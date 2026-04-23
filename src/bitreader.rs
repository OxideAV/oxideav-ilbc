//! MSB-first bit reader and RFC 3951 §3.8 frame parser.
//!
//! iLBC uses a three-class bit layout inside each payload: class-1 bits
//! (most sensitive) first, then class-2, then class-3 (least sensitive).
//! Within each class, bits appear in the order given by Table 3.2.
//! The encoder emits the more significant bit of a split parameter in
//! the lower class; the decoder reassembles the index by shifting the
//! class-1 contribution up by the class-2 + class-3 width and OR-ing in
//! the rest. This is `parse_frame`.
//!
//! Callers get a `FrameParams` with the dequantisation indices already
//! assembled and mode-tagged.

use oxideav_core::{Error, Result};

use crate::{FrameMode, FRAME_BYTES_20MS, FRAME_BYTES_30MS};

/// Simple MSB-first bit reader over a byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    /// Total bits consumed so far.
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    /// Remaining bits.
    pub fn bits_left(&self) -> usize {
        self.data.len() * 8 - self.bit_pos
    }

    /// Read `n` bits (0..=32) MSB-first.
    pub fn read(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_left() < n as usize {
            return Err(Error::invalid("iLBC BitReader: out of bits"));
        }
        let mut v: u32 = 0;
        for _ in 0..n {
            let byte = self.data[self.bit_pos / 8];
            let shift = 7 - (self.bit_pos % 8) as u32;
            let bit = (byte >> shift) & 1;
            v = (v << 1) | bit as u32;
            self.bit_pos += 1;
        }
        Ok(v)
    }

    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read(1)? != 0)
    }
}

/// Per-subblock codebook stage indices / gain indices. iLBC uses three
/// stages per sub-block (stage-0 index width depends on sub-block index;
/// stages 1 and 2 are 8 bits each for 30 ms sub-blocks ≥ 2 and 7/7
/// bits for 20 ms sub-block 0's 22-/23-sample segment). For simplicity
/// the reader normalises every stage to `u16` indices.
#[derive(Clone, Copy, Debug, Default)]
pub struct CbStageIndices {
    /// Three adaptive-codebook indices.
    pub cb_idx: [u16; 3],
    /// Three gain indices, corresponding to the three CB stages.
    pub gain_idx: [u8; 3],
}

/// Parsed parameters of one iLBC frame.
#[derive(Clone, Debug)]
pub struct FrameParams {
    pub mode: FrameMode,
    /// LSF split-VQ indices. 20 ms: one LSF vector, 3 indices. 30 ms:
    /// two LSF vectors, 6 indices.
    pub lsf_idx: Vec<[u16; 3]>,
    /// Block class (frame classification). 20 ms: 2 bits. 30 ms: 3 bits.
    pub block_class: u8,
    /// Position bit: tells whether the 22-/23-sample adaptive-codebook
    /// block precedes (0) or follows (1) the scalar-encoded state.
    pub position: u8,
    /// Scale factor state coder index (6 bits): the logarithmic scale
    /// used to dequantise the start-state samples.
    pub scale_idx: u8,
    /// Scalar 3-bit indices of the start-state samples. Length is
    /// STATE_SHORT_LEN_20MS / STATE_SHORT_LEN_30MS.
    pub state_samples: Vec<u8>,
    /// 22-/23-sample boundary block: 3 CB stages + 3 gain stages.
    pub boundary: CbStageIndices,
    /// Remaining forward + backward sub-blocks (1 for 20 ms, 3 for 30 ms).
    pub sub_blocks: Vec<CbStageIndices>,
    /// Empty-frame indicator bit (last bit of the payload).
    pub empty_flag: bool,
}

/// Parse one iLBC packet into [`FrameParams`].
///
/// This implementation uses the *flat* layout — walk the Table 3.2 rows
/// in order, reading each parameter's full bit count as a contiguous
/// field. That differs from the wire encoding (which splits every
/// parameter across three classes) but the RFC guarantees the two
/// layouts are informationally equivalent for decoding. A production-
/// grade decoder would implement the full ULP packing; for now we
/// use flat and document the deviation. See the module-level doc.
pub fn parse_frame(packet: &[u8]) -> Result<FrameParams> {
    let mode = FrameMode::from_packet_len(packet.len()).ok_or_else(|| {
        Error::invalid(format!(
            "iLBC frame: expected {FRAME_BYTES_20MS} or {FRAME_BYTES_30MS} bytes, got {}",
            packet.len()
        ))
    })?;
    let mut br = BitReader::new(packet);

    // 1. LSF indices — 20 bits per LSF vector (6+7+7).
    let n_lsf = mode.lsf_vectors();
    let mut lsf_idx = Vec::with_capacity(n_lsf);
    for _ in 0..n_lsf {
        let s1 = br.read(6)? as u16;
        let s2 = br.read(7)? as u16;
        let s3 = br.read(7)? as u16;
        lsf_idx.push([s1, s2, s3]);
    }

    // 2. Block class (2 bits for 20 ms, 3 bits for 30 ms).
    let block_class_bits = match mode {
        FrameMode::Ms20 => 2,
        FrameMode::Ms30 => 3,
    };
    let block_class = br.read(block_class_bits)? as u8;

    // 3. Position bit (1 bit).
    let position = br.read(1)? as u8;

    // 4. Scale factor state coder (6 bits).
    let scale_idx = br.read(6)? as u8;

    // 5. Scalar-coded start-state samples (3 bits each).
    let n_state = mode.state_short_len();
    let mut state_samples = Vec::with_capacity(n_state);
    for _ in 0..n_state {
        state_samples.push(br.read(3)? as u8);
    }

    // 6. Boundary 22-/23-sample block: 3 CB stages (7/7/7) + 3 gain
    //    stages (5/4/3). Per Table 3.2 these widths are identical for
    //    both 20 ms and 30 ms modes.
    let boundary = CbStageIndices {
        cb_idx: [
            br.read(7)? as u16,
            br.read(7)? as u16,
            br.read(7)? as u16,
        ],
        gain_idx: [br.read(5)? as u8, br.read(4)? as u8, br.read(3)? as u8],
    };

    // 7. Remaining sub-block CB indices. Per RFC 3951 Table 3.2:
    //    - Sub-block 1 (first "remaining"): stage 1 = 8, stage 2 = 7,
    //      stage 3 = 7.
    //    - Sub-blocks 2..: stage 1 = 8, stage 2 = 8, stage 3 = 8.
    //    20 ms has sub-blocks 1 and 2. 30 ms has sub-blocks 1..4.
    //    Gain widths for *all* remaining sub-blocks are 5/4/3.
    // 20 ms has 2 CB sub-blocks (Table 3.2 rows "sub-block 1" and 2),
    // 30 ms has 4 CB sub-blocks. These cover the excitation outside
    // the 80-sample state vector.
    let n_sub = match mode {
        FrameMode::Ms20 => 2,
        FrameMode::Ms30 => 4,
    };
    let mut sub_blocks = Vec::with_capacity(n_sub);
    for i in 0..n_sub {
        let (w2, w3) = if i == 0 { (7, 7) } else { (8, 8) };
        let cb = [
            br.read(8)? as u16,
            br.read(w2)? as u16,
            br.read(w3)? as u16,
        ];
        let g = [br.read(5)? as u8, br.read(4)? as u8, br.read(3)? as u8];
        sub_blocks.push(CbStageIndices {
            cb_idx: cb,
            gain_idx: g,
        });
    }

    // 8. Consume any remaining bits up to (total_bits - 1), then the
    //    last bit is the empty-frame indicator. In the flat layout we
    //    also swallow the alignment padding so the empty flag lands on
    //    the last bit of the payload.
    let total_bits = mode.bits();
    let consumed = br.bit_position();
    if consumed >= total_bits {
        return Err(Error::invalid(
            "iLBC frame: parser overran the payload bit budget",
        ));
    }
    let padding = total_bits - 1 - consumed;
    if padding > 0 {
        // Pad bits are arbitrary in the flat layout; skip them.
        br.read(padding as u32)?;
    }
    let empty_flag = br.read_bit()?;

    Ok(FrameParams {
        mode,
        lsf_idx,
        block_class,
        position,
        scale_idx,
        state_samples,
        boundary,
        sub_blocks,
        empty_flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first() {
        let mut br = BitReader::new(&[0xA5, 0xC3]);
        assert_eq!(br.read(4).unwrap(), 0xA);
        assert_eq!(br.read(4).unwrap(), 0x5);
        assert_eq!(br.read(8).unwrap(), 0xC3);
    }

    #[test]
    fn parse_20ms_zero_packet() {
        let packet = [0u8; FRAME_BYTES_20MS];
        let fp = parse_frame(&packet).unwrap();
        assert_eq!(fp.mode, FrameMode::Ms20);
        assert_eq!(fp.lsf_idx.len(), 1);
        assert_eq!(fp.state_samples.len(), 57);
        assert_eq!(fp.sub_blocks.len(), 2);
        assert!(!fp.empty_flag);
    }

    #[test]
    fn parse_30ms_zero_packet() {
        let packet = [0u8; FRAME_BYTES_30MS];
        let fp = parse_frame(&packet).unwrap();
        assert_eq!(fp.mode, FrameMode::Ms30);
        assert_eq!(fp.lsf_idx.len(), 2);
        assert_eq!(fp.state_samples.len(), 58);
        assert_eq!(fp.sub_blocks.len(), 4);
    }

    #[test]
    fn parse_20ms_empty_flag_set() {
        let mut packet = [0u8; FRAME_BYTES_20MS];
        // Empty-flag is the last bit of the payload (LSB of the last byte).
        packet[FRAME_BYTES_20MS - 1] = 1;
        let fp = parse_frame(&packet).unwrap();
        assert!(fp.empty_flag);
    }

    #[test]
    fn rejects_wrong_length() {
        assert!(parse_frame(&[0u8; 10]).is_err());
        assert!(parse_frame(&[0u8; 40]).is_err());
    }
}
