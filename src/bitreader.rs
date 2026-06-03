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
//! assembled and mode-tagged. Field semantics map onto RFC 3951
//! `iLBC_decode`'s state as follows:
//!
//! - [`FrameParams::lsf_idx`]      `lsf_i`
//! - [`FrameParams::block_class`]  `start` (1-based start sub-frame)
//! - [`FrameParams::position`]     `state_first` (1=leading, 0=trailing)
//! - [`FrameParams::scale_idx`]    `idxForMax` (RFC §3.5.2)
//! - [`FrameParams::state_samples`] `idxVec`
//! - [`FrameParams::boundary`]     `extra_cb_index` / `extra_gain_index`
//! - [`FrameParams::sub_blocks`]   `cb_index` / `gain_index`
//! - [`FrameParams::empty_flag`]   trailing bit (`last_bit` in RFC)

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

/// Parse one iLBC packet into [`FrameParams`] using the RFC 3951 §3.8
/// ULP (uneven-level-protection) bit layout.
///
/// The bit reader makes three passes over the payload, accumulating
/// the class-1 high bits, class-2 mid bits, and class-3 low bits of
/// each parameter in the order given by Table 3.2. The per-parameter
/// per-class bit widths live in [`crate::ulp`] (mirroring
/// `ULP_20msTbl` / `ULP_30msTbl` in RFC 3951 §A.41).
///
/// The single trailing bit of the payload is the empty-frame
/// indicator (RFC §3.8 `last_bit`).
pub fn parse_frame(packet: &[u8]) -> Result<FrameParams> {
    let mode = FrameMode::from_packet_len(packet.len()).ok_or_else(|| {
        Error::invalid(format!(
            "iLBC frame: expected {FRAME_BYTES_20MS} or {FRAME_BYTES_30MS} bytes, got {}",
            packet.len()
        ))
    })?;
    let n_state = mode.state_short_len();
    let mut br = BitReader::new(packet);
    let (logical, empty_flag) = crate::ulp::unpack_logical(mode, n_state, |n| br.read(n))?;

    // Translate LogicalParams to FrameParams. `cb_index` / `cb_gain`
    // are stored row-major: `[sub_block_idx][stage] -> u32`. Boundary
    // block is `extra_*` (3 stages).
    let lsf_idx = logical
        .lsf_idx
        .chunks_exact(3)
        .map(|c| [c[0] as u16, c[1] as u16, c[2] as u16])
        .collect::<Vec<_>>();
    let boundary = CbStageIndices {
        cb_idx: [
            logical.extra_cb_index[0] as u16,
            logical.extra_cb_index[1] as u16,
            logical.extra_cb_index[2] as u16,
        ],
        gain_idx: [
            logical.extra_cb_gain[0] as u8,
            logical.extra_cb_gain[1] as u8,
            logical.extra_cb_gain[2] as u8,
        ],
    };
    let sub_blocks = logical
        .cb_index
        .iter()
        .zip(logical.cb_gain.iter())
        .map(|(cb, g)| CbStageIndices {
            cb_idx: [cb[0] as u16, cb[1] as u16, cb[2] as u16],
            gain_idx: [g[0] as u8, g[1] as u8, g[2] as u8],
        })
        .collect();
    Ok(FrameParams {
        mode,
        lsf_idx,
        block_class: logical.start as u8,
        position: logical.state_first as u8,
        scale_idx: logical.idx_for_max as u8,
        state_samples: logical.state_samples.iter().map(|v| *v as u8).collect(),
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
