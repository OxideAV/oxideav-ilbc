//! MSB-first bit writer mirroring [`crate::bitreader::BitReader`].
//!
//! The writer expects the same flat layout the decoder parses: walk
//! Table 3.2 top-to-bottom, emitting each parameter's full bit count as a
//! single contiguous MSB-first field. The final bit is the empty-frame
//! indicator (bit 303 for 20 ms, bit 399 for 30 ms).

use oxideav_core::{Error, Result};

use crate::bitreader::CbStageIndices;
use crate::{FrameMode, FRAME_BYTES_20MS, FRAME_BYTES_30MS};

/// Simple MSB-first bit writer over a pre-allocated byte buffer.
pub struct BitWriter<'a> {
    data: &'a mut [u8],
    bit_pos: usize,
}

impl<'a> BitWriter<'a> {
    pub fn new(data: &'a mut [u8]) -> Self {
        for b in data.iter_mut() {
            *b = 0;
        }
        Self { data, bit_pos: 0 }
    }

    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    pub fn bits_left(&self) -> usize {
        self.data.len() * 8 - self.bit_pos
    }

    /// Write the low `n` bits of `value`, MSB-first.
    pub fn write(&mut self, value: u32, n: u32) -> Result<()> {
        debug_assert!(n <= 32);
        if self.bits_left() < n as usize {
            return Err(Error::invalid("iLBC BitWriter: out of space"));
        }
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte_idx = self.bit_pos / 8;
            let shift = 7 - (self.bit_pos % 8);
            self.data[byte_idx] |= bit << shift;
            self.bit_pos += 1;
        }
        Ok(())
    }

    pub fn write_bit(&mut self, b: bool) -> Result<()> {
        self.write(if b { 1 } else { 0 }, 1)
    }
}

/// Parameters that must be packed into the iLBC payload. The field
/// layout mirrors [`crate::bitreader::FrameParams`], except we take
/// owned state_samples / sub_blocks slices (the writer doesn't mutate).
#[derive(Clone, Debug)]
pub struct PackParams {
    pub mode: FrameMode,
    pub lsf_idx: Vec<[u16; 3]>,
    pub block_class: u8,
    pub position: u8,
    pub scale_idx: u8,
    pub state_samples: Vec<u8>,
    pub boundary: CbStageIndices,
    pub sub_blocks: Vec<CbStageIndices>,
    pub empty_flag: bool,
}

/// Pack a frame into its byte payload using the flat layout the decoder
/// parses.
pub fn pack_frame(params: &PackParams) -> Result<Vec<u8>> {
    let mode = params.mode;
    let nbytes = match mode {
        FrameMode::Ms20 => FRAME_BYTES_20MS,
        FrameMode::Ms30 => FRAME_BYTES_30MS,
    };
    let mut buf = vec![0u8; nbytes];
    {
        let mut bw = BitWriter::new(&mut buf);

        // 1. LSF indices.
        if params.lsf_idx.len() != mode.lsf_vectors() {
            return Err(Error::invalid(format!(
                "iLBC pack: expected {} LSF vectors, got {}",
                mode.lsf_vectors(),
                params.lsf_idx.len()
            )));
        }
        for idx in &params.lsf_idx {
            bw.write(idx[0] as u32, 6)?;
            bw.write(idx[1] as u32, 7)?;
            bw.write(idx[2] as u32, 7)?;
        }
        // 2. Block class.
        let bits_bc = match mode {
            FrameMode::Ms20 => 2,
            FrameMode::Ms30 => 3,
        };
        bw.write(params.block_class as u32, bits_bc)?;
        // 3. Position bit.
        bw.write(params.position as u32, 1)?;
        // 4. Scale idx (6 bits).
        bw.write(params.scale_idx as u32, 6)?;
        // 5. State samples (3 bits each).
        if params.state_samples.len() != mode.state_short_len() {
            return Err(Error::invalid(format!(
                "iLBC pack: expected {} state samples, got {}",
                mode.state_short_len(),
                params.state_samples.len()
            )));
        }
        for &s in &params.state_samples {
            bw.write((s & 0x7) as u32, 3)?;
        }
        // 6. Boundary block: 7/7/7 CB, 5/4/3 gain.
        bw.write(params.boundary.cb_idx[0] as u32, 7)?;
        bw.write(params.boundary.cb_idx[1] as u32, 7)?;
        bw.write(params.boundary.cb_idx[2] as u32, 7)?;
        bw.write(params.boundary.gain_idx[0] as u32, 5)?;
        bw.write(params.boundary.gain_idx[1] as u32, 4)?;
        bw.write(params.boundary.gain_idx[2] as u32, 3)?;
        // 7. Sub-blocks: (8,7,7)/(5,4,3) for sub_block 0 and (8,8,8)/(5,4,3) for later.
        let expected_sb = mode.cb_sub_blocks();
        if params.sub_blocks.len() != expected_sb {
            return Err(Error::invalid(format!(
                "iLBC pack: expected {} sub-blocks, got {}",
                expected_sb,
                params.sub_blocks.len()
            )));
        }
        for (i, sb) in params.sub_blocks.iter().enumerate() {
            let (w2, w3) = if i == 0 { (7, 7) } else { (8, 8) };
            bw.write(sb.cb_idx[0] as u32, 8)?;
            bw.write(sb.cb_idx[1] as u32, w2)?;
            bw.write(sb.cb_idx[2] as u32, w3)?;
            bw.write(sb.gain_idx[0] as u32, 5)?;
            bw.write(sb.gain_idx[1] as u32, 4)?;
            bw.write(sb.gain_idx[2] as u32, 3)?;
        }
        // 8. Padding to (total_bits - 1), then empty-frame indicator.
        let total_bits = mode.bits();
        let consumed = bw.bit_position();
        if consumed >= total_bits {
            return Err(Error::invalid("iLBC pack: exceeded payload bit budget"));
        }
        let padding = total_bits - 1 - consumed;
        if padding > 0 {
            bw.write(0, padding as u32)?;
        }
        bw.write_bit(params.empty_flag)?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::parse_frame;

    fn synthetic_params_20ms() -> PackParams {
        PackParams {
            mode: FrameMode::Ms20,
            lsf_idx: vec![[5, 17, 42]],
            block_class: 1,
            position: 1,
            scale_idx: 20,
            state_samples: vec![4; 57],
            boundary: CbStageIndices {
                cb_idx: [10, 20, 30],
                gain_idx: [7, 3, 1],
            },
            sub_blocks: vec![
                CbStageIndices {
                    cb_idx: [100, 50, 60],
                    gain_idx: [12, 8, 2],
                },
                CbStageIndices {
                    cb_idx: [200, 80, 70],
                    gain_idx: [14, 7, 3],
                },
            ],
            empty_flag: false,
        }
    }

    #[test]
    fn pack_20ms_round_trip() {
        let params = synthetic_params_20ms();
        let bytes = pack_frame(&params).unwrap();
        assert_eq!(bytes.len(), FRAME_BYTES_20MS);
        let fp = parse_frame(&bytes).unwrap();
        assert_eq!(fp.mode, FrameMode::Ms20);
        assert_eq!(fp.lsf_idx, params.lsf_idx);
        assert_eq!(fp.block_class, params.block_class);
        assert_eq!(fp.position, params.position);
        assert_eq!(fp.scale_idx, params.scale_idx);
        assert_eq!(fp.state_samples, params.state_samples);
        assert_eq!(fp.boundary.cb_idx, params.boundary.cb_idx);
        assert_eq!(fp.boundary.gain_idx, params.boundary.gain_idx);
        for (a, b) in fp.sub_blocks.iter().zip(params.sub_blocks.iter()) {
            assert_eq!(a.cb_idx, b.cb_idx);
            assert_eq!(a.gain_idx, b.gain_idx);
        }
        assert_eq!(fp.empty_flag, params.empty_flag);
    }

    #[test]
    fn pack_30ms_round_trip() {
        let mut params = synthetic_params_20ms();
        params.mode = FrameMode::Ms30;
        params.lsf_idx = vec![[1, 2, 3], [4, 5, 6]];
        params.state_samples = vec![3; 58];
        params.sub_blocks = vec![
            CbStageIndices {
                cb_idx: [10, 20, 30],
                gain_idx: [1, 2, 3],
            },
            CbStageIndices {
                cb_idx: [40, 50, 60],
                gain_idx: [4, 5, 6],
            },
            CbStageIndices {
                cb_idx: [70, 80, 90],
                gain_idx: [7, 8, 1],
            },
            CbStageIndices {
                cb_idx: [100, 110, 120],
                gain_idx: [10, 11, 2],
            },
        ];
        let bytes = pack_frame(&params).unwrap();
        assert_eq!(bytes.len(), FRAME_BYTES_30MS);
        let fp = parse_frame(&bytes).unwrap();
        assert_eq!(fp.mode, FrameMode::Ms30);
        assert_eq!(fp.lsf_idx, params.lsf_idx);
        assert_eq!(fp.state_samples, params.state_samples);
        for (a, b) in fp.sub_blocks.iter().zip(params.sub_blocks.iter()) {
            assert_eq!(a.cb_idx, b.cb_idx);
            assert_eq!(a.gain_idx, b.gain_idx);
        }
    }

    #[test]
    fn pack_empty_flag_bit() {
        let mut params = synthetic_params_20ms();
        params.empty_flag = true;
        let bytes = pack_frame(&params).unwrap();
        // Last bit is the LSB of the final byte.
        assert_eq!(bytes[FRAME_BYTES_20MS - 1] & 1, 1);
        let fp = parse_frame(&bytes).unwrap();
        assert!(fp.empty_flag);
    }
}
