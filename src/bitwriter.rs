//! MSB-first bit writer mirroring [`crate::bitreader::BitReader`].
//!
//! The writer emits the same RFC 3951 §3.8 ULP layout the decoder
//! parses: three passes per payload, each parameter contributing its
//! class-N high bits on pass N. Per-parameter widths come from
//! [`crate::ulp`]. The final bit is the empty-frame indicator
//! (bit 303 for 20 ms, bit 399 for 30 ms).

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

/// Pack a frame into its byte payload using the RFC 3951 §3.8 ULP
/// layout. Field-count validation runs up front; per-class widths
/// come from [`crate::ulp`].
pub fn pack_frame(params: &PackParams) -> Result<Vec<u8>> {
    let mode = params.mode;
    let nbytes = match mode {
        FrameMode::Ms20 => FRAME_BYTES_20MS,
        FrameMode::Ms30 => FRAME_BYTES_30MS,
    };

    if params.lsf_idx.len() != mode.lsf_vectors() {
        return Err(Error::invalid(format!(
            "iLBC pack: expected {} LSF vectors, got {}",
            mode.lsf_vectors(),
            params.lsf_idx.len()
        )));
    }
    if params.state_samples.len() != mode.state_short_len() {
        return Err(Error::invalid(format!(
            "iLBC pack: expected {} state samples, got {}",
            mode.state_short_len(),
            params.state_samples.len()
        )));
    }
    let expected_sb = mode.cb_sub_blocks();
    if params.sub_blocks.len() != expected_sb {
        return Err(Error::invalid(format!(
            "iLBC pack: expected {} sub-blocks, got {}",
            expected_sb,
            params.sub_blocks.len()
        )));
    }

    // Build LogicalParams (RFC §3.8 named view of the wire fields)
    // from the in-tree FrameParams-style PackParams, then run the ULP
    // emit list through the bit writer.
    let logical = crate::ulp::LogicalParams {
        lsf_idx: params
            .lsf_idx
            .iter()
            .flat_map(|row| [row[0] as u32, row[1] as u32, row[2] as u32])
            .collect(),
        start: params.block_class as u32,
        state_first: params.position as u32,
        idx_for_max: params.scale_idx as u32,
        state_samples: params.state_samples.iter().map(|v| *v as u32).collect(),
        extra_cb_index: [
            params.boundary.cb_idx[0] as u32,
            params.boundary.cb_idx[1] as u32,
            params.boundary.cb_idx[2] as u32,
        ],
        extra_cb_gain: [
            params.boundary.gain_idx[0] as u32,
            params.boundary.gain_idx[1] as u32,
            params.boundary.gain_idx[2] as u32,
        ],
        cb_index: params
            .sub_blocks
            .iter()
            .map(|sb| {
                [
                    sb.cb_idx[0] as u32,
                    sb.cb_idx[1] as u32,
                    sb.cb_idx[2] as u32,
                ]
            })
            .collect(),
        cb_gain: params
            .sub_blocks
            .iter()
            .map(|sb| {
                [
                    sb.gain_idx[0] as u32,
                    sb.gain_idx[1] as u32,
                    sb.gain_idx[2] as u32,
                ]
            })
            .collect(),
    };
    let emit_list = crate::ulp::pack_emit_list(mode, &logical, params.empty_flag);

    let mut buf = vec![0u8; nbytes];
    {
        let mut bw = BitWriter::new(&mut buf);
        for (value, width) in emit_list {
            bw.write(value, width)?;
        }
        // The emit list packs every payload bit (class 1 + class 2 +
        // class 3 + empty-frame indicator); the writer should be at
        // exactly `mode.bits()` here.
        debug_assert_eq!(bw.bit_position(), mode.bits());
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
