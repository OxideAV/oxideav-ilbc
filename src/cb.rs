//! Multistage adaptive codebook (RFC 3951 §3.6, §4.3, §4.4).
//!
//! The iLBC excitation for each 40-sample sub-block is the sum of three
//! scaled codebook vectors, each drawn from a 147-sample *adaptive
//! codebook memory* whose contents evolve with previously-decoded
//! sub-blocks:
//!
//! ```text
//!     cbvec = gain[0]*cbvec[0] + gain[1]*cbvec[1] + gain[2]*cbvec[2]
//! ```
//!
//! Each stage's index picks a segment from the codebook memory (with
//! filter / interpolation variants per §3.6.3). Gains are successively-
//! rescaled — stage k's gain is quantised relative to stage k-1's
//! dequantised value.
//!
//! This module implements the decoder-side construction. It uses a
//! simplified base codebook that takes contiguous windows out of the
//! memory buffer, which is the common case for iLBC — augmented / bit-
//! reversed codebooks (§3.6.3.2 / §3.6.3.3) are not exercised.

use crate::{CB_LMEM, SUBL};

/// Successive-rescale gain dequantisation (RFC 3951 §3.6.4.2).
///
/// Stage 0 uses `gain_sq5Tbl` (positive-only, 5 bits) with an implicit
/// scale of 1.0. Stages 1 and 2 use `gain_sq4Tbl` (4 bits) and
/// `gain_sq3Tbl` (3 bits) respectively, scaled by the absolute value of
/// the previous stage's dequantised gain (clamped to a floor of 0.1).
///
/// All three tables are transcribed verbatim from RFC 3951 Appendix A.8.
pub const GAIN_SQ5_TBL: [f32; 32] = [
    0.037476, 0.075012, 0.112488, 0.150024, 0.187500, 0.224976, 0.262512, 0.299988, 0.337524,
    0.375000, 0.412476, 0.450012, 0.487488, 0.525024, 0.562500, 0.599976, 0.637512, 0.674988,
    0.712524, 0.750000, 0.787476, 0.825012, 0.862488, 0.900024, 0.937500, 0.974976, 1.012512,
    1.049988, 1.087524, 1.125000, 1.162476, 1.200012,
];
pub const GAIN_SQ4_TBL: [f32; 16] = [
    -1.049988, -0.900024, -0.750000, -0.599976, -0.450012, -0.299988, -0.150024, 0.000000,
    0.150024, 0.299988, 0.450012, 0.599976, 0.750000, 0.900024, 1.049988, 1.200012,
];
pub const GAIN_SQ3_TBL: [f32; 8] = [
    -1.000000, -0.659973, -0.330017, 0.000000, 0.250000, 0.500000, 0.750000, 1.000000,
];

/// Gain-floor used by the reference `gaindequant` when the external
/// scale falls below 0.1 (RFC 3951 Appendix A.22).
const GAIN_SCALE_FLOOR: f32 = 0.1;

/// Dequantise a 3-stage gain vector given the three raw indices.
///
/// Mirrors the RFC reference `gaindequant` call sequence:
///   `gain[0] = gain_sq5Tbl[i0] * max(1.0, 0.1)`
///   `gain[1] = gain_sq4Tbl[i1] * max(|gain[0]|, 0.1)`
///   `gain[2] = gain_sq3Tbl[i2] * max(|gain[1]|, 0.1)`
pub fn decode_gains(indices: &[u8; 3]) -> [f32; 3] {
    let scale0 = 1.0f32.max(GAIN_SCALE_FLOOR);
    let g0 = GAIN_SQ5_TBL[(indices[0] as usize) % GAIN_SQ5_TBL.len()] * scale0;
    let scale1 = g0.abs().max(GAIN_SCALE_FLOOR);
    let g1 = GAIN_SQ4_TBL[(indices[1] as usize) % GAIN_SQ4_TBL.len()] * scale1;
    let scale2 = g1.abs().max(GAIN_SCALE_FLOOR);
    let g2 = GAIN_SQ3_TBL[(indices[2] as usize) % GAIN_SQ3_TBL.len()] * scale2;
    [g0, g1, g2]
}

/// Extract a 40-sample codebook vector from the 147-sample adaptive
/// codebook memory.
///
/// The `index` is 7 or 8 bits; the number of base-codebook positions is
/// `CB_LMEM - SUBL + 1 = 108`. Indices beyond that (108..255) would
/// normally select augmented / expanded codebooks per §3.6.3; we wrap
/// them modulo the base range.
pub fn extract_cbvec(cb_mem: &[f32; CB_LMEM], index: u16) -> [f32; SUBL] {
    let base_positions = CB_LMEM - SUBL + 1; // 108
    let pos = (index as usize) % base_positions;
    let mut out = [0.0f32; SUBL];
    for n in 0..SUBL {
        out[n] = cb_mem[pos + n];
    }
    out
}

/// Build the decoded excitation for one sub-block:
///   cbvec = Σ gain[k] * cbvec_k
pub fn construct_excitation(
    cb_mem: &[f32; CB_LMEM],
    cb_idx: &[u16; 3],
    gain_idx: &[u8; 3],
) -> [f32; SUBL] {
    let gains = decode_gains(gain_idx);
    let mut exc = [0.0f32; SUBL];
    for stage in 0..3 {
        let v = extract_cbvec(cb_mem, cb_idx[stage]);
        for n in 0..SUBL {
            exc[n] += gains[stage] * v[n];
        }
    }
    exc
}

/// Update the adaptive codebook memory: shift left by `SUBL` and
/// append `new_excitation` at the tail.
pub fn update_cb_memory(cb_mem: &mut [f32; CB_LMEM], new_excitation: &[f32; SUBL]) {
    for i in 0..(CB_LMEM - SUBL) {
        cb_mem[i] = cb_mem[i + SUBL];
    }
    for i in 0..SUBL {
        cb_mem[CB_LMEM - SUBL + i] = new_excitation[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_tables_monotone() {
        for k in 1..GAIN_SQ5_TBL.len() {
            assert!(GAIN_SQ5_TBL[k] > GAIN_SQ5_TBL[k - 1]);
        }
        for k in 1..GAIN_SQ4_TBL.len() {
            assert!(GAIN_SQ4_TBL[k] > GAIN_SQ4_TBL[k - 1]);
        }
        for k in 1..GAIN_SQ3_TBL.len() {
            assert!(GAIN_SQ3_TBL[k] > GAIN_SQ3_TBL[k - 1]);
        }
    }

    #[test]
    fn decode_gains_finite() {
        let g = decode_gains(&[16, 8, 4]);
        for v in g.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn gain_tables_bit_exact() {
        // Verbatim RFC 3951 Appendix A.8 endpoints / mid-points.
        assert_eq!(GAIN_SQ5_TBL[0], 0.037476);
        assert_eq!(GAIN_SQ5_TBL[31], 1.200012);
        assert_eq!(GAIN_SQ4_TBL[0], -1.049988);
        assert_eq!(GAIN_SQ4_TBL[7], 0.000000);
        assert_eq!(GAIN_SQ4_TBL[15], 1.200012);
        assert_eq!(GAIN_SQ3_TBL[0], -1.000000);
        assert_eq!(GAIN_SQ3_TBL[1], -0.659973);
        assert_eq!(GAIN_SQ3_TBL[3], 0.000000);
        assert_eq!(GAIN_SQ3_TBL[7], 1.000000);
    }

    #[test]
    fn extract_cbvec_inside_mem() {
        let mut mem = [0.0f32; CB_LMEM];
        for i in 0..CB_LMEM {
            mem[i] = i as f32;
        }
        let v = extract_cbvec(&mem, 10);
        assert_eq!(v[0], 10.0);
        assert_eq!(v[SUBL - 1], 10.0 + (SUBL as f32) - 1.0);
    }

    #[test]
    fn update_mem_shifts_correctly() {
        let mut mem = [0.0f32; CB_LMEM];
        for i in 0..CB_LMEM {
            mem[i] = i as f32;
        }
        let new_exc: [f32; SUBL] = core::array::from_fn(|i| -(i as f32));
        update_cb_memory(&mut mem, &new_exc);
        assert_eq!(mem[0], SUBL as f32); // old index SUBL moved to 0
        assert_eq!(mem[CB_LMEM - SUBL], 0.0);
        assert_eq!(mem[CB_LMEM - 1], -((SUBL - 1) as f32));
    }

    #[test]
    fn construct_excitation_finite() {
        let mem: [f32; CB_LMEM] = core::array::from_fn(|i| (i as f32 * 0.1).sin());
        let e = construct_excitation(&mem, &[5, 10, 20], &[16, 8, 4]);
        for v in e.iter() {
            assert!(v.is_finite());
        }
    }
}
