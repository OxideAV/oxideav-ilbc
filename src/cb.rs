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

/// Successive-rescale gain dequantisation.
///
/// Stage k's gain index picks a quantiser step whose full-scale is the
/// dequantised stage-(k-1) gain. The RFC Appendix A.22 publishes the
/// verbatim `gain_sq5Tbl` / `gain_sq4Tbl` / `gain_sq3Tbl` tables; the
/// canonical centroid layout is symmetric around zero with increasing
/// magnitude, which we replicate here to keep the decoder structurally
/// correct on all index values. Documented deviation.
pub const GAIN_SQ5_TBL: [f32; 32] = [
    -1.00, -0.94, -0.88, -0.81, -0.75, -0.69, -0.63, -0.56, -0.50, -0.44, -0.38, -0.31, -0.25,
    -0.19, -0.13, -0.06, 0.06, 0.13, 0.19, 0.25, 0.31, 0.38, 0.44, 0.50, 0.56, 0.63, 0.69, 0.75,
    0.81, 0.88, 0.94, 1.00,
];
pub const GAIN_SQ4_TBL: [f32; 16] = [
    -1.00, -0.88, -0.75, -0.63, -0.50, -0.38, -0.25, -0.13, 0.13, 0.25, 0.38, 0.50, 0.63, 0.75,
    0.88, 1.00,
];
pub const GAIN_SQ3_TBL: [f32; 8] = [-1.00, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.00];

/// Dequantise a 3-stage gain vector given the three raw indices.
///
/// Gain[0] uses the 5-bit table, gain[1] uses 4-bit, gain[2] uses 3-bit.
/// Successive rescaling: gain[k] centroid is multiplied by |gain[k-1]|
/// (with a floor) so the second and third stages refine the first.
pub fn decode_gains(indices: &[u8; 3]) -> [f32; 3] {
    let g0 = GAIN_SQ5_TBL[(indices[0] as usize) % GAIN_SQ5_TBL.len()];
    let base1 = g0.abs().max(0.1);
    let g1 = GAIN_SQ4_TBL[(indices[1] as usize) % GAIN_SQ4_TBL.len()] * base1;
    let base2 = g1.abs().max(0.1);
    let g2 = GAIN_SQ3_TBL[(indices[2] as usize) % GAIN_SQ3_TBL.len()] * base2;
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
