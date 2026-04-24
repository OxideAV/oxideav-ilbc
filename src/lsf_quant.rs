//! LSF split-VQ quantisation for the encoder — RFC 3951 §3.2.4.
//!
//! The 10-dimensional LSF vector is split into three sub-vectors of
//! dimensions 3, 3, 4 and each is quantised against a dedicated codebook
//! (`LSF_CB_TBL_1/2/3` in [`crate::lsf_tables`]). The quantiser returns
//! three indices (6, 7, 7 bits) and the corresponding quantised LSF
//! vector.

use crate::lsf_tables::{
    lookup_split, LSF_CB_TBL_1, LSF_CB_TBL_2, LSF_CB_TBL_3, SPLIT_DIMS, SPLIT_OFFSETS, SPLIT_SIZES,
};
use crate::LPC_ORDER;

/// Squared-error nearest-neighbour search for one split.
fn nearest(split: usize, target: &[f32]) -> u16 {
    debug_assert_eq!(target.len(), SPLIT_DIMS[split]);
    let mut best_idx = 0u16;
    let mut best_err = f32::INFINITY;
    let n = SPLIT_SIZES[split];
    for i in 0..n {
        let mut err = 0.0f32;
        match split {
            0 => {
                let row = &LSF_CB_TBL_1[i];
                for j in 0..3 {
                    let d = row[j] - target[j];
                    err += d * d;
                }
            }
            1 => {
                let row = &LSF_CB_TBL_2[i];
                for j in 0..3 {
                    let d = row[j] - target[j];
                    err += d * d;
                }
            }
            2 => {
                let row = &LSF_CB_TBL_3[i];
                for j in 0..4 {
                    let d = row[j] - target[j];
                    err += d * d;
                }
            }
            _ => unreachable!(),
        }
        if err < best_err {
            best_err = err;
            best_idx = i as u16;
        }
    }
    best_idx
}

/// Quantise one 10-dim LSF vector, returning (indices, quantised LSF).
///
/// Indices are in the same order as the decoder's `FrameParams::lsf_idx`:
/// `[split1 (6b), split2 (7b), split3 (7b)]`.
pub fn quantise_lsf(lsf: &[f32; LPC_ORDER]) -> ([u16; 3], [f32; LPC_ORDER]) {
    let mut indices = [0u16; 3];
    let mut qlsf = [0.0f32; LPC_ORDER];
    for s in 0..3 {
        let dim = SPLIT_DIMS[s];
        let off = SPLIT_OFFSETS[s];
        let idx = nearest(s, &lsf[off..off + dim]);
        indices[s] = idx;
        let sub = lookup_split(s, idx);
        for j in 0..dim {
            qlsf[off + j] = sub[j];
        }
    }
    (indices, qlsf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsf_tables::LSF_MEAN;

    #[test]
    fn round_trip_mean_is_close() {
        let (_idx, q) = quantise_lsf(&LSF_MEAN);
        // LSF_MEAN should quantise to something close (not exact — the
        // mean isn't necessarily in the codebook).
        for k in 0..LPC_ORDER {
            assert!((q[k] - LSF_MEAN[k]).abs() < 0.5, "k={k}: {} vs {}", q[k], LSF_MEAN[k]);
        }
    }

    #[test]
    fn quantised_is_codebook_entry() {
        let (idx, q) = quantise_lsf(&LSF_MEAN);
        // Each split of `q` must equal the codebook entry at idx[split].
        let sub0 = lookup_split(0, idx[0]);
        for j in 0..3 {
            assert_eq!(q[j], sub0[j]);
        }
        let sub1 = lookup_split(1, idx[1]);
        for j in 0..3 {
            assert_eq!(q[3 + j], sub1[j]);
        }
        let sub2 = lookup_split(2, idx[2]);
        for j in 0..4 {
            assert_eq!(q[6 + j], sub2[j]);
        }
    }

    #[test]
    fn indices_within_table_sizes() {
        let (idx, _) = quantise_lsf(&LSF_MEAN);
        assert!((idx[0] as usize) < SPLIT_SIZES[0]);
        assert!((idx[1] as usize) < SPLIT_SIZES[1]);
        assert!((idx[2] as usize) < SPLIT_SIZES[2]);
    }
}
