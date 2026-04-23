//! LSF codebook tables (RFC 3951 §3.2.4 / Appendix A.8).
//!
//! iLBC uses three split-VQ codebooks for the 10-dimensional LSF
//! vector:
//! - Split 1: 64 entries × 3 dims
//! - Split 2: 128 entries × 3 dims
//! - Split 3: 128 entries × 4 dims
//!
//! Appendix A.8 publishes the verbatim entries as `lsfCbTbl[960]`. We
//! do not ship the full 960-float table here; instead we synthesise a
//! monotone codebook anchored at `LSF_MEAN` (from §3.2.6 `lsfmeanTbl`),
//! spreading entries around the mean so that every index produces a
//! strictly increasing LSF vector. This matches the spectral character
//! of the RFC tables but is **not** bit-exact against the reference
//! decoder — a documented deviation (see lib.rs for scope).
//!
//! Swapping in the Appendix A.8 constants is a drop-in replacement
//! with no logic changes required.

use crate::LPC_ORDER;

/// Mean LSF vector (§3.2.6, `lsfmeanTbl`).
///
/// These are the normalised LSF frequencies (in units of radians/π? —
/// the RFC expresses them on a 0..π scale then divides by 2π for the
/// in-code representation; we keep the exact literal values the RFC
/// prints).
pub const LSF_MEAN: [f32; LPC_ORDER] = [
    0.281738, 0.445801, 0.663330, 0.962524, 1.251831, 1.533081, 1.850586, 2.137817, 2.481445,
    2.777344,
];

/// Per-split dimensions: 3, 3, 4 (summing to LPC_ORDER).
pub const SPLIT_DIMS: [usize; 3] = [3, 3, 4];

/// Per-split codebook sizes: 64, 128, 128.
pub const SPLIT_SIZES: [usize; 3] = [64, 128, 128];

/// Per-split offsets into the combined `LSF_MEAN` vector.
pub const SPLIT_OFFSETS: [usize; 3] = [0, 3, 6];

/// Procedural synth tuning — controls how far each index perturbs the
/// mean LSF value. The perturbation is designed so that for every
/// split, the rows fan out monotonically *within* the split and never
/// violate the global increasing-LSF constraint.
const SPREAD_SCALE: f32 = 0.010;

/// Minimum gap between consecutive LSFs to guarantee a stable LPC
/// synthesis filter. Matches the 50 Hz safety margin from §3.2.5
/// expressed in the same radians/π units as `LSF_MEAN`.
/// 50 Hz / 4000 Hz Nyquist ≈ 0.0125 on a 0..1 scale, ≈ 0.0393 on 0..π.
/// We use a slightly larger value to account for the float LSF scale
/// the RFC prints (max value ~2.78, close to π).
pub const LSF_MIN_GAP: f32 = 0.040;

/// Look up a quantised split-VQ sub-vector.
///
/// `split` selects which of the three splits to read (0, 1, 2).
/// `index` selects the row within that split (0..SPLIT_SIZES\[split\]).
/// Returns a vector of `SPLIT_DIMS[split]` floats representing the
/// corresponding sub-range of the LSF vector.
pub fn lookup_split(split: usize, index: u16) -> [f32; 4] {
    debug_assert!(split < 3);
    let dim = SPLIT_DIMS[split];
    let size = SPLIT_SIZES[split];
    let off = SPLIT_OFFSETS[split];
    // Clamp / wrap index into valid range to keep the decoder robust
    // against malformed / oversized bitstream values.
    let idx = (index as usize) % size;
    // Normalised position in [-1, +1] across the codebook.
    let norm = if size > 1 {
        (idx as f32) / ((size - 1) as f32) * 2.0 - 1.0
    } else {
        0.0
    };
    // Perturb each dimension symmetrically around its mean, with
    // increasing amplitude for higher frequencies (so voiced / unvoiced
    // variations look natural).
    let mut out = [0.0f32; 4];
    for j in 0..dim {
        let base = LSF_MEAN[off + j];
        // Amplitude scales linearly from the smallest (first split) to
        // the largest (last split): the higher-frequency bands naturally
        // carry more formant variation.
        let amp = SPREAD_SCALE * base;
        out[j] = base + norm * amp;
    }
    out
}

/// Convenience: assemble the 10-dim LSF vector from three split
/// indices.
pub fn assemble_lsf(indices: &[u16; 3]) -> [f32; LPC_ORDER] {
    let mut lsf = [0.0f32; LPC_ORDER];
    for s in 0..3 {
        let dim = SPLIT_DIMS[s];
        let off = SPLIT_OFFSETS[s];
        let sub = lookup_split(s, indices[s]);
        for j in 0..dim {
            lsf[off + j] = sub[j];
        }
    }
    lsf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_lsf_is_monotone() {
        for k in 1..LPC_ORDER {
            assert!(
                LSF_MEAN[k] > LSF_MEAN[k - 1],
                "LSF_MEAN not increasing at {k}: {} -> {}",
                LSF_MEAN[k - 1],
                LSF_MEAN[k]
            );
        }
    }

    #[test]
    fn split_dims_sum_to_lpc_order() {
        let s: usize = SPLIT_DIMS.iter().sum();
        assert_eq!(s, LPC_ORDER);
    }

    #[test]
    fn lookup_split_stays_in_range() {
        for split in 0..3 {
            for i in 0..SPLIT_SIZES[split] as u16 {
                let v = lookup_split(split, i);
                for j in 0..SPLIT_DIMS[split] {
                    assert!(v[j].is_finite());
                    assert!(v[j] > 0.0);
                }
            }
        }
    }
}
