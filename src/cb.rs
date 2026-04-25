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
//! Each stage's index picks a segment from the codebook memory. The
//! codebook layout for a `cbveclen`-sample target has four regions
//! (RFC §3.6.3 and the reference `getCBvec` in Appendix A.24):
//!
//! - **Base codebook**: `lMem - cbveclen + 1` vectors, one per starting
//!   position in the memory buffer.
//! - **Augmented base** (only when `cbveclen == SUBL`): 20 vectors,
//!   indices `base_size - cbveclen/2 .. base_size`, built by the linear
//!   interpolation described in §3.6.3.3 (with the 5-sample `pi`/`po`
//!   blend weights `[0.0, 0.2, 0.4, 0.6, 0.8]`).
//! - **Expanded codebook**: `lMem - cbveclen + 1` vectors produced by
//!   filtering the memory buffer with the 8-tap `cbfiltersTbl`
//!   (§3.6.3.2).
//! - **Augmented expanded** (only when `cbveclen == SUBL`): 20 vectors
//!   analogous to the augmented-base, but built from the filtered
//!   memory.
//!
//! Gains are successively-rescaled — stage k's gain is quantised
//! relative to stage k-1's dequantised value.

use crate::{CB_LMEM, SUBL};

/// Length of the 8-tap `cbfiltersTbl`, per §3.6.3.2.
const CB_FILTERLEN: usize = 8;
/// Half the filter length — used to offset the filtered output so that
/// the expansion compensates for the filter's group delay.
const CB_HALFFILTERLEN: usize = CB_FILTERLEN / 2;

/// Successive-rescale gain dequantisation (RFC 3951 §3.6.4.2).
///
/// Stage 0 uses `gain_sq5Tbl` (positive-only, 5 bits) with an implicit
/// scale of 1.0. Stages 1 and 2 use `gain_sq4Tbl` (4 bits) and
/// `gain_sq3Tbl` (3 bits) respectively, scaled by the absolute value of
/// the previous stage's dequantised gain (clamped to a floor of 0.1).
///
/// All three tables are transcribed verbatim from RFC 3951 Appendix A.8.
#[allow(clippy::excessive_precision)] // RFC 3951 Appendix A.8 verbatim
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

/// CB expansion filter — verbatim from RFC 3951 Appendix A.8
/// `cbfiltersTbl` (8 taps). Used by the codebook expansion procedure
/// of §3.6.3 when the index selects the expanded region of the
/// adaptive codebook memory. The reference `getCBvec` consumes this
/// table tail-first (`pp1 = &cbfiltersTbl[CB_FILTERLEN-1]; pp1--`).
pub const CB_FILTERS_TBL: [f32; CB_FILTERLEN] = [
    -0.034180, 0.108887, -0.184326, 0.806152, 0.713379, -0.144043, 0.083740, -0.033691,
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

/// Construct the augmented codebook vector corresponding to a given
/// sample delay `index` (20..=39), taken from the tail of `mem`.
///
/// Mirrors `createAugmentedVec` (RFC 3951 Appendix A.12):
///   1. Copy `index` samples from `mem[lMem-index..]` to `cbvec[0..index]`.
///   2. Interpolate the next 5 samples between `mem[lMem-5..]` (`po`)
///      and `mem[lMem-index-5..]` (`pi`) with blend weights
///      `[0.0, 0.2, 0.4, 0.6, 0.8]` for `pi`.
///   3. Copy the final `SUBL-index` samples again from
///      `mem[lMem-index..]`.
pub(crate) fn create_augmented_vec(
    mem: &[f32],
    lmem: usize,
    index: usize,
    cbvec: &mut [f32; SUBL],
) {
    debug_assert!((20..=39).contains(&index));
    let ilow = index - 5;

    // Copy first non-interpolated part (length `index`).
    let pp_start = lmem - index;
    cbvec[..index].copy_from_slice(&mem[pp_start..pp_start + index]);

    // Interpolation: 5 samples, indices ilow..index — RFC 3951 §3.6.3
    // augmented codebook construction (`createAugmentedVec`).
    let alfa1 = 0.2_f32;
    let mut alfa = 0.0_f32;
    let ppo_start = lmem - 5;
    let ppi_start = lmem - index - 5;
    for (offset, j) in (ilow..index).enumerate() {
        let ppo = mem[ppo_start + offset];
        let ppi = mem[ppi_start + offset];
        cbvec[j] = (1.0 - alfa) * ppo + alfa * ppi;
        alfa += alfa1;
    }

    // Copy second non-interpolated part (length SUBL - index).
    let tail = SUBL - index;
    cbvec[index..index + tail].copy_from_slice(&mem[pp_start..pp_start + tail]);
}

/// Apply the 8-tap `cbfiltersTbl` FIR filter to the codebook memory
/// buffer, producing a filtered version the expanded codebook section
/// is drawn from. Matches the reference `getCBvec` inner convolution:
///
/// ```text
///     tempbuff2 = [0] * CB_HALFFILTERLEN || mem || [0] * (CB_HALFFILTERLEN + 1)
///     for n in 0..lMem + CB_HALFFILTERLEN:
///         out[n] = Σ_{j=0..CB_FILTERLEN} tempbuff2[n + j] * cbfiltersTbl[CB_FILTERLEN-1-j]
/// ```
///
/// Returns a length-`lMem` buffer aligned to the original memory
/// (the filter's group delay of 4 samples is compensated).
fn filter_cb_memory(mem: &[f32]) -> Vec<f32> {
    let lmem = mem.len();
    let pad = CB_HALFFILTERLEN;
    let total = lmem + CB_FILTERLEN; // pad + mem + (pad+1), we only need `lmem` outputs

    // Build padded buffer: CB_HALFFILTERLEN zeros, then mem, then tail.
    let mut padded = vec![0.0f32; total + 1];
    padded[pad..pad + lmem].copy_from_slice(mem);

    // Convolve: out[n] = Σ_{j=0..CB_FILTERLEN} padded[n + j] * tbl[CB_FILTERLEN-1-j]
    // which with tbl[CB_FILTERLEN-1-j] traversal is exactly the reference
    // loop (pp1 decrementing from &cbfiltersTbl[CB_FILTERLEN-1]).
    let mut out = vec![0.0f32; lmem];
    for (n, out_n) in out.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for j in 0..CB_FILTERLEN {
            s += padded[n + j] * CB_FILTERS_TBL[CB_FILTERLEN - 1 - j];
        }
        *out_n = s;
    }
    out
}

/// Extract a codebook vector of length `cbveclen` from the `lMem`-sample
/// adaptive codebook memory, for the given codebook `index`.
///
/// Implements the full four-region layout from RFC 3951 §3.6.3 and
/// Appendix A.24 (`getCBvec`): base codebook, augmented base (only when
/// `cbveclen == SUBL`), expanded codebook (FIR-filtered memory),
/// augmented expanded. Indices beyond the legal range wrap modulo the
/// total codebook size, preserving the previous behaviour for malformed
/// packets.
pub fn extract_cbvec_veclen(cb_mem: &[f32], index: u16, cbveclen: usize) -> Vec<f32> {
    let lmem = cb_mem.len();
    let base_size_no_aug = lmem - cbveclen + 1;
    let base_size = if cbveclen == SUBL {
        base_size_no_aug + cbveclen / 2
    } else {
        base_size_no_aug
    };
    // Total codebook size: base + expanded (same layout for expanded).
    let total_size = 2 * base_size;
    let index = (index as usize) % total_size;

    let mut cbvec = vec![0.0f32; cbveclen];

    if index < base_size_no_aug {
        // Region 1: base, non-interpolated. `k = index + cbveclen`,
        // vector = mem[lMem-k .. lMem-k+cbveclen].
        let k = index + cbveclen;
        let start = lmem - k;
        cbvec[..cbveclen].copy_from_slice(&cb_mem[start..start + cbveclen]);
    } else if index < base_size {
        // Region 2: augmented base (only when cbveclen == SUBL).
        // index in [base_size_no_aug .. base_size_no_aug + cbveclen/2).
        // Per the reference: k = 2*(index - base_size_no_aug) + cbveclen,
        // then ihigh = k/2 which is the augmented-vector "index"
        // parameter in createAugmentedVec (20..39).
        let k = 2 * (index - base_size_no_aug) + cbveclen;
        let aug_idx = k / 2; // 20..=39 when cbveclen == 40.
        let mut out_arr = [0.0f32; SUBL];
        // The reference carves the augmented region out of `mem`
        // directly; the `buffer` parameter of createAugmentedVec is
        // `mem + lMem`. We pass `lmem` as the length sentinel.
        create_augmented_vec(cb_mem, lmem, aug_idx, &mut out_arr);
        cbvec[..cbveclen].copy_from_slice(&out_arr[..cbveclen]);
    } else {
        // Regions 3-4: expanded (filtered) codebook.
        let filtered = filter_cb_memory(cb_mem);
        let sub_idx = index - base_size;
        if sub_idx < base_size_no_aug {
            // Region 3: expanded base, non-interpolated.
            let k = sub_idx + cbveclen;
            let start = lmem - k;
            cbvec[..cbveclen].copy_from_slice(&filtered[start..start + cbveclen]);
        } else {
            // Region 4: augmented expanded.
            let k = 2 * (sub_idx - base_size_no_aug) + cbveclen;
            let aug_idx = k / 2;
            let mut out_arr = [0.0f32; SUBL];
            create_augmented_vec(&filtered, lmem, aug_idx, &mut out_arr);
            cbvec[..cbveclen].copy_from_slice(&out_arr[..cbveclen]);
        }
    }

    cbvec
}

/// Extract a 40-sample codebook vector from the 147-sample adaptive
/// codebook memory, using the four-region layout of §3.6.3
/// (base / augmented base / expanded / augmented expanded).
pub fn extract_cbvec(cb_mem: &[f32; CB_LMEM], index: u16) -> [f32; SUBL] {
    let v = extract_cbvec_veclen(cb_mem, index, SUBL);
    let mut out = [0.0f32; SUBL];
    out.copy_from_slice(&v);
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
        for (e, &v_n) in exc.iter_mut().zip(v.iter()) {
            *e += gains[stage] * v_n;
        }
    }
    exc
}

/// Update the adaptive codebook memory: shift left by `SUBL` and
/// append `new_excitation` at the tail.
pub fn update_cb_memory(cb_mem: &mut [f32; CB_LMEM], new_excitation: &[f32; SUBL]) {
    cb_mem.copy_within(SUBL.., 0);
    cb_mem[CB_LMEM - SUBL..].copy_from_slice(new_excitation);
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
    fn cb_filters_bit_exact() {
        // Verbatim RFC 3951 Appendix A.8 `cbfiltersTbl` endpoints.
        assert_eq!(CB_FILTERS_TBL.len(), 8);
        assert_eq!(CB_FILTERS_TBL[0], -0.034180);
        assert_eq!(CB_FILTERS_TBL[3], 0.806152);
        assert_eq!(CB_FILTERS_TBL[4], 0.713379);
        assert_eq!(CB_FILTERS_TBL[7], -0.033691);
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
        // With cbveclen == SUBL == 40 and lMem == 147:
        //   base_size_no_aug = 108, base_size = 128, total = 256.
        // Index 10 is in the base region. Reference formula:
        //   k = index + cbveclen = 50, start = lMem - k = 97.
        //   cbvec[j] = mem[97 + j], i.e. 97..137.
        let mem: [f32; CB_LMEM] = core::array::from_fn(|i| i as f32);
        let v = extract_cbvec(&mem, 10);
        assert_eq!(v[0], 97.0);
        assert_eq!(v[SUBL - 1], 97.0 + (SUBL as f32) - 1.0);
    }

    #[test]
    fn extract_cbvec_augmented_base() {
        // Augmented base region begins at index 108 (base_size_no_aug).
        // Index 108 -> aug_idx = (2*0 + 40)/2 = 20.
        let mem: [f32; CB_LMEM] = core::array::from_fn(|i| i as f32);
        let v_aug = extract_cbvec(&mem, 108);
        // Region layout for aug_idx=20:
        //   j in [0..15): mem[127..142] (first non-interpolated part)
        //   j in [15..20): interpolation between mem[142..147] (po) and
        //                  mem[122..127] (pi) with pi weights
        //                  [0.0, 0.2, 0.4, 0.6, 0.8].
        //   j in [20..40): mem[127..147] (second non-interpolated part)
        for (j, &got) in v_aug.iter().take(15).enumerate() {
            assert_eq!(got, 127.0 + j as f32);
        }
        let pi_w = [0.0_f32, 0.2, 0.4, 0.6, 0.8];
        for (k, &w) in pi_w.iter().enumerate() {
            let expected = (1.0 - w) * (142.0 + k as f32) + w * (122.0 + k as f32);
            let got = v_aug[15 + k];
            assert!(
                (got - expected).abs() < 1e-4,
                "aug-base j={} got {} want {}",
                15 + k,
                got,
                expected
            );
        }
        for (j, &got) in v_aug[20..40].iter().enumerate() {
            assert_eq!(got, 127.0 + j as f32);
        }
    }

    #[test]
    fn extract_cbvec_augmented_near_top() {
        // Augmented base index 127 -> last augmented vector, aug_idx = 39.
        let mem: [f32; CB_LMEM] = core::array::from_fn(|i| i as f32);
        let v = extract_cbvec(&mem, 127);
        // For aug_idx=39: first non-interpolated part has 39 samples
        // (j=0..34 is the plain copy; j=34..39 is interpolated).
        for (j, &got) in v.iter().take(34).enumerate() {
            assert_eq!(got, (CB_LMEM - 39) as f32 + j as f32);
        }
        // Last sample (SUBL - 39 = 1) comes from mem[lMem - 39] = mem[108].
        assert_eq!(v[39], (CB_LMEM - 39) as f32);
    }

    #[test]
    fn extract_cbvec_expanded_base() {
        // base_size = 128, so index 128 starts the expanded region.
        // The expanded codebook is FIR-filtered memory, so values will
        // differ from plain mem — just check it's finite and bounded.
        let mem: [f32; CB_LMEM] = core::array::from_fn(|i| (i as f32 * 0.1).sin());
        let v = extract_cbvec(&mem, 128);
        for &x in v.iter() {
            assert!(x.is_finite());
            assert!(x.abs() < 10.0);
        }
    }

    #[test]
    fn create_augmented_vec_interpolation_weights() {
        // Sanity: interpolated region blends pi/po with weights
        // [0.0, 0.2, 0.4, 0.6, 0.8]. With po == 1.0 and pi == 0.0,
        // cbvec[ilow..index] should equal those weights.
        let lmem = CB_LMEM;
        let mut mem = [0.0f32; CB_LMEM];
        // Set the last 5 "po" samples to 1.0; leave "pi" samples 0.
        mem[lmem - 5..lmem].fill(1.0);
        let index = 25;
        let mut out = [0.0f32; SUBL];
        create_augmented_vec(&mem, lmem, index, &mut out);
        // Positions ilow..index are the interpolated slots.
        let expected = [0.0_f32, 0.2, 0.4, 0.6, 0.8];
        for (k, &w) in expected.iter().enumerate() {
            let v = out[index - 5 + k];
            assert!((v - (1.0 - w)).abs() < 1e-5);
        }
    }

    #[test]
    fn update_mem_shifts_correctly() {
        let mut mem: [f32; CB_LMEM] = core::array::from_fn(|i| i as f32);
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
