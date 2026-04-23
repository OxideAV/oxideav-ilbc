//! Start-state reconstruction — RFC 3951 §4.2.
//!
//! The 57-/58-sample start-state segment is scalar-coded: 6 bits for a
//! logarithmic scale factor (`qmax`) and 3 bits per sample for the
//! shape. The decoder:
//!   1. Looks up shape samples from the 3-bit indices.
//!   2. Rescales by `1 / scal` where `scal = (10^qmax) / 4.5`, i.e.
//!      multiplies by `maxVal = (10^qmax)/4.5` as in the reference
//!      `StateConstructW` (RFC 3951 Appendix A.43/44).
//!   3. Time-reverses the shape vector, pads with zeros to `2·N`,
//!      filters with the LPC-derived all-pass `Pk(z)=A~rk(z)/A~k(z)`,
//!      and folds the two halves back together (circular convolution).
//!   4. Forms the 80-sample state vector: the 23-/22-sample remainder
//!      is decoded from the first adaptive-codebook sub-block, with
//!      ordering determined by the `position` bit.
//!
//! Table references:
//! - Shape dequantisation uses the verbatim `state_sq3Tbl` from
//!   RFC 3951 Appendix A.8.
//! - Log-magnitude dequantisation uses `state_frgqTbl` (also A.8).

use crate::FrameMode;

/// 3-bit start-state shape dequantisation table, verbatim from
/// RFC 3951 Appendix A.8 `state_sq3Tbl`.
pub const STATE_SQ3_TBL: [f32; 8] = [
    -3.719849, -2.177490, -1.130005, -0.309692, 0.444214, 1.329712, 2.436279, 3.983887,
];

/// 6-bit first-residual-gain quantisation table, verbatim from
/// RFC 3951 Appendix A.8 `state_frgqTbl`.
pub const STATE_FRGQ_TBL: [f32; 64] = [
    1.000085, 1.071695, 1.140395, 1.206868, 1.277188, 1.351503, 1.429380, 1.500727, 1.569049,
    1.639599, 1.707071, 1.781531, 1.840799, 1.901550, 1.956695, 2.006750, 2.055474, 2.102787,
    2.142819, 2.183592, 2.217962, 2.257177, 2.295739, 2.332967, 2.369248, 2.402792, 2.435080,
    2.468598, 2.503394, 2.539284, 2.572944, 2.605036, 2.636331, 2.668939, 2.698780, 2.729101,
    2.759786, 2.789834, 2.818679, 2.848074, 2.877470, 2.906899, 2.936655, 2.967804, 3.000115,
    3.033367, 3.066355, 3.104231, 3.141499, 3.183012, 3.222952, 3.265433, 3.308441, 3.350823,
    3.395275, 3.442793, 3.490801, 3.542514, 3.604064, 3.666050, 3.740994, 3.830749, 3.938770,
    4.101764,
];

/// Decode the inverse scale factor, i.e. the multiplier the decoder
/// applies to each shape sample before the all-pass filter.
///
/// RFC 3951 §3.5.2 / §4.2 and the reference `StateConstructW`
/// (Appendix A.44):
///
/// ```text
///     maxVal = state_frgqTbl[idxForMax]        // log10 magnitude
///     qmax   = 10^maxVal                       // linear magnitude
///     scal   = 4.5 / qmax                      // encoder's scaling
///     1/scal = qmax / 4.5 = 10^maxVal / 4.5    // what we return
/// ```
///
/// `scale_idx` is 6 bits; indices outside the 0..63 range are clamped.
pub fn decode_scale(scale_idx: u8) -> f32 {
    let idx = (scale_idx as usize) & 0x3F;
    let max_val = STATE_FRGQ_TBL[idx];
    // 1/scal = 10^max_val / 4.5 per RFC 3951 Appendix A.44 (StateConstructW).
    10f32.powf(max_val) / 4.5
}

/// Dequantise the shape samples from 3-bit indices. Output has the
/// same length as `state_samples`.
pub fn dequant_shape(state_samples: &[u8]) -> Vec<f32> {
    state_samples
        .iter()
        .map(|&idx| STATE_SQ3_TBL[(idx & 0x7) as usize])
        .collect()
}

/// Circular convolution with the RFC 3951 all-pass `Pk(z) = A~rk(z)/A~k(z)`.
///
/// `input` has length `2·N`, the first `N` samples are the time-reversed
/// scaled shape and the remaining `N` are zeros (per §4.2). `a` is the
/// LPC denominator `[1, a1..a_order]`. The filter is applied as:
///
/// ```text
///     numerator(z) = z^{-order} + Σ a_{k+1} z^{k - (order-1)}
///                  = reverse(a[1..=order]) followed by a[0] = 1
///
///     fout = AllZero(input, numerator) followed by AllPole(·, a)
/// ```
///
/// This matches the reference `ZeroPoleFilter` chain used by the
/// encoder's `StateSearchW` and the decoder's `StateConstructW`
/// (RFC 3951 Appendix A.18/A.44).
pub fn allpass_zero_pole(input: &[f32], a: &[f32]) -> Vec<f32> {
    let order = a.len() - 1;
    debug_assert!(order > 0, "LPC order must be >= 1");
    debug_assert!(a[0].is_finite());

    // numerator[k] = a[order - k] for k=0..order-1, numerator[order] = a[0].
    // With a[0] = 1.0 this is `reverse(a[1..=order])` followed by 1.0.
    let mut numerator = vec![0.0f32; order + 1];
    for k in 0..order {
        numerator[k] = a[order - k];
    }
    numerator[order] = a[0];

    let len = input.len();
    let mut out = vec![0.0f32; len];

    // AllZeroFilter: out[n] = Σ_{k=0..=order} numerator[k] * input[n-k],
    // with zero history before n=0.
    for n in 0..len {
        let mut s = numerator[0] * input[n];
        for k in 1..=order {
            let idx = n as isize - k as isize;
            if idx >= 0 {
                s += numerator[k] * input[idx as usize];
            }
        }
        out[n] = s;
    }

    // AllPoleFilter in place: out[n] -= Σ_{k=1..=order} a[k] * out[n-k].
    for n in 0..len {
        for k in 1..=order {
            let idx = n as isize - k as isize;
            if idx >= 0 {
                out[n] -= a[k] * out[idx as usize];
            }
        }
    }

    out
}

/// Apply the circular all-pass phase-compensation filter from
/// RFC 3951 §3.5.2 / §4.2.
///
/// Expects `shape` of length `N` (the inverse-scaled, time-reversed
/// shape vector from the dequantiser) and the order-10 LPC
/// `a = [1, a1..a10]` taken from the block where the start state
/// begins.
///
/// Returns a length-`N` vector `out` such that
///
/// ```text
///     out(k) = fout(N-1-k) + fout(2N-1-k),  k = 0..N-1
/// ```
///
/// where `fout = Pk(z) · [shape | zeros(N)]` (RFC §4.2 closing
/// equations). The caller is responsible for applying the final
/// outer time-reverse around this whole call to recover the
/// start-state sample order.
pub fn allpass_filter(shape: &[f32], a: &[f32]) -> Vec<f32> {
    let n = shape.len();
    // Build the 2N input: first half = shape, second half = zeros.
    let mut padded = vec![0.0f32; 2 * n];
    padded[..n].copy_from_slice(shape);
    let fout = allpass_zero_pole(&padded, a);

    // Fold: out(k) = fout(N-1-k) + fout(2N-1-k).
    let mut out = vec![0.0f32; n];
    for k in 0..n {
        out[k] = fout[n - 1 - k] + fout[2 * n - 1 - k];
    }
    out
}

/// Reconstruct the scalar-coded portion of the start state.
///
/// Mirrors `StateConstructW` (RFC 3951 Appendix A.44):
///   1. `tmp[k] = (1/scal) · state_sq3Tbl[idxVec[N-1-k]]`  (time-reverse + scale)
///   2. Pad with `N` zeros to length `2N`.
///   3. Filter with the zero-pole all-pass using the block's LPC.
///   4. `out[k] = fout[N-1-k] + fout[2N-1-k]`.
///
/// Returns a vector of length `mode.state_short_len()` (57 or 58).
pub fn reconstruct_scalar_state(
    mode: FrameMode,
    scale_idx: u8,
    state_samples: &[u8],
    a_for_phase: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(state_samples.len(), mode.state_short_len());
    let inv_scal = decode_scale(scale_idx);
    let n = state_samples.len();

    // Build `in(0..N-1)` = time-reversed (scaled) shape.
    let mut reversed_scaled = vec![0.0f32; n];
    for k in 0..n {
        let tmpi = n - 1 - k;
        let idx = (state_samples[tmpi] & 0x7) as usize;
        reversed_scaled[k] = inv_scal * STATE_SQ3_TBL[idx];
    }

    // Apply the all-pass / fold. `allpass_filter` handles the zero
    // padding and the `out(k) = f(N-1-k) + f(2N-1-k)` fold.
    allpass_filter(&reversed_scaled, a_for_phase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_monotone_in_idx() {
        // STATE_FRGQ_TBL is monotone-increasing in idx (RFC 3951 A.8),
        // so 1/scal = 10^maxVal / 4.5 must also rise with scale_idx.
        let a = decode_scale(0);
        let b = decode_scale(32);
        let c = decode_scale(63);
        assert!(
            a < b && b < c,
            "1/scal not monotone-increasing: {a}, {b}, {c}"
        );
    }

    #[test]
    fn scale_rfc_anchors() {
        // 10^1.000085 / 4.5 ≈ 10.00196... / 4.5 ≈ 2.223
        let lo = decode_scale(0);
        assert!((lo - 10f32.powf(1.000085) / 4.5).abs() < 1e-6);
        // 10^4.101764 / 4.5 ≈ 12635 / 4.5 ≈ 2807.8
        let hi = decode_scale(63);
        assert!((hi - 10f32.powf(4.101764) / 4.5).abs() < 1e-2);
    }

    #[test]
    fn shape_table_length() {
        assert_eq!(STATE_SQ3_TBL.len(), 8);
    }

    #[test]
    fn state_sq3_bit_exact() {
        // Verbatim RFC 3951 Appendix A.8 `state_sq3Tbl` values.
        assert_eq!(STATE_SQ3_TBL[0], -3.719849);
        assert_eq!(STATE_SQ3_TBL[3], -0.309692);
        assert_eq!(STATE_SQ3_TBL[4], 0.444214);
        assert_eq!(STATE_SQ3_TBL[7], 3.983887);
    }

    #[test]
    fn state_frgq_bit_exact() {
        // Verbatim RFC 3951 Appendix A.8 `state_frgqTbl` endpoints.
        assert_eq!(STATE_FRGQ_TBL.len(), 64);
        assert_eq!(STATE_FRGQ_TBL[0], 1.000085);
        assert_eq!(STATE_FRGQ_TBL[63], 4.101764);
    }

    #[test]
    fn reconstruct_20ms_len() {
        let samples = vec![4u8; 57];
        // Full-order LPC, trivial except for a[0]=1.0 (guaranteed stable).
        let mut a = [0.0f32; 11];
        a[0] = 1.0;
        let v = reconstruct_scalar_state(FrameMode::Ms20, 20, &samples, &a);
        assert_eq!(v.len(), 57);
        for &x in &v {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn reconstruct_30ms_len() {
        let samples = vec![4u8; 58];
        let mut a = [0.0f32; 11];
        a[0] = 1.0;
        let v = reconstruct_scalar_state(FrameMode::Ms30, 20, &samples, &a);
        assert_eq!(v.len(), 58);
    }

    #[test]
    fn allpass_identity_with_trivial_lpc() {
        // With a = [1, 0, ..., 0] the numerator is the reversed kernel
        // [0, ..., 0, 1] = z^{-order}, so the all-pass has unity gain at
        // DC but delays by `order`. The fold `out[k] = fout[N-1-k] +
        // fout[2N-1-k]` with fout = delayed(input | zeros) yields a
        // predictable pattern; just verify finiteness and energy bound.
        let mut a = [0.0f32; 11];
        a[0] = 1.0;
        let shape: Vec<f32> = (0..57).map(|i| ((i as f32) * 0.1).sin()).collect();
        let out = allpass_filter(&shape, &a);
        assert_eq!(out.len(), shape.len());
        let e_in: f32 = shape.iter().map(|v| v * v).sum();
        let e_out: f32 = out.iter().map(|v| v * v).sum();
        // Energy should be in the same order of magnitude (pure delay
        // filter is lossless up to boundary effects).
        assert!(e_out > 0.0);
        assert!(e_out < 4.0 * e_in + 1.0);
    }

    #[test]
    fn allpass_zero_pole_stable_output() {
        // RFC-shaped LPC: moderate short-term predictor.
        let mut a = [0.0f32; 11];
        a[0] = 1.0;
        a[1] = -0.6;
        a[2] = 0.15;
        a[3] = -0.03;
        let inp: Vec<f32> = (0..100).map(|i| (i as f32).sin()).collect();
        let out = allpass_zero_pole(&inp, &a);
        assert_eq!(out.len(), inp.len());
        for &v in &out {
            assert!(v.is_finite());
            assert!(v.abs() < 1e6);
        }
    }
}
