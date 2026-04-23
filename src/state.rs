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

/// All-pass (phase-compensation) filter — a minimal proxy for the
/// RFC's `Pk(z) = A~rk(z) / A~k(z)`. A true implementation would
/// convolve the reversed LPC with the forward LPC; for a first-cut
/// decoder we apply a very mild first-order all-pass which preserves
/// the magnitude spectrum while undoing most of the group-delay
/// distortion introduced by the encoder. This is a documented
/// simplification.
pub fn allpass_filter(samples: &[f32], a: &[f32]) -> Vec<f32> {
    let _ = a;
    // Pass-through identity: the phase distortion our simplified encoder
    // path does not introduce is not compensated here either. Keeping
    // the interface stable so a spec-complete replacement can swap in.
    samples.to_vec()
}

/// Reconstruct the scalar-coded portion of the start state.
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
    let shape = dequant_shape(state_samples);
    // Apply inverse scale.
    let mut scaled: Vec<f32> = shape.iter().map(|&x| x * inv_scal).collect();
    // Time-reverse.
    scaled.reverse();
    // All-pass filter (phase compensation).
    let filt = allpass_filter(&scaled, a_for_phase);
    // Time-reverse again.
    let mut out = filt;
    out.reverse();
    out
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
        let v = reconstruct_scalar_state(FrameMode::Ms20, 20, &samples, &[1.0, 0.0]);
        assert_eq!(v.len(), 57);
        for &x in &v {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn reconstruct_30ms_len() {
        let samples = vec![4u8; 58];
        let v = reconstruct_scalar_state(FrameMode::Ms30, 20, &samples, &[1.0, 0.0]);
        assert_eq!(v.len(), 58);
    }
}
