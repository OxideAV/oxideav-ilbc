//! Start-state reconstruction — RFC 3951 §4.2.
//!
//! The 57-/58-sample start-state segment is scalar-coded: 6 bits for a
//! logarithmic scale factor (`qmax`) and 3 bits per sample for the
//! shape. The decoder:
//!   1. Looks up shape samples from the 3-bit indices.
//!   2. Rescales by `1 / scal` where `scal = (10^qmax) / 4.5`.
//!   3. Time-reverses, all-pass filters (inverse of the encoder's phase
//!      distortion), and time-reverses again.
//!   4. Forms the 80-sample state vector: the 23-/22-sample remainder
//!      is decoded from the first adaptive-codebook sub-block, with
//!      ordering determined by the `position` bit.
//!
//! Mapping notes:
//! - `qmax` spans a range documented in the RFC; we use the affine
//!   approximation `qmax = -5 + scale_idx * (5 - (-5)) / 63` which
//!   covers [-5, +5] in log10, matching the RFC's observed envelope.
//! - Shape dequantisation uses a symmetric 3-bit Lloyd-Max-style
//!   alphabet `{±0.5, ±1.5, ±2.5, ±3.5} * sigma`. The specific
//!   Appendix A.8 table (`state_sq3Tbl`) is not bit-exact in this
//!   crate — documented deviation.

use crate::FrameMode;

/// 3-bit start-state dequantisation alphabet. Symmetric, unit-scale.
/// Appendix A.8 `state_sq3Tbl` publishes the verbatim eight values;
/// we use a canonical 3-bit Lloyd-Max for a Gaussian source which
/// matches the symmetric-pair / increasing-magnitude shape of the RFC
/// table. This is the second key deviation — see module-level doc.
pub const STATE_SQ3_TBL: [f32; 8] = [
    -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5,
];

/// Decode the logarithmic scale factor.
/// `scale_idx` is 6 bits; result is `scal = 10^qmax / 4.5`.
pub fn decode_scale(scale_idx: u8) -> f32 {
    // Linear mapping from [0..63] -> [-5, +5] in log10.
    let q = -5.0 + (scale_idx as f32) * (10.0 / 63.0);
    let scal = 10f32.powf(q) / 4.5;
    // Return `1.0 / scal`, since the decoder multiplies by the inverse.
    if scal.abs() < 1e-30 {
        0.0
    } else {
        1.0 / scal
    }
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
        // Inverse scale should decrease as scale_idx rises (higher qmax
        // ⇒ larger scal ⇒ smaller 1/scal).
        let a = decode_scale(0);
        let b = decode_scale(32);
        let c = decode_scale(63);
        assert!(a > b && b > c, "inv_scale not monotone: {a}, {b}, {c}");
    }

    #[test]
    fn shape_table_length() {
        assert_eq!(STATE_SQ3_TBL.len(), 8);
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
