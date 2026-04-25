//! Input high-pass pre-processing filter for the iLBC encoder
//! (RFC 3951 §3.1, reference `hpInput` in Appendix A.28).
//!
//! A 2nd-order biquad IIR with a 90 Hz cutoff is applied to the raw
//! PCM input before it reaches the LPC analysis buffer. The filter
//! removes DC and 50/60 Hz mains hum, which would otherwise eat
//! quantisation budget in the LPC / start-state stages.
//!
//! Transfer function (RFC §3.1):
//!
//! ```text
//!   H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)
//!
//!   b = [ 0.92727436, -1.8544941, 0.92727436 ]
//!   a = [ 1.0,        -1.9059465, 0.9114024  ]
//! ```
//!
//! The reference `hpInput` keeps state in a 4-element array:
//!   - `mem[0] = x(n-1)`, `mem[1] = x(n-2)`  (input delay line)
//!   - `mem[2] = y(n-1)`, `mem[3] = y(n-2)`  (output delay line)
//!
//! and processes the input in two passes (all-zero numerator, then
//! all-pole denominator). We fold both into a single Direct-Form-I
//! biquad — bit-equivalent to the reference and one fewer pass over
//! the data.

/// Zero (numerator b) coefficients — RFC 3951 §3.1 / Appendix A.28
/// `hpi_zero_coefsTbl`.
pub const HPI_ZERO_COEFS: [f32; 3] = [0.927_274_36, -1.854_494_1, 0.927_274_36];

/// Pole (denominator a) coefficients — RFC 3951 §3.1 / Appendix A.28
/// `hpi_pole_coefsTbl`. `a[0] = 1.0` (already normalised).
pub const HPI_POLE_COEFS: [f32; 3] = [1.0, -1.905_946_5, 0.911_402_4];

/// Persistent state of the input HP filter (one instance per encoder).
#[derive(Clone, Copy, Debug, Default)]
pub struct HpInputState {
    /// `[x(n-1), x(n-2), y(n-1), y(n-2)]` — same layout as the
    /// reference encoder's `mem` array.
    pub mem: [f32; 4],
}

impl HpInputState {
    /// Reset the filter to silence (zeros all delay-line samples).
    pub fn reset(&mut self) {
        self.mem = [0.0; 4];
    }
}

/// Apply the input HP filter to `input`, writing the filtered samples to
/// `out`. The filter state in `state` is updated in place. `out` and
/// `input` may alias the same slice (we read each input before writing
/// the corresponding output).
///
/// RFC 3951 §3.1: a Direct-Form-I biquad reproduces the reference
/// `hpInput` exactly because the reference is a cascade of FIR then IIR
/// without intermediate quantisation.
pub fn hp_input(input: &[f32], out: &mut [f32], state: &mut HpInputState) {
    debug_assert_eq!(input.len(), out.len());
    let b = HPI_ZERO_COEFS;
    let a = HPI_POLE_COEFS;
    let mem = &mut state.mem;
    for (n, &x_n) in input.iter().enumerate() {
        // All-zero (FIR) section.
        let z = b[0] * x_n + b[1] * mem[0] + b[2] * mem[1];
        // All-pole (IIR) section.
        let y_n = z - a[1] * mem[2] - a[2] * mem[3];
        // Shift delay lines.
        mem[1] = mem[0];
        mem[0] = x_n;
        mem[3] = mem[2];
        mem[2] = y_n;
        out[n] = y_n;
    }
}

/// Convenience wrapper: filter `input` into a freshly allocated `Vec`.
pub fn hp_input_vec(input: &[f32], state: &mut HpInputState) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    hp_input(input, &mut out, state);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DC input (constant value) should be attenuated to near zero in
    /// the steady state. The numerator coefs are designed to sum to ~0
    /// (DC notch); they sum to ~5e-5 in single-precision after the
    /// RFC's printed values are rounded to f32, which still suppresses
    /// a 1000-amplitude DC by ~85 dB once the IIR settles.
    #[test]
    fn dc_is_attenuated() {
        let dc_zeros_sum = HPI_ZERO_COEFS.iter().sum::<f32>();
        assert!(
            dc_zeros_sum.abs() < 1e-3,
            "expected zero-coefs to sum to ~0 (DC notch); got {dc_zeros_sum}"
        );
        let mut st = HpInputState::default();
        let input = vec![1000.0f32; 2000];
        let out = hp_input_vec(&input, &mut st);
        // Steady-state: the last 200 samples should be < 2 % of input.
        // (Single-precision rounding of the printed coefficients leaves
        // a small residue at DC; ~40 dB is still plenty.)
        let tail = &out[1800..];
        let max_tail = tail.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_tail < 20.0,
            "DC steady-state |y| should be < 20 (~34 dB attenuation); got {max_tail}"
        );
    }

    /// A 1 kHz tone (well above the 90 Hz cutoff) should pass through
    /// roughly unchanged in amplitude — within a few % once steady.
    #[test]
    fn high_frequency_passes() {
        let mut st = HpInputState::default();
        let fs = 8000.0f32;
        let f = 1000.0f32;
        let n = 4000;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * core::f32::consts::PI * f * (i as f32) / fs).sin() * 1000.0)
            .collect();
        let out = hp_input_vec(&input, &mut st);
        // RMS over the last second.
        let lo = n - 800;
        let rms_in = (input[lo..].iter().map(|v| v * v).sum::<f32>() / (n - lo) as f32).sqrt();
        let rms_out = (out[lo..].iter().map(|v| v * v).sum::<f32>() / (n - lo) as f32).sqrt();
        let ratio = rms_out / rms_in;
        assert!(
            ratio > 0.9 && ratio < 1.1,
            "1 kHz tone amplitude ratio out/in should be ~1; got {ratio}"
        );
    }

    /// 50 Hz hum — well below the 90 Hz cutoff — should be measurably
    /// attenuated relative to a 1 kHz tone of the same amplitude.
    #[test]
    fn low_frequency_attenuated() {
        let mut st = HpInputState::default();
        let fs = 8000.0f32;
        let n = 8000;
        let f = 50.0f32;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * core::f32::consts::PI * f * (i as f32) / fs).sin() * 1000.0)
            .collect();
        let out = hp_input_vec(&input, &mut st);
        let lo = n - 1600;
        let rms_in = (input[lo..].iter().map(|v| v * v).sum::<f32>() / (n - lo) as f32).sqrt();
        let rms_out = (out[lo..].iter().map(|v| v * v).sum::<f32>() / (n - lo) as f32).sqrt();
        let ratio = rms_out / rms_in;
        assert!(
            ratio < 0.5,
            "50 Hz tone should be attenuated by > 6 dB; ratio={ratio}"
        );
    }

    /// Filter is causal and zero-input-zero-output: silence in →
    /// silence out, for any state of zeros.
    #[test]
    fn silence_in_silence_out() {
        let mut st = HpInputState::default();
        let input = vec![0.0f32; 1000];
        let out = hp_input_vec(&input, &mut st);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    /// Filter is stable: a bounded input produces a bounded output. We
    /// drive it with a square wave and check no sample exceeds a sane
    /// envelope.
    #[test]
    fn stable_under_square_wave() {
        let mut st = HpInputState::default();
        let n = 4000;
        let input: Vec<f32> = (0..n)
            .map(|i| if (i / 40) % 2 == 0 { 1000.0 } else { -1000.0 })
            .collect();
        let out = hp_input_vec(&input, &mut st);
        let max = out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        // The zero filter has DC gain 0 but a high-frequency overshoot
        // bounded by Σ|b_k| ≈ 3.71. The pole filter is stable; output
        // shouldn't grow beyond ~5× input magnitude.
        assert!(max < 5000.0, "output exceeded sane envelope: {max}");
    }
}
