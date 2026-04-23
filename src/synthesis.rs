//! LPC synthesis, pitch-emphasis post-filter, and packet-loss
//! concealment — RFC 3951 §4.5, §4.7, §4.8.
//!
//! The synthesis filter is 10th-order all-pole:
//!
//! ```text
//!     y(n) = x(n) - Σ a[i] * y(n-i),  i = 1..10
//! ```
//!
//! After synthesis we apply a mild pitch-emphasis post-filter as a
//! stand-in for the RFC's more elaborate enhancer (§4.6). This keeps
//! the decoder's output perceptually sensible without the full six-
//! PSSQ combiner — a documented scope deviation.
//!
//! PLC (§4.5): if a frame is lost (or the empty-frame indicator is
//! set), we emit a dampened extrapolation from the previous block's
//! excitation and filter state.

use crate::{FrameMode, LPC_ORDER, SUBL};

/// Enhancer polyphase interpolation filter — verbatim from RFC 3951
/// Appendix A.8 `polyphaserTbl`. 4-phase filter of length 7 per
/// phase (ENH_UPS0=4, 2*ENH_FL0+1=7), total 28 taps. Used by the
/// §4.6 per-block periodic enhancer to upsample the candidate by 4×.
/// Not yet wired into the current (simplified) enhancer path.
pub const POLYPHASER_TBL: [f32; 28] = [
    0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.015625, -0.076904,
    0.288330, 0.862061, -0.106445, 0.018799, -0.015625, 0.023682, -0.124268, 0.601563, 0.601563,
    -0.124268, 0.023682, -0.023682, 0.018799, -0.106445, 0.862061, 0.288330, -0.076904, 0.015625,
    -0.018799,
];

/// Enhancer sub-block centre positions — verbatim from RFC 3951
/// Appendix A.8 `enh_plocsTbl`. Eight centres spaced 80 samples apart
/// across the two-frame enhancer window.
pub const ENH_PLOCS_TBL: [f32; 8] = [
    40.0, 120.0, 200.0, 280.0, 360.0, 440.0, 520.0, 600.0,
];

/// LPC synthesis filter state (per-frame).
#[derive(Clone)]
pub struct SynthState {
    /// Filter memory (last LPC_ORDER output samples, reversed).
    pub mem: [f32; LPC_ORDER],
    /// Last frame's final LPC (for PLC).
    pub last_a: [f32; LPC_ORDER + 1],
    /// Last frame's final excitation amplitude RMS (for PLC attenuation).
    pub last_rms: f32,
    /// Consecutive PLC frame count — drives attenuation.
    pub plc_count: u32,
    /// Post-filter memory for the pitch-emphasis pass.
    pub post_mem: f32,
    /// Simple pseudorandom seed for PLC innovation.
    pub plc_seed: u32,
}

impl SynthState {
    pub fn new() -> Self {
        let mut last_a = [0.0f32; LPC_ORDER + 1];
        last_a[0] = 1.0;
        Self {
            mem: [0.0; LPC_ORDER],
            last_a,
            last_rms: 0.0,
            plc_count: 0,
            post_mem: 0.0,
            plc_seed: 0x1234_5678,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for SynthState {
    fn default() -> Self {
        Self::new()
    }
}

/// One-subblock LPC synthesis. Updates `mem` in place.
pub fn synthesise(
    excitation: &[f32; SUBL],
    a: &[f32; LPC_ORDER + 1],
    mem: &mut [f32; LPC_ORDER],
    out: &mut [f32; SUBL],
) {
    for n in 0..SUBL {
        let mut s = excitation[n];
        for k in 1..=LPC_ORDER {
            // mem[k-1] is y(n-k); we'll refresh it below.
            s -= a[k] * mem[k - 1];
        }
        out[n] = s;
        // Shift memory: y(n) becomes the new mem[0], prior entries move.
        for k in (1..LPC_ORDER).rev() {
            mem[k] = mem[k - 1];
        }
        mem[0] = s;
    }
}

/// Mild pitch-emphasis post-filter approximating RFC §4.8.
/// y_post(n) = y(n) + 0.25 * y_post(n-1). Single-pole low-pass that
/// removes HF spectral tilt on telephone-band speech. The filter
/// memory lives on the `SynthState`.
pub fn pitch_emphasis_post(samples: &mut [f32; SUBL], mem: &mut f32) {
    let alpha = 0.25_f32;
    for n in 0..SUBL {
        let y = samples[n] + alpha * *mem;
        *mem = y;
        samples[n] = y;
    }
}

/// Synthesise an entire frame, sub-block by sub-block.
pub fn synthesise_frame(
    excitation: &[f32],
    a_per_sub: &[[f32; LPC_ORDER + 1]],
    state: &mut SynthState,
    out: &mut [f32],
) {
    let n_sub = a_per_sub.len();
    debug_assert_eq!(excitation.len(), n_sub * SUBL);
    debug_assert_eq!(out.len(), n_sub * SUBL);

    for sb in 0..n_sub {
        let mut exc = [0.0f32; SUBL];
        exc.copy_from_slice(&excitation[sb * SUBL..(sb + 1) * SUBL]);
        let mut y = [0.0f32; SUBL];
        synthesise(&exc, &a_per_sub[sb], &mut state.mem, &mut y);
        pitch_emphasis_post(&mut y, &mut state.post_mem);
        out[sb * SUBL..(sb + 1) * SUBL].copy_from_slice(&y);
    }
    // Cache the last LPC and excitation RMS for PLC use on future frames.
    state.last_a = a_per_sub[n_sub - 1];
    let last_exc = &excitation[(n_sub - 1) * SUBL..];
    let mut sum_sq = 0.0f32;
    for &v in last_exc.iter() {
        sum_sq += v * v;
    }
    state.last_rms = (sum_sq / SUBL as f32).sqrt();
    state.plc_count = 0;
}

/// Generate a concealed frame (RFC 3951 §4.5.2). Produces `mode.samples()`
/// output samples using the last-seen LPC filter, attenuating by
/// ~0.85 per consecutive concealed frame so prolonged loss decays to
/// silence.
pub fn conceal_frame(state: &mut SynthState, mode: FrameMode, out: &mut [f32]) {
    state.plc_count = state.plc_count.saturating_add(1);
    let atten = 0.85_f32.powi(state.plc_count as i32);
    let sigma = state.last_rms * atten;
    let n_sub = mode.sub_blocks();
    for sb in 0..n_sub {
        let mut exc = [0.0f32; SUBL];
        for n in 0..SUBL {
            // xorshift32 pseudorandom
            let mut s = state.plc_seed;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            state.plc_seed = s;
            // Map to [-1, +1].
            let r = ((s as i32) as f32) / (i32::MAX as f32);
            exc[n] = r * sigma;
        }
        let mut y = [0.0f32; SUBL];
        synthesise(&exc, &state.last_a, &mut state.mem, &mut y);
        pitch_emphasis_post(&mut y, &mut state.post_mem);
        out[sb * SUBL..(sb + 1) * SUBL].copy_from_slice(&y);
    }
    // Decay the cached RMS so the next concealed frame attenuates
    // further on top of the exponent.
    state.last_rms *= 0.9;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyphaser_bit_exact() {
        // Verbatim RFC 3951 Appendix A.8 endpoints.
        assert_eq!(POLYPHASER_TBL.len(), 28);
        assert_eq!(POLYPHASER_TBL[0], 0.000000);
        assert_eq!(POLYPHASER_TBL[3], 1.000000);
        assert_eq!(POLYPHASER_TBL[16], 0.601563);
        assert_eq!(POLYPHASER_TBL[17], 0.601563);
        assert_eq!(POLYPHASER_TBL[27], -0.018799);
    }

    #[test]
    fn enh_plocs_bit_exact() {
        assert_eq!(ENH_PLOCS_TBL.len(), 8);
        assert_eq!(ENH_PLOCS_TBL[0], 40.0);
        assert_eq!(ENH_PLOCS_TBL[7], 600.0);
        // Spacing must be exactly 80 samples.
        for k in 1..ENH_PLOCS_TBL.len() {
            assert_eq!(ENH_PLOCS_TBL[k] - ENH_PLOCS_TBL[k - 1], 80.0);
        }
    }

    #[test]
    fn synthesise_zero_excitation_zero_output() {
        let mut a = [0.0f32; LPC_ORDER + 1];
        a[0] = 1.0;
        let exc = [0.0f32; SUBL];
        let mut mem = [0.0f32; LPC_ORDER];
        let mut out = [0.0f32; SUBL];
        synthesise(&exc, &a, &mut mem, &mut out);
        for &v in out.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn synthesise_impulse_bounded() {
        // Stable LPC: a(1..)=0 (trivial).
        let mut a = [0.0f32; LPC_ORDER + 1];
        a[0] = 1.0;
        let mut exc = [0.0f32; SUBL];
        exc[0] = 1.0;
        let mut mem = [0.0f32; LPC_ORDER];
        let mut out = [0.0f32; SUBL];
        synthesise(&exc, &a, &mut mem, &mut out);
        assert_eq!(out[0], 1.0);
        for &v in out.iter().skip(1) {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn conceal_produces_bounded_output() {
        let mut state = SynthState::new();
        state.last_rms = 100.0;
        let mut out = vec![0.0f32; 160];
        conceal_frame(&mut state, FrameMode::Ms20, &mut out);
        for &v in out.iter() {
            assert!(v.is_finite());
            assert!(v.abs() < 1e6);
        }
    }

    #[test]
    fn conceal_attenuates_over_frames() {
        let mut state = SynthState::new();
        state.last_rms = 1000.0;
        let mut energies = Vec::new();
        for _ in 0..10 {
            let mut out = vec![0.0f32; 160];
            conceal_frame(&mut state, FrameMode::Ms20, &mut out);
            let e: f32 = out.iter().map(|v| v * v).sum();
            energies.push(e);
        }
        // Last should be much smaller than first.
        assert!(energies.last().unwrap() < &(energies[0] * 0.5));
    }
}
