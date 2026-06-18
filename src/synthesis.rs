//! LPC synthesis and packet-loss concealment — RFC 3951 §4.5, §4.7, §4.8.
//!
//! The synthesis filter is 10th-order all-pole:
//!
//! ```text
//!     y(n) = x(n) - Σ a[i] * y(n-i),  i = 1..10
//! ```
//!
//! The §4.6 excitation enhancer is in the [`crate::enhancer`] module
//! and is invoked by the frame decoder *before* the LPC synthesis
//! filter runs.
//!
//! PLC (§4.5): if a frame is lost (or the empty-frame indicator is set),
//! the lost block is substituted by a pitch-synchronous repetition of the
//! previous block's saved excitation, mixed with a random excitation in a
//! ratio set by the voicing level measured on the last good block, and
//! filtered through that block's LP filter — energy-dampened for
//! consecutive losses. See [`conceal_frame`] / [`analyse_pitch`].

use crate::{FrameMode, LPC_ORDER, SUBL};

// Re-export the enhancer tables here for backward compatibility with
// existing tests / downstream users that imported them from this module.
pub use crate::enhancer::{ENH_PLOCS_TBL, POLYPHASER_TBL};

/// Length of the previous-excitation history the PLC keeps for the
/// §4.5.2 pitch-synchronous repetition + correlation analysis. One 30 ms
/// frame is 240 samples; the longest pitch lag we search (`PLC_PITCH_MAX`)
/// must fit comfortably inside this window for the correlation analysis.
pub const PLC_HIST_LEN: usize = 240;

/// Shortest pitch period the §4.5.2 correlation analysis searches, in
/// 8 kHz samples (≈ 400 Hz — the top of the speech-pitch range).
pub const PLC_PITCH_MIN: usize = 20;

/// Longest pitch period the §4.5.2 correlation analysis searches, in
/// 8 kHz samples (≈ 66 Hz — the bottom of the speech-pitch range).
pub const PLC_PITCH_MAX: usize = 120;

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
    /// Most-recent decoded excitation (the signal fed to the LPC
    /// synthesis filter), kept for the §4.5.2 pitch-synchronous PLC.
    /// `PLC_HIST_LEN` samples, oldest first. Records the "entire decoded
    /// excitation signal" §4.5.1 says to save against a future loss.
    pub exc_hist: [f32; PLC_HIST_LEN],
    /// Pitch lag (in samples) measured on the last good block, reused for
    /// consecutive concealed blocks per §4.5.2.
    pub plc_pitch: usize,
    /// Voicing level ∈ [0, 1] measured on the last good block: the degree
    /// to which the previous excitation was periodic. Drives the
    /// periodic-vs-random excitation mix in §4.5.2.
    pub plc_voicing: f32,
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
            exc_hist: [0.0; PLC_HIST_LEN],
            plc_pitch: PLC_PITCH_MIN,
            plc_voicing: 0.0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Append a freshly-decoded block of excitation to the rolling
    /// history (oldest first), shifting older samples out. §4.5.1: the
    /// entire decoded excitation is saved so the next block, if lost, can
    /// be substituted pitch-synchronously.
    fn push_excitation(&mut self, exc: &[f32]) {
        let n = exc.len();
        if n >= PLC_HIST_LEN {
            self.exc_hist.copy_from_slice(&exc[n - PLC_HIST_LEN..]);
        } else {
            // Slide existing history left by `n`, then append.
            self.exc_hist.copy_within(n.., 0);
            self.exc_hist[PLC_HIST_LEN - n..].copy_from_slice(exc);
        }
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

/// Kept for backward compatibility (tests / external tools). Applies a
/// first-order low-pass smoothing: `y_post(n) = y(n) + 0.25*y_post(n-1)`.
/// Not used by the main decode pipeline; the RFC-proper §4.6 enhancer
/// handles periodic enhancement on the excitation side, and §4.8
/// mandates only an optional 65 Hz HP post-filter.
pub fn pitch_emphasis_post(samples: &mut [f32; SUBL], mem: &mut f32) {
    let alpha = 0.25_f32;
    for s in samples.iter_mut() {
        let y = *s + alpha * *mem;
        *mem = y;
        *s = y;
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

    // §4.5.1: record the entire decoded excitation, then run the §4.5.2
    // correlation analysis up-front so a *following* lost block can be
    // substituted pitch-synchronously using the same pitch + voicing
    // measured here.
    state.push_excitation(excitation);
    let (pitch, voicing) = analyse_pitch(&state.exc_hist);
    state.plc_pitch = pitch;
    state.plc_voicing = voicing;
}

/// §4.5.2 correlation analysis on the previous block's excitation.
///
/// Returns `(pitch_lag, voicing)` where `pitch_lag ∈ [PLC_PITCH_MIN,
/// PLC_PITCH_MAX]` is the lag maximising the normalised cross-correlation
/// of the most-recent window with itself shifted back by that lag, and
/// `voicing ∈ [0, 1]` is that peak normalised correlation — the "degree
/// to which the previous block's excitation was a voiced or roughly
/// periodic signal" the RFC uses to set the periodic-vs-random mix.
pub fn analyse_pitch(hist: &[f32; PLC_HIST_LEN]) -> (usize, f32) {
    // Correlate the trailing window (one short block) against earlier
    // copies of itself. The window must be short enough that the longest
    // lag still has a full window of past samples in the history.
    let win = PLC_HIST_LEN - PLC_PITCH_MAX;
    let base = PLC_HIST_LEN - win; // start of the trailing window
    let mut best_lag = PLC_PITCH_MIN;
    let mut best_norm = 0.0f32;
    // Energy of the trailing window (denominator term 1).
    let mut e_cur = 0.0f32;
    for i in 0..win {
        let v = hist[base + i];
        e_cur += v * v;
    }
    if e_cur <= f32::MIN_POSITIVE {
        return (PLC_PITCH_MIN, 0.0);
    }
    for lag in PLC_PITCH_MIN..=PLC_PITCH_MAX {
        let mut cross = 0.0f32;
        let mut e_lag = 0.0f32;
        for i in 0..win {
            let cur = hist[base + i];
            let prev = hist[base + i - lag];
            cross += cur * prev;
            e_lag += prev * prev;
        }
        if e_lag <= f32::MIN_POSITIVE {
            continue;
        }
        // Normalised cross-correlation in [-1, 1].
        let norm = cross / (e_cur * e_lag).sqrt();
        if norm > best_norm {
            best_norm = norm;
            best_lag = lag;
        }
    }
    // Voicing is the clamped peak correlation. A negative or tiny peak
    // means little periodicity → mostly-random substitution.
    let voicing = best_norm.clamp(0.0, 1.0);
    (best_lag, voicing)
}

/// Generate a concealed frame (RFC 3951 §4.5.2).
///
/// The lost block is substituted by a **pitch-synchronous repetition** of
/// the previous block's excitation — extended forward by copying from
/// `plc_pitch` samples back — **mixed with a random excitation**, where
/// the relative weight of the two components is the voicing level
/// measured on the last good block (§4.5.2: "a random excitation is mixed
/// with the new pitch periodic excitation, and the relative use of the
/// two components is computed from the correlation measure"). The mixed
/// excitation is filtered through the last LP filter of the previous
/// block. For consecutive lost blocks the same pitch + voicing are
/// reused and the excitation energy is dampened (§4.5.2 final paragraph).
pub fn conceal_frame(state: &mut SynthState, mode: FrameMode, out: &mut [f32]) {
    state.plc_count = state.plc_count.saturating_add(1);
    // Energy dampening for consecutive losses (RFC: "the energy of the
    // substituted excitation for consecutive lost blocks is decreased").
    let atten = 0.85_f32.powi(state.plc_count as i32);

    let n = mode.samples();
    let pitch = state.plc_pitch.clamp(PLC_PITCH_MIN, PLC_PITCH_MAX);
    let voicing = state.plc_voicing.clamp(0.0, 1.0);
    // Periodic / random energy split. sqrt-weighting keeps the *power* of
    // the mix constant across the voicing sweep (the two components are
    // uncorrelated, so their energies add).
    let w_per = voicing.sqrt();
    let w_rand = (1.0 - voicing).sqrt();

    // The periodic component continues the previous excitation: the first
    // `pitch` synthetic samples come from the tail of the saved history,
    // and beyond that we wrap onto the synthetic samples already produced,
    // so the pitch period repeats seamlessly.
    let hist = &state.exc_hist;
    let mut periodic = vec![0.0f32; n];
    for i in 0..n {
        periodic[i] = if i < pitch {
            hist[PLC_HIST_LEN - pitch + i]
        } else {
            periodic[i - pitch]
        };
    }
    // RMS of the periodic continuation — used to scale the random part to
    // the same level so the mix energy tracks the last good block.
    let mut per_ss = 0.0f32;
    for &v in &periodic {
        per_ss += v * v;
    }
    let per_rms = (per_ss / n as f32).sqrt();
    // Random level: fall back to the cached excitation RMS when the
    // periodic continuation is silent (fully-random unvoiced case).
    let rand_level = if per_rms > f32::MIN_POSITIVE {
        per_rms
    } else {
        state.last_rms
    };

    let mut exc_buf = vec![0.0f32; n];
    for (i, slot) in exc_buf.iter_mut().enumerate() {
        // xorshift32 pseudorandom in [-1, +1].
        let mut s = state.plc_seed;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        state.plc_seed = s;
        let r = ((s as i32) as f32) / (i32::MAX as f32);
        let rand = r * rand_level;
        *slot = atten * (w_per * periodic[i] + w_rand * rand);
    }

    // Filter the substituted excitation through the last LP filter,
    // sub-block by sub-block (preserving the synthesis-filter memory).
    let n_sub = mode.sub_blocks();
    for sb in 0..n_sub {
        let mut exc = [0.0f32; SUBL];
        exc.copy_from_slice(&exc_buf[sb * SUBL..(sb + 1) * SUBL]);
        let mut y = [0.0f32; SUBL];
        synthesise(&exc, &state.last_a, &mut state.mem, &mut y);
        out[sb * SUBL..(sb + 1) * SUBL].copy_from_slice(&y);
    }

    // Feed the substituted excitation back into the history so a further
    // consecutive loss continues the pitch-synchronous repetition from
    // the synthetic signal, and decay the cached RMS for the next block.
    state.push_excitation(&exc_buf);
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
        // Seed a periodic history so the periodic component is non-trivial.
        let lag = 40usize;
        for i in 0..PLC_HIST_LEN {
            state.exc_hist[i] = ((i % lag) as f32 - lag as f32 / 2.0) * 10.0;
        }
        state.plc_pitch = lag;
        state.plc_voicing = 0.9;
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

    #[test]
    fn analyse_pitch_finds_seeded_period() {
        // A pure period-`lag` sawtooth must yield voicing≈1 and pitch=lag.
        let lag = 53usize;
        let mut hist = [0.0f32; PLC_HIST_LEN];
        for (i, h) in hist.iter_mut().enumerate() {
            *h = ((i % lag) as f32 - lag as f32 / 2.0) * 7.0;
        }
        let (pitch, voicing) = analyse_pitch(&hist);
        assert_eq!(pitch, lag);
        assert!(voicing > 0.95, "voicing {voicing}");
    }

    #[test]
    fn analyse_pitch_low_voicing_on_silence() {
        let hist = [0.0f32; PLC_HIST_LEN];
        let (_pitch, voicing) = analyse_pitch(&hist);
        assert_eq!(voicing, 0.0);
    }

    #[test]
    fn conceal_periodic_repeats_pitch() {
        // With full voicing the concealed excitation should reproduce the
        // saved periodic excitation (modulo the LP filter + dampening), so
        // the output of a single concealed block on an all-pass filter
        // (last_a = identity) must be periodic with `pitch`.
        let mut state = SynthState::new();
        let lag = 40usize;
        for i in 0..PLC_HIST_LEN {
            state.exc_hist[i] = ((i % lag) as f32 - lag as f32 / 2.0) * 5.0;
        }
        state.plc_pitch = lag;
        state.plc_voicing = 1.0; // fully voiced → no random component
                                 // identity LP filter so output == excitation.
        let mut id = [0.0f32; LPC_ORDER + 1];
        id[0] = 1.0;
        state.last_a = id;
        let mut out = vec![0.0f32; 160];
        conceal_frame(&mut state, FrameMode::Ms20, &mut out);
        // Output beyond the first pitch period repeats with period `lag`.
        for i in lag..160 {
            assert!(
                (out[i] - out[i - lag]).abs() < 1e-3,
                "non-periodic at {i}: {} vs {}",
                out[i],
                out[i - lag]
            );
        }
    }

    #[test]
    fn conceal_unvoiced_is_aperiodic() {
        // With zero voicing the substitution is pure random — it must NOT
        // be pitch-periodic.
        let mut state = SynthState::new();
        let lag = 40usize;
        for i in 0..PLC_HIST_LEN {
            state.exc_hist[i] = ((i % lag) as f32 - lag as f32 / 2.0) * 5.0;
        }
        state.last_rms = 100.0;
        state.plc_pitch = lag;
        state.plc_voicing = 0.0;
        let mut id = [0.0f32; LPC_ORDER + 1];
        id[0] = 1.0;
        state.last_a = id;
        let mut out = vec![0.0f32; 160];
        conceal_frame(&mut state, FrameMode::Ms20, &mut out);
        let mut max_diff = 0.0f32;
        for i in lag..160 {
            max_diff = max_diff.max((out[i] - out[i - lag]).abs());
        }
        assert!(max_diff > 1.0, "unexpectedly periodic: max_diff {max_diff}");
    }
}
