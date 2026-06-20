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

/// Largest concealed-frame length in samples (30 ms = 240). RFC 3951
/// Appendix A.14 calls this `BLOCKL_MAX`; the §4.5.2 `randvec` scratch is
/// sized to it.
pub const BLOCKL_MAX: usize = 240;

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

    // ---- RFC 3951 Appendix A.14 `doThePLC` residual-domain state ----
    //
    // These mirror the `iLBC_Dec_Inst_t` fields the §4.5 / Appendix A.14 example
    // keeps so the residual-domain concealer ([`conceal_residual`]) can
    // reproduce the documented algorithm exactly. The PCM-domain
    // [`conceal_frame`] above keeps its own `exc_hist`-based state for
    // backward compatibility; the two are independent.
    /// The full decoded residual of the previous block (`prevResidual`),
    /// length = `blockl` (160 or 240). Source for the pitch-synchronous
    /// repetition + the random `randvec` copies in §4.5.2.
    pub prev_residual: [f32; BLOCKL_MAX],
    /// Length in samples of the valid prefix of `prev_residual`.
    pub prev_residual_len: usize,
    /// Packet-loss indicator of the previous block (`prevPLI`): 1 if the
    /// previous block was concealed, 0 if it was received. Selects the
    /// §4.5.2 "previous frame lost" vs "previous frame received" branch.
    pub prev_pli: u8,
    /// Pitch lag recorded by the last concealment (`prevLag`), reused when
    /// the previous block was itself concealed.
    pub prev_lag: i32,
    /// Periodicity measure recorded by the last concealment (`per`),
    /// reused across consecutive losses.
    pub per: f32,
    /// `doThePLC`'s LCG seed (`seed`), distinct from the PCM-domain
    /// `plc_seed`. RFC uses `seed = seed*69069 + 1`.
    pub plc_res_seed: u32,
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
            prev_residual: [0.0; BLOCKL_MAX],
            prev_residual_len: 0,
            prev_pli: 0,
            prev_lag: 20,
            per: 0.0,
            plc_res_seed: 0,
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

/// Synthesise an entire frame, sub-block by sub-block, *without* updating
/// any PLC bookkeeping. Used by both the good-frame path (via
/// [`synthesise_frame`]) and the residual-domain concealment path, which
/// manages its own §A.14 state and must not have it clobbered by the
/// §4.5.1 PCM-domain recording.
pub fn synthesise_blocks(
    excitation: &[f32],
    a_per_sub: &[[f32; LPC_ORDER + 1]],
    mem: &mut [f32; LPC_ORDER],
    out: &mut [f32],
) {
    let n_sub = a_per_sub.len();
    debug_assert_eq!(excitation.len(), n_sub * SUBL);
    debug_assert_eq!(out.len(), n_sub * SUBL);
    for sb in 0..n_sub {
        let mut exc = [0.0f32; SUBL];
        exc.copy_from_slice(&excitation[sb * SUBL..(sb + 1) * SUBL]);
        let mut y = [0.0f32; SUBL];
        synthesise(&exc, &a_per_sub[sb], mem, &mut y);
        out[sb * SUBL..(sb + 1) * SUBL].copy_from_slice(&y);
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
    synthesise_blocks(excitation, a_per_sub, &mut state.mem, out);
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

/// Cross-correlation + pitch-gain for the §4.5.2 pitch prediction of the
/// last sub-frame at a given lag. Mirrors RFC 3951 Appendix A.14
/// `compCorr`: it correlates the trailing `s_range` samples of `buffer`
/// against the same window shifted back by `lag`.
///
/// Returns `(cc, gc, pm)` where
/// * `cc = ftmp1² / ftmp2` — the cross-correlation criterion the lag
///   search maximises,
/// * `gc = |ftmp1 / ftmp2|` — the pitch gain (unused here but kept to
///   match the §A.14 `compCorr` signature for clarity),
/// * `pm = |ftmp1| / (sqrt(ftmp2)·sqrt(ftmp3))` — the normalised
///   periodicity in `[0, 1]` that becomes the recorded `per`.
fn comp_corr(buffer: &[f32], lag: usize, b_len: usize, s_range: usize) -> (f32, f32, f32) {
    // Guard against reading before the start of the buffer (A.14:
    // "if ((bLen-sRange-lag)<0) sRange=bLen-lag").
    let s_range = if (b_len as isize) - (s_range as isize) - (lag as isize) < 0 {
        b_len.saturating_sub(lag)
    } else {
        s_range
    };
    let mut ftmp1 = 0.0f32; // cross
    let mut ftmp2 = 0.0f32; // energy of lagged window
    let mut ftmp3 = 0.0f32; // energy of current window
    let base = b_len - s_range;
    for i in 0..s_range {
        let cur = buffer[base + i];
        let prev = buffer[base + i - lag];
        ftmp1 += cur * prev;
        ftmp2 += prev * prev;
        ftmp3 += cur * cur;
    }
    if ftmp2 > 0.0 {
        let cc = ftmp1 * ftmp1 / ftmp2;
        let gc = (ftmp1 / ftmp2).abs();
        let pm = ftmp1.abs() / (ftmp2.sqrt() * ftmp3.sqrt());
        (cc, gc, pm)
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Residual-domain packet-loss concealment — RFC 3951 §4.5.2, following
/// the Appendix A.14 `doThePLC` example.
///
/// Produces a concealed *residual* (excitation) block of `mode.samples()`
/// samples from `state.prev_residual`, to be fed through the same enhancer
/// and LPC synthesis path as a received block (the §4.5.3 smooth merge
/// into a subsequent good block is then handled implicitly by the
/// enhancer's cross-block correlation, exactly as in the RFC 3951 §A.44 decoder
/// driver).
///
/// `inlag` is the pitch lag the enhancer measured on the last good block
/// (RFC 3951 §A.44 passes `last_lag` here). The concealed residual is also
/// written back into `state.prev_residual` so a further consecutive loss
/// continues from the synthetic signal, and `state.prev_pli` /
/// `state.prev_lag` / `state.per` / `state.plc_count` are updated per the
/// §A.14 state machine.
pub fn conceal_residual(state: &mut SynthState, mode: FrameMode, inlag: i32, out: &mut [f32]) {
    let blockl = mode.samples();
    debug_assert_eq!(out.len(), blockl);
    // First loss after a clean run has no recorded residual: emit silence.
    if state.prev_residual_len != blockl {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        state.prev_pli = 1;
        state.plc_count = state.plc_count.saturating_add(1);
        return;
    }

    state.plc_count = state.plc_count.saturating_add(1);
    let prev = &state.prev_residual[..blockl];

    // Pitch lag + periodicity. If the previous block was received, refine
    // the lag in a ±3 window around the enhancer's `inlag`; if it was
    // itself concealed, reuse the recorded lag + periodicity (A.14).
    let (mut lag, mut max_per) = if state.prev_pli != 1 {
        let inlag = inlag.max(PLC_PITCH_MIN as i32);
        let lo = (inlag - 3).max(1) as usize;
        let mut best_lag = lo;
        let (mut maxcc, _gc, mut best_per) = comp_corr(prev, lo, blockl, 60);
        for i in (inlag - 2)..=(inlag + 3) {
            if i < 1 || i as usize >= blockl {
                continue;
            }
            let (cc, _g, pm) = comp_corr(prev, i as usize, blockl, 60);
            if cc > maxcc {
                maxcc = cc;
                best_lag = i as usize;
                best_per = pm;
            }
        }
        (best_lag as i32, best_per)
    } else {
        (state.prev_lag, state.per)
    };
    if lag < 1 {
        lag = PLC_PITCH_MIN as i32;
    }

    // Consecutive-loss energy downscaling (A.14 `use_gain`). The reference
    // ladder is evaluated against `consPLICount * blockl` in multiples of
    // 320 samples (40 ms).
    let span = state.plc_count as usize * blockl;
    let mut use_gain = 1.0f32;
    if span > 4 * 320 {
        use_gain = 0.0;
    } else if span > 3 * 320 {
        use_gain = 0.5;
    } else if span > 2 * 320 {
        use_gain = 0.7;
    } else if span > 320 {
        use_gain = 0.9;
    }

    // Periodic-vs-random mix factor from the periodicity measure (A.14
    // `pitchfact`): full pitch above sqrt(per) 0.7, linear ramp down to
    // 0.4, pure noise below.
    let ftmp = max_per.max(0.0).sqrt();
    let pitchfact = if ftmp > 0.7 {
        1.0
    } else if ftmp > 0.4 {
        (ftmp - 0.4) / (0.7 - 0.4)
    } else {
        0.0
    };

    // Avoid repeating the same pitch cycle for short lags (A.14).
    let use_lag = if lag < 80 { 2 * lag } else { lag } as usize;

    // Build the concealed residual sample by sample.
    let mut randvec = [0.0f32; BLOCKL_MAX];
    let mut energy = 0.0f32;
    for i in 0..blockl {
        // Noise component: a randomly-delayed copy of the previous
        // residual (A.14 randlag = 50 + seed%70 over the 0x7fff_ffff mask).
        state.plc_res_seed = state.plc_res_seed.wrapping_mul(69069).wrapping_add(1) & 0x7fff_ffff;
        let randlag = 50 + (state.plc_res_seed as i32) % 70;
        let pick = i as i32 - randlag;
        randvec[i] = if pick < 0 {
            prev[(blockl as i32 + pick) as usize]
        } else {
            randvec[pick as usize]
        };

        // Pitch-repetition component.
        let pick = i as i32 - use_lag as i32;
        let pitch_sample = if pick < 0 {
            prev[(blockl as i32 + pick) as usize]
        } else {
            out[pick as usize]
        };

        // Mix, with the per-80-sample intra-block taper (A.14): 1.0 for
        // the first 80, 0.95 for the next 80, 0.9 beyond.
        let taper = if i < 80 {
            1.0
        } else if i < 160 {
            0.95
        } else {
            0.9
        };
        out[i] = taper * use_gain * (pitchfact * pitch_sample + (1.0 - pitchfact) * randvec[i]);
        energy += out[i] * out[i];
    }

    // Below 30 dB RMS, fall back to pure noise (A.14).
    if (energy / blockl as f32).sqrt() < 30.0 {
        max_per = 0.0;
        out[..blockl].copy_from_slice(&randvec[..blockl]);
    }

    // Update §A.14 state machine.
    state.prev_lag = lag;
    state.per = max_per;
    state.prev_pli = 1;
    state.prev_residual[..blockl].copy_from_slice(&out[..blockl]);
    state.prev_residual_len = blockl;
}

/// §4.5.1 / A.14 (no-loss branch): record the decoded residual + final LP
/// filter so a *following* lost block can be concealed, and clear the
/// consecutive-loss counter. Called once per received frame.
pub fn plc_record_good(state: &mut SynthState, decresidual: &[f32], last_a: &[f32; LPC_ORDER + 1]) {
    let n = decresidual.len().min(BLOCKL_MAX);
    state.prev_residual[..n].copy_from_slice(&decresidual[..n]);
    state.prev_residual_len = n;
    state.last_a = *last_a;
    state.prev_pli = 0;
    state.plc_count = 0;
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

    // ---- Appendix A.14 `doThePLC` residual-domain concealment ----

    fn seed_periodic_residual(state: &mut SynthState, blockl: usize, lag: usize, amp: f32) {
        for i in 0..blockl {
            state.prev_residual[i] = ((i % lag) as f32 - lag as f32 / 2.0) * amp;
        }
        state.prev_residual_len = blockl;
        state.prev_pli = 0;
    }

    #[test]
    fn conceal_residual_first_loss_without_history_is_silent() {
        // No recorded residual yet (prev_residual_len == 0) → silence, and
        // prev_pli flips to 1 so the next loss takes the recorded-state path.
        let mut state = SynthState::new();
        let mut out = vec![0.0f32; 160];
        conceal_residual(&mut state, FrameMode::Ms20, 40, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        assert_eq!(state.prev_pli, 1);
        assert_eq!(state.plc_count, 1);
    }

    #[test]
    fn conceal_residual_doubles_short_lag() {
        // A.14: for lag<80 the repetition uses 2*lag to avoid repeating the
        // same pitch cycle. A pure period-`lag` residual concealed with full
        // voicing must therefore reproduce with period `lag` (since 2*lag is
        // also a period) and stay bounded.
        let mut state = SynthState::new();
        let blockl = 240usize; // 30 ms so we exercise the 80/160 tapers
        let lag = 40usize;
        seed_periodic_residual(&mut state, blockl, lag, 200.0);
        let mut out = vec![0.0f32; blockl];
        // inlag at the true lag so the ±3 search locks on.
        conceal_residual(&mut state, FrameMode::Ms30, lag as i32, &mut out);
        // Strongly periodic input → high periodicity → pitchfact≈1, so the
        // pitch-repetition dominates and the first 80 samples (taper=1,
        // use_gain=1) repeat with the residual's period.
        for i in (2 * lag)..80 {
            assert!(
                (out[i] - out[i - lag]).abs() < out[i].abs().max(1.0) * 0.5,
                "non-periodic at {i}: {} vs {}",
                out[i],
                out[i - lag]
            );
        }
    }

    #[test]
    fn conceal_residual_taper_drops_energy_across_block() {
        // The §4.5.2 intra-block taper is 1.0 / 0.95 / 0.9 across the three
        // 80-sample thirds of a 30 ms block, so (for a stationary-energy
        // periodic residual) the last third must carry less energy than the
        // first.
        let mut state = SynthState::new();
        let blockl = 240usize;
        seed_periodic_residual(&mut state, blockl, 50, 300.0);
        let mut out = vec![0.0f32; blockl];
        conceal_residual(&mut state, FrameMode::Ms30, 50, &mut out);
        let e0: f32 = out[0..80].iter().map(|v| v * v).sum();
        let e2: f32 = out[160..240].iter().map(|v| v * v).sum();
        assert!(e2 < e0, "third-third energy {e2} not below first {e0}");
    }

    #[test]
    fn conceal_residual_consecutive_losses_attenuate() {
        // A.14 `use_gain` ladder: once consPLICount*blockl exceeds 320 the
        // gain steps down. With 30 ms blocks the 2nd loss already crosses
        // 320 (480), so its energy must drop below the 1st loss.
        let mut state = SynthState::new();
        let blockl = 240usize;
        seed_periodic_residual(&mut state, blockl, 50, 400.0);
        let mut out = vec![0.0f32; blockl];
        conceal_residual(&mut state, FrameMode::Ms30, 50, &mut out);
        let e_first: f32 = out.iter().map(|v| v * v).sum();
        conceal_residual(&mut state, FrameMode::Ms30, 50, &mut out);
        let e_second: f32 = out.iter().map(|v| v * v).sum();
        assert!(
            e_second < e_first,
            "consecutive loss not attenuated: {e_second} vs {e_first}"
        );
    }

    #[test]
    fn conceal_residual_low_energy_falls_back_to_noise() {
        // A.14: if the mixed residual RMS is below 30, the block becomes the
        // pure `randvec` and `per` is zeroed. Seed a tiny residual so the
        // mix lands under the threshold.
        let mut state = SynthState::new();
        let blockl = 160usize;
        for i in 0..blockl {
            state.prev_residual[i] = ((i % 40) as f32 - 20.0) * 0.1; // peak ≈ 2
        }
        state.prev_residual_len = blockl;
        state.prev_pli = 0;
        let mut out = vec![0.0f32; blockl];
        conceal_residual(&mut state, FrameMode::Ms20, 40, &mut out);
        // per is zeroed in the noise-fallback path.
        assert_eq!(state.per, 0.0);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn conceal_residual_records_state_for_next_loss() {
        // After a loss, prev_pli==1 and prev_lag is recorded so a following
        // loss reuses it (skips the ±3 search).
        let mut state = SynthState::new();
        seed_periodic_residual(&mut state, 240, 53, 250.0);
        let mut out = vec![0.0f32; 240];
        conceal_residual(&mut state, FrameMode::Ms30, 53, &mut out);
        assert_eq!(state.prev_pli, 1);
        assert!(state.prev_lag >= 50 && state.prev_lag <= 56);
    }

    #[test]
    fn plc_record_good_clears_loss_state() {
        let mut state = SynthState::new();
        state.plc_count = 5;
        state.prev_pli = 1;
        let res = vec![10.0f32; 160];
        let mut a = [0.0f32; LPC_ORDER + 1];
        a[0] = 1.0;
        plc_record_good(&mut state, &res, &a);
        assert_eq!(state.plc_count, 0);
        assert_eq!(state.prev_pli, 0);
        assert_eq!(state.prev_residual_len, 160);
        assert_eq!(state.prev_residual[0], 10.0);
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
