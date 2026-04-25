//! Start-state encoder — RFC 3951 §3.5.
//!
//! Mirrors the reference `StateSearchW` / `StateConstructW` pipeline:
//!
//! 1. Compute the residual = LPC analysis filter applied to the input
//!    sub-block.
//! 2. Pick the two consecutive sub-blocks with the highest (weighted)
//!    energy as the start-state location (`start` index, 2 bits for
//!    20 ms / 3 bits for 30 ms).
//! 3. Decide whether the 57/58-sample start-state slice is the first or
//!    the last `STATE_SHORT_LEN` samples of the two-sub-block span
//!    (`position` bit).
//! 4. Pass the 57/58-sample residual through the all-pass filter
//!    `Pk(z) = A~rk(z) / A~k(z)` with the *quantised* LPC coefficients,
//!    folded by `ccres(k) = filtered(k) + filtered(k+STATE_SHORT_LEN)`.
//! 5. Find the largest magnitude sample, log10 it, quantise to 6 bits
//!    against `state_frgqTbl` → `scale_idx`.
//! 6. Multiply all samples by `scal = 4.5 / 10^qmax` and quantise each
//!    one to the nearest entry of `state_sq3Tbl` (3 bits). These are
//!    the `state_samples` indices.
//!
//! The perceptual DPCM loop of §3.5.3 is intentionally simplified to a
//! direct scalar quantiser — the RFC calls the weighting OPTIONAL, and
//! the decoder's reconstruction ignores it (see `crate::state`).

use crate::state::{STATE_FRGQ_TBL, STATE_SQ3_TBL};
use crate::{FrameMode, LPC_ORDER, SUBL};

/// Apply the LPC analysis filter A(z) to a block of samples, given the
/// prior filter memory. Produces the residual `e(n) = x(n) + Σ a[k] x(n-k)`.
///
/// `mem` holds the previous `LPC_ORDER` input samples in time-reversed
/// order (`mem[0]` is the most recent). It is updated in place.
pub fn lpc_analysis_filter(
    input: &[f32],
    a: &[f32; LPC_ORDER + 1],
    mem: &mut [f32; LPC_ORDER],
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), out.len());
    // RFC 3951 §3.2.3 LPC analysis filter A(z): e(n) = Σ a[k]·x(n-k).
    for (n, &x_n) in input.iter().enumerate() {
        let mut s = a[0] * x_n;
        for k in 1..=LPC_ORDER {
            s += a[k] * mem[k - 1];
        }
        out[n] = s;
        // Shift memory: newest sample at index 0.
        for k in (1..LPC_ORDER).rev() {
            mem[k] = mem[k - 1];
        }
        mem[0] = x_n;
    }
}

/// Down-weighting triangular window used at sub-block edges for start-
/// state selection (RFC 3951 §3.5.1 `sampEn_win`).
const SAMP_EN_WIN: [f32; 5] = [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0];

/// Mid-biasing bell for start-state selection (`ssqEn_win`).
const SSQ_EN_WIN_20MS: [f32; 3] = [0.9, 1.0, 0.9];
const SSQ_EN_WIN_30MS: [f32; 5] = [0.8, 0.9, 1.0, 0.9, 0.8];

/// Select the start-state position (0-based index of the first sub-block
/// in the two-sub-block span). Returns a value in `0..(n_sub - 1)`.
pub fn select_start_state(mode: FrameMode, residual: &[f32]) -> usize {
    let n_sub = mode.sub_blocks();
    debug_assert_eq!(residual.len(), n_sub * SUBL);
    let en_win: &[f32] = match mode {
        FrameMode::Ms20 => &SSQ_EN_WIN_20MS,
        FrameMode::Ms30 => &SSQ_EN_WIN_30MS,
    };
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    let pairs = n_sub - 1;
    for (nsub_1, &bias) in en_win.iter().enumerate().take(pairs) {
        // ssqn over [nsub_1*SUBL .. (nsub_1+2)*SUBL), with triangular
        // weighting on the first and last 5 samples (RFC 3951 §3.5.1).
        let mut ssqn = 0.0f32;
        let start = nsub_1 * SUBL;
        let end = start + 2 * SUBL;
        for (offset, &w) in SAMP_EN_WIN.iter().enumerate() {
            let r = residual[start + offset];
            ssqn += w * r * r;
        }
        for &r in &residual[(start + 5)..(end - 5)] {
            ssqn += r * r;
        }
        for (offset, &w) in SAMP_EN_WIN.iter().rev().enumerate() {
            let r = residual[end - 5 + offset];
            ssqn += w * r * r;
        }
        let score = bias * ssqn;
        if score > best_val {
            best_val = score;
            best_idx = nsub_1;
        }
    }
    best_idx
}

/// Decide the `position` bit: whether to keep the first or last
/// STATE_SHORT_LEN samples of the two-sub-block span.
///
/// Returns `(position, slice_lo, slice_hi)`:
///   - `position = 1`: keep the first `STATE_SHORT_LEN` samples.
///   - `position = 0`: keep the last `STATE_SHORT_LEN` samples.
///
/// Whichever boundary segment has the **lower** energy is the one we
/// discard.
pub fn select_position(mode: FrameMode, residual: &[f32], start_idx: usize) -> (u8, usize, usize) {
    let n_short = mode.state_short_len();
    let span_start = start_idx * SUBL;
    let span_end = span_start + 2 * SUBL;
    let boundary = 2 * SUBL - n_short; // 23 or 22
    let mut e_last = 0.0f32;
    for &r in &residual[(span_end - boundary)..span_end] {
        e_last += r * r;
    }
    let mut e_first = 0.0f32;
    for &r in &residual[span_start..(span_start + boundary)] {
        e_first += r * r;
    }
    if e_last <= e_first {
        // Drop the trailing boundary.
        (1, span_start, span_start + n_short)
    } else {
        // Drop the leading boundary.
        (0, span_end - n_short, span_end)
    }
}

/// All-pass filter `Pk(z) = A~rk(z) / A~k(z)` applied as in the reference
/// encoder: input is the 57/58-sample state residual, followed by
/// STATE_SHORT_LEN zeros; we filter with the all-zero numerator then an
/// all-pole denominator, and fold the two halves.
pub fn allpass_forward(state_residual: &[f32], a: &[f32; LPC_ORDER + 1]) -> Vec<f32> {
    let n = state_residual.len();
    let mut padded = vec![0.0f32; 2 * n];
    padded[..n].copy_from_slice(state_residual);
    let fout = crate::state::allpass_zero_pole(&padded, a);
    let mut ccres = vec![0.0f32; n];
    for (k, c) in ccres.iter_mut().enumerate() {
        *c = fout[k] + fout[k + n];
    }
    ccres
}

/// Quantise `max_val = log10(max |ccres|)` against STATE_FRGQ_TBL.
pub fn quantise_scale(ccres: &[f32]) -> u8 {
    let mut max_mag = 1e-9f32;
    for &v in ccres {
        let a = v.abs();
        if a > max_mag {
            max_mag = a;
        }
    }
    let log10_mag = max_mag.log10();
    let mut best_idx = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &v) in STATE_FRGQ_TBL.iter().enumerate() {
        let e = (v - log10_mag).abs();
        if e < best_err {
            best_err = e;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Nearest-neighbour scalar quantisation of `x` against STATE_SQ3_TBL.
pub fn quantise_shape_sample(x: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &v) in STATE_SQ3_TBL.iter().enumerate() {
        let e = (v - x).abs();
        if e < best_err {
            best_err = e;
            best = i as u8;
        }
    }
    best
}

/// Output of the start-state encoder.
#[derive(Clone, Debug)]
pub struct StateEncodeResult {
    /// Start-state position (0-based index of the first sub-block in the
    /// two-sub-block span). Encoded as `block_class = start_idx + 1`.
    pub start_idx: usize,
    /// Position bit (0 or 1).
    pub position: u8,
    /// 6-bit scale index into STATE_FRGQ_TBL.
    pub scale_idx: u8,
    /// 3-bit shape indices, length `STATE_SHORT_LEN`.
    pub state_samples: Vec<u8>,
    /// Quantised (reconstructed) start-state samples in the residual
    /// domain, i.e. what the decoder's `reconstruct_scalar_state` yields.
    /// Length `STATE_SHORT_LEN`.
    pub reconstructed: Vec<f32>,
    /// Range in the frame residual corresponding to the state window
    /// (`STATE_SHORT_LEN` samples).
    pub state_range: (usize, usize),
    /// Range in the frame residual for the full two-sub-block span
    /// `[start_idx*SUBL, (start_idx+2)*SUBL)`.
    pub span_range: (usize, usize),
}

/// Full state encoding pipeline. `residual` is the LPC-filtered whole-
/// frame residual (length `mode.samples()`); `a_for_phase` is the
/// quantised LPC polynomial of the first sub-block in the start-state
/// span, matching the decoder's reconstruction path.
pub fn encode_state(
    mode: FrameMode,
    residual: &[f32],
    a_for_phase: &[f32; LPC_ORDER + 1],
) -> StateEncodeResult {
    let start_idx = select_start_state(mode, residual);
    let (position, slice_start, slice_end) = select_position(mode, residual, start_idx);
    let state_residual = &residual[slice_start..slice_end];
    let ccres = allpass_forward(state_residual, a_for_phase);
    let scale_idx = quantise_scale(&ccres);
    let qmax = STATE_FRGQ_TBL[scale_idx as usize];
    let scal = 4.5 / 10f32.powf(qmax);
    let state_samples: Vec<u8> = ccres
        .iter()
        .map(|&v| quantise_shape_sample(v * scal))
        .collect();
    let reconstructed =
        crate::state::reconstruct_scalar_state(mode, scale_idx, &state_samples, a_for_phase);
    let span_range = (start_idx * SUBL, (start_idx + 2) * SUBL);
    StateEncodeResult {
        start_idx,
        position,
        scale_idx,
        state_samples,
        reconstructed,
        state_range: (slice_start, slice_end),
        span_range,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FRAME_SAMPLES_20MS;

    #[test]
    fn select_start_picks_high_energy() {
        let mut r = vec![0.0f32; FRAME_SAMPLES_20MS];
        r[40..120].fill(1000.0);
        let idx = select_start_state(FrameMode::Ms20, &r);
        assert_eq!(idx, 1);
    }

    #[test]
    fn select_position_drops_low_energy_boundary() {
        let mut r = vec![0.0f32; FRAME_SAMPLES_20MS];
        r[0..57].fill(1000.0);
        let (pos, lo, hi) = select_position(FrameMode::Ms20, &r, 0);
        assert_eq!(pos, 1);
        assert_eq!(lo, 0);
        assert_eq!(hi, 57);
    }

    #[test]
    fn quantise_scale_returns_valid_index() {
        let ccres = vec![100.0f32; 57];
        let idx = quantise_scale(&ccres);
        assert!((idx as usize) < STATE_FRGQ_TBL.len());
    }

    #[test]
    fn quantise_shape_finds_best() {
        assert_eq!(quantise_shape_sample(-3.719849), 0);
        assert_eq!(quantise_shape_sample(3.983887), 7);
        assert_eq!(quantise_shape_sample(0.0), 3);
    }

    #[test]
    fn encode_state_20ms_runs() {
        let r: Vec<f32> = (0..FRAME_SAMPLES_20MS)
            .map(|i| ((i as f32) * 0.3).sin() * 500.0)
            .collect();
        let mut a = [0.0f32; LPC_ORDER + 1];
        a[0] = 1.0;
        let res = encode_state(FrameMode::Ms20, &r, &a);
        assert_eq!(res.state_samples.len(), 57);
        assert!(res.start_idx <= 2);
        assert!(res.scale_idx < 64);
        for &s in &res.state_samples {
            assert!(s < 8);
        }
    }

    #[test]
    fn lpc_analysis_filter_identity() {
        // a = [1, 0, ..., 0] → output == input.
        let mut a = [0.0f32; LPC_ORDER + 1];
        a[0] = 1.0;
        let input: Vec<f32> = (0..40).map(|i| (i as f32) * 0.5).collect();
        let mut mem = [0.0f32; LPC_ORDER];
        let mut out = vec![0.0f32; 40];
        lpc_analysis_filter(&input, &a, &mut mem, &mut out);
        for (got, expected) in out.iter().zip(input.iter()) {
            assert_eq!(got, expected);
        }
    }
}
