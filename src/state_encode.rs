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
//! The §3.5.3 perceptual DPCM noise-shaping loop is implemented in
//! [`abs_quant_w`] (Appendix A.46 `AbsQuantW`) but the encoder uses it
//! only when `state_dpcm=on`; the default is the direct scalar quantiser
//! above. The RFC calls the weighting RECOMMENDED, not REQUIRED, and the
//! decoder's reconstruction ignores it either way (see `crate::state` /
//! Appendix A.44 `StateConstructW`), so both quantisers emit `state_sq3Tbl`
//! indices that decode identically.

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

/// Perceptual-weighting chirp factor `Wk(z) = 1/Ak(z/0.4222)`.
///
/// RFC 3951 §3.4 (line 820): `LPC_CHIRP_WEIGHTDENUM = 0.4222`. The same
/// constant drives the codebook-search weighting filter in
/// [`crate::cb_search`].
const LPC_CHIRP_WEIGHTDENUM: f32 = 0.4222;

/// Build the bandwidth-expanded weighting denominator `weightDenum =
/// Ak(z / 0.4222)` from a sub-block LPC polynomial `[1, a1..a10]`.
pub fn weight_denum_pub(a: &[f32; LPC_ORDER + 1]) -> [f32; LPC_ORDER + 1] {
    weight_denum(a)
}

/// Build the bandwidth-expanded weighting denominator `weightDenum =
/// Ak(z / 0.4222)` from a sub-block LPC polynomial `[1, a1..a10]`.
fn weight_denum(a: &[f32; LPC_ORDER + 1]) -> [f32; LPC_ORDER + 1] {
    let mut wd = *a;
    let mut c = 1.0f32;
    for w in wd.iter_mut() {
        *w *= c;
        c *= LPC_CHIRP_WEIGHTDENUM;
    }
    wd
}

/// Predictive noise-shaping DPCM quantiser of the scaled start state —
/// RFC 3951 §3.5.3 / Appendix A.46 `AbsQuantW`.
///
/// `scaled` is the scaled all-pass output `fout * scal` (length
/// `state_short_len`). `wd_first` / `wd_second` are the bandwidth-
/// expanded weighting denominators (`Ak(z/0.4222)`) of the two
/// sub-blocks straddling the start state — `wd_first` is sub-block
/// `start-1`, `wd_second` is sub-block `start`. `state_first` is the
/// position bit (1 ⇒ scalar slot is leading, 0 ⇒ trailing).
///
/// The reference filters the input through the weighting filter `Wk(z)`
/// to form the weighted-speech samples `x[n]`, then runs the sample-by-
/// sample DPCM loop of Figure 3.3: predict `y[n]` via `Pk(z) =
/// 1 - 1/Wk(z)`, quantise `d[n] = x[n] - y[n]` with the 3-bit
/// `state_sq3Tbl`, and feed the chosen value back through `Wk(z)` so the
/// prediction memory tracks the quantised reconstruction.
///
/// The decoder ([`crate::state::reconstruct_scalar_state`]) reads the
/// chosen indices back through `state_sq3Tbl` directly — no inverse
/// weighting is applied at decode time, so this routine is a drop-in,
/// decoder-compatible replacement for the per-sample nearest-neighbour
/// quantiser: it only changes which indices we emit, not how they are
/// interpreted.
///
/// `state_first` selects where the weighting filter switches from the
/// first sub-block's coefficients to the second (RFC reference advances
/// `weightDenum` at `n == SUBL` for `state_first`, else at `n ==
/// state_short_len - SUBL`).
pub fn abs_quant_w(
    scaled: &[f32],
    wd_first: &[f32; LPC_ORDER + 1],
    wd_second: &[f32; LPC_ORDER + 1],
    state_first: bool,
) -> Vec<u8> {
    let len = scaled.len();
    // Weighted-speech buffer `x[n]` — the input filtered through Wk(z) =
    // 1/weightDenum, with the coefficient switch at the sub-block edge.
    // The reference filters `in` in place with a single AllPoleFilter
    // state that runs continuously across the boundary (the second
    // `AllPoleFilter(&in[split], ...)` call reads `in[split-1..]`, the
    // already-filtered tail, as its initial filter memory), so we carry
    // the filter memory across the denominator switch rather than
    // resetting it.
    let split = if state_first {
        SUBL.min(len)
    } else {
        len.saturating_sub(SUBL)
    };
    let mut x = scaled.to_vec();
    let mut wmem = [0.0f32; LPC_ORDER];
    for (n, xn) in x.iter_mut().enumerate() {
        let wd = if n < split { wd_first } else { wd_second };
        let mut s = *xn;
        for k in 1..=LPC_ORDER {
            s -= wd[k] * wmem[k - 1];
        }
        *xn = s;
        for k in (1..LPC_ORDER).rev() {
            wmem[k] = wmem[k - 1];
        }
        wmem[0] = s;
    }

    // DPCM loop in the weighted-speech domain (Figure 3.3). `synt_out`
    // holds the Wk(z)-synthesised quantised values; `synt_mem[k]` is the
    // value `k+1` samples back (most recent at index 0).
    let mut out = vec![0u8; len];
    let mut synt_mem = [0.0f32; LPC_ORDER];
    for n in 0..len {
        // Pick the active weighting denominator for this sample — the
        // reference advances `weightDenum` at `n == SUBL` (state_first)
        // or `n == state_short_len - SUBL` (else), i.e. exactly `split`.
        let wd = if n < split { wd_first } else { wd_second };
        // Prediction y[n] = -Σ wd[k]·synt_out[n-k] (AllPole on a zero
        // input sample): `synt_out[n] = 0` then AllPoleFilter(1).
        let mut y = 0.0f32;
        for k in 1..=LPC_ORDER {
            y -= wd[k] * synt_mem[k - 1];
        }
        // Target d[n] = x[n] - y[n], quantise with state_sq3Tbl.
        let to_q = x[n] - y;
        let idx = quantise_shape_sample(to_q);
        out[n] = idx;
        // Reconstruct synt_out[n] = u[n] - Σ wd[k]·synt_out[n-k] (second
        // AllPoleFilter pass — feeds Wk(z) with the chosen value).
        let mut synt = STATE_SQ3_TBL[idx as usize];
        for k in 1..=LPC_ORDER {
            synt -= wd[k] * synt_mem[k - 1];
        }
        // Shift filter memory: newest at index 0.
        for k in (1..LPC_ORDER).rev() {
            synt_mem[k] = synt_mem[k - 1];
        }
        synt_mem[0] = synt;
    }
    out
}

/// In-place all-pole filter `1/Coef` (RFC 3951 Appendix A.30
/// `AllPoleFilter`) with zero initial state. `coef[0]` is assumed 1.0.
/// Used by the §3.5.3 tests to synthesise weighted-domain signals; the
/// `abs_quant_w` filtering is inlined so its state can carry across the
/// sub-block weighting-denominator switch (see the reference's single
/// continuous `AllPoleFilter` run over `in`).
#[cfg(test)]
fn all_pole_in_place(buf: &mut [f32], coef: &[f32; LPC_ORDER + 1]) {
    let mut mem = [0.0f32; LPC_ORDER];
    for x in buf.iter_mut() {
        let mut s = *x;
        for k in 1..=LPC_ORDER {
            s -= coef[k] * mem[k - 1];
        }
        *x = s;
        for k in (1..LPC_ORDER).rev() {
            mem[k] = mem[k - 1];
        }
        mem[0] = s;
    }
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

    #[test]
    fn weight_denum_chirps_by_0_4222() {
        // Ak(z/0.4222): a[i] *= 0.4222^i. a[0] is untouched (×1).
        let mut a = [0.0f32; LPC_ORDER + 1];
        a[0] = 1.0;
        a[1] = 0.5;
        a[2] = -0.25;
        let wd = weight_denum_pub(&a);
        assert_eq!(wd[0], 1.0);
        assert!((wd[1] - 0.5 * 0.4222).abs() < 1e-7);
        assert!((wd[2] - (-0.25) * 0.4222 * 0.4222).abs() < 1e-7);
    }

    #[test]
    fn abs_quant_w_identity_weight_matches_direct() {
        // With an identity weighting filter (Wk(z) = 1, all chirped
        // coeffs zero), Pk(z) = 1 - 1/Wk(z) = 0: no prediction, no
        // pre-filtering. AbsQuantW then degenerates to the per-sample
        // nearest-neighbour quantiser, so its indices must match the
        // direct `quantise_shape_sample` path bit-for-bit. RFC 3951
        // Figure 3.3 with a flat weighting filter.
        let mut id = [0.0f32; LPC_ORDER + 1];
        id[0] = 1.0;
        let scaled: Vec<f32> = (0..57)
            .map(|i| ((i as f32) * 0.21).sin() * 3.0 - 0.4)
            .collect();
        for &state_first in &[true, false] {
            let dpcm = abs_quant_w(&scaled, &id, &id, state_first);
            let direct: Vec<u8> = scaled.iter().map(|&v| quantise_shape_sample(v)).collect();
            assert_eq!(dpcm, direct, "state_first={state_first}");
        }
    }

    #[test]
    fn abs_quant_w_produces_valid_indices() {
        // Non-trivial weighting filter: every emitted index must be a
        // valid 3-bit state_sq3Tbl index (0..8). Length = 30 ms state.
        let mut a = [0.0f32; LPC_ORDER + 1];
        a[0] = 1.0;
        a[1] = -0.6;
        a[2] = 0.18;
        a[3] = -0.04;
        let wd = weight_denum_pub(&a);
        let scaled: Vec<f32> = (0..58).map(|i| ((i as f32) * 0.37).cos() * 2.5).collect();
        for &state_first in &[true, false] {
            let out = abs_quant_w(&scaled, &wd, &wd, state_first);
            assert_eq!(out.len(), 58);
            for &idx in &out {
                assert!(idx < 8, "index out of 3-bit range: {idx}");
            }
        }
    }

    #[test]
    fn abs_quant_w_noise_shaping_lowers_weighted_error() {
        // The DPCM loop shapes quantisation noise so the weighted-domain
        // reconstruction tracks the weighted-domain target better than
        // an open-loop per-sample quantiser would. We verify the loop
        // actually closes: feed a ramp through a real weighting filter
        // and check the reconstructed weighted signal has lower total
        // squared error against the weighted target than the open-loop
        // (no-feedback) quantiser. RFC 3951 §3.5.3.
        let mut a = [0.0f32; LPC_ORDER + 1];
        a[0] = 1.0;
        a[1] = -0.9;
        a[2] = 0.4;
        let wd = weight_denum_pub(&a);
        let scaled: Vec<f32> = (0..57).map(|i| (i as f32 - 28.0) * 0.12).collect();

        // Weighted target x[n] (the same pre-filtering AbsQuantW does).
        let mut x = scaled.clone();
        all_pole_in_place(&mut x, &wd);

        // Closed-loop indices.
        let idx_cl = abs_quant_w(&scaled, &wd, &wd, true);
        // Reconstruct the closed-loop weighted signal: synthesise the
        // chosen sq3 values through Wk(z) = 1/wd.
        let mut recon_cl: Vec<f32> = idx_cl.iter().map(|&i| STATE_SQ3_TBL[i as usize]).collect();
        all_pole_in_place(&mut recon_cl, &wd);

        // Open-loop indices: per-sample NN on the weighted target.
        let idx_ol: Vec<u8> = x.iter().map(|&v| quantise_shape_sample(v)).collect();
        let mut recon_ol: Vec<f32> = idx_ol.iter().map(|&i| STATE_SQ3_TBL[i as usize]).collect();
        all_pole_in_place(&mut recon_ol, &wd);

        let err_cl: f32 = x
            .iter()
            .zip(&recon_cl)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        let err_ol: f32 = x
            .iter()
            .zip(&recon_ol)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        assert!(
            err_cl <= err_ol,
            "closed-loop weighted error {err_cl} should not exceed open-loop {err_ol}"
        );
    }
}
