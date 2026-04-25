//! LSF dequantisation, stability check, interpolation, and conversion
//! to LPC (RFC 3951 §3.2.5, §3.2.6, §3.2.7, §4.1).
//!
//! For the 20 ms mode one LSF vector is received; for the 30 ms mode
//! two. Sub-block LPC coefficients are produced by linearly
//! interpolating the LSF vectors across the sub-blocks and converting
//! the resulting LSFs to LPC per-sub-block.
//!
//! The decoder also maintains the "last LSF of the previous frame"
//! state for the first sub-block of the next frame; on the very first
//! frame this is initialised to `LSF_MEAN`.

use crate::lsf_tables::{assemble_lsf, LSF_MEAN, LSF_MIN_GAP};
use crate::{FrameMode, LPC_ORDER};

/// LSF decoder state — holds the last frame's (quantised) LSF vector
/// so the next frame's first sub-block can interpolate against it.
#[derive(Clone)]
pub struct LsfState {
    pub last_lsf: [f32; LPC_ORDER],
    pub first_frame: bool,
}

impl LsfState {
    pub fn new() -> Self {
        Self {
            last_lsf: LSF_MEAN,
            first_frame: true,
        }
    }

    /// Reset to first-frame conditions. Used on seek / error recovery.
    pub fn reset(&mut self) {
        self.last_lsf = LSF_MEAN;
        self.first_frame = true;
    }
}

impl Default for LsfState {
    fn default() -> Self {
        Self::new()
    }
}

/// Enforce strict increase of LSF values with at least `LSF_MIN_GAP`
/// between neighbours (RFC 3951 §3.2.5).
///
/// The RFC's procedure iterates twice, nudging any violating pair by
/// half the gap deficit — we apply a simpler forward pass which is
/// sufficient on any monotone-plus-noise input and is stable for our
/// synthesised codebook.
pub fn stabilise_lsf(lsf: &mut [f32; LPC_ORDER]) {
    // Clamp first LSF away from 0.
    if lsf[0] < LSF_MIN_GAP * 0.5 {
        lsf[0] = LSF_MIN_GAP * 0.5;
    }
    for k in 1..LPC_ORDER {
        let min_k = lsf[k - 1] + LSF_MIN_GAP;
        if lsf[k] < min_k {
            lsf[k] = min_k;
        }
    }
    // Also cap the top LSF slightly below π so the filter stays stable.
    let pi = std::f32::consts::PI;
    if lsf[LPC_ORDER - 1] > pi - LSF_MIN_GAP * 0.5 {
        lsf[LPC_ORDER - 1] = pi - LSF_MIN_GAP * 0.5;
        // Re-walk downward to preserve the gap.
        for k in (1..LPC_ORDER).rev() {
            let max_k = lsf[k] - LSF_MIN_GAP;
            if lsf[k - 1] > max_k {
                lsf[k - 1] = max_k;
            }
        }
    }
}

/// Dequantise one LSF vector from its three split-VQ indices. The
/// result is stabilised in place before return.
pub fn dequant_lsf(indices: &[u16; 3]) -> [f32; LPC_ORDER] {
    let mut lsf = assemble_lsf(indices);
    stabilise_lsf(&mut lsf);
    lsf
}

/// Linear interpolation between two LSF vectors.
/// `alpha=0` returns `a`, `alpha=1` returns `b`.
pub fn interpolate(a: &[f32; LPC_ORDER], b: &[f32; LPC_ORDER], alpha: f32) -> [f32; LPC_ORDER] {
    let mut out = [0.0f32; LPC_ORDER];
    for k in 0..LPC_ORDER {
        out[k] = a[k] * (1.0 - alpha) + b[k] * alpha;
    }
    // Re-stabilise after interpolation — mixing two monotone vectors
    // can still tighten gaps at split boundaries.
    stabilise_lsf(&mut out);
    out
}

/// Convert a 10-dim LSF vector (in radians, 0..π) to a 10th-order
/// LPC polynomial. Uses the sum/difference polynomial trick:
/// - P(z) has zeros at the even-indexed LSFs, Q(z) at the odd ones.
/// - A(z) = (P(z) + Q(z)) / 2  (with the canonical constant term 1).
///
/// Returns an array where `a[0] = 1.0` and `a[1..=10]` are the LPC
/// coefficients in predictor convention: y(n) = x(n) - Σ a[i] y(n-i).
pub fn lsf_to_lpc(lsf: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER + 1] {
    // Build P(z) and Q(z) as polynomials with roots at cos(lsf[i]).
    // Each root `r = cos(lsf[i])` contributes a factor (1 - 2r z^-1 + z^-2).
    // P gets even-indexed LSFs, Q gets odd-indexed LSFs.
    let mut p = [0.0f64; LPC_ORDER + 2];
    let mut q = [0.0f64; LPC_ORDER + 2];
    p[0] = 1.0;
    q[0] = 1.0;

    let half = LPC_ORDER / 2; // 5
    for i in 0..half {
        // Even-indexed LSF (0, 2, 4, 6, 8) -> P.
        let r_even = (lsf[2 * i] as f64).cos();
        // Multiply p(z) by (1 - 2*r_even z^-1 + z^-2):
        // Walk from high degree to low to avoid in-place clobbering.
        let len = 2 * (i + 1);
        for k in (0..=len).rev() {
            let v2 = if k >= 2 { p[k - 2] } else { 0.0 };
            let v1 = if k >= 1 { p[k - 1] } else { 0.0 };
            p[k] = p[k] + v2 - 2.0 * r_even * v1;
        }
        // Odd-indexed LSF -> Q.
        let r_odd = (lsf[2 * i + 1] as f64).cos();
        for k in (0..=len).rev() {
            let v2 = if k >= 2 { q[k - 2] } else { 0.0 };
            let v1 = if k >= 1 { q[k - 1] } else { 0.0 };
            q[k] = q[k] + v2 - 2.0 * r_odd * v1;
        }
    }
    // Multiply P by (1 + z^-1) and Q by (1 - z^-1) per the standard LSF
    // factorisation.
    let mut pp = [0.0f64; LPC_ORDER + 2];
    let mut qq = [0.0f64; LPC_ORDER + 2];
    for k in 0..=LPC_ORDER {
        pp[k] = p[k] + if k == 0 { 0.0 } else { p[k - 1] };
        qq[k] = q[k] - if k == 0 { 0.0 } else { q[k - 1] };
    }
    // A(z) = (P(z) + Q(z)) / 2 — canonical form keeping a[0]=1.
    let mut a = [0.0f32; LPC_ORDER + 1];
    for k in 0..=LPC_ORDER {
        a[k] = ((pp[k] + qq[k]) * 0.5) as f32;
    }
    // Normalise so a[0] == 1 exactly (numeric robustness).
    let a0 = a[0];
    if a0.abs() > 1e-12 {
        for a_k in a.iter_mut() {
            *a_k /= a0;
        }
    } else {
        a[0] = 1.0;
    }
    a
}

/// Produce per-subblock LPC coefficients from the frame's LSF vector(s)
/// plus the last-frame tail state.
///
/// Returns a vector of length `mode.sub_blocks()` where each entry is a
/// full `[1, a1, ..., a10]` LPC row (predictor convention). The state is
/// advanced (its `last_lsf` becomes the current frame's tail LSF).
pub fn decode_and_interpolate(
    state: &mut LsfState,
    mode: FrameMode,
    lsf_vectors: &[[f32; LPC_ORDER]],
) -> Vec<[f32; LPC_ORDER + 1]> {
    let n_sub = mode.sub_blocks();
    let mut a_per_sub: Vec<[f32; LPC_ORDER + 1]> = Vec::with_capacity(n_sub);

    match mode {
        FrameMode::Ms20 => {
            // One LSF vector per frame. RFC 3.2.7: weight (4-n)/4 for old,
            // n/4 for new, for sub-block n=1..4 (so the 4th uses the
            // current LSF fully, the 1st uses mostly the previous tail).
            let cur = &lsf_vectors[0];
            for n in 1..=n_sub {
                let alpha = (n as f32) / (n_sub as f32);
                let lsf_k = interpolate(&state.last_lsf, cur, alpha);
                a_per_sub.push(lsf_to_lpc(&lsf_k));
            }
            state.last_lsf = *cur;
        }
        FrameMode::Ms30 => {
            // Two LSF vectors: lsf1 (first half) and lsf2 (second half).
            // §3.2.6: sub-block 1 = mean(prev_tail, lsf1); sub-blocks 2..5
            // interpolate between lsf1 and lsf2; sub-block 6 = lsf2.
            let lsf1 = &lsf_vectors[0];
            let lsf2 = &lsf_vectors[1];
            // Sub-block 0 (the "first sub-block" in the RFC) is the
            // average.
            let lsf_sb0 = interpolate(&state.last_lsf, lsf1, 0.5);
            a_per_sub.push(lsf_to_lpc(&lsf_sb0));
            // Sub-blocks 1..4 interpolate between lsf1 and lsf2.
            // The RFC says 2..5 linearly interpolate, with sub-block 2
            // = lsf1 and sub-block 5 = lsf2. We have 6 sub-blocks total
            // (indices 0..5). We already handled sub-block 0. Now
            // sub-blocks 1..4 linearly interpolate, with alpha = (n)/4
            // for n=0..3, i.e. alpha ∈ {0, 1/4, 2/4, 3/4}. Finally
            // sub-block 5 = lsf2.
            for n in 0..(n_sub - 2) {
                let alpha = (n as f32) / ((n_sub - 2) as f32);
                let lsf_k = interpolate(lsf1, lsf2, alpha);
                a_per_sub.push(lsf_to_lpc(&lsf_k));
            }
            // Final sub-block uses lsf2 outright.
            a_per_sub.push(lsf_to_lpc(lsf2));
            state.last_lsf = *lsf2;
        }
    }

    state.first_frame = false;
    a_per_sub
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stabilise_enforces_gap() {
        let mut lsf = [0.1f32; LPC_ORDER];
        stabilise_lsf(&mut lsf);
        for k in 1..LPC_ORDER {
            assert!(lsf[k] >= lsf[k - 1] + LSF_MIN_GAP * 0.999);
        }
    }

    #[test]
    fn interpolate_endpoints() {
        let a = LSF_MEAN;
        let mut b = LSF_MEAN;
        for v in b.iter_mut() {
            *v *= 1.1;
        }
        let out = interpolate(&a, &b, 0.0);
        for k in 0..LPC_ORDER {
            assert!((out[k] - a[k]).abs() < 1e-3);
        }
    }

    #[test]
    fn lsf_to_lpc_mean_is_stable() {
        let a = lsf_to_lpc(&LSF_MEAN);
        // a[0] must be exactly 1.
        assert!((a[0] - 1.0).abs() < 1e-6);
        // Sum A(1) > 0 for a stable all-pole filter at DC.
        let a1: f32 = a.iter().sum();
        assert!(a1 > 0.0, "A(1) not positive: {}", a1);
    }

    #[test]
    fn decode_interpolate_20ms_produces_four_subs() {
        let mut s = LsfState::new();
        let lsf = LSF_MEAN;
        let per = decode_and_interpolate(&mut s, FrameMode::Ms20, &[lsf]);
        assert_eq!(per.len(), 4);
        for row in &per {
            assert!((row[0] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn decode_interpolate_30ms_produces_six_subs() {
        let mut s = LsfState::new();
        let lsf1 = LSF_MEAN;
        let mut lsf2 = LSF_MEAN;
        for v in lsf2.iter_mut() {
            *v *= 1.05;
        }
        let per = decode_and_interpolate(&mut s, FrameMode::Ms30, &[lsf1, lsf2]);
        assert_eq!(per.len(), 6);
    }
}
