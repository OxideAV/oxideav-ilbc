//! LPC analysis for the iLBC encoder — RFC 3951 §3.2.
//!
//! Pipeline:
//!   1. Window the (high-pass filtered) input buffer.
//!   2. Compute autocorrelation coefficients up to LPC_ORDER.
//!   3. Lag-window (spectral smoothing + 40 dB noise floor).
//!   4. Levinson-Durbin -> LPC predictor coefficients.
//!   5. Apply chirp bandwidth expansion (0.9^i).
//!   6. Convert to LSF (radians 0..pi).
//!
//! For the 20 ms mode, a single asymmetric 240-sample window is used,
//! centred on the third sub-block. For the 30 ms mode two windows are
//! used: a symmetric Hanning centered on the second sub-block's mid-
//! point, and an asymmetric window centered on the fifth sub-block.
//!
//! The resulting LSF vectors are *unquantised* — split-VQ quantisation
//! is applied in [`crate::lsf_quant`].

use crate::LPC_ORDER;

/// Number of "look-back" samples: 80 for 20 ms, 60 for 30 ms.
pub const LPC_LOOKBACK_20MS: usize = 80;
pub const LPC_LOOKBACK_30MS: usize = 60;

/// Window length: BLOCKL + lookback. 240 samples for both modes.
pub const LPC_WINLEN: usize = 240;

/// Chirp bandwidth-expansion factor recommended by the RFC (§3.2.2).
pub const LPC_CHIRP: f32 = 0.9;

/// Symmetric Hanning window (240 samples) — RFC 3951 §3.2.1 `lpc_winTbl`.
///
/// ```text
///   lpc_winTbl[i] = 0.5*(1.0 - cos(2π(i+1)/(BLOCKL+1))),  i=0..119
///   lpc_winTbl[i] = lpc_winTbl[BLOCKL - i - 1],            i=120..239
/// ```
/// where BLOCKL == 240.
pub fn hanning_window() -> [f32; LPC_WINLEN] {
    let mut w = [0.0f32; LPC_WINLEN];
    // Standard mirrored Hanning over 240 samples.
    let blockl = LPC_WINLEN as f32;
    for (i, w_i) in w.iter_mut().enumerate().take(LPC_WINLEN / 2) {
        *w_i =
            0.5 * (1.0 - (2.0 * core::f32::consts::PI * (i as f32 + 1.0) / (blockl + 1.0)).cos());
    }
    for i in (LPC_WINLEN / 2)..LPC_WINLEN {
        w[i] = w[LPC_WINLEN - i - 1];
    }
    w
}

/// Asymmetric window — RFC 3951 §3.2.1 `lpc_asymwinTbl`, 240 samples.
///
/// ```text
///   lpc_asymwinTbl[i] = (sin(π(i+1)/441))^2,  i=0..219
///   lpc_asymwinTbl[i] = cos((i-220)π/40),     i=220..239
/// ```
pub fn asymmetric_window() -> [f32; LPC_WINLEN] {
    let mut w = [0.0f32; LPC_WINLEN];
    for (i, w_i) in w.iter_mut().enumerate().take(220) {
        let s = (core::f32::consts::PI * (i as f32 + 1.0) / 441.0).sin();
        *w_i = s * s;
    }
    for (i, w_i) in w.iter_mut().enumerate().take(LPC_WINLEN).skip(220) {
        *w_i = ((i as f32 - 220.0) * core::f32::consts::PI / 40.0).cos();
    }
    w
}

/// Lag window — RFC 3951 §3.2.1 `lpc_lagwinTbl`.
///
/// ```text
///   lpc_lagwinTbl[0] = 1.0001  (40 dB white-noise floor)
///   lpc_lagwinTbl[i] = exp(-0.5 * (2π * 60.0 * i / FS)^2), i=1..10, FS=8000
/// ```
pub fn lag_window() -> [f32; LPC_ORDER + 1] {
    let mut w = [0.0f32; LPC_ORDER + 1];
    w[0] = 1.0001;
    for (i, w_i) in w.iter_mut().enumerate().skip(1) {
        let x = 2.0 * core::f32::consts::PI * 60.0 * (i as f32) / 8000.0;
        *w_i = (-0.5 * x * x).exp();
    }
    w
}

/// Autocorrelation of the windowed signal up to lag `LPC_ORDER`.
///
/// `x` has length `LPC_WINLEN`. Output has length `LPC_ORDER + 1`.
pub fn autocorrelate(x: &[f32; LPC_WINLEN]) -> [f32; LPC_ORDER + 1] {
    let mut r = [0.0f32; LPC_ORDER + 1];
    for lag in 0..=LPC_ORDER {
        let mut s = 0.0f64;
        for n in 0..(LPC_WINLEN - lag) {
            s += (x[n] as f64) * (x[n + lag] as f64);
        }
        r[lag] = s as f32;
    }
    r
}

/// Levinson-Durbin recursion — produces `[1, a1, ..., a10]` in predictor
/// convention (y(n) = x(n) - Σ a[k] y(n-k)).
///
/// Returns `[1.0; LPC_ORDER+1]` if the recursion fails (near-singular
/// autocorrelation), which yields a pass-through filter.
pub fn levinson_durbin(r: &[f32; LPC_ORDER + 1]) -> [f32; LPC_ORDER + 1] {
    let mut a = [0.0f64; LPC_ORDER + 1];
    a[0] = 1.0;
    if r[0] <= 0.0 {
        let mut out = [0.0f32; LPC_ORDER + 1];
        out[0] = 1.0;
        return out;
    }
    let mut err = r[0] as f64;
    let mut tmp = [0.0f64; LPC_ORDER + 1];
    for i in 1..=LPC_ORDER {
        // Reflection coefficient.
        let mut acc = -(r[i] as f64);
        for j in 1..i {
            acc -= a[j] * (r[i - j] as f64);
        }
        if err.abs() < 1e-30 {
            let mut out = [0.0f32; LPC_ORDER + 1];
            out[0] = 1.0;
            return out;
        }
        let k = acc / err;
        // Update: new_a[j] = a[j] + k * a[i-j], for j=1..i-1. new_a[i] = k.
        tmp[..i].copy_from_slice(&a[..i]);
        a[i] = k;
        for j in 1..i {
            a[j] = tmp[j] + k * tmp[i - j];
        }
        err *= 1.0 - k * k;
        if err <= 0.0 {
            // Loss of stability — bail out with predictor so far.
            break;
        }
    }
    let mut out = [0.0f32; LPC_ORDER + 1];
    for k in 0..=LPC_ORDER {
        out[k] = a[k] as f32;
    }
    out
}

/// In-place chirp bandwidth expansion: a[i] *= chirp^i for i=0..=LPC_ORDER.
pub fn chirp_expand(a: &mut [f32; LPC_ORDER + 1], chirp: f32) {
    let mut c = 1.0f32;
    for a_k in a.iter_mut() {
        *a_k *= c;
        c *= chirp;
    }
}

/// Multiply autocorrelation coefficients by the lag window in-place.
pub fn apply_lag_window(r: &mut [f32; LPC_ORDER + 1]) {
    let w = lag_window();
    for (r_i, &w_i) in r.iter_mut().zip(w.iter()) {
        *r_i *= w_i;
    }
}

/// Compute an LPC polynomial from a windowed block of length `LPC_WINLEN`.
/// Combines autocorrelation, lag-windowing, Levinson-Durbin, and chirp
/// expansion. The returned polynomial has `a[0] = 1`.
pub fn block_lpc(windowed: &[f32; LPC_WINLEN]) -> [f32; LPC_ORDER + 1] {
    let mut r = autocorrelate(windowed);
    apply_lag_window(&mut r);
    let mut a = levinson_durbin(&r);
    chirp_expand(&mut a, LPC_CHIRP);
    a
}

/// Convert an LPC polynomial `[1, a1..a10]` to an LSF vector in radians
/// 0..π. Uses a simple root-searching approach: evaluate the symmetric
/// P(z) = A(z) + z^-(p+1) A(z^-1) and antisymmetric Q(z) = A(z) - z^-(p+1)
/// A(z^-1) polynomials on the unit circle, find sign changes, refine by
/// bisection.
///
/// This is adequate for the encoder's quantisation purposes: quantised
/// LSFs go through [`crate::lsf::stabilise_lsf`] before they reach
/// synthesis.
pub fn lpc_to_lsf(a: &[f32; LPC_ORDER + 1]) -> [f32; LPC_ORDER] {
    // Build P(z) = A(z) + z^{-(p+1)} A(z^{-1}), Q(z) = A(z) - z^{-(p+1)} A(z^{-1})
    // symmetric / antisymmetric polynomials of length p+2. Coefficient at z^{-k}
    // is a[k] ± a[p+1-k] with a[p+1] = 0.
    let p = LPC_ORDER;
    let mut psum = [0.0f64; LPC_ORDER + 2];
    let mut qsum = [0.0f64; LPC_ORDER + 2];
    for k in 0..=(p + 1) {
        let ak = if k <= p { a[k] as f64 } else { 0.0 };
        let ar = if (p + 1 - k) <= p {
            a[p + 1 - k] as f64
        } else {
            0.0
        };
        psum[k] = ak + ar;
        qsum[k] = ak - ar;
    }
    // Remove trivial roots: P has a root at z=-1 (divide by 1+z^{-1}),
    // Q has a root at z=+1 (divide by 1-z^{-1}). After division both
    // are symmetric/antisymmetric polynomials of length p+1.
    // Since we only need sign-changes on the unit circle, we can
    // evaluate P and Q as-is — the trivial roots are at ω=π (cos(π)=-1,
    // contributing to P) and ω=0 (contributing to Q). Those aren't
    // interior LSFs.

    // Evaluate |P(e^{jω})| and |Q(e^{jω})| on a grid and find
    // alternating zero-crossings. Because P and Q have interlacing
    // roots, we search P first over the grid, then Q, alternating.
    let n_grid = 512;
    let mut grid = [0.0f64; 513];
    for (i, g) in grid.iter_mut().enumerate() {
        *g = core::f64::consts::PI * (i as f64) / (n_grid as f64);
    }
    let eval = |coeffs: &[f64; LPC_ORDER + 2], omega: f64| -> f64 {
        let mut s = 0.0f64;
        for (k, &c) in coeffs.iter().enumerate().take(p + 2) {
            s += c * (-(k as f64) * omega).cos();
            // Only real part matters for symmetric/antisymmetric on unit circle.
        }
        s
    };
    // Collect roots of P and Q on (0, π).
    let mut p_roots: Vec<f64> = Vec::new();
    let mut q_roots: Vec<f64> = Vec::new();
    let mut prev_p = eval(&psum, grid[0]);
    let mut prev_q = eval(&qsum, grid[0]);
    for i in 1..=n_grid {
        let cur_p = eval(&psum, grid[i]);
        let cur_q = eval(&qsum, grid[i]);
        if prev_p * cur_p < 0.0 {
            let root = bisect(&psum, grid[i - 1], grid[i], &eval);
            if root > 1e-5 && root < core::f64::consts::PI - 1e-5 {
                p_roots.push(root);
            }
        }
        if prev_q * cur_q < 0.0 {
            let root = bisect(&qsum, grid[i - 1], grid[i], &eval);
            if root > 1e-5 && root < core::f64::consts::PI - 1e-5 {
                q_roots.push(root);
            }
        }
        prev_p = cur_p;
        prev_q = cur_q;
    }

    // LSFs: interleave q, p, q, p, ... starting with Q (lowest root = ω_1).
    // RFC ordering: LSF[0] < LSF[1] < ... < LSF[9], alternating between
    // the two polynomials. We'll just merge and sort, keeping 10 smallest.
    let mut all: Vec<f64> = p_roots.into_iter().chain(q_roots).collect();
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // If root-finding was unstable, fall back to LSF_MEAN.
    let mut lsf = [0.0f32; LPC_ORDER];
    if all.len() < LPC_ORDER {
        // Fallback: LSF_MEAN.
        lsf.copy_from_slice(&crate::lsf_tables::LSF_MEAN);
        return lsf;
    }
    for k in 0..LPC_ORDER {
        lsf[k] = all[k] as f32;
    }
    lsf
}

fn bisect<F>(coeffs: &[f64; LPC_ORDER + 2], mut lo: f64, mut hi: f64, eval: &F) -> f64
where
    F: Fn(&[f64; LPC_ORDER + 2], f64) -> f64,
{
    let mut f_lo = eval(coeffs, lo);
    // 30 iterations -> ~10^-9 radians.
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        let f_mid = eval(coeffs, mid);
        if f_lo * f_mid <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    0.5 * (lo + hi)
}

/// Helper: compute an LSF vector from a windowed 240-sample input. Uses
/// the symmetric Hanning window if `use_hanning` is true, otherwise the
/// asymmetric window.
pub fn windowed_lsf(input: &[f32; LPC_WINLEN], use_hanning: bool) -> [f32; LPC_ORDER] {
    let w = if use_hanning {
        hanning_window()
    } else {
        asymmetric_window()
    };
    let mut windowed = [0.0f32; LPC_WINLEN];
    for i in 0..LPC_WINLEN {
        windowed[i] = input[i] * w[i];
    }
    let a = block_lpc(&windowed);
    lpc_to_lsf(&a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hanning_window_ends_and_peak() {
        let w = hanning_window();
        // Endpoints close to 0.
        assert!(w[0] < 0.001);
        assert!(w[LPC_WINLEN - 1] < 0.001);
        // Peak near the middle ~1.0.
        assert!(w[119] > 0.99);
        assert!(w[120] > 0.99);
    }

    #[test]
    fn asym_window_starts_and_transitions() {
        let w = asymmetric_window();
        // Starts near 0 (sin(pi/441)^2 ~ small).
        assert!(w[0] < 1e-3);
        // Peak near sample 219.
        assert!(w[219] > 0.99);
        // Last sample: cos(19*pi/40).
        let expected = (19.0 * core::f32::consts::PI / 40.0).cos();
        assert!((w[239] - expected).abs() < 1e-5);
    }

    #[test]
    fn levinson_durbin_constant_signal() {
        // Signal = DC constant: all autocorrelation coefficients equal.
        // LPC of pure DC should be a[0]=1, a[1]=-1, rest 0 (a 1-tap
        // predictor that outputs y[n] = x[n] - x[n-1]).
        let mut r = [0.0f32; LPC_ORDER + 1];
        for v in r.iter_mut() {
            *v = 1.0;
        }
        // Apply a tiny lag-window to avoid singular recursion.
        r[0] *= 1.0001;
        let a = levinson_durbin(&r);
        assert_eq!(a[0], 1.0);
        // Predictor is not guaranteed to be exact with lag-window, but
        // should be finite.
        for v in &a {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn block_lpc_silence_is_identity_ish() {
        // All-zero input: a[0]=1, rest tiny or zero.
        let x = [0.0f32; LPC_WINLEN];
        let a = block_lpc(&x);
        assert_eq!(a[0], 1.0);
    }

    #[test]
    fn lpc_to_lsf_monotone_for_impulse() {
        // Minimum phase stable LPC from a real-ish input.
        let input: [f32; LPC_WINLEN] = core::array::from_fn(|i| ((i as f32) * 0.1).sin() * 1000.0);
        let w = asymmetric_window();
        let windowed: [f32; LPC_WINLEN] = core::array::from_fn(|i| input[i] * w[i]);
        let a = block_lpc(&windowed);
        let lsf = lpc_to_lsf(&a);
        // Check monotone ordering.
        for k in 1..LPC_ORDER {
            assert!(
                lsf[k] > lsf[k - 1],
                "LSF not monotone at {k}: {} -> {}",
                lsf[k - 1],
                lsf[k]
            );
        }
        // In range (0, π).
        for &v in &lsf {
            assert!(v > 0.0 && v < core::f32::consts::PI);
        }
    }

    #[test]
    fn chirp_expand_scales() {
        let mut a = [1.0f32; LPC_ORDER + 1];
        chirp_expand(&mut a, 0.9);
        assert_eq!(a[0], 1.0);
        assert!((a[1] - 0.9).abs() < 1e-6);
        assert!((a[2] - 0.81).abs() < 1e-5);
    }
}
