#![allow(
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::unnecessary_cast,
    clippy::approx_constant,
    clippy::excessive_precision
)]
//! Synthetic-signal SNR measurement for the start-state reconstruction.
//!
//! We can't call a full iLBC encoder (there isn't one in this crate),
//! but we can exercise `state::reconstruct_scalar_state` against a
//! synthetic mini-encoder that mirrors RFC 3951 Appendix A.46
//! (`StateSearchW`): take a residual, all-pass filter it, quantise the
//! log-amplitude (6 bits) and per-sample shape (3 bits), then let the
//! decoder reconstruct. The SNR of reconstructed vs. the original
//! residual tells us how well the decode-side all-pass / scale inversion
//! tracks the encoder's operations.
//!
//! This test is printed rather than asserted — it's a diagnostic run
//! via `cargo test -- --nocapture state_snr`.

use oxideav_ilbc::state::{reconstruct_scalar_state, STATE_FRGQ_TBL, STATE_SQ3_TBL};
use oxideav_ilbc::FrameMode;

/// All-zero filter: `out[n] = sum_k coef[k] * in[n-k]`, coef[0] = 1.0.
fn all_zero(inp: &[f32], coef: &[f32], out: &mut [f32], order: usize) {
    assert_eq!(inp.len(), out.len());
    for n in 0..inp.len() {
        let mut s = coef[0] * inp[n];
        for k in 1..=order {
            let idx = n as isize - k as isize;
            if idx >= 0 {
                s += coef[k] * inp[idx as usize];
            }
        }
        out[n] = s;
    }
}

/// All-pole filter in place: `out[n] -= sum_k coef[k] * out[n-k]`.
fn all_pole(io: &mut [f32], coef: &[f32], order: usize) {
    for n in 0..io.len() {
        for k in 1..=order {
            let idx = n as isize - k as isize;
            if idx >= 0 {
                io[n] -= coef[k] * io[idx as usize];
            }
        }
    }
}

fn sort_sq(value: f32, tbl: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = (tbl[0] - value).abs();
    for i in 1..tbl.len() {
        let d = (tbl[i] - value).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn encode_state(residual: &[f32], a: &[f32]) -> (u8, Vec<u8>) {
    let n = residual.len();
    let order = a.len() - 1;

    // Numerator = reverse(a) with a[0] at end (per RFC 3951 §3.5.2 /
    // Appendix A.46 StateSearchW).
    let mut numerator = vec![0.0f32; order + 1];
    for k in 0..order {
        numerator[k] = a[order - k];
    }
    numerator[order] = a[0];

    // Double buffer: residual || 0, then circular-all-pass.
    let mut tmp = vec![0.0f32; 2 * n];
    tmp[..n].copy_from_slice(residual);

    let mut fout = vec![0.0f32; 2 * n];
    all_zero(&tmp, &numerator, &mut fout, order);
    all_pole(&mut fout, a, order);

    // Circular fold.
    let mut folded = vec![0.0f32; n];
    for k in 0..n {
        folded[k] = fout[k] + fout[k + n];
    }

    // Max amplitude (signed), then log10, quantise.
    let mut max_val_signed = folded[0];
    for &v in folded.iter().skip(1) {
        if v * v > max_val_signed * max_val_signed {
            max_val_signed = v;
        }
    }
    let mut max_val = max_val_signed.abs();
    if max_val < 10.0 {
        max_val = 10.0;
    }
    max_val = max_val.log10();
    let idx_for_max = sort_sq(max_val, STATE_FRGQ_TBL.as_slice()) as u8;

    // Apply scal = 4.5 / qmax.
    let qmax = 10f32.powf(STATE_FRGQ_TBL[idx_for_max as usize]);
    let scal = 4.5 / qmax;
    for v in folded.iter_mut() {
        *v *= scal;
    }

    // Predictive scalar quantisation — for the SNR benchmark we use a
    // simple direct scalar quantisation (no DPCM), which upper-bounds
    // the achievable SNR but still exercises the shape table.
    let mut idx_vec = vec![0u8; n];
    for k in 0..n {
        idx_vec[k] = sort_sq(folded[k], STATE_SQ3_TBL.as_slice()) as u8;
    }

    (idx_for_max, idx_vec)
}

fn snr_db(reference: &[f32], test: &[f32]) -> f32 {
    assert_eq!(reference.len(), test.len());
    let mut s_sig = 0.0f64;
    let mut s_err = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let d = (t - r) as f64;
        s_sig += (r as f64) * (r as f64);
        s_err += d * d;
    }
    if s_err < 1e-30 {
        return 1e6;
    }
    10.0 * (s_sig / s_err).log10() as f32
}

/// Stable test LPC — a low-pass all-pole from a 2nd-order prototype
/// padded to order 10. Gains are chosen so the filter is strictly
/// stable and produces reasonable speech-like spectral tilt.
fn test_lpc() -> [f32; 11] {
    let mut a = [0.0f32; 11];
    a[0] = 1.0;
    a[1] = -0.6;
    a[2] = 0.15;
    a[3] = -0.03;
    a[4] = 0.01;
    a
}

#[test]
fn state_roundtrip_snr_20ms() {
    let a = test_lpc();
    // Synthetic residual: sine + noise at moderate amplitude.
    let n = 57usize;
    let mut residual = vec![0.0f32; n];
    for k in 0..n {
        let t = k as f32;
        residual[k] = 500.0 * (2.0 * std::f32::consts::PI * t / 13.0).sin()
            + 300.0 * (2.0 * std::f32::consts::PI * t / 7.5).cos();
    }

    let (idx_for_max, idx_vec) = encode_state(&residual, &a);
    let recon = reconstruct_scalar_state(FrameMode::Ms20, idx_for_max, &idx_vec, &a);

    let snr = snr_db(&residual, &recon);
    println!("state_roundtrip_snr_20ms: SNR = {:.2} dB", snr);
}

#[test]
fn state_roundtrip_snr_30ms() {
    let a = test_lpc();
    let n = 58usize;
    let mut residual = vec![0.0f32; n];
    for k in 0..n {
        let t = k as f32;
        residual[k] = 800.0 * (2.0 * std::f32::consts::PI * t / 11.0).sin()
            + 250.0 * (2.0 * std::f32::consts::PI * t / 5.0).cos();
    }

    let (idx_for_max, idx_vec) = encode_state(&residual, &a);
    let recon = reconstruct_scalar_state(FrameMode::Ms30, idx_for_max, &idx_vec, &a);

    let snr = snr_db(&residual, &recon);
    println!("state_roundtrip_snr_30ms: SNR = {:.2} dB", snr);
}
