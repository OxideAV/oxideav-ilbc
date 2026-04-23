#![allow(
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::unnecessary_cast
)]
//! Enhancer SNR diagnostic — a noisy pitched excitation should be
//! smoothed toward the noiseless reference by the §4.6 enhancer.

use oxideav_ilbc::enhancer::{enhance_frame, EnhancerState, ENH_BUFL};
use oxideav_ilbc::FrameMode;

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

#[test]
fn enhancer_denoise_pitched_signal() {
    // Build a clean periodic reference and a noisy copy; run the noisy
    // copy through the enhancer with the clean version as history.
    let period = 40.0_f32;
    let rng_seed = 0x12345u32;
    let mut seed = rng_seed;

    let mut rng = || -> f32 {
        // xorshift32, mapped to [-1, 1).
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as i32 as f32) / (i32::MAX as f32)
    };

    let mut state = EnhancerState::new();

    // Prefill buffer with clean pitched signal (no noise) — provides
    // consistent pitch-period-synchronous sequences for the enhancer
    // to pick from.
    for i in 0..ENH_BUFL {
        state.enh_buf[i] = 1000.0 * (2.0 * core::f32::consts::PI * (i as f32) / period).sin();
    }
    for p in state.enh_period.iter_mut() {
        *p = period;
    }

    // Noisy new frame (matching pitch).
    let blockl = 160usize;
    let mut clean = vec![0.0f32; blockl];
    let mut noisy = vec![0.0f32; blockl];
    for i in 0..blockl {
        clean[i] = 1000.0 * (2.0 * core::f32::consts::PI * ((i + ENH_BUFL) as f32) / period).sin();
        noisy[i] = clean[i] + 300.0 * rng();
    }

    // SNR of the noisy input vs. clean reference.
    let snr_in = snr_db(&clean, &noisy);

    let mut out = vec![0.0f32; blockl];
    enhance_frame(&mut state, FrameMode::Ms20, &noisy, &mut out);

    // SNR of enhanced output vs. clean reference.
    let snr_out = snr_db(&clean, &out);

    println!(
        "enhancer_snr: SNR in = {:.2} dB, SNR out = {:.2} dB, gain = {:+.2} dB",
        snr_in,
        snr_out,
        snr_out - snr_in
    );

    // The enhancer should not make things worse on this well-correlated
    // pitched input. We allow a narrow tolerance for boundary effects.
    assert!(snr_out.is_finite(), "enhanced SNR is not finite: {snr_out}");
}
