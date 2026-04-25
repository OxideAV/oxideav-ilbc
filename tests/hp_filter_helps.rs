//! Integration test: enabling the optional HP pre-processing filter
//! (RFC 3951 §3.1) should produce a *cleaner* output than leaving it
//! off when the input contains DC + 50 Hz mains hum on top of voiced
//! speech-like content.
//!
//! Methodology:
//! 1. Build a "clean" reference signal (sustained voiced 130 Hz +
//!    harmonics, ~1 s).
//! 2. Add a strong DC bias and a 50 Hz mains hum to make a "dirty"
//!    input.
//! 3. Encode the dirty input twice — once with `hp_filter=on`, once
//!    without — and decode each through the iLBC decoder.
//! 4. Compute SNR of each decoded output against the **clean** reference.
//! 5. Assert the HP-filter-on case beats the HP-filter-off case by at
//!    least a few dB.

use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_core::{CodecRegistry, Encoder};

use oxideav_ilbc::{FrameMode, CODEC_ID_STR, SAMPLE_RATE};

fn gen_voiced_clean(samples: usize) -> Vec<f32> {
    let f0 = 130.0f32;
    (0..samples)
        .map(|n| {
            let t = n as f32 / SAMPLE_RATE as f32;
            let mut v = 0.0f32;
            for h in 1..5 {
                v +=
                    ((2.0 * core::f32::consts::PI * h as f32 * f0 * t).sin()) * (3000.0 / h as f32);
            }
            v
        })
        .collect()
}

fn add_hum_and_dc(clean: &[f32], dc: f32, hum_hz: f32, hum_amp: f32) -> Vec<i16> {
    clean
        .iter()
        .enumerate()
        .map(|(n, &v)| {
            let t = n as f32 / SAMPLE_RATE as f32;
            let dirty = v + dc + hum_amp * (2.0 * core::f32::consts::PI * hum_hz * t).sin();
            dirty.round().clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

fn pcm_to_audio_frame(pcm: &[i16]) -> Frame {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels: 1,
        sample_rate: SAMPLE_RATE,
        samples: pcm.len() as u32,
        pts: Some(0),
        time_base: TimeBase::new(1, SAMPLE_RATE as i64),
        data: vec![bytes],
    })
}

fn round_trip(mode: FrameMode, hp: bool, pcm: &[i16]) -> Vec<i16> {
    let mut reg = CodecRegistry::new();
    oxideav_ilbc::register(&mut reg);

    let mut enc_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    enc_params.sample_rate = Some(SAMPLE_RATE);
    enc_params.channels = Some(1);
    enc_params.sample_format = Some(SampleFormat::S16);
    let mut options = CodecOptions::new();
    if mode == FrameMode::Ms30 {
        options = options.set("frame_ms", "30");
    }
    if hp {
        options = options.set("hp_filter", "on");
    }
    enc_params.options = options;
    let mut enc: Box<dyn Encoder> = reg.make_encoder(&enc_params).expect("encoder");

    let mut dec_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    dec_params.sample_rate = Some(SAMPLE_RATE);
    dec_params.channels = Some(1);
    let mut dec = reg.make_decoder(&dec_params).expect("decoder");

    enc.send_frame(&pcm_to_audio_frame(pcm)).unwrap();
    enc.flush().unwrap();

    let mut decoded = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        let decoder_pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), pkt.data.clone());
        dec.send_packet(&decoder_pkt).unwrap();
        if let Frame::Audio(a) = dec.receive_frame().unwrap() {
            for chunk in a.data[0].chunks_exact(2) {
                decoded.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
    }
    decoded
}

fn per_frame_best_snr_avg(
    reference: &[i16],
    test: &[i16],
    frame_len: usize,
    skip_frames: usize,
) -> f64 {
    let n_frames = reference.len() / frame_len;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for f in skip_frames..n_frames {
        let lo = f * frame_len;
        let hi = lo + frame_len;
        if hi > test.len() {
            break;
        }
        let r = &reference[lo..hi];
        let t = &test[lo..hi];
        let max_lag = (frame_len / 2) as isize;
        let mut best = f64::NEG_INFINITY;
        for lag in -max_lag..=max_lag {
            let mut s_sig = 0.0f64;
            let mut s_err = 0.0f64;
            for (i, &r_sample) in r.iter().enumerate() {
                let j = i as isize + lag;
                if j < 0 || j as usize >= t.len() {
                    continue;
                }
                let rv = r_sample as f64;
                let tv = t[j as usize] as f64;
                s_sig += rv * rv;
                s_err += (rv - tv) * (rv - tv);
            }
            if s_err < 1e-9 {
                best = f64::INFINITY;
                break;
            }
            let snr = 10.0 * (s_sig / s_err).log10();
            if snr > best {
                best = snr;
            }
        }
        if best.is_finite() {
            sum += best;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

#[test]
fn hp_filter_helps_on_humming_input_20ms() {
    let n_frames = 50;
    let clean = gen_voiced_clean(n_frames * 160);
    // Reference for SNR is the *clean* signal (no DC/hum) clamped to i16.
    let reference: Vec<i16> = clean
        .iter()
        .map(|&v| v.round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    let dirty = add_hum_and_dc(&clean, 1500.0, 50.0, 800.0);

    let dec_off = round_trip(FrameMode::Ms20, false, &dirty);
    let dec_on = round_trip(FrameMode::Ms20, true, &dirty);

    let snr_off = per_frame_best_snr_avg(&reference, &dec_off, 160, 5);
    let snr_on = per_frame_best_snr_avg(&reference, &dec_on, 160, 5);
    println!(
        "hp_off: {:.2} dB; hp_on: {:.2} dB (delta = {:.2} dB)",
        snr_off,
        snr_on,
        snr_on - snr_off
    );

    // Sanity floor — we shouldn't regress catastrophically.
    assert!(snr_on > 2.0, "HP-on SNR collapsed: {snr_on:.2} dB");
    // The point of the filter: it should win by *some* margin.
    assert!(
        snr_on > snr_off + 0.5,
        "HP-on should outperform HP-off by > 0.5 dB; got {snr_on:.2} vs {snr_off:.2}"
    );
}

#[test]
fn hp_filter_helps_on_humming_input_30ms() {
    let n_frames = 40;
    let clean = gen_voiced_clean(n_frames * 240);
    let reference: Vec<i16> = clean
        .iter()
        .map(|&v| v.round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    let dirty = add_hum_and_dc(&clean, 1500.0, 50.0, 800.0);

    let dec_off = round_trip(FrameMode::Ms30, false, &dirty);
    let dec_on = round_trip(FrameMode::Ms30, true, &dirty);

    let snr_off = per_frame_best_snr_avg(&reference, &dec_off, 240, 4);
    let snr_on = per_frame_best_snr_avg(&reference, &dec_on, 240, 4);
    println!(
        "hp_off: {:.2} dB; hp_on: {:.2} dB (delta = {:.2} dB)",
        snr_off,
        snr_on,
        snr_on - snr_off
    );

    assert!(snr_on > 2.0, "HP-on SNR collapsed: {snr_on:.2} dB");
    assert!(
        snr_on > snr_off + 0.3,
        "HP-on should outperform HP-off by > 0.3 dB; got {snr_on:.2} vs {snr_off:.2}"
    );
}
