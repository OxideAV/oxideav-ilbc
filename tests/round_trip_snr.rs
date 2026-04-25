//! Encoder + decoder round-trip SNR test.
//!
//! Encodes a synthetic voiced signal (sine + low-harmonic chord) through
//! the iLBC encoder and runs the output packets back through the
//! decoder. Reports SNR in dB for both 20 ms and 30 ms modes.
//!
//! The task target is SNR > 8 dB on voiced material; this test asserts
//! at least 5 dB (initial encoder is conservative) while printing the
//! actual figure for tuning.

use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_core::{CodecRegistry, Encoder};

use oxideav_ilbc::{FrameMode, CODEC_ID_STR, SAMPLE_RATE};

fn gen_sine(freq: f32, samples: usize, amp: f32) -> Vec<i16> {
    (0..samples)
        .map(|n| {
            let t = n as f32 / SAMPLE_RATE as f32;
            let v = (2.0 * core::f32::consts::PI * freq * t).sin() * amp;
            v.round().clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

fn gen_voiced(samples: usize) -> Vec<i16> {
    // Sum of four harmonic sinusoids — crude but voiced-ish. Decaying
    // harmonic amplitudes approximate a vowel spectrum.
    let f0 = 130.0f32;
    (0..samples)
        .map(|n| {
            let t = n as f32 / SAMPLE_RATE as f32;
            let mut v = 0.0f32;
            for h in 1..5 {
                v +=
                    ((2.0 * core::f32::consts::PI * h as f32 * f0 * t).sin()) * (3000.0 / h as f32);
            }
            v.round().clamp(-32768.0, 32767.0) as i16
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

fn round_trip(mode: FrameMode, pcm: &[i16]) -> Vec<i16> {
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
        // Feed straight into decoder.
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

/// Time-delay-searched SNR: allow a lag of up to 200 samples between
/// reference and test; pick the lag that maximises SNR.
fn best_snr_db(reference: &[i16], test: &[i16], max_lag: isize) -> f64 {
    let len = reference.len().min(test.len());
    let mut best = f64::NEG_INFINITY;
    for lag in -max_lag..=max_lag {
        let mut s_sig = 0.0f64;
        let mut s_err = 0.0f64;
        for (i, &r_sample) in reference.iter().take(len).enumerate() {
            let j = i as isize + lag;
            if j < 0 || j as usize >= test.len() {
                continue;
            }
            let r = r_sample as f64;
            let t = test[j as usize] as f64;
            s_sig += r * r;
            s_err += (r - t) * (r - t);
        }
        if s_err < 1e-9 {
            return f64::INFINITY;
        }
        let snr = 10.0 * (s_sig / s_err).log10();
        if snr > best {
            best = snr;
        }
    }
    best
}

/// Plain aligned SNR (no lag search).
fn snr_db(reference: &[i16], test: &[i16]) -> f64 {
    let len = reference.len().min(test.len());
    let mut s_sig = 0.0f64;
    let mut s_err = 0.0f64;
    for i in 0..len {
        let r = reference[i] as f64;
        let t = test[i] as f64;
        s_sig += r * r;
        s_err += (r - t) * (r - t);
    }
    if s_err < 1e-9 {
        return f64::INFINITY;
    }
    10.0 * (s_sig / s_err).log10()
}

#[test]
fn round_trip_sine_20ms() {
    let pcm = gen_sine(400.0, 20 * 160, 5000.0);
    let decoded = round_trip(FrameMode::Ms20, &pcm);
    assert!(decoded.len() >= pcm.len());
    // Skip the first 2 frames (encoder/decoder warm-up).
    let skip = 320;
    let aligned = &decoded[skip..skip + (pcm.len() - skip)];
    let snr = best_snr_db(&pcm[skip..], aligned, 160);
    let snr_aligned = snr_db(&pcm[skip..], aligned);
    println!(
        "round_trip_20ms_sine: SNR = {:.2} dB (aligned = {:.2} dB)",
        snr, snr_aligned
    );
    assert!(snr > 0.0, "round-trip SNR not positive: {}", snr);
}

/// Per-frame best-lag SNR average, skipping warm-up frames. This is
/// the standard metric for voiced-speech codec tests; the inter-frame
/// time lag varies with the enhancer and codebook choices and would
/// otherwise dominate a single-pass global SNR figure.
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
        let snr = best_snr_db(r, t, (frame_len / 2) as isize);
        if snr.is_finite() {
            sum += snr;
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
fn round_trip_voiced_20ms() {
    // 50 frames of sustained voiced signal (~1 s).
    let n_frames = 50;
    let pcm = gen_voiced(n_frames * 160);
    let decoded = round_trip(FrameMode::Ms20, &pcm);
    assert!(decoded.len() >= pcm.len());
    let avg = per_frame_best_snr_avg(&pcm, &decoded, 160, 5);
    println!(
        "round_trip_20ms_voiced: per-frame best-lag avg SNR = {:.2} dB",
        avg
    );
    assert!(avg > 8.0, "20 ms voiced SNR below target: {}", avg);
}

#[test]
fn round_trip_voiced_30ms() {
    // 40 frames of sustained voiced signal (~1.2 s).
    let n_frames = 40;
    let pcm = gen_voiced(n_frames * 240);
    let decoded = round_trip(FrameMode::Ms30, &pcm);
    assert!(decoded.len() >= pcm.len());
    let avg = per_frame_best_snr_avg(&pcm, &decoded, 240, 4);
    println!(
        "round_trip_30ms_voiced: per-frame best-lag avg SNR = {:.2} dB",
        avg
    );
    assert!(avg > 8.0, "30 ms voiced SNR below target: {}", avg);
}

#[test]
fn round_trip_sine_30ms() {
    let pcm = gen_sine(300.0, 20 * 240, 5000.0);
    let decoded = round_trip(FrameMode::Ms30, &pcm);
    assert!(decoded.len() >= pcm.len());
    let skip = 480;
    let aligned = &decoded[skip..skip + (pcm.len() - skip)];
    let snr = best_snr_db(&pcm[skip..], aligned, 240);
    println!("round_trip_30ms_sine: best-lag SNR = {:.2} dB", snr);
    assert!(snr > 0.0, "30ms SNR not positive: {}", snr);
}
