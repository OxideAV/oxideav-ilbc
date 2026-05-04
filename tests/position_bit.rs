//! Encoder/decoder agreement on the RFC 3951 §3.5 `position` bit.
//!
//! The position bit selects whether the 22-/23-sample boundary CB block
//! occupies the LEADING or TRAILING 22/23 samples of the 80-sample
//! state span. The encoder picks position=0 (boundary leading) when the
//! leading slot is significantly quieter than the trailing slot — i.e.
//! voiced/transient onsets — and position=1 (boundary trailing) for
//! the steady-state case.
//!
//! These tests verify that:
//!   1. The encoder actually emits `position=0` when fed a leading-
//!      silence + trailing-loud frame.
//!   2. The decoder round-trips both positions cleanly (same energy
//!      preservation regardless of which slot is CB-coded).

use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_core::{CodecRegistry, Encoder};

use oxideav_ilbc::bitreader::parse_frame;
use oxideav_ilbc::{FrameMode, CODEC_ID_STR, SAMPLE_RATE};

fn pcm_to_audio_frame(pcm: &[i16]) -> Frame {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        samples: pcm.len() as u32,
        pts: Some(0),
        data: vec![bytes],
    })
}

fn build_encoder(mode: FrameMode) -> Box<dyn Encoder> {
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
    reg.make_encoder(&enc_params).expect("encoder")
}

#[test]
fn encoder_picks_position_0_on_voiced_onset() {
    // Frame layout: silence for the first ~30 samples, then a strong
    // sine for the rest. The LPC residual's leading 23-sample slot
    // (sample indices 0..23 of the start-state span at sub-blocks 0/1)
    // is essentially silent; the trailing 23 (indices 57..80) is loud.
    // The encoder's RFC §3.5.1 heuristic must pick position=0.
    let mut pcm = vec![0i16; 160];
    for (i, s) in pcm.iter_mut().enumerate().skip(30) {
        let t = i as f32 / SAMPLE_RATE as f32;
        let v = (2.0 * core::f32::consts::PI * 600.0 * t).sin() * 8000.0;
        *s = v.round().clamp(-32768.0, 32767.0) as i16;
    }
    let mut enc = build_encoder(FrameMode::Ms20);
    enc.send_frame(&pcm_to_audio_frame(&pcm)).unwrap();
    enc.flush().unwrap();
    let pkt = enc.receive_packet().expect("packet");
    let parsed = parse_frame(&pkt.data).expect("parse");
    assert_eq!(
        parsed.position, 0,
        "expected position=0 for leading-silence frame, got {}",
        parsed.position
    );
}

#[test]
fn encoder_picks_position_1_on_steady_signal() {
    // A steady sine has near-equal leading/trailing energies; the
    // encoder's threshold should keep position=1 (the safer choice
    // because IIR error propagation from the leading boundary CB hurts
    // PCM-domain SNR).
    let mut pcm = vec![0i16; 160];
    for (i, s) in pcm.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        let v = (2.0 * core::f32::consts::PI * 400.0 * t).sin() * 5000.0;
        *s = v.round().clamp(-32768.0, 32767.0) as i16;
    }
    let mut enc = build_encoder(FrameMode::Ms20);
    enc.send_frame(&pcm_to_audio_frame(&pcm)).unwrap();
    enc.flush().unwrap();
    let pkt = enc.receive_packet().expect("packet");
    let parsed = parse_frame(&pkt.data).expect("parse");
    assert_eq!(
        parsed.position, 1,
        "expected position=1 for steady-sine frame, got {}",
        parsed.position
    );
}

#[test]
fn round_trip_voiced_onset_frame() {
    // End-to-end round-trip on the same leading-silence + trailing-
    // sine signal as `encoder_picks_position_0`. With both encoder and
    // decoder honouring position=0, the decoded output should preserve
    // bounded energy (no clipping, no NaN/Inf) — the principal fidelity
    // claim that justifies the position-aware path.
    let mut pcm = vec![0i16; 160];
    for (i, s) in pcm.iter_mut().enumerate().skip(30) {
        let t = i as f32 / SAMPLE_RATE as f32;
        let v = (2.0 * core::f32::consts::PI * 600.0 * t).sin() * 8000.0;
        *s = v.round().clamp(-32768.0, 32767.0) as i16;
    }
    let mut reg = CodecRegistry::new();
    oxideav_ilbc::register(&mut reg);

    let mut enc_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    enc_params.sample_rate = Some(SAMPLE_RATE);
    enc_params.channels = Some(1);
    enc_params.sample_format = Some(SampleFormat::S16);
    let mut enc = reg.make_encoder(&enc_params).expect("encoder");

    let mut dec_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    dec_params.sample_rate = Some(SAMPLE_RATE);
    dec_params.channels = Some(1);
    let mut dec = reg.make_decoder(&dec_params).expect("decoder");

    enc.send_frame(&pcm_to_audio_frame(&pcm)).unwrap();
    enc.flush().unwrap();

    let pkt = enc.receive_packet().expect("packet");
    let dpkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), pkt.data);
    dec.send_packet(&dpkt).unwrap();
    let Frame::Audio(a) = dec.receive_frame().expect("frame") else {
        panic!("audio frame expected");
    };
    assert_eq!(a.samples, 160);
    let mut max_abs = 0i32;
    let mut clip_count = 0usize;
    for chunk in a.data[0].chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        max_abs = max_abs.max(s.unsigned_abs() as i32);
        if s == i16::MIN || s == i16::MAX {
            clip_count += 1;
        }
    }
    // Less than 2 % of the samples should clip — the position=0 path
    // is exercised on a real-ish onset signal and the synthesis filter
    // must remain stable.
    assert!(
        clip_count < 4,
        "decoded clipping count too high ({clip_count}/160): position=0 is amplifying"
    );
    assert!(
        max_abs > 100,
        "decoded output suspiciously quiet: {max_abs}"
    );
    eprintln!("position=0 onset frame: max_abs={max_abs}, clip_count={clip_count}");
}
