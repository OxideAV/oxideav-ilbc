//! Integration test for the RFC 3952 RTP depacketiser → iLBC decoder
//! handoff. Exercises the public `rtp::Depacketiser` against frames
//! the public encoder emits, then re-decodes them, end-to-end.
//!
//! The test stitches multiple iLBC frames into a single RTP payload
//! (RFC 3952 §3 explicitly allows aggregation when every frame
//! shares the SDP-pinned mode — RFC 3952 §4.2). Each split-out
//! frame is fed verbatim into the decoder; the test asserts:
//!   1. the depacketiser produces the right frame count for the
//!      payload size,
//!   2. every depacketised frame is the right byte length for the
//!      mode,
//!   3. each frame decodes to the per-mode PCM sample count
//!      (160 / 240) with no panic,
//!   4. the SDP `mode=20|30` path picks the same mode as the
//!      explicit constructor.
//!
//! Nothing in the test reads any external library source or runs a
//! fixture pipeline through one — every iLBC frame the test handles
//! is synthesised in-test by our own encoder.

use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, Error, Frame, Packet, SampleFormat,
    TimeBase,
};
use oxideav_ilbc::rtp::{
    detect_mode_from_payload_len, Depacketiser, Packetiser, TS_DELTA_20MS, TS_DELTA_30MS,
};
use oxideav_ilbc::{
    decoder, encoder, FrameMode, CODEC_ID_STR, FRAME_BYTES_20MS, FRAME_BYTES_30MS,
    FRAME_SAMPLES_20MS, FRAME_SAMPLES_30MS, SAMPLE_RATE,
};

/// Build N successive iLBC frames from a deterministic synthetic
/// signal. Returns the wire bytes.
fn encode_n_frames(mode: FrameMode, n: usize) -> Vec<Vec<u8>> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    if matches!(mode, FrameMode::Ms30) {
        params.options = CodecOptions::new().set("frame_ms", "30");
    }
    let mut enc = encoder::make_encoder(&params).expect("make_encoder");

    let samples_per_frame = mode.samples();
    let total_samples = samples_per_frame * n;
    // Triangle ramp keeps the encoder out of degenerate start-state
    // paths and gives the LPC analyser something stable to chew on
    // without leaking any actual speech material.
    let mut pcm_bytes = Vec::<u8>::with_capacity(total_samples * 2);
    for i in 0..total_samples {
        let phase = (i % 32) as i16;
        let v = (phase - 16) * 256;
        pcm_bytes.extend_from_slice(&v.to_le_bytes());
    }

    enc.send_frame(&Frame::Audio(AudioFrame {
        samples: total_samples as u32,
        pts: Some(0),
        data: vec![pcm_bytes],
    }))
    .expect("send_frame");
    enc.flush().expect("flush");

    let mut out: Vec<Vec<u8>> = Vec::with_capacity(n);
    loop {
        match enc.receive_packet() {
            Ok(p) => out.push(p.data.clone()),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("encode: {e:?}"),
        }
    }
    out.truncate(n);
    assert_eq!(out.len(), n, "encoder produced unexpected frame count");
    for f in &out {
        assert_eq!(f.len(), mode.bytes());
    }
    out
}

/// Decode every iLBC frame in `frames` through a fresh decoder and
/// return the total PCM sample count. Asserts no panic.
fn decode_through(frames: &[Vec<u8>], mode: FrameMode) -> usize {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec = decoder::make_decoder(&params).expect("make_decoder");
    let tb = TimeBase::new(1, SAMPLE_RATE as i64);

    let mut total_samples = 0usize;
    for (i, f) in frames.iter().enumerate() {
        let pkt = Packet::new(0, tb, f.clone()).with_pts(i as i64);
        dec.send_packet(&pkt).expect("send_packet");
        match dec.receive_frame().expect("receive_frame") {
            Frame::Audio(a) => {
                let got = a.samples as usize;
                assert_eq!(
                    got,
                    mode.samples(),
                    "decoded sample count off for mode {mode:?}",
                );
                total_samples += got;
            }
            other => panic!("non-audio frame from iLBC decoder: {other:?}"),
        }
    }
    total_samples
}

#[test]
fn rtp_depacketiser_aggregates_three_20ms_frames_and_decoder_eats_them() {
    let frames = encode_n_frames(FrameMode::Ms20, 3);
    let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();

    let pk = Packetiser::with_max_frames_per_packet(FrameMode::Ms20, 8);
    let body = pk.pack_single(&refs).expect("pack_single");
    assert_eq!(body.len(), 3 * FRAME_BYTES_20MS);

    let dk = Depacketiser::new(FrameMode::Ms20);
    assert_eq!(dk.frame_count(body.len()), Some(3));
    let split = dk.depacketise_owned(&body).expect("depacketise_owned");

    let total = decode_through(&split, FrameMode::Ms20);
    assert_eq!(total, 3 * FRAME_SAMPLES_20MS);
}

#[test]
fn rtp_depacketiser_aggregates_two_30ms_frames_and_decoder_eats_them() {
    let frames = encode_n_frames(FrameMode::Ms30, 2);
    let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();

    let pk = Packetiser::new(FrameMode::Ms30);
    let body = pk.pack_single(&refs).expect("pack_single");
    assert_eq!(body.len(), 2 * FRAME_BYTES_30MS);

    let dk = Depacketiser::new(FrameMode::Ms30);
    let split = dk.depacketise_owned(&body).expect("depacketise_owned");
    assert_eq!(split.len(), 2);

    let total = decode_through(&split, FrameMode::Ms30);
    assert_eq!(total, 2 * FRAME_SAMPLES_30MS);
}

#[test]
fn rtp_pack_series_then_depacketise_then_decode_20ms() {
    // 5 frames at cap=2 → 3 packets sized 2 + 2 + 1. Each packet
    // body is depacketised and fed to a *shared* decoder so per-
    // packet boundaries do not reset decoder state.
    let frames = encode_n_frames(FrameMode::Ms20, 5);
    let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
    let pk = Packetiser::with_max_frames_per_packet(FrameMode::Ms20, 2);
    let series = pk.pack_series(&refs).expect("pack_series");
    assert_eq!(series.len(), 3);
    let mut concat: Vec<Vec<u8>> = Vec::new();
    let dk = Depacketiser::new(FrameMode::Ms20);
    let mut expected_ts = 0u32;
    for (i, (body, ts)) in series.iter().enumerate() {
        assert_eq!(*ts, expected_ts, "ts at packet {i}");
        let split = dk.depacketise_owned(body).expect("depacketise");
        expected_ts = expected_ts.wrapping_add(split.len() as u32 * TS_DELTA_20MS);
        concat.extend(split);
    }
    assert_eq!(concat.len(), 5);
    let total = decode_through(&concat, FrameMode::Ms20);
    assert_eq!(total, 5 * FRAME_SAMPLES_20MS);
}

#[test]
fn rtp_sdp_fmtp_mode_pins_session_and_drives_decoder() {
    // Same content as the 20 ms aggregation test, but the
    // depacketiser is constructed from an SDP fmtp string per
    // RFC 3952 §4.2 instead of by explicit FrameMode.
    let frames = encode_n_frames(FrameMode::Ms20, 2);
    let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
    let pk = Packetiser::new(FrameMode::Ms20);
    let body = pk.pack_single(&refs).expect("pack");
    let dk = Depacketiser::from_sdp_fmtp("mode=20; ptime=20; maxptime=160").expect("sdp parse");
    assert_eq!(dk.mode(), FrameMode::Ms20);
    let split = dk.depacketise_owned(&body).expect("depacketise");
    assert_eq!(split.len(), 2);
    let total = decode_through(&split, FrameMode::Ms20);
    assert_eq!(total, 2 * FRAME_SAMPLES_20MS);
}

#[test]
fn rtp_sdp_fmtp_mode_30_pins_session_and_drives_decoder() {
    let frames = encode_n_frames(FrameMode::Ms30, 2);
    let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
    let pk = Packetiser::new(FrameMode::Ms30);
    let body = pk.pack_single(&refs).expect("pack");
    let dk = Depacketiser::from_sdp_fmtp("MODE=30; ptime=30").expect("sdp parse");
    assert_eq!(dk.mode(), FrameMode::Ms30);
    assert_eq!(dk.ts_delta(), TS_DELTA_30MS);
    let split = dk.depacketise_owned(&body).expect("depacketise");
    assert_eq!(split.len(), 2);
    let total = decode_through(&split, FrameMode::Ms30);
    assert_eq!(total, 2 * FRAME_SAMPLES_30MS);
}

#[test]
fn rtp_length_only_mode_detection_picks_clean_20ms_payloads() {
    let frames = encode_n_frames(FrameMode::Ms20, 1);
    let body = frames[0].clone();
    assert_eq!(
        detect_mode_from_payload_len(body.len()),
        Some(FrameMode::Ms20),
    );
}

#[test]
fn rtp_length_only_mode_detection_picks_clean_30ms_payloads() {
    let frames = encode_n_frames(FrameMode::Ms30, 1);
    let body = frames[0].clone();
    assert_eq!(
        detect_mode_from_payload_len(body.len()),
        Some(FrameMode::Ms30),
    );
}
