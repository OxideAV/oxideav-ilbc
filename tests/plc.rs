//! End-to-end packet-loss-concealment tests against real fixture
//! bitstreams (RFC 3951 §4.5, Appendix A.14 `doThePLC`).
//!
//! Each test decodes a genuine `docs/audio/ilbc/fixtures/*` stream, then
//! re-decodes it with selected frames replaced by an *empty-frame
//! indicator* (RFC 3951 §3.8: last bit set ⇒ "treat as lost, run PLC").
//! The concealed output is exercised through the public registry decoder,
//! so it covers the residual-domain concealer wired through the §4.6
//! enhancer + §4.7 synthesis path.
//!
//! Source: RFC 3951 §4.5 + the `docs/audio/ilbc/fixtures/` corpus only.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Decoder, Frame, Packet, SampleFormat, TimeBase,
};

fn fixture(name: &str) -> Option<Vec<u8>> {
    let p = PathBuf::from("../../docs/audio/ilbc/fixtures")
        .join(name)
        .join("input.lbc");
    fs::read(p).ok()
}

/// Split a storage-format `.lbc` file into (frame_bytes, frames) via
/// the public `oxideav_ilbc::storage` parser.
fn split(input: &[u8]) -> (usize, Vec<Vec<u8>>) {
    let sf = oxideav_ilbc::storage::parse(input).expect("storage-format parse");
    (sf.frame_size(), sf.frames().map(|f| f.to_vec()).collect())
}

fn make_dec() -> Box<dyn Decoder> {
    let mut reg = CodecRegistry::new();
    oxideav_ilbc::register_codecs(&mut reg);
    let mut params = CodecParameters::audio(CodecId::new(oxideav_ilbc::CODEC_ID_STR));
    params.sample_rate = Some(oxideav_ilbc::SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    reg.first_decoder(&params).expect("decoder")
}

/// Decode a list of frame payloads to interleaved i16 PCM.
fn decode(frames: &[Vec<u8>]) -> Vec<Vec<i16>> {
    let mut dec = make_dec();
    let mut out = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let pkt = Packet::new(
            0,
            TimeBase::new(1, oxideav_ilbc::SAMPLE_RATE as i64),
            f.clone(),
        )
        .with_pts(i as i64 * 160);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(a) = dec.receive_frame().unwrap() else {
            panic!("audio frame expected");
        };
        let pcm: Vec<i16> = a.data[0]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        out.push(pcm);
    }
    out
}

/// Replace frame `idx` with an empty-frame-indicator packet of the same
/// size (all zero bytes + the §3.8 trailing "lost" bit set).
fn lose(frames: &mut [Vec<u8>], idx: usize) {
    let fb = frames[idx].len();
    let mut empty = vec![0u8; fb];
    empty[fb - 1] = 1; // §3.8 empty-frame indicator
    frames[idx] = empty;
}

fn energy(pcm: &[i16]) -> f64 {
    pcm.iter().map(|&s| (s as f64) * (s as f64)).sum()
}

/// A concealed voiced block carries real energy (pitch continuation), and
/// a run of losses is energy-dampened per the §A.14 `use_gain` ladder.
#[test]
fn consecutive_losses_attenuate_on_voiced_stream() {
    let Some(raw) = fixture("mode-20ms-voice-like") else {
        eprintln!("fixture absent; skipping");
        return;
    };
    let (_fb, frames) = split(&raw);
    assert!(frames.len() > 20, "need a few frames");

    // Lose a run of consecutive frames in the middle of the stream, after
    // the decoder has built up a voiced excitation history.
    let first_lost = 10usize;
    let run = 6usize;
    let mut lossy = frames.clone();
    for k in 0..run {
        lose(&mut lossy, first_lost + k);
    }
    let pcm = decode(&lossy);

    // Each concealed block is finite + bounded (no runaway from feeding the
    // synthetic excitation back into the §A.14 prevResidual).
    for blk in &pcm[first_lost..first_lost + run] {
        for &s in blk {
            assert!(s.abs() < 32000, "concealed sample near rail: {s}");
        }
    }

    // The §A.14 `use_gain` ladder: with 20 ms blocks, consPLICount*blockl
    // crosses 320 from the 3rd consecutive loss on, so the tail of the
    // loss run must carry less energy than its head.
    let e_head = energy(&pcm[first_lost]);
    let e_tail = energy(&pcm[first_lost + run - 1]);
    assert!(
        e_tail <= e_head,
        "loss-run energy did not dampen: head {e_head} tail {e_tail}"
    );
}

/// A single lost frame in a steady stream is concealed (non-silent) and
/// the stream recovers cleanly on the next received frame — the §4.5.3
/// smooth-merge path via the enhancer must not blow up or zero the output.
#[test]
fn single_loss_conceals_and_recovers() {
    let Some(raw) = fixture("mode-20ms-voice-like") else {
        eprintln!("fixture absent; skipping");
        return;
    };
    let (_fb, frames) = split(&raw);
    let lost = 12usize;
    assert!(frames.len() > lost + 4);

    let clean = decode(&frames);
    let mut lossy = frames.clone();
    lose(&mut lossy, lost);
    let concealed = decode(&lossy);

    // Same frame/sample geometry.
    assert_eq!(clean.len(), concealed.len());

    // The concealed block is bounded.
    for &s in &concealed[lost] {
        assert!(s.abs() < 32000, "concealed sample near rail: {s}");
    }

    // Recovery: a few frames after the loss the concealed-path output must
    // re-converge toward the clean decode (the §4.5.3 merge prevents a
    // permanent divergence). Compare RMS-level energy a few frames later.
    let recover = lost + 3;
    let e_clean = energy(&clean[recover]).max(1.0);
    let e_conc = energy(&concealed[recover]).max(1.0);
    let ratio = e_conc / e_clean;
    assert!(
        (0.25..=4.0).contains(&ratio),
        "post-loss energy diverged: clean {e_clean} concealed {e_conc} ratio {ratio}"
    );
}

/// Losing the very first frame (no recorded residual) must not panic and
/// must emit a bounded, finite block — the §A.14 "no history" guard.
#[test]
fn first_frame_loss_is_safe() {
    let Some(raw) = fixture("mode-30ms-voice-like") else {
        eprintln!("fixture absent; skipping");
        return;
    };
    let (_fb, mut frames) = split(&raw);
    lose(&mut frames, 0);
    let pcm = decode(&frames);
    for &s in &pcm[0] {
        assert!(s.abs() < 32000, "first-loss sample near rail: {s}");
    }
}
