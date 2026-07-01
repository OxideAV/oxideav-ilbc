//! Integration tests for the public [`oxideav_ilbc::storage`] module
//! against the real `.lbc` storage-format fixtures under
//! `docs/audio/ilbc/fixtures/`.
//!
//! These prove the storage parser recovers the same `(mode, frames)`
//! split that the per-fixture drivers previously computed with their
//! private helpers, and that a parsed file drives the decoder end to
//! end producing the expected sample count.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
use oxideav_ilbc::{storage, FrameMode, CODEC_ID_STR, SAMPLE_RATE};

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/audio/ilbc/fixtures").join(name)
}

fn read_fixture(name: &str, file: &str) -> Option<Vec<u8>> {
    fs::read(fixture_dir(name).join(file)).ok()
}

/// The `containerless-vs-rtp-style-pair` fixture ships the same
/// bitstream in three carriages; `storage_header.lbc` is the storage
/// form (`#!iLBC20\n` + 25 × 38-byte frames), and `raw_no_header.bin`
/// is the same frames with the magic stripped.
#[test]
fn parses_containerless_storage_header() {
    let Some(lbc) = read_fixture("containerless-vs-rtp-style-pair", "storage_header.lbc") else {
        eprintln!("skipping: fixture corpus absent");
        return;
    };
    let raw = read_fixture("containerless-vs-rtp-style-pair", "raw_no_header.bin")
        .expect("raw_no_header.bin present when storage_header.lbc is");

    let sf = storage::parse(&lbc).expect("storage_header.lbc parses");
    assert_eq!(sf.mode, FrameMode::Ms20);
    assert_eq!(sf.frame_size(), 38);
    // 959 B file = 9 B magic + 25 × 38 B.
    assert_eq!(sf.frame_count(), 25);

    // The magic-stripped body must equal raw_no_header.bin verbatim.
    assert_eq!(sf.body(), raw.as_slice());

    // Each frame is exactly one raw frame.
    let frames: Vec<&[u8]> = sf.frames().collect();
    assert_eq!(frames.len(), 25);
    let rejoined: Vec<u8> = frames.concat();
    assert_eq!(rejoined, raw);
}

/// `detect_mode` on the storage header must agree with the RTP
/// length-based mode hint on the same fixture.
#[test]
fn detect_mode_agrees_across_carriages() {
    let Some(lbc) = read_fixture("containerless-vs-rtp-style-pair", "storage_header.lbc") else {
        return;
    };
    assert_eq!(storage::detect_mode(&lbc), Some(FrameMode::Ms20));
}

/// Round-trip: strip the magic, re-wrap it, and confirm byte-identity
/// with the original storage file.
#[test]
fn wrap_body_reproduces_storage_file() {
    let Some(lbc) = read_fixture("containerless-vs-rtp-style-pair", "storage_header.lbc") else {
        return;
    };
    let sf = storage::parse(&lbc).unwrap();
    let rebuilt = storage::wrap_body(sf.mode, sf.body()).unwrap();
    assert_eq!(rebuilt, lbc);

    // And `write` from the frame slices reproduces it too.
    let frames: Vec<&[u8]> = sf.frames().collect();
    let rebuilt2 = storage::write(sf.mode, &frames).unwrap();
    assert_eq!(rebuilt2, lbc);
}

fn decode_all(frames: &[&[u8]]) -> usize {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec = oxideav_ilbc::decoder::make_decoder(&params).expect("make decoder");

    let tb = TimeBase::new(1, SAMPLE_RATE as i64);
    let mut total = 0usize;
    for (i, frame) in frames.iter().enumerate() {
        let pkt = Packet::new(0, tb, frame.to_vec()).with_pts(i as i64);
        dec.send_packet(&pkt).expect("send_packet");
        if let Ok(Frame::Audio(a)) = dec.receive_frame() {
            // Interleaved S16 mono: 2 bytes/sample on plane 0.
            total += a.data[0].len() / 2;
        }
    }
    total
}

/// A parsed storage file drives the decoder to the expected total
/// sample count: 25 frames × 160 samples = 4000 samples for the 20 ms
/// containerless fixture.
#[test]
fn parsed_storage_file_drives_decoder() {
    let Some(lbc) = read_fixture("containerless-vs-rtp-style-pair", "storage_header.lbc") else {
        return;
    };
    let sf = storage::parse(&lbc).unwrap();
    let frames: Vec<&[u8]> = sf.frames().collect();
    let samples = decode_all(&frames);
    assert_eq!(samples, sf.frame_count() * sf.mode.samples());
    assert_eq!(samples, 25 * 160);
}

/// The transition fixture ships a 20 ms half and a 30 ms half as two
/// separate storage files; each parses to its own pinned mode.
#[test]
fn transition_halves_parse_independently() {
    let Some(a) = read_fixture("transition-mid-stream", "part_a_20ms.lbc") else {
        return;
    };
    let b = read_fixture("transition-mid-stream", "part_b_30ms.lbc")
        .expect("part_b present when part_a is");

    let sa = storage::parse(&a).expect("part_a parses");
    let sb = storage::parse(&b).expect("part_b parses");
    assert_eq!(sa.mode, FrameMode::Ms20);
    assert_eq!(sb.mode, FrameMode::Ms30);
    assert_eq!(sa.frame_count(), 20);
    assert_eq!(sb.frame_count(), 20);
}
