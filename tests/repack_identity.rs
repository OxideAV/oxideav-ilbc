//! ULP pack/unpack inverse conformance on real fixture bitstreams.
//!
//! `tests/bitstream_trace.rs` pins the *unpacker* against a static
//! per-frame index trace. This test pins the complementary property: the
//! §3.8 ULP *packer* ([`oxideav_ilbc::bitwriter::pack_frame`]) is the
//! exact inverse of the unpacker ([`oxideav_ilbc::bitreader::parse_frame`])
//! on genuine `.lbc` payloads. Parsing a real frame and re-packing the
//! recovered indices must reproduce the original 38/50 bytes bit-for-bit —
//! if any wire bit were dropped, reordered, or defaulted on either side,
//! the round-trip would diverge.
//!
//! Every payload byte is covered: RFC 3951 §3.8 packs class-1 + class-2 +
//! class-3 fields plus the empty-frame indicator with no reserved/padding
//! bits, so byte-identity is the correct expectation (not merely
//! index-equality).

use std::fs;
use std::path::{Path, PathBuf};

use oxideav_ilbc::bitreader::{parse_frame, FrameParams};
use oxideav_ilbc::bitwriter::{pack_frame, PackParams};
use oxideav_ilbc::{storage, FrameMode};

fn fixtures_root() -> PathBuf {
    PathBuf::from("../../docs/audio/ilbc/fixtures")
}

/// Rebuild the packer's input view from a parsed frame. `PackParams` and
/// `FrameParams` carry the same wire fields; this is a straight copy.
fn pack_params_from(fp: &FrameParams) -> PackParams {
    PackParams {
        mode: fp.mode,
        lsf_idx: fp.lsf_idx.clone(),
        block_class: fp.block_class,
        position: fp.position,
        scale_idx: fp.scale_idx,
        state_samples: fp.state_samples.clone(),
        boundary: fp.boundary,
        sub_blocks: fp.sub_blocks.clone(),
        empty_flag: fp.empty_flag,
    }
}

/// Parse every frame in `body` (chunked at `mode`'s frame size), re-pack
/// it, and assert byte-identity. Returns the number of frames checked.
fn check_body(label: &str, mode: FrameMode, body: &[u8]) -> usize {
    let sz = mode.bytes();
    assert_eq!(
        body.len() % sz,
        0,
        "{label}: body {} not a whole number of {sz}-byte frames",
        body.len()
    );
    let mut n = 0;
    for (i, frame) in body.chunks_exact(sz).enumerate() {
        let fp = parse_frame(frame)
            .unwrap_or_else(|e| panic!("{label} frame #{i}: parse failed: {e:?}"));
        assert_eq!(
            fp.mode, mode,
            "{label} frame #{i}: mode disagrees with magic"
        );
        let repacked = pack_frame(&pack_params_from(&fp))
            .unwrap_or_else(|e| panic!("{label} frame #{i}: pack failed: {e:?}"));
        assert_eq!(
            repacked, frame,
            "{label} frame #{i}: parse->pack is not byte-identical"
        );
        n += 1;
    }
    n
}

/// Drive one storage-format `.lbc` file (single pinned mode).
fn check_lbc_file(dir: &Path, fname: &str) -> usize {
    let path = dir.join(fname);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: {e}", path.display());
            return 0;
        }
    };
    let sf = storage::parse(&bytes)
        .unwrap_or_else(|e| panic!("{}: storage parse: {e:?}", path.display()));
    check_body(fname, sf.mode, sf.body())
}

#[test]
fn repack_is_byte_identical_on_single_mode_fixtures() {
    let root = fixtures_root();
    // Every fixture directory that ships an `input.lbc` storage file.
    let cases = [
        "mode-20ms-mono-1s-sine",
        "mode-20ms-mono-1s-noise",
        "mode-20ms-silence",
        "mode-20ms-voice-like",
        "mode-20ms-dtmf-tones",
        "mode-20ms-step-impulse",
        "mode-30ms-mono-1s-sine",
        "mode-30ms-mono-1s-noise",
        "mode-30ms-silence",
        "mode-30ms-voice-like",
    ];
    let mut total = 0;
    let mut dirs_seen = 0;
    for case in cases {
        let dir = root.join(case);
        if !dir.join("input.lbc").exists() {
            eprintln!("skip {case}: no input.lbc");
            continue;
        }
        dirs_seen += 1;
        let n = check_lbc_file(&dir, "input.lbc");
        assert!(n > 0, "{case}: input.lbc yielded no frames");
        total += n;
    }
    if dirs_seen == 0 {
        eprintln!("no fixtures present; skipping");
        return;
    }
    eprintln!("repack-ok: {total} frames across {dirs_seen} single-mode fixtures");
}

#[test]
fn repack_is_byte_identical_across_a_mid_stream_mode_transition() {
    let dir = fixtures_root().join("transition-mid-stream");
    if !dir.exists() {
        eprintln!("skip transition-mid-stream: absent");
        return;
    }
    let mut total = 0;
    for (fname, mode) in [
        ("part_a_20ms.lbc", FrameMode::Ms20),
        ("part_b_30ms.lbc", FrameMode::Ms30),
    ] {
        let bytes = match fs::read(dir.join(fname)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip {fname}: {e}");
                continue;
            }
        };
        let sf = storage::parse(&bytes).expect("transition half parse");
        assert_eq!(sf.mode, mode, "{fname}: magic mode");
        total += check_body(fname, sf.mode, sf.body());
    }
    eprintln!("repack-ok transition: {total} frames across the 20ms->30ms boundary");
}
