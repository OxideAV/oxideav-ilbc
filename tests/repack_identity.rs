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

use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, CodecRegistry, Encoder, Frame, SampleFormat,
};
use oxideav_ilbc::bitreader::{parse_frame, FrameParams};
use oxideav_ilbc::bitwriter::{pack_frame, PackParams};
use oxideav_ilbc::{storage, FrameMode, CODEC_ID_STR, SAMPLE_RATE};

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

/// Encode `pcm` in `mode` and return the produced packet payloads.
fn encode(mode: FrameMode, pcm: &[i16]) -> Vec<Vec<u8>> {
    let mut reg = CodecRegistry::new();
    oxideav_ilbc::register_codecs(&mut reg);
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    if mode == FrameMode::Ms30 {
        params.options = CodecOptions::new().set("frame_ms", "30");
    }
    let mut enc: Box<dyn Encoder> = reg.first_encoder(&params).expect("encoder");

    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples: pcm.len() as u32,
        pts: Some(0),
        data: vec![bytes],
    }))
    .unwrap();
    enc.flush().unwrap();

    let mut out = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        out.push(pkt.data.clone());
    }
    out
}

/// The encoder must emit canonical §3.8 ULP: every produced packet is
/// the mode's fixed size, carries a clear empty-frame indicator (a real
/// encoded frame is never a lost marker), and survives a parse->repack
/// cycle byte-for-bit. This closes the loop from the packer side —
/// bitstream_trace.rs pins the unpacker, the fixture repack test pins
/// the packer on captured data, and this pins the packer on freshly
/// encoded data.
#[test]
fn encoder_output_is_canonical_and_repacks_identically() {
    // Deterministic voiced-ish sweep so the encoder exercises non-trivial
    // start-state / codebook / gain index choices (silence would coast on
    // near-zero indices).
    let voiced: Vec<i16> = (0..40 * 160)
        .map(|n| {
            let t = n as f32 / SAMPLE_RATE as f32;
            let mut v = 0.0f32;
            for h in 1..5 {
                v +=
                    (2.0 * std::f32::consts::PI * (110 * h) as f32 * t).sin() * (4000.0 / h as f32);
            }
            v.clamp(-32768.0, 32767.0) as i16
        })
        .collect();

    for mode in [FrameMode::Ms20, FrameMode::Ms30] {
        let sz = mode.bytes();
        let packets = encode(mode, &voiced);
        assert!(!packets.is_empty(), "{mode:?}: encoder produced no packets");
        let mut checked = 0;
        for (i, pkt) in packets.iter().enumerate() {
            assert_eq!(
                pkt.len(),
                sz,
                "{mode:?} packet #{i}: length {} != frame size {sz}",
                pkt.len()
            );
            let fp =
                parse_frame(pkt).unwrap_or_else(|e| panic!("{mode:?} packet #{i}: parse: {e:?}"));
            assert_eq!(fp.mode, mode, "{mode:?} packet #{i}: mode");
            assert!(
                !fp.empty_flag,
                "{mode:?} packet #{i}: encoder set the empty-frame indicator on a real frame"
            );
            let repacked = pack_frame(&pack_params_from(&fp))
                .unwrap_or_else(|e| panic!("{mode:?} packet #{i}: pack: {e:?}"));
            assert_eq!(
                &repacked, pkt,
                "{mode:?} packet #{i}: encoder output is not parse->pack stable"
            );
            checked += 1;
        }
        eprintln!("encoder-canonical-ok {mode:?}: {checked} packets");
    }
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
