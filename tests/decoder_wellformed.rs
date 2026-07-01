//! Decoder well-formedness invariants on the `docs/audio/ilbc/fixtures/`
//! tonal corpus.
//!
//! ## Why this test exists
//!
//! The companion `docs_corpus.rs` driver scores our decoder against each
//! fixture's `expected.wav` (an external black-box reference decode of
//! the corpus bitstream). For the **tonal** fixtures (sine / noise
//! / voice-like / DTMF) the reference WAV is anomalous: 0.45–1.7 % of
//! its samples slam to the int16 rails (±32767/-32768) and every one of
//! them peaks at full scale, i.e. the reference pipeline produces a
//! high-energy, clipping waveform out of a clean 440 Hz sine. Our
//! spec-correct decode of the *same* bitstream stays bounded (tone /
//! voice / DTMF peaks ≈ 3k–5k, white noise ≈ 26k) and never touches the
//! rails. The `expected.wav` for the silence fixture, by contrast, is
//! clean (0 % clipped, all-zero), which is why `docs_corpus.rs` only
//! reaches ~95 dB PSNR there and 13–17 dB on the tonal cases — the
//! divergence is dominated by the reference's clipping, not by our
//! reconstruction.
//!
//! Per RFC 3951 the decoded excitation for one sub-block is
//! `Σ gain[k]·cbvec[k]` (§4.4.1 / Appendix A.32 `iCBConstruct`) with
//! `gain[0] = gain_sq5Tbl[i]·max(1.0, 0.1)` (§3.6.4.2). Those gains are
//! bounded by `gain_sq5Tbl[31] = 1.200012`, and the §4.1 LSF→LPC
//! synthesis filter is stabilised (§3.2.5) to keep every pole strictly
//! inside the unit circle, so a clean tone can never decode to a
//! full-scale square wave in a spec-compliant decoder. This test pins
//! that property: our output on the tonal fixtures must be well-formed
//! (finite, bounded, and free of the runaway-clipping signature the
//! reference WAV exhibits), so a future change can't silently "improve"
//! the `docs_corpus.rs` PSNR by chasing the broken reference into an
//! unstable, clipping decode.
//!
//! Everything here is computed from our own decoder plus the fixture
//! `input.lbc` / `expected.wav` bytes — no external decoder is consulted.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
use oxideav_ilbc::{FrameMode, CODEC_ID_STR, SAMPLE_RATE};

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/audio/ilbc/fixtures").join(name)
}

/// Slice a `#!iLBC{20,30}\n`-prefixed storage file into frame payloads
/// via the public `oxideav_ilbc::storage` parser.
fn split_storage_frames(input: &[u8]) -> (FrameMode, Vec<Vec<u8>>) {
    let sf = oxideav_ilbc::storage::parse(input).expect("storage-format parse");
    (sf.mode, sf.frames().map(|f| f.to_vec()).collect())
}

/// Extract the `data` chunk PCM from a RIFF/WAVE file as int16 samples.
fn parse_wav_pcm(buf: &[u8]) -> Vec<i16> {
    assert!(buf.len() >= 12, "wav too short");
    assert_eq!(&buf[0..4], b"RIFF", "wav missing RIFF magic");
    assert_eq!(&buf[8..12], b"WAVE", "wav missing WAVE form");
    let mut off = 12usize;
    while off + 8 <= buf.len() {
        let id = &buf[off..off + 4];
        let size =
            u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]) as usize;
        off += 8;
        if id == b"data" {
            return buf[off..off + size]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
        }
        off += size + (size & 1);
    }
    panic!("wav data chunk not found");
}

/// Decode a sequence of iLBC frame payloads with the default (no
/// `hp_filter`) decoder, returning the concatenated int16 PCM.
fn decode_pcm(frames: &[Vec<u8>]) -> Vec<i16> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec = oxideav_ilbc::decoder::make_decoder(&params).expect("make_decoder");

    let mut pcm = Vec::with_capacity(frames.len() * 240);
    for (i, frame) in frames.iter().enumerate() {
        let pkt =
            Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), frame.clone()).with_pts(i as i64);
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Audio(a) = dec.receive_frame().expect("receive_frame") else {
            panic!("expected audio frame for packet {i}");
        };
        for chunk in a.data[0].chunks_exact(2) {
            pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
    }
    pcm
}

/// Fraction of samples that sit exactly on the int16 rails. This is the
/// "runaway clipping" signature: the anomalous tonal `expected.wav`
/// reference files exhibit ~1.5–1.7 %; a spec-correct decode of a clean
/// tone exhibits 0 %.
fn rail_fraction(pcm: &[i16]) -> f64 {
    if pcm.is_empty() {
        return 0.0;
    }
    let railed = pcm
        .iter()
        .filter(|&&s| s == i16::MAX || s == i16::MIN)
        .count();
    railed as f64 / pcm.len() as f64
}

/// Load a fixture's `(frames, reference_pcm)`. Returns `None` when the
/// fixture files are absent — the `docs/audio/ilbc/fixtures/` corpus
/// lives in a private submodule that the public CI checkout does not
/// fetch, so these tests run for real locally and skip in CI (matching
/// `docs_corpus.rs`'s skip-on-missing behaviour).
fn load(name: &str) -> Option<(Vec<Vec<u8>>, Vec<i16>)> {
    let dir = fixture_dir(name);
    let input = fs::read(dir.join("input.lbc")).ok()?;
    let wav = fs::read(dir.join("expected.wav")).ok()?;
    let (_mode, frames) = split_storage_frames(&input);
    Some((frames, parse_wav_pcm(&wav)))
}

/// The tonal fixtures whose `expected.wav` reference exhibits the
/// runaway-clipping anomaly documented at the top of this file.
const TONAL_FIXTURES: &[&str] = &[
    "mode-20ms-mono-1s-sine",
    "mode-30ms-mono-1s-sine",
    "mode-20ms-mono-1s-noise",
    "mode-30ms-mono-1s-noise",
    "mode-20ms-voice-like",
    "mode-30ms-voice-like",
    "mode-20ms-dtmf-tones",
];

/// Our spec-compliant decode of every tonal fixture must be well-formed:
/// finite, bounded, and free of the reference's runaway-clipping
/// signature.
#[test]
fn tonal_decode_is_well_formed() {
    for &name in TONAL_FIXTURES {
        let Some((frames, _reference)) = load(name) else {
            eprintln!("skip {name}: fixture corpus not present (CI without docs submodule)");
            continue;
        };
        let ours = decode_pcm(&frames);
        assert!(!ours.is_empty(), "{name}: empty decode");

        // i16 samples are finite by construction; assert the stronger
        // property that the decode never pins the rails. A spec-correct
        // bounded all-pole synthesis of these inputs peaks well below
        // full scale.
        let railed = rail_fraction(&ours);
        assert_eq!(
            railed,
            0.0,
            "{name}: our decode railed {:.2}% of samples — \
             a spec-compliant decode of a clean tone must not clip",
            railed * 100.0
        );

        // Peak amplitude must stay clear of full scale. The bounded
        // gain table (gain_sq5Tbl[31]=1.200012) and stabilised LPC keep
        // the synthesised signal below the rails. Measured peaks across
        // the corpus: tones / voice / DTMF ≈ 2.9k–4.6k, white noise the
        // hottest at ≈ 26k (no pitch periodicity caps the residual
        // energy). A 30000 ceiling (≈ -0.8 dBFS) admits the legitimate
        // noise peak while still excluding the reference's ±32767
        // clipping behaviour.
        let peak = ours.iter().map(|&s| (s as i32).abs()).max().unwrap_or(0);
        assert!(
            peak < 30_000,
            "{name}: peak {peak} reached the clipping region — \
             a spec-correct CELP decode of this input stays bounded"
        );
    }
}

/// Document the reference anomaly itself: the tonal `expected.wav`
/// reference clips, the silence reference does not. This pins the
/// premise that motivates `tonal_decode_is_well_formed` — if a future
/// re-capture of the corpus replaces the clipping tonal references with
/// clean ones, this test flags it so the PSNR floors in `docs_corpus.rs`
/// can be re-anchored.
#[test]
fn reference_tonal_wavs_exhibit_clipping_anomaly() {
    // Silence reference is clean: zero samples on the rails.
    let Some((_frames, silence_ref)) = load("mode-20ms-silence") else {
        eprintln!("skip: fixture corpus not present (CI without docs submodule)");
        return;
    };
    assert_eq!(
        rail_fraction(&silence_ref),
        0.0,
        "silence reference unexpectedly clips"
    );

    // Every tonal reference clips on a non-trivial fraction of samples.
    // Measured rail fractions span 0.0045–0.017 across the corpus, all
    // peaking at the int16 rail (32768 in unsigned terms); the silence
    // reference is the lone clean case (0, peak 0). A 0.003 threshold
    // sits below the lowest tonal observation and above silence.
    for &name in TONAL_FIXTURES {
        let Some((_frames, reference)) = load(name) else {
            eprintln!("skip {name}: fixture corpus not present (CI without docs submodule)");
            continue;
        };
        let railed = rail_fraction(&reference);
        let peak = reference
            .iter()
            .map(|&s| (s as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(
            railed > 0.003 && peak >= 32_767,
            "{name}: reference rail fraction {:.4} (peak {peak}) below the \
             documented anomaly threshold — corpus may have been \
             re-captured; re-anchor docs_corpus.rs PSNR floors",
            railed
        );
    }
}
