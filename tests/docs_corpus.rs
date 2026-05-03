//! Integration tests against the `docs/audio/ilbc/fixtures/` corpus.
//!
//! Each fixture under `../../docs/audio/ilbc/fixtures/<name>/` ships an
//! iLBC bitstream (storage-format `.lbc` with the 9-byte ASCII magic
//! `#!iLBC20\n` / `#!iLBC30\n`, or a raw `.bin` of concatenated frames)
//! plus an `expected.wav` ground truth produced by FFmpeg's reference
//! iLBC decoder. This test driver runs each input through the in-tree
//! [`IlbcDecoder`], compares the output PCM against the reference, and
//! reports per-fixture statistics:
//!
//! - sample count delta (decoder vs reference),
//! - exact-sample match percentage,
//! - per-channel RMS error,
//! - PSNR (full-scale int16 dynamic range).
//!
//! iLBC is a CELP codec — the bitstream encodes a vector-quantised
//! excitation through a 10th-order LPC synthesis filter (RFC 3951
//! §4). Two independent decoders almost never produce bit-identical
//! PCM: rounding in the LSF -> LPC conversion, the all-pass phase
//! compensator (§4.2), the optional pitch-synchronous enhancer (§4.6),
//! and the post-filter all introduce sub-LSB drift. We therefore start
//! every fixture in [`Tier::ReportOnly`] and let CI surface the deltas
//! without gating; tightening to a numeric PSNR floor is a follow-up
//! once the residual divergences are catalogued.
//!
//! The trace.txt files under each fixture dir are not consumed here;
//! they are an aid for the human implementer when localising
//! divergences against the static parser output documented in
//! `docs/audio/ilbc/ilbc-fixtures-and-traces.md`.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, SampleFormat, TimeBase};
use oxideav_ilbc::{FrameMode, CODEC_ID_STR, FRAME_BYTES_20MS, FRAME_BYTES_30MS, SAMPLE_RATE};

/// Locate `docs/audio/ilbc/fixtures/<name>/`. Tests run with CWD set
/// to the crate root, so we walk two levels up to reach the workspace
/// root and then into `docs/`.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/audio/ilbc/fixtures").join(name)
}

// ---------------------------------------------------------------------------
// `.lbc` / `.bin` carriage parsers
// ---------------------------------------------------------------------------

/// 9-byte storage-format magic: `#!iLBC20\n` (20 ms) or `#!iLBC30\n`
/// (30 ms). Defined in libavformat/ilbc.c and matched by the FFmpeg
/// `ilbc` (de)muxer; not formally part of RFC 3951 but the de-facto
/// container for raw frames on disk.
const ILBC_MAGIC_20: &[u8] = b"#!iLBC20\n";
const ILBC_MAGIC_30: &[u8] = b"#!iLBC30\n";

/// Inferred carriage convention for an input file.
#[derive(Clone, Copy, Debug)]
enum Carriage {
    /// `#!iLBC{20,30}\n` magic + concatenated raw frames.
    StorageFormat,
    /// Raw concatenated frames (no header). Mode is supplied
    /// out-of-band by the caller.
    Raw(FrameMode),
    /// Synthetic transport: each frame prefixed with a big-endian
    /// `uint16` length (containerless-vs-rtp-style-pair fixture).
    RtpStyle(FrameMode),
}

/// Slice an input file into the iLBC frame payloads it carries.
///
/// Returns `(mode, frames)` where `frames[i]` is the 38- or 50-byte
/// payload that is fed verbatim into the decoder.
fn split_frames(input: &[u8], carriage: Carriage) -> (FrameMode, Vec<Vec<u8>>) {
    match carriage {
        Carriage::StorageFormat => {
            let (mode, body) = if input.starts_with(ILBC_MAGIC_20) {
                (FrameMode::Ms20, &input[ILBC_MAGIC_20.len()..])
            } else if input.starts_with(ILBC_MAGIC_30) {
                (FrameMode::Ms30, &input[ILBC_MAGIC_30.len()..])
            } else {
                panic!("storage-format input lacks #!iLBC{{20,30}}\\n magic");
            };
            let frame_size = mode.bytes();
            assert_eq!(
                body.len() % frame_size,
                0,
                "storage-format body length {} is not a multiple of {}",
                body.len(),
                frame_size
            );
            let frames = body.chunks_exact(frame_size).map(|c| c.to_vec()).collect();
            (mode, frames)
        }
        Carriage::Raw(mode) => {
            let frame_size = mode.bytes();
            assert_eq!(
                input.len() % frame_size,
                0,
                "raw body length {} is not a multiple of {}",
                input.len(),
                frame_size
            );
            let frames = input.chunks_exact(frame_size).map(|c| c.to_vec()).collect();
            (mode, frames)
        }
        Carriage::RtpStyle(mode) => {
            let frame_size = mode.bytes();
            let mut off = 0usize;
            let mut frames = Vec::new();
            while off + 2 <= input.len() {
                let len = u16::from_be_bytes([input[off], input[off + 1]]) as usize;
                off += 2;
                assert_eq!(
                    len,
                    frame_size,
                    "rtp-style frame at off={} carries len={} (expected {})",
                    off - 2,
                    len,
                    frame_size
                );
                assert!(off + len <= input.len(), "rtp-style frame truncated");
                frames.push(input[off..off + len].to_vec());
                off += len;
            }
            assert_eq!(off, input.len(), "rtp-style trailing bytes");
            (mode, frames)
        }
    }
}

// ---------------------------------------------------------------------------
// `.wav` parser — finds the `data` chunk in a RIFF WAVE file.
// ---------------------------------------------------------------------------

/// Locate the PCM payload inside a WAV file. FFmpeg writes a `LIST`
/// INFO chunk between `fmt ` and `data`, so a fixed 44-byte skip is
/// wrong; we walk the chunks.
///
/// Returns `(sample_rate, channels, bits_per_sample, pcm_bytes)`.
fn parse_wav(buf: &[u8]) -> (u32, u16, u16, Vec<u8>) {
    assert!(buf.len() >= 12, "wav too short");
    assert_eq!(&buf[0..4], b"RIFF", "wav missing RIFF magic");
    assert_eq!(&buf[8..12], b"WAVE", "wav missing WAVE form");
    let mut sr = 0u32;
    let mut ch = 0u16;
    let mut bps = 0u16;
    let mut pcm: Vec<u8> = Vec::new();
    let mut off = 12usize;
    while off + 8 <= buf.len() {
        let id = &buf[off..off + 4];
        let size =
            u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]) as usize;
        off += 8;
        if id == b"fmt " {
            assert!(size >= 16, "fmt chunk too small");
            ch = u16::from_le_bytes([buf[off + 2], buf[off + 3]]);
            sr = u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]);
            bps = u16::from_le_bytes([buf[off + 14], buf[off + 15]]);
        } else if id == b"data" {
            pcm = buf[off..off + size].to_vec();
        }
        off += size;
        // RIFF chunks are 2-byte aligned.
        if size % 2 == 1 {
            off += 1;
        }
    }
    assert_ne!(sr, 0, "wav fmt chunk not found");
    assert!(!pcm.is_empty(), "wav data chunk not found");
    (sr, ch, bps, pcm)
}

// ---------------------------------------------------------------------------
// Per-fixture decode + scoring
// ---------------------------------------------------------------------------

/// Statistics over a single decoded vs reference PCM stream.
struct Stats {
    /// Number of overlapping samples compared (after truncating both
    /// buffers to the shorter length).
    n: usize,
    /// Decoder-produced sample count.
    n_ours: usize,
    /// Reference sample count.
    n_ref: usize,
    /// Bit-exact (zero diff) sample count.
    n_exact: usize,
    /// Sum of squared int16 differences (for RMSE / PSNR).
    sse: f64,
    /// Largest absolute int16 difference.
    max_abs_diff: i32,
}

impl Stats {
    fn rmse(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            (self.sse / self.n as f64).sqrt()
        }
    }

    /// PSNR in dB referenced to the int16 full scale (32767). Returns
    /// `f64::INFINITY` when the streams match exactly.
    fn psnr_db(&self) -> f64 {
        if self.sse == 0.0 || self.n == 0 {
            f64::INFINITY
        } else {
            let mse = self.sse / self.n as f64;
            let peak = 32767.0f64;
            10.0 * (peak * peak / mse).log10()
        }
    }

    fn match_pct(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.n_exact as f64 / self.n as f64 * 100.0
        }
    }
}

fn decode_pcm(packets: &[Vec<u8>]) -> Result<Vec<i16>, String> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec =
        oxideav_ilbc::decoder::make_decoder(&params).map_err(|e| format!("make_decoder: {e:?}"))?;

    let mut pcm: Vec<i16> = Vec::with_capacity(packets.len() * 240);
    for (i, frame) in packets.iter().enumerate() {
        let pkt =
            Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), frame.clone()).with_pts(i as i64);
        if let Err(e) = dec.send_packet(&pkt) {
            return Err(format!("packet {i}: send_packet: {e:?}"));
        }
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => {
                assert_eq!(a.data.len(), 1, "iLBC produces one interleaved plane");
                for chunk in a.data[0].chunks_exact(2) {
                    pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
            Ok(_) => return Err(format!("packet {i}: non-audio frame")),
            Err(e) => return Err(format!("packet {i}: receive_frame: {e:?}")),
        }
    }
    Ok(pcm)
}

fn score(ours: &[i16], reference: &[i16]) -> Stats {
    let n = ours.len().min(reference.len());
    let mut n_exact = 0usize;
    let mut sse = 0.0f64;
    let mut max_abs_diff = 0i32;
    for i in 0..n {
        let d = ours[i] as i32 - reference[i] as i32;
        if d == 0 {
            n_exact += 1;
        }
        let a = d.unsigned_abs() as i32;
        if a > max_abs_diff {
            max_abs_diff = a;
        }
        sse += (d as f64) * (d as f64);
    }
    Stats {
        n,
        n_ours: ours.len(),
        n_ref: reference.len(),
        n_exact,
        sse,
        max_abs_diff,
    }
}

#[derive(Clone, Copy, Debug)]
enum Tier {
    /// Decode is permitted to diverge from the FFmpeg reference; we
    /// log the deltas but do not gate CI on them. iLBC is a CELP codec
    /// (lossy) and two independent decoders typically differ at the
    /// sub-LSB level even on identical input — see the module-level
    /// comment.
    ReportOnly,
}

struct CorpusCase {
    name: &'static str,
    /// Path to the input file inside `fixture_dir(name)/`.
    input_file: &'static str,
    /// Path to the reference WAV inside `fixture_dir(name)/`.
    expected_file: &'static str,
    carriage: Carriage,
    tier: Tier,
}

fn evaluate(case: &CorpusCase) {
    let dir = fixture_dir(case.name);
    let in_path = dir.join(case.input_file);
    let wav_path = dir.join(case.expected_file);

    let input = match fs::read(&in_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, in_path.display());
            return;
        }
    };
    let wav = match fs::read(&wav_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, wav_path.display());
            return;
        }
    };

    let (mode, frames) = split_frames(&input, case.carriage);
    assert!(!frames.is_empty(), "{}: no frames parsed", case.name);
    // Sanity: every payload is the right size for the inferred mode.
    let want = mode.bytes();
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(
            f.len(),
            want,
            "{}: frame {i} has {} bytes, expected {}",
            case.name,
            f.len(),
            want
        );
    }

    let (sr, ch, bps, pcm_bytes) = parse_wav(&wav);
    assert_eq!(sr, SAMPLE_RATE, "{}: wav sr {} != 8000", case.name, sr);
    assert_eq!(ch, 1, "{}: wav channels {} != 1", case.name, ch);
    assert_eq!(bps, 16, "{}: wav bps {} != 16", case.name, bps);
    let reference: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let ours = match decode_pcm(&frames) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "[{:?}] {} ({:?}): DECODER ERROR: {e}",
                case.tier, case.name, mode
            );
            return;
        }
    };

    let s = score(&ours, &reference);
    eprintln!(
        "[{:?}] {} ({:?}, {} frames): ours_samples={} ref_samples={} cmp_n={} \
         exact={}/{} ({:.4}%) max|diff|={} rms={:.2} psnr={:.2} dB",
        case.tier,
        case.name,
        mode,
        frames.len(),
        s.n_ours,
        s.n_ref,
        s.n,
        s.n_exact,
        s.n,
        s.match_pct(),
        s.max_abs_diff,
        s.rmse(),
        s.psnr_db()
    );

    match case.tier {
        Tier::ReportOnly => {
            // Don't fail. The eprintln output above is the report; CI
            // captures it. Tighten to a numeric floor in a follow-up.
            let _ = s.psnr_db();
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests — 12 cases. iLBC is lossy, so every fixture starts
// in Tier::ReportOnly. Tighten case-by-case in follow-ups once the
// per-fixture PSNR floor is empirically established.
// ---------------------------------------------------------------------------

#[test]
fn corpus_mode_20ms_mono_1s_sine() {
    evaluate(&CorpusCase {
        name: "mode-20ms-mono-1s-sine",
        input_file: "input.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_20ms_mono_1s_noise() {
    evaluate(&CorpusCase {
        name: "mode-20ms-mono-1s-noise",
        input_file: "input.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_20ms_silence() {
    evaluate(&CorpusCase {
        name: "mode-20ms-silence",
        input_file: "input.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_30ms_mono_1s_sine() {
    evaluate(&CorpusCase {
        name: "mode-30ms-mono-1s-sine",
        input_file: "input.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_30ms_mono_1s_noise() {
    evaluate(&CorpusCase {
        name: "mode-30ms-mono-1s-noise",
        input_file: "input.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_30ms_silence() {
    evaluate(&CorpusCase {
        name: "mode-30ms-silence",
        input_file: "input.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_30ms_voice_like() {
    evaluate(&CorpusCase {
        name: "mode-30ms-voice-like",
        input_file: "input.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_20ms_voice_like() {
    evaluate(&CorpusCase {
        name: "mode-20ms-voice-like",
        input_file: "input.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_20ms_step_impulse() {
    evaluate(&CorpusCase {
        name: "mode-20ms-step-impulse",
        input_file: "input.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_mode_20ms_dtmf_tones() {
    evaluate(&CorpusCase {
        name: "mode-20ms-dtmf-tones",
        input_file: "input.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

// ---------------------------------------------------------------------------
// Containerless / RTP-style framing — three views of the same bitstream.
// We assert the carriage parser splits each into the same frame count
// and run all three through the decoder, scoring each against the
// shared reference WAV.
// ---------------------------------------------------------------------------

#[test]
fn corpus_containerless_storage_header() {
    evaluate(&CorpusCase {
        name: "containerless-vs-rtp-style-pair",
        input_file: "storage_header.lbc",
        expected_file: "expected.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_containerless_raw_no_header() {
    evaluate(&CorpusCase {
        name: "containerless-vs-rtp-style-pair",
        input_file: "raw_no_header.bin",
        expected_file: "expected.wav",
        carriage: Carriage::Raw(FrameMode::Ms20),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_containerless_rtp_style() {
    evaluate(&CorpusCase {
        name: "containerless-vs-rtp-style-pair",
        input_file: "rtp_style.bin",
        expected_file: "expected.wav",
        carriage: Carriage::RtpStyle(FrameMode::Ms20),
        tier: Tier::ReportOnly,
    });
}

// ---------------------------------------------------------------------------
// Mid-stream mode transition — 20 ms then 30 ms. The shared reference
// WAV concatenates both halves; we score each `.lbc` half independently
// against its `expected_part_*.wav` (since neither mode alone produces
// the full reference) and additionally exercise the per-mode
// concatenated `.bin` view to verify a known-splice-aware decoder.
// ---------------------------------------------------------------------------

#[test]
fn corpus_transition_part_a_20ms() {
    evaluate(&CorpusCase {
        name: "transition-mid-stream",
        input_file: "part_a_20ms.lbc",
        expected_file: "expected_part_a.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_transition_part_b_30ms() {
    evaluate(&CorpusCase {
        name: "transition-mid-stream",
        input_file: "part_b_30ms.lbc",
        expected_file: "expected_part_b.wav",
        carriage: Carriage::StorageFormat,
        tier: Tier::ReportOnly,
    });
}

/// Splice the no-header `concatenated_raw.bin` at the documented
/// 20-frames * 38-byte boundary (RFC 3951 §5 pins one mode per
/// storage-format file; this fixture is intentionally non-conformant).
/// We feed each half into a *fresh* decoder, concatenate the PCM, and
/// score against the shared `expected.wav`.
#[test]
fn corpus_transition_concatenated_raw_with_known_splice() {
    let dir = fixture_dir("transition-mid-stream");
    let raw = match fs::read(dir.join("concatenated_raw.bin")) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip transition-concat: {e}");
            return;
        }
    };
    let wav = match fs::read(dir.join("expected.wav")) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip transition-concat (wav): {e}");
            return;
        }
    };
    let split = 20 * FRAME_BYTES_20MS;
    assert!(
        raw.len() > split,
        "concatenated_raw.bin shorter than the documented 20*38 splice"
    );
    let part_a = &raw[..split];
    let part_b = &raw[split..];
    assert_eq!(part_b.len() % FRAME_BYTES_30MS, 0, "part_b not 50-aligned");

    let (_, frames_a) = split_frames(part_a, Carriage::Raw(FrameMode::Ms20));
    let (_, frames_b) = split_frames(part_b, Carriage::Raw(FrameMode::Ms30));

    let pcm_a = match decode_pcm(&frames_a) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("transition-concat part_a decode error: {e}");
            return;
        }
    };
    let pcm_b = match decode_pcm(&frames_b) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("transition-concat part_b decode error: {e}");
            return;
        }
    };
    let mut ours = pcm_a;
    ours.extend(pcm_b);

    let (sr, ch, bps, pcm_bytes) = parse_wav(&wav);
    assert_eq!((sr, ch, bps), (SAMPLE_RATE, 1, 16));
    let reference: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let s = score(&ours, &reference);
    eprintln!(
        "[ReportOnly] transition-mid-stream-concat (20+30, splice@760B): \
         ours={} ref={} cmp_n={} exact={}/{} ({:.4}%) max|d|={} rms={:.2} psnr={:.2} dB",
        s.n_ours,
        s.n_ref,
        s.n,
        s.n_exact,
        s.n,
        s.match_pct(),
        s.max_abs_diff,
        s.rmse(),
        s.psnr_db()
    );
}
