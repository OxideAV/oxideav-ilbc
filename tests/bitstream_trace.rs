//! Bitstream-unpack cross-check against the `docs/audio/ilbc/fixtures/`
//! static `trace.txt` files (RFC 3951 §3.8 ULP).
//!
//! Each fixture ships a `trace.txt` produced by a *static* bitstream
//! analyser: it walks every frame's §3.8 unequal-level-protection (ULP)
//! bit packing and records one line per parameter — the LSF split-VQ
//! indices, the start-state header (`start_subframe` / `state_first` /
//! `scale_factor_idx_ifm`), the first 16 start-state-sample MSBs, the
//! boundary 22/23-sample block, and every adaptive-codebook sub-block's
//! three-stage `cb_idx` / `gain_idx`. The analyser performs **no**
//! numeric dequantisation — it only extracts bit positions, so the
//! trace is decoder-implementation-independent. Two unpackers that agree
//! on every trace line have a bit-exact §3.8 inverse-ULP regardless of
//! any downstream CELP float drift.
//!
//! This is the property the `docs_corpus.rs` PSNR floors cannot pin:
//! PSNR measures the synthesised PCM (subject to LSF→LPC rounding, the
//! §4.2 all-pass phase compensator, the §4.6 enhancer, and the §4.8
//! post-filter), whereas the trace pins the *integer indices* recovered
//! by [`oxideav_ilbc::bitreader::parse_frame`] straight off the wire.
//!
//! The fields map onto [`FrameParams`] (see `src/bitreader.rs`):
//!
//! | trace token                | FrameParams field                  |
//! | -------------------------- | ---------------------------------- |
//! | `start_subframe` / `start` | `block_class`                      |
//! | `state_first`              | `position`                         |
//! | `scale_factor_idx_ifm`     | `scale_idx`                        |
//! | `trailing_bit`             | `empty_flag`                       |
//! | `LSF_DECODE split_vq=[…]`  | `lsf_idx` (flattened)              |
//! | `BLOCK_DECODE block_idx=0` | `boundary`                         |
//! | `BLOCK_DECODE block_idx=k` | `sub_blocks[k-1]` (k ≥ 1)          |
//! | `idx[0:16]_msb=<bits>`     | `state_samples[0..16] >> 2`        |

use std::fs;
use std::path::{Path, PathBuf};

use oxideav_ilbc::bitreader::{parse_frame, FrameParams};
use oxideav_ilbc::FrameMode;

/// `#!iLBC20\n` / `#!iLBC30\n` storage-format magic (de-facto container
/// convention; not part of RFC 3951 proper).
const ILBC_MAGIC_20: &[u8] = b"#!iLBC20\n";
const ILBC_MAGIC_30: &[u8] = b"#!iLBC30\n";

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/audio/ilbc/fixtures").join(name)
}

// ---------------------------------------------------------------------------
// Expected per-frame view parsed out of trace.txt.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct TraceFrame {
    mode: Option<FrameMode>,
    start_subframe: u8,
    state_first: u8,
    scale_idx: u8,
    trailing_bit: u8,
    /// LSF split-VQ indices, flattened in wire order (3 for 20 ms,
    /// 6 for 30 ms).
    lsf: Vec<u16>,
    /// First-16 start-state-sample MSB bitstring (`'0'`/`'1'`), or empty.
    state_msb: String,
    /// `BLOCK_DECODE` rows in `block_idx` order: `(cb_idx[3], gain_idx[3])`.
    blocks: Vec<([u16; 3], [u8; 3])>,
}

/// Parse a `key=val` token list out of a trace line tail.
fn kv<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    for tok in line.split_whitespace() {
        if let Some(rest) = tok.strip_prefix(key) {
            if let Some(v) = rest.strip_prefix('=') {
                return Some(v);
            }
        }
    }
    None
}

/// Parse a `name=[ a b c ]` bracketed integer list. The static analyser
/// space-pads numeric cells, so we tolerate arbitrary internal spacing.
fn parse_bracketed(line: &str, key: &str) -> Vec<i64> {
    // Find `key=[` then read until the matching `]`.
    let needle = format!("{key}=[");
    let Some(start) = line.find(&needle) else {
        return Vec::new();
    };
    let after = &line[start + needle.len()..];
    let Some(end) = after.find(']') else {
        return Vec::new();
    };
    after[..end]
        .split_whitespace()
        .filter_map(|s| s.parse::<i64>().ok())
        .collect()
}

fn parse_trace(text: &str) -> Vec<TraceFrame> {
    let mut frames: Vec<TraceFrame> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("FRAME ") {
            let mode = kv(trimmed, "mode").and_then(|m| match m {
                "20ms" => Some(FrameMode::Ms20),
                "30ms" => Some(FrameMode::Ms30),
                _ => None,
            });
            frames.push(TraceFrame {
                mode,
                start_subframe: kv(trimmed, "start_subframe")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                state_first: kv(trimmed, "state_first")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                scale_idx: kv(trimmed, "scale_factor_idx_ifm")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                trailing_bit: kv(trimmed, "trailing_bit")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                ..Default::default()
            });
            continue;
        }
        let Some(cur) = frames.last_mut() else {
            continue;
        };
        if trimmed.starts_with("LSF_DECODE") {
            cur.lsf = parse_bracketed(trimmed, "split_vq")
                .into_iter()
                .map(|v| v as u16)
                .collect();
        } else if trimmed.starts_with("START_STATE_SAMPLE_PCM") {
            // `idx[0:16]_msb=<bits>` — the key contains brackets/colon, so
            // grab the substring after `_msb=`.
            if let Some(pos) = trimmed.find("_msb=") {
                cur.state_msb = trimmed[pos + "_msb=".len()..]
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_string();
            }
        } else if trimmed.starts_with("BLOCK_DECODE") {
            let cb = parse_bracketed(trimmed, "cb_idx");
            let g = parse_bracketed(trimmed, "gain_idx");
            assert_eq!(cb.len(), 3, "BLOCK_DECODE cb_idx not 3 wide: {trimmed}");
            assert_eq!(g.len(), 3, "BLOCK_DECODE gain_idx not 3 wide: {trimmed}");
            cur.blocks.push((
                [cb[0] as u16, cb[1] as u16, cb[2] as u16],
                [g[0] as u8, g[1] as u8, g[2] as u8],
            ));
        }
        // CB_VECTOR_USE / START_STATE_INFO lines carry no information
        // beyond what BLOCK_DECODE / the FRAME header already pinned.
    }
    frames
}

/// Read the input file named in the trace's `# iLBC trace for <file>`
/// header line and split it into the per-frame payload list, driven by
/// the per-frame mode the trace records.
fn read_input_for_trace(dir: &Path, trace: &str, expected: &[TraceFrame]) -> Vec<Vec<u8>> {
    let header = trace
        .lines()
        .find(|l| l.starts_with("# iLBC trace for "))
        .expect("trace missing header line");
    let fname = header.trim_start_matches("# iLBC trace for ").trim();
    let bytes = fs::read(dir.join(fname)).expect("read input file named in trace");

    // Strip the storage-format magic if present.
    let body: &[u8] = if bytes.starts_with(ILBC_MAGIC_20) {
        &bytes[ILBC_MAGIC_20.len()..]
    } else if bytes.starts_with(ILBC_MAGIC_30) {
        &bytes[ILBC_MAGIC_30.len()..]
    } else {
        &bytes
    };

    // Walk the body using each frame's mode from the trace (the
    // transition fixture switches mode mid-stream).
    let mut out = Vec::with_capacity(expected.len());
    let mut off = 0usize;
    for tf in expected {
        let mode = tf.mode.expect("trace FRAME line missing a parseable mode");
        let sz = mode.bytes();
        assert!(
            off + sz <= body.len(),
            "input body exhausted at frame off={off} (need {sz}, have {})",
            body.len() - off
        );
        out.push(body[off..off + sz].to_vec());
        off += sz;
    }
    assert_eq!(
        off,
        body.len(),
        "input body has trailing bytes after {} frames",
        expected.len()
    );
    out
}

/// Assert one parsed `FrameParams` matches its trace expectation.
fn assert_frame_matches(fixture: &str, idx: usize, tf: &TraceFrame, fp: &FrameParams) {
    let ctx = || format!("{fixture} frame #{idx}");

    assert_eq!(
        Some(fp.mode),
        tf.mode,
        "{}: mode mismatch (parsed {:?})",
        ctx(),
        fp.mode
    );
    assert_eq!(
        fp.block_class,
        tf.start_subframe,
        "{}: block_class/start_subframe",
        ctx()
    );
    assert_eq!(
        fp.position,
        tf.state_first,
        "{}: position/state_first",
        ctx()
    );
    assert_eq!(fp.scale_idx, tf.scale_idx, "{}: scale_idx/ifm", ctx());
    assert_eq!(
        u8::from(fp.empty_flag),
        tf.trailing_bit,
        "{}: empty_flag/trailing_bit",
        ctx()
    );

    // LSF split-VQ — flatten FrameParams' per-set [u16; 3] view to the
    // trace's flat ordering.
    let lsf_flat: Vec<u16> = fp.lsf_idx.iter().flatten().copied().collect();
    assert_eq!(lsf_flat, tf.lsf, "{}: LSF split-VQ indices", ctx());

    // Boundary block (block_idx=0) + sub-blocks (block_idx=1..N) — the
    // trace lists them in that exact wire order.
    assert_eq!(
        tf.blocks.len(),
        1 + fp.sub_blocks.len(),
        "{}: BLOCK_DECODE count (trace {} vs parsed boundary+{} subs)",
        ctx(),
        tf.blocks.len(),
        fp.sub_blocks.len()
    );
    let (ref_cb0, ref_g0) = tf.blocks[0];
    assert_eq!(fp.boundary.cb_idx, ref_cb0, "{}: boundary cb_idx", ctx());
    assert_eq!(fp.boundary.gain_idx, ref_g0, "{}: boundary gain_idx", ctx());
    for (k, sb) in fp.sub_blocks.iter().enumerate() {
        let (ref_cb, ref_g) = tf.blocks[k + 1];
        assert_eq!(sb.cb_idx, ref_cb, "{}: sub_block[{k}] cb_idx", ctx());
        assert_eq!(sb.gain_idx, ref_g, "{}: sub_block[{k}] gain_idx", ctx());
    }

    // Start-state sample MSBs — the trace prints the first 16 sample
    // MSBs as a bitstring; each parsed 3-bit sample's MSB is `s >> 2`.
    if !tf.state_msb.is_empty() {
        let n = tf.state_msb.len().min(fp.state_samples.len());
        let got: String = fp.state_samples[..n]
            .iter()
            .map(|s| if (s >> 2) & 1 == 1 { '1' } else { '0' })
            .collect();
        assert_eq!(
            got,
            tf.state_msb[..n],
            "{}: start-state MSB fingerprint",
            ctx()
        );
    }
}

/// Drive a single fixture's trace cross-check end to end.
fn check_fixture(name: &str) {
    let dir = fixture_dir(name);
    let trace_path = dir.join("trace.txt");
    let trace = match fs::read_to_string(&trace_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("skip {name}: missing {} ({e})", trace_path.display());
            return;
        }
    };
    let expected = parse_trace(&trace);
    assert!(!expected.is_empty(), "{name}: no FRAME lines in trace");

    let frames = read_input_for_trace(&dir, &trace, &expected);
    assert_eq!(
        frames.len(),
        expected.len(),
        "{name}: input frame count {} != trace frame count {}",
        frames.len(),
        expected.len()
    );

    for (idx, (payload, tf)) in frames.iter().zip(expected.iter()).enumerate() {
        let fp =
            parse_frame(payload).unwrap_or_else(|e| panic!("{name} frame #{idx}: parse: {e:?}"));
        assert_frame_matches(name, idx, tf, &fp);
    }

    eprintln!("trace-ok {name}: {} frames bit-exact", expected.len());
}

// ---------------------------------------------------------------------------
// One test per fixture. Every fixture carries a static trace.txt; the
// trace is the decoder-implementation-independent §3.8 ULP oracle.
// ---------------------------------------------------------------------------

#[test]
fn trace_mode_20ms_mono_1s_sine() {
    check_fixture("mode-20ms-mono-1s-sine");
}

#[test]
fn trace_mode_20ms_mono_1s_noise() {
    check_fixture("mode-20ms-mono-1s-noise");
}

#[test]
fn trace_mode_20ms_silence() {
    check_fixture("mode-20ms-silence");
}

#[test]
fn trace_mode_20ms_voice_like() {
    check_fixture("mode-20ms-voice-like");
}

#[test]
fn trace_mode_20ms_dtmf_tones() {
    check_fixture("mode-20ms-dtmf-tones");
}

#[test]
fn trace_mode_20ms_step_impulse() {
    check_fixture("mode-20ms-step-impulse");
}

#[test]
fn trace_mode_30ms_mono_1s_sine() {
    check_fixture("mode-30ms-mono-1s-sine");
}

#[test]
fn trace_mode_30ms_mono_1s_noise() {
    check_fixture("mode-30ms-mono-1s-noise");
}

#[test]
fn trace_mode_30ms_silence() {
    check_fixture("mode-30ms-silence");
}

#[test]
fn trace_mode_30ms_voice_like() {
    check_fixture("mode-30ms-voice-like");
}

#[test]
fn trace_containerless_storage_header() {
    check_fixture("containerless-vs-rtp-style-pair");
}

#[test]
fn trace_transition_mid_stream() {
    // The trace covers part_a_20ms.lbc (20 × 38 B) followed by
    // part_b_30ms.lbc (20 × 50 B), but its header names only the first
    // file. Drive the two halves explicitly and concatenate.
    let dir = fixture_dir("transition-mid-stream");
    let trace_path = dir.join("trace.txt");
    let trace = match fs::read_to_string(&trace_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("skip transition-mid-stream: missing trace ({e})");
            return;
        }
    };
    let expected = parse_trace(&trace);
    assert_eq!(
        expected.len(),
        40,
        "transition trace expected 40 frames, got {}",
        expected.len()
    );

    // Read both halves, strip magic, chunk per documented mode.
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(40);
    for (fname, mode) in [
        ("part_a_20ms.lbc", FrameMode::Ms20),
        ("part_b_30ms.lbc", FrameMode::Ms30),
    ] {
        let bytes = fs::read(dir.join(fname)).expect("read transition half");
        let body: &[u8] = if bytes.starts_with(ILBC_MAGIC_20) {
            &bytes[ILBC_MAGIC_20.len()..]
        } else if bytes.starts_with(ILBC_MAGIC_30) {
            &bytes[ILBC_MAGIC_30.len()..]
        } else {
            &bytes
        };
        let sz = mode.bytes();
        assert_eq!(body.len() % sz, 0, "{fname} body not {sz}-aligned");
        for chunk in body.chunks_exact(sz) {
            payloads.push(chunk.to_vec());
        }
    }
    assert_eq!(payloads.len(), 40, "transition input frame count");

    for (idx, (payload, tf)) in payloads.iter().zip(expected.iter()).enumerate() {
        let fp = parse_frame(payload)
            .unwrap_or_else(|e| panic!("transition frame #{idx}: parse: {e:?}"));
        assert_frame_matches("transition-mid-stream", idx, tf, &fp);
    }
    eprintln!("trace-ok transition-mid-stream: 40 frames bit-exact (20×20ms + 20×30ms)");
}
