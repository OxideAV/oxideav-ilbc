#![no_main]

//! Focused fuzz target for the RFC 3952 RTP gap-fill / packet-loss
//! concealment surface on `oxideav_ilbc::rtp::Depacketiser`. Companion
//! to the existing `rtp_depacketise` target (which exercises the
//! happy-path depacketise / packetise round-trip) and `sdp_fmtp` (which
//! exercises the SDP `fmtp` parser). This target hands the entire fuzz
//! input to the concealment-frame builders and the sequence-number
//! gap arithmetic so libFuzzer can spend its whole iteration budget
//! exploring the dropped-frame path the round-240 helpers opened up.
//!
//! Attacker model
//! ==============
//!
//! An iLBC RTP receiver that detects packet loss via the 16-bit RTP
//! sequence-number gap (RFC 3550 §3.3 numbering) drives the
//! [`crate::synthesis`] §4.5 dampened pitch-synchronous PLC by handing
//! the decoder one RFC 3951 §3.8 empty-marker frame per missing audio
//! frame. The depacketiser exposes four helpers and one free function
//! that together translate the upstream sequence-number delta into
//! that frame list:
//!
//! 1. [`rtp_seq_gap`] — folds the 16-bit `(last, now)` pair into a
//!    signed-arc gap count, treating the > 2^15 direction as a
//!    backward jump that yields 0.
//! 2. [`Depacketiser::gap_frame_count`] — multiplies a packet-count gap
//!    by the steady-state `frames_per_payload` aggregation to give the
//!    iLBC frame count to conceal. Defensively saturates on overflow.
//! 3. [`Depacketiser::conceal_gap`] / [`Depacketiser::concealment_payload`]
//!    — emit the per-frame buffers (`Vec<Vec<u8>>`) and the concatenated
//!    body (`Option<Vec<u8>>`) shapes of the empty-marker frame list.
//! 4. [`Depacketiser::depacketise_with_gap_fill`] — chains the gap-fill
//!    onto a live payload, returning the concealment frames followed
//!    by the depacketised live frames.
//!
//! Each of these must be panic-free on hostile input — arbitrary u16
//! pairs, arbitrary gap-packet counts, arbitrary aggregations, and
//! arbitrary live-payload bytes. The contracts asserted on every
//! iteration are entirely structural; the round-240 unit tests
//! enumerate the happy / sad path matrix.
//!
//! Fuzz input layout
//! =================
//!
//! ```text
//!   byte 0   : seed
//!                bit 0    → mode = 20 ms (0) / 30 ms (1)
//!                bits 1..3 → frames_per_payload nibble
//!                            (0..=7; clamped to >=1 unless explicitly
//!                             zero to drive the "no observation yet"
//!                             guard)
//!                bits 4..7 → gap_packets cap nibble (0..=15)
//!   bytes 1..3 : RTP seq pair as little-endian u16 (last)
//!                + the next byte fills the low half of `now`.
//!   byte 3      : high half of `now`.
//!   byte 4      : `missing_frames_raw` argument fed directly to the
//!                 `conceal_gap` / `concealment_payload` helpers (lets
//!                 the fuzzer hit the count surface without going
//!                 through `gap_frame_count`).
//!   bytes 5..  : RTP payload body (post-12-byte-RTP-header bytes)
//!                fed to `depacketise_with_gap_fill`.
//! ```
//!
//! Properties asserted on every input
//! ==================================
//!
//! * [`rtp_seq_gap`] is panic-free for any (last, now) `u16` pair and
//!   never reports a gap larger than `0x7FFF` — the signed-arc fold.
//!   The diagonal `now == last` and the in-order step `now == last + 1`
//!   both report `0` (no concealment).
//! * [`Depacketiser::gap_frame_count`] is panic-free for any `(gap, agg)`
//!   `usize` pair: it returns `gap * agg` clamped to `usize::MAX` via
//!   saturating multiplication. `agg == 0` always yields `0` (the
//!   defensive "no observation yet" guard documented on the helper).
//! * [`Depacketiser::conceal_gap(n)`] returns exactly `n` frames, each
//!   `mode.bytes()` long, each carrying the RFC 3951 §3.8 empty-frame
//!   indicator at the LSB of the final byte and zero elsewhere.
//! * [`Depacketiser::concealment_payload(n)`] returns `Some(body)` for
//!   `n >= 1` and `None` for `n == 0`. The body length is exactly
//!   `n * mode.bytes()`, every per-mode-frame slice carries the
//!   empty-marker indicator, and feeding the body back through
//!   [`Depacketiser::depacketise`] yields `n` frames that compare
//!   byte-equal to the per-frame buffers produced by `conceal_gap(n)`.
//! * [`Depacketiser::depacketise_with_gap_fill`] mirrors the
//!   `depacketise(body)` accept / reject decision on the live tail —
//!   accepting iff the body is a positive multiple of `mode.bytes()`.
//!   On accept, the returned `Vec` has exactly
//!   `missing + body.len() / mode.bytes()` frames, where `missing`
//!   equals `gap_frame_count(gap_packets, frames_per_payload)`. The
//!   first `missing` are empty-marker frames; the trailing
//!   `body.len() / fs` are the depacketised live frames in order.
//! * On any [`Depacketiser::depacketise_with_gap_fill`] reject, the
//!   live body really is not a positive multiple of the per-mode
//!   frame size — the gap-fill never invents a successful decode.
//!
//! The work bound on every iteration is the smaller of the fuzz-derived
//! `missing_frames` count and a 1024-frame ceiling so an attacker
//! cannot drive the harness into an OOM-on-allocation tarpit (the
//! production helper is unbounded; the cap here protects the fuzzer
//! iteration budget). The same cap is applied to
//! `gap_frame_count(gap_packets * frames_per_payload)` before driving
//! the depacketise_with_gap_fill call.

use libfuzzer_sys::fuzz_target;
use oxideav_ilbc::rtp::{empty_marker_frame, rtp_seq_gap, Depacketiser};
use oxideav_ilbc::FrameMode;

/// Cap on the per-iteration concealment-frame count. Production code
/// is unbounded; the fuzzer needs a ceiling so a malicious seed can't
/// drive the harness into a multi-second allocation that starves
/// libFuzzer's iteration budget.
const FUZZ_MISSING_FRAMES_CAP: usize = 1024;

fuzz_target!(|data: &[u8]| {
    // Step 1: decode the fixed-shape header (`seed`, RTP seq pair,
    // direct `missing_frames` byte). Bail early on short inputs — the
    // fuzz harness is content with no-op runs on near-empty input.
    if data.len() < 5 {
        // Still exercise the two purely-numeric helpers on the empty
        // / near-empty cases so libFuzzer sees the `(0, 0)` corner.
        let _ = rtp_seq_gap(0, 0);
        let _ = Depacketiser::new(FrameMode::Ms20).gap_frame_count(0, 0);
        return;
    }

    let seed = data[0];
    let mode = if (seed & 0x01) != 0 {
        FrameMode::Ms30
    } else {
        FrameMode::Ms20
    };

    // 0..=7 from bits 1..3. A 0 stays 0 so the fuzzer can hit the
    // "no payload observed yet → frames_per_payload == 0" guard
    // documented on `gap_frame_count`.
    let frames_per_payload = ((seed >> 1) & 0x07) as usize;

    // 0..=15 from bits 4..7. Used for `gap_packets`.
    let gap_packets = ((seed >> 4) & 0x0F) as usize;

    let last = u16::from_le_bytes([data[1], data[2]]);
    let now = u16::from_le_bytes([data[3], data[4 % data.len()]]);
    let missing_frames_raw = data[4] as usize;
    let body = if data.len() > 5 { &data[5..] } else { &[][..] };

    // Step 2: `rtp_seq_gap` is panic-free on every (last, now) pair
    // and bounded above by 0x7FFF (the signed-arc fold).
    let arc_gap = rtp_seq_gap(last, now);
    assert!(
        arc_gap <= 0x7FFF,
        "rtp_seq_gap exceeded signed-arc ceiling: last={last}, now={now}, arc_gap={arc_gap}",
    );
    // The in-order step and the diagonal both yield zero.
    assert_eq!(
        rtp_seq_gap(last, last),
        0,
        "rtp_seq_gap diagonal must be 0: last={last}",
    );
    assert_eq!(
        rtp_seq_gap(last, last.wrapping_add(1)),
        0,
        "rtp_seq_gap in-order step must be 0: last={last}",
    );

    // Step 3: build the depacketiser and exercise gap_frame_count on
    // the seed-derived (gap_packets, frames_per_payload) pair plus a
    // saturation-edge pair to drive the saturating_mul boundary.
    let depk = Depacketiser::new(mode);

    let missing_seed = depk.gap_frame_count(gap_packets, frames_per_payload);
    assert_eq!(
        missing_seed,
        gap_packets.saturating_mul(frames_per_payload),
        "gap_frame_count diverged from saturating_mul",
    );

    // Defensive "no observation yet" guard documented on the helper:
    // frames_per_payload == 0 ⇒ always 0 frames concealed.
    assert_eq!(
        depk.gap_frame_count(usize::MAX, 0),
        0,
        "gap_frame_count must yield 0 on frames_per_payload == 0 (defensive guard)",
    );
    // Saturation edge: usize::MAX * usize::MAX saturates to usize::MAX.
    assert_eq!(
        depk.gap_frame_count(usize::MAX, usize::MAX),
        usize::MAX,
        "gap_frame_count must saturate on overflow",
    );

    // Step 4: drive `conceal_gap` and `concealment_payload` on the
    // direct `missing_frames_raw` count (capped at the fuzz ceiling)
    // and check the structural invariants on every emitted frame.
    let missing_direct = missing_frames_raw.min(FUZZ_MISSING_FRAMES_CAP);
    let fs = mode.bytes();
    let per_frame_marker = empty_marker_frame(mode);

    let conceal_vec = depk.conceal_gap(missing_direct);
    assert_eq!(
        conceal_vec.len(),
        missing_direct,
        "conceal_gap frame count mismatch",
    );
    for (i, frame) in conceal_vec.iter().enumerate() {
        assert_eq!(frame.len(), fs, "conceal_gap frame {i} wrong length");
        assert_eq!(
            *frame, per_frame_marker,
            "conceal_gap frame {i} diverged from empty_marker_frame template",
        );
    }

    let conceal_body = depk.concealment_payload(missing_direct);
    match (missing_direct, &conceal_body) {
        (0, None) => {}
        (n, Some(body)) if n >= 1 => {
            assert_eq!(
                body.len(),
                n * fs,
                "concealment_payload body length mismatch (missing={n}, fs={fs})",
            );
            // Every per-mode slice of the body must carry the
            // RFC 3951 §3.8 empty-frame indicator.
            for (k, chunk) in body.chunks_exact(fs).enumerate() {
                assert_eq!(
                    chunk[fs - 1] & 0x01,
                    0x01,
                    "concealment_payload slice {k} missing empty-frame indicator",
                );
                for (j, b) in chunk.iter().enumerate().take(fs - 1) {
                    assert_eq!(
                        *b, 0,
                        "concealment_payload slice {k} non-zero pre-marker byte at {j}",
                    );
                }
            }
            // Round-trip through `depacketise` must yield N frames each
            // equal to the per-frame `conceal_gap` template.
            let depacked = depk
                .depacketise(body)
                .expect("concealment_payload body must depacketise cleanly");
            assert_eq!(
                depacked.len(),
                n,
                "concealment_payload depacketise frame count mismatch",
            );
            for (k, slice) in depacked.iter().enumerate() {
                assert_eq!(
                    *slice,
                    per_frame_marker.as_slice(),
                    "depacketised concealment slice {k} diverged from empty_marker_frame",
                );
            }
        }
        (n, body) => panic!(
            "concealment_payload shape contract broken: missing={n}, body.is_some={}",
            body.is_some(),
        ),
    }

    // Step 5: drive `depacketise_with_gap_fill` on the body plus the
    // seed-derived (gap_packets, frames_per_payload). Cap the missing
    // frames the helper will produce so we don't OOM on a malicious
    // seed pair like (255, 7) which would request 1785 frames.
    let missing_for_gap_fill = missing_seed.min(FUZZ_MISSING_FRAMES_CAP);
    // To stay bounded we feed `frames_per_payload` directly only when
    // the product is below the cap; otherwise pass a (1, missing_cap)
    // pair that yields the same number of concealment frames.
    let (gap_arg, agg_arg) = if missing_for_gap_fill == missing_seed {
        (gap_packets, frames_per_payload)
    } else {
        (1usize, missing_for_gap_fill)
    };

    match depk.depacketise_with_gap_fill(gap_arg, agg_arg, body) {
        Ok(frames) => {
            let live_count = body.len() / fs;
            assert_eq!(
                frames.len(),
                missing_for_gap_fill + live_count,
                "depacketise_with_gap_fill frame count mismatch",
            );
            // First `missing_for_gap_fill` frames are empty-markers;
            // the rest are the depacketised live frames in order.
            for (i, frame) in frames.iter().enumerate().take(missing_for_gap_fill) {
                assert_eq!(
                    frame.len(),
                    fs,
                    "gap_fill concealment frame {i} wrong length",
                );
                assert_eq!(
                    *frame, per_frame_marker,
                    "gap_fill concealment frame {i} diverged from empty_marker_frame",
                );
            }
            for k in 0..live_count {
                let frame_idx = missing_for_gap_fill + k;
                let want = &body[k * fs..(k + 1) * fs];
                assert_eq!(
                    frames[frame_idx].as_slice(),
                    want,
                    "gap_fill live frame {k} diverged from input body slice",
                );
            }
        }
        Err(_) => {
            // The helper rejects iff `depacketise(body)` would reject —
            // empty body or non-multiple length. The gap-fill path
            // must never invent a successful decode on a malformed
            // live tail.
            assert!(
                body.is_empty() || body.len() % fs != 0,
                "depacketise_with_gap_fill rejected a structurally-valid body \
                 (mode={mode:?}, fs={fs}, body.len={})",
                body.len(),
            );
        }
    }
});
