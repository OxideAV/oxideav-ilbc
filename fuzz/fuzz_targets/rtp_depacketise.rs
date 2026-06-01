#![no_main]

//! Fuzz the iLBC RFC 3952 RTP depacketiser end-to-end against
//! arbitrary fuzz-supplied bytes for both the SDP `fmtp` parameter
//! string and the post-RTP-header payload body.
//!
//! Attacker model
//! ==============
//!
//! An iLBC RTP receiver has two parser surfaces facing untrusted
//! input. Both have to be panic-free on hostile data; both also have
//! to be tight enough that a malformed payload never feeds the
//! downstream iLBC decoder a buffer the decoder will index out of
//! bounds.
//!
//! 1. **SDP `a=fmtp:<pt> ...` value** — a string an RTSP / SIP / WebRTC
//!    signalling implementation hands to the codec layer to pin the
//!    session mode. RFC 3952 §4.2 only defines the `mode=20|30`
//!    parameter, but real-world SDP can carry arbitrary additional
//!    `;`-separated `key=value` parameters (`ptime=`, `maxptime=`,
//!    vendor-specific keys, etc.). The [`parse_mode_from_fmtp`] / the
//!    [`Depacketiser::from_sdp_fmtp`] wrapper has to walk that list,
//!    accept the well-formed iLBC pin, and reject everything else
//!    with a typed `Err(Error::Invalid)`. Never panic on weird input
//!    (empty string, missing key, malformed key=value pair, garbage
//!    UTF-8, very long input, embedded NULs the lossy decode keeps).
//!
//! 2. **RTP payload body** (post-12-byte-RTP-header bytes). RFC 3952
//!    §3 says the payload is one or more iLBC frames concatenated,
//!    all sharing the session-pinned mode (38 B per 20 ms frame /
//!    50 B per 30 ms frame). [`Depacketiser::depacketise`] /
//!    [`Depacketiser::depacketise_owned`] must split exactly on the
//!    mode boundary, accept any non-empty whole multiple of the
//!    per-mode frame size, and reject everything else with a typed
//!    error. The depacketised frames must never overlap each other
//!    or read past the input slice.
//!
//! Fuzz input layout
//! =================
//!
//! ```text
//!   byte 0   : seed
//!                bit 0 → mode = 20 ms (0) / 30 ms (1) for the
//!                        constructor + bounded-length path.
//!   byte 1   : `fmtp_len` (saturating cap at remaining bytes).
//!   bytes 2.. : `fmtp_len` bytes treated as the SDP `fmtp` value
//!               (lossy-UTF-8 decoded — the parser is a byte / `str`
//!               splitter and takes any `&str` even if the bytes were
//!               nonsense; we use `String::from_utf8_lossy` so the
//!               fuzzer can drive both ASCII and arbitrary UTF-8).
//!   trailing bytes : RTP payload body fed to all three depacketise
//!                    paths (one per FrameMode constructor + one for
//!                    `from_sdp_fmtp` if its construction succeeded).
//! ```
//!
//! Properties asserted on every input
//! ----------------------------------
//!
//! * The [`parse_mode_from_fmtp`] entry point never panics on any
//!   `&str` — `Ok(FrameMode)` for a well-formed `mode=20|30`, typed
//!   `Err(Error::Invalid)` otherwise. The return value is consumed
//!   but not asserted on (the fuzzer's correctness witness is
//!   "no panic"; the unit tests already enumerate the happy / sad
//!   paths).
//! * Constructing a [`Depacketiser`] from the seed byte's chosen
//!   `FrameMode` is infallible.
//! * On every call to [`Depacketiser::depacketise`] (borrowed) and
//!   [`Depacketiser::depacketise_owned`] (cloned):
//!     - return value is either `Ok(Vec<...>)` or
//!       `Err(Error::Invalid)`; nothing else;
//!     - on `Ok`, every yielded slice has length exactly equal to
//!       the depacketiser's `frame_size()`, the slice count matches
//!       `frame_count(payload.len()).unwrap()`, and the concatenated
//!       slices reconstitute the input byte-for-byte (no data loss /
//!       reordering / overlap);
//!     - on `Err`, the input length really is not a positive
//!       multiple of `frame_size()` (the contract).
//! * [`detect_mode_from_payload_len`] never panics on any `usize`
//!   (the fuzzer uses the trailing-payload length as the argument).
//! * [`empty_marker_frame`] returns a buffer of the right length
//!   with the LSB of the last byte set to 1 (RFC 3951 §3.8 / §4.5
//!   empty-frame indicator) — verified once per fuzz iteration to
//!   exercise the cached LUT-style allocation path.
//! * [`Packetiser::pack_series`] is panic-free on a fuzz-derived
//!   list of frame slices when each slice happens to match the
//!   per-mode length; the returned `(body, ts_offset)` tuples sum
//!   their body lengths back to the input total and the timestamps
//!   are monotone-non-decreasing.

use libfuzzer_sys::fuzz_target;
use oxideav_ilbc::rtp::{
    detect_mode_from_payload_len, empty_marker_frame, parse_mode_from_fmtp, Depacketiser,
    Packetiser,
};
use oxideav_ilbc::FrameMode;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let seed = data[0];
    let mode_30ms = (seed & 0x01) != 0;
    let mode = if mode_30ms {
        FrameMode::Ms30
    } else {
        FrameMode::Ms20
    };

    // Length-based mode detection on the trailing-payload length: must
    // never panic. We don't assert the result — the unit tests already
    // pin the happy / sad path matrix.
    let _ = detect_mode_from_payload_len(data.len());
    let _ = detect_mode_from_payload_len(usize::MAX); // saturation path

    // Empty-marker helper: exercise the allocation path and verify the
    // RFC 3951 §3.8 empty-frame-indicator invariant on every iteration.
    let m = empty_marker_frame(mode);
    assert_eq!(m.len(), mode.bytes());
    assert_eq!(*m.last().unwrap() & 0x01, 0x01);

    // Split the trailing input into (fmtp slice, body slice).
    let rest = &data[1..];
    let fmtp_len = if rest.is_empty() {
        0
    } else {
        // saturating slice — `rest[0]` capped at `rest.len() - 1` so the
        // body always gets at least zero bytes (we tolerate an empty
        // body; the depacketiser rejects empty payloads with a typed
        // error, which is part of the contract under test).
        (rest[0] as usize).min(rest.len().saturating_sub(1))
    };
    let (fmtp_bytes, body) = if rest.is_empty() {
        (&[][..], &[][..])
    } else {
        let after_lenbyte = &rest[1..];
        let fl = fmtp_len.min(after_lenbyte.len());
        (&after_lenbyte[..fl], &after_lenbyte[fl..])
    };

    // SDP fmtp parser — `String::from_utf8_lossy` keeps the surface
    // testable on non-UTF-8 input (the parser only ever takes `&str`,
    // never raw bytes, so we have to bridge the type). Must not panic
    // on any byte sequence; either an `Ok(FrameMode)` or a typed
    // `Err(Error::Invalid)`.
    let fmtp_str = String::from_utf8_lossy(fmtp_bytes);
    let _ = parse_mode_from_fmtp(&fmtp_str);
    let from_sdp = Depacketiser::from_sdp_fmtp(&fmtp_str).ok();

    // Constructor from the seed-pinned mode — infallible.
    let depk = Depacketiser::new(mode);
    assert_eq!(depk.mode(), mode);
    assert_eq!(depk.frame_size(), mode.bytes());

    // Drive both the borrowed and owned depacketise paths against the
    // body. Both must agree (same frame count, same per-frame length,
    // same byte content).
    let borrowed = depk.depacketise(body);
    let owned = depk.depacketise_owned(body);

    match (borrowed, owned) {
        (Ok(slices), Ok(vecs)) => {
            let fs = depk.frame_size();
            // Frame-count agreement with the `frame_count` accessor.
            assert_eq!(depk.frame_count(body.len()), Some(slices.len()));
            assert_eq!(slices.len(), vecs.len());
            assert!(
                !slices.is_empty(),
                "non-empty Ok payload must produce ≥1 frame"
            );
            for (i, s) in slices.iter().enumerate() {
                assert_eq!(s.len(), fs, "frame {i} not exactly frame_size bytes");
                assert_eq!(
                    vecs[i].len(),
                    fs,
                    "owned frame {i} not exactly frame_size bytes"
                );
                assert_eq!(vecs[i].as_slice(), *s, "owned and borrowed frames diverge");
            }
            // Reconstitution: concatenated slices == input body
            // (no data loss / reordering / overlap).
            let mut acc = Vec::with_capacity(body.len());
            for s in &slices {
                acc.extend_from_slice(s);
            }
            assert_eq!(acc.as_slice(), body, "depacketise lost or reordered bytes");
        }
        (Err(_), Err(_)) => {
            // Negative-path contract: the input length really must
            // not be a positive multiple of the per-mode frame size.
            let fs = depk.frame_size();
            assert!(
                body.is_empty() || body.len() % fs != 0,
                "depacketiser rejected a structurally-valid payload (len={}, fs={fs})",
                body.len(),
            );
        }
        (a, b) => {
            // The borrowed and owned variants must agree on the
            // accept / reject decision. If they don't, that's a real
            // bug — neither variant should ever diverge from the
            // other on the same input.
            panic!(
                "depacketise borrowed/owned disagree on body: borrowed={:?}, owned={:?}",
                a.is_ok(),
                b.is_ok(),
            );
        }
    }

    // If the SDP fmtp parser produced a depacketiser, drive it across
    // the same body. It might be pinned to a different mode than the
    // seed-chosen one, so we only check the structural contract (not
    // a cross-depacketiser equality).
    if let Some(d) = from_sdp {
        let fs = d.frame_size();
        match d.depacketise(body) {
            Ok(slices) => {
                assert_eq!(d.frame_count(body.len()), Some(slices.len()));
                for s in &slices {
                    assert_eq!(s.len(), fs);
                }
            }
            Err(_) => {
                assert!(
                    body.is_empty() || body.len() % fs != 0,
                    "sdp-pinned depacketiser rejected a structurally-valid payload",
                );
            }
        }
    }

    // Packetiser-side fuzzing: build a list of well-formed frame
    // buffers out of the input bytes (chunking by the seed-pinned
    // frame size and zero-padding the tail) and round-trip them
    // through `pack_series` + `depacketise`. The point is to drive
    // `pack_series` on attacker-chosen frame counts.
    let fs = mode.bytes();
    let frame_count_cap = 32usize; // bound the work
    let mut owned_frames: Vec<Vec<u8>> = Vec::new();
    let mut start = 0usize;
    while start + fs <= body.len() && owned_frames.len() < frame_count_cap {
        owned_frames.push(body[start..start + fs].to_vec());
        start += fs;
    }
    if !owned_frames.is_empty() {
        let refs: Vec<&[u8]> = owned_frames.iter().map(|v| v.as_slice()).collect();
        // Vary the per-packet cap deterministically from the seed.
        let cap = ((seed >> 1) as usize & 0x07).clamp(1, 8);
        let pk = Packetiser::with_max_frames_per_packet(mode, cap);
        let series = pk
            .pack_series(&refs)
            .expect("pack_series on well-formed equal-length frames must succeed");
        // Sum of body lengths == sum of input frame lengths.
        let sum_body: usize = series.iter().map(|(b, _)| b.len()).sum();
        assert_eq!(sum_body, owned_frames.len() * fs);
        // Timestamps strictly non-decreasing; first packet starts at 0.
        assert_eq!(series.first().map(|(_, t)| *t), Some(0));
        let mut prev = 0u32;
        for (_b, t) in &series {
            assert!(*t >= prev, "pack_series timestamps regressed");
            prev = *t;
        }
        // Round-trip through the depacketiser: every emitted body is
        // a clean multiple of `fs`, depacketises into frame_count
        // frames, and concatenating across all packets reproduces
        // the input.
        let mut got: Vec<Vec<u8>> = Vec::new();
        for (b, _t) in &series {
            assert!(b.len() % fs == 0);
            for s in depk.depacketise(b).unwrap() {
                got.push(s.to_vec());
            }
        }
        assert_eq!(got, owned_frames, "pack→depack lost data");
    }
});
