#![no_main]

//! Focused fuzz target for the RFC 3952 §4.2 SDP `fmtp` parser
//! surface. Companion to the `rtp_depacketise` target — that target
//! exercises depacketise / packetise round-trips end-to-end and only
//! threads a small length-prefixed slice of the fuzz input into the
//! `fmtp` string parser. This target hands the *entire* fuzz input
//! to the parser as the `fmtp` value, so libFuzzer can spend its
//! whole iteration budget exploring grammar edge cases of the
//! `;`-separated `key=value` list.
//!
//! Attacker model
//! ==============
//!
//! An iLBC RTP receiver gets an SDP `a=fmtp:<pt> ...` line from the
//! signalling channel. RFC 3952 §4.2 reserves the iLBC-specific
//! `mode=20|30` parameter inside that list; the parser
//! ([`parse_mode_from_fmtp`]) is the choke-point that turns that
//! list into a typed [`FrameMode`].
//!
//! The parser MUST be panic-free on any byte sequence the caller
//! happens to drop in front of it:
//!
//! * empty string;
//! * arbitrary UTF-8 (the parser only takes `&str`, but real SDP
//!   sources occasionally interleave invalid sequences — the lossy
//!   bridge below covers that surface);
//! * trailing / leading / internal whitespace of any size;
//! * a `;`-list of arbitrary length with arbitrary keys (the parser
//!   case-insensitively matches `mode` and skips every other key);
//! * a piece that elides the `=` separator entirely, a piece whose
//!   value is empty, a piece carrying several `=` characters in a
//!   row, a piece with a non-ASCII key, and so on.
//!
//! Fuzz input layout
//! =================
//!
//! ```text
//!   bytes 0..  : raw SDP `fmtp` value, lossy-UTF-8 decoded.
//! ```
//!
//! Properties asserted on every input
//! ==================================
//!
//! * [`parse_mode_from_fmtp`] is panic-free for any `&str`. Its
//!   return value is one of `Ok(FrameMode::Ms20)`,
//!   `Ok(FrameMode::Ms30)`, or `Err(Error::Invalid)`. Nothing else
//!   reaches the caller.
//! * [`Depacketiser::from_sdp_fmtp`] mirrors the same decision on
//!   the same input (it is a thin wrapper around
//!   `parse_mode_from_fmtp` plus a [`Depacketiser::new`] call). When
//!   one returns `Ok`, the other does too, and the pinned mode
//!   matches.
//! * On any `Ok(mode)`:
//!     - [`format_mode_fmtp(mode)`] returns the bare `mode=20` or
//!       `mode=30` token;
//!     - [`build_fmtp(mode, None)`] returns the same bare token
//!       (a `None` cap stays inside the parameter list rather than
//!       emitting a trailing `maxptime`);
//!     - `parse_mode_from_fmtp(&format_mode_fmtp(mode))` and
//!       `parse_mode_from_fmtp(&build_fmtp(mode, None))` both
//!       round-trip to the same `mode`.
//! * For each accepted `mode`, [`build_fmtp(mode, Some(cap))`] for
//!   `cap ∈ {0, 1, 2, 8, 255}` is well-formed: a `cap` of 0 or 1
//!   collapses to the bare `mode=N` token (a `maxptime` equal to
//!   one per-frame `ptime` is a no-op); a `cap > 1` emits a
//!   `;maxptime=<ms>` suffix whose integer value is
//!   `cap * frame_duration_ms(mode)`. Each emission also
//!   round-trips through `parse_mode_from_fmtp` to the same mode.
//! * On any `Err(_)`, [`Depacketiser::from_sdp_fmtp`] also returns
//!   `Err(_)`; the agreement is the parser's only structural
//!   contract.

use libfuzzer_sys::fuzz_target;
use oxideav_ilbc::rtp::{
    build_fmtp, format_mode_fmtp, frame_duration_ms, parse_mode_from_fmtp, Depacketiser,
};

fuzz_target!(|data: &[u8]| {
    // The parser only takes `&str` — bridge arbitrary bytes via
    // lossy UTF-8 so the fuzzer can explore the full byte space
    // even though the source type is a `str`. (The parser is a
    // `split(';')` + `split_once('=')` + `trim()` chain; from its
    // perspective the lossy bridge keeps every byte sequence the
    // caller could realistically present.)
    let fmtp = String::from_utf8_lossy(data);

    // Panic-freedom contract on the public parser.
    let parsed = parse_mode_from_fmtp(&fmtp);

    // The thin `Depacketiser::from_sdp_fmtp` wrapper must match the
    // parser's accept / reject decision on the same input.
    let from_sdp = Depacketiser::from_sdp_fmtp(&fmtp);
    match (&parsed, &from_sdp) {
        (Ok(a), Ok(d)) => assert_eq!(
            *a,
            d.mode(),
            "from_sdp_fmtp and parse_mode_from_fmtp disagree on accepted mode",
        ),
        (Err(_), Err(_)) => {}
        (a, b) => panic!(
            "from_sdp_fmtp and parse_mode_from_fmtp disagree on accept/reject: \
             parse={:?}, from_sdp={:?}",
            a.is_ok(),
            b.is_ok(),
        ),
    }

    // Outbound builders + round-trip checks (only meaningful when
    // the parser accepted the input).
    if let Ok(mode) = parsed {
        // `format_mode_fmtp` is a tiny "mode={20|30}" formatter; the
        // parser must accept its output back as the same mode.
        let token = format_mode_fmtp(mode);
        let re = parse_mode_from_fmtp(&token)
            .expect("format_mode_fmtp output must parse back into a FrameMode");
        assert_eq!(re, mode, "format_mode_fmtp lost the mode on round-trip");

        // `build_fmtp(mode, None)` collapses to the same bare token —
        // the documented invariant.
        let bare = build_fmtp(mode, None);
        assert_eq!(
            bare, token,
            "build_fmtp(mode, None) must equal format_mode_fmtp(mode)",
        );

        // The `maxptime` suffix emission contract: cap 0 / 1 collapse
        // to the bare token (per-frame ptime == maxptime is a no-op);
        // cap > 1 emits `;maxptime=<cap * frame_duration_ms(mode)>`.
        for cap in [0usize, 1, 2, 8, 255] {
            let v = build_fmtp(mode, Some(cap));
            if cap <= 1 {
                assert_eq!(
                    v, token,
                    "build_fmtp({mode:?}, Some({cap})) must collapse to bare mode",
                );
            } else {
                let want_ms = (cap as u32).saturating_mul(frame_duration_ms(mode));
                let want = format!("{token};maxptime={want_ms}");
                assert_eq!(v, want, "build_fmtp({mode:?}, Some({cap})) emission shape",);
            }
            // Every emission round-trips through the parser to the
            // same mode regardless of cap.
            let back = parse_mode_from_fmtp(&v)
                .expect("build_fmtp output must parse back into a FrameMode");
            assert_eq!(
                back, mode,
                "build_fmtp({mode:?}, Some({cap})) lost the mode on parse round-trip",
            );
        }
    }
});
