//! RFC 3952 — RTP Payload Format for iLBC Speech.
//!
//! This module implements the depacketiser and packetiser for the
//! iLBC payload format defined by RFC 3952. It deliberately does not
//! own any RTP transport state — the 12-byte fixed RTP header
//! (sequence number, timestamp, SSRC…) is codec-agnostic and lives
//! one layer above this module. What is in scope here is the *iLBC
//! payload* that an RTP receiver hands us after stripping that fixed
//! header, plus the SDP `mode` parameter that pins the session's
//! frame mode.
//!
//! Authoritative facts used by this module
//! ---------------------------------------
//!
//! * RFC 3952 §3 *Payload Format* — the iLBC RTP payload is the
//!   encoded bitstream itself, with no per-packet codec header. A
//!   single RTP packet MAY carry one or more iLBC frames, and all
//!   frames in a packet MUST share the same operating mode (20 ms
//!   or 30 ms). RFC 3951 §3.8 / Table 3.2 sets the per-mode frame
//!   size in bytes: 38 for 20 ms (304 bits) and 50 for 30 ms
//!   (400 bits).
//! * RFC 3952 §4.2 *Mapping to SDP* — the SDP `a=fmtp:<pt> mode=N`
//!   parameter pins one frame mode for the whole RTP session, where
//!   `N` is `20` or `30`. There is no per-packet mode indicator in
//!   the wire bitstream.
//!
//! These two facts make a strict depacketiser stateless once it
//! knows the session mode: split the payload into fixed-size
//! chunks, hand each chunk to the iLBC decoder, done.
//!
//! Out of scope (and intentionally so)
//! ------------------------------------
//!
//! * The RTP fixed header. Codec-agnostic.
//! * RTCP. Out-of-band.
//! * Real-world packet loss handling. That belongs in the decoder's
//!   PLC path (see [`crate::synthesis`] for our pitch-synchronous
//!   PLC); when an RTP packet is lost, the caller hands the
//!   decoder an [`empty`-marked frame](`empty_marker_frame`) for
//!   the relevant slot and the decoder runs PLC.
//! * SDP parsing in general. This module ships a *very* narrow
//!   `fmtp` parameter parser for `mode=`; everything else in the
//!   `a=fmtp:` line is left to the caller.
//!
//! The depacketiser refuses to consume any input until the session
//! mode is locked, mirroring the RFC's "the mode is signalled
//! out-of-band" requirement. See [`Depacketiser::new`] /
//! [`Depacketiser::from_sdp_fmtp`].

use oxideav_core::{Error, Result};

use crate::{FrameMode, FRAME_BYTES_20MS, FRAME_BYTES_30MS};

/// Default RTP clock rate for iLBC: 8 kHz, matching the codec's
/// narrowband sample rate (RFC 3951 §1 / §3.8). The RTP timestamp
/// in an iLBC RTP stream ticks once per audio sample; one 20 ms
/// frame advances it by 160, one 30 ms frame by 240.
pub const RTP_CLOCK_RATE_HZ: u32 = 8_000;

/// RTP timestamp delta per 20 ms iLBC frame.
pub const TS_DELTA_20MS: u32 = 160;

/// RTP timestamp delta per 30 ms iLBC frame.
pub const TS_DELTA_30MS: u32 = 240;

/// A depacketiser that splits an RTP payload into iLBC frames.
///
/// The mode is pinned at construction time and never changes for the
/// life of the depacketiser — RFC 3952 §4.2 forbids per-packet mode
/// changes within a session.
#[derive(Clone, Debug)]
pub struct Depacketiser {
    mode: FrameMode,
}

impl Depacketiser {
    /// Construct a depacketiser pinned to the supplied mode.
    pub fn new(mode: FrameMode) -> Self {
        Self { mode }
    }

    /// Construct a depacketiser from the SDP `a=fmtp:<pt> ...` parameter
    /// payload (everything after the `<pt> ` token). Recognises the
    /// `mode=20|30` parameter; everything else is ignored.
    ///
    /// Returns an error when `mode` is missing or carries an unknown
    /// value — the receiver MUST know the mode before depacketising,
    /// so silently picking a default would mask a real interop bug.
    pub fn from_sdp_fmtp(fmtp_value: &str) -> Result<Self> {
        let mode = parse_mode_from_fmtp(fmtp_value)?;
        Ok(Self::new(mode))
    }

    /// Mode this depacketiser will split payloads against.
    pub fn mode(&self) -> FrameMode {
        self.mode
    }

    /// Number of bytes consumed per iLBC frame in this mode.
    pub fn frame_size(&self) -> usize {
        self.mode.bytes()
    }

    /// RTP timestamp delta per iLBC frame in this mode.
    pub fn ts_delta(&self) -> u32 {
        match self.mode {
            FrameMode::Ms20 => TS_DELTA_20MS,
            FrameMode::Ms30 => TS_DELTA_30MS,
        }
    }

    /// How many iLBC frames a payload of length `payload_len` would
    /// carry, assuming it is well-formed for this depacketiser's mode.
    /// Returns `None` when `payload_len` is not a clean multiple of
    /// the per-frame byte count.
    pub fn frame_count(&self, payload_len: usize) -> Option<usize> {
        let fs = self.frame_size();
        if payload_len == 0 || payload_len % fs != 0 {
            None
        } else {
            Some(payload_len / fs)
        }
    }

    /// Split an RTP payload into its iLBC frames.
    ///
    /// `payload` is the bytes the RTP receiver hands us *after*
    /// stripping the 12-byte fixed RTP header. The slice MUST be
    /// either empty (which is rejected) or a positive multiple of
    /// the per-mode frame size; anything else is rejected per
    /// RFC 3952 §3 (the iLBC payload format has no padding /
    /// header / framing of its own — every byte is part of one of
    /// the contained iLBC frames).
    pub fn depacketise<'a>(&self, payload: &'a [u8]) -> Result<Vec<&'a [u8]>> {
        let fs = self.frame_size();
        if payload.is_empty() {
            return Err(Error::invalid(
                "iLBC RTP depacketiser: empty payload (RFC 3952 §3 requires ≥1 frame)",
            ));
        }
        if payload.len() % fs != 0 {
            return Err(Error::invalid(
                "iLBC RTP depacketiser: payload length is not a multiple of the mode's frame size",
            ));
        }
        Ok(payload.chunks_exact(fs).collect())
    }

    /// Owned variant of [`depacketise`] that copies each frame
    /// into its own `Vec<u8>`. Convenient for handing frames to the
    /// decoder across the public `oxideav_core::Decoder::send_packet`
    /// boundary, which requires an owned `Packet`.
    pub fn depacketise_owned(&self, payload: &[u8]) -> Result<Vec<Vec<u8>>> {
        Ok(self
            .depacketise(payload)?
            .into_iter()
            .map(|s| s.to_vec())
            .collect())
    }

    /// Emit `missing_frames` empty-marker frames sized for this
    /// depacketiser's mode. Each frame is the buffer produced by
    /// [`empty_marker_frame`] (all-zero except the LSB of the last
    /// byte). Feeding these to the decoder advances the PLC path one
    /// frame at a time (RFC 3951 §3.8 empty-frame indicator → §4.5
    /// dampened pitch-synchronous concealment), so the output PCM
    /// stream stays aligned with the wall-clock duration the gap
    /// represented.
    ///
    /// Returns an empty `Vec` when `missing_frames == 0` (a vacuous
    /// "no gap" call is fine — the typical caller passes the result
    /// of [`Self::gap_frame_count`] which can legitimately be zero).
    pub fn conceal_gap(&self, missing_frames: usize) -> Vec<Vec<u8>> {
        (0..missing_frames)
            .map(|_| empty_marker_frame(self.mode))
            .collect()
    }

    /// Concatenated form of [`Self::conceal_gap`]: emits one
    /// `Vec<u8>` of `missing_frames × frame_size` bytes that
    /// [`Self::depacketise`] would split back into individual
    /// empty-marker frames. Useful when the caller plans to drive the
    /// gap fill through the same decode path as a normal RTP body
    /// (one call to `depacketise`, then iterate the frames).
    ///
    /// Returns `None` when `missing_frames == 0` — a zero-length
    /// payload is a `depacketise` rejection per RFC 3952 §3, so
    /// callers handle the no-gap case by skipping the call.
    pub fn concealment_payload(&self, missing_frames: usize) -> Option<Vec<u8>> {
        if missing_frames == 0 {
            return None;
        }
        let fs = self.frame_size();
        let mut out = vec![0u8; missing_frames * fs];
        // Place the empty-frame indicator at the LSB of each frame's
        // final byte (RFC 3951 §3.8).
        for k in 0..missing_frames {
            out[(k + 1) * fs - 1] = 0x01;
        }
        Some(out)
    }

    /// Translate an RTP sequence-number gap into the iLBC-frame count
    /// the gap represented, assuming each missing packet carried
    /// `frames_per_payload` frames (a common steady-state
    /// invariant — RFC 3952 §3 makes no per-packet header surface
    /// that would let the receiver introspect the lost packets, so
    /// the typical sender uses a constant aggregation through the
    /// session).
    ///
    /// `gap_packets` is the number of *missing* RTP packets, computed
    /// upstream from the RTP fixed-header sequence-number delta
    /// (`seq_now - seq_last - 1`, with 16-bit wrap handled by the
    /// caller). `frames_per_payload` is most usefully the value
    /// [`Self::frame_count`] reported on the most recently received
    /// payload — that derivation is exposed on [`Self`] so a caller
    /// can chain `frame_count(last_payload.len())` straight into
    /// `gap_frame_count`.
    ///
    /// A `frames_per_payload` of 0 yields 0 — a defensive guard so a
    /// post-restart caller that has not yet observed a payload (and
    /// therefore cannot know the steady-state aggregation) does not
    /// over-conceal.
    pub fn gap_frame_count(&self, gap_packets: usize, frames_per_payload: usize) -> usize {
        gap_packets.saturating_mul(frames_per_payload)
    }

    /// One-shot helper that produces the full "concealment then
    /// payload" frame list a decoder driver feeds the iLBC decoder
    /// when an RTP packet arrives after a gap.
    ///
    /// `gap_packets` is the number of missing RTP packets the caller
    /// detected (typically `seq_now - seq_last - 1` with 16-bit wrap
    /// handled by [`rtp_seq_gap`]). `frames_per_payload` is the
    /// steady-state aggregation the session is using; pass the result
    /// of [`Self::frame_count`] on a previously-received payload.
    /// `payload` is the newly-arrived RTP body (post-RTP-header
    /// bytes).
    ///
    /// Returns a `Vec` containing first the gap-fill empty-marker
    /// frames (in time order) and then the depacketised frames from
    /// `payload`. Rejects `payload` for the same reasons as
    /// [`Self::depacketise`] — empty or not a multiple of the
    /// per-mode frame size.
    pub fn depacketise_with_gap_fill(
        &self,
        gap_packets: usize,
        frames_per_payload: usize,
        payload: &[u8],
    ) -> Result<Vec<Vec<u8>>> {
        let live = self.depacketise_owned(payload)?;
        let missing = self.gap_frame_count(gap_packets, frames_per_payload);
        if missing == 0 {
            return Ok(live);
        }
        let mut out = Vec::with_capacity(missing + live.len());
        out.extend(self.conceal_gap(missing));
        out.extend(live);
        Ok(out)
    }
}

/// Difference between two 16-bit RTP sequence numbers, accounting for
/// the 65 536-element circular space defined by RFC 3550 §3.3.
/// Returns the number of *missing* sequence numbers strictly between
/// `last` and `now` (i.e. `(now - last - 1) mod 2^16`, interpreted
/// as the shorter of the two arc directions on the circle).
///
/// The convention treats deltas larger than `2^15` as backward jumps
/// (out-of-order delivery or a deliberate sequence reset) and reports
/// `0` for them — concealing a frame for a packet that arrived early
/// would lower output quality with no upside. Callers that want to
/// react to backward jumps should compare `last` and `now` directly
/// before calling this helper.
///
/// Returns `0` when `now == last` (duplicate) or `now == last + 1`
/// (in-order delivery, no gap).
pub fn rtp_seq_gap(last: u16, now: u16) -> usize {
    let delta = now.wrapping_sub(last) as i32;
    // Wrap delta into signed 16-bit space: values in (2^15, 2^16) are
    // backward jumps.
    let signed = if delta >= 0x8000 {
        delta - 0x1_0000
    } else {
        delta
    };
    if signed <= 1 {
        0
    } else {
        (signed - 1) as usize
    }
}

/// Build an RTP payload from one or more iLBC frames.
///
/// All frames MUST be the same mode (RFC 3952 §3 — every frame in a
/// packet shares the session mode); the packetiser rejects any
/// length that does not match its pinned mode. Returns the
/// concatenated bytes ready to hand to the RTP sender (which will
/// then prepend the 12-byte fixed RTP header).
#[derive(Clone, Debug)]
pub struct Packetiser {
    mode: FrameMode,
    max_frames_per_packet: usize,
}

impl Packetiser {
    /// Construct a packetiser pinned to the supplied mode with a
    /// default cap of 8 iLBC frames per packet (the largest count
    /// that fits comfortably inside an IPv4 RTP packet over a
    /// 1500-byte MTU after subtracting the RTP / UDP / IP overhead).
    pub fn new(mode: FrameMode) -> Self {
        Self::with_max_frames_per_packet(mode, 8)
    }

    /// Construct a packetiser with an explicit per-packet frame cap.
    pub fn with_max_frames_per_packet(mode: FrameMode, max_frames_per_packet: usize) -> Self {
        Self {
            mode,
            max_frames_per_packet: max_frames_per_packet.max(1),
        }
    }

    /// Mode this packetiser emits.
    pub fn mode(&self) -> FrameMode {
        self.mode
    }

    /// Number of bytes per emitted iLBC frame.
    pub fn frame_size(&self) -> usize {
        self.mode.bytes()
    }

    /// RTP timestamp delta per emitted iLBC frame.
    pub fn ts_delta(&self) -> u32 {
        match self.mode {
            FrameMode::Ms20 => TS_DELTA_20MS,
            FrameMode::Ms30 => TS_DELTA_30MS,
        }
    }

    /// Maximum number of iLBC frames per emitted RTP payload.
    pub fn max_frames_per_packet(&self) -> usize {
        self.max_frames_per_packet
    }

    /// Pack one or more iLBC frames into a single RTP payload. The
    /// returned `Vec<u8>` is the body of *one* RTP packet — the
    /// caller is responsible for not exceeding the path MTU.
    pub fn pack_single(&self, frames: &[&[u8]]) -> Result<Vec<u8>> {
        if frames.is_empty() {
            return Err(Error::invalid(
                "iLBC RTP packetiser: at least one iLBC frame is required",
            ));
        }
        if frames.len() > self.max_frames_per_packet {
            return Err(Error::invalid(
                "iLBC RTP packetiser: frame count exceeds max_frames_per_packet",
            ));
        }
        let fs = self.frame_size();
        let mut out = Vec::with_capacity(fs * frames.len());
        for (i, f) in frames.iter().enumerate() {
            if f.len() != fs {
                let _ = i; // silenced — message is enough
                return Err(Error::invalid(
                    "iLBC RTP packetiser: frame length does not match the pinned mode",
                ));
            }
            out.extend_from_slice(f);
        }
        Ok(out)
    }

    /// Pack a series of iLBC frames into as many RTP payloads as
    /// needed, capped at `max_frames_per_packet` frames each.
    /// Returns `(payload, ts_offset)` pairs where `ts_offset` is the
    /// RTP-timestamp delta of the first frame in `payload` relative
    /// to the first frame of the whole input series — the caller
    /// adds this to its base RTP timestamp to label each emitted
    /// packet.
    pub fn pack_series(&self, frames: &[&[u8]]) -> Result<Vec<(Vec<u8>, u32)>> {
        let fs = self.frame_size();
        for f in frames {
            if f.len() != fs {
                return Err(Error::invalid(
                    "iLBC RTP packetiser: frame length does not match the pinned mode",
                ));
            }
        }
        let cap = self.max_frames_per_packet;
        let per_frame_ts = self.ts_delta();
        let mut out = Vec::with_capacity(frames.len().div_ceil(cap));
        let mut ts_offset: u32 = 0;
        for chunk in frames.chunks(cap) {
            let body = self.pack_single(chunk)?;
            out.push((body, ts_offset));
            ts_offset = ts_offset.wrapping_add(per_frame_ts * chunk.len() as u32);
        }
        Ok(out)
    }
}

/// Sentinel byte used to mark an iLBC frame as "empty" so the
/// decoder runs PLC instead of trusting the payload. RFC 3951 §3.8
/// places the empty-frame indicator at the LSB of the last byte;
/// for an entirely-PLC slot we hand the decoder a buffer that is
/// all-zero except for that LSB set to 1. This is the conventional
/// "lost packet" surrogate when an RTP packet does not arrive.
pub fn empty_marker_frame(mode: FrameMode) -> Vec<u8> {
    let mut buf = vec![0u8; mode.bytes()];
    *buf.last_mut().expect("non-empty frame buffer") = 0x01;
    buf
}

/// Parse the SDP `a=fmtp:<pt> ...` value (the substring after the
/// payload-type token) and extract the `mode=20|30` parameter per
/// RFC 3952 §4.2. The parameter list is `;`-separated and
/// case-insensitive on parameter keys; the `mode` *value* is one of
/// the bare integers `20` or `30`.
///
/// Returns an error when the parameter is missing or carries an
/// unsupported value (e.g. `mode=10`, `mode=40`, or non-numeric).
pub fn parse_mode_from_fmtp(fmtp_value: &str) -> Result<FrameMode> {
    for piece in fmtp_value.split(';') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let (k, v) = match piece.split_once('=') {
            Some(kv) => kv,
            None => continue,
        };
        if !k.trim().eq_ignore_ascii_case("mode") {
            continue;
        }
        return match v.trim() {
            "20" => Ok(FrameMode::Ms20),
            "30" => Ok(FrameMode::Ms30),
            _ => Err(Error::invalid(
                "iLBC SDP fmtp: 'mode' value must be 20 or 30 (RFC 3952 §4.2)",
            )),
        };
    }
    Err(Error::invalid(
        "iLBC SDP fmtp: missing 'mode' parameter (RFC 3952 §4.2 pins the session mode)",
    ))
}

/// Detect the mode of a candidate iLBC RTP payload from its byte
/// length alone. Useful when an interop test rig has lost track of
/// the SDP-pinned mode but the payload is known to contain `k * 38`
/// or `k * 50` bytes for some k ≥ 1. Returns `None` when the length
/// is ambiguous (the only mutually-ambiguous lengths in the
/// realistic packet-size range are `0` and the LCM `38*50=1900`
/// bytes; both are reported as `None`).
pub fn detect_mode_from_payload_len(payload_len: usize) -> Option<FrameMode> {
    if payload_len == 0 {
        return None;
    }
    let by_20 = payload_len % FRAME_BYTES_20MS == 0;
    let by_30 = payload_len % FRAME_BYTES_30MS == 0;
    match (by_20, by_30) {
        (true, false) => Some(FrameMode::Ms20),
        (false, true) => Some(FrameMode::Ms30),
        _ => None,
    }
}

/// Per-frame duration in milliseconds. 20 ms for [`FrameMode::Ms20`]
/// and 30 ms for [`FrameMode::Ms30`]; this is also the
/// recommended-`ptime` (packetisation time) value when only one
/// iLBC frame is carried per RTP packet, and the building block of
/// `maxptime` when more than one frame is aggregated. RFC 3952 §4.2
/// pins one frame mode for the whole session, so the per-frame
/// duration is constant for the session's lifetime.
pub fn frame_duration_ms(mode: FrameMode) -> u32 {
    match mode {
        FrameMode::Ms20 => 20,
        FrameMode::Ms30 => 30,
    }
}

/// Emit just the `mode=20` / `mode=30` token for an SDP `a=fmtp:<pt>`
/// parameter list, per RFC 3952 §4.2. The output is the inverse of
/// [`parse_mode_from_fmtp`] on the same value and is intended as a
/// building block for callers stitching together an outbound
/// `a=fmtp:<pt> ...` value from several parameters; the bare key=value
/// pair carries no leading or trailing whitespace and no `;`
/// terminator, so the caller can join several parameters with
/// `;` separators of its choice.
pub fn format_mode_fmtp(mode: FrameMode) -> String {
    format!("mode={}", frame_duration_ms(mode))
}

/// Emit the body (everything after the `a=fmtp:<pt> ` token) of an
/// outbound SDP `fmtp` attribute pinning the iLBC session to the
/// supplied mode and optional per-packet aggregation cap.
///
/// The emitted parameter list is always `mode=N` where `N` is `20` or
/// `30`, optionally followed by `;maxptime=M` when
/// `max_frames_per_packet` is supplied and greater than 1. RFC 3952
/// §4.2 names `mode` as the only iLBC-specific fmtp parameter; the
/// `maxptime` attribute (RFC 4566 §6) is a session-level cap on
/// packetisation time and is honoured by SDP-aware receivers as a
/// hint for the maximum frame count per packet (`maxptime` /
/// per-frame ms).
///
/// The output is the inverse of [`Depacketiser::from_sdp_fmtp`] +
/// [`parse_mode_from_fmtp`]: an `fmtp` value emitted here round-trips
/// back to the same [`FrameMode`] without any whitespace shaping.
pub fn build_fmtp(mode: FrameMode, max_frames_per_packet: Option<usize>) -> String {
    let mut out = format_mode_fmtp(mode);
    if let Some(n) = max_frames_per_packet {
        if n > 1 {
            let cap_ms = (n as u32).saturating_mul(frame_duration_ms(mode));
            out.push_str(&format!(";maxptime={cap_ms}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameMode;

    // ----- mode parsing -----

    #[test]
    fn parse_mode_accepts_20_and_30() {
        assert_eq!(parse_mode_from_fmtp("mode=20").unwrap(), FrameMode::Ms20);
        assert_eq!(parse_mode_from_fmtp("mode=30").unwrap(), FrameMode::Ms30);
    }

    #[test]
    fn parse_mode_is_case_insensitive_on_key() {
        assert_eq!(parse_mode_from_fmtp("MODE=20").unwrap(), FrameMode::Ms20);
        assert_eq!(parse_mode_from_fmtp("Mode=30").unwrap(), FrameMode::Ms30);
    }

    #[test]
    fn parse_mode_skips_unrelated_parameters() {
        assert_eq!(
            parse_mode_from_fmtp("ptime=20;mode=30;maxptime=240").unwrap(),
            FrameMode::Ms30,
        );
    }

    #[test]
    fn parse_mode_trims_whitespace() {
        assert_eq!(
            parse_mode_from_fmtp(" mode = 20 ; ptime=20 ").unwrap(),
            FrameMode::Ms20,
        );
    }

    #[test]
    fn parse_mode_rejects_unknown_value() {
        assert!(parse_mode_from_fmtp("mode=10").is_err());
        assert!(parse_mode_from_fmtp("mode=40").is_err());
        assert!(parse_mode_from_fmtp("mode=hello").is_err());
    }

    #[test]
    fn parse_mode_rejects_missing_parameter() {
        assert!(parse_mode_from_fmtp("").is_err());
        assert!(parse_mode_from_fmtp("ptime=20").is_err());
        assert!(parse_mode_from_fmtp("ptime=20;maxptime=240").is_err());
    }

    // ----- length-based mode detection -----

    #[test]
    fn detect_mode_picks_20ms_for_pure_38_multiples() {
        for k in 1..=8 {
            let want = if (k * FRAME_BYTES_20MS) % FRAME_BYTES_30MS == 0 {
                None
            } else {
                Some(FrameMode::Ms20)
            };
            assert_eq!(
                detect_mode_from_payload_len(k * FRAME_BYTES_20MS),
                want,
                "k={k}",
            );
        }
    }

    #[test]
    fn detect_mode_picks_30ms_for_pure_50_multiples() {
        for k in 1..=8 {
            let want = if (k * FRAME_BYTES_30MS) % FRAME_BYTES_20MS == 0 {
                None
            } else {
                Some(FrameMode::Ms30)
            };
            assert_eq!(
                detect_mode_from_payload_len(k * FRAME_BYTES_30MS),
                want,
                "k={k}",
            );
        }
    }

    #[test]
    fn detect_mode_returns_none_on_empty_or_ambiguous() {
        assert_eq!(detect_mode_from_payload_len(0), None);
        // 1900 = 38*50 = 50*38 — the smallest mutually-ambiguous
        // payload length above zero. A real receiver hits this only
        // at unusually large per-packet aggregations; we report
        // None so callers fall back to the SDP-pinned mode.
        assert_eq!(detect_mode_from_payload_len(1900), None);
        // Non-multiples are also rejected.
        assert_eq!(detect_mode_from_payload_len(37), None);
        assert_eq!(detect_mode_from_payload_len(39), None);
        assert_eq!(detect_mode_from_payload_len(51), None);
    }

    // ----- depacketiser: single-frame payloads -----

    #[test]
    fn depacketise_single_20ms_frame() {
        let depk = Depacketiser::new(FrameMode::Ms20);
        let payload = vec![0u8; FRAME_BYTES_20MS];
        let frames = depk.depacketise(&payload).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), FRAME_BYTES_20MS);
        assert_eq!(depk.frame_count(payload.len()), Some(1));
    }

    #[test]
    fn depacketise_single_30ms_frame() {
        let depk = Depacketiser::new(FrameMode::Ms30);
        let payload = vec![0u8; FRAME_BYTES_30MS];
        let frames = depk.depacketise(&payload).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), FRAME_BYTES_30MS);
    }

    // ----- depacketiser: multi-frame payloads -----

    #[test]
    fn depacketise_three_20ms_frames_per_packet() {
        let depk = Depacketiser::new(FrameMode::Ms20);
        let mut payload = Vec::new();
        for i in 0..3u8 {
            // Per-frame body distinct so we can tell them apart.
            let mut f = vec![i; FRAME_BYTES_20MS];
            *f.last_mut().unwrap() = 0; // clear empty-frame bit
            payload.extend_from_slice(&f);
        }
        let frames = depk.depacketise(&payload).unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0][0], 0);
        assert_eq!(frames[1][0], 1);
        assert_eq!(frames[2][0], 2);
        assert_eq!(depk.frame_count(payload.len()), Some(3));
        assert_eq!(depk.ts_delta(), TS_DELTA_20MS);
    }

    #[test]
    fn depacketise_four_30ms_frames_per_packet() {
        let depk = Depacketiser::new(FrameMode::Ms30);
        let payload = vec![0u8; 4 * FRAME_BYTES_30MS];
        let frames = depk.depacketise(&payload).unwrap();
        assert_eq!(frames.len(), 4);
        for f in &frames {
            assert_eq!(f.len(), FRAME_BYTES_30MS);
        }
        assert_eq!(depk.ts_delta(), TS_DELTA_30MS);
    }

    // ----- depacketiser: rejection paths -----

    #[test]
    fn depacketise_rejects_empty_payload() {
        let depk = Depacketiser::new(FrameMode::Ms20);
        assert!(depk.depacketise(&[]).is_err());
        assert_eq!(depk.frame_count(0), None);
    }

    #[test]
    fn depacketise_rejects_short_payload() {
        let depk = Depacketiser::new(FrameMode::Ms20);
        let payload = vec![0u8; FRAME_BYTES_20MS - 1];
        assert!(depk.depacketise(&payload).is_err());
    }

    #[test]
    fn depacketise_rejects_payload_not_a_multiple_of_frame_size() {
        let depk = Depacketiser::new(FrameMode::Ms20);
        let payload = vec![0u8; FRAME_BYTES_20MS + 1];
        assert!(depk.depacketise(&payload).is_err());
    }

    #[test]
    fn depacketise_rejects_30ms_size_on_20ms_session() {
        let depk = Depacketiser::new(FrameMode::Ms20);
        let payload = vec![0u8; FRAME_BYTES_30MS];
        assert!(depk.depacketise(&payload).is_err());
    }

    // ----- depacketiser: owned variant -----

    #[test]
    fn depacketise_owned_returns_independent_buffers() {
        let depk = Depacketiser::new(FrameMode::Ms20);
        let payload = vec![0xABu8; 2 * FRAME_BYTES_20MS];
        let frames = depk.depacketise_owned(&payload).unwrap();
        assert_eq!(frames.len(), 2);
        for f in &frames {
            assert_eq!(f.len(), FRAME_BYTES_20MS);
            assert_eq!(f[0], 0xAB);
        }
    }

    // ----- SDP-driven depacketiser construction -----

    #[test]
    fn depacketiser_from_sdp_picks_mode() {
        let d20 = Depacketiser::from_sdp_fmtp("mode=20;ptime=20").unwrap();
        assert_eq!(d20.mode(), FrameMode::Ms20);
        assert_eq!(d20.frame_size(), FRAME_BYTES_20MS);
        let d30 = Depacketiser::from_sdp_fmtp("mode=30;ptime=30").unwrap();
        assert_eq!(d30.mode(), FrameMode::Ms30);
        assert_eq!(d30.frame_size(), FRAME_BYTES_30MS);
    }

    #[test]
    fn depacketiser_from_sdp_propagates_missing_mode() {
        assert!(Depacketiser::from_sdp_fmtp("ptime=20").is_err());
    }

    // ----- packetiser -----

    #[test]
    fn packetiser_packs_single_frame() {
        let pkt = Packetiser::new(FrameMode::Ms20);
        let f = vec![0u8; FRAME_BYTES_20MS];
        let body = pkt.pack_single(&[&f]).unwrap();
        assert_eq!(body.len(), FRAME_BYTES_20MS);
    }

    #[test]
    fn packetiser_packs_multiple_frames() {
        let pkt = Packetiser::new(FrameMode::Ms30);
        let f1 = vec![1u8; FRAME_BYTES_30MS];
        let f2 = vec![2u8; FRAME_BYTES_30MS];
        let body = pkt.pack_single(&[&f1, &f2]).unwrap();
        assert_eq!(body.len(), 2 * FRAME_BYTES_30MS);
        assert_eq!(body[0], 1);
        assert_eq!(body[FRAME_BYTES_30MS], 2);
    }

    #[test]
    fn packetiser_rejects_empty_input() {
        let pkt = Packetiser::new(FrameMode::Ms20);
        assert!(pkt.pack_single(&[]).is_err());
    }

    #[test]
    fn packetiser_rejects_wrong_size_frame() {
        let pkt = Packetiser::new(FrameMode::Ms20);
        let bad = vec![0u8; FRAME_BYTES_30MS];
        assert!(pkt.pack_single(&[&bad]).is_err());
    }

    #[test]
    fn packetiser_respects_max_frames_per_packet_cap() {
        let pkt = Packetiser::with_max_frames_per_packet(FrameMode::Ms20, 2);
        let f = vec![0u8; FRAME_BYTES_20MS];
        // 2 frames OK, 3 frames over cap.
        assert!(pkt.pack_single(&[&f, &f]).is_ok());
        assert!(pkt.pack_single(&[&f, &f, &f]).is_err());
    }

    #[test]
    fn packetiser_pack_series_chunks_at_cap_with_timestamps() {
        let pkt = Packetiser::with_max_frames_per_packet(FrameMode::Ms20, 2);
        let f = vec![0u8; FRAME_BYTES_20MS];
        let frames: Vec<&[u8]> = (0..5).map(|_| f.as_slice()).collect();
        let series = pkt.pack_series(&frames).unwrap();
        // 5 frames at cap=2 → 3 packets sized 2 + 2 + 1.
        assert_eq!(series.len(), 3);
        assert_eq!(series[0].0.len(), 2 * FRAME_BYTES_20MS);
        assert_eq!(series[1].0.len(), 2 * FRAME_BYTES_20MS);
        assert_eq!(series[2].0.len(), FRAME_BYTES_20MS);
        // Timestamps step by frames-in-prior-packet × 160 samples.
        assert_eq!(series[0].1, 0);
        assert_eq!(series[1].1, 2 * TS_DELTA_20MS);
        assert_eq!(series[2].1, 4 * TS_DELTA_20MS);
    }

    #[test]
    fn packetiser_with_zero_cap_clamps_to_one() {
        // Defensive: cap=0 would deadlock pack_series; we clamp.
        let pkt = Packetiser::with_max_frames_per_packet(FrameMode::Ms20, 0);
        assert_eq!(pkt.max_frames_per_packet(), 1);
    }

    // ----- round-trip: pack then depacketise -----

    #[test]
    fn roundtrip_20ms_pack_then_depacketise() {
        let pk = Packetiser::with_max_frames_per_packet(FrameMode::Ms20, 4);
        let dk = Depacketiser::new(FrameMode::Ms20);
        let frames: Vec<Vec<u8>> = (0..7)
            .map(|i| {
                let mut f = vec![i as u8; FRAME_BYTES_20MS];
                *f.last_mut().unwrap() = 0;
                f
            })
            .collect();
        let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
        let series = pk.pack_series(&refs).unwrap();
        // 7 frames @ cap=4 → packets of 4 + 3.
        assert_eq!(series.len(), 2);
        // Concatenated, the depacketised frames must match the
        // original frames in order.
        let mut got: Vec<Vec<u8>> = Vec::new();
        for (body, _ts) in &series {
            for f in dk.depacketise(body).unwrap() {
                got.push(f.to_vec());
            }
        }
        assert_eq!(got, frames);
    }

    #[test]
    fn roundtrip_30ms_pack_then_depacketise() {
        let pk = Packetiser::new(FrameMode::Ms30);
        let dk = Depacketiser::new(FrameMode::Ms30);
        let f1 = vec![0xAAu8; FRAME_BYTES_30MS];
        let f2 = vec![0x55u8; FRAME_BYTES_30MS];
        let body = pk.pack_single(&[&f1, &f2]).unwrap();
        let frames = dk.depacketise_owned(&body).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], f1);
        assert_eq!(frames[1], f2);
    }

    // ----- empty-marker helper -----

    #[test]
    fn empty_marker_frame_sets_only_last_bit() {
        let m20 = empty_marker_frame(FrameMode::Ms20);
        assert_eq!(m20.len(), FRAME_BYTES_20MS);
        assert_eq!(m20[m20.len() - 1], 0x01);
        for b in &m20[..m20.len() - 1] {
            assert_eq!(*b, 0);
        }
        let m30 = empty_marker_frame(FrameMode::Ms30);
        assert_eq!(m30.len(), FRAME_BYTES_30MS);
        assert_eq!(m30[m30.len() - 1], 0x01);
    }

    #[test]
    fn empty_marker_passes_through_depacketiser() {
        let dk = Depacketiser::new(FrameMode::Ms20);
        let m = empty_marker_frame(FrameMode::Ms20);
        let frames = dk.depacketise(&m).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], m.as_slice());
    }

    // ----- per-frame duration helper -----

    #[test]
    fn frame_duration_matches_mode() {
        assert_eq!(frame_duration_ms(FrameMode::Ms20), 20);
        assert_eq!(frame_duration_ms(FrameMode::Ms30), 30);
    }

    // ----- outbound SDP fmtp builders -----

    #[test]
    fn format_mode_emits_bare_key_value() {
        assert_eq!(format_mode_fmtp(FrameMode::Ms20), "mode=20");
        assert_eq!(format_mode_fmtp(FrameMode::Ms30), "mode=30");
    }

    #[test]
    fn format_mode_round_trips_through_parse() {
        for m in [FrameMode::Ms20, FrameMode::Ms30] {
            let s = format_mode_fmtp(m);
            assert_eq!(parse_mode_from_fmtp(&s).unwrap(), m);
        }
    }

    #[test]
    fn build_fmtp_without_cap_emits_just_mode() {
        assert_eq!(build_fmtp(FrameMode::Ms20, None), "mode=20");
        assert_eq!(build_fmtp(FrameMode::Ms30, None), "mode=30");
        // Cap of 1 collapses to "just mode=N" — maxptime equal to one
        // per-frame ptime would be a no-op for an SDP receiver.
        assert_eq!(build_fmtp(FrameMode::Ms20, Some(1)), "mode=20");
        assert_eq!(build_fmtp(FrameMode::Ms30, Some(1)), "mode=30");
    }

    #[test]
    fn build_fmtp_with_cap_emits_maxptime_in_ms() {
        // 4 × 20 ms = 80 ms / 4 × 30 ms = 120 ms.
        assert_eq!(build_fmtp(FrameMode::Ms20, Some(4)), "mode=20;maxptime=80");
        assert_eq!(build_fmtp(FrameMode::Ms30, Some(4)), "mode=30;maxptime=120");
        // Default packetiser cap of 8.
        assert_eq!(build_fmtp(FrameMode::Ms20, Some(8)), "mode=20;maxptime=160");
        assert_eq!(build_fmtp(FrameMode::Ms30, Some(8)), "mode=30;maxptime=240");
    }

    #[test]
    fn build_fmtp_round_trips_with_parser_and_packetiser_cap() {
        // build → parse round-trips the mode; the parser ignores
        // maxptime (it's not iLBC-specific).
        for m in [FrameMode::Ms20, FrameMode::Ms30] {
            for cap in [None, Some(1), Some(2), Some(8)] {
                let s = build_fmtp(m, cap);
                assert_eq!(parse_mode_from_fmtp(&s).unwrap(), m, "fmtp={s:?}");
                // The depacketiser construction path is also happy.
                let dk = Depacketiser::from_sdp_fmtp(&s).unwrap();
                assert_eq!(dk.mode(), m);
            }
        }
    }

    // ----- dropped-frame concealment helpers -----

    #[test]
    fn conceal_gap_emits_n_marker_frames_per_mode() {
        for (mode, fs) in [
            (FrameMode::Ms20, FRAME_BYTES_20MS),
            (FrameMode::Ms30, FRAME_BYTES_30MS),
        ] {
            let dk = Depacketiser::new(mode);
            for n in [0_usize, 1, 3, 8] {
                let frames = dk.conceal_gap(n);
                assert_eq!(frames.len(), n, "mode={mode:?} n={n}");
                for f in &frames {
                    assert_eq!(f.len(), fs);
                    // Last byte LSB set, every other byte zero (RFC 3951 §3.8).
                    assert_eq!(*f.last().unwrap(), 0x01);
                    for b in &f[..f.len() - 1] {
                        assert_eq!(*b, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn concealment_payload_matches_concatenated_marker_frames() {
        for mode in [FrameMode::Ms20, FrameMode::Ms30] {
            let dk = Depacketiser::new(mode);
            // Zero-gap shortcut returns None.
            assert!(dk.concealment_payload(0).is_none());

            for n in [1_usize, 2, 5] {
                let payload = dk.concealment_payload(n).expect("non-empty for n>=1");
                let split = dk.depacketise_owned(&payload).unwrap();
                assert_eq!(split.len(), n);
                let expected = dk.conceal_gap(n);
                assert_eq!(split, expected);
            }
        }
    }

    #[test]
    fn gap_frame_count_scales_with_steady_aggregation() {
        let dk = Depacketiser::new(FrameMode::Ms20);
        assert_eq!(dk.gap_frame_count(0, 4), 0);
        assert_eq!(dk.gap_frame_count(3, 0), 0);
        assert_eq!(dk.gap_frame_count(1, 1), 1);
        assert_eq!(dk.gap_frame_count(2, 4), 8);
        // Saturating multiplication avoids overflow panics for
        // pathologically large inputs.
        assert_eq!(dk.gap_frame_count(usize::MAX, 8), usize::MAX);
    }

    #[test]
    fn depacketise_with_gap_fill_prefixes_marker_frames() {
        let dk = Depacketiser::new(FrameMode::Ms20);
        let live = vec![0xAAu8; 2 * FRAME_BYTES_20MS];
        // 2 missing packets at 2 frames each → 4 PLC frames + 2 live.
        let frames = dk.depacketise_with_gap_fill(2, 2, &live).unwrap();
        assert_eq!(frames.len(), 6);
        // First four are empty-marker frames (LSB of last byte set,
        // every other byte zero).
        for f in &frames[..4] {
            assert_eq!(f.len(), FRAME_BYTES_20MS);
            assert_eq!(*f.last().unwrap(), 0x01);
            for b in &f[..f.len() - 1] {
                assert_eq!(*b, 0);
            }
        }
        // Last two are the live payload.
        for f in &frames[4..] {
            assert_eq!(f.len(), FRAME_BYTES_20MS);
            assert_eq!(f[0], 0xAA);
        }
    }

    #[test]
    fn depacketise_with_gap_fill_zero_gap_matches_owned_path() {
        let dk = Depacketiser::new(FrameMode::Ms30);
        let live = vec![0x55u8; 3 * FRAME_BYTES_30MS];
        let with_gap = dk.depacketise_with_gap_fill(0, 1, &live).unwrap();
        let no_gap = dk.depacketise_owned(&live).unwrap();
        assert_eq!(with_gap, no_gap);
        // Zero frames-per-payload also degenerates cleanly.
        let with_zero = dk.depacketise_with_gap_fill(5, 0, &live).unwrap();
        assert_eq!(with_zero, no_gap);
    }

    #[test]
    fn depacketise_with_gap_fill_rejects_malformed_payload() {
        let dk = Depacketiser::new(FrameMode::Ms20);
        // Empty payload — same rejection rule as depacketise.
        assert!(dk.depacketise_with_gap_fill(1, 1, &[]).is_err());
        // Wrong-mode size.
        let bad = vec![0u8; FRAME_BYTES_30MS];
        assert!(dk.depacketise_with_gap_fill(1, 1, &bad).is_err());
    }

    #[test]
    fn depacketise_with_gap_fill_round_trips_through_decoder_state() {
        // The gap-fill output is contiguous concealment frames followed by
        // the live payload's frames in time order — concretely, splitting
        // the concatenated bytes through depacketise again yields the
        // same frame list.
        let dk = Depacketiser::new(FrameMode::Ms20);
        let live = vec![0xC3u8; 2 * FRAME_BYTES_20MS];
        let frames = dk.depacketise_with_gap_fill(3, 2, &live).unwrap();
        let concat: Vec<u8> = frames.iter().flatten().copied().collect();
        // 3 gap × 2 frames + 2 live = 8 frames × 38 bytes = 304 bytes.
        assert_eq!(concat.len(), 8 * FRAME_BYTES_20MS);
        let re = dk.depacketise_owned(&concat).unwrap();
        assert_eq!(re, frames);
    }

    // ----- RTP sequence-number gap arithmetic -----

    #[test]
    fn rtp_seq_gap_in_order_or_duplicate_is_zero() {
        assert_eq!(rtp_seq_gap(100, 101), 0); // in-order
        assert_eq!(rtp_seq_gap(100, 100), 0); // duplicate
    }

    #[test]
    fn rtp_seq_gap_counts_missing_packets() {
        assert_eq!(rtp_seq_gap(100, 102), 1);
        assert_eq!(rtp_seq_gap(100, 105), 4);
        assert_eq!(rtp_seq_gap(0, 1000), 999);
    }

    #[test]
    fn rtp_seq_gap_wraps_around_16_bit_boundary() {
        // last=65530, now=5 → wrapped delta = 11, gap = 10.
        assert_eq!(rtp_seq_gap(65_530, 5), 10);
        // last=65535, now=0 → in-order across wrap.
        assert_eq!(rtp_seq_gap(65_535, 0), 0);
        // last=65535, now=1 → 1 missing.
        assert_eq!(rtp_seq_gap(65_535, 1), 1);
    }

    #[test]
    fn rtp_seq_gap_backward_jump_reports_zero() {
        // Out-of-order delivery: now < last by less than the wrap
        // midpoint → backward jump → no concealment.
        assert_eq!(rtp_seq_gap(100, 50), 0);
        // A backward jump of 1 position (re-ordered duplicate of the
        // previous packet) also reports zero.
        assert_eq!(rtp_seq_gap(1_000, 999), 0);
        // A wrap-aware backward jump (last just past the wrap, now
        // just before it) — `now - last` mod 2^16 is in the upper
        // half, which the RFC 3550 wrap convention classifies as
        // out-of-order rather than a near-complete-roll forward.
        assert_eq!(rtp_seq_gap(5, 65_000), 0);
    }

    #[test]
    fn rtp_seq_gap_chains_into_depacketiser_with_gap_fill() {
        let dk = Depacketiser::new(FrameMode::Ms20);
        let last_seq = 65_534_u16;
        let now_seq = 1_u16; // wraps; gap = 2 missing packets.
        let gap = rtp_seq_gap(last_seq, now_seq);
        assert_eq!(gap, 2);

        let live = vec![0x77u8; FRAME_BYTES_20MS];
        let frames = dk.depacketise_with_gap_fill(gap, 1, &live).unwrap();
        // 2 missing × 1 frame/payload + 1 live = 3 total.
        assert_eq!(frames.len(), 3);
        // First two are empty-marker, third is the live payload.
        for f in &frames[..2] {
            assert_eq!(*f.last().unwrap(), 0x01);
        }
        assert_eq!(frames[2][0], 0x77);
    }
}
