//! iLBC storage format — the de-facto `#!iLBC{20,30}\n` on-disk framing
//! (RFC 3951 §5 "storage format", reference [1]).
//!
//! An iLBC file on disk is a 9-byte ASCII magic header followed by a run
//! of fixed-size frames. The magic pins the frame mode for the whole
//! file (there is no per-frame mode indicator on the wire):
//!
//! - `#!iLBC20\n` → 20 ms mode → 38-byte frames.
//! - `#!iLBC30\n` → 30 ms mode → 50-byte frames.
//!
//! This magic is not carried inside the RFC 3951 payload bitstream
//! itself; it is the container convention every `.lbc` file in the wild
//! uses so that a bare decoder can recover the mode from the file alone.
//! Because the magic pins one mode per file, mid-stream mode changes are
//! out of scope (they require RTP/SDP renegotiation — see [`crate::rtp`]).
//!
//! ```
//! use oxideav_ilbc::{storage, FrameMode};
//!
//! // Two 20 ms frames + magic.
//! let mut file = b"#!iLBC20\n".to_vec();
//! file.extend(std::iter::repeat_n(0u8, 2 * 38));
//!
//! let parsed = storage::parse(&file).unwrap();
//! assert_eq!(parsed.mode, FrameMode::Ms20);
//! assert_eq!(parsed.frames().count(), 2);
//! ```

use crate::{FrameMode, FRAME_BYTES_20MS, FRAME_BYTES_30MS};
use oxideav_core::Error;

/// 9-byte storage-format magic for 20 ms mode.
pub const MAGIC_20MS: &[u8; 9] = b"#!iLBC20\n";
/// 9-byte storage-format magic for 30 ms mode.
pub const MAGIC_30MS: &[u8; 9] = b"#!iLBC30\n";
/// Length of the storage-format magic header (identical for both modes).
pub const MAGIC_LEN: usize = 9;

/// The storage magic that pins a given [`FrameMode`].
pub fn magic_for(mode: FrameMode) -> &'static [u8; 9] {
    match mode {
        FrameMode::Ms20 => MAGIC_20MS,
        FrameMode::Ms30 => MAGIC_30MS,
    }
}

/// Detect the frame mode implied by a storage-format magic header
/// without consuming or validating the frame body. Returns `None` when
/// the buffer does not start with a recognised `#!iLBC{20,30}\n` magic.
pub fn detect_mode(buf: &[u8]) -> Option<FrameMode> {
    if buf.starts_with(MAGIC_20MS) {
        Some(FrameMode::Ms20)
    } else if buf.starts_with(MAGIC_30MS) {
        Some(FrameMode::Ms30)
    } else {
        None
    }
}

/// A parsed storage-format file: the pinned [`FrameMode`] plus a borrow
/// of the frame body (everything after the 9-byte magic).
///
/// The body length is validated to be a whole multiple of the mode's
/// frame size at [`parse`] time, so [`StorageFile::frames`] never yields
/// a short trailing chunk.
#[derive(Clone, Copy, Debug)]
pub struct StorageFile<'a> {
    /// Frame mode pinned by the magic header.
    pub mode: FrameMode,
    /// The frame body — concatenated fixed-size frames, magic stripped.
    body: &'a [u8],
}

impl<'a> StorageFile<'a> {
    /// Byte size of one frame in this file's mode (38 or 50).
    pub fn frame_size(&self) -> usize {
        self.mode.bytes()
    }

    /// Number of frames in the file.
    pub fn frame_count(&self) -> usize {
        self.body.len() / self.frame_size()
    }

    /// Iterate the fixed-size frame payloads, each ready to hand to the
    /// decoder verbatim.
    pub fn frames(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        self.body.chunks_exact(self.frame_size())
    }

    /// The raw frame body (magic stripped).
    pub fn body(&self) -> &'a [u8] {
        self.body
    }
}

/// Parse a storage-format `.lbc` buffer into its mode + frame body.
///
/// # Errors
/// - The buffer does not start with a `#!iLBC{20,30}\n` magic.
/// - The body length after the magic is not a whole multiple of the
///   mode's frame size (a truncated or corrupt file).
pub fn parse(buf: &[u8]) -> Result<StorageFile<'_>, Error> {
    let mode = detect_mode(buf).ok_or_else(|| {
        Error::invalid("iLBC storage: input lacks #!iLBC20\\n / #!iLBC30\\n magic header")
    })?;
    let body = &buf[MAGIC_LEN..];
    let frame_size = mode.bytes();
    if body.len() % frame_size != 0 {
        return Err(Error::invalid(format!(
            "iLBC storage: {mode:?} body length {} is not a multiple of frame size {frame_size}",
            body.len()
        )));
    }
    Ok(StorageFile { mode, body })
}

/// Serialise a run of frames into a storage-format buffer: the mode's
/// magic header followed by the concatenated frames.
///
/// # Errors
/// Returns an error if any frame's length does not match the mode's
/// fixed frame size.
pub fn write(mode: FrameMode, frames: &[&[u8]]) -> Result<Vec<u8>, Error> {
    let frame_size = mode.bytes();
    let mut out = Vec::with_capacity(MAGIC_LEN + frames.len() * frame_size);
    out.extend_from_slice(magic_for(mode));
    for (i, frame) in frames.iter().enumerate() {
        if frame.len() != frame_size {
            return Err(Error::invalid(format!(
                "iLBC storage: frame {i} has {} bytes, expected {frame_size} for {mode:?}",
                frame.len()
            )));
        }
        out.extend_from_slice(frame);
    }
    Ok(out)
}

/// Wrap an already-concatenated raw frame body with the mode's magic
/// header. The body length must be a whole multiple of the frame size.
///
/// # Errors
/// Returns an error if `body` is not a whole number of frames.
pub fn wrap_body(mode: FrameMode, body: &[u8]) -> Result<Vec<u8>, Error> {
    let frame_size = mode.bytes();
    if body.len() % frame_size != 0 {
        return Err(Error::invalid(format!(
            "iLBC storage: body length {} is not a multiple of frame size {frame_size} for {mode:?}",
            body.len()
        )));
    }
    let mut out = Vec::with_capacity(MAGIC_LEN + body.len());
    out.extend_from_slice(magic_for(mode));
    out.extend_from_slice(body);
    Ok(out)
}

/// One-shot decode of a storage-format `.lbc` buffer to interleaved
/// mono S16 PCM.
///
/// Parses the magic to recover the mode, then drives every frame through
/// the iLBC decoder (respecting the §3.8 empty-frame indicator — a
/// lost-marked frame is concealed, not decoded from garbage). Returns
/// the concatenated 16-bit samples in decode order.
///
/// This is the "read a file, get audio" convenience; callers wanting
/// per-frame control (timestamps, incremental output, mid-stream mode
/// changes) should drive [`crate::decoder::make_decoder`] directly.
///
/// # Errors
/// Propagates [`parse`] errors (bad magic / ragged body) and any decoder
/// error.
pub fn decode(buf: &[u8]) -> Result<Vec<i16>, Error> {
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};

    let sf = parse(buf)?;

    let mut params = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
    params.sample_rate = Some(crate::SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec = crate::decoder::make_decoder(&params)?;

    let tb = TimeBase::new(1, crate::SAMPLE_RATE as i64);
    let mut pcm: Vec<i16> = Vec::with_capacity(sf.frame_count() * sf.mode.samples());
    for (i, frame) in sf.frames().enumerate() {
        let pkt = Packet::new(0, tb, frame.to_vec()).with_pts((i * sf.mode.samples()) as i64);
        dec.send_packet(&pkt)?;
        if let Frame::Audio(a) = dec.receive_frame()? {
            for plane in &a.data {
                for chunk in plane.chunks_exact(2) {
                    pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
        }
    }
    Ok(pcm)
}

/// Bit-mask of the empty-frame indicator within the last payload byte.
///
/// RFC 3951 §3.8 packs the empty-frame indicator as the final (class-3)
/// bit of the payload, which lands at the LSB of the last byte. When it
/// is set the decoder MUST treat the frame as lost and run PLC. The RFC
/// notes this bit "can be set to 1 to indicate lost frame for file
/// storage format" — i.e. the storage form uses it as a per-frame
/// packet-loss marker.
pub const EMPTY_FRAME_INDICATOR_MASK: u8 = 0x01;

/// Report whether a frame carries the empty-frame indicator (LSB of its
/// last byte set), i.e. it is marked lost / to-be-concealed.
///
/// Returns `false` for an empty slice (no last byte to inspect).
pub fn is_lost(frame: &[u8]) -> bool {
    frame
        .last()
        .is_some_and(|&b| b & EMPTY_FRAME_INDICATOR_MASK != 0)
}

/// Return a copy of `frame` with the empty-frame indicator set, marking
/// it as a lost frame for the file storage format (RFC 3951 §3.8). The
/// decoder will conceal such a frame rather than trusting its payload,
/// so the remaining payload bits are irrelevant — but they are preserved
/// here so the operation is reversible with [`clear_lost`].
pub fn mark_lost(frame: &[u8]) -> Vec<u8> {
    let mut out = frame.to_vec();
    if let Some(last) = out.last_mut() {
        *last |= EMPTY_FRAME_INDICATOR_MASK;
    }
    out
}

/// Return a copy of `frame` with the empty-frame indicator cleared. Note
/// that the RFC recommends the encoder set this bit to zero anyway, so
/// on a well-formed non-lost frame this is a no-op.
pub fn clear_lost(frame: &[u8]) -> Vec<u8> {
    let mut out = frame.to_vec();
    if let Some(last) = out.last_mut() {
        *last &= !EMPTY_FRAME_INDICATOR_MASK;
    }
    out
}

/// Assert the two known frame sizes are exactly what the magic implies —
/// a compile-time-ish sanity that the module and [`crate::FrameMode`]
/// agree on sizing.
const _: () = {
    assert!(FRAME_BYTES_20MS == 38);
    assert!(FRAME_BYTES_30MS == 50);
};

#[cfg(test)]
mod tests {
    use super::*;

    fn body(mode: FrameMode, n: usize) -> Vec<u8> {
        let fs = mode.bytes();
        (0..n * fs).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn detect_mode_from_magic() {
        assert_eq!(detect_mode(MAGIC_20MS), Some(FrameMode::Ms20));
        assert_eq!(detect_mode(MAGIC_30MS), Some(FrameMode::Ms30));
        assert_eq!(detect_mode(b"#!iLBC99\n"), None);
        assert_eq!(detect_mode(b""), None);
        assert_eq!(detect_mode(b"#!iLBC2"), None);
    }

    #[test]
    fn parse_20ms_splits_frames() {
        let mut buf = MAGIC_20MS.to_vec();
        buf.extend(body(FrameMode::Ms20, 3));
        let sf = parse(&buf).unwrap();
        assert_eq!(sf.mode, FrameMode::Ms20);
        assert_eq!(sf.frame_size(), 38);
        assert_eq!(sf.frame_count(), 3);
        let frames: Vec<&[u8]> = sf.frames().collect();
        assert_eq!(frames.len(), 3);
        assert!(frames.iter().all(|f| f.len() == 38));
    }

    #[test]
    fn parse_30ms_splits_frames() {
        let mut buf = MAGIC_30MS.to_vec();
        buf.extend(body(FrameMode::Ms30, 4));
        let sf = parse(&buf).unwrap();
        assert_eq!(sf.mode, FrameMode::Ms30);
        assert_eq!(sf.frame_size(), 50);
        assert_eq!(sf.frame_count(), 4);
        assert_eq!(sf.frames().count(), 4);
    }

    #[test]
    fn parse_empty_body_is_zero_frames() {
        let sf = parse(MAGIC_20MS).unwrap();
        assert_eq!(sf.frame_count(), 0);
        assert_eq!(sf.frames().count(), 0);
    }

    #[test]
    fn parse_rejects_missing_magic() {
        let err = parse(b"no magic here at all").unwrap_err();
        assert!(format!("{err}").contains("magic"));
    }

    #[test]
    fn parse_rejects_partial_frame() {
        let mut buf = MAGIC_20MS.to_vec();
        buf.extend(body(FrameMode::Ms20, 2));
        buf.truncate(buf.len() - 5); // shave 5 bytes off the last frame
        let err = parse(&buf).unwrap_err();
        assert!(format!("{err}").contains("multiple"));
    }

    #[test]
    fn write_then_parse_roundtrips() {
        let raw = body(FrameMode::Ms30, 5);
        let frames: Vec<&[u8]> = raw.chunks_exact(50).collect();
        let file = write(FrameMode::Ms30, &frames).unwrap();
        assert_eq!(&file[..MAGIC_LEN], MAGIC_30MS);
        let sf = parse(&file).unwrap();
        assert_eq!(sf.mode, FrameMode::Ms30);
        assert_eq!(sf.frame_count(), 5);
        let back: Vec<&[u8]> = sf.frames().collect();
        assert_eq!(back, frames);
    }

    #[test]
    fn write_rejects_wrong_size_frame() {
        let good = vec![0u8; 38];
        let bad = vec![0u8; 37];
        let err = write(FrameMode::Ms20, &[&good, &bad]).unwrap_err();
        assert!(format!("{err}").contains("frame 1"));
    }

    #[test]
    fn wrap_body_matches_manual_concat() {
        let raw = body(FrameMode::Ms20, 3);
        let wrapped = wrap_body(FrameMode::Ms20, &raw).unwrap();
        let mut manual = MAGIC_20MS.to_vec();
        manual.extend_from_slice(&raw);
        assert_eq!(wrapped, manual);
    }

    #[test]
    fn wrap_body_rejects_ragged_body() {
        let err = wrap_body(FrameMode::Ms30, &[0u8; 49]).unwrap_err();
        assert!(format!("{err}").contains("multiple"));
    }

    #[test]
    fn magic_for_is_stable() {
        assert_eq!(magic_for(FrameMode::Ms20), MAGIC_20MS);
        assert_eq!(magic_for(FrameMode::Ms30), MAGIC_30MS);
    }

    #[test]
    fn mark_and_detect_lost_frame() {
        let frame = vec![0u8; 38];
        assert!(!is_lost(&frame));
        let lost = mark_lost(&frame);
        assert!(is_lost(&lost));
        // Only the indicator bit changed.
        assert_eq!(lost.last(), Some(&0x01));
        assert_eq!(&lost[..37], &frame[..37]);
    }

    #[test]
    fn mark_lost_preserves_other_payload_bits() {
        let frame: Vec<u8> = (0..50).map(|i| (0xF0 | (i & 0x0E)) as u8).collect();
        let lost = mark_lost(&frame);
        assert!(is_lost(&lost));
        // Every byte but the last is untouched; the last differs only
        // in bit 0.
        assert_eq!(&lost[..49], &frame[..49]);
        assert_eq!(lost[49], frame[49] | 0x01);
    }

    #[test]
    fn clear_lost_is_inverse_of_mark_on_even_last_byte() {
        let frame: Vec<u8> = (0..38).map(|i| (i * 2) as u8).collect(); // last byte even
        assert!(!is_lost(&frame));
        let lost = mark_lost(&frame);
        let cleared = clear_lost(&lost);
        assert!(!is_lost(&cleared));
        assert_eq!(cleared, frame);
    }

    #[test]
    fn is_lost_on_empty_slice_is_false() {
        assert!(!is_lost(&[]));
    }
}
