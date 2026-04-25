//! Pure-Rust iLBC (Internet Low Bit Rate Codec) decoder — RFC 3951.
//!
//! iLBC is an 8 kHz narrowband speech codec designed for VoIP over
//! lossy channels. This crate implements the decoder side only.
//!
//! Pipeline (all pure-Rust, RFC 3951 §4):
//! - [`bitreader`]: 304-/400-bit payload -> split-VQ LSF indices + start-
//!   state + per-subblock codebook indices + gain indices + empty-frame
//!   flag (§3.8 / Table 3.2).
//! - [`lsf`]: split-VQ LSF dequantisation (§4.1), stability check
//!   (§3.2.5 — 50 Hz safety margin), linear interpolation across
//!   sub-blocks (§3.2.6 / §3.2.7), LSF -> LPC conversion.
//! - [`state`]: start-state reconstruction (§4.2) — scalar dequant +
//!   all-pass phase compensator.
//! - [`cb`]: multistage adaptive-codebook construction and gain
//!   dequantisation (§4.3 / §4.4).
//! - [`synthesis`]: 10th-order LPC synthesis (§4.7), pitch-emphasis
//!   post-filter (§4.8 simplified), and a dampened pitch-synchronous
//!   PLC unit (§4.5).
//! - [`decoder`]: wires those pieces into the `oxideav_core::Decoder`
//!   trait.
//!
//! The frame mode is inferred from packet length:
//! - 38 bytes  -> 20 ms / 160 samples / 304 bits
//! - 50 bytes  -> 30 ms / 240 samples / 400 bits

#![allow(
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::too_many_arguments,
    clippy::manual_range_contains,
    clippy::manual_memcpy
)]

pub mod bitreader;
pub mod bitwriter;
pub mod cb;
pub mod cb_search;
pub mod codec;
pub mod decoder;
pub mod encoder;
pub mod enhancer;
pub mod lpc_analysis;
pub mod lsf;
pub mod lsf_quant;
pub mod lsf_tables;
pub mod state;
pub mod state_encode;
pub mod synthesis;

use oxideav_core::CodecRegistry;

/// Codec id string.
pub const CODEC_ID_STR: &str = "ilbc";

/// Sample rate — iLBC is strictly narrowband 8 kHz.
pub const SAMPLE_RATE: u32 = 8_000;

/// Samples per 5 ms sub-block (40 samples @ 8 kHz).
pub const SUBL: usize = 40;

/// LPC order (10th-order all-pole short-term predictor).
pub const LPC_ORDER: usize = 10;

/// Length of the start state in scalar-coded samples — RFC 3951 §4.2
/// calls this `STATE_SHORT_LEN`. Depends on frame mode:
/// - 20 ms  -> 57 samples
/// - 30 ms  -> 58 samples
pub const STATE_SHORT_LEN_20MS: usize = 57;
pub const STATE_SHORT_LEN_30MS: usize = 58;

/// Full start-state vector length in samples (the 80-sample window that
/// anchors the adaptive codebook).
pub const STATE_LEN: usize = 80;

/// Circular-codebook memory length for the adaptive codebook
/// (RFC §4.3 `CB_LMEM`).
pub const CB_LMEM: usize = 147;

/// 20 ms frame: 160 samples, 304 payload bits, 38 bytes.
pub const FRAME_SAMPLES_20MS: usize = 160;
pub const FRAME_BYTES_20MS: usize = 38;
pub const FRAME_BITS_20MS: usize = 304;

/// 30 ms frame: 240 samples, 400 payload bits, 50 bytes.
pub const FRAME_SAMPLES_30MS: usize = 240;
pub const FRAME_BYTES_30MS: usize = 50;
pub const FRAME_BITS_30MS: usize = 400;

/// iLBC frame mode — 20 ms or 30 ms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameMode {
    /// 20 ms / 160 samples / 38 bytes / 15.20 kbit/s.
    Ms20,
    /// 30 ms / 240 samples / 50 bytes / 13.33 kbit/s.
    Ms30,
}

impl FrameMode {
    /// Detect the frame mode from the payload byte length.
    pub fn from_packet_len(len: usize) -> Option<Self> {
        match len {
            FRAME_BYTES_20MS => Some(FrameMode::Ms20),
            FRAME_BYTES_30MS => Some(FrameMode::Ms30),
            _ => None,
        }
    }

    /// Number of PCM samples produced per frame.
    pub fn samples(self) -> usize {
        match self {
            FrameMode::Ms20 => FRAME_SAMPLES_20MS,
            FrameMode::Ms30 => FRAME_SAMPLES_30MS,
        }
    }

    /// Number of 40-sample sub-blocks per frame (for LPC / synthesis).
    /// 20 ms = 4, 30 ms = 6.
    pub fn sub_blocks(self) -> usize {
        self.samples() / SUBL
    }

    /// Number of 40-sample codebook sub-blocks carried in the
    /// bitstream (Table 3.2 "sub-block 1..N"). 20 ms = 2, 30 ms = 4.
    /// These encode the forward + backward excitation around the
    /// 80-sample state vector; the state vector itself is scalar-coded
    /// plus a 22-/23-sample boundary CB block.
    pub fn cb_sub_blocks(self) -> usize {
        match self {
            FrameMode::Ms20 => 2,
            FrameMode::Ms30 => 4,
        }
    }

    /// Byte count of one packet in this mode.
    pub fn bytes(self) -> usize {
        match self {
            FrameMode::Ms20 => FRAME_BYTES_20MS,
            FrameMode::Ms30 => FRAME_BYTES_30MS,
        }
    }

    /// Bit count of one packet in this mode.
    pub fn bits(self) -> usize {
        match self {
            FrameMode::Ms20 => FRAME_BITS_20MS,
            FrameMode::Ms30 => FRAME_BITS_30MS,
        }
    }

    /// Length of the scalar-coded start-state vector.
    pub fn state_short_len(self) -> usize {
        match self {
            FrameMode::Ms20 => STATE_SHORT_LEN_20MS,
            FrameMode::Ms30 => STATE_SHORT_LEN_30MS,
        }
    }

    /// Number of LSF vectors transmitted: 20 ms has 1 LSF vector, 30 ms
    /// has 2 (each is three split-VQ indices).
    pub fn lsf_vectors(self) -> usize {
        match self {
            FrameMode::Ms20 => 1,
            FrameMode::Ms30 => 2,
        }
    }
}

/// Register the iLBC codec with the codec registry.
pub fn register(reg: &mut CodecRegistry) {
    codec::register(reg);
}
