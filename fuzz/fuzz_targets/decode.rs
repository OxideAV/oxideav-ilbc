#![no_main]

//! Decode arbitrary fuzz-supplied bytes through the iLBC framing path.
//!
//! The contract under test is purely that the calls *return*: any
//! 38-byte (20 ms) or 50-byte (30 ms) iLBC payload must yield either
//! `Ok(Frame::Audio)` with the mode's natural 160 / 240-sample S16
//! buffer or a typed `Err(oxideav_core::Error::…)`. Neither path may
//! panic, integer-overflow (in debug builds), index out of bounds,
//! abort, or allocate an attacker-controlled buffer.
//!
//! Three entry points are exercised on every input:
//!
//! 1. **Raw decode of the whole input as one packet.** Catches lengths
//!    outside `{38, 50}` (the `make_decoder` factory wrapper returns
//!    `Err(Error::Invalid)` immediately) and also length-39 / length-49
//!    boundary cases the bit-reader-out-of-bits path needs to detect.
//! 2. **Sliding 38-byte windows** through the entire payload, one
//!    after another, all sent through the *same* decoder instance.
//!    This exercises the 20 ms decode pipeline plus the inter-frame
//!    enhancer + post-filter state carry-over, plus the
//!    `prev_a_per_sub` enhancer-delay LPC shift of RFC §4.7.
//! 3. **Sliding 50-byte windows** through the entire payload, again
//!    one after another through a separate decoder instance. This
//!    covers the 30 ms decode pipeline with the same state-carry
//!    contracts.
//!
//! All four return values are intentionally discarded; the
//! out-of-channel decoder state mutation is what's being driven.
//!
//! The `parse_packet` re-export is also called directly on the raw
//! input — that surface skips the registration / RuntimeContext glue
//! and goes straight to the §3.8 bitreader frame parser, useful for
//! detecting any out-of-bits / table-lookup-out-of-range issue the
//! `Decoder` wrapper might mask.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
// `SampleFormat::S16` is referenced through the `default_params`
// helper below to keep the decoder's stream-level format pinned to
// what the iLBC pipeline emits. The slim per-frame `AudioFrame` shape
// does not carry sample format directly.
use oxideav_ilbc::decoder::{make_decoder, parse_packet};
use oxideav_ilbc::{
    CODEC_ID_STR, FRAME_BYTES_20MS, FRAME_BYTES_30MS, FRAME_SAMPLES_20MS, FRAME_SAMPLES_30MS,
    SAMPLE_RATE,
};

fn default_params() -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    p.sample_rate = Some(SAMPLE_RATE);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    p
}

fn time_base() -> TimeBase {
    TimeBase::new(1, SAMPLE_RATE as i64)
}

/// Drive a single packet through the decoder and assert structural
/// invariants on the returned frame (if any).
fn drive(dec: &mut Box<dyn oxideav_core::Decoder>, bytes: &[u8]) {
    let pkt = Packet::new(0, time_base(), bytes.to_vec());
    if dec.send_packet(&pkt).is_err() {
        return;
    }
    let frame = match dec.receive_frame() {
        Ok(f) => f,
        Err(_) => return,
    };
    if let Frame::Audio(a) = frame {
        let len = bytes.len();
        // Per the slim AudioFrame shape (`samples` + `data` only — sample
        // rate / channel count / sample format are stream-level on
        // `CodecParameters`, not per-frame), the decoder's correctness
        // witness is the per-frame sample count and the per-mode S16
        // byte count.
        let pcm_bytes = a.data.first().map(|d| d.len()).unwrap_or(0);
        if len == FRAME_BYTES_20MS {
            assert_eq!(
                pcm_bytes,
                FRAME_SAMPLES_20MS * 2,
                "20 ms decode must emit 160 S16 samples"
            );
            assert_eq!(a.samples as usize, FRAME_SAMPLES_20MS);
        } else if len == FRAME_BYTES_30MS {
            assert_eq!(
                pcm_bytes,
                FRAME_SAMPLES_30MS * 2,
                "30 ms decode must emit 240 S16 samples"
            );
            assert_eq!(a.samples as usize, FRAME_SAMPLES_30MS);
        }
        // Any other length means the make_decoder wrapper rejected the
        // packet upstream and we wouldn't have reached the receive path.
    }
}

fuzz_target!(|data: &[u8]| {
    // (1) Bitreader-only path — no Decoder state needed, just the
    // §3.8 frame parser. Works on any length; non-{38,50} returns
    // Err(Error::Invalid) before any table lookup runs.
    let _ = parse_packet(data);

    // (2) Whole-input-as-one-packet decode.
    {
        let params = default_params();
        if let Ok(mut dec) = make_decoder(&params) {
            drive(&mut dec, data);
        }
    }

    // (3) Sliding 20 ms windows on the same decoder instance.
    if data.len() >= FRAME_BYTES_20MS {
        let params = default_params();
        if let Ok(mut dec) = make_decoder(&params) {
            // Cap the iteration count so the fuzzer doesn't spend its
            // time budget on pathological-length inputs (a 16 KiB
            // input would otherwise produce 16384 - 38 ~= 16346
            // sliding windows). 256 windows is enough to drive the
            // enhancer + post-filter state to steady state.
            let max_windows = (data.len() - FRAME_BYTES_20MS + 1).min(256);
            for i in 0..max_windows {
                drive(&mut dec, &data[i..i + FRAME_BYTES_20MS]);
            }
        }
    }

    // (4) Sliding 30 ms windows on a fresh decoder instance.
    if data.len() >= FRAME_BYTES_30MS {
        let params = default_params();
        if let Ok(mut dec) = make_decoder(&params) {
            let max_windows = (data.len() - FRAME_BYTES_30MS + 1).min(256);
            for i in 0..max_windows {
                drive(&mut dec, &data[i..i + FRAME_BYTES_30MS]);
            }
        }
    }
});
