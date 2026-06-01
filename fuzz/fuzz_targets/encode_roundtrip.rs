#![no_main]

//! Drive arbitrary fuzz-supplied S16 PCM bytes through the iLBC
//! **encoder** and push every emitted packet straight back through
//! the decoder. The contract under test:
//!
//! 1. `make_encoder` either constructs an `IlbcEncoder` for the given
//!    options or returns a typed `Err(Error::Unsupported)`. Neither
//!    path panics.
//! 2. `Encoder::send_frame` on arbitrary S16 input either accepts the
//!    samples (queueing them into the per-mode 160 / 240-sample frame
//!    buffer) or returns a typed `Err(Error::Invalid)`. Never panics.
//! 3. `Encoder::receive_packet` returns a `Packet` whose `data.len()`
//!    is exactly the mode's natural byte count (38 / 50), never
//!    something the decoder will reject as a length mismatch.
//! 4. Every accepted packet is bytes-the-decoder-ingests-without-
//!    panicking. CELP rounding drift means the *samples* won't be
//!    bit-exact (that's what `tests/round_trip_snr.rs` exercises on
//!    deterministic signals), but `Decoder::send_packet` +
//!    `receive_frame` always succeed and yield a 160-/240-sample
//!    audio frame.
//! 5. `Encoder::flush` is panic-free regardless of whether the input
//!    ended on a frame boundary.
//!
//! ## Fuzz input layout
//!
//! ```text
//!   byte 0      : mode + options seed
//!                   bit 0 → frame_ms = 20 or 30 ms (0 → 20, 1 → 30)
//!                   bit 1 → hp_filter (1 → on)
//!                   bit 2 → state_dpcm (1 → on)
//!   bytes 1..   : interleaved S16 LE samples; trailing partial
//!                 sample-pair is dropped.
//! ```
//!
//! The dispatcher feeds the encoder one big `send_frame` with all the
//! sample bytes, then drains all packets, then `flush` to make sure
//! a partial last frame is either emitted (with zero-padding) or
//! quietly dropped without panicking.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
use oxideav_ilbc::decoder::make_decoder;
use oxideav_ilbc::encoder::make_encoder;
use oxideav_ilbc::{
    CODEC_ID_STR, FRAME_BYTES_20MS, FRAME_BYTES_30MS, FRAME_SAMPLES_20MS, FRAME_SAMPLES_30MS,
    SAMPLE_RATE,
};

fn time_base() -> TimeBase {
    TimeBase::new(1, SAMPLE_RATE as i64)
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let seed = data[0];
    let payload = &data[1..];

    let frame_ms_30 = (seed & 0x01) != 0;
    let hp_filter_on = (seed & 0x02) != 0;
    let state_dpcm_on = (seed & 0x04) != 0;

    // Snap the payload to an even byte count (one S16 sample = 2 bytes).
    // We also cap the input so the fuzzer can't burn its time budget
    // on a multi-second clip — 16 KiB is ~8000 samples = 50 frames in
    // 20 ms mode / ~33 frames in 30 ms mode, more than enough to drive
    // the encoder's `prev_a_per_sub` carry-over to steady state.
    let cap = payload.len().min(16 * 1024);
    let payload = &payload[..cap & !1];
    if payload.len() < 2 {
        return;
    }

    // Build the encoder.
    let mut enc_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    enc_params.sample_rate = Some(SAMPLE_RATE);
    enc_params.channels = Some(1);
    enc_params.sample_format = Some(SampleFormat::S16);
    if frame_ms_30 {
        enc_params.options = enc_params.options.set("frame_ms", "30");
    }
    if hp_filter_on {
        enc_params.options = enc_params.options.set("hp_filter", "on");
    }
    if state_dpcm_on {
        enc_params.options = enc_params.options.set("state_dpcm", "on");
    }
    let mut enc = match make_encoder(&enc_params) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Same decoder params (mode is selected per-packet from byte length,
    // not from the params).
    let mut dec_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    dec_params.sample_rate = Some(SAMPLE_RATE);
    dec_params.channels = Some(1);
    dec_params.sample_format = Some(SampleFormat::S16);
    let mut dec = match make_decoder(&dec_params) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Push one big audio frame through.
    let af = AudioFrame {
        samples: (payload.len() / 2) as u32,
        pts: None,
        data: vec![payload.to_vec()],
    };
    if enc.send_frame(&Frame::Audio(af)).is_err() {
        // The encoder rejected the input wholesale (e.g. odd byte
        // count slipped past — defensive). Done.
        return;
    }

    // Drain every accepted packet through the decoder.
    let mut drained = 0usize;
    while let Ok(pkt) = enc.receive_packet() {
        let len = pkt.data.len();
        // Contract (3): the encoder MUST emit only well-formed lengths.
        assert!(
            len == FRAME_BYTES_20MS || len == FRAME_BYTES_30MS,
            "iLBC encoder emitted packet of length {len}, expected 38 or 50",
        );
        // Mode is locked in `make_encoder`, so every packet from this
        // encoder must carry the same byte length.
        if frame_ms_30 {
            assert_eq!(
                len, FRAME_BYTES_30MS,
                "30 ms encoder emitted a non-50-byte packet",
            );
        } else {
            assert_eq!(
                len, FRAME_BYTES_20MS,
                "20 ms encoder emitted a non-38-byte packet",
            );
        }

        // Contract (4): the decoder ingests every emitted packet.
        let dec_pkt = Packet::new(0, time_base(), pkt.data.clone());
        if dec.send_packet(&dec_pkt).is_err() {
            // The encoder produced bytes the decoder refused. The
            // strict assert is the meaningful failure mode here, not
            // a panic — flag it.
            panic!("encoder emitted bytes the decoder refused");
        }
        let frame = dec
            .receive_frame()
            .expect("decoder must produce a frame for any encoder-emitted packet");
        if let Frame::Audio(a) = frame {
            // Per the slim `AudioFrame` shape (`samples` + `data` only,
            // sample rate / channels / format live on `CodecParameters`),
            // the decoder's correctness witness is the per-frame sample
            // count and the per-mode S16 byte count.
            let expected = if frame_ms_30 {
                FRAME_SAMPLES_30MS
            } else {
                FRAME_SAMPLES_20MS
            };
            assert_eq!(a.samples as usize, expected);
            let pcm_bytes = a.data.first().map(|d| d.len()).unwrap_or(0);
            assert_eq!(pcm_bytes, expected * 2);
        } else {
            panic!("iLBC decoder produced a non-audio frame");
        }

        drained += 1;
        // Belt-and-braces cap so the assertion loop can't run forever.
        if drained > 1024 {
            break;
        }
    }

    // Contract (5): flush is panic-free regardless of input alignment.
    let _ = enc.flush();
    while let Ok(pkt) = enc.receive_packet() {
        let len = pkt.data.len();
        assert!(len == FRAME_BYTES_20MS || len == FRAME_BYTES_30MS);
        let dec_pkt = Packet::new(0, time_base(), pkt.data.clone());
        if dec.send_packet(&dec_pkt).is_ok() {
            let _ = dec.receive_frame();
        }
    }
});
