//! Encoder forward-path determinism + structural invariants.
//!
//! The iLBC encoder (RFC 3951 §3) is a deterministic function of its
//! input PCM block — there is no randomness or adaptive dithering in the
//! analysis path. Encoding the same input twice must therefore produce
//! byte-identical packets, and every emitted packet must be exactly the
//! mode's frame size (38 bytes / 304 bits for 20 ms, 50 bytes / 400 bits
//! for 30 ms — RFC 3951 §3.8). These tests pin both so a future change to
//! the LPC analysis, the §3.6 codebook search, or the §3.8 ULP packing
//! can't silently introduce non-determinism or a wrong-length frame.

use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, Encoder, Frame, SampleFormat,
};
use oxideav_core::{CodecRegistry, Packet, TimeBase};

use oxideav_ilbc::{FrameMode, CODEC_ID_STR, FRAME_BYTES_20MS, FRAME_BYTES_30MS, SAMPLE_RATE};

fn gen_voiced(samples: usize) -> Vec<i16> {
    let f0 = 130.0f32;
    (0..samples)
        .map(|n| {
            let t = n as f32 / SAMPLE_RATE as f32;
            let mut v = 0.0f32;
            for h in 1..5 {
                v +=
                    ((2.0 * core::f32::consts::PI * h as f32 * f0 * t).sin()) * (3000.0 / h as f32);
            }
            v.round().clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

fn pcm_to_audio_frame(pcm: &[i16]) -> Frame {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        samples: pcm.len() as u32,
        pts: Some(0),
        data: vec![bytes],
    })
}

fn encode_all(mode: FrameMode, pcm: &[i16]) -> Vec<Vec<u8>> {
    let mut reg = CodecRegistry::new();
    oxideav_ilbc::register_codecs(&mut reg);
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.sample_rate = Some(SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut options = CodecOptions::new();
    if mode == FrameMode::Ms30 {
        options = options.set("frame_ms", "30");
    }
    params.options = options;
    let mut enc: Box<dyn Encoder> = reg.first_encoder(&params).expect("encoder");

    enc.send_frame(&pcm_to_audio_frame(pcm)).unwrap();
    enc.flush().unwrap();

    let mut packets = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        packets.push(pkt.data.clone());
    }
    packets
}

fn frame_bytes(mode: FrameMode) -> usize {
    match mode {
        FrameMode::Ms20 => FRAME_BYTES_20MS,
        FrameMode::Ms30 => FRAME_BYTES_30MS,
    }
}

fn assert_deterministic_and_sized(mode: FrameMode) {
    // A whole second of voiced input — many frames so any per-frame
    // analysis-state carry-over (the §3.6 codebook memory window, the
    // §4.2.1 LPC look-back) is exercised across frame boundaries.
    let pcm = gen_voiced(SAMPLE_RATE as usize);

    let first = encode_all(mode, &pcm);
    let second = encode_all(mode, &pcm);

    assert!(!first.is_empty(), "{mode:?}: encoder produced no packets");
    assert_eq!(
        first.len(),
        second.len(),
        "{mode:?}: re-encode produced a different packet count ({} vs {})",
        first.len(),
        second.len()
    );

    let want = frame_bytes(mode);
    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert_eq!(
            a.len(),
            want,
            "{mode:?}: packet {i} is {} bytes, expected {want} (RFC 3951 §3.8 frame size)",
            a.len()
        );
        assert_eq!(
            a, b,
            "{mode:?}: packet {i} differs between two encodes of identical input — \
             the §3 analysis path is not deterministic"
        );
    }
}

#[test]
fn encode_is_deterministic_20ms() {
    assert_deterministic_and_sized(FrameMode::Ms20);
}

#[test]
fn encode_is_deterministic_30ms() {
    assert_deterministic_and_sized(FrameMode::Ms30);
}

/// A second encode→decode→encode pass must also be deterministic: feeding
/// the decoder output of one encode back through the encoder yields the
/// same bytes every time. This exercises the full forward path on a
/// realistic (already band-limited, CELP-shaped) input rather than a
/// synthetic tone, catching non-determinism that only surfaces on signals
/// near the quantiser cell boundaries.
#[test]
fn reencode_of_decoded_output_is_deterministic_20ms() {
    let pcm = gen_voiced(SAMPLE_RATE as usize);

    // First encode → decode.
    let packets = encode_all(FrameMode::Ms20, &pcm);
    let mut reg = CodecRegistry::new();
    oxideav_ilbc::register_codecs(&mut reg);
    let mut dec_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    dec_params.sample_rate = Some(SAMPLE_RATE);
    dec_params.channels = Some(1);
    let mut dec = reg.first_decoder(&dec_params).expect("decoder");
    let mut decoded = Vec::new();
    for p in &packets {
        let dp = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), p.clone());
        dec.send_packet(&dp).unwrap();
        if let Frame::Audio(a) = dec.receive_frame().unwrap() {
            for c in a.data[0].chunks_exact(2) {
                decoded.push(i16::from_le_bytes([c[0], c[1]]));
            }
        }
    }

    // Re-encode the decoded PCM twice — must be byte-identical.
    let r1 = encode_all(FrameMode::Ms20, &decoded);
    let r2 = encode_all(FrameMode::Ms20, &decoded);
    assert_eq!(r1, r2, "re-encode of decoded output is non-deterministic");
    assert!(!r1.is_empty());
}
