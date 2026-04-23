//! Smoke-test the iLBC decoder end-to-end via the public registry API.

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};

/// Build a decoder through the registry like a downstream user would.
fn make_dec() -> Box<dyn oxideav_codec::Decoder> {
    let mut reg = CodecRegistry::new();
    oxideav_ilbc::register(&mut reg);
    let mut params = CodecParameters::audio(CodecId::new(oxideav_ilbc::CODEC_ID_STR));
    params.sample_rate = Some(oxideav_ilbc::SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    reg.make_decoder(&params).expect("decoder")
}

#[test]
fn registry_roundtrip_20ms() {
    let mut dec = make_dec();
    // Varied non-zero byte pattern so every LSF / scale / CB / gain
    // field lands at a mid-range index.
    let packet: Vec<u8> = (0..oxideav_ilbc::FRAME_BYTES_20MS as u8)
        .map(|i| 0x55 ^ i.wrapping_mul(17))
        .collect();
    let pkt = Packet::new(
        0,
        TimeBase::new(1, oxideav_ilbc::SAMPLE_RATE as i64),
        packet,
    )
    .with_pts(0);
    dec.send_packet(&pkt).unwrap();
    let Frame::Audio(a) = dec.receive_frame().unwrap() else {
        panic!("audio frame expected");
    };
    assert_eq!(a.samples, 160);
    assert_eq!(a.sample_rate, 8_000);
    assert_eq!(a.channels, 1);
    // Ensure at least one non-zero sample is produced (sanity that the
    // pipeline is actually exercising the synthesis path).
    let any_nonzero = a.data[0]
        .chunks_exact(2)
        .any(|c| i16::from_le_bytes([c[0], c[1]]) != 0);
    assert!(any_nonzero, "decoder produced all-zero PCM");
}

#[test]
fn registry_roundtrip_30ms() {
    let mut dec = make_dec();
    let packet: Vec<u8> = (0..oxideav_ilbc::FRAME_BYTES_30MS as u8)
        .map(|i| i.wrapping_mul(31).wrapping_add(7))
        .collect();
    let pkt = Packet::new(
        0,
        TimeBase::new(1, oxideav_ilbc::SAMPLE_RATE as i64),
        packet,
    );
    dec.send_packet(&pkt).unwrap();
    let Frame::Audio(a) = dec.receive_frame().unwrap() else {
        panic!("audio frame expected");
    };
    assert_eq!(a.samples, 240);
    assert_eq!(a.sample_rate, 8_000);
}

#[test]
fn stream_of_mixed_frames() {
    let mut dec = make_dec();
    // Alternate 20 / 30 ms frames and check each decodes to the right
    // sample count.
    let p20: Vec<u8> = vec![0x33; oxideav_ilbc::FRAME_BYTES_20MS];
    let p30: Vec<u8> = vec![0x55; oxideav_ilbc::FRAME_BYTES_30MS];
    let mut pts: i64 = 0;
    for i in 0..6 {
        let (data, expected_samples) = if i % 2 == 0 {
            (p20.clone(), 160)
        } else {
            (p30.clone(), 240)
        };
        let pkt = Packet::new(0, TimeBase::new(1, 8000), data).with_pts(pts);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(a) = dec.receive_frame().unwrap() else {
            panic!();
        };
        assert_eq!(a.samples as usize, expected_samples);
        pts += expected_samples as i64;
    }
}
