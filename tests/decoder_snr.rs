#![allow(
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::unnecessary_cast
)]
//! End-to-end decoder stability / repeatability SNR diagnostic.
//!
//! We don't have an iLBC encoder in the workspace, so we can't do a
//! true waveform roundtrip. Instead:
//!
//! 1. Feed a fixed synthetic packet stream through the decoder.
//! 2. Record the PCM output across 20 frames.
//! 3. Decode the same stream again and compute the "SNR" between
//!    the two passes — this must be infinite (bit-identical), which
//!    is an integrity check.
//! 4. Additionally, compute the "effective per-frame SNR" of the
//!    decoded signal vs. a matched concealment-only run — higher
//!    means the codec has more decoded content relative to PLC noise.
//!
//! The second measure is printed rather than asserted — diagnostic
//! only. Run with `cargo test -- --nocapture decoder_snr`.

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};

fn make_dec() -> Box<dyn oxideav_codec::Decoder> {
    let mut reg = CodecRegistry::new();
    oxideav_ilbc::register(&mut reg);
    let mut params = CodecParameters::audio(CodecId::new(oxideav_ilbc::CODEC_ID_STR));
    params.sample_rate = Some(oxideav_ilbc::SAMPLE_RATE);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    reg.make_decoder(&params).expect("decoder")
}

fn decode_stream(packets: &[Vec<u8>]) -> Vec<i16> {
    let mut dec = make_dec();
    let mut out = Vec::new();
    for (i, p) in packets.iter().enumerate() {
        let pkt = Packet::new(
            0,
            TimeBase::new(1, oxideav_ilbc::SAMPLE_RATE as i64),
            p.clone(),
        )
        .with_pts(i as i64 * 160);
        dec.send_packet(&pkt).unwrap();
        if let Frame::Audio(a) = dec.receive_frame().unwrap() {
            for chunk in a.data[0].chunks_exact(2) {
                out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
    }
    out
}

fn energy(pcm: &[i16]) -> f64 {
    pcm.iter().map(|&s| (s as f64) * (s as f64)).sum()
}

fn snr_db(reference: &[i16], test: &[i16]) -> f64 {
    assert_eq!(reference.len(), test.len());
    let mut s_sig = 0.0;
    let mut s_err = 0.0;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let d = (t as f64) - (r as f64);
        s_sig += (r as f64) * (r as f64);
        s_err += d * d;
    }
    if s_err < 1e-9 {
        return f64::INFINITY;
    }
    10.0 * (s_sig / s_err).log10()
}

#[test]
fn decoder_determinism_and_energy() {
    // 20 varied 20-ms packets.
    let packets: Vec<Vec<u8>> = (0..20u8)
        .map(|seed| {
            (0..oxideav_ilbc::FRAME_BYTES_20MS as u8)
                .map(|i| (0x55 ^ i.wrapping_mul(17).wrapping_add(seed)) as u8)
                .collect()
        })
        .collect();

    let pcm_a = decode_stream(&packets);
    let pcm_b = decode_stream(&packets);

    assert_eq!(pcm_a.len(), 20 * 160);
    assert_eq!(pcm_a, pcm_b, "decoder is not bit-deterministic");

    let e = energy(&pcm_a);
    let n = pcm_a.len() as f64;
    println!(
        "decoder_snr: {} samples, mean energy {:.0}, rms {:.2}",
        pcm_a.len(),
        e / n,
        (e / n).sqrt()
    );

    // Concealment-only: send empty packets.
    let empty_pkts: Vec<Vec<u8>> = (0..20)
        .map(|_| vec![0u8; oxideav_ilbc::FRAME_BYTES_20MS])
        .collect();
    // Flip the empty-flag bit so the decoder always runs PLC.
    let empty_pkts: Vec<Vec<u8>> = empty_pkts
        .into_iter()
        .map(|mut p| {
            let last = p.len() - 1;
            p[last] |= 1;
            p
        })
        .collect();
    let plc_pcm = decode_stream(&empty_pkts);
    let e_plc = energy(&plc_pcm);
    println!(
        "decoder_snr: PLC-only mean energy {:.0}, rms {:.2}",
        e_plc / n,
        (e_plc / n).sqrt()
    );

    // Ratio decoded / PLC in dB. Higher = more signal-dependent content.
    if e_plc > 0.0 && e > 0.0 {
        let ratio_db = 10.0 * (e / e_plc).log10();
        println!("decoder_snr: decoded-vs-PLC ratio = {:.2} dB", ratio_db);
    }

    // Self-SNR must be infinite.
    let s = snr_db(&pcm_a, &pcm_b);
    println!("decoder_snr: self-SNR = {} (expected inf)", s);
}
