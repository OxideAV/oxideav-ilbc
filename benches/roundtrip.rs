//! Criterion benchmarks for the iLBC encode -> decode round-trip.
//!
//! Round 180 (depth-mode benchmarks): companion to `decode.rs` and
//! `encode.rs`. These scenarios time the full encoder + decoder pair
//! through the public trait surface, in the order a transcoder
//! actually pays — S16 PCM in, 38-/50-byte iLBC packets, S16 PCM
//! out. The round-trip number bounds what a streaming pipeline
//! observes, which neither half alone captures: an encoder tweak
//! that speeds the CB search but bloats the LSF quantiser, or a
//! decoder tweak that defers cost into the enhancer, shows up here
//! as a net regression even when one half looks like a win.
//!
//! All inputs are synthesised in-bench from a deterministic
//! xorshift seed; no `docs/` fixtures or external files are read.
//!
//! Scenarios:
//!
//!   - **roundtrip_mono_8k_20ms_1s**: 1 s of mono S16 PCM at 8 kHz,
//!     20 ms framing — the canonical low-latency VoIP scenario.
//!   - **roundtrip_mono_8k_30ms_1s**: same source, 30 ms framing —
//!     lower bitrate, larger per-packet working set.
//!   - **roundtrip_mono_8k_20ms_3s**: 3 s clip, 20 ms framing — long
//!     enough for the enhancer's pitch buffer and the encoder's
//!     `prev_a_per_sub` carry-over to settle into steady state.
//!
//! Run with:
//!     cargo bench -p oxideav-ilbc --bench roundtrip

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, Decoder, Encoder, Error, Frame, Packet,
    SampleFormat, TimeBase,
};
use oxideav_ilbc::{decoder, encoder, FrameMode, CODEC_ID_STR, SAMPLE_RATE};

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn build_pcm_bytes(n_samples: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n_samples * 2);
    let f0 = 145.0f32;
    let sr = SAMPLE_RATE as f32;
    for n in 0..n_samples {
        let t = n as f32 / sr;
        let mut v = 0.0f32;
        for h in 1..=4 {
            v += (2.0 * core::f32::consts::PI * (h as f32) * f0 * t).sin() * (4200.0 / h as f32);
        }
        let noise = (xorshift32(&mut state) as i32 >> 22) as f32;
        let s = (v + noise).round().clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn ilbc_params(mode: FrameMode) -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    p.sample_rate = Some(SAMPLE_RATE);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    if mode == FrameMode::Ms30 {
        p.options = CodecOptions::new().set("frame_ms", "30");
    }
    p
}

/// Encode the full clip, then decode every emitted packet. Returns
/// the total decoded PCM byte count for `black_box`.
fn run_roundtrip(mode: FrameMode, pcm_bytes: &[u8], samples: u32) -> usize {
    let enc_params = ilbc_params(mode);
    let dec_params = ilbc_params(FrameMode::Ms20); // mode is detected from packet length
    let mut enc: Box<dyn Encoder> = encoder::make_encoder(&enc_params).expect("make_encoder");
    let mut dec: Box<dyn Decoder> = decoder::make_decoder(&dec_params).expect("make_decoder");

    enc.send_frame(&Frame::Audio(AudioFrame {
        samples,
        pts: Some(0),
        data: vec![pcm_bytes.to_vec()],
    }))
    .expect("send_frame");
    enc.flush().expect("flush");

    let mut total = 0usize;
    loop {
        match enc.receive_packet() {
            Ok(p) => {
                let dec_pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), p.data.clone());
                dec.send_packet(&dec_pkt).expect("send_packet");
                if let Frame::Audio(a) = dec.receive_frame().expect("receive_frame") {
                    total += a.data[0].len();
                }
            }
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("unexpected encoder error: {:?}", e),
        }
    }
    total
}

fn bench_roundtrip_mono_8k_20ms_1s(c: &mut Criterion) {
    let n = 8_000;
    let pcm = build_pcm_bytes(n, 0xCAFE_F00D);
    let mut g = c.benchmark_group("roundtrip_mono_8k_20ms_1s");
    g.throughput(Throughput::Bytes((n * 2) as u64));
    g.bench_function(BenchmarkId::from_parameter("mono/8k/20ms/1s"), |b| {
        b.iter(|| {
            let bytes = run_roundtrip(FrameMode::Ms20, criterion::black_box(&pcm), n as u32);
            criterion::black_box(bytes);
        });
    });
    g.finish();
}

fn bench_roundtrip_mono_8k_30ms_1s(c: &mut Criterion) {
    let n = 8_000;
    let pcm = build_pcm_bytes(n, 0xBEEF_1234);
    let mut g = c.benchmark_group("roundtrip_mono_8k_30ms_1s");
    g.throughput(Throughput::Bytes((n * 2) as u64));
    g.bench_function(BenchmarkId::from_parameter("mono/8k/30ms/1s"), |b| {
        b.iter(|| {
            let bytes = run_roundtrip(FrameMode::Ms30, criterion::black_box(&pcm), n as u32);
            criterion::black_box(bytes);
        });
    });
    g.finish();
}

fn bench_roundtrip_mono_8k_20ms_3s(c: &mut Criterion) {
    let n = 24_000;
    let pcm = build_pcm_bytes(n, 0xDECA_FBAD);
    let mut g = c.benchmark_group("roundtrip_mono_8k_20ms_3s");
    g.throughput(Throughput::Bytes((n * 2) as u64));
    g.bench_function(BenchmarkId::from_parameter("mono/8k/20ms/3s"), |b| {
        b.iter(|| {
            let bytes = run_roundtrip(FrameMode::Ms20, criterion::black_box(&pcm), n as u32);
            criterion::black_box(bytes);
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_roundtrip_mono_8k_20ms_1s,
    bench_roundtrip_mono_8k_30ms_1s,
    bench_roundtrip_mono_8k_20ms_3s,
);
criterion_main!(benches);
