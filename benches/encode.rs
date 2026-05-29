//! Criterion benchmarks for the iLBC encoder hot path.
//!
//! Round 180 (depth-mode benchmarks): companion to `decode.rs` and
//! `roundtrip.rs`. Each scenario synthesises a deterministic mono
//! S16 PCM clip at the codec's native 8 kHz and runs the production
//! `oxideav_ilbc::encoder::make_encoder` trait object through a
//! `send_frame -> flush -> receive_packet*` cycle. The timed body
//! covers windowing plus Levinson-Durbin LPC, split-VQ LSF
//! quantisation with stability check and interpolation, residual
//! analysis, the start-state classifier and scalar coder, and the
//! symmetric forward and backward adaptive-codebook walks.
//!
//! Scenarios cover both frame modes plus a longer 20 ms clip so the
//! per-frame setup cost amortises.
//!
//!   - **encode_mono_8k_20ms_1s**: 1 s of mono S16 PCM at 8 kHz,
//!     20 ms framing (15.20 kbit/s, 50 packets/s). Smallest-latency
//!     working set; the LPC + LSF quantise + CB search runs the
//!     short variant (cb_sub_blocks=2) on every packet.
//!   - **encode_mono_8k_30ms_1s**: 1 s of the same source, 30 ms
//!     framing (13.33 kbit/s, ~33 packets/s). Doubles the LSF
//!     vectors transmitted per frame and runs the longer
//!     cb_sub_blocks=4 codebook walk.
//!   - **encode_mono_8k_20ms_3s**: 3 s clip, 20 ms framing. Longer
//!     run lets the encoder's per-frame book-keeping (cb_mem state,
//!     `prev_a_per_sub` carry-over) reach steady state so the bench
//!     picks up the hot-loop cost rather than initialisation.
//!
//! Run with:
//!     cargo bench -p oxideav-ilbc --bench encode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, Encoder, Error, Frame, SampleFormat,
};
use oxideav_ilbc::{encoder, FrameMode, CODEC_ID_STR, SAMPLE_RATE};

/// Deterministic xorshift32 — shared with the sibling bench files.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Mono S16 voiced-like clip (fundamental + three harmonics + small
/// noise floor). Same generator as `benches/decode.rs` so the two
/// scenarios share an input distribution.
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

/// Drive the encoder through one full clip and count the emitted
/// packets. The encoder is rebuilt per iteration so the bench
/// captures `make_encoder` allocation + every frame's per-packet
/// cost — the cost a streaming sender actually pays at the start of
/// a new call.
fn run_encode(mode: FrameMode, pcm_bytes: &[u8], samples: u32) -> usize {
    let params = ilbc_params(mode);
    let mut enc: Box<dyn Encoder> = encoder::make_encoder(&params).expect("make_encoder");
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples,
        pts: Some(0),
        data: vec![pcm_bytes.to_vec()],
    }))
    .expect("send_frame");
    enc.flush().expect("flush");
    let mut count = 0usize;
    loop {
        match enc.receive_packet() {
            Ok(_) => count += 1,
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("unexpected encoder error: {:?}", e),
        }
    }
    count
}

fn bench_encode_mono_8k_20ms_1s(c: &mut Criterion) {
    let n = 8_000;
    let pcm = build_pcm_bytes(n, 0xCAFE_F00D);
    let mut g = c.benchmark_group("encode_mono_8k_20ms_1s");
    g.throughput(Throughput::Bytes((n * 2) as u64));
    g.bench_function(BenchmarkId::from_parameter("mono/8k/20ms/1s"), |b| {
        b.iter(|| {
            let n_pkts = run_encode(FrameMode::Ms20, criterion::black_box(&pcm), n as u32);
            criterion::black_box(n_pkts);
        });
    });
    g.finish();
}

fn bench_encode_mono_8k_30ms_1s(c: &mut Criterion) {
    let n = 8_000;
    let pcm = build_pcm_bytes(n, 0xBEEF_1234);
    let mut g = c.benchmark_group("encode_mono_8k_30ms_1s");
    g.throughput(Throughput::Bytes((n * 2) as u64));
    g.bench_function(BenchmarkId::from_parameter("mono/8k/30ms/1s"), |b| {
        b.iter(|| {
            let n_pkts = run_encode(FrameMode::Ms30, criterion::black_box(&pcm), n as u32);
            criterion::black_box(n_pkts);
        });
    });
    g.finish();
}

fn bench_encode_mono_8k_20ms_3s(c: &mut Criterion) {
    let n = 24_000;
    let pcm = build_pcm_bytes(n, 0xDECA_FBAD);
    let mut g = c.benchmark_group("encode_mono_8k_20ms_3s");
    g.throughput(Throughput::Bytes((n * 2) as u64));
    g.bench_function(BenchmarkId::from_parameter("mono/8k/20ms/3s"), |b| {
        b.iter(|| {
            let n_pkts = run_encode(FrameMode::Ms20, criterion::black_box(&pcm), n as u32);
            criterion::black_box(n_pkts);
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_encode_mono_8k_20ms_1s,
    bench_encode_mono_8k_30ms_1s,
    bench_encode_mono_8k_20ms_3s,
);
criterion_main!(benches);
