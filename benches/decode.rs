//! Criterion benchmarks for the iLBC decoder hot path.
//!
//! Round 180 (depth-mode benchmarks): paired with `encode.rs` and
//! `roundtrip.rs`. Each scenario synthesises a deterministic mono
//! S16 PCM clip at the codec's native 8 kHz, encodes it once with the
//! production `oxideav_ilbc::encoder::make_encoder` trait object (so
//! the bench sees a realistic mix of LSF split-VQ outputs, start-state
//! scalar codes, and adaptive-codebook gain indices), and then
//! iterates the decoder on the resulting packet stream. The encode
//! work is **outside** the timed region — only the decoder's
//! per-packet `send_packet -> receive_frame` cost is measured, so the
//! number reflects what a streaming receiver actually pays.
//!
//! Scenarios:
//!
//!   - **decode_mono_8k_20ms_1s**: 1 s of mono S16 PCM at 8 kHz coded
//!     as 50 × 20 ms frames. Exercises the 20 ms split-VQ LSF
//!     decoder, the 57-sample start-state path, and the smaller
//!     cb_sub_blocks=2 codebook walk.
//!   - **decode_mono_8k_30ms_1s**: same source, coded as 33 × 30 ms
//!     frames. Activates the 30 ms split-VQ table (2 LSF vectors),
//!     58-sample start state, and the cb_sub_blocks=4 codebook walk.
//!     Higher per-frame cost than 20 ms but fewer frames per second.
//!   - **decode_mono_8k_20ms_3s**: 3 s of PCM, 20 ms framing. Longer
//!     run lets the enhancer's pitch-buffer history reach steady
//!     state so the bench picks up the post-filter cost rather than
//!     the cold-start ramp.
//!
//! Run with:
//!     cargo bench -p oxideav-ilbc --bench decode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, Encoder, Error, Frame, Packet,
    SampleFormat, TimeBase,
};
use oxideav_ilbc::{decoder, encoder, FrameMode, CODEC_ID_STR, SAMPLE_RATE};

/// Deterministic xorshift32 — shared with the sibling bench files so
/// every scenario draws from the same input distribution.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build a mono S16 clip that mixes a fundamental, three harmonics,
/// and a small noise floor. The harmonic content keeps the LPC
/// analysis producing non-trivial residuals; the noise floor keeps the
/// codebook search from degenerating to a single bucket. Returned as
/// little-endian S16 bytes ready for `AudioFrame::data[0]`.
fn build_pcm_bytes(n_samples: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n_samples * 2);
    let f0 = 145.0f32; // adult-male-ish pitch
    let sr = SAMPLE_RATE as f32;
    for n in 0..n_samples {
        let t = n as f32 / sr;
        let mut v = 0.0f32;
        for h in 1..=4 {
            v += (2.0 * core::f32::consts::PI * (h as f32) * f0 * t).sin() * (4200.0 / h as f32);
        }
        // Low-amplitude noise — keeps the residual non-degenerate.
        let noise = (xorshift32(&mut state) as i32 >> 22) as f32;
        let s = (v + noise).round().clamp(-32768.0, 32767.0) as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn ilbc_params() -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    p.sample_rate = Some(SAMPLE_RATE);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    p
}

/// Encode `pcm_bytes` with the production encoder into a packet
/// stream. Done once in scenario setup; the timed bench body only
/// iterates the decoder over the captured packets.
fn prepare_packets(mode: FrameMode, pcm_bytes: &[u8], samples: u32) -> Vec<Packet> {
    let mut params = ilbc_params();
    if mode == FrameMode::Ms30 {
        params.options = CodecOptions::new().set("frame_ms", "30");
    }
    let mut enc: Box<dyn Encoder> = encoder::make_encoder(&params).expect("make_encoder");
    enc.send_frame(&Frame::Audio(AudioFrame {
        samples,
        pts: Some(0),
        data: vec![pcm_bytes.to_vec()],
    }))
    .expect("send_frame");
    enc.flush().expect("flush");
    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(Packet::new(
                0,
                TimeBase::new(1, SAMPLE_RATE as i64),
                p.data.clone(),
            )),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("unexpected encoder error: {:?}", e),
        }
    }
    packets
}

/// Run the decoder over the packet stream and return the total
/// number of decoded PCM bytes. The return value is fed to
/// `black_box` so the compiler can't elide the work.
fn run_decode(packets: &[Packet]) -> usize {
    let params = ilbc_params();
    let mut dec = decoder::make_decoder(&params).expect("make_decoder");
    let mut total = 0usize;
    for pkt in packets {
        dec.send_packet(pkt).expect("send_packet");
        if let Frame::Audio(a) = dec.receive_frame().expect("receive_frame") {
            total += a.data[0].len();
        }
    }
    total
}

fn bench_decode_mono_8k_20ms_1s(c: &mut Criterion) {
    let n = 8_000;
    let pcm = build_pcm_bytes(n, 0xCAFE_F00D);
    let packets = prepare_packets(FrameMode::Ms20, &pcm, n as u32);
    let mut g = c.benchmark_group("decode_mono_8k_20ms_1s");
    g.throughput(Throughput::Bytes((n * 2) as u64));
    g.bench_function(BenchmarkId::from_parameter("mono/8k/20ms/1s"), |b| {
        b.iter(|| {
            let total = run_decode(criterion::black_box(&packets));
            criterion::black_box(total);
        });
    });
    g.finish();
}

fn bench_decode_mono_8k_30ms_1s(c: &mut Criterion) {
    let n = 8_000;
    let pcm = build_pcm_bytes(n, 0xBEEF_1234);
    let packets = prepare_packets(FrameMode::Ms30, &pcm, n as u32);
    let mut g = c.benchmark_group("decode_mono_8k_30ms_1s");
    g.throughput(Throughput::Bytes((n * 2) as u64));
    g.bench_function(BenchmarkId::from_parameter("mono/8k/30ms/1s"), |b| {
        b.iter(|| {
            let total = run_decode(criterion::black_box(&packets));
            criterion::black_box(total);
        });
    });
    g.finish();
}

fn bench_decode_mono_8k_20ms_3s(c: &mut Criterion) {
    let n = 24_000;
    let pcm = build_pcm_bytes(n, 0xDECA_FBAD);
    let packets = prepare_packets(FrameMode::Ms20, &pcm, n as u32);
    let mut g = c.benchmark_group("decode_mono_8k_20ms_3s");
    g.throughput(Throughput::Bytes((n * 2) as u64));
    g.bench_function(BenchmarkId::from_parameter("mono/8k/20ms/3s"), |b| {
        b.iter(|| {
            let total = run_decode(criterion::black_box(&packets));
            criterion::black_box(total);
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_decode_mono_8k_20ms_1s,
    bench_decode_mono_8k_30ms_1s,
    bench_decode_mono_8k_20ms_3s,
);
criterion_main!(benches);
