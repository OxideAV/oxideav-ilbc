//! Criterion benchmarks for the RFC 3952 RTP payload-format hot path.
//!
//! Round 235 (depth-mode benchmarks): companion to `decode.rs`,
//! `encode.rs`, and `roundtrip.rs`. Those three time the audio
//! codec proper; this one times the *transport* surface a streaming
//! receiver / sender pays once per RTP packet — the
//! [`oxideav_ilbc::rtp::Packetiser`] that fans iLBC frames into
//! RTP payload bodies, and the [`oxideav_ilbc::rtp::Depacketiser`]
//! that splits an inbound payload back into per-frame slices the
//! decoder ingests.
//!
//! The RTP layer is not on the audio decode path — it sits one
//! layer up, after the kernel hands the RTP payload to the codec
//! library. But for a streaming endpoint terminating thousands of
//! RTP sessions, the pack / depack cost compounds quickly: every
//! 20 ms a 20 ms-mode session pays one
//! `Packetiser::pack_series` call (transmit side) and one
//! `Depacketiser::depacketise` call (receive side). Each of those
//! allocates a `Vec<u8>` of `frame_size * frames_per_packet`
//! bytes on the transmit side and `frames_per_packet`
//! `&[u8]` slices on the receive side. Both are O(n) in the
//! packed frame count, but the constants matter — a needless
//! per-call allocation in the steady-state path shows up as
//! measurable CPU at scale.
//!
//! These benches A/B-test those calls so any future cleanup of
//! the allocation pattern (e.g. amortising the per-packet
//! `Vec::with_capacity`, switching `chunks_exact` to a hand-rolled
//! slice walk, returning an iterator instead of a `Vec`) has a
//! stable baseline. The depacketise + pack-series round-trip in
//! particular models the "B2BUA" / SFU / WebRTC-gateway
//! workload where the same iLBC bytes enter as one packet and
//! leave as another after a possible re-aggregation.
//!
//! No `docs/` fixtures or external files are read. Every input
//! buffer is built in-bench from a deterministic xorshift seed,
//! sized to the per-mode frame width (38 / 50 bytes).
//!
//! Scenarios:
//!
//!   - **rtp_pack_20ms_50f**: pack 50 × 38-byte iLBC frames (1 s
//!     of 20 ms audio) through `Packetiser::pack_series` with the
//!     default 8-frames-per-packet cap. Times the transmit side
//!     for a typical 1 s outbound burst.
//!   - **rtp_pack_30ms_33f**: pack 33 × 50-byte iLBC frames (~1 s
//!     of 30 ms audio) through `Packetiser::pack_series`. 33 frames
//!     × 50 bytes is the 30 ms-mode analogue of the 1 s 20 ms
//!     scenario.
//!   - **rtp_depack_20ms_8f**: depacketise one RTP payload
//!     carrying 8 × 38-byte iLBC frames (`Depacketiser::depacketise`,
//!     borrowed). Models a single inbound packet at the receiver.
//!   - **rtp_depack_30ms_5f**: depacketise one RTP payload carrying
//!     5 × 50-byte iLBC frames (250 bytes — fits comfortably in a
//!     1500-byte IPv4 MTU after RTP / UDP / IP overhead).
//!   - **rtp_depack_owned_20ms_8f**: same input as `_depack_20ms_8f`
//!     but through `depacketise_owned`, which copies each frame
//!     into its own `Vec<u8>` (the caller path that hands frames
//!     to the audio decoder's `Packet`-owning `send_packet`).
//!   - **rtp_roundtrip_20ms_50f**: full pack → depack chain on a
//!     50-frame batch — a B2BUA-style relay model where the same
//!     frame buffers are repacked and split again.
//!
//! Run with:
//!     cargo bench -p oxideav-ilbc --bench rtp

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_ilbc::rtp::{Depacketiser, Packetiser};
use oxideav_ilbc::{FrameMode, FRAME_BYTES_20MS, FRAME_BYTES_30MS};

/// Deterministic xorshift32 — shared with the sibling bench files
/// so every scenario draws from the same input distribution.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build `n_frames` iLBC frame buffers of `frame_size` bytes each.
/// Contents are pseudo-random per the seed; the depacketiser /
/// packetiser does not inspect the bytes (only the length), so any
/// random fill keeps the bench honest by preventing any
/// `chunks_exact` short-circuit on a zero buffer.
fn build_frames(n_frames: usize, frame_size: usize, seed: u32) -> Vec<Vec<u8>> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n_frames);
    for _ in 0..n_frames {
        let mut frame = Vec::with_capacity(frame_size);
        for _ in 0..frame_size {
            frame.push((xorshift32(&mut state) & 0xFF) as u8);
        }
        out.push(frame);
    }
    out
}

/// Materialise a single concatenated payload of `n_frames * frame_size`
/// bytes from the same xorshift seed. Used by the depacketise benches
/// so the timed body just calls `depacketise(&payload)` on a pre-built
/// buffer.
fn build_payload(n_frames: usize, frame_size: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let total = n_frames * frame_size;
    let mut out = Vec::with_capacity(total);
    for _ in 0..total {
        out.push((xorshift32(&mut state) & 0xFF) as u8);
    }
    out
}

fn bench_rtp_pack_20ms_50f(c: &mut Criterion) {
    let frames = build_frames(50, FRAME_BYTES_20MS, 0xC0DE_BABE);
    let frame_refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
    let pk = Packetiser::new(FrameMode::Ms20);
    let mut g = c.benchmark_group("rtp_pack_20ms_50f");
    g.throughput(Throughput::Bytes((50 * FRAME_BYTES_20MS) as u64));
    g.bench_function(BenchmarkId::from_parameter("50x38B/cap8"), |b| {
        b.iter(|| {
            let series = pk
                .pack_series(criterion::black_box(&frame_refs))
                .expect("pack_series");
            criterion::black_box(series);
        });
    });
    g.finish();
}

fn bench_rtp_pack_30ms_33f(c: &mut Criterion) {
    let frames = build_frames(33, FRAME_BYTES_30MS, 0xFEED_FACE);
    let frame_refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
    let pk = Packetiser::new(FrameMode::Ms30);
    let mut g = c.benchmark_group("rtp_pack_30ms_33f");
    g.throughput(Throughput::Bytes((33 * FRAME_BYTES_30MS) as u64));
    g.bench_function(BenchmarkId::from_parameter("33x50B/cap8"), |b| {
        b.iter(|| {
            let series = pk
                .pack_series(criterion::black_box(&frame_refs))
                .expect("pack_series");
            criterion::black_box(series);
        });
    });
    g.finish();
}

fn bench_rtp_depack_20ms_8f(c: &mut Criterion) {
    let payload = build_payload(8, FRAME_BYTES_20MS, 0x1234_5678);
    let dp = Depacketiser::new(FrameMode::Ms20);
    let mut g = c.benchmark_group("rtp_depack_20ms_8f");
    g.throughput(Throughput::Bytes((8 * FRAME_BYTES_20MS) as u64));
    g.bench_function(BenchmarkId::from_parameter("8x38B"), |b| {
        b.iter(|| {
            let frames = dp
                .depacketise(criterion::black_box(&payload))
                .expect("depacketise");
            criterion::black_box(frames);
        });
    });
    g.finish();
}

fn bench_rtp_depack_30ms_5f(c: &mut Criterion) {
    let payload = build_payload(5, FRAME_BYTES_30MS, 0x8765_4321);
    let dp = Depacketiser::new(FrameMode::Ms30);
    let mut g = c.benchmark_group("rtp_depack_30ms_5f");
    g.throughput(Throughput::Bytes((5 * FRAME_BYTES_30MS) as u64));
    g.bench_function(BenchmarkId::from_parameter("5x50B"), |b| {
        b.iter(|| {
            let frames = dp
                .depacketise(criterion::black_box(&payload))
                .expect("depacketise");
            criterion::black_box(frames);
        });
    });
    g.finish();
}

fn bench_rtp_depack_owned_20ms_8f(c: &mut Criterion) {
    let payload = build_payload(8, FRAME_BYTES_20MS, 0xABCD_EF01);
    let dp = Depacketiser::new(FrameMode::Ms20);
    let mut g = c.benchmark_group("rtp_depack_owned_20ms_8f");
    g.throughput(Throughput::Bytes((8 * FRAME_BYTES_20MS) as u64));
    g.bench_function(BenchmarkId::from_parameter("8x38B/owned"), |b| {
        b.iter(|| {
            let frames = dp
                .depacketise_owned(criterion::black_box(&payload))
                .expect("depacketise_owned");
            criterion::black_box(frames);
        });
    });
    g.finish();
}

fn bench_rtp_roundtrip_20ms_50f(c: &mut Criterion) {
    let frames = build_frames(50, FRAME_BYTES_20MS, 0x0BAD_BEEF);
    let frame_refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
    let pk = Packetiser::new(FrameMode::Ms20);
    let dp = Depacketiser::new(FrameMode::Ms20);
    let mut g = c.benchmark_group("rtp_roundtrip_20ms_50f");
    g.throughput(Throughput::Bytes((50 * FRAME_BYTES_20MS) as u64));
    g.bench_function(BenchmarkId::from_parameter("pack+depack/50x38B"), |b| {
        b.iter(|| {
            let series = pk
                .pack_series(criterion::black_box(&frame_refs))
                .expect("pack_series");
            let mut total = 0usize;
            for (body, _ts) in &series {
                let parts = dp.depacketise(body).expect("depacketise");
                total += parts.len();
            }
            criterion::black_box(total);
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_rtp_pack_20ms_50f,
    bench_rtp_pack_30ms_33f,
    bench_rtp_depack_20ms_8f,
    bench_rtp_depack_30ms_5f,
    bench_rtp_depack_owned_20ms_8f,
    bench_rtp_roundtrip_20ms_50f,
);
criterion_main!(benches);
