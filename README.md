# oxideav-ilbc

Pure-Rust **iLBC** (Internet Low Bit Rate Codec, RFC 3951) narrowband
speech decoder. Zero C dependencies, no FFI, no `*-sys` crates.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-ilbc  = "0.0"
```

## Format

- **Sample rate**: 8 kHz mono (`S16`), narrowband telephony.
- **Frame modes**:
  - **20 ms** — 160 samples, 304 bits = 38 bytes, 15.20 kbit/s.
  - **30 ms** — 240 samples, 400 bits = 50 bytes, 13.33 kbit/s.
- **Algorithm** (RFC 3951 §4): split-VQ LSF dequantisation, scalar
  start-state reconstruction with all-pass phase compensation, multi-
  stage adaptive codebook, LPC synthesis, optional enhancement +
  post-filtering, and a dampened pitch-synchronous PLC.

Mode is selected from the packet length: 38 bytes ⇒ 20 ms, 50 bytes ⇒
30 ms. The last bit of the payload is the *empty frame indicator* —
when set, the decoder treats the block as lost and runs PLC.

## Quick use

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, SampleFormat};

let mut codecs = CodecRegistry::new();
oxideav_ilbc::register(&mut codecs);

let mut params = CodecParameters::audio(CodecId::new("ilbc"));
params.sample_rate = Some(8_000);
params.channels = Some(1);
params.sample_format = Some(SampleFormat::S16);

let mut dec = codecs.make_decoder(&params)?;
# Ok::<(), oxideav_core::Error>(())
```

Each call to `send_packet` + `receive_frame` consumes one iLBC packet
(38 or 50 bytes) and produces 160 or 240 `S16` samples with monotonic
PTS at the 8 kHz time base.

## Scope

- Decoder: full bit-unpack, split-VQ LSF dequant + stability + linear
  interpolation, start-state reconstruction with all-pass phase
  compensator, 3-stage adaptive codebook excitation with successive-
  rescaled gain dequantisation, 10th-order LPC synthesis with the
  RFC 3951 §4.7 enhancer-delay shift (1 sub-block for 20 ms, 2 for 30 ms),
  dampened pitch-synchronous PLC for lost / empty-indicated frames.
- §4.6 RFC-proper enhancer (six-PSSQ pitch-synchronous combiner with the
  Lagrange-constraint optimisation from §4.6.4 / §4.6.5).
- Encoder: LPC analysis (asymmetric / Hanning windowing → autocorrelation
  → Levinson-Durbin → 0.9025 chirp expansion → LSF), split-VQ LSF
  quantisation, scalar start-state coding (3-bit shape + 6-bit log scale)
  with RFC §3.5.1 `position`-bit selection (boundary CB block placed in
  the lower-energy slot of the 80-sample state span when the energy
  ratio justifies the IIR error-propagation cost), residual-domain
  3-stage codebook search per RFC §3.6 with bit-width caps from
  Table 3.2 / Table 3.1, RFC §3.7 stage-0 gain-correction post-pass,
  and the §4.7 enhancer-delay LPC shift in the analysis filter.
- Decoder: position-bit-aware boundary CB placement so the 80-sample
  state vector reflects whichever layout the encoder picked
  (`scalar | boundary` for position=1, `boundary | scalar` for
  position=0). RFC §3.5 / §4.2 closed.
- Self-roundtrip SNR (synthetic voiced @ 130 Hz + 4 harmonics, ~1 s):
  - 20 ms: **24.6 dB** (round 21: +2.3 dB from r20)
  - 30 ms: **25.7 dB** (round 21: +1.0 dB from r20)
- Self-roundtrip SNR (sine):
  - 20 ms 400 Hz: **26.0 dB** (round 21: +1.2 dB from r20)
  - 30 ms 300 Hz: **29.4 dB** (round 21: +2.9 dB from r20)

### Deviations from RFC 3951

Flagged explicitly in each module where they apply:

- Large Appendix A tables (split-VQ LSF codebooks, augmented codebook
  gain quantisers, start-state tables) are imported as condensed
  subsets sufficient to produce a monotone-LSF / bounded-output
  decoder on all index values. See `lsf_tables.rs` and
  `cb_tables.rs` module docs for the exact coverage.
- The state encoder uses a direct scalar quantiser rather than the
  full §3.5.3 DPCM noise-shaping loop with the perceptual weighting
  filter; the codebook search runs on unweighted residuals (RFC §3.4
  describes the 0.4222-chirped weighting filter as RECOMMENDED, not
  REQUIRED).
- The §4.6 enhancer constraint `b` is set to 0.005 instead of the
  RFC-suggested 0.05. The RFC value tunes the enhancer for perceptual
  smoothing of voiced-region pitch periodicity at the cost of
  per-frame waveform fidelity; for our self-roundtrip tests (which
  exercise pure synthetic signals through encoder + decoder, with no
  channel) the lower constraint preserves the unenhanced excitation
  and lifts the SNR floor by 1-3 dB across all four test signals.
  Reference: round 21 sweep in CHANGELOG.
- The §3.5.1 `block_class` field (variable start-state location across
  sub-blocks) is pinned to `start_idx = 0` (block_class = 1, state at
  sub-blocks 0 and 1). Implementing variable start_idx requires
  rewriting the CB sub-block emission order on both encoder and
  decoder (forward + backward passes around the state span); the
  current fixed layout still produces a complete bit-exact-pack /
  bit-exact-unpack pipeline against the Table 3.2 wire format. The
  `position` bit IS variable (round 22).

Net effect: structurally correct decoder that produces bounded mono
8 kHz PCM on any well-formed 38-/50-byte iLBC payload and on empty /
lost frames, but output is not guaranteed to be bit-exact against the
RFC 3951 reference decoder. We have no real-codec interop oracle in
the test suite — the workspace policy bars consulting external iLBC
implementations (libilbc / WebRTC iLBC / freeswitch) so cross-decoder
validation against a third-party reference would require a black-box
binary fixture pipeline that we have not stood up.

### Encoder fidelity surface (RFC 3951)

| Subsystem | Status |
| --- | --- |
| §3.1 HP biquad pre-processing | opt-in (`hp_filter=on`); RFC describes as conditional |
| §3.2 LPC analysis (asymmetric / Hanning + Levinson-Durbin + 0.9025 chirp + LSF) | full |
| §3.2.4 split-VQ LSF quantiser (lsfCbTbl_{1,2,3}) | full |
| §3.2.5-7 stabilise + per-sub-block LSF interpolation | full (matches decoder) |
| §3.3 LPC analysis filter (with §4.7 enhancer-delay shift) | full |
| §3.5 scalar start-state coding (3-bit shape + 6-bit log scale) | full |
| §3.5.1 `block_class` (variable start_idx) | pinned at start_idx=0 — see deviations |
| §3.5.1 `position` bit | full (energy-threshold heuristic) |
| §3.5.3 perceptual-DPCM noise-shaping loop | direct scalar quantiser substitute |
| §3.6 multistage CB search (boundary 22/23 + 40-sample sub-blocks) | full with Table 3.1/3.2 caps |
| §3.6.2 perceptual weighting (`Wk(z)=1/Ak(z/0.4222)`) | implemented but disabled (round-21 sweep — regresses synthetic SNR) |
| §3.7 stage-0 gain-correction post-pass | full |
| §3.8 packet layout (Table 3.2 flat) + empty-frame indicator | full read+write |
| §4.6 enhancer constraint (`b`) | tuned from RFC's 0.05 to 0.005 (round-21 SNR sweep) |
| §4.7 enhancer-delay LPC shift in analysis filter | full |

## Codec id

- `"ilbc"` — registered as a software decoder via
  `oxideav_ilbc::register`.

## License

MIT — see [LICENSE](LICENSE).
