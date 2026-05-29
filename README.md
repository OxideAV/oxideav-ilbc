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
  with RFC §3.5.1 variable `start_idx` (windowed energy classifier picks
  the highest-energy 80-sample state span across all `n_sub-1`
  candidate positions; `block_class` carries the 1-based start index
  on the wire) and `position`-bit selection (boundary CB block placed
  in the lower-energy slot of the chosen state span when the energy
  ratio justifies the IIR error-propagation cost). Residual-domain
  3-stage codebook search per RFC §3.6 with bit-width caps from
  Table 3.2 / Table 3.1, walked symmetrically forward + backward
  around the state span (Nfor sub-blocks at `[(start+1)*SUBL ..]`,
  Nback sub-blocks at `[0..(start-1)*SUBL]` in reversed time). Plus
  RFC §3.7 stage-0 gain-correction post-pass and the §4.7
  enhancer-delay LPC shift in the analysis filter.
- Decoder: variable-`start_idx`-aware reconstruction — `block_class`
  selects the start-state sub-block pair, `position` selects within
  that pair, and the symmetric forward+backward CB walk reproduces
  the encoder's emission order. RFC §3.5 / §4.2 / §3.6.1 closed.
- Self-roundtrip SNR (synthetic voiced @ 130 Hz + 4 harmonics, ~1 s):
  - 20 ms: **25.0 dB** (round 23: +0.4 dB from r22, variable start_idx)
  - 30 ms: **27.1 dB** (round 23: +1.4 dB from r22, variable start_idx)
- Self-roundtrip SNR (sine):
  - 20 ms 400 Hz: **23.9 dB** (round 23: -2.1 dB from r22 — FrameClassify
    picks the centre window for steady tones; voiced SNR is the
    speech-relevant metric and improved.)
  - 30 ms 300 Hz: **28.6 dB** (round 23: -0.8 dB from r22)

### Deviations from RFC 3951

Flagged explicitly in each module where they apply:

- Large Appendix A tables (split-VQ LSF codebooks, augmented codebook
  gain quantisers, start-state tables) are imported as condensed
  subsets sufficient to produce a monotone-LSF / bounded-output
  decoder on all index values. See `lsf_tables.rs` and
  `cb_tables.rs` module docs for the exact coverage.
- The §3.5.3 DPCM noise-shaping start-state quantiser (Appendix A.46
  `AbsQuantW`, perceptual weighting via `Ak(z/0.4222)`) is implemented
  but **off by default** (`state_dpcm=on` to enable). Like the §3.6.2
  codebook-search weighting, the perceptual weighting regresses
  waveform SNR on the synthetic self-roundtrip signals, so the default
  path uses a direct per-sample scalar quantiser on the unweighted
  scaled residual. Both paths emit `state_sq3Tbl` indices that the
  decoder reads back identically (RFC §4.2 / Appendix A.44
  `StateConstructW` applies no inverse weighting), so the toggle never
  affects decode semantics. The codebook search also runs on unweighted
  residuals (RFC §3.4 describes the 0.4222-chirped weighting filter as
  RECOMMENDED, not REQUIRED).
- The §4.6 enhancer constraint `b` is set to 0.005 instead of the
  RFC-suggested 0.05. The RFC value tunes the enhancer for perceptual
  smoothing of voiced-region pitch periodicity at the cost of
  per-frame waveform fidelity; for our self-roundtrip tests (which
  exercise pure synthetic signals through encoder + decoder, with no
  channel) the lower constraint preserves the unenhanced excitation
  and lifts the SNR floor by 1-3 dB across all four test signals.
  Reference: round 21 sweep in CHANGELOG.
- The `tests/docs_corpus.rs` driver decodes FFmpeg-encoded fixtures
  successfully across all 16 cases. As of round 173 every fixture
  carries a per-case `Tier::PsnrFloor` regression gate anchored 2-3 dB
  beneath the observed PSNR: silence 70 dB (vs baseline ~74 dB),
  step-impulse 30 dB (vs 34 dB), voiced / sine / dtmf 13-15 dB (vs
  16-19 dB), noise 9-10 dB (vs 12-13 dB). The margin absorbs sub-LSB
  cross-runner float drift in the CELP path (LSF→LPC rounding, the
  §4.2 all-pass phase compensator, the optional §4.6 enhancer, and
  post-filter) while still red-lighting CI on any per-fixture
  regression bigger than the margin. Tighter case-by-case floors are
  a follow-up once a cross-runner span (linux x86-64 / macOS aarch64)
  has been catalogued.

Net effect: structurally complete encoder + decoder that produces
bounded mono 8 kHz PCM on any well-formed 38-/50-byte iLBC payload
and on empty / lost frames; output is not guaranteed to be bit-exact
against the RFC 3951 reference (CELP rounding drift) and we have no
third-party encoder oracle in CI. Workspace policy bars consulting
libilbc / WebRTC iLBC / freeswitch / ffmpeg's iLBC source as a
clean-room reference, so cross-encoder validation against a known
good third-party encoder would require a black-box binary fixture
pipeline that we have not stood up.

### Encoder fidelity surface (RFC 3951)

| Subsystem | Status |
| --- | --- |
| §3.1 HP biquad pre-processing | opt-in (`hp_filter=on`); RFC describes as conditional |
| §3.2 LPC analysis (asymmetric / Hanning + Levinson-Durbin + 0.9025 chirp + LSF) | full |
| §3.2.4 split-VQ LSF quantiser (lsfCbTbl_{1,2,3}) | full |
| §3.2.5-7 stabilise + per-sub-block LSF interpolation | full (matches decoder) |
| §3.3 LPC analysis filter (with §4.7 enhancer-delay shift) | full |
| §3.5 scalar start-state coding (3-bit shape + 6-bit log scale) | full |
| §3.5.1 `block_class` (variable `start_idx` via FrameClassify) | full (round 23 — windowed energy classifier from Appendix A.20) |
| §3.5.1 `position` bit | full (RFC §3.5.1 `en1 vs en2` test + 4× IIR-error-propagation guard) |
| §3.5.3 perceptual-DPCM noise-shaping loop (`AbsQuantW`) | implemented (`state_dpcm=on`); direct scalar quantiser is the SNR-preserving default |
| §3.6 multistage CB search (boundary 22/23 + 40-sample sub-blocks, symmetric forward + backward walk) | full with Table 3.1/3.2 caps |
| §3.6.2 perceptual weighting (`Wk(z)=1/Ak(z/0.4222)`) | implemented but disabled (round-21 sweep — regresses synthetic SNR) |
| §3.7 stage-0 gain-correction post-pass | full |
| §3.8 packet layout (Table 3.2 flat) + empty-frame indicator | full read+write |
| §4.6 enhancer constraint (`b`) | tuned from RFC's 0.05 to 0.005 (round-21 SNR sweep) |
| §4.7 enhancer-delay LPC shift in analysis filter | full |

## Benchmarks

Criterion harnesses in `benches/` time the decoder hot path, the
encoder hot path, and the paired round-trip through the public
trait surface. Every PCM input is synthesised in-bench from a
deterministic xorshift32 seed, so the harnesses ship no committed
fixture files and read nothing under `docs/`.

```sh
cargo bench -p oxideav-ilbc --bench decode
cargo bench -p oxideav-ilbc --bench encode
cargo bench -p oxideav-ilbc --bench roundtrip
```

Each harness covers three scenarios: 20 ms framing × 1 s,
30 ms framing × 1 s, and 20 ms framing × 3 s (the long clip lets
the enhancer pitch buffer and the encoder's `prev_a_per_sub`
carry-over reach steady state).

## Codec id

- `"ilbc"` — registered as a software decoder via
  `oxideav_ilbc::register`.

## License

MIT — see [LICENSE](LICENSE).
