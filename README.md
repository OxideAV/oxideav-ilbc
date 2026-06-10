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
  dampened pitch-synchronous PLC for lost / empty-indicated frames,
  optional §4.8 65 Hz output HP post-filter (opt-in via `hp_filter=on`
  in `CodecParameters::options`).
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

- The split-VQ LSF codebooks (`LSF_CB_TBL_{1,2,3}`, the full
  64×3 + 128×3 + 128×4 = 1088-entry table), the LSF mean vector, the
  three gain quantisers (`GAIN_SQ{3,4,5}_TBL`), and the start-state
  scalar quantiser (`STATE_SQ3_TBL`) are the complete Appendix A
  tables, transcribed verbatim from the RFC 3951 decimal listing.
  `tests/table_provenance.rs` cross-checks all 1208 of these numeric
  facts against the independently extracted fixed-point (Q-domain)
  tables under `docs/audio/ilbc/tables/`: every crate `f32`, scaled
  into the matching Q-domain and rounded to nearest, equals the docs
  integer exactly (LSF + state at Q13, gains at Q14). Other large
  Appendix A tables (the enhancer / PLC / windowing coefficient
  tables) are imported as the subsets the decode + encode paths
  exercise; see the per-module docs for the exact coverage.
- **Wire-format bit layout matches RFC 3951 §3.8 ULP** as of round 219
  (previous rounds used a simplified flat layout — encoder and decoder
  agreed on it but it was incompatible with reference iLBC payloads).
  Per-parameter class-1 / class-2 / class-3 widths come from RFC
  Appendix A.41 `ULP_20msTbl` / `ULP_30msTbl`; the three-pass
  pack/unpack mirrors Appendix A.42 `unpack` / `packsplit` /
  `packcombine`. See `src/ulp.rs`.
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
  beneath the observed PSNR: silence 70 dB (post-r219 observed
  ~95 dB on 20 ms / ~97 dB on 30 ms, with 65-76 % sample-exact match
  after the ULP bit-layout fix), step-impulse 30 dB (observed
  ~39 dB), voiced / sine / dtmf 13-15 dB (observed 13-21 dB),
  noise 9-10 dB (observed 12-13 dB). The margin absorbs sub-LSB
  cross-runner float drift in the CELP path (LSF→LPC rounding, the
  §4.2 all-pass phase compensator, the optional §4.6 enhancer, and
  post-filter) while still red-lighting CI on any per-fixture
  regression bigger than the margin. Tighter case-by-case floors are
  a follow-up once a cross-runner span (linux x86-64 / macOS aarch64)
  has been catalogued.

Net effect: structurally complete encoder + decoder that produces
bounded mono 8 kHz PCM on any well-formed 38-/50-byte iLBC payload
and on empty / lost frames; bit layout matches RFC 3951 §3.8 ULP as
of round 219 (silence on either mode is sample-exact 65-76 % of the
time against the reference WAV, with sub-LSB CELP-pipeline drift on
the remaining samples). Output is not bit-exact end-to-end against
the RFC 3951 reference on speech / tones (CELP rounding drift) and
we have no third-party encoder oracle in CI. Workspace policy bars
consulting any external iLBC implementation as a clean-room
reference, so cross-encoder validation against a known-good
third-party encoder would require a black-box binary fixture
pipeline that we have not stood up.

### Decoder post-processing surface (RFC 3951)

| Subsystem | Status |
| --- | --- |
| §4.6 enhancer (six-PSSQ pitch-synchronous combiner) | full |
| §4.7 enhancer-delay LPC shift in synthesis filter | full |
| §4.8 65 Hz output HP biquad post-filter (`hpOutput`) | opt-in (`hp_filter=on`); RFC §4.8 marks as "if desired" |
| §4.5 dampened pitch-synchronous PLC | full (empty-frame indicator + zero-length packet) |

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
encoder hot path, the paired round-trip through the public trait
surface, and the RFC 3952 RTP packetiser / depacketiser. Every
input is synthesised in-bench from a deterministic xorshift32
seed, so the harnesses ship no committed fixture files and read
nothing under `docs/`.

```sh
cargo bench -p oxideav-ilbc --bench decode
cargo bench -p oxideav-ilbc --bench encode
cargo bench -p oxideav-ilbc --bench roundtrip
cargo bench -p oxideav-ilbc --bench rtp
```

The audio harnesses (`decode`, `encode`, `roundtrip`) each cover
three scenarios: 20 ms framing × 1 s, 30 ms framing × 1 s, and
20 ms framing × 3 s (the long clip lets the enhancer pitch
buffer and the encoder's `prev_a_per_sub` carry-over reach steady
state).

The `rtp` harness covers six scenarios on the RFC 3952 transport
surface (added round 235): per-mode `pack_series` on a 1 s frame
batch (50 × 38 B / 33 × 50 B), per-mode `depacketise` on one RTP
payload (8 × 38 B / 5 × 50 B), the owned-variant
`depacketise_owned` against the same input as the borrowed 20 ms
case, and a B2BUA-style `pack_series → depacketise` round-trip on
a 50-frame batch. Useful when A/B-testing future allocation /
chunking cleanups in `rtp::{Packetiser, Depacketiser}` — the
owned-vs-borrowed depacketise pair already pins the per-frame
`Vec<u8>` clone cost relative to the zero-copy slice walk
(measured ~11× gap on the 8-frame 20 ms input).

## Fuzzing

A `cargo-fuzz` harness in `fuzz/` exercises every attacker-facing
parse the crate ships, asserting *panic-freedom* + structural
invariants on arbitrary fuzz-supplied bytes:

```sh
cargo +nightly fuzz run decode
cargo +nightly fuzz run encode_roundtrip
cargo +nightly fuzz run rtp_depacketise
cargo +nightly fuzz run sdp_fmtp
cargo +nightly fuzz run rtp_gap_fill
```

- **`decode`** — drives raw byte payloads through the §3.8 bit-reader,
  through `make_decoder` + `send_packet` / `receive_frame`, and across
  sliding 20 ms (38 B) and 30 ms (50 B) windows on the *same* decoder
  instance to exercise the inter-frame enhancer + post-filter +
  `prev_a_per_sub` LPC-shift carry-over. Asserts the per-mode sample
  count and S16 byte count on every accepted packet.
- **`encode_roundtrip`** — drives arbitrary S16 PCM bytes through the
  encoder (mode + `hp_filter` + `state_dpcm` toggled from a seed
  byte) and pushes every emitted packet straight back through the
  decoder. Asserts every encoder-emitted packet is exactly 38 or 50
  bytes and that the decoder produces the matching `n*160` /
  `n*240`-sample audio frame without panicking, plus a panic-free
  `flush`.
- **`rtp_depacketise`** — drives the RFC 3952 RTP surface
  (`parse_mode_from_fmtp`, `Depacketiser::from_sdp_fmtp`,
  `Depacketiser::depacketise` borrowed + owned, `pack_series`,
  `detect_mode_from_payload_len`, `empty_marker_frame`). Asserts the
  borrowed and owned depacketise variants agree on every input, that
  every accepted depacketisation reconstitutes the input
  byte-for-byte, and that a `Packetiser::pack_series` →
  `Depacketiser::depacketise` round-trip preserves the original
  frame list and emits monotone-non-decreasing per-packet RTP
  timestamps.
- **`sdp_fmtp`** (round 243) — focused companion to `rtp_depacketise`
  that hands the *entire* fuzz input to `parse_mode_from_fmtp` as
  the SDP `a=fmtp:<pt> ...` value. `rtp_depacketise` threads only a
  length-prefixed slice of its input into the parser (the rest is
  the depacketise body); splitting the parser onto its own iteration
  budget lets libFuzzer spend its whole budget exploring the
  `;`-separated `key=value` grammar of RFC 3952 §4.2. Asserts
  panic-freedom on every byte sequence, that
  `Depacketiser::from_sdp_fmtp` agrees with `parse_mode_from_fmtp`
  on the accept / reject decision, and that every accepted mode
  round-trips through `format_mode_fmtp` and `build_fmtp`
  (`{ None, Some(0), Some(1), Some(2), Some(8), Some(255) }` cap
  ladder) back to the same `FrameMode`.
- **`rtp_gap_fill`** (round 246) — focused companion to
  `rtp_depacketise` that exercises the round-240 dropped-frame
  helpers (`rtp_seq_gap`, `Depacketiser::gap_frame_count`,
  `Depacketiser::conceal_gap`, `Depacketiser::concealment_payload`,
  `Depacketiser::depacketise_with_gap_fill`). Splits them onto a
  dedicated iteration budget so libFuzzer can spend its whole
  budget on the 16-bit RTP sequence-number arc fold, the saturating
  multiplication of `gap_packets * frames_per_payload`, the
  RFC 3951 §3.8 empty-frame-indicator placement on every emitted
  concealment slice, and the `depacketise(body) ↔
  depacketise_with_gap_fill` accept-decision parity. A 1024-frame
  cap on the per-iteration concealment count keeps the harness
  inside libFuzzer's iteration budget even on adversarial
  `(gap_packets, frames_per_payload)` pairs.

The targets share `oxideav-ilbc-fuzz` as a nested workspace
(`fuzz/Cargo.toml`) so the umbrella's `crates/*` glob does not pull
them in. Workspace policy bars consulting any external iLBC
implementation as a cross-decode oracle, so the five targets cover
the attacker surface end-to-end without an external comparison.

## RTP payload format (RFC 3952)

The `rtp` module implements the iLBC RTP payload format defined by
RFC 3952. It is deliberately a pure depacketiser / packetiser — the
12-byte fixed RTP header lives one layer above this crate and is
codec-agnostic.

Covered:

- **§3 *Payload Format*** — the iLBC RTP payload is the encoded
  bitstream itself with no per-packet codec header. One RTP packet
  MAY carry one or more iLBC frames, and all frames in a packet
  share the SDP-pinned mode (20 ms / 38 B or 30 ms / 50 B). The
  depacketiser splits a payload into fixed-size chunks for the
  decoder; the packetiser aggregates frames up to a per-packet
  cap and emits per-packet RTP-timestamp offsets (160 samples
  per 20 ms frame, 240 per 30 ms frame).
- **§4.2 *Mapping to SDP*** — `Depacketiser::from_sdp_fmtp` parses
  the `mode=20|30` parameter from an `a=fmtp:<pt> ...` line and
  pins the depacketiser to that mode. The `mode` key is matched
  case-insensitively per the SDP convention; unknown values and a
  missing parameter are hard errors (the receiver MUST know the
  mode out of band — falling back to a silent default would mask
  interop bugs). The outbound counterpart `format_mode_fmtp(mode)`
  emits the bare `mode=20` / `mode=30` token, and
  `build_fmtp(mode, max_frames_per_packet)` stitches the mode token
  together with an optional `;maxptime=M` cap (where
  `M = N * frame_ms`, mirroring `Packetiser::with_max_frames_per_packet`).
  A cap of 0 or 1 collapses to a bare `mode=N`. The emitted string
  round-trips back through `parse_mode_from_fmtp` and
  `Depacketiser::from_sdp_fmtp` to the same `FrameMode`.
- **Length-only mode hint** — `detect_mode_from_payload_len`
  inspects a payload whose mode has been lost in transit and
  reports `FrameMode::Ms20` / `Ms30` / `None` (ambiguous). Useful
  as a corruption check, not as a primary mode source.
- **Empty-frame surrogate** — `empty_marker_frame(mode)` yields a
  buffer the decoder treats as "packet lost; run PLC" (RFC 3951
  §3.8 / §4.5 — empty-frame indicator at LSB of the last byte).
- **Dropped-frame concealment** (round 240) — when an RTP receiver
  detects a sequence-number gap, the `Depacketiser::conceal_gap`,
  `Depacketiser::concealment_payload`, and
  `Depacketiser::depacketise_with_gap_fill` helpers produce the
  matching number of empty-marker frames to feed the decoder so
  the output PCM stream stays aligned with the wall-clock duration
  of the loss. The companion `rtp_seq_gap` free function does the
  RFC 3550 §3.3 16-bit-wrap-aware sequence-number arithmetic
  (forward deltas → "missing packets"; in-order, duplicate, and
  backward jumps collapse to zero).
- **Inbound `ptime` / `maxptime`** (round 258) — the
  `parse_ptime_from_fmtp` and `parse_maxptime_from_fmtp` free
  functions extract the optional RFC 4566 §6 `ptime` /
  `maxptime` parameters from an inbound `fmtp` line, and
  `max_frames_per_packet_from_fmtp(fmtp_value, mode)` derives a
  `Packetiser` per-packet cap straight from the parsed values
  (prefers `maxptime` over `ptime` when both are present;
  clamps degenerate sub-frame caps to 1). Closes the inbound
  mirror of the round-226 `build_fmtp` `;maxptime=M` emission:
  `build_fmtp(mode, Some(N))` round-trips through
  `max_frames_per_packet_from_fmtp(..., mode)` back to `N` for
  any `N >= 2`.
- **Outbound `ptime`** (round 267) — `build_fmtp_with_ptime(mode,
  ptime_frames, maxptime_frames)` closes the outbound mirror of
  `parse_ptime_from_fmtp`: the round-226 `build_fmtp` only emitted
  `mode` + `maxptime`, so there was no way to advertise the RFC 4566
  §6 *typical* per-packet aggregation. The new builder takes both
  aggregations in frames, converts to ms via `frame_duration_ms`, and
  emits `mode=N;ptime=P;maxptime=M` in canonical SDP order. A `ptime`
  of one whole frame is kept (unlike the `maxptime` cap, which a
  single frame collapses away), and when both are present `ptime` is
  clamped to never exceed `maxptime` (RFC 4566 §6 invariant). The
  emitted value round-trips back through `parse_ptime_from_fmtp` /
  `parse_maxptime_from_fmtp` / `parse_mode_from_fmtp` to the same
  `(mode, ptime, maxptime)` triple.

The depacketiser → decoder handoff is exercised end-to-end by
`tests/rtp_depacketiser_drives_decoder.rs`: encoder emits 2–5
frames per scenario, packetiser aggregates them under a per-packet
cap, depacketiser splits the body back into individual iLBC
payloads, and the decoder produces the correct `n * 160` or
`n * 240`-sample PCM stream. The round-240 gap-fill helpers are
covered by three additional integration tests that drive the
pre-roll → concealment → live-payload transition through the
decoder on both modes, with one chain through the 16-bit-wrap
sequence-arithmetic path.

## Codec id

- `"ilbc"` — registered as a software decoder via
  `oxideav_ilbc::register`.

## License

MIT — see [LICENSE](LICENSE).
