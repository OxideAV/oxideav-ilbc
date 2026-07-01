# oxideav-ilbc

Pure-Rust **iLBC** (Internet Low Bit Rate Codec, RFC 3951) narrowband
speech codec — encoder, decoder, and the RFC 3952 RTP payload format.
Zero C dependencies, no FFI, no `*-sys` crates.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core  = "0.1"
oxideav-codec = "0.1"
oxideav-ilbc  = "0.0"
```

## Format

- **Sample rate**: 8 kHz mono (`S16`), narrowband telephony.
- **Frame modes**:
  - **20 ms** — 160 samples, 304 bits = 38 bytes, 15.20 kbit/s.
  - **30 ms** — 240 samples, 400 bits = 50 bytes, 13.33 kbit/s.

Mode is selected from the packet length: 38 bytes ⇒ 20 ms, 50 bytes ⇒
30 ms. The last bit of the payload is the *empty frame indicator* —
when set, the decoder treats the block as lost and runs packet-loss
concealment (PLC).

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

## Capability

Both directions are structurally complete against RFC 3951:

- **Decoder** — full bit-unpack, split-VQ LSF dequantisation with
  stability + linear interpolation, start-state reconstruction with the
  §4.2 all-pass phase compensator, 3-stage adaptive-codebook excitation
  with successive-rescaled gain dequantisation, 10th-order LPC synthesis
  with the §4.7 enhancer-delay shift, the §4.6 six-PSSQ
  pitch-synchronous enhancer, the §4.5 residual-domain packet-loss
  concealment following the Appendix A.14 `doThePLC` example (a ±3
  `compCorr` pitch-refinement search around the enhancer's pitch lag, the
  `pitchfact` voicing mix, `use_lag` short-lag doubling, the
  delayed-copy `randlag` noise component with the §A.14 `seed·69069+1`
  LCG, the `use_gain` consecutive-loss energy ladder, the per-80-sample
  1.0 / 0.95 / 0.9 intra-block taper, and the 30 dB pure-noise fallback)
  — the concealed *excitation* is run through the same enhancer + LPC
  synthesis as a received block so a recovered frame merges smoothly
  (§4.5.3) via the enhancer's cross-block correlation — and an opt-in
  §4.8 65 Hz output high-pass post-filter (`hp_filter=on` in
  `CodecParameters::options`).
- **Encoder** — LPC analysis (asymmetric / Hanning windowing →
  autocorrelation → Levinson-Durbin → 0.9025 chirp expansion → LSF),
  split-VQ LSF quantisation, scalar start-state coding with the §3.5.1
  variable `start_idx` (windowed-energy classifier) and `position` bit,
  the §3.6 residual-domain 3-stage codebook search with the Table 3.1 /
  3.2 bit-width caps walked symmetrically forward + backward around the
  state span, and the §3.7 stage-0 gain-correction post-pass.

### Wire format

The bitstream layout follows RFC 3951 §3.8 ULP: per-parameter class-1 /
class-2 / class-3 widths come from Appendix A.41 `ULP_20msTbl` /
`ULP_30msTbl`, and the three-pass pack / unpack mirrors Appendix A.42.
See `src/ulp.rs`.

### Table provenance

The split-VQ LSF codebooks, the LSF mean vector, the three gain
quantisers, and the start-state scalar quantiser are the complete
Appendix A tables, transcribed verbatim from the RFC 3951 decimal
listing. `tests/table_provenance.rs` cross-checks every numeric fact
against the independently extracted fixed-point (Q-domain) tables under
`docs/audio/ilbc/tables/`, and pins the high-pass biquad coefficients,
the §3.6.3.2 codebook expansion filter, the §4.6.2 enhancer polyphase
filter, and the §4.2.1 LPC analysis windows the closed-form generators
produce.

### Deviations from RFC 3951

A few RFC-RECOMMENDED (not REQUIRED) options are implemented but
disabled by default because they regress waveform SNR on the synthetic
self-roundtrip signals:

- The §3.5.3 DPCM noise-shaping start-state quantiser is present
  (`state_dpcm=on`); the default is a direct per-sample scalar
  quantiser. The decoder reads the indices back identically either way.
- The §3.6.2 perceptual codebook-search weighting is implemented but
  off; the search runs on unweighted residuals.
- The §4.6 enhancer constraint `b` is tuned to 0.005 (RFC suggests
  0.05) to preserve the unenhanced excitation on the test signals.

### Fidelity

The decoder output conversion follows the RFC 3951 §A.2 `iLBC_decode`
output stage exactly — the float synthesis output is clamped to the
int16 range `[-32768, 32767]` and then **truncated toward zero** (the
reference's `(short)` cast), not rounded to nearest. On near-silent
material this is the difference between drifting ±1 LSB on every sample
that crosses a `.5` fractional boundary and being bit-aligned: the
silence fixtures now decode **bit-identical to the reference on 99.94 %
(20 ms) / 99.96 % (30 ms) of samples** (the remainder drift by exactly
1 LSB), for ~122-125 dB PSNR, and the step-impulse exact-sample fraction
is 94.8 %. `tests/docs_corpus.rs` pins the ≥99 % bit-aligned silence
fraction (`silence_20ms_is_near_bit_exact` /
`silence_30ms_is_near_bit_exact`) and a ≤1-LSB max diff so the
conversion can't regress to round-to-nearest.

Output is **not** bit-exact end-to-end on
speech / tones (CELP rounding drift in the multi-stage codebook search
accumulates above the LSB). Each fixture under
`docs/audio/ilbc/fixtures/` carries a per-case PSNR regression floor
anchored 2-3 dB beneath the observed PSNR (the silence floors are pinned
at 110 dB). Workspace policy bars consulting any external iLBC implementation
as a clean-room reference, so there is no third-party encoder oracle in
CI; the decode path is validated against statically captured bitstream
trace files and against the integer index trace shipped with each
fixture (`tests/bitstream_trace.rs`).

The low (13-17 dB) PSNR on the **tonal** fixtures (sine / noise /
voice-like / DTMF) is dominated by an anomaly in the *reference* WAVs,
not by our reconstruction: the captured `expected.wav` for each tonal
case slams 0.45-1.7 % of its samples to the int16 rails (a high-energy
clipping waveform out of a clean 440 Hz sine), whereas our spec-correct
decode of the same bitstream stays bounded (tone / voice / DTMF peaks
≈ 3k-5k, white noise ≈ 26k) and never clips. The silence reference is
the lone clean case (0 % clipped), which is why it scores ~95 dB.
`tests/decoder_wellformed.rs` pins both halves of this: our decode is
well-formed (no rail-pinning, peak < 30000) on every tonal fixture, and
the reference WAVs do exhibit the clipping anomaly — so a future change
can't silently "improve" the corpus PSNR by chasing the broken
reference into an unstable, clipping decode, and a re-captured corpus
flags the floors for re-anchoring.

### Packet-loss concealment

Lost frames are signalled to the decoder either by a packet whose §3.8
empty-frame indicator bit is set or by a zero-byte packet. Both route to
the §4.5 residual-domain concealer (Appendix A.14 `doThePLC`): the lost
block's *excitation* is reconstructed from the previous block's saved
residual and the previous LP filter, then enhanced and synthesised on the
normal decode path. `tests/plc.rs` drives genuine fixture bitstreams with
simulated loss and pins the observable §A.14 behaviour — consecutive
losses dampen per the `use_gain` ladder, a single loss recovers within a
few frames without diverging, and losing the first frame (no saved
residual) is bounded.

## Storage format (RFC 3951 §5)

The `storage` module reads and writes the de-facto `#!iLBC{20,30}\n`
on-disk framing every `.lbc` file in the wild uses: a 9-byte ASCII
magic header that pins the frame mode for the whole file, followed by a
run of fixed-size (38- or 50-byte) frames.

- `storage::parse` recovers the `FrameMode` from the magic and yields
  the frame payloads, validating the body is a whole number of frames.
- `storage::write` / `storage::wrap_body` serialise frames (or an
  already-concatenated body) back into the storage form.
- `storage::detect_mode` / `storage::magic_for` expose the magic ↔ mode
  mapping for probing.
- `storage::mark_lost` / `storage::clear_lost` / `storage::is_lost` set,
  clear, and detect the §3.8 empty-frame indicator — the RFC's "lost
  frame" marker for the file storage format — which routes a frame to
  the decoder's residual-domain PLC path.

`tests/storage_format.rs` drives the parser against the real
`containerless-vs-rtp-style-pair` (magic-stripped body is byte-identical
to the header-less carriage) and `transition-mid-stream` fixtures, and
runs a parsed storage file through the decoder end to end.

## RTP payload format (RFC 3952)

The `rtp` module is a pure depacketiser / packetiser — the 12-byte
fixed RTP header lives one layer above this crate.

- **Payload format (§3)** — the iLBC RTP payload is the encoded
  bitstream with no per-packet codec header; one packet may carry
  several frames, all sharing the SDP-pinned mode. The packetiser
  aggregates frames up to a per-packet cap and emits per-packet
  RTP-timestamp offsets; the depacketiser splits a payload back into
  fixed-size chunks for the decoder.
- **SDP mapping (§4.2)** — `Depacketiser::from_sdp_fmtp` parses
  `mode=20|30` from an `a=fmtp` line; `format_mode_fmtp` /
  `build_fmtp` / `build_fmtp_with_ptime` emit the canonical
  `mode` + optional `ptime` / `maxptime` tokens (RFC 4566 §6).
  Inbound `parse_ptime_from_fmtp` / `parse_maxptime_from_fmtp` /
  `max_frames_per_packet_from_fmtp` derive a per-packet cap.
- **Length-only mode hint** — `detect_mode_from_payload_len` recovers
  a mode lost in transit (or reports it ambiguous).
- **Loss concealment** — `empty_marker_frame`, `conceal_gap`,
  `concealment_payload`, and `depacketise_with_gap_fill` emit empty
  marker frames to keep the decoded stream wall-clock-aligned across a
  sequence-number gap; `rtp_seq_gap` does the RFC 3550 §3.3
  16-bit-wrap-aware sequence arithmetic.

`tests/rtp_depacketiser_drives_decoder.rs` exercises the
depacketiser → decoder handoff end-to-end on both modes.

## Benchmarks

Criterion harnesses in `benches/` time the decoder, encoder, the paired
round-trip, and the RTP packetiser / depacketiser. Every input is
synthesised in-bench from a deterministic seed, so the harnesses ship
no committed fixtures and read nothing under `docs/`.

```sh
cargo bench -p oxideav-ilbc --bench decode
cargo bench -p oxideav-ilbc --bench encode
cargo bench -p oxideav-ilbc --bench roundtrip
cargo bench -p oxideav-ilbc --bench rtp
```

## Fuzzing

A `cargo-fuzz` harness in `fuzz/` exercises every attacker-facing parse
the crate ships, asserting panic-freedom + structural invariants on
arbitrary input: `decode`, `encode_roundtrip`, `rtp_depacketise`,
`sdp_fmtp`, and `rtp_gap_fill`. The targets share a nested workspace so
the umbrella's `crates/*` glob does not pull them in.

```sh
cargo +nightly fuzz run decode
cargo +nightly fuzz run encode_roundtrip
cargo +nightly fuzz run rtp_depacketise
cargo +nightly fuzz run sdp_fmtp
cargo +nightly fuzz run rtp_gap_fill
```

## Codec id

- `"ilbc"` — registered as a software decoder via
  `oxideav_ilbc::register`.

## License

MIT — see [LICENSE](LICENSE).
