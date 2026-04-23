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
  rescaled gain dequantisation, 10th-order LPC synthesis, dampened
  pitch-synchronous PLC for lost / empty-indicated frames.
- The RFC's enhancer (§4.6) is **not** mandatory for interoperability;
  this crate ships a simplified pitch-emphasis variant good enough for
  voiced-speech clarity without risking over-enhancement.
- Encoder: not implemented. Decoder-only crate.

### Deviations from RFC 3951

Flagged explicitly in each module where they apply:

- Large Appendix A tables (split-VQ LSF codebooks, augmented codebook
  gain quantisers, start-state tables) are imported as condensed
  subsets sufficient to produce a monotone-LSF / bounded-output
  decoder on all index values. See `lsf_tables.rs` and
  `cb_tables.rs` module docs for the exact coverage.
- The enhancer is a simplified pitch-emphasis filter rather than the
  RFC §4.6 six-PSSQ combiner.

Net effect: structurally correct decoder that produces bounded mono
8 kHz PCM on any well-formed 38-/50-byte iLBC payload and on empty /
lost frames, but output is not guaranteed to be bit-exact against the
RFC 3951 reference decoder.

## Codec id

- `"ilbc"` — registered as a software decoder via
  `oxideav_ilbc::register`.

## License

MIT — see [LICENSE](LICENSE).
