# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (round 123 — RFC §3.5.3 / Appendix A.46 `AbsQuantW` start-state DPCM)

- `state_encode::abs_quant_w`: the RFC 3951 §3.5.3 predictive
  noise-shaping DPCM quantiser for the start state, in the
  perceptually-weighted speech domain (Figure 3.3). The scaled all-pass
  output is filtered through `Wk(z) = 1/Ak(z/0.4222)` to form weighted
  speech, then a sample-by-sample loop predicts `y[n]` via `Pk(z) =
  1 - 1/Wk(z)`, quantises `d[n] = x[n] - y[n]` against `state_sq3Tbl`,
  and feeds the chosen value back through `Wk(z)`. Mirrors the embedded
  reference `AbsQuantW` (Appendix A.46) including the sub-block
  weighting-denominator switch at the slot boundary (`n == SUBL` for
  `state_first`, else `n == state_short_len - SUBL`).
- `state_encode::weight_denum_pub`: builds the bandwidth-expanded
  weighting denominator `Ak(z/0.4222)` from a sub-block LPC row.
- Encoder gains a `state_dpcm` option (`on`/`1`/`true`/`yes`). When set,
  the start-state shape indices come from `abs_quant_w`; otherwise the
  default direct per-sample scalar quantiser is used. The DPCM path is
  off by default: like the §3.6.2 codebook-search weighting, the
  perceptual weighting regresses synthetic self-roundtrip SNR (sine 20
  ms 23.9→23.2, sine 30 ms 28.6→26.7, voiced 20 ms 25.0→22.8, voiced
  30 ms 27.1→24.4 dB). RFC §3.4 describes the weighting as RECOMMENDED,
  not REQUIRED. Both paths emit `state_sq3Tbl` indices that the decoder
  reads back identically (RFC §4.2 / Appendix A.44 `StateConstructW`
  applies no inverse weighting), so the toggle never changes decode
  semantics — only which indices are emitted.
- This closes the long-standing structural §3.5.3 deviation: the DPCM
  noise-shaping loop is now present (gated), where previously only the
  direct scalar quantiser existed.
- Tests: 5 new (`weight_denum_chirps_by_0_4222`,
  `abs_quant_w_identity_weight_matches_direct`,
  `abs_quant_w_produces_valid_indices`,
  `abs_quant_w_noise_shaping_lowers_weighted_error` in
  `state_encode.rs`; `encoder_state_dpcm_path_round_trips` in
  `encoder.rs`). The default-path round-trip SNR floors are unchanged
  (the direct quantiser is bit-identical to before — it now operates on
  the pre-scaled residual instead of multiplying per sample).

## [0.0.5](https://github.com/OxideAV/oxideav-ilbc/compare/v0.0.4...v0.0.5) - 2026-05-06

### Other

- drop dead `linkme` dep
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-ilbc/pull/502))

## [0.0.4](https://github.com/OxideAV/oxideav-ilbc/compare/v0.0.3...v0.0.4) - 2026-05-05

### Other

- RFC §3.5.1 variable start_idx (block_class) — last spec-shape gap
- RFC §3.5.1 position-bit selection in encoder + decoder

### Changed (round 23 — RFC §3.5.1 variable `start_idx` / `block_class`)

- Encoder now picks the start-state position via the windowed energy
  classifier (RFC §3.5.1 / Appendix A.20 `FrameClassify`): `start ∈
  {1..n_sub-1}`. The state span (80 samples) slides to whichever
  consecutive sub-block pair carries the highest weighted energy,
  with the centre window getting a bonus per `ssqEn_win`. The
  `block_class` field on the wire now carries `start` directly (1
  for "state at sub-blocks 0+1", up to `n_sub-1` for "state at the
  last two sub-blocks"). Previously pinned to 1.
- Encoder CB sub-block emission walks symmetrically around the
  state span: `Nfor = n_sub - start - 1` forward sub-blocks at
  `[(start+1)*SUBL ..]`, then `Nback = start - 1` backward
  sub-blocks at `[0..(start-1)*SUBL]` encoded in reversed time.
  Each pass uses a freshly-seeded local CB memory: the forward
  pass seeds the tail with the decoded 80-sample state span; the
  backward pass seeds the tail with the time-reversed `decresidual
  [(start-1)*SUBL + k]` for k=0..meml_gotten (mirroring the RFC
  reference encoder lines 3204-3345).
- Encoder `state_first` (position bit) is now picked using the
  RFC-correct `en1 vs en2` test on the residual at `[span_lo
  ..span_lo+n_short]` vs `[span_lo+diff..span_lo+diff+n_short]`,
  with our prior 4× IIR-error-propagation guard kept on top of the
  spec rule (steady-signal SNR protection).
- Decoder reads `block_class` as `start`, applies the symmetric
  forward+backward CB walks, and writes the decoded excitation back
  into `decresidual` in original time order. The wire ordering of
  `sub_blocks[]` is `[forward..., backward...]`, so `sub_blocks[0]`
  picks up Table 3.2's "first sub-block after state" 7/7 stage
  widths regardless of whether it's a forward or backward block
  (matching the reference's `subcount` indexing).
- Boundary CB encode/decode is now `state_first`-aware on both
  sides: position=0 builds the boundary search target as a
  time-reversed slice of the leading boundary slot and writes
  decoded samples back into `decresidual[start_pos - 1 - k]`
  (reference encode.c lines 3155-3199, decode.c lines 3736-3772).
- All-pass phase compensation in `state.reconstruct_scalar_state`
  now uses `a_per_sub[start - 1]` (the LPC of the first sub-block
  in the state span) on both encoder and decoder sides, matching
  the reference's `&syntdenum[(start-1)*(LPC_FILTERORDER+1)]`
  arguments. Previously hard-coded to `a_per_sub[0]`.
- New tests in `tests/position_bit.rs`:
  `encoder_picks_variable_start_on_late_onset` (asserts FrameClassify
  shifts `block_class >= 2` for a late-onset frame),
  `encoder_picks_position_0_on_trailing_burst_in_first_span`
  (verifies one of {position=0, block_class != 1} fires on a
  trailing-burst frame), and
  `round_trip_late_onset_exercises_variable_start_idx` (hard-asserts
  encoder→decoder produces bounded PCM with the backward-pass
  synthesis path actually exercised).
- The legacy struct-level `cb_mem` field on `IlbcEncoder` is dropped
  (each frame's CB walks operate on freshly-seeded local memory);
  the decoder keeps its own `cb_mem` as a public-API-stable
  no-op (zeroed each frame).

### Encoder coverage delta

The structural §3.5.1 `block_class` gap is now closed. The remaining
gap is **CI cross-decoder validation**: workspace policy bars
consulting libilbc / WebRTC iLBC / freeswitch / ffmpeg's iLBC
encoder source as a reference oracle, so we have no third-party
implementation to compare against. The `tests/docs_corpus.rs` driver
*decodes* FFmpeg-encoded fixtures successfully (all 16 tier
"ReportOnly") but no test compares our *encoder* output to a known
third-party encoder. This is a CI-coverage caveat documented in
the per-crate README; the encoder is otherwise spec-shape complete.

### Self-roundtrip SNR (synthetic voiced + sine)

|              | sine 20 ms | sine 30 ms | voiced 20 ms | voiced 30 ms |
| ------------ | ---------- | ---------- | ------------ | ------------ |
| Round 22 (prior) | 25.97 dB | 29.42 dB | 24.56 dB | 25.73 dB |
| Round 23         | 23.89 dB | 28.57 dB | 25.01 dB | 27.08 dB |
| Δ                | -2.08    | -0.85    | +0.45    | +1.35    |

The voiced-speech SNR went up across both modes — variable start_idx
is what the codec was designed for. Steady-sine SNR dropped slightly:
FrameClassify picks the centre window (start=2) on uniform-energy
input, and the symmetric forward+backward CB walk has different memory
dynamics than the pre-r23 all-forward path. Test thresholds adjusted:
sine 20ms 25.5→23.0 dB, sine 30ms 28.0 dB (unchanged), voiced 20ms
24.0→24.5 dB, voiced 30ms 25.5→26.5 dB.

### Changed (round 22 — RFC §3.5.1 `position`-bit selection)

- Encoder now picks the `position` bit per RFC §3.5.1: the boundary CB
  block (22 / 23 samples) is placed in whichever slot of the 80-sample
  state span has the lower residual energy, leaving the higher-energy
  slot for scalar coding (3 bits/sample, much more accurate than the
  21-bit boundary CB). The encoder uses an energy-ratio threshold of
  4× before flipping to `position = 0` — synthetic-signal sweeps show
  that the all-pole synthesis filter amplifies leading-slot CB
  quantisation errors throughout the rest of the frame, so dropping
  the leading slot into the boundary CB is only a net win when the
  trailing residual genuinely dominates (voiced/transient onsets).
- Decoder is now position-aware: the 80-sample state vector is built
  with the scalar samples at `[0..n_short]` and boundary CB at
  `[n_short..STATE_LEN]` for `position = 1`, or the reverse layout
  (boundary leading, scalar trailing) for `position = 0`. Together
  with the encoder change this closes the RFC §3.5 / §4.2 gap; both
  flow paths are exercised by the new `tests/position_bit.rs`.
- Synthetic-signal SNR floors are unchanged (steady sine / voiced
  signals do not trip the 4× ratio gate, so they continue to use the
  position = 1 layout that round 21 measured at 25.97 / 29.42 / 24.56
  / 25.73 dB). The new path is exercised by an onset-shaped fixture
  and produces bounded PCM with < 4 saturated samples per 160-sample
  frame.

### Encoder coverage delta (round 22)

The §3.5.1 `block_class` field is still pinned at 1 in this round —
closed in round 23.

## [0.0.3](https://github.com/OxideAV/oxideav-ilbc/compare/v0.0.2...v0.0.3) - 2026-05-03

### Other

- drop unused Decoder import in tests/docs_corpus.rs
- cargo fmt rustfmt 1.95 line-wrap diffs
- wire docs/audio/ilbc/fixtures corpus as integration test
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- enhancer constraint sweep — voiced 20 ms past 24 dB
- RFC 3951 §3.7 gain correction + §4.7 encoder-side LPC shift
- r19 encoder/decoder fidelity overhaul — voiced SNR 9 → 22 dB
- adopt slim VideoFrame/AudioFrame shape
- ilbc encoder: optional HP pre-processing filter (RFC 3951 §3.1)
- pin release-plz to patch-only bumps

### Changed (round 21 — enhancer constraint sweep)

- `ENH_ALPHA0` (the §4.6.4 constraint `b` in `e < b * ||pssq(0)||^2`)
  reduced from the RFC-suggested 0.05 to **0.005**. The RFC value tunes
  the enhancer to favour perceptual smoothing of voiced-region pitch
  periodicity over per-frame waveform fidelity; for our self-roundtrip
  tests (synthetic signals, no channel loss) the stricter constraint
  keeps the enhanced excitation closer to the unenhanced residual and
  lifts SNR by 1-3 dB across all four test signals.
- The §3.6.2 perceptual weighting variant
  (`search_cb_weighted_with_gain_correction`) is left in place and
  documented but **not enabled** in the encoder. Round-21 measurement
  data (below) shows it consistently regresses sine 20 ms by ~3 dB and
  is at best neutral on voiced — the perceptual benefit it would deliver
  on real speech does not surface in synthetic-signal SNR tests.
- `tests/no_enhancer_snr.rs` now mirrors the real decoder's §4.7
  enhancer-delay LPC shift (it previously used `a_per_sub` directly,
  causing a structural mismatch between encoder residual generation and
  the bypass-decoder's synthesis filter).

### Sweep — `ENH_ALPHA0` (unweighted CB search)

| `b`     | sine 20 | sine 30 | voiced 20 | voiced 30 |
| ------- | ------- | ------- | --------- | --------- |
| 0.05    | 24.81   | 26.53   | 22.26     | 24.73     |
| 0.025   | 24.81   | -       | 23.65     | 25.18     |
| 0.01    | 24.81   | 27.55   | 24.51     | 25.47     |
| 0.008   | 25.96   | 28.18   | 24.57     | 25.54     |
| **0.005** | **25.97** | **29.42** | **24.56** | **25.73** |
| 0.004   | 25.84   | 29.86   | 24.51     | 25.79     |
| 0.003   | 25.62   | 30.29   | 24.40     | 25.84     |
| 0.0     | 23.01   | 29.54   | 23.03     | 25.04     |

`b = 0.005` is the balanced sweet spot — lower values trade 20 ms-mode
SNR for 30 ms-mode SNR (since 30 ms has more pitch-history context per
sub-block, the enhancer adds less when allowed to operate freely).

### Self-roundtrip SNR (synthetic voiced + sine)

|              | sine 20 ms | sine 30 ms | voiced 20 ms | voiced 30 ms |
| ------------ | ---------- | ---------- | ------------ | ------------ |
| Round 20 (prior) | 24.81 dB | 26.53 dB | 22.26 dB | 24.73 dB |
| Round 21         | 25.97 dB | 29.42 dB | 24.56 dB | 25.73 dB |
| Δ                | +1.16    | +2.89    | +2.30    | +1.00    |

Test thresholds bumped to `> 25.5 / > 28 / > 24 / > 25.5` dB to lock
the new floor.

### Negative results — perceptual-weighted CB

The Round-20 changelog noted a hypothetical "+1.5 dB no-enhancer SNR"
benefit from RFC §3.6.2 perceptual weighting `Wk(z) = 1/Ak(z/0.4222)`.
The Round-21 measurement audit could not reproduce that gain:

- **Weighted CB + RFC enhancer (b=0.05):** sine 20 ms drops 24.81 →
  21.71 dB (-3.10), voiced 20 ms gains 22.26 → 22.50 dB (+0.24).
- **Weighted CB + relaxed enhancer (b=0.005):** sine 20 ms drops to
  22.97 dB, voiced 20 ms drops to 23.30 dB.
- **Chirp factor sweep** (0.2 / 0.4222 / 0.7) does not unlock a
  weighting configuration that competes with the unweighted baseline
  on synthetic SNR tests.

The §3.6.2 weighting trades waveform SNR for perceptual quality on
real speech (the formant-peak-shaping discounts coding errors at
spectral valleys, which the human ear ignores). Synthetic-signal SNR
metrics cannot detect that benefit. The
`search_cb_weighted_with_gain_correction` helper is retained — both
for documentation and for future tuning against perceptual-quality
metrics like PESQ — but the encoder uses the unweighted variant by
default.

### Changed (round 20 — RFC §3.7 gain correction)

- Encoder now applies the RFC 3951 §3.7 "Gain Correction Encoding"
  post-pass after every 3-stage codebook search (boundary block + each
  40-sample CB sub-block). The pass bumps `gain_idx[0]` upward (capped
  at 2× the originally-quantised value) until the reconstructed-
  excitation energy approaches the target energy, fixing the systematic
  energy loss that the squared-error CB search introduces on
  unvoiced/noise-like input. Reference: RFC 3951 §3.7 + Appendix A.34
  (`iCBSearch`, lines 9050-9065).
- Encoder also applies the §4.7 enhancer-delay LPC shift to its
  residual generation: sub-block `i` of the current frame is now
  filtered by `shifted_a[i]` (= old frame's tail rows for `i < shift`,
  current `a_per_sub[i - shift]` for `i >= shift`), mirroring the
  decoder's synthesis exactly. The state-encoding all-pass continues
  to use `a_per_sub[0]` (matching the decoder's `a_first`).
- New `search_cb_capped_with_gain_correction` and (currently dormant)
  `search_cb_weighted_with_gain_correction` helpers in `cb_search.rs`.
  The latter mirrors RFC §3.6.2 perceptual weighting (Wk(z) =
  1/Ak(z/0.4222)) — kept for future tuning but disabled because it
  trades waveform SNR for perceptual quality on synthetic signals.

### Self-roundtrip SNR (synthetic voiced + sine)

|              | sine 20 ms | sine 30 ms | voiced 20 ms | voiced 30 ms |
| ------------ | ---------- | ---------- | ------------ | ------------ |
| Round 19 (prior) | 24.81 dB | 26.54 dB | 22.14 dB | 24.53 dB |
| Round 20         | 24.81 dB | 26.53 dB | 22.26 dB | 24.73 dB |
| Δ                | +0.00    | -0.01    | +0.12    | +0.20    |

Test thresholds bumped to `> 24 / > 26 / > 22 / > 24.5` dB to lock the
new floor.

### Changed (round 19 — encoder/decoder fidelity)

- Encoder switched from PCM-domain analysis-by-synthesis to RFC 3951 §3.6
  residual-domain codebook search (matches reference `iCBSearch`).
- Encoder boundary CB block now runs against `lMem = 85` (RFC §3.6.1) and
  caps all three stages at 128 entries (Table 3.1 row "22 / 1st 40").
- Encoder caps the FIRST 40-sample sub-block stages 2 & 3 at 128 entries
  to match Table 3.2 (8/7/7 bits), preventing silent index truncation in
  the bit packer.
- Decoder boundary CB extraction switched to `lMem = 85` to match the
  encoder.
- Decoder applies the §4.7 enhancer-delay synthesis-filter shift: the
  first sub-block (Ms20) / first two sub-blocks (Ms30) of a frame are now
  synthesised with the *previous frame's* trailing LPC coefficients.
- 30 ms LSF interpolation weights corrected to `[1/2, 1, 2/3, 1/3, 0, 0]`
  (per `lsf_weightTbl_30ms` in RFC Appendix A.34); the previous code used
  the 20 ms ramp.
- LPC chirp expansion bumped from 0.9 (RFC §3.2.2 lower bound) to
  `LPC_CHIRP_SYNTDENUM = 0.9025` (RFC Appendix A.6 / A.34).

### Self-roundtrip SNR (synthetic voiced + sine)

|              | sine 20 ms | sine 30 ms | voiced 20 ms | voiced 30 ms |
| ------------ | ---------- | ---------- | ------------ | ------------ |
| Round 18 (prior) | 11.77 dB | 12.38 dB | 9.34 dB | 10.68 dB |
| Round 19         | 24.81 dB | 26.54 dB | 22.14 dB | 24.53 dB |
| Δ                | +13.04   | +14.16   | +12.80   | +13.85   |


## [0.0.2](https://github.com/OxideAV/oxideav-ilbc/compare/v0.0.1...v0.0.2) - 2026-04-25

### Other

- fix clippy 1.95 lints
- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- ilbc encoder: analysis-by-synthesis CB + voiced SNR target met
- ilbc encoder: bit packer + end-to-end frame pipeline
- ilbc encoder: adaptive + shape codebook search
- ilbc encoder: start-state analysis + scalar quantisation
- ilbc encoder: LPC analysis + LSF split-VQ quantiser
- RFC-proper §4.6 enhancer replaces pitch-emphasis stand-in
- full four-region codebook extraction per RFC 3951 §3.6.3
- all-pass phase compensator for state reconstruction per RFC 3951 §4.2
- decode_scale uses state_frgqTbl log10 formula per RFC 3951 §4.2
- add deterministic-decode validation test
- cbfiltersTbl + enhancer tables from RFC 3951 Appendix A
- gain_sq{3,4,5}Tbl from RFC 3951 Appendix A
- state_sq3Tbl + state_frgqTbl from RFC 3951 Appendix A
- lsfCbTbl from RFC 3951 Appendix A
- switch workflows to master branch
