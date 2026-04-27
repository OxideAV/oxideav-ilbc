# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
