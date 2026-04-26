# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3](https://github.com/OxideAV/oxideav-ilbc/compare/v0.0.2...v0.0.3) - 2026-04-26

### Other

- adopt slim VideoFrame/AudioFrame shape
- ilbc encoder: optional HP pre-processing filter (RFC 3951 §3.1)
- pin release-plz to patch-only bumps

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
