//! Numeric-table provenance cross-check against the independently
//! extracted normative tables under `docs/audio/ilbc/tables/`.
//!
//! The crate ships its split-VQ LSF codebook, LSF mean vector, the three
//! gain quantiser codebooks, and the start-state scalar quantiser as
//! `f32` constants transcribed from RFC 3951 Appendix A. The docs tree
//! carries the *same* normative tables in their integer fixed-point
//! (Q-domain) form, extracted by a pure data-only extractor that reads
//! only the constants file (no algorithmic source). The two derivations
//! are independent: one is the RFC's decimal listing, the other is the
//! fixed-point constant table. They must agree exactly once the
//! fixed-point integer is mapped back to the rational the RFC prints
//! (`int / 2^q`, rounded to nearest).
//!
//! This driver loads each docs CSV, scales every crate `f32` into the
//! matching Q-domain, rounds to nearest, and asserts equality element by
//! element. A single transcription typo in any codebook entry — the kind
//! the PSNR-floor gates in `docs_corpus.rs` would only catch indirectly
//! and only for the indices a given fixture happens to exercise —
//! red-lights here immediately, on the *whole* table, with no dependence
//! on the CELP synthesis pipeline.
//!
//! Provenance: the docs CSVs are declarative numeric facts (per
//! *Feist v. Rural*) extracted under `docs/audio/ilbc/tables/`. The test
//! reads them as plain text; there is no algorithmic source involved on
//! either side.

use std::fs;
use std::path::PathBuf;

use oxideav_ilbc::cb::{GAIN_SQ3_TBL, GAIN_SQ4_TBL, GAIN_SQ5_TBL};
use oxideav_ilbc::lsf_tables::{LSF_CB_TBL_1, LSF_CB_TBL_2, LSF_CB_TBL_3, LSF_MEAN};
use oxideav_ilbc::state::STATE_SQ3_TBL;

fn tables_dir() -> PathBuf {
    PathBuf::from("../../docs/audio/ilbc/tables")
}

/// Read a whitespace/newline-separated list of `i64` from a docs CSV.
/// Returns `None` (and the caller skips) when the docs submodule is not
/// checked out, matching the `read_or_skip` convention the other
/// docs-backed integration drivers use.
fn read_int_csv(name: &str) -> Option<Vec<i64>> {
    let path = tables_dir().join(name);
    match fs::read_to_string(&path) {
        Ok(text) => Some(
            text.split_whitespace()
                .map(|tok| {
                    tok.parse::<i64>()
                        .unwrap_or_else(|e| panic!("{}: non-integer token {tok:?}: {e}", name))
                })
                .collect(),
        ),
        Err(e) => {
            eprintln!("skip {name} ({}): {e}", path.display());
            None
        }
    }
}

/// Scale a crate `f32` into Q`q` and round to nearest (ties away from
/// zero, matching the symmetric rounding the RFC decimal listing
/// implies for the fixed-point constants).
fn to_q(value: f32, q: u32) -> i64 {
    let scaled = f64::from(value) * f64::from(1u32 << q);
    scaled.round() as i64
}

/// Assert every crate value, scaled into Q`q`, equals the corresponding
/// docs integer. `docs` may be longer than `crate_vals` (the gain CSVs
/// carry a trailing saturation sentinel the crate does not store); the
/// leading `crate_vals.len()` entries must match.
fn assert_q_match(label: &str, crate_vals: &[f32], docs: &[i64], q: u32) {
    assert!(
        docs.len() >= crate_vals.len(),
        "{label}: docs table ({}) shorter than crate table ({})",
        docs.len(),
        crate_vals.len()
    );
    for (i, (&c, &d)) in crate_vals.iter().zip(docs.iter()).enumerate() {
        let got = to_q(c, q);
        assert_eq!(
            got, d,
            "{label}: entry #{i} mismatch — crate {c} → Q{q} {got}, docs {d}"
        );
    }
}

#[test]
fn lsf_codebook_matches_docs_q13() {
    let Some(docs) = read_int_csv("lsf-quantizer-codebook.csv") else {
        return;
    };
    // The docs codebook is the three split-VQ stages packed end-to-end
    // (64×3 + 128×3 + 128×4 = 1088 entries), so flatten the crate tables
    // in the same order and compare against the single docs vector.
    let mut flat: Vec<f32> = Vec::with_capacity(1088);
    for row in LSF_CB_TBL_1.iter() {
        flat.extend_from_slice(row);
    }
    for row in LSF_CB_TBL_2.iter() {
        flat.extend_from_slice(row);
    }
    for row in LSF_CB_TBL_3.iter() {
        flat.extend_from_slice(row);
    }
    assert_eq!(
        flat.len(),
        1088,
        "crate LSF split-VQ flat length (expected 64*3 + 128*3 + 128*4)"
    );
    assert_eq!(docs.len(), 1088, "docs LSF codebook length");
    assert_q_match("lsf-quantizer-codebook", &flat, &docs, 13);
}

#[test]
fn lsf_mean_matches_docs_q13() {
    let Some(docs) = read_int_csv("lsf-mean-Q13.csv") else {
        return;
    };
    assert_eq!(docs.len(), LSF_MEAN.len(), "docs LSF-mean length");
    assert_q_match("lsf-mean", &LSF_MEAN, &docs, 13);
}

#[test]
fn gain_sq3_matches_docs_q14() {
    let Some(docs) = read_int_csv("gain-codebook-3bit-Q14.csv") else {
        return;
    };
    // The docs CSV carries a trailing 32767 saturation sentinel the
    // crate omits (the 3-bit index space is 0..=7).
    assert_eq!(GAIN_SQ3_TBL.len(), 8, "crate gain SQ3 length");
    assert_q_match("gain-codebook-3bit", &GAIN_SQ3_TBL, &docs, 14);
}

#[test]
fn gain_sq4_matches_docs_q14() {
    let Some(docs) = read_int_csv("gain-codebook-4bit-Q14.csv") else {
        return;
    };
    assert_eq!(GAIN_SQ4_TBL.len(), 16, "crate gain SQ4 length");
    assert_q_match("gain-codebook-4bit", &GAIN_SQ4_TBL, &docs, 14);
}

#[test]
fn gain_sq5_matches_docs_q14() {
    let Some(docs) = read_int_csv("gain-codebook-5bit-Q14.csv") else {
        return;
    };
    assert_eq!(GAIN_SQ5_TBL.len(), 32, "crate gain SQ5 length");
    assert_q_match("gain-codebook-5bit", &GAIN_SQ5_TBL, &docs, 14);
}

#[test]
fn state_sq3_matches_docs_q13() {
    let Some(docs) = read_int_csv("state-quantizer-3bit-Q15.csv") else {
        return;
    };
    // The docs `.meta` labels this table Q15 after the WebRTC storage
    // domain, but the RFC decimal listing the crate transcribes is the
    // integer divided by 2^13 (e.g. -30473 / 8192 = -3.719849). The
    // crate float therefore maps back at Q13, not Q15. The numeric
    // facts are identical; only the documented Q-domain label differs.
    assert_eq!(STATE_SQ3_TBL.len(), 8, "crate state SQ3 length");
    assert_q_match("state-quantizer-3bit", &STATE_SQ3_TBL, &docs, 13);
}
