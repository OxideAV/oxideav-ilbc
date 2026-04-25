//! Multistage codebook search for the iLBC encoder (RFC 3951 §3.6).
//!
//! For each target sub-block (length 40 for regular sub-blocks, 22/23
//! for the start-state boundary block) the encoder:
//!
//!   1. Builds the 4-region adaptive codebook (base / augmented / expanded
//!      / augmented expanded) from the current 147-sample codebook memory
//!      — reusing the decoder's [`crate::cb::extract_cbvec_veclen`].
//!   2. At stage 0, picks the codebook vector with the largest
//!      `(target·cbvec)^2 / ||cbvec||^2` subject to:
//!        * `target·cbvec > 0`
//!        * `|gain| < CB_MAXGAIN = 1.3`
//!      and quantises the positive gain against `GAIN_SQ5_TBL` (5 bits).
//!   3. At stages 1 and 2, subtracts the gain-scaled stage-k vector from
//!      the target, searches for the best remaining match, and quantises
//!      the gain (signed) against `GAIN_SQ4_TBL` (4 bits) / `GAIN_SQ3_TBL`
//!      (3 bits) respectively. The rescaling is `scale = max(0.1,
//!      |previous_dequantised_gain|)` as in `gaindequant` (RFC §3.6.4.2).
//!   4. Returns the three (idx, gain_idx) pairs, plus the reconstructed
//!      excitation sub-block — identical to what the decoder would
//!      build from those indices.
//!
//! Codebook sizes per RFC Table 3.1 (40-sample sub-blocks):
//!   - total = 256 (base 108 + aug-base 20 + expanded 108 + aug-exp 20)
//!   - stage 0 uses 8 bits
//!   - stages 1 and 2 use 7 bits (for the first 40-sample sub-block
//!     after the state) or 8 bits (for later sub-blocks).
//!
//! For the 22/23-sample boundary block:
//!   - total = 128, stages 0/1/2 all 7 bits.

use crate::cb::{extract_cbvec_veclen, update_cb_memory, GAIN_SQ3_TBL, GAIN_SQ4_TBL, GAIN_SQ5_TBL};
use crate::{CB_LMEM, LPC_ORDER, SUBL};

/// Maximum absolute gain (RFC 3951 §3.6.4.1).
const CB_MAXGAIN: f32 = 1.3;

/// Gain-floor used when rescaling successive-stage gain tables
/// (RFC §3.6.4.2).
const GAIN_SCALE_FLOOR: f32 = 0.1;

/// Result of one-sub-block codebook search.
#[derive(Clone, Copy, Debug)]
pub struct CbSearchResult {
    pub cb_idx: [u16; 3],
    pub gain_idx: [u8; 3],
}

/// Number of codebook entries for a target of length `cbveclen` given
/// `lMem`. Matches the layout in [`crate::cb::extract_cbvec_veclen`].
fn total_cb_size(lmem: usize, cbveclen: usize) -> usize {
    let base_size_no_aug = lmem - cbveclen + 1;
    let base_size = if cbveclen == SUBL {
        base_size_no_aug + cbveclen / 2
    } else {
        base_size_no_aug
    };
    2 * base_size
}

/// Stage-0 gain: positive, 5-bit, scale = 1.0.
fn quantise_gain0(gain: f32) -> (u8, f32) {
    // Clamp positive and limit to CB_MAXGAIN.
    let g = gain.clamp(0.0, CB_MAXGAIN);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &v) in GAIN_SQ5_TBL.iter().enumerate() {
        let e = (v - g).abs();
        if e < best_err {
            best_err = e;
            best = i as u8;
        }
    }
    (best, GAIN_SQ5_TBL[best as usize])
}

/// Stage-1 gain: signed, 4-bit, scale = max(0.1, |g_prev|).
fn quantise_gain1(gain: f32, prev_abs: f32) -> (u8, f32) {
    let scale = prev_abs.max(GAIN_SCALE_FLOOR);
    // Normalised gain to quantise against the raw table.
    let norm = gain / scale;
    let norm = norm.clamp(-CB_MAXGAIN, CB_MAXGAIN);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &v) in GAIN_SQ4_TBL.iter().enumerate() {
        let e = (v - norm).abs();
        if e < best_err {
            best_err = e;
            best = i as u8;
        }
    }
    (best, GAIN_SQ4_TBL[best as usize] * scale)
}

/// Stage-2 gain: signed, 3-bit, scale = max(0.1, |g_prev|).
fn quantise_gain2(gain: f32, prev_abs: f32) -> (u8, f32) {
    let scale = prev_abs.max(GAIN_SCALE_FLOOR);
    let norm = gain / scale;
    let norm = norm.clamp(-CB_MAXGAIN, CB_MAXGAIN);
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &v) in GAIN_SQ3_TBL.iter().enumerate() {
        let e = (v - norm).abs();
        if e < best_err {
            best_err = e;
            best = i as u8;
        }
    }
    (best, GAIN_SQ3_TBL[best as usize] * scale)
}

/// Find the codebook vector that maximises `(target·cbvec)^2 / ||cbvec||^2`
/// subject to the stage-0 constraint `target·cbvec > 0` and a gain
/// magnitude cap. Returns (best_idx, best_gain, best_vec).
fn search_stage(
    cb_mem: &[f32],
    cbveclen: usize,
    target: &[f32],
    stage0: bool,
) -> (u16, f32, Vec<f32>) {
    let total = total_cb_size(cb_mem.len(), cbveclen);
    let mut best_idx = 0u16;
    let mut best_measure = f32::NEG_INFINITY;
    let mut best_gain = 0.0f32;
    let mut best_vec = vec![0.0f32; cbveclen];

    for i in 0..total {
        let v = extract_cbvec_veclen(cb_mem, i as u16, cbveclen);
        let mut dot = 0.0f32;
        let mut nrm = 0.0f32;
        for n in 0..cbveclen {
            dot += target[n] * v[n];
            nrm += v[n] * v[n];
        }
        if nrm < 1e-12 {
            continue;
        }
        if stage0 && dot <= 0.0 {
            continue;
        }
        let gain = dot / nrm;
        if gain.abs() > CB_MAXGAIN {
            continue;
        }
        let measure = (dot * dot) / nrm;
        let signed_measure = if stage0 {
            // Stage 0 requires dot>0, so measure is positive by
            // construction.
            measure
        } else {
            // Stages 1/2: the measure is `dot^2/nrm`, always positive,
            // but we'd like to reward both positive and negative
            // contributions the same. Use the unsigned measure.
            measure
        };
        if signed_measure > best_measure {
            best_measure = signed_measure;
            best_idx = i as u16;
            best_gain = gain;
            best_vec = v;
        }
    }
    if best_measure.is_infinite() {
        // No valid candidate — fall back to the first vector with gain 0.
        let v = extract_cbvec_veclen(cb_mem, 0, cbveclen);
        return (0, 0.0, v);
    }
    (best_idx, best_gain, best_vec)
}

/// Multistage codebook search. Searches 3 stages, updating the target
/// after each one. Also reconstructs the excitation sub-block exactly
/// as the decoder would from the chosen indices.
pub fn search_cb(cb_mem: &[f32], cbveclen: usize, target: &[f32]) -> (CbSearchResult, Vec<f32>) {
    debug_assert_eq!(target.len(), cbveclen);

    let mut t = target.to_vec();
    let mut cb_idx = [0u16; 3];
    let mut gain_idx = [0u8; 3];
    let mut reconstructed = vec![0.0f32; cbveclen];
    let mut prev_abs = 1.0f32;

    for stage in 0..3 {
        let stage0 = stage == 0;
        let (idx, gain, vec) = search_stage(cb_mem, cbveclen, &t, stage0);
        cb_idx[stage] = idx;
        let (g_idx, g_deq) = match stage {
            0 => quantise_gain0(gain),
            1 => quantise_gain1(gain, prev_abs),
            _ => quantise_gain2(gain, prev_abs),
        };
        gain_idx[stage] = g_idx;
        prev_abs = g_deq.abs();
        // Subtract the quantised-gain contribution from the target,
        // and add it to the reconstruction.
        for n in 0..cbveclen {
            t[n] -= g_deq * vec[n];
            reconstructed[n] += g_deq * vec[n];
        }
    }

    (CbSearchResult { cb_idx, gain_idx }, reconstructed)
}

/// Search a 40-sample sub-block. Convenience wrapper for the main path.
pub fn search_cb_subl(
    cb_mem: &[f32; CB_LMEM],
    target: &[f32; SUBL],
) -> (CbSearchResult, [f32; SUBL]) {
    let (res, rec) = search_cb(cb_mem, SUBL, target);
    let mut out = [0.0f32; SUBL];
    out.copy_from_slice(&rec);
    (res, out)
}

/// Analysis-by-synthesis codebook search — searches for the excitation
/// that, after passing through `1/A(z)` with zero initial memory,
/// best matches the perceptually significant residual.
///
/// `target_pcm` is the weighted PCM residual already net of the zero-
/// input response of the synth filter. `a` is the LPC denominator.
///
/// Internally, we pre-compute the zero-state response of `1/A(z)` for
/// each candidate codebook vector — the synth filter is linear, so we
/// can decompose each reconstructed candidate as `gain * ZSR + 0`.
/// Classic analysis-by-synthesis: maximise
/// `(target·ZSR)^2 / ||ZSR||^2`.
///
/// Returns the 3-stage CB indices plus the reconstructed excitation
/// sub-block (which, when run through `1/A(z)` with the same memory,
/// reproduces the chosen PCM).
pub fn search_cb_abs(
    cb_mem: &[f32; CB_LMEM],
    a: &[f32; LPC_ORDER + 1],
    target_pcm: &[f32; SUBL],
) -> (CbSearchResult, [f32; SUBL]) {
    // Pre-compute ZSR for all 256 codebook vectors (worst case).
    // total_cb_size for 40-sample target with 147-sample memory = 256.
    let total = total_cb_size(CB_LMEM, SUBL);
    let mut zsrs: Vec<[f32; SUBL]> = Vec::with_capacity(total);
    for i in 0..total {
        let cbv = extract_cbvec_veclen(cb_mem, i as u16, SUBL);
        let mut arr = [0.0f32; SUBL];
        arr.copy_from_slice(&cbv);
        let zsr = zero_state_response(&arr, a);
        zsrs.push(zsr);
    }

    let mut t = *target_pcm;
    let mut cb_idx = [0u16; 3];
    let mut gain_idx = [0u8; 3];
    let mut excitation = [0.0f32; SUBL];
    let mut prev_abs = 1.0f32;

    for stage in 0..3 {
        let stage0 = stage == 0;
        let mut best_idx = 0u16;
        let mut best_measure = f32::NEG_INFINITY;
        let mut best_gain = 0.0f32;
        for i in 0..total {
            let zsr = &zsrs[i];
            let mut dot = 0.0f32;
            let mut nrm = 0.0f32;
            for n in 0..SUBL {
                dot += t[n] * zsr[n];
                nrm += zsr[n] * zsr[n];
            }
            if nrm < 1e-12 {
                continue;
            }
            if stage0 && dot <= 0.0 {
                continue;
            }
            let gain = dot / nrm;
            if gain.abs() > CB_MAXGAIN {
                continue;
            }
            let measure = (dot * dot) / nrm;
            if measure > best_measure {
                best_measure = measure;
                best_idx = i as u16;
                best_gain = gain;
            }
        }
        if best_measure.is_infinite() {
            // Nothing valid; bail with zeroed stage.
            cb_idx[stage] = 0;
            gain_idx[stage] = match stage {
                0 => 0,
                1 => GAIN_SQ4_TBL.iter().position(|&v| v == 0.0).unwrap_or(7) as u8,
                _ => GAIN_SQ3_TBL.iter().position(|&v| v == 0.0).unwrap_or(3) as u8,
            };
            continue;
        }
        cb_idx[stage] = best_idx;
        let (g_idx, g_deq) = match stage {
            0 => quantise_gain0(best_gain),
            1 => quantise_gain1(best_gain, prev_abs),
            _ => quantise_gain2(best_gain, prev_abs),
        };
        gain_idx[stage] = g_idx;
        prev_abs = g_deq.abs();
        // Update PCM target: subtract g_deq * ZSR. Add the excitation
        // contribution `g_deq * cbvec` to the reconstructed excitation.
        let zsr = &zsrs[best_idx as usize];
        let cbv = extract_cbvec_veclen(cb_mem, best_idx, SUBL);
        for n in 0..SUBL {
            t[n] -= g_deq * zsr[n];
            excitation[n] += g_deq * cbv[n];
        }
    }
    (CbSearchResult { cb_idx, gain_idx }, excitation)
}

/// Compute the zero-state response of the all-pole synth filter
/// `1/A(z)` fed `input`.
fn zero_state_response(input: &[f32; SUBL], a: &[f32; LPC_ORDER + 1]) -> [f32; SUBL] {
    let mut mem = [0.0f32; LPC_ORDER];
    let mut out = [0.0f32; SUBL];
    for n in 0..SUBL {
        let mut s = input[n];
        for k in 1..=LPC_ORDER {
            s -= a[k] * mem[k - 1];
        }
        out[n] = s;
        for k in (1..LPC_ORDER).rev() {
            mem[k] = mem[k - 1];
        }
        mem[0] = s;
    }
    out
}

/// Update the 147-sample codebook memory after encoding one excitation
/// sub-block — identical to the decoder's `update_cb_memory`.
pub fn push_excitation(cb_mem: &mut [f32; CB_LMEM], excitation: &[f32; SUBL]) {
    update_cb_memory(cb_mem, excitation);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cb::{construct_excitation, extract_cbvec};

    #[test]
    fn cb_size_40_matches_decoder() {
        assert_eq!(total_cb_size(CB_LMEM, SUBL), 256);
        assert_eq!(total_cb_size(CB_LMEM, 40), 256);
    }

    #[test]
    fn cb_size_22_is_128() {
        // 22/23 segment: base (85-22+1=64 for lMem=85; here we use
        // lMem=CB_LMEM but the RFC uses 85 for the boundary). Our
        // wrapper uses whatever `cb_mem.len()` is.
        assert_eq!(total_cb_size(85, 22), 2 * (85 - 22 + 1));
    }

    #[test]
    fn gain_quantisers_endpoints() {
        // Positive max gain → top of gain_sq5Tbl.
        let (i, _) = quantise_gain0(10.0);
        assert_eq!(i, 31);
        let (i, _) = quantise_gain0(-10.0);
        assert_eq!(i, 0);

        let (i, _) = quantise_gain1(10.0, 1.0);
        assert_eq!(i, 15);
        let (i, _) = quantise_gain1(-10.0, 1.0);
        assert_eq!(i, 0);
    }

    #[test]
    fn search_cb_matches_with_known_index() {
        // Build a memory where index 0 cbvec has known content.
        // Then construct the ideal `excitation = gain0*v0` and search:
        // the first stage must pick idx 0 with a positive gain close to
        // `gain0`.
        let mut mem = [0.0f32; CB_LMEM];
        // Fill with a simple ramp so extracted vectors differ.
        for i in 0..CB_LMEM {
            mem[i] = ((i as f32) * 0.1).sin() * 100.0;
        }
        let v0 = extract_cbvec(&mem, 0);
        let mut target = [0.0f32; SUBL];
        let gain = 0.5_f32;
        for n in 0..SUBL {
            target[n] = gain * v0[n];
        }
        let (res, _rec) = search_cb_subl(&mem, &target);
        // Stage 0 should pick a vector strongly correlated with target.
        // With this specific target the best idx should be 0 (exact
        // match). The codebook vectors are derived from `mem` slides
        // so the exact idx 0 match is ideal.
        assert_eq!(res.cb_idx[0], 0);
        // Gain idx should place us near 0.5 ∈ gain_sq5Tbl. Table index 12
        // = 0.487488, index 13 = 0.525024 — either is acceptable.
        assert!(res.gain_idx[0] == 12 || res.gain_idx[0] == 13);
    }

    #[test]
    fn search_cb_round_trip_stable() {
        let mut mem = [0.0f32; CB_LMEM];
        for i in 0..CB_LMEM {
            mem[i] = ((i as f32) * 0.3).cos() * 50.0;
        }
        let target = [12.3f32; SUBL];
        let (res, rec) = search_cb_subl(&mem, &target);
        // Reconstruction should match the decoder's own construction.
        let dec = construct_excitation(&mem, &res.cb_idx, &res.gain_idx);
        for n in 0..SUBL {
            assert!(
                (rec[n] - dec[n]).abs() < 1e-3,
                "n={n}: rec={} dec={}",
                rec[n],
                dec[n]
            );
        }
    }

    #[test]
    fn search_cb_zero_target_finds_any_valid() {
        let mut mem = [0.0f32; CB_LMEM];
        for i in 0..CB_LMEM {
            mem[i] = (i as f32).cos();
        }
        let target = [0.0f32; SUBL];
        let (res, _) = search_cb_subl(&mem, &target);
        // Any index is fine; just verify bounds.
        assert!(res.cb_idx[0] < 256);
        assert!(res.gain_idx[0] < 32);
    }
}
