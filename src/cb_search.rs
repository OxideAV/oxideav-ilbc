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
//!      * `target·cbvec > 0`
//!      * `|gain| < CB_MAXGAIN = 1.3`
//!
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

/// LPC chirp factor for the perceptual weighting filter
/// `Wk(z) = 1/Ak(z/LPC_CHIRP_WEIGHTDENUM)`.
/// RFC 3951 §3.4 (line 820): `LPC_CHIRP_WEIGHTDENUM = 0.4222`.
const LPC_CHIRP_WEIGHTDENUM: f32 = 0.4222;

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

/// Multistage codebook search. Searches 3 stages, updating the target
/// after each one. Also reconstructs the excitation sub-block exactly
/// as the decoder would from the chosen indices.
pub fn search_cb(cb_mem: &[f32], cbveclen: usize, target: &[f32]) -> (CbSearchResult, Vec<f32>) {
    search_cb_capped(cb_mem, cbveclen, target, &[usize::MAX; 3])
}

/// Same as [`search_cb`] but with a per-stage cap on the search index
/// range. Use this where the bitstream allocates fewer bits to a stage
/// than the natural codebook size (per RFC 3951 Table 3.1 / Table 3.2).
///
/// `caps[k]` limits stage `k`'s search to indices `0..caps[k].min(total)`.
pub fn search_cb_capped(
    cb_mem: &[f32],
    cbveclen: usize,
    target: &[f32],
    caps: &[usize; 3],
) -> (CbSearchResult, Vec<f32>) {
    debug_assert_eq!(target.len(), cbveclen);

    let mut t = target.to_vec();
    let mut cb_idx = [0u16; 3];
    let mut gain_idx = [0u8; 3];
    let mut reconstructed = vec![0.0f32; cbveclen];
    let mut prev_abs = 1.0f32;

    for stage in 0..3 {
        let stage0 = stage == 0;
        let (idx, gain, vec) = search_stage_capped(cb_mem, cbveclen, &t, stage0, caps[stage]);
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

/// As [`search_cb_capped`] but with the RFC 3951 §3.7 "Gain Correction
/// Encoding" post-pass: bumps `gain_idx[0]` upward (capped at 2× the
/// original quantised value) until the reconstructed-excitation energy
/// approaches the target energy, fixing the systematic energy loss that
/// the squared-error CB search introduces on unvoiced/noise-like input.
///
/// Reference: RFC 3951 §3.7 + Appendix A.34 (`iCBSearch`, lines 9050-9065).
///
/// Returns the (possibly-rescaled) indices plus the recomputed
/// excitation — recomputed because the encoder must keep its codebook
/// memory in lockstep with the decoder, which sees the rescaled gains.
pub fn search_cb_capped_with_gain_correction(
    cb_mem: &[f32],
    cbveclen: usize,
    target: &[f32],
    caps: &[usize; 3],
) -> (CbSearchResult, Vec<f32>) {
    let (mut res, reconstructed) = search_cb_capped(cb_mem, cbveclen, target, caps);

    // Target energy (unweighted residual domain — matches what the
    // search itself optimised against).
    let mut tene = 0.0f32;
    for &t in target.iter() {
        tene += t * t;
    }
    // Coded-vector energy (sum of three gain-scaled CB vecs == reconstructed).
    let mut cene = 0.0f32;
    for &c in reconstructed.iter() {
        cene += c * c;
    }

    // Quantised stage-0 gain (scale 1.0 ⇒ value == table[index]).
    let g0_quant = GAIN_SQ5_TBL[res.gain_idx[0] as usize];
    let g0_sq = g0_quant * g0_quant;
    if tene <= 0.0 || cene <= 0.0 || g0_sq <= 0.0 {
        return (res, reconstructed);
    }

    // Per RFC §3.7 / iCBSearch:
    //   for i in gain_idx[0]..32: if cene*tbl[i]^2 < tene*g0^2 and
    //                              tbl[j] < 2*g0  ⇒ j = i
    // Reference compares `gain_sq5Tbl[j]` (i.e. the *currently-best*
    // candidate) against the cap, not `gain_sq5Tbl[i]`. We mirror that
    // exactly so the bump can stop early when even the running pick
    // already saturates the 2× ceiling.
    let start_idx = res.gain_idx[0] as usize;
    let mut j = start_idx;
    let cap_2x = 2.0 * g0_quant;
    let target_term = tene * g0_sq;
    for (i, &v_i) in GAIN_SQ5_TBL.iter().enumerate().skip(start_idx) {
        let ftmp = cene * v_i * v_i;
        if ftmp < target_term && GAIN_SQ5_TBL[j] < cap_2x {
            j = i;
        }
    }

    if j == start_idx {
        // No bump — return original.
        return (res, reconstructed);
    }

    // Rebuild the excitation with the bumped stage-0 gain. Stages 1 and
    // 2 are dequantised against `max(0.1, |g_prev_dec|)`, which scales
    // *linearly* with g0 as long as the floor isn't hit; we therefore
    // re-derive all three dequantised gains from the (possibly new)
    // stage-0 index using the same chain the decoder will run.
    res.gain_idx[0] = j as u8;
    let new_exc = rebuild_excitation(cb_mem, cbveclen, &res.cb_idx, &res.gain_idx);
    (res, new_exc)
}

/// Search variant matching RFC 3951 §3.4 / §3.6.2: the codebook memory
/// and target are passed through the perceptual weighting filter
/// `Wk(z) = 1/Ak(z/LPC_CHIRP_WEIGHTDENUM)` (i.e. an all-pole synthesis
/// filter on the BW-expanded LPC) before the multistage NN search runs.
/// The reconstruction is done in the **unweighted** domain — the
/// returned excitation matches what the decoder will build from the
/// chosen indices, so the encoder's CB memory stays in lockstep with
/// the decoder.
///
/// `a` is the *unquantised* sub-block LPC `[1, a1..a10]`. The weighting
/// filter chirps it inward by 0.4222 to flatten formant peaks before
/// the search. `weight_state` is the 10-sample synthesis-filter memory;
/// it is reset to zero at the start of each sub-block (matches the
/// reference encoder, which also memsets `weightState` between
/// sub-frames — see RFC 3951 Appendix A.34 / A.38 lines 3257, 3294).
///
/// Reference: `iCBSearch` in Appendix A.34 (the `AllPoleFilter(buf+...,
/// weightDenum, ...)` call before the multistage loop).
pub fn search_cb_weighted_with_gain_correction(
    cb_mem: &[f32],
    cbveclen: usize,
    target: &[f32],
    a: &[f32; LPC_ORDER + 1],
    caps: &[usize; 3],
) -> (CbSearchResult, Vec<f32>) {
    debug_assert_eq!(target.len(), cbveclen);

    // Build the BW-expanded LPC for the weighting filter.
    let mut a_bw = *a;
    let mut c = 1.0f32;
    for a_k in a_bw.iter_mut() {
        *a_k *= c;
        c *= LPC_CHIRP_WEIGHTDENUM;
    }

    // Filter (cb_mem | target) through `1 / a_bw` with shared memory,
    // starting from a zero state (matches the reference's
    // `memset(weightState, 0, ...)` between sub-frames).
    let lmem = cb_mem.len();
    let mut buf = Vec::with_capacity(lmem + cbveclen);
    buf.extend_from_slice(cb_mem);
    buf.extend_from_slice(target);
    let mut wmem = [0.0f32; LPC_ORDER];
    for x in buf.iter_mut() {
        let mut s = *x;
        for k in 1..=LPC_ORDER {
            s -= a_bw[k] * wmem[k - 1];
        }
        *x = s;
        for k in (1..LPC_ORDER).rev() {
            wmem[k] = wmem[k - 1];
        }
        wmem[0] = s;
    }
    // Slice out the weighted halves.
    let (w_cb_mem, w_target) = buf.split_at(lmem);

    // Run the standard multistage NN search in the weighted domain.
    let (res_w, w_recon) = search_cb_capped(w_cb_mem, cbveclen, w_target, caps);

    // §3.7 gain correction in the **weighted** domain — matches the
    // reference `iCBSearch` which computes `tene` and `cene` from the
    // weighted target and weighted reconstruction (see RFC 3951
    // Appendix A.34, lines 8622-8633 and 9050-9065).
    let mut res = res_w;
    let mut tene = 0.0f32;
    for &t in w_target.iter() {
        tene += t * t;
    }
    let mut cene = 0.0f32;
    for &c in w_recon.iter() {
        cene += c * c;
    }
    let g0_quant = GAIN_SQ5_TBL[res.gain_idx[0] as usize];
    let g0_sq = g0_quant * g0_quant;
    if tene > 0.0 && cene > 0.0 && g0_sq > 0.0 {
        let start_idx = res.gain_idx[0] as usize;
        let mut j = start_idx;
        let cap_2x = 2.0 * g0_quant;
        let target_term = tene * g0_sq;
        for (i, &v_i) in GAIN_SQ5_TBL.iter().enumerate().skip(start_idx) {
            let ftmp = cene * v_i * v_i;
            if ftmp < target_term && GAIN_SQ5_TBL[j] < cap_2x {
                j = i;
            }
        }
        if j != start_idx {
            res.gain_idx[0] = j as u8;
        }
    }

    // Rebuild the **unweighted** excitation from the (possibly-bumped)
    // indices, so the encoder's `cb_mem` evolves identically to the
    // decoder's.
    let unweighted_recon = rebuild_excitation(cb_mem, cbveclen, &res.cb_idx, &res.gain_idx);
    (res, unweighted_recon)
}

/// Re-extract the codebook vectors at the chosen indices and combine
/// them with the dequantised gains from those gain indices. Mirrors the
/// decoder's `construct_excitation` / `construct_excitation_veclen`
/// routines but stays self-contained for use inside the search.
fn rebuild_excitation(
    cb_mem: &[f32],
    cbveclen: usize,
    cb_idx: &[u16; 3],
    gain_idx: &[u8; 3],
) -> Vec<f32> {
    // Decoder-equivalent gain chain.
    const FLOOR: f32 = GAIN_SCALE_FLOOR;
    let g0 = GAIN_SQ5_TBL[gain_idx[0] as usize] * 1.0_f32.max(FLOOR);
    let s1 = g0.abs().max(FLOOR);
    let g1 = GAIN_SQ4_TBL[gain_idx[1] as usize] * s1;
    let s2 = g1.abs().max(FLOOR);
    let g2 = GAIN_SQ3_TBL[gain_idx[2] as usize] * s2;
    let gains = [g0, g1, g2];

    let mut out = vec![0.0f32; cbveclen];
    for stage in 0..3 {
        let v = extract_cbvec_veclen(cb_mem, cb_idx[stage], cbveclen);
        for (o, &v_n) in out.iter_mut().zip(v.iter()) {
            *o += gains[stage] * v_n;
        }
    }
    out
}

/// As [`search_stage`] but limit the search range to `cap` candidates.
fn search_stage_capped(
    cb_mem: &[f32],
    cbveclen: usize,
    target: &[f32],
    stage0: bool,
    cap: usize,
) -> (u16, f32, Vec<f32>) {
    let total = total_cb_size(cb_mem.len(), cbveclen).min(cap);
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
        if measure > best_measure {
            best_measure = measure;
            best_idx = i as u16;
            best_gain = gain;
            best_vec = v;
        }
    }
    if best_measure.is_infinite() {
        let v = extract_cbvec_veclen(cb_mem, 0, cbveclen);
        return (0, 0.0, v);
    }
    (best_idx, best_gain, best_vec)
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
        for (i, zsr) in zsrs.iter().take(total).enumerate() {
            let mut dot = 0.0f32;
            let mut nrm = 0.0f32;
            for (&t_n, &z_n) in t.iter().zip(zsr.iter()) {
                dot += t_n * z_n;
                nrm += z_n * z_n;
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
        for ((t_n, e_n), (&z_n, &c_n)) in t
            .iter_mut()
            .zip(excitation.iter_mut())
            .zip(zsr.iter().zip(cbv.iter()))
        {
            *t_n -= g_deq * z_n;
            *e_n += g_deq * c_n;
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
        // Fill with a simple ramp so extracted vectors differ.
        let mem: [f32; CB_LMEM] = core::array::from_fn(|i| ((i as f32) * 0.1).sin() * 100.0);
        let v0 = extract_cbvec(&mem, 0);
        let gain = 0.5_f32;
        let target: [f32; SUBL] = core::array::from_fn(|n| gain * v0[n]);
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
        let mem: [f32; CB_LMEM] = core::array::from_fn(|i| ((i as f32) * 0.3).cos() * 50.0);
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
        let mem: [f32; CB_LMEM] = core::array::from_fn(|i| (i as f32).cos());
        let target = [0.0f32; SUBL];
        let (res, _) = search_cb_subl(&mem, &target);
        // Any index is fine; just verify bounds.
        assert!(res.cb_idx[0] < 256);
        assert!(res.gain_idx[0] < 32);
    }
}
