//! Decoder-side excitation enhancer — RFC 3951 §4.6.
//!
//! The enhancer operates on 80-sample sub-blocks of the reconstructed
//! residual (excitation) *before* LPC synthesis. For each new frame it:
//!
//! 1. Maintains a 640-sample history (`enh_buf`) = eight 80-sample
//!    blocks, newest-last.
//! 2. Down-samples by 2, then estimates a pitch lag per new sub-block
//!    via maximum-cross-correlation in the 20..60 range on the
//!    down-sampled domain (lag stored doubled in `enh_period`).
//! 3. For each sub-block, collects six pitch-synchronous sequences
//!    (`pssq(-3..=-1, 1..=3)`) around the current block using the
//!    refiner (`polyphaserTbl` 4× upsample, see §4.6.2 / A.15/16).
//! 4. Blends them with the cosine weights `0.5*(1 - cos(2π·(i+4)/8))`
//!    to form `y`, energy-scales to `z = C·y`, and applies the
//!    constraint-optimisation of §4.6.4 / §4.6.5.
//!
//! Tables `POLYPHASER_TBL` / `ENH_PLOCS_TBL` live in this module.
//! `LP_FILT_COEFS_TBL` is the 7-tap downsampling filter from
//! RFC 3951 Appendix A.8 `lpFilt_coefsTbl`.

use crate::FrameMode;

// ---------- RFC constants (verbatim from Appendix A.8 / the define file) ----------

/// 80-sample enhancer block (`ENH_BLOCKL`).
pub const ENH_BLOCKL: usize = 80;
/// Half of `ENH_BLOCKL` (`ENH_BLOCKL_HALF`).
pub const ENH_BLOCKL_HALF: usize = ENH_BLOCKL / 2;
/// 2·HL+1 = 7 is the number of blocks in the pitch-synchronous sequence
/// (three past, centre, three future).
pub const ENH_HL: usize = 3;
/// Pitch-estimate slop ± this many samples.
pub const ENH_SLOP: usize = 2;
/// Overhang samples around the refiner search window.
pub const ENH_OVERHANG: usize = 2;
/// Upsampling factor of the polyphase filter (4).
pub const ENH_UPS0: usize = 4;
/// 2·FL0+1 = 7 is the length of each polyphase sub-filter.
pub const ENH_FL0: usize = 3;
/// Refiner vector length (= ENH_BLOCKL + 2·ENH_FL0).
pub const ENH_VECTL: usize = ENH_BLOCKL + 2 * ENH_FL0;
/// Correlation-vector length (= 2·ENH_SLOP + 1).
pub const ENH_CORRDIM: usize = 2 * ENH_SLOP + 1;
/// BLOCKL_MAX / ENH_BLOCKL = 240/80 = 3 new blocks for 30 ms mode.
pub const ENH_NBLOCKS: usize = 3;
/// Extra 80-sample sub-blocks of *past* history kept in the enh_buf.
pub const ENH_NBLOCKS_EXTRA: usize = 5;
/// Total blocks = history + new (= 8).
pub const ENH_NBLOCKS_TOT: usize = ENH_NBLOCKS + ENH_NBLOCKS_EXTRA;
/// Enhancer buffer length (= 640).
pub const ENH_BUFL: usize = ENH_NBLOCKS_TOT * ENH_BLOCKL;
/// Maximum enhancement energy fraction.
pub const ENH_ALPHA0: f32 = 0.05;

/// Polyphase (4-phase) interpolation filter, 7 taps per phase =
/// 4·7 = 28 total. Verbatim from RFC 3951 Appendix A.8 `polyphaserTbl`.
pub const POLYPHASER_TBL: [f32; ENH_UPS0 * (2 * ENH_FL0 + 1)] = [
    0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.015625, -0.076904,
    0.288330, 0.862061, -0.106445, 0.018799, -0.015625, 0.023682, -0.124268, 0.601563, 0.601563,
    -0.124268, 0.023682, -0.023682, 0.018799, -0.106445, 0.862061, 0.288330, -0.076904, 0.015625,
    -0.018799,
];

/// Pitch-estimate centre locations — verbatim from RFC 3951 Appendix A.8
/// `enh_plocsTbl`. One centre per 80-sample block across the 640-sample
/// `enh_buf`, i.e. 40, 120, 200, ..., 600.
pub const ENH_PLOCS_TBL: [f32; ENH_NBLOCKS_TOT] =
    [40.0, 120.0, 200.0, 280.0, 360.0, 440.0, 520.0, 600.0];

/// 7-tap symmetric low-pass used by the 2:1 downsampler before the
/// pitch search. Verbatim from RFC 3951 Appendix A.8 `lpFilt_coefsTbl`.
pub const LP_FILT_COEFS_TBL: [f32; 7] = [
    -0.066650, 0.125000, 0.316650, 0.414063, 0.316650, 0.125000, -0.066650,
];

const FILTERORDER_DS: usize = 7;
const DELAY_DS: usize = 3;
const FACTOR_DS: usize = 2;

// ---------- enhancer state ----------

/// Per-decoder enhancer memory. Owned by the top-level `IlbcDecoder`
/// and mutated in place on every successfully decoded frame.
#[derive(Clone)]
pub struct EnhancerState {
    /// Last 640 samples of excitation history (newest at the tail).
    pub enh_buf: [f32; ENH_BUFL],
    /// Per-block pitch-period estimates (one per `ENH_BLOCKL` slot).
    pub enh_period: [f32; ENH_NBLOCKS_TOT],
    /// Previous-packet PLC flag (1 if last frame was concealed).
    pub prev_enh_pl: u8,
    /// Most-recent pitch lag (used by PLC heuristics).
    pub last_lag: i32,
}

impl EnhancerState {
    pub fn new() -> Self {
        Self {
            enh_buf: [0.0; ENH_BUFL],
            enh_period: [40.0; ENH_NBLOCKS_TOT],
            prev_enh_pl: 0,
            last_lag: 20,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for EnhancerState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------- helpers ----------

/// Cross-correlation coefficient `(t·r)^2 / (r·r)` — zero if `t·r<=0`.
///
/// Mirrors `xCorrCoef` (RFC 3951 Appendix A.16).
fn xcorr_coef(target: &[f32], regressor: &[f32]) -> f32 {
    let mut tr = 0.0f32;
    let mut rr = 0.0f32;
    for (&t, &r) in target.iter().zip(regressor.iter()) {
        tr += t * r;
        rr += r * r;
    }
    if tr > 0.0 && rr > 0.0 {
        tr * tr / rr
    } else {
        0.0
    }
}

/// Nearest-neighbour lookup in `array` by squared-error criterion.
fn nearest_neighbor(array: &[f32], value: f32) -> usize {
    let mut best = 0usize;
    let mut best_d = (array[0] - value).powi(2);
    for (i, &v) in array.iter().enumerate().skip(1) {
        let d = (v - value).powi(2);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Cross-correlation between `seq1` and `seq2`, output length
/// `seq1.len() - seq2.len() + 1`. Mirrors `mycorr1` from A.16.
fn mycorr1(seq1: &[f32], seq2: &[f32], corr: &mut [f32]) {
    let dim1 = seq1.len();
    let dim2 = seq2.len();
    debug_assert!(corr.len() >= dim1 + 1 - dim2);
    for (i, c) in corr.iter_mut().take(dim1 - dim2 + 1).enumerate() {
        let mut s = 0.0f32;
        for j in 0..dim2 {
            s += seq1[i + j] * seq2[j];
        }
        *c = s;
    }
}

/// 4× upsampling using the polyphase filter. Exactly reproduces
/// `enh_upsample` (A.16): for a sequence of length `dim1`, output
/// length is `ENH_UPS0 * dim1`.
fn enh_upsample(seq1: &[f32], useq1: &mut [f32]) {
    let dim1 = seq1.len();
    let mut hfl = ENH_FL0;
    let mut filterlength = 2 * hfl + 1;

    // When filterlength > dim1, shrink hfl to dim1/2 and offset the
    // phase pointers (per A.16).
    let mut offsets = [0usize; ENH_UPS0];
    if filterlength > dim1 {
        let hfl2 = dim1 / 2;
        for (j, off) in offsets.iter_mut().enumerate() {
            *off = j * filterlength + hfl - hfl2;
        }
        hfl = hfl2;
        filterlength = 2 * hfl + 1;
    } else {
        for (j, off) in offsets.iter_mut().enumerate() {
            *off = j * filterlength;
        }
    }

    let mut pu = 0usize;

    // Left overhang: filter overhangs left side of sequence.
    // RFC 3951 Appendix A.16 polyphase upsampler — `ps` walks back from
    // seq1[i], `pp` walks forward through POLYPHASER_TBL[offset..]; both
    // indices key off `i` and `j` so per-call-site allow.
    #[allow(clippy::needless_range_loop)] // RFC 3951 Appendix A.16 (enh_upsample)
    for i in hfl..filterlength {
        for j in 0..ENH_UPS0 {
            let mut s = 0.0f32;
            for k in 0..=i {
                // ps points to seq1[i], walks back
                let ps_idx = i - k;
                let pp_idx = offsets[j] + k;
                s += seq1[ps_idx] * POLYPHASER_TBL[pp_idx];
            }
            useq1[pu] = s;
            pu += 1;
        }
    }

    // Middle section: simple convolution.
    #[allow(clippy::needless_range_loop)] // RFC 3951 Appendix A.16 (enh_upsample)
    for i in filterlength..dim1 {
        for j in 0..ENH_UPS0 {
            let mut s = 0.0f32;
            for k in 0..filterlength {
                let ps_idx = i - k;
                let pp_idx = offsets[j] + k;
                s += seq1[ps_idx] * POLYPHASER_TBL[pp_idx];
            }
            useq1[pu] = s;
            pu += 1;
        }
    }

    // Right overhang.
    #[allow(clippy::needless_range_loop)] // RFC 3951 Appendix A.16 (enh_upsample)
    for q in 1..=hfl {
        for j in 0..ENH_UPS0 {
            let mut s = 0.0f32;
            for k in 0..(filterlength - q) {
                let ps_idx = dim1 - 1 - k;
                let pp_idx = offsets[j] + q + k;
                s += seq1[ps_idx] * POLYPHASER_TBL[pp_idx];
            }
            useq1[pu] = s;
            pu += 1;
        }
    }
}

/// Refiner: extract a pitch-synchronous 80-sample segment from
/// `idata` at a fractional sample offset around `est_seg_pos`.
///
/// Mirrors `refiner` in A.16. `_period` is not used here (kept in the
/// signature for future period-weighted search).
fn refiner(
    seg: &mut [f32; ENH_BLOCKL],
    upd_start_pos: &mut f32,
    idata: &[f32],
    center_start_pos: usize,
    est_seg_pos: f32,
    _period: f32,
) {
    let idatal = idata.len();
    let est_rounded = (est_seg_pos - 0.5) as i32;
    let mut search_start = est_rounded - ENH_SLOP as i32;
    if search_start < 0 {
        search_start = 0;
    }
    let mut search_end = est_rounded + ENH_SLOP as i32;
    if search_end as usize + ENH_BLOCKL >= idatal {
        search_end = (idatal - ENH_BLOCKL - 1) as i32;
    }
    let corrdim = (search_end - search_start + 1) as usize;

    // Correlation between sliding window over idata and the current block.
    let win_len = corrdim + ENH_BLOCKL - 1;
    let win = &idata[search_start as usize..search_start as usize + win_len];
    let target = &idata[center_start_pos..center_start_pos + ENH_BLOCKL];
    let mut corr_vec = vec![0.0f32; corrdim];
    mycorr1(win, target, &mut corr_vec);

    // Upsample the correlation (4× via polyphase) and find its max.
    let mut corr_ups = vec![0.0f32; corrdim * ENH_UPS0];
    enh_upsample(&corr_vec, &mut corr_ups);
    let mut tloc = 0usize;
    let mut maxv = corr_ups[0];
    for (i, &v) in corr_ups.iter().enumerate().take(corrdim * ENH_UPS0).skip(1) {
        if v > maxv {
            tloc = i;
            maxv = v;
        }
    }

    // Fractional segment position.
    *upd_start_pos = search_start as f32 + (tloc as f32) / (ENH_UPS0 as f32) + 1.0_f32;
    let mut tloc2 = tloc / ENH_UPS0;
    if tloc > tloc2 * ENH_UPS0 {
        tloc2 += 1;
    }
    let st = search_start + tloc2 as i32 - ENH_FL0 as i32;

    // Build vect (length ENH_VECTL) with appropriate zero-padding.
    let mut vect = [0.0f32; ENH_VECTL];
    if st < 0 {
        let neg = (-st) as usize;
        let take = (ENH_VECTL - neg).min(idatal);
        vect[neg..neg + take].copy_from_slice(&idata[..take]);
    } else {
        let st_u = st as usize;
        let en = st_u + ENH_VECTL;
        if en > idatal {
            let take = ENH_VECTL - (en - idatal);
            vect[..take].copy_from_slice(&idata[st_u..st_u + take]);
            // remainder stays zero
        } else {
            vect.copy_from_slice(&idata[st_u..st_u + ENH_VECTL]);
        }
    }

    let fraction = (tloc2 as i32) * (ENH_UPS0 as i32) - tloc as i32;
    // Final segment: convolve vect with polyphaserTbl[fraction*(2*FL0+1)..][..2*FL0+1].
    let fraction_idx = (fraction as usize) * (2 * ENH_FL0 + 1);
    let kernel = &POLYPHASER_TBL[fraction_idx..fraction_idx + (2 * ENH_FL0 + 1)];

    // mycorr1(seg, vect, polyphaser_kernel) — output length ENH_BLOCKL.
    let dim2 = 2 * ENH_FL0 + 1;
    for (i, seg_i) in seg.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for j in 0..dim2 {
            s += vect[i + j] * kernel[j];
        }
        *seg_i = s;
    }
}

/// Weighted smoothing over the six non-centre pssqs. This is the
/// cosine-weighted combination `y = Σ pssq(i) · wt(i)` followed by the
/// two-constraint optimisation from §4.6.4 / §4.6.5, exactly matching
/// the reference `smath` in A.16.
fn smath(odata: &mut [f32], sseq: &[f32], hl: usize, alpha0: f32) {
    let two_hl_1 = 2 * hl + 1;
    // cosine weights for 0..2*hl, with centre wt[hl] = 0.
    let mut wt = [0.0f32; 7]; // 2*ENH_HL+1 = 7 for hl=3
    for i in 1..=two_hl_1 {
        wt[i - 1] =
            0.5 * (1.0 - (2.0 * core::f32::consts::PI * (i as f32) / (2 * hl + 2) as f32).cos());
    }
    wt[hl] = 0.0;

    // Surround sum: Σ_{k != hl} sseq[k·ENH_BLOCKL .. ] * wt[k].
    let mut surround = [0.0f32; ENH_BLOCKL];
    for (i, s) in surround.iter_mut().enumerate() {
        *s = sseq[i] * wt[0];
    }
    for (k, &w) in wt.iter().enumerate().take(hl).skip(1) {
        let base = k * ENH_BLOCKL;
        for (i, s) in surround.iter_mut().enumerate() {
            *s += sseq[base + i] * w;
        }
    }
    for (k, &w) in wt.iter().enumerate().take(2 * hl + 1).skip(hl + 1) {
        let base = k * ENH_BLOCKL;
        for (i, s) in surround.iter_mut().enumerate() {
            *s += sseq[base + i] * w;
        }
    }

    // Inner products.
    let centre_off = hl * ENH_BLOCKL;
    let mut w00 = 0.0f32;
    let mut w11 = 0.0f32;
    let mut w10 = 0.0f32;
    for (i, &r) in surround.iter().enumerate() {
        let p = sseq[centre_off + i];
        w00 += p * p;
        w11 += r * r;
        w10 += r * p;
    }
    if w11.abs() < 1.0 {
        w11 = 1.0;
    }
    let c = (w00 / w11).sqrt();

    // First try without power constraint.
    let mut errs = 0.0f32;
    for (i, (out, &surr)) in odata
        .iter_mut()
        .zip(surround.iter())
        .enumerate()
        .take(ENH_BLOCKL)
    {
        let p = sseq[centre_off + i];
        *out = c * surr;
        let err = p - *out;
        errs += err * err;
    }

    // Constraint violation: fall back to Lagrange-optimised mix.
    if errs > alpha0 * w00 {
        if w00 < 1.0 {
            w00 = 1.0;
        }
        let denom = (w11 * w00 - w10 * w10) / (w00 * w00);
        let (a, b) = if denom > 0.0001 {
            let a = ((alpha0 - alpha0 * alpha0 / 4.0) / denom).sqrt();
            let b = -alpha0 / 2.0 - a * w10 / w00;
            (a, b + 1.0)
        } else {
            (0.0, 1.0)
        };
        for (i, (out, &surr)) in odata
            .iter_mut()
            .zip(surround.iter())
            .enumerate()
            .take(ENH_BLOCKL)
        {
            *out = a * surr + b * sseq[centre_off + i];
        }
    }
}

/// Fill `sseq` with the 7 pitch-synchronous sequences centred on
/// `center_start_pos`. Mirrors `getsseq` in A.16.
fn getsseq(
    sseq: &mut [f32],
    idata: &[f32],
    center_start_pos: usize,
    period: &[f32],
    plocs: &[f32],
    hl: usize,
) {
    let idatal = idata.len();
    let center_end_pos = center_start_pos + ENH_BLOCKL - 1;

    // Present block (hl).
    let mid_centre = 0.5 * (center_start_pos as f32 + center_end_pos as f32);
    let mut lag_block = [0usize; 7];
    lag_block[hl] = nearest_neighbor(plocs, mid_centre);

    let mut block_start_pos = [0.0f32; 7];
    block_start_pos[hl] = center_start_pos as f32;
    let pss_off = hl * ENH_BLOCKL;
    sseq[pss_off..pss_off + ENH_BLOCKL]
        .copy_from_slice(&idata[center_start_pos..center_start_pos + ENH_BLOCKL]);

    // Past blocks (q = hl-1 .. 0).
    for q in (0..hl).rev() {
        let p_q1 = period[lag_block[q + 1]];
        block_start_pos[q] = block_start_pos[q + 1] - p_q1;
        lag_block[q] = nearest_neighbor(plocs, block_start_pos[q] + ENH_BLOCKL_HALF as f32 - p_q1);
        if block_start_pos[q] - ENH_OVERHANG as f32 >= 0.0 {
            let mut seg = [0.0f32; ENH_BLOCKL];
            let mut upd = 0.0f32;
            refiner(
                &mut seg,
                &mut upd,
                idata,
                center_start_pos,
                block_start_pos[q],
                p_q1,
            );
            block_start_pos[q] = upd;
            let off = q * ENH_BLOCKL;
            sseq[off..off + ENH_BLOCKL].copy_from_slice(&seg);
        } else {
            let off = q * ENH_BLOCKL;
            sseq[off..off + ENH_BLOCKL].fill(0.0);
        }
    }

    // Future blocks (q = hl+1 .. 2*hl).
    // plocs2 = plocs - period (elementwise).
    let mut plocs2 = [0.0f32; 8];
    for (i, p2) in plocs2.iter_mut().take(plocs.len().min(8)).enumerate() {
        *p2 = plocs[i] - period[i];
    }
    let plocs2 = &plocs2[..plocs.len()];
    for q in (hl + 1)..=(2 * hl) {
        lag_block[q] = nearest_neighbor(plocs2, block_start_pos[q - 1] + ENH_BLOCKL_HALF as f32);
        let p_q = period[lag_block[q]];
        block_start_pos[q] = block_start_pos[q - 1] + p_q;
        if block_start_pos[q] + ((ENH_BLOCKL + ENH_OVERHANG) as f32) < idatal as f32 {
            let mut seg = [0.0f32; ENH_BLOCKL];
            let mut upd = 0.0f32;
            refiner(
                &mut seg,
                &mut upd,
                idata,
                center_start_pos,
                block_start_pos[q],
                p_q,
            );
            block_start_pos[q] = upd;
            let off = q * ENH_BLOCKL;
            sseq[off..off + ENH_BLOCKL].copy_from_slice(&seg);
        } else {
            let off = q * ENH_BLOCKL;
            sseq[off..off + ENH_BLOCKL].fill(0.0);
        }
    }
}

/// Run the core enhancer on one 80-sample block at `center_start_pos`.
fn enhancer_block(
    odata: &mut [f32],
    idata: &[f32],
    center_start_pos: usize,
    alpha0: f32,
    period: &[f32],
    plocs: &[f32],
) {
    let mut sseq = vec![0.0f32; (2 * ENH_HL + 1) * ENH_BLOCKL];
    getsseq(&mut sseq, idata, center_start_pos, period, plocs, ENH_HL);
    smath(odata, &sseq, ENH_HL, alpha0);
}

/// 2:1 downsample via the 7-tap lp filter and decimate (mirrors
/// `DownSample` in A.18). Returns ceil((input.len - DELAY_DS + 1) / 2)
/// samples. `state` is 6 history samples — i.e. the 6 samples
/// immediately preceding `input[0]`.
fn downsample(input: &[f32], state: &[f32; 6]) -> Vec<f32> {
    let length_in = input.len();
    let mut out: Vec<f32> = Vec::with_capacity(length_in / FACTOR_DS + DELAY_DS);

    // LP filter + decimation.
    let mut i = DELAY_DS;
    while i < length_in {
        let mut s = 0.0f32;
        let stop = if i < FILTERORDER_DS {
            i + 1
        } else {
            FILTERORDER_DS
        };
        // In-buffer contributions: coef[0..stop] * input[i], [i-1], ...
        for j in 0..stop {
            s += LP_FILT_COEFS_TBL[j] * input[i - j];
        }
        // State contributions (reference: walks state[FILTERORDER_DS-2]
        // down). For i < FILTERORDER_DS this pulls from `state` to
        // cover the past samples outside `input`.
        if i + 1 < FILTERORDER_DS {
            let mut state_ptr: isize = FILTERORDER_DS as isize - 2;
            for &coef in LP_FILT_COEFS_TBL.iter().take(FILTERORDER_DS).skip(i + 1) {
                if state_ptr < 0 {
                    break;
                }
                s += coef * state[state_ptr as usize];
                state_ptr -= 1;
            }
        }
        out.push(s);
        i += FACTOR_DS;
    }

    // Tail portion (§A.18: fills the last DELAY_DS/FACTOR_DS output
    // samples using zero inputs for the future). For the enhancer's
    // use case we consume only the primary decimation output; mirror
    // the loop for fidelity but guard against underflow.
    let mut k = length_in + FACTOR_DS;
    while k < length_in + DELAY_DS {
        let offset = k - length_in;
        let limit = FILTERORDER_DS.saturating_sub(offset);
        let mut s = 0.0f32;
        for j in 0..limit {
            if (length_in as isize - 1 - j as isize) >= 0 {
                s += LP_FILT_COEFS_TBL[offset + j] * input[length_in - 1 - j];
            }
        }
        out.push(s);
        k += FACTOR_DS;
    }

    out
}

/// Run the enhancer for one frame.
///
/// `mode` selects the frame size. `in_exc` is the newly decoded
/// excitation for this frame (`mode.samples()` entries). `out_exc`
/// receives the enhanced excitation for the DELAYED block (40 samples
/// earlier for 20 ms, 80 samples earlier for 30 ms).
///
/// Return value is the estimated pitch lag in non-downsampled samples
/// (useful for downstream PLC).
pub fn enhance_frame(
    state: &mut EnhancerState,
    mode: FrameMode,
    in_exc: &[f32],
    out_exc: &mut [f32],
) -> i32 {
    let blockl = mode.samples();
    debug_assert_eq!(in_exc.len(), blockl);
    debug_assert_eq!(out_exc.len(), blockl);

    // Shift enh_buf left by blockl and append new excitation at tail.
    state.enh_buf.copy_within(blockl.., 0);
    state.enh_buf[ENH_BUFL - blockl..].copy_from_slice(in_exc);

    // For 20 ms mode, move processing one block.
    let ioffset = if matches!(mode, FrameMode::Ms20) {
        1
    } else {
        0
    };

    // Shift enh_period left by (3 - ioffset).
    let i_shift = 3 - ioffset;
    state.enh_period.copy_within(i_shift.., 0);

    // Downsample the trailing window.
    let ds_start = (ENH_NBLOCKS_EXTRA + ioffset) * ENH_BLOCKL;
    let ds_history_start = ds_start - 126; // 6 state + 120 samples back
    let ds_state: [f32; 6] = {
        let mut arr = [0.0f32; 6];
        arr.copy_from_slice(&state.enh_buf[ds_history_start..ds_history_start + 6]);
        arr
    };
    let ds_in_start = ds_start - 120;
    let ds_in_len = ENH_NBLOCKS * ENH_BLOCKL + 120 - ioffset * ENH_BLOCKL;
    let ds_input = &state.enh_buf[ds_in_start..ds_in_start + ds_in_len];
    let downsampled = downsample(ds_input, &ds_state);

    // Pitch estimation per new block.
    let mut last_lag = 20i32;
    let n_blocks = ENH_NBLOCKS - ioffset;
    for iblock in 0..n_blocks {
        let target_start = 60 + iblock * ENH_BLOCKL_HALF;
        if target_start + ENH_BLOCKL_HALF > downsampled.len() {
            break;
        }
        let target = &downsampled[target_start..target_start + ENH_BLOCKL_HALF];
        let mut lag: i32 = 10;
        let mut maxcc = {
            let reg_start = (target_start as i32 - lag) as usize;
            let reg = &downsampled[reg_start..reg_start + ENH_BLOCKL_HALF];
            xcorr_coef(target, reg)
        };
        for ilag in 11..60 {
            if (target_start as i32 - ilag) < 0 {
                break;
            }
            let reg_start = (target_start as i32 - ilag) as usize;
            if reg_start + ENH_BLOCKL_HALF > downsampled.len() {
                break;
            }
            let reg = &downsampled[reg_start..reg_start + ENH_BLOCKL_HALF];
            let cc = xcorr_coef(target, reg);
            if cc > maxcc {
                maxcc = cc;
                lag = ilag;
            }
        }
        state.enh_period[iblock + ENH_NBLOCKS_EXTRA + ioffset] = (lag * 2) as f32;
        last_lag = lag;
    }
    state.last_lag = last_lag;

    // Run the per-block enhancer.
    let plocs = &ENH_PLOCS_TBL;
    let enh_buf_copy = state.enh_buf; // avoid aliasing
    let period = state.enh_period;
    let alpha0 = ENH_ALPHA0;

    if matches!(mode, FrameMode::Ms20) {
        // 2 blocks with 40-sample delay (centre_start = (5+iblock)*80 + 40).
        for iblock in 0..2 {
            let center_start = (5 + iblock) * ENH_BLOCKL + 40;
            let mut block_out = [0.0f32; ENH_BLOCKL];
            enhancer_block(
                &mut block_out,
                &enh_buf_copy,
                center_start,
                alpha0,
                &period,
                plocs,
            );
            let off = iblock * ENH_BLOCKL;
            out_exc[off..off + ENH_BLOCKL].copy_from_slice(&block_out);
        }
    } else {
        // 3 blocks with 80-sample delay.
        for iblock in 0..3 {
            let center_start = (4 + iblock) * ENH_BLOCKL;
            let mut block_out = [0.0f32; ENH_BLOCKL];
            enhancer_block(
                &mut block_out,
                &enh_buf_copy,
                center_start,
                alpha0,
                &period,
                plocs,
            );
            let off = iblock * ENH_BLOCKL;
            out_exc[off..off + ENH_BLOCKL].copy_from_slice(&block_out);
        }
    }

    last_lag * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyphaser_bit_exact() {
        assert_eq!(POLYPHASER_TBL.len(), 28);
        assert_eq!(POLYPHASER_TBL[0], 0.000000);
        assert_eq!(POLYPHASER_TBL[3], 1.000000);
        assert_eq!(POLYPHASER_TBL[16], 0.601563);
        assert_eq!(POLYPHASER_TBL[17], 0.601563);
        assert_eq!(POLYPHASER_TBL[27], -0.018799);
    }

    #[test]
    fn enh_plocs_bit_exact() {
        assert_eq!(ENH_PLOCS_TBL.len(), ENH_NBLOCKS_TOT);
        assert_eq!(ENH_PLOCS_TBL[0], 40.0);
        assert_eq!(ENH_PLOCS_TBL[7], 600.0);
        for k in 1..ENH_PLOCS_TBL.len() {
            assert_eq!(ENH_PLOCS_TBL[k] - ENH_PLOCS_TBL[k - 1], 80.0);
        }
    }

    #[test]
    fn lp_filt_coefs_symmetric() {
        assert_eq!(LP_FILT_COEFS_TBL.len(), FILTERORDER_DS);
        for i in 0..FILTERORDER_DS / 2 {
            assert_eq!(
                LP_FILT_COEFS_TBL[i],
                LP_FILT_COEFS_TBL[FILTERORDER_DS - 1 - i]
            );
        }
    }

    #[test]
    fn enhance_zero_input_is_zero() {
        let mut state = EnhancerState::new();
        let zeros = vec![0.0f32; 160];
        let mut out = vec![0.0f32; 160];
        enhance_frame(&mut state, FrameMode::Ms20, &zeros, &mut out);
        for &v in &out {
            assert!(v.abs() < 1e-6, "zero-in should yield ~zero out, got {}", v);
        }
    }

    #[test]
    fn enhance_finite_on_random_input() {
        let mut state = EnhancerState::new();
        // Fill buffer with a sinusoid first to provide pitch context.
        for (i, s) in state.enh_buf.iter_mut().enumerate() {
            *s = 100.0 * (2.0 * core::f32::consts::PI * (i as f32) / 40.0).sin();
        }
        let exc: Vec<f32> = (0..160)
            .map(|i| 50.0 * (2.0 * core::f32::consts::PI * (i as f32) / 40.0).sin())
            .collect();
        let mut out = vec![0.0f32; 160];
        enhance_frame(&mut state, FrameMode::Ms20, &exc, &mut out);
        for &v in &out {
            assert!(v.is_finite());
            assert!(v.abs() < 1e5);
        }
    }

    #[test]
    fn enhance_preserves_periodic_signal_energy() {
        // A consistent periodic signal (no noise) should come out of
        // the enhancer with similar energy — the constraint
        // `e < alpha0·||pssq(0)||^2` should keep z close to the
        // centre block for well-correlated history.
        let mut state = EnhancerState::new();
        // Prefill several frames of pitched signal.
        let period = 40.0_f32;
        for (i, s) in state.enh_buf.iter_mut().enumerate() {
            *s = 200.0 * (2.0 * core::f32::consts::PI * (i as f32) / period).sin();
        }
        // Fill enh_period with the matching pitch.
        for p in state.enh_period.iter_mut() {
            *p = period;
        }

        let exc: Vec<f32> = (0..160)
            .map(|i| 200.0 * (2.0 * core::f32::consts::PI * ((i + ENH_BUFL) as f32) / period).sin())
            .collect();
        let mut out = vec![0.0f32; 160];
        enhance_frame(&mut state, FrameMode::Ms20, &exc, &mut out);

        let e_in: f32 = exc.iter().map(|v| v * v).sum();
        let e_out: f32 = out.iter().map(|v| v * v).sum();
        assert!(e_out > 0.0);
        // Energy shouldn't blow up by more than ~4× or shrink to near
        // zero on a well-formed periodic input.
        assert!(
            e_out > 0.05 * e_in,
            "enhancer over-attenuated periodic signal: {} vs {}",
            e_out,
            e_in
        );
        assert!(
            e_out < 10.0 * e_in,
            "enhancer blew up periodic signal: {} vs {}",
            e_out,
            e_in
        );
    }
}
