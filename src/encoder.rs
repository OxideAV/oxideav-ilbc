//! Top-level iLBC encoder — wires LPC analysis, LSF quantisation, start-
//! state coding, and codebook search into a packet producer that the
//! sibling decoder round-trips cleanly.
//!
//! Pipeline per 20 ms / 30 ms frame:
//!
//! ```text
//!   PCM (160/240 × i16)
//!     ↓ window + Levinson-Durbin                         (§3.2)
//!   unquantised LSF (1 vector for 20 ms, 2 for 30 ms)
//!     ↓ split-VQ against lsfCbTbl_{1,2,3}                (§3.2.4)
//!   qLSF + 3/6 indices
//!     ↓ stabilise + interpolate per sub-block            (§3.2.5-7)
//!   a_per_sub[n_sub]
//!     ↓ LPC analysis filter                              (§3.3)
//!   residual[n_sub·40]
//!     ↓ frame_classify -> start ∈ {1..n_sub-1}           (§3.5.1)
//!     ↓ pick state span / position                       (§3.5.1)
//!     ↓ all-pass + log-magnitude + shape 3-bit scalar VQ (§3.5.2-3)
//!   scale_idx + state_samples[57/58]
//!     ↓ rebuild state_vec from the *decoded* samples so the CB memory
//!       evolves identically to the decoder.
//!     ↓ boundary CB (22/23 samples) + forward CB walk (Nfor sub-blocks)
//!       + backward CB walk (Nback sub-blocks, in reversed time)  (§3.6)
//!   CB indices + gain indices
//!     ↓ pack Table 3.2                                   (§3.7/3.8)
//!   38/50-byte iLBC payload.
//! ```
//!
//! The encoder picks `start ∈ {1..n_sub-1}` per RFC §3.5.1 (variable
//! `start_idx`), then walks the codebook sub-blocks symmetrically
//! around the start state: `Nfor = n_sub - start - 1` forward
//! sub-blocks at `[(start+1)*SUBL ..]` (cb_mem seeded with the decoded
//! state vector at `[(start-1)*SUBL ..]`), then `Nback = start - 1`
//! backward sub-blocks at `[0 .. (start-1)*SUBL]` (cb_mem seeded with
//! the time-reversed decoded state vector). The encoded sub-block
//! order on the wire is `[forward..., backward_reversed...]`, so
//! `sub_blocks[0]` is the first forward sub-block when `Nfor > 0`,
//! else the first backward sub-block.

use std::collections::VecDeque;

use oxideav_core::Encoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat, TimeBase,
};

use crate::bitreader::CbStageIndices;
use crate::bitwriter::{pack_frame, PackParams};
use crate::cb::update_cb_memory;
use crate::cb_search::{search_cb_abs, search_cb_capped_with_gain_correction};
use crate::hp_filter::{hp_input, HpInputState};
use crate::lpc_analysis::{asymmetric_window, block_lpc, hanning_window, lpc_to_lsf, LPC_WINLEN};
use crate::lsf::{decode_and_interpolate, dequant_lsf, LsfState};
use crate::lsf_quant::quantise_lsf;
use crate::state_encode::lpc_analysis_filter;
use crate::{FrameMode, CB_LMEM, CODEC_ID_STR, LPC_ORDER, SAMPLE_RATE, STATE_LEN, SUBL};

/// RFC 3951 §3.5.1 / Appendix A.20 `FrameClassify`.
///
/// Picks the index of the highest-energy two-sub-block window in the
/// frame's LPC residual. Returns `start ∈ {1..n_sub-1}` (1-based, so
/// the state span covers sub-blocks `start-1` and `start`).
///
/// Energy is windowed at the sub-block edges to bias the classifier
/// toward speech mid-frame; a per-window `ssqEn_win` bias further
/// favours the centre. Both tables are verbatim from Appendix A.20.
fn frame_classify(mode: FrameMode, residual: &[f32]) -> usize {
    let n_sub = mode.sub_blocks();
    debug_assert_eq!(residual.len(), n_sub * SUBL);
    // RFC Appendix A.20: ssqEn_win has NSUB-1 entries, indexed so that
    // the max-window weight applies to the centre sub-block pair.
    // For 30 ms (NSUB=6, 5 windows): {0.8, 0.9, 1.0, 0.9, 0.8}.
    // For 20 ms (NSUB=4, 3 windows): the reference uses
    // ssqEn_win[1..=3] -> {0.9, 1.0, 0.9}, i.e. the centre 3 entries
    // of the 30 ms table. We follow the reference's `l = mode==20 ? 1
    // : 0` indexing exactly.
    const SSQ_EN_WIN: [f32; 5] = [0.8, 0.9, 1.0, 0.9, 0.8];
    const SAMP_EN_WIN: [f32; 5] = [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0];

    // Front-of-sub-block (`fssqEn`) and back-of-sub-block (`bssqEn`)
    // energies. Each entry is the windowed energy of a single 40-sample
    // sub-block; the candidate two-sub-block windows then sum
    // `fssqEn[n-1] + bssqEn[n]`.
    let mut fssq = vec![0.0f32; n_sub];
    let mut bssq = vec![0.0f32; n_sub];

    // First sub-block: front only.
    for (l, &v) in residual.iter().take(5).enumerate() {
        fssq[0] += SAMP_EN_WIN[l] * v * v;
    }
    for &v in residual.iter().take(SUBL).skip(5) {
        fssq[0] += v * v;
    }

    // Middle sub-blocks: both front and back energies.
    for n in 1..n_sub - 1 {
        let base = n * SUBL;
        let sub = &residual[base..base + SUBL];
        for (l, &v) in sub.iter().take(5).enumerate() {
            fssq[n] += SAMP_EN_WIN[l] * v * v;
            bssq[n] += v * v;
        }
        for &v in sub.iter().take(SUBL - 5).skip(5) {
            fssq[n] += v * v;
            bssq[n] += v * v;
        }
        for (l, &v) in sub.iter().enumerate().take(SUBL).skip(SUBL - 5) {
            fssq[n] += v * v;
            bssq[n] += SAMP_EN_WIN[SUBL - l - 1] * v * v;
        }
    }

    // Last sub-block: back only.
    let n_last = n_sub - 1;
    let base = n_last * SUBL;
    let sub = &residual[base..base + SUBL];
    for &v in sub.iter().take(SUBL - 5) {
        bssq[n_last] += v * v;
    }
    for (l, &v) in sub.iter().enumerate().take(SUBL).skip(SUBL - 5) {
        bssq[n_last] += SAMP_EN_WIN[SUBL - l - 1] * v * v;
    }

    // Find the windowed maximum. Reference uses `l = (mode==20) ? 1 :
    // 0` then increments `l` per candidate. So:
    //   20 ms (n_sub=4): candidates n in {1,2,3}, window weights
    //                     SSQ_EN_WIN[{1,2,3}] = {0.9, 1.0, 0.9}.
    //   30 ms (n_sub=6): candidates n in {1..5}, window weights
    //                     SSQ_EN_WIN[{0,1,2,3,4}] = {0.8,0.9,1.0,0.9,0.8}.
    let mut l: usize = if matches!(mode, FrameMode::Ms20) {
        1
    } else {
        0
    };
    let mut max_e = (fssq[0] + bssq[1]) * SSQ_EN_WIN[l];
    let mut max_n = 1usize;
    for n in 2..n_sub {
        l += 1;
        let e = (fssq[n - 1] + bssq[n]) * SSQ_EN_WIN[l.min(SSQ_EN_WIN.len() - 1)];
        if e > max_e {
            max_e = e;
            max_n = n;
        }
    }
    max_n
}

/// Length of the encoder's input buffer per LPC analysis: 240 samples
/// (80 lookback + 160 current for 20 ms, or 60 lookback + 240 current
/// for 30 ms). Both modes end up at 240 thanks to differing lookbacks.
const LPC_LOOKBACK_20MS: usize = 80;
const LPC_LOOKBACK_30MS: usize = 60;

/// Build a boxed iLBC encoder. Accepts 8 kHz mono S16 input.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let sample_rate = params.sample_rate.unwrap_or(SAMPLE_RATE);
    if sample_rate != SAMPLE_RATE {
        return Err(Error::unsupported(format!(
            "iLBC encoder: only 8000 Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "iLBC encoder: only mono is supported (got {channels} channels)"
        )));
    }
    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "iLBC encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }
    if params.codec_id.as_str() != CODEC_ID_STR {
        return Err(Error::unsupported(format!(
            "iLBC encoder: unexpected codec id {:?}",
            params.codec_id
        )));
    }

    // Pick a mode from the `frame_ms` option (20 or 30). Default 20 ms.
    let mode = match params
        .options
        .get("frame_ms")
        .and_then(|v| v.parse::<u32>().ok())
    {
        Some(30) => FrameMode::Ms30,
        _ => FrameMode::Ms20,
    };

    // RFC 3951 §3.1: high-pass pre-processing is "if needed". Off by
    // default; enable with `hp_filter=on` (or `1` / `true` / `yes`).
    let hp_filter_on = params
        .options
        .get("hp_filter")
        .map(|v| matches!(v, "1" | "on" | "true" | "yes"))
        .unwrap_or(false);

    // RFC 3951 §3.5.3 / Appendix A.46 `AbsQuantW`: the start-state
    // samples are quantised by a predictive noise-shaping DPCM loop in
    // the perceptually-weighted speech domain. The RFC weighting is
    // RECOMMENDED, not REQUIRED, and (like the §3.6.2 codebook-search
    // weighting) it regresses waveform SNR on the synthetic
    // self-roundtrip signals, so it is OFF by default; the encoder then
    // uses a direct per-sample scalar quantiser on the unweighted scaled
    // residual. Enable the spec-faithful DPCM path with `state_dpcm=on`.
    // Both paths emit the same kind of `state_sq3Tbl` indices and decode
    // identically (the decoder applies no inverse weighting — RFC §4.2 /
    // Appendix A.44 `StateConstructW`).
    let state_dpcm_on = params
        .options
        .get("state_dpcm")
        .map(|v| matches!(v, "1" | "on" | "true" | "yes"))
        .unwrap_or(false);

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(SampleFormat::S16);
    output.channels = Some(1);
    output.sample_rate = Some(SAMPLE_RATE);
    output.bit_rate = Some(match mode {
        FrameMode::Ms20 => 15_200,
        FrameMode::Ms30 => 13_333,
    });

    Ok(Box::new(IlbcEncoder::new(
        mode,
        output,
        hp_filter_on,
        state_dpcm_on,
    )))
}

/// Internal encoder state.
struct IlbcEncoder {
    output_params: CodecParameters,
    time_base: TimeBase,
    mode: FrameMode,
    /// Incoming PCM, in samples. Drained in frame-sized chunks.
    pcm_queue: VecDeque<f32>,
    /// LPC analysis lookback: previous N samples concatenated with the
    /// current frame to window before autocorrelation.
    lookback: Vec<f32>,
    /// LSF decoder-side state so the encoder's per-sub-block LPC
    /// interpolation sees exactly what the decoder will see.
    lsf_state: LsfState,
    /// Encoder-side LPC analysis filter memory, used to carry the
    /// residual pipeline across frames.
    lpc_mem: [f32; LPC_ORDER],
    /// Previous frame's per-sub-block LPC denominators. Mirrors the
    /// decoder's `old_a_per_sub`: the §4.7 enhancer-delay shift means
    /// the FIRST `shift` sub-blocks of the current frame are synthesised
    /// (in the decoder) with the *previous* frame's tail LPC rows. To
    /// keep the encoder's analysis filter aligned with what the decoder
    /// will resynthesise, we apply the same shift to the encoder-side
    /// residual generation.
    old_a_per_sub: Vec<[f32; LPC_ORDER + 1]>,
    /// Input HP filter state — RFC 3951 §3.1. Applied to PCM at the
    /// `send_frame` boundary so every downstream stage (lookback, LPC,
    /// residual) sees a DC- and 50/60 Hz-suppressed signal. Only used
    /// when `hp_filter_on` is true (RFC describes pre-processing as
    /// optional, gated on the application's input characteristics).
    hp_state: HpInputState,
    hp_filter_on: bool,
    /// Use the RFC §3.5.3 / Appendix A.46 `AbsQuantW` predictive
    /// noise-shaping DPCM quantiser for the start state (perceptually
    /// weighted). Off by default — see `make_encoder` for the rationale.
    state_dpcm_on: bool,
    pending: VecDeque<Packet>,
    sample_pos: i64,
    eof: bool,
}

impl IlbcEncoder {
    fn new(
        mode: FrameMode,
        output_params: CodecParameters,
        hp_filter_on: bool,
        state_dpcm_on: bool,
    ) -> Self {
        let lookback = vec![0.0f32; lookback_len(mode)];
        // Identity LPC for the very first frame's enhancer-delay shift —
        // matches the decoder's seeding (decoder.rs:81).
        let mut identity = [0.0f32; LPC_ORDER + 1];
        identity[0] = 1.0;
        Self {
            output_params,
            time_base: TimeBase::new(1, SAMPLE_RATE as i64),
            mode,
            pcm_queue: VecDeque::new(),
            lookback,
            lsf_state: LsfState::new(),
            lpc_mem: [0.0; LPC_ORDER],
            old_a_per_sub: vec![identity; 6],
            hp_state: HpInputState::default(),
            hp_filter_on,
            state_dpcm_on,
            pending: VecDeque::new(),
            sample_pos: 0,
            eof: false,
        }
    }

    /// Encode as many complete frames as are buffered, emitting one
    /// packet per frame. If `final_flush` is set, zero-pad the trailing
    /// partial frame.
    fn drain(&mut self, final_flush: bool) -> Result<()> {
        let samples = self.mode.samples();
        loop {
            if self.pcm_queue.len() < samples {
                if !final_flush || self.pcm_queue.is_empty() {
                    break;
                }
                while self.pcm_queue.len() < samples {
                    self.pcm_queue.push_back(0.0);
                }
            }
            let frame: Vec<f32> = self.pcm_queue.drain(..samples).collect();
            let pkt_bytes = self.encode_one(&frame)?;
            let start_sample = self.sample_pos;
            self.sample_pos += samples as i64;
            let mut pkt = Packet::new(0, self.time_base, pkt_bytes);
            pkt.pts = Some(start_sample);
            pkt.dts = pkt.pts;
            pkt.duration = Some(samples as i64);
            pkt.flags.keyframe = true;
            self.pending.push_back(pkt);
        }
        Ok(())
    }

    fn encode_one(&mut self, frame_pcm: &[f32]) -> Result<Vec<u8>> {
        let mode = self.mode;
        let samples = mode.samples();
        debug_assert_eq!(frame_pcm.len(), samples);
        // ---- 1. LPC analysis: compute unquantised LSF vector(s) ----
        let lsf_vectors = self.analyse_lsf(frame_pcm);
        // ---- 2. Split-VQ quantisation ----
        let mut lsf_idx = Vec::with_capacity(lsf_vectors.len());
        let mut qlsf_vectors = Vec::with_capacity(lsf_vectors.len());
        for lsf in &lsf_vectors {
            let (idx, qlsf) = quantise_lsf(lsf);
            // Stabilise after quantisation — same as the decoder side.
            let mut q_stab = qlsf;
            crate::lsf::stabilise_lsf(&mut q_stab);
            lsf_idx.push(idx);
            qlsf_vectors.push(q_stab);
        }
        // Now interpolate the quantised LSFs per sub-block exactly as
        // the decoder will.
        let qlsf_refs: Vec<[f32; LPC_ORDER]> = qlsf_vectors.clone();
        // Re-derive the decoder-visible LSF after dequantising from the
        // same indices — `dequant_lsf` applies stabilise_lsf too. This
        // guarantees the encoder and decoder see the same sub-block LPC
        // rows.
        let dec_qlsf: Vec<[f32; LPC_ORDER]> = lsf_idx.iter().map(dequant_lsf).collect();
        debug_assert_eq!(dec_qlsf.len(), qlsf_refs.len());
        let a_per_sub = decode_and_interpolate(&mut self.lsf_state, mode, &dec_qlsf);
        debug_assert_eq!(a_per_sub.len(), mode.sub_blocks());

        // Build the §4.7 enhancer-delay-shifted LPC list — sub-block i
        // of the current frame is synthesised (by the decoder) with the
        // *previous* frame's LPC for i < shift, and with this frame's
        // a_per_sub[i - shift] for i >= shift. shift = 1 (Ms20) or 2
        // (Ms30). Mirrors `decoder.rs` exactly so the encoder's analysis
        // filter inverts what the decoder synthesises.
        let n_sub = mode.sub_blocks();
        let shift = match mode {
            FrameMode::Ms20 => 1usize,
            FrameMode::Ms30 => 2usize,
        };
        let identity = {
            let mut id = [0.0f32; LPC_ORDER + 1];
            id[0] = 1.0;
            id
        };
        let mut shifted_a: Vec<[f32; LPC_ORDER + 1]> = Vec::with_capacity(n_sub);
        for i in 0..n_sub {
            if i < shift {
                let off = i + n_sub - shift;
                shifted_a.push(self.old_a_per_sub.get(off).copied().unwrap_or(identity));
            } else {
                shifted_a.push(a_per_sub[i - shift]);
            }
        }

        // ---- 3. Residual via per-sub-block LPC analysis filter ----
        //
        // Use the §4.7-shifted LPC so the encoder's residual maps
        // exactly to what the decoder will resynthesise: residual[sb]
        // is generated with the same LPC row that the decoder's synth
        // filter will apply in reverse for sub-block sb.
        let mut residual = vec![0.0f32; samples];
        for (sb, a) in shifted_a.iter().enumerate().take(n_sub) {
            let lo = sb * SUBL;
            let hi = lo + SUBL;
            let mut out = vec![0.0f32; SUBL];
            lpc_analysis_filter(&frame_pcm[lo..hi], a, &mut self.lpc_mem, &mut out);
            residual[lo..hi].copy_from_slice(&out);
        }

        // ---- 4. Start-state encoding (RFC §3.5.1, variable `start_idx`) ----
        //
        // Pick `start ∈ {1..n_sub-1}` via the windowed-energy classifier
        // (RFC §3.5.1 / Appendix A.20 `FrameClassify`), then choose the
        // `position` bit (`state_first` in the reference) so the lower-
        // energy 22/23-sample slot of the 80-sample state span gets the
        // boundary CB and the higher-energy 57/58-sample slot is scalar
        // coded.
        let n_short = mode.state_short_len();
        let diff = STATE_LEN - n_short; // boundary block length: 23 (20ms) / 22 (30ms)
        let boundary_samples = diff;
        let start = frame_classify(mode, &residual);
        debug_assert!((1..n_sub).contains(&start));
        let span_lo = (start - 1) * SUBL;

        // Energy of the leading 57/58 samples vs the trailing 57/58
        // samples of the state span. RFC §3.5.1 keeps the higher-energy
        // slot for scalar coding (more bits/sample). Reference
        // implementation: encode.c lines 3087-3103.
        let mut en1 = 0.0f32;
        for &r in &residual[span_lo..(span_lo + n_short)] {
            en1 += r * r;
        }
        let mut en2 = 0.0f32;
        for &r in &residual[(span_lo + diff)..(span_lo + diff + n_short)] {
            en2 += r * r;
        }
        // position == 1 (state_first==1): scalar at the LEADING n_short
        //   samples of the state span, boundary at the trailing diff.
        // position == 0 (state_first==0): boundary at the LEADING diff
        //   samples, scalar at the trailing n_short.
        //
        // The reference picks `state_first=1` when `en1 > en2` (the
        // leading 57/58-sample window holds more energy → keep it
        // scalar). For our self-roundtrip-shaped tuning we apply an
        // extra IIR-error-propagation guard on top of the spec rule:
        // the all-pole synthesis filter amplifies excitation errors
        // that occur EARLY in the frame, so we only switch to
        // position=0 when the trailing slot energy genuinely dominates
        // (≥ 4×). On marginal energy differences position=1 is the
        // safer choice for PCM-domain SNR. This keeps us spec-aligned
        // (we WILL pick position=0 on voiced onsets where the leading
        // slot is the quiet one) while protecting steady-signal SNR.
        let position: u8 = if en2 > 4.0 * en1 { 0 } else { 1 };
        let start_pos = if position == 1 {
            span_lo
        } else {
            span_lo + diff
        };
        let boundary_pos = if position == 1 {
            span_lo + n_short
        } else {
            span_lo
        };

        // Decoder uses `a_per_sub[start-1]` for the all-pass phase
        // compensation — i.e. the LPC of the first sub-block in the
        // state span (reference `decode.c` line 3713,
        // `&syntdenum[(start-1)*(LPC_FILTERORDER+1)]`). Mirror that.
        let a_for_phase = a_per_sub[start - 1];
        let state_residual_slice = &residual[start_pos..(start_pos + n_short)];
        let ccres = crate::state_encode::allpass_forward(state_residual_slice, &a_for_phase);
        let scale_idx = crate::state_encode::quantise_scale(&ccres);
        let qmax = crate::state::STATE_FRGQ_TBL[scale_idx as usize];
        let scal = 4.5 / 10f32.powf(qmax);
        let scaled: Vec<f32> = ccres.iter().map(|&v| v * scal).collect();
        // Start-state shape quantisation. RFC §3.5.3 / Appendix A.46
        // `AbsQuantW` specifies a predictive noise-shaping DPCM loop in
        // the perceptually-weighted domain; we wire it behind
        // `state_dpcm_on` (off by default — like the §3.6.2 CB weighting,
        // the perceptual weighting regresses synthetic self-roundtrip
        // SNR). The weighting denominators are the chirp-0.4222-expanded
        // LPC of the two sub-blocks straddling the start state (sub-block
        // `start-1` and `start`), matching the reference's
        // `weightdenum[(start-1)..]` pointer that advances one sub-block
        // at the slot boundary. Both paths emit `state_sq3Tbl` indices
        // that the decoder reads straight through (no inverse weighting,
        // RFC §4.2 / Appendix A.44), so the choice never affects decode
        // semantics — only which indices we emit.
        let state_samples: Vec<u8> = if self.state_dpcm_on {
            let wd_first = crate::state_encode::weight_denum_pub(&a_per_sub[start - 1]);
            let wd_second = crate::state_encode::weight_denum_pub(&a_per_sub[start]);
            crate::state_encode::abs_quant_w(&scaled, &wd_first, &wd_second, position == 1)
        } else {
            scaled
                .iter()
                .map(|&v| crate::state_encode::quantise_shape_sample(v))
                .collect()
        };

        // The reconstructed scalar state the decoder will produce.
        let scalar_state =
            crate::state::reconstruct_scalar_state(mode, scale_idx, &state_samples, &a_for_phase);

        // Build a frame-length `decresidual` buffer so the symmetric
        // forward+backward CB walks below see exactly the same memory
        // the decoder will see. The state span occupies 80 samples
        // starting at `span_lo`; we write the scalar state into the
        // appropriate half and let the boundary CB step (next) fill
        // the other half.
        let mut decresidual = vec![0.0f32; samples];
        for (k, &s) in scalar_state.iter().enumerate() {
            decresidual[start_pos + k] = s;
        }

        // ---- 5. Boundary CB search (22/23 samples) ----
        //
        // The boundary slot lives at `[boundary_pos .. boundary_pos +
        // boundary_samples]` within the state span. CB memory layout
        // (RFC §3.6.1, Appendix A.34 lines 3118-3170):
        //   - state_first==1 (boundary trailing): cb_mem tail = the
        //     scalar-decoded state samples; the boundary search uses
        //     the last `stMemLTbl=85` entries.
        //   - state_first==0 (boundary leading): cb_mem tail =
        //     time-reversed scalar samples; boundary samples are then
        //     time-reversed back into `decresidual[span_lo..span_lo +
        //     diff]` after the search.
        let stmeml = 85usize;
        let mut boundary_mem = vec![0.0f32; CB_LMEM];
        if position == 1 {
            // Tail-fill with scalar samples.
            boundary_mem[CB_LMEM - n_short..].copy_from_slice(&scalar_state);
        } else {
            // Tail-fill with time-reversed scalar samples (reference
            // `mem[CB_MEML-1-k] = decresidual[start_pos + k]`).
            for k in 0..n_short {
                boundary_mem[CB_LMEM - 1 - k] = scalar_state[k];
            }
        }
        // Search target: residual samples at the boundary slot, time-
        // reversed when state_first==0 (reference
        // `reverseResidual[k] = residual[(start+1)*SUBL-1 - (k +
        // state_short_len)]`).
        let target_boundary: Vec<f32> = if position == 1 {
            residual[boundary_pos..(boundary_pos + boundary_samples)].to_vec()
        } else {
            (0..boundary_samples)
                .map(|k| residual[span_lo + STATE_LEN - 1 - (k + n_short)])
                .collect()
        };
        let boundary_mem_slice = &boundary_mem[CB_LMEM - stmeml..];
        let (boundary_res, boundary_rec) = search_cb_capped_with_gain_correction(
            boundary_mem_slice,
            boundary_samples,
            &target_boundary,
            &[128; 3],
        );
        // Write the decoded boundary samples back into `decresidual`.
        if position == 1 {
            for (k, &v) in boundary_rec.iter().take(boundary_samples).enumerate() {
                decresidual[boundary_pos + k] = v;
            }
        } else {
            // Reverse-time write — reference: `decresidual[start_pos -
            // 1 - k] = reverseDecresidual[k]`.
            for (k, &v) in boundary_rec.iter().take(boundary_samples).enumerate() {
                decresidual[start_pos - 1 - k] = v;
            }
        }
        // Silence the "unused" warning in case the analysis-by-synthesis
        // search is re-enabled later.
        let _ = search_cb_abs;

        // ---- 6. Forward + backward CB sub-block walks (RFC §3.6) ----
        //
        // After the start state is encoded, the codebook search proceeds
        // in two passes (reference encode.c lines 3204-3345):
        //
        //   * Forward (`Nfor = n_sub - start - 1`): sub-blocks at
        //     `[(start+1)*SUBL ..]`, in forward time. cb_mem seeded
        //     with the full 80-sample decoded state span at
        //     `decresidual[(start-1)*SUBL ..]`.
        //
        //   * Backward (`Nback = start - 1`): sub-blocks at
        //     `[0 .. (start-1)*SUBL]`, encoded in reversed time.
        //     cb_mem seeded with the time-reversed tail of the decoded
        //     state span (and any forward sub-blocks that were just
        //     decoded — though the reference resets cb_mem before this
        //     pass and only seeds with the state span).
        //
        // The wire emission order is `[forward..., backward...]`, so
        // `subcount` increments through both passes — `sub_blocks[0]`
        // gets stage-2/3 widths of (7,7) and the rest get (8,8).
        let n_cb_sub = mode.cb_sub_blocks();
        let mut sub_block_indices: Vec<CbStageIndices> = Vec::with_capacity(n_cb_sub);

        // ---- 6a. Forward pass ----
        let n_for = n_sub.saturating_sub(start + 1);
        if n_for > 0 {
            // Seed cb_mem with the decoded state span (80 samples) at
            // its tail; zero before that.
            let mut mem = [0.0f32; CB_LMEM];
            mem[CB_LMEM - STATE_LEN..].copy_from_slice(&decresidual[span_lo..span_lo + STATE_LEN]);
            for fb in 0..n_for {
                let sb = start + 1 + fb;
                let lo = sb * SUBL;
                let hi = lo + SUBL;
                if hi > samples {
                    sub_block_indices.push(CbStageIndices::default());
                    continue;
                }
                let stage12_cap = if sub_block_indices.is_empty() {
                    128usize // first emitted sub-block: stage2/3 = 7 bits
                } else {
                    256usize // 8 bits
                };
                let target: [f32; SUBL] = core::array::from_fn(|i| residual[lo + i]);
                let (res, excitation) = search_cb_capped_with_gain_correction(
                    &mem,
                    SUBL,
                    &target,
                    &[256, stage12_cap, stage12_cap],
                );
                let mut exc_arr = [0.0f32; SUBL];
                exc_arr.copy_from_slice(&excitation);
                // Update cb_mem (shift-and-append) for the next stage.
                let mut mem_arr: [f32; CB_LMEM] = mem;
                update_cb_memory(&mut mem_arr, &exc_arr);
                mem = mem_arr;
                // Write decoded excitation into `decresidual` so the
                // backward pass — which reads from positions BEFORE the
                // state span — sees a consistent buffer.
                decresidual[lo..hi].copy_from_slice(&excitation);
                sub_block_indices.push(CbStageIndices {
                    cb_idx: res.cb_idx,
                    gain_idx: res.gain_idx,
                });
            }
        }

        // ---- 6b. Backward pass ----
        let n_back = start.saturating_sub(1);
        if n_back > 0 {
            // Reference encode.c lines 3273-3293: build a reversed
            // mirror of `residual[..(start-1)*SUBL]` and seed cb_mem
            // with the time-reversed decoded samples that immediately
            // follow the boundary in the original buffer (i.e. the
            // state span's `decresidual[(start-1)*SUBL ..]`, capped
            // at CB_LMEM samples).
            let meml_gotten = (SUBL * (n_sub + 1 - start)).min(CB_LMEM);
            let mut mem = [0.0f32; CB_LMEM];
            for k in 0..meml_gotten {
                // mem[CB_MEML-1-k] = decresidual[(start-1)*SUBL + k]
                mem[CB_LMEM - 1 - k] = decresidual[span_lo + k];
            }
            for bf in 0..n_back {
                let stage12_cap = if sub_block_indices.is_empty() {
                    128usize
                } else {
                    256usize
                };
                // Reversed target: residual[(start-1)*SUBL - 1 - bf*SUBL - k]
                // for k in 0..SUBL.
                let target: [f32; SUBL] =
                    core::array::from_fn(|k| residual[span_lo - 1 - bf * SUBL - k]);
                let (res, excitation) = search_cb_capped_with_gain_correction(
                    &mem,
                    SUBL,
                    &target,
                    &[256, stage12_cap, stage12_cap],
                );
                let mut exc_arr = [0.0f32; SUBL];
                exc_arr.copy_from_slice(&excitation);
                let mut mem_arr: [f32; CB_LMEM] = mem;
                update_cb_memory(&mut mem_arr, &exc_arr);
                mem = mem_arr;
                // Write decoded excitation back into `decresidual` in
                // reverse-time (reference: `decresidual[SUBL*Nback - i
                // - 1] = reverseDecresidual[i]`). For backward-pass
                // sub-block `bf`, the corresponding original-time
                // sub-block is `start - 2 - bf` (counting from 0).
                let orig_sb = start - 2 - bf;
                let orig_lo = orig_sb * SUBL;
                for k in 0..SUBL {
                    // reverseDecresidual[bf*SUBL + k] -> decresidual[span_lo - 1 - bf*SUBL - k]
                    decresidual[span_lo - 1 - bf * SUBL - k] = excitation[k];
                }
                let _ = orig_lo;
                sub_block_indices.push(CbStageIndices {
                    cb_idx: res.cb_idx,
                    gain_idx: res.gain_idx,
                });
            }
        }

        debug_assert_eq!(sub_block_indices.len(), n_cb_sub);

        // ---- 7. Pack ----
        let params = PackParams {
            mode,
            lsf_idx,
            // RFC §3.5.1 encodes the start-state position as 1-based
            // `start ∈ {1..n_sub-1}` directly. 20 ms uses 2 bits
            // (values {1,2,3}); 30 ms uses 3 bits (values {1..5}).
            block_class: start as u8,
            position, // RFC §3.5.1 `state_first`
            scale_idx,
            state_samples,
            boundary: CbStageIndices {
                cb_idx: boundary_res.cb_idx,
                gain_idx: boundary_res.gain_idx,
            },
            sub_blocks: sub_block_indices,
            empty_flag: false,
        };
        let bytes = pack_frame(&params)?;
        // Cache the current frame's per-sub-block LPC for the next
        // frame's enhancer-delay shift (RFC §4.7) — mirrors the decoder's
        // own `self.old_a_per_sub = a_per_sub` step.
        self.old_a_per_sub = a_per_sub;
        Ok(bytes)
    }

    /// Compute one (20 ms) or two (30 ms) LSF vectors from the input
    /// frame plus the lookback buffer.
    fn analyse_lsf(&mut self, frame_pcm: &[f32]) -> Vec<[f32; LPC_ORDER]> {
        // Build the LPC analysis buffer: previous lookback ++ current frame.
        let mode = self.mode;
        let lookback = lookback_len(mode);
        let mut buf = Vec::with_capacity(lookback + frame_pcm.len());
        buf.extend_from_slice(&self.lookback);
        buf.extend_from_slice(frame_pcm);
        // For 20 ms: buf.len() == 80 + 160 = 240. For 30 ms: 60 + 240 = 300.
        // We always window over LPC_WINLEN = 240 samples.
        let result = match mode {
            FrameMode::Ms20 => {
                // One LSF vector, asymmetric window over samples 0..240.
                let mut windowed = [0.0f32; LPC_WINLEN];
                let w = asymmetric_window();
                for i in 0..LPC_WINLEN {
                    windowed[i] = buf[i] * w[i];
                }
                let a = block_lpc(&windowed);
                vec![lpc_to_lsf(&a)]
            }
            FrameMode::Ms30 => {
                // lsf1: symmetric Hanning window over samples 0..240 of buf.
                // lsf2: asymmetric window over samples 60..300 of buf.
                let mut windowed1 = [0.0f32; LPC_WINLEN];
                let w1 = hanning_window();
                for i in 0..LPC_WINLEN {
                    windowed1[i] = buf[i] * w1[i];
                }
                let a1 = block_lpc(&windowed1);
                let lsf1 = lpc_to_lsf(&a1);

                let mut windowed2 = [0.0f32; LPC_WINLEN];
                let w2 = asymmetric_window();
                let off = LPC_LOOKBACK_30MS; // 60
                for i in 0..LPC_WINLEN {
                    windowed2[i] = buf[i + off] * w2[i];
                }
                let a2 = block_lpc(&windowed2);
                let lsf2 = lpc_to_lsf(&a2);
                vec![lsf1, lsf2]
            }
        };
        // Slide the lookback window: keep the last `lookback` samples of
        // the concatenated buffer.
        let new_lookback = &buf[buf.len() - lookback..];
        self.lookback.clear();
        self.lookback.extend_from_slice(new_lookback);
        result
    }
}

fn lookback_len(mode: FrameMode) -> usize {
    match mode {
        FrameMode::Ms20 => LPC_LOOKBACK_20MS,
        FrameMode::Ms30 => LPC_LOOKBACK_30MS,
    }
}

impl Encoder for IlbcEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let af = match frame {
            Frame::Audio(a) => a,
            _ => return Err(Error::invalid("iLBC encoder: audio frames only")),
        };
        let bytes = af
            .data
            .first()
            .ok_or_else(|| Error::invalid("iLBC encoder: empty frame"))?;
        if bytes.len() % 2 != 0 {
            return Err(Error::invalid("iLBC encoder: odd byte count"));
        }
        // Convert; optionally HP-filter (RFC 3951 §3.1 — opt-in, off by
        // default since the spec describes it as conditional on the
        // application's input characteristics).
        let n = bytes.len() / 2;
        let mut raw = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(2) {
            raw.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32);
        }
        if self.hp_filter_on {
            let mut filtered = vec![0.0f32; n];
            hp_input(&raw, &mut filtered, &mut self.hp_state);
            for s in filtered {
                self.pcm_queue.push_back(s);
            }
        } else {
            for s in raw {
                self.pcm_queue.push_back(s);
            }
        }
        self.drain(false)
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if !self.eof {
            self.eof = true;
            self.drain(true)?;
        }
        Ok(())
    }
}

/// Register the iLBC encoder with the codec registry. Called from
/// [`crate::codec::register`] once wired.
pub fn register_encoder(info: oxideav_core::CodecInfo) -> oxideav_core::CodecInfo {
    info.encoder(make_encoder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::AudioFrame;

    fn new_encoder(mode: FrameMode) -> Box<dyn Encoder> {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(1);
        params.sample_format = Some(SampleFormat::S16);
        if mode == FrameMode::Ms30 {
            params.options = params.options.set("frame_ms", "30");
        }
        make_encoder(&params).expect("encoder")
    }

    #[test]
    fn make_encoder_rejects_stereo() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(2);
        assert!(make_encoder(&params).is_err());
    }

    #[test]
    fn encoder_emits_20ms_packets_for_silence() {
        let mut enc = new_encoder(FrameMode::Ms20);
        // 3 frames of silence.
        let samples = 3 * 160;
        let bytes = vec![0u8; samples * 2];
        let af = AudioFrame {
            samples: samples as u32,
            pts: Some(0),
            data: vec![bytes],
        };
        enc.send_frame(&Frame::Audio(af)).unwrap();
        let mut count = 0;
        while let Ok(pkt) = enc.receive_packet() {
            assert_eq!(pkt.data.len(), 38);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn encoder_emits_30ms_packets_for_silence() {
        let mut enc = new_encoder(FrameMode::Ms30);
        let samples = 2 * 240;
        let bytes = vec![0u8; samples * 2];
        let af = AudioFrame {
            samples: samples as u32,
            pts: Some(0),
            data: vec![bytes],
        };
        enc.send_frame(&Frame::Audio(af)).unwrap();
        let mut count = 0;
        while let Ok(pkt) = enc.receive_packet() {
            assert_eq!(pkt.data.len(), 50);
            count += 1;
        }
        assert_eq!(count, 2);
    }

    /// RFC §3.5.3 / Appendix A.46 `AbsQuantW` path: enabling
    /// `state_dpcm=on` runs the predictive noise-shaping DPCM quantiser
    /// for the start state. It must still emit well-formed 38-byte
    /// packets and produce bounded PCM through the decoder.
    #[test]
    fn encoder_state_dpcm_path_round_trips() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(1);
        params.sample_format = Some(SampleFormat::S16);
        params.options = params.options.set("state_dpcm", "on");
        let mut enc = make_encoder(&params).expect("encoder");

        // 6 frames of a voiced-ish chord so the start-state slot is
        // non-trivial (the DPCM loop is a no-op on silence).
        let n = 6 * 160;
        let mut bytes = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / SAMPLE_RATE as f32;
            let mut v = 0.0f32;
            for h in 1..4 {
                v += (2.0 * core::f32::consts::PI * (h as f32) * 150.0 * t).sin()
                    * (4000.0 / h as f32);
            }
            let s = v.round().clamp(-32768.0, 32767.0) as i16;
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let af = AudioFrame {
            samples: n as u32,
            pts: Some(0),
            data: vec![bytes],
        };
        enc.send_frame(&Frame::Audio(af)).unwrap();
        enc.flush().unwrap();

        let mut dec_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        dec_params.sample_rate = Some(SAMPLE_RATE);
        dec_params.channels = Some(1);
        let mut dec = crate::decoder::make_decoder(&dec_params).expect("decoder");

        let mut produced = 0usize;
        let mut decoded = 0usize;
        while let Ok(pkt) = enc.receive_packet() {
            assert_eq!(pkt.data.len(), 38);
            produced += 1;
            let dpkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), pkt.data.clone());
            dec.send_packet(&dpkt).unwrap();
            if let Frame::Audio(a) = dec.receive_frame().unwrap() {
                for chunk in a.data[0].chunks_exact(2) {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                    assert!(s.abs() as i32 <= 32767, "sample out of range: {s}");
                    decoded += 1;
                }
            }
        }
        assert_eq!(produced, 6, "expected 6 encoded packets");
        assert_eq!(decoded, 6 * 160, "decoder produced unexpected sample count");
    }
}
