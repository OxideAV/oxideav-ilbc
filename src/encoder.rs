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
//!     ↓ pick state span / position                       (§3.5.1)
//!     ↓ all-pass + log-magnitude + shape 3-bit scalar VQ (§3.5.2-3)
//!   scale_idx + state_samples[57/58]
//!     ↓ rebuild state_vec from the *decoded* samples so the CB memory
//!       evolves identically to the decoder.
//!     ↓ for each CB sub-block (boundary 22/23 + 40-sample sub-blocks)
//!       run a 3-stage CB search                          (§3.6)
//!   CB indices + gain indices
//!     ↓ pack Table 3.2                                   (§3.7/3.8)
//!   38/50-byte iLBC payload.
//! ```
//!
//! The decoder currently pins `start_idx = 0` (the state vector drives
//! sub-blocks 0 and 1 unconditionally and CB sub-blocks start at
//! sub-block 2). We emit `block_class = 1` to match, independent of
//! where the speech energy actually peaks. The CB targets for
//! sub-blocks 2..n_sub are the residual samples `[80, frame_len)`.

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

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(SampleFormat::S16);
    output.channels = Some(1);
    output.sample_rate = Some(SAMPLE_RATE);
    output.bit_rate = Some(match mode {
        FrameMode::Ms20 => 15_200,
        FrameMode::Ms30 => 13_333,
    });

    Ok(Box::new(IlbcEncoder::new(mode, output, hp_filter_on)))
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
    /// 147-sample adaptive codebook memory, kept in lockstep with the
    /// decoder's own `cb_mem`.
    cb_mem: [f32; CB_LMEM],
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
    pending: VecDeque<Packet>,
    sample_pos: i64,
    eof: bool,
}

impl IlbcEncoder {
    fn new(mode: FrameMode, output_params: CodecParameters, hp_filter_on: bool) -> Self {
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
            cb_mem: [0.0; CB_LMEM],
            old_a_per_sub: vec![identity; 6],
            hp_state: HpInputState::default(),
            hp_filter_on,
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

        // ---- 4. Start-state encoding. We pin the state span to
        //         sub-blocks 0 and 1 (block_class = 1) but pick the
        //         `position` bit per RFC §3.5.1 so the lower-energy
        //         half of the state span is the one that gets boundary-
        //         CB-coded (22/23 samples / 21 bits) instead of the
        //         scalar-coded one (57/58 × 3 bits + 6-bit scale).
        //
        // The decoder uses `a_per_sub[0]` (the current frame's first
        // sub-block LPC) for the state's all-pass phase compensation
        // (decoder.rs:109). We mirror that exactly so encoder and
        // decoder stay in lockstep on the scalar-state reconstruction.
        let a_for_phase = a_per_sub[0];
        let n_short = mode.state_short_len();
        let boundary_samples = match mode {
            FrameMode::Ms20 => 23usize,
            FrameMode::Ms30 => 22usize,
        };
        // Energy of the leading boundary slot vs the trailing boundary
        // slot. The slot we keep for scalar coding is the one with the
        // higher energy (more bits per sample preserves it best); the
        // opposite slot is what the boundary CB sees as its target.
        let mut e_leading_boundary = 0.0f32;
        for &r in &residual[0..boundary_samples] {
            e_leading_boundary += r * r;
        }
        let mut e_trailing_boundary = 0.0f32;
        for &r in &residual[(STATE_LEN - boundary_samples)..STATE_LEN] {
            e_trailing_boundary += r * r;
        }
        // position == 1: scalar at [0..n_short], boundary at [n_short..STATE_LEN].
        //                Drop the trailing slot (more energy → bigger error
        //                if CB-coded) means we KEEP it scalar — i.e. pick
        //                position=1 when the leading boundary slot has
        //                LESS energy than the trailing one.
        // position == 0: boundary at [0..boundary_samples], scalar at [boundary_samples..STATE_LEN].
        // RFC §3.5.1 picks position based on which boundary slot has
        // less energy ("drop the quieter one into the CB block"). In
        // practice the all-pole synthesis filter at the decoder side
        // amplifies excitation errors that occur EARLY in the frame
        // (samples ripple through the LPC feedback for the rest of the
        // frame), so dropping CB content into the leading slot
        // (position = 0) costs measurable PCM-domain SNR even when the
        // residual energies say it should help. We require a
        // significant energy ratio (≥ 4×) before switching to position
        // = 0 — small/marginal energy differences are not worth the
        // IIR error-propagation penalty. This keeps us spec-compliant
        // (we WILL pick position=0 when the leading slot is genuinely
        // the quiet one, e.g. voiced onsets) while protecting steady-
        // signal SNR.
        let position: u8 = if e_trailing_boundary > 4.0 * e_leading_boundary {
            // Trailing slot dominates the energy — drop the leading
            // (quieter) slot into the boundary CB.
            0
        } else {
            1
        };
        let scalar_offset = if position == 1 { 0 } else { boundary_samples };
        let boundary_offset = if position == 1 { n_short } else { 0 };

        let state_residual_slice = &residual[scalar_offset..(scalar_offset + n_short)];
        let ccres = crate::state_encode::allpass_forward(state_residual_slice, &a_for_phase);
        let scale_idx = crate::state_encode::quantise_scale(&ccres);
        let qmax = crate::state::STATE_FRGQ_TBL[scale_idx as usize];
        let scal = 4.5 / 10f32.powf(qmax);
        let state_samples: Vec<u8> = ccres
            .iter()
            .map(|&v| crate::state_encode::quantise_shape_sample(v * scal))
            .collect();

        // The reconstructed scalar state the decoder will produce.
        let scalar_state =
            crate::state::reconstruct_scalar_state(mode, scale_idx, &state_samples, &a_for_phase);
        // Build the 80-sample state_vec exactly as the decoder will:
        // scalar_state in its position-selected slot, boundary slot
        // zero (the boundary CB pass below fills it in).
        let mut state_vec = [0.0f32; STATE_LEN];
        for (k, &s) in scalar_state.iter().enumerate() {
            let dst = scalar_offset + k;
            if dst < STATE_LEN {
                state_vec[dst] = s;
            }
        }

        // Reset the CB memory the way the decoder does at the start of
        // every frame: zero-pad before CB_LMEM-STATE_LEN, then the
        // (partially populated) state vector goes at the tail.
        let pad = CB_LMEM - STATE_LEN;
        self.cb_mem[..pad].fill(0.0);
        self.cb_mem[pad..].copy_from_slice(&state_vec);

        // ---- 5. Boundary CB search (22/23 samples) ----
        // Target = residual at the boundary slot.
        let target_boundary: Vec<f32> =
            residual[boundary_offset..(boundary_offset + boundary_samples)].to_vec();
        // Per RFC 3951 §3.6.1 the boundary block uses lMem = 85 samples
        // (not the full 147), so the search runs against the last 85
        // entries of the CB memory; per Table 3.1 each stage has 128
        // entries (64 base + 64 expanded), so the cap is 128 throughout.
        let boundary_mem = &self.cb_mem[CB_LMEM - 85..];
        let (boundary_res, boundary_rec) = search_cb_capped_with_gain_correction(
            boundary_mem,
            boundary_samples,
            &target_boundary,
            &[128; 3],
        );
        // Update cb_mem / state_vec exactly as the decoder will. The
        // decoder adds `boundary_exc[i]` to `excitation[boundary_offset
        // + i]` and also pushes `boundary_exc` as a full SUBL-sample
        // block into the CB memory (padded with zeros).
        let mut boundary_block = [0.0f32; SUBL];
        let copy_n = boundary_samples.min(SUBL);
        boundary_block[..copy_n].copy_from_slice(&boundary_rec[..copy_n]);
        update_cb_memory(&mut self.cb_mem, &boundary_block);

        // ---- 6. Remaining 40-sample sub-blocks: residual-domain CB search ----
        // RFC 3951 §3.6 (and the reference `iCBSearch` in Appendix A.34) does
        // the codebook search in the residual domain — target = LPC residual,
        // codebook memory = previously-decoded residual. Our `search_cb`
        // reproduces that flow; the perceptual weighting filter is omitted
        // (it is OPTIONAL per RFC §3.4) which makes the search identical to
        // the reference's behaviour with the weighting filter set to identity.
        let n_cb_sub = mode.cb_sub_blocks();
        let mut sub_block_indices = Vec::with_capacity(n_cb_sub);
        for cb_i in 0..n_cb_sub {
            let sb = 2 + cb_i;
            let lo = sb * SUBL;
            let hi = lo + SUBL;
            if hi > samples {
                sub_block_indices.push(CbStageIndices::default());
                continue;
            }
            // Per Table 3.2, the FIRST 40-sample sub-block after the state
            // (`cb_i == 0`) has codebook size 128 for stages 1 and 2 (8/7/7
            // bits); subsequent sub-blocks have 256 for all stages. We cap
            // the search range here so the encoder never picks an index it
            // cannot encode.
            let stage12_cap = if cb_i == 0 { 128usize } else { 256usize };
            let target: [f32; SUBL] = core::array::from_fn(|i| residual[lo + i]);
            let (res, excitation) = search_cb_capped_with_gain_correction(
                &self.cb_mem,
                SUBL,
                &target,
                &[256, stage12_cap, stage12_cap],
            );
            let mut exc_arr = [0.0f32; SUBL];
            exc_arr.copy_from_slice(&excitation);
            update_cb_memory(&mut self.cb_mem, &exc_arr);
            sub_block_indices.push(CbStageIndices {
                cb_idx: res.cb_idx,
                gain_idx: res.gain_idx,
            });
        }
        // Silence the "unused" warning in case the analysis-by-synthesis
        // search is re-enabled later.
        let _ = search_cb_abs;

        // ---- 7. Pack ----
        let params = PackParams {
            mode,
            lsf_idx,
            block_class: 1, // start_idx = 0 ⇒ block_class index = 1 (1-based)
            position,       // RFC §3.5.1 selection — see step 4 above
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
}
