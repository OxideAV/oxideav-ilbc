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
use crate::cb_search::{search_cb, search_cb_abs};
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

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.sample_format = Some(SampleFormat::S16);
    output.channels = Some(1);
    output.sample_rate = Some(SAMPLE_RATE);
    output.bit_rate = Some(match mode {
        FrameMode::Ms20 => 15_200,
        FrameMode::Ms30 => 13_333,
    });

    Ok(Box::new(IlbcEncoder::new(mode, output)))
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
    pending: VecDeque<Packet>,
    sample_pos: i64,
    eof: bool,
}

impl IlbcEncoder {
    fn new(mode: FrameMode, output_params: CodecParameters) -> Self {
        let lookback = vec![0.0f32; lookback_len(mode)];
        Self {
            output_params,
            time_base: TimeBase::new(1, SAMPLE_RATE as i64),
            mode,
            pcm_queue: VecDeque::new(),
            lookback,
            lsf_state: LsfState::new(),
            lpc_mem: [0.0; LPC_ORDER],
            cb_mem: [0.0; CB_LMEM],
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

        // ---- 3. Residual via per-sub-block LPC analysis filter ----
        //
        // Open-loop analysis using the encoder's input PCM history. This
        // matches the RFC reference. Closed-loop alternatives (feeding
        // the decoder's synth memory into the analysis filter) were
        // tried and gave worse SNR in this project's simplified
        // pipeline — they only help when the decoder-side synth memory
        // truly equals the target output, which isn't the case with
        // 3-bit scalar state quantisation and a 3-stage CB.
        let mut residual = vec![0.0f32; samples];
        for (sb, a) in a_per_sub.iter().enumerate().take(mode.sub_blocks()) {
            let lo = sb * SUBL;
            let hi = lo + SUBL;
            let mut out = vec![0.0f32; SUBL];
            lpc_analysis_filter(&frame_pcm[lo..hi], a, &mut self.lpc_mem, &mut out);
            residual[lo..hi].copy_from_slice(&out);
        }

        // ---- 4. Start-state encoding. We pin the state to sub-blocks
        //         0 and 1 to match the decoder's wiring.
        // We still call `encode_state` for the scale + shape quantisation,
        // but override its selected start_idx to 0 and always pick
        // position = 1 (keep the first STATE_SHORT_LEN samples of the
        // 80-sample span).
        let a_for_phase = a_per_sub[0];
        let state_residual_slice = &residual[0..mode.state_short_len()];
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
        // The decoder seeds the 80-sample state vector with the scalar
        // state at indices 0..STATE_SHORT_LEN, leaving the boundary span
        // (STATE_SHORT_LEN..80) to be filled by the boundary CB search.
        let mut state_vec = [0.0f32; STATE_LEN];
        let copy_len = scalar_state.len().min(STATE_LEN);
        state_vec[..copy_len].copy_from_slice(&scalar_state[..copy_len]);

        // Reset the CB memory the way the decoder does at the start of
        // every frame: zero-pad before STATE_LEN-CB_LMEM, then the state
        // vector goes at the tail.
        let pad = CB_LMEM - STATE_LEN;
        self.cb_mem[..pad].fill(0.0);
        self.cb_mem[pad..].copy_from_slice(&state_vec);

        // ---- 5. Boundary CB search (22/23 samples) ----
        let boundary_samples = match mode {
            FrameMode::Ms20 => 23usize,
            FrameMode::Ms30 => 22usize,
        };
        // Target: the residual positions [STATE_SHORT_LEN..STATE_LEN).
        let target_boundary: Vec<f32> = residual
            [mode.state_short_len()..STATE_LEN.min(mode.state_short_len() + boundary_samples)]
            .to_vec();
        // Pad to `boundary_samples` in case STATE_LEN < state_short_len+boundary
        // (shouldn't happen — boundary is always 2*SUBL - state_short_len).
        let mut target_boundary = target_boundary;
        while target_boundary.len() < boundary_samples {
            target_boundary.push(0.0);
        }
        let (boundary_res, boundary_rec) =
            search_cb(&self.cb_mem, boundary_samples, &target_boundary);
        // Update cb_mem / state_vec exactly as the decoder will. The
        // decoder adds `boundary_exc[i]` to `excitation[STATE_LEN -
        // boundary_samples + i]` and also pushes `boundary_exc` as a
        // full SUBL-sample block into the CB memory (padded with zeros).
        let mut boundary_block = [0.0f32; SUBL];
        let copy_n = boundary_samples.min(SUBL);
        boundary_block[..copy_n].copy_from_slice(&boundary_rec[..copy_n]);
        update_cb_memory(&mut self.cb_mem, &boundary_block);

        // ---- 6. Remaining 40-sample sub-blocks: analysis-by-synthesis ----
        // For each 40-sample sub-block the encoder searches for the
        // excitation whose zero-state response of `1/A(z)` best matches
        // the *PCM* target (input - ZIR). This is strictly better than
        // residual-domain matching when the decoder's filter memory
        // drifts from the encoder's input history.
        let n_cb_sub = mode.cb_sub_blocks();
        let mut sub_block_indices = Vec::with_capacity(n_cb_sub);
        // Running synth memory for the analysis-by-synthesis loop. It
        // starts at the state-sub-block boundary: the decoder's synth
        // mem at sub-block 2's start is whatever 1/A(z) produces from
        // excitation[0..80] = state_vec + boundary. We compute that here.
        let mut synth_mem;
        {
            // Run 1/A(z) from zero mem through state_vec to get synth_mem.
            // This approximates the decoder's synth memory at sub-block 2's
            // start. Using a_per_sub[0] for sub-blocks 0-1 (two halves).
            let mut tmp_mem = [0.0f32; LPC_ORDER];
            for (sb, a) in a_per_sub.iter().enumerate().take(2.min(mode.sub_blocks())) {
                let sb_start = sb * SUBL;
                let mut exc = [0.0f32; SUBL];
                // Build the excitation the decoder uses for this state
                // sub-block.
                for (i, e) in exc.iter_mut().enumerate() {
                    if sb_start + i < STATE_LEN {
                        *e = state_vec[sb_start + i];
                    }
                    if sb_start + i >= STATE_LEN - boundary_samples && sb_start + i < STATE_LEN {
                        let b_off = sb_start + i - (STATE_LEN - boundary_samples);
                        if b_off < boundary_samples.min(SUBL) {
                            *e += boundary_rec[b_off];
                        }
                    }
                }
                // RFC 3951 §4.7 eq.: y(n) = x(n) - Σ a[k]·y(n-k), 1≤k≤LPC_ORDER.
                #[allow(clippy::needless_range_loop)] // RFC 3951 §4.7 LPC synthesis
                for n in 0..SUBL {
                    let mut s = exc[n];
                    for k in 1..=LPC_ORDER {
                        s -= a[k] * tmp_mem[k - 1];
                    }
                    for k in (1..LPC_ORDER).rev() {
                        tmp_mem[k] = tmp_mem[k - 1];
                    }
                    tmp_mem[0] = s;
                }
            }
            synth_mem = tmp_mem;
        }
        for cb_i in 0..n_cb_sub {
            let sb = 2 + cb_i;
            let lo = sb * SUBL;
            let hi = lo + SUBL;
            if hi > samples {
                sub_block_indices.push(CbStageIndices::default());
                continue;
            }
            let a = &a_per_sub[sb];
            // Compute zero-input response (ZIR) of 1/A(z) for this
            // sub-block given `synth_mem`. RFC 3951 §4.7.
            let zir = {
                let mut mem = synth_mem;
                let mut out = [0.0f32; SUBL];
                for out_n in out.iter_mut() {
                    let mut s = 0.0f32;
                    for k in 1..=LPC_ORDER {
                        s -= a[k] * mem[k - 1];
                    }
                    *out_n = s;
                    for k in (1..LPC_ORDER).rev() {
                        mem[k] = mem[k - 1];
                    }
                    mem[0] = s;
                }
                out
            };
            // PCM target = input - ZIR.
            let pcm_target: [f32; SUBL] = core::array::from_fn(|i| frame_pcm[lo + i] - zir[i]);
            let (res, excitation) = search_cb_abs(&self.cb_mem, a, &pcm_target);
            // Advance synth_mem using the PCM output = ZIR + ZSR(excitation).
            // We compute the actual output sample-by-sample. RFC 3951 §4.7.
            let mut mem = synth_mem;
            for &exc_n in excitation.iter() {
                let mut s = exc_n;
                for k in 1..=LPC_ORDER {
                    s -= a[k] * mem[k - 1];
                }
                for k in (1..LPC_ORDER).rev() {
                    mem[k] = mem[k - 1];
                }
                mem[0] = s;
            }
            synth_mem = mem;
            update_cb_memory(&mut self.cb_mem, &excitation);
            sub_block_indices.push(CbStageIndices {
                cb_idx: res.cb_idx,
                gain_idx: res.gain_idx,
            });
        }
        // Silence the "unused" warning in case the simpler search is
        // re-enabled later.
        let _ = search_cb;

        // ---- 7. Pack ----
        let params = PackParams {
            mode,
            lsf_idx,
            block_class: 1, // start_idx = 0 ⇒ block_class index = 1 (1-based)
            position: 1,    // decoder pins state to sub-blocks 0-1
            scale_idx,
            state_samples,
            boundary: CbStageIndices {
                cb_idx: boundary_res.cb_idx,
                gain_idx: boundary_res.gain_idx,
            },
            sub_blocks: sub_block_indices,
            empty_flag: false,
        };
        pack_frame(&params)
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
        if af.channels != 1 || af.sample_rate != SAMPLE_RATE {
            return Err(Error::invalid("iLBC encoder: input must be mono, 8000 Hz"));
        }
        if af.format != SampleFormat::S16 {
            return Err(Error::invalid(
                "iLBC encoder: input sample format must be S16",
            ));
        }
        let bytes = af
            .data
            .first()
            .ok_or_else(|| Error::invalid("iLBC encoder: empty frame"))?;
        if bytes.len() % 2 != 0 {
            return Err(Error::invalid("iLBC encoder: odd byte count"));
        }
        for chunk in bytes.chunks_exact(2) {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f32;
            self.pcm_queue.push_back(s);
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
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: SAMPLE_RATE,
            samples: samples as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, SAMPLE_RATE as i64),
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
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: SAMPLE_RATE,
            samples: samples as u32,
            pts: Some(0),
            time_base: TimeBase::new(1, SAMPLE_RATE as i64),
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
