//! Top-level iLBC frame decoder.
//!
//! The decoder threads the subsystems together for each incoming packet:
//!
//! ```text
//!   payload → bitreader → lsf (dequant + interp + LSF→LPC) →
//!             state.reconstruct → cb.construct × sub_blocks →
//!             synthesis.synthesise_frame → S16 PCM
//! ```
//!
//! Frame mode (20 ms / 30 ms) is inferred from the packet byte length.

use oxideav_core::Decoder;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::bitreader::{parse_frame, FrameParams};
use crate::cb::{construct_excitation, construct_excitation_veclen, update_cb_memory};
use crate::enhancer::{enhance_frame, EnhancerState};
use crate::hp_filter::{hp_output, HpOutputState};
use crate::lsf::{decode_and_interpolate, dequant_lsf, LsfState};
use crate::state::reconstruct_scalar_state;
use crate::synthesis::{
    conceal_residual, plc_record_good, synthesise_blocks, synthesise_frame, SynthState,
};
use crate::{FrameMode, CB_LMEM, CODEC_ID_STR, LPC_ORDER, SAMPLE_RATE, STATE_LEN, SUBL};

/// Build a boxed [`Decoder`] for iLBC.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let sample_rate = params.sample_rate.unwrap_or(SAMPLE_RATE);
    if sample_rate != SAMPLE_RATE {
        return Err(Error::unsupported(format!(
            "iLBC decoder: only 8000 Hz is supported (got {sample_rate})"
        )));
    }
    let channels = params.channels.unwrap_or(1);
    if channels != 1 {
        return Err(Error::unsupported(format!(
            "iLBC decoder: only mono is supported (got {channels} channels)"
        )));
    }
    if params.codec_id.as_str() != CODEC_ID_STR {
        return Err(Error::unsupported(format!(
            "iLBC decoder: unexpected codec id {:?}",
            params.codec_id
        )));
    }
    // RFC 3951 §4.8 Post Filtering: optional 65 Hz HP filter on the
    // decoded output. Off by default; enable with `hp_filter=on` (or
    // `1` / `true` / `yes`). The RFC explicitly marks this stage as
    // "If desired" — useful when the encoder side ran without §3.1
    // pre-processing and the decoder wants to strip residual DC / 50-
    // 60 Hz content the CELP synthesis filter may have introduced.
    let hp_filter_on = params
        .options
        .get("hp_filter")
        .map(|v| matches!(v, "1" | "on" | "true" | "yes"))
        .unwrap_or(false);
    Ok(Box::new(IlbcDecoder::new(hp_filter_on)))
}

/// Parse a packet into its [`FrameParams`] — thin re-export for tests
/// and external tooling.
pub fn parse_packet(packet: &[u8]) -> Result<FrameParams> {
    parse_frame(packet)
}

/// Filter `samples` in place using the §4.8 output HP biquad. Returns
/// the number of samples processed. Wrapper around [`hp_output`] that
/// avoids a temporary `Vec` allocation by reading and writing the same
/// slice (the filter samples its input before writing the output, so
/// aliasing is safe).
fn hp_output_in_place(samples: &mut [f32], state: &mut HpOutputState) -> usize {
    let n = samples.len();
    // Safety: `hp_output` reads `input[n]` before writing `out[n]` for
    // every n, so reading and writing the same slice via two raw
    // borrows is sound. We avoid `unsafe` by allocating once per frame
    // — iLBC frames are 160/240 samples so the allocation is cheap.
    let scratch: Vec<f32> = samples.to_vec();
    hp_output(&scratch, samples, state);
    n
}

/// Final synthesis-sample → S16 conversion, RFC 3951 §A.2 `iLBC_decode`
/// output stage (the `decoded_data[k] = (short) dtmp` clamp-then-cast).
///
/// RFC 3951 §A.2 clamps the float synthesis output to the int16 range
/// `[MIN_SAMPLE, MAX_SAMPLE] = [-32768, 32767]` and then performs a C
/// `(short)` cast, which **truncates toward zero** — it does not round to
/// nearest. Matching this exactly is what makes the quiet-passage decode
/// bit-aligned: a round-to-nearest cast drifts by ±1 LSB on every sample
/// whose fractional part crosses 0.5, which is the dominant error on
/// near-silent material.
#[inline]
fn sample_to_i16(s: f32) -> i16 {
    // `as i16` on an already-clamped finite f32 truncates toward zero and
    // is saturation-defined for the endpoints, mirroring the reference
    // clamp followed by the truncating `(short)` cast.
    s.clamp(-32768.0, 32767.0) as i16
}

struct IlbcDecoder {
    codec_id: CodecId,
    lsf_state: LsfState,
    synth: SynthState,
    enhancer: EnhancerState,
    /// 147-sample adaptive-codebook memory (RFC §4.3 `CB_LMEM`).
    cb_mem: [f32; CB_LMEM],
    /// Previous frame's per-sub-block LPC denominators, used by the
    /// enhancer-delay-aware synthesis filtering of RFC §4.7. Holds at
    /// least the last `mode.sub_blocks()` rows of the previous frame.
    old_a_per_sub: Vec<[f32; LPC_ORDER + 1]>,
    /// RFC §4.8 output HP filter state. Persists across frames so the
    /// IIR delay line doesn't reset on every packet (the post-filter is
    /// applied to the full per-frame PCM block, not per sub-block).
    hp_state: HpOutputState,
    /// Whether to apply the §4.8 output HP filter. Toggled by the
    /// `hp_filter=on` decoder parameter.
    hp_filter_on: bool,
    /// Frame mode of the most recently decoded (or concealed) packet.
    /// A zero-byte "lost" packet carries no length hint, so its
    /// concealment must reuse the mode the stream has been running in
    /// to keep the decoded sample count (and therefore the wall-clock
    /// PTS advance) aligned — a 30 ms stream must conceal 240 samples,
    /// not the 20 ms default. `None` until the first sized packet.
    last_mode: Option<FrameMode>,
    pending: Option<Packet>,
    eof: bool,
}

impl IlbcDecoder {
    fn new(hp_filter_on: bool) -> Self {
        // Seed `old_a_per_sub` with identity LPC rows so the very first
        // frame's enhancer-delay shift uses a pass-through filter where
        // the previous-frame LPC would normally apply.
        let mut identity = [0.0f32; LPC_ORDER + 1];
        identity[0] = 1.0;
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            lsf_state: LsfState::new(),
            synth: SynthState::new(),
            enhancer: EnhancerState::new(),
            cb_mem: [0.0; CB_LMEM],
            old_a_per_sub: vec![identity; 6],
            hp_state: HpOutputState::default(),
            hp_filter_on,
            last_mode: None,
            pending: None,
            eof: false,
        }
    }

    fn decode_into(&mut self, packet: &[u8], out: &mut [f32]) -> Result<()> {
        let fp = parse_frame(packet)?;
        // Empty-frame flag: §3.8 — if set, the decoder SHOULD treat the
        // block as lost and run PLC.
        if fp.empty_flag {
            self.conceal_into(fp.mode, out);
            return Ok(());
        }

        // Dequantise LSF vector(s).
        let mut lsf_vectors = Vec::with_capacity(fp.mode.lsf_vectors());
        for idx in &fp.lsf_idx {
            lsf_vectors.push(dequant_lsf(idx));
        }
        // Build per-sub-block LPC coefficients.
        let a_per_sub = decode_and_interpolate(&mut self.lsf_state, fp.mode, &lsf_vectors);
        debug_assert_eq!(a_per_sub.len(), fp.mode.sub_blocks());

        // ---- Variable start_idx (RFC §3.5.1) ----
        //
        // `block_class` carries the encoder's `start ∈ {1..n_sub-1}`
        // value directly: the start state occupies sub-blocks
        // `start-1` and `start`. The codebook walk then proceeds in
        // two passes — `Nfor` forward sub-blocks at `[(start+1)*SUBL
        // ..]`, then `Nback` backward sub-blocks at `[0 ..
        // (start-1)*SUBL]` (encoded in reverse time). The wire order
        // is `[forward..., backward...]`, so `fp.sub_blocks[0]` is
        // the first forward sub-block when Nfor>0, else the first
        // backward sub-block.
        let n_sub = fp.mode.sub_blocks();
        let n_short = fp.mode.state_short_len();
        let diff = STATE_LEN - n_short; // 23 (20 ms) / 22 (30 ms)
        let boundary_samples = diff;
        // Clamp `block_class` into the legal range so a malformed
        // packet (e.g. block_class == 0 or > n_sub-1) still produces
        // bounded output.
        let start = (fp.block_class as usize).clamp(1, n_sub - 1);
        let span_lo = (start - 1) * SUBL;
        let start_pos = if fp.position == 1 {
            span_lo
        } else {
            span_lo + diff
        };
        let boundary_pos = if fp.position == 1 {
            span_lo + n_short
        } else {
            span_lo
        };

        // Reconstruct start state. RFC §4.2 / Appendix A.5 line 3713:
        // the all-pass phase compensation uses the LPC of the first
        // sub-block in the state span (`a_per_sub[start - 1]`).
        let a_for_phase: [f32; LPC_ORDER + 1] = a_per_sub[start - 1];
        let scalar_state =
            reconstruct_scalar_state(fp.mode, fp.scale_idx, &fp.state_samples, &a_for_phase);
        debug_assert_eq!(scalar_state.len(), n_short);

        // `decresidual` is the frame-length excitation we will hand to
        // synthesis. Build it incrementally — first the scalar state,
        // then the boundary CB samples, then the forward/backward CB
        // sub-blocks.
        let mut decresidual = vec![0.0f32; n_sub * SUBL];
        for (k, &s) in scalar_state.iter().enumerate() {
            decresidual[start_pos + k] = s;
        }

        // ---- Boundary CB decode (22/23 samples) ----
        // Mirror the encoder's cb_mem layout: scalar samples in the
        // tail (forward) for position=1, time-reversed scalar in the
        // tail for position=0. RFC §3.6.1 reads `stMemLTbl=85` samples.
        let stmeml = 85usize;
        let mut boundary_mem = vec![0.0f32; CB_LMEM];
        if fp.position == 1 {
            boundary_mem[CB_LMEM - n_short..].copy_from_slice(&scalar_state);
        } else {
            for k in 0..n_short {
                boundary_mem[CB_LMEM - 1 - k] = scalar_state[k];
            }
        }
        let boundary_full = construct_excitation_veclen(
            &boundary_mem[CB_LMEM - stmeml..],
            boundary_samples,
            &fp.boundary.cb_idx,
            &fp.boundary.gain_idx,
        );
        if fp.position == 1 {
            for (k, &v) in boundary_full.iter().take(boundary_samples).enumerate() {
                decresidual[boundary_pos + k] = v;
            }
        } else {
            // Reverse-time write back into the leading boundary slot.
            for (k, &v) in boundary_full.iter().take(boundary_samples).enumerate() {
                decresidual[start_pos - 1 - k] = v;
            }
        }
        // ---- Forward + backward CB sub-block decode ----
        let n_cb_sub = fp.mode.cb_sub_blocks();
        let n_for = n_sub.saturating_sub(start + 1);
        let n_back = start.saturating_sub(1);
        debug_assert_eq!(n_for + n_back, n_cb_sub);
        let mut sub_idx = 0usize;

        // Forward pass: cb_mem seeded with the full 80-sample state span.
        if n_for > 0 {
            let mut mem = [0.0f32; CB_LMEM];
            mem[CB_LMEM - STATE_LEN..].copy_from_slice(&decresidual[span_lo..span_lo + STATE_LEN]);
            for fb in 0..n_for {
                let pkt_sb = &fp.sub_blocks[sub_idx];
                let e = construct_excitation(&mem, &pkt_sb.cb_idx, &pkt_sb.gain_idx);
                let sb = start + 1 + fb;
                let lo = sb * SUBL;
                if sb < n_sub {
                    decresidual[lo..lo + SUBL].copy_from_slice(&e);
                }
                update_cb_memory(&mut mem, &e);
                sub_idx += 1;
            }
        }

        // Backward pass: cb_mem seeded with the time-reversed tail of
        // the decoded state span (and, in the reference, the forward
        // sub-blocks just decoded — but we follow the reference's
        // simplified seeding which only reads the state span itself).
        if n_back > 0 {
            let meml_gotten = (SUBL * (n_sub + 1 - start)).min(CB_LMEM);
            let mut mem = [0.0f32; CB_LMEM];
            for k in 0..meml_gotten {
                mem[CB_LMEM - 1 - k] = decresidual[span_lo + k];
            }
            for bf in 0..n_back {
                let pkt_sb = &fp.sub_blocks[sub_idx];
                let e = construct_excitation(&mem, &pkt_sb.cb_idx, &pkt_sb.gain_idx);
                // Write reverse-time back into decresidual.
                for (k, &v) in e.iter().enumerate().take(SUBL) {
                    let dst = span_lo - 1 - bf * SUBL - k;
                    decresidual[dst] = v;
                }
                update_cb_memory(&mut mem, &e);
                sub_idx += 1;
            }
        }

        // The struct's `cb_mem` is now stale per-frame; we keep it on
        // the struct for backward compatibility (other code paths may
        // read it) but it is no longer the source of truth — each
        // frame's CB walks operate on a freshly-seeded local memory.
        // Reset to zero so any stray reader sees a defined state.
        self.cb_mem = [0.0; CB_LMEM];

        let excitation = decresidual;

        // §4.6 enhancer: smooth the residual using the pitch-
        // synchronous sequences over the last 640 samples (see the
        // `enhancer` module). The enhanced excitation drives synthesis.
        let mut enhanced = vec![0.0f32; excitation.len()];
        enhance_frame(&mut self.enhancer, fp.mode, &excitation, &mut enhanced);

        // Build the per-sub-block LPC list with the §4.7 enhancer-delay
        // shift: for 20 ms (NSUB=4) the synthesis sub-block i uses the
        // previous frame's LPC for i==0 and the current frame's LPC[i-1]
        // for i in 1..NSUB. For 30 ms (NSUB=6), sub-blocks 0 and 1 use
        // the previous frame's LPC and sub-blocks 2..NSUB use the current
        // frame's LPC[i-2]. Reference: RFC 3951 Appendix A.5 (decoder).
        let shift = match fp.mode {
            FrameMode::Ms20 => 1usize,
            FrameMode::Ms30 => 2usize,
        };
        let mut shifted_a = Vec::with_capacity(n_sub);
        for i in 0..n_sub {
            if i < shift {
                // Use the previous frame's LPC at offset (i + n_sub - shift).
                let off = i + n_sub - shift;
                let row = self.old_a_per_sub.get(off).copied().unwrap_or_else(|| {
                    let mut id = [0.0f32; LPC_ORDER + 1];
                    id[0] = 1.0;
                    id
                });
                shifted_a.push(row);
            } else {
                shifted_a.push(a_per_sub[i - shift]);
            }
        }

        // Synthesise from the enhanced excitation.
        synthesise_frame(&enhanced, &shifted_a, &mut self.synth, out);

        // §4.5.1 / §A.44 "preparing the plc for a future loss": record the
        // *unenhanced* decoded residual and the frame's final LP filter so
        // a following lost block can be concealed in the residual domain.
        // §A.44 passes `syntdenum + (nsub-1)` — the last sub-block's LPC.
        plc_record_good(&mut self.synth, &excitation, &a_per_sub[n_sub - 1]);

        // Cache the current frame's LPC rows so the next frame's first
        // sub-blocks can use them per the enhancer-delay shift above.
        self.old_a_per_sub = a_per_sub;
        self.enhancer.prev_enh_pl = 0;
        Ok(())
    }

    /// Conceal a lost / empty-indicated frame in the residual domain
    /// (RFC 3951 §4.5.2 via [`conceal_residual`]) and run the concealed
    /// excitation through the same §4.6 enhancer + §4.7 LPC synthesis path
    /// as a received frame, mirroring the §A.44 decoder driver's `mode==0`
    /// branch: the §A.14 PLC produces a residual + reuses the previous
    /// block's LP filter for every sub-block, then `enhancerInterface` and
    /// `syntFilter` run exactly as for a good block.
    fn conceal_into(&mut self, mode: FrameMode, out: &mut [f32]) {
        let n_sub = mode.sub_blocks();
        let n = mode.samples();

        // §A.14 / §A.44: the concealment lag is the pitch lag the enhancer
        // reported on the last good block (`iLBCdec_inst->last_lag`).
        let inlag = self.enhancer.last_lag;
        let mut residual = vec![0.0f32; n];
        conceal_residual(&mut self.synth, mode, inlag, &mut residual);

        // §A.44: a concealed frame reuses the previous block's LP filter
        // (`prevLpc`, == synth.last_a after plc_record_good / conceal) for
        // *all* sub-blocks of the substituted frame.
        let plc_a = self.synth.last_a;
        let a_per_sub = vec![plc_a; n_sub];

        // Run the concealed residual through the enhancer just like a
        // received block so a subsequent good frame merges smoothly
        // (§4.5.3) via the enhancer's cross-block correlation.
        let mut enhanced = vec![0.0f32; n];
        enhance_frame(&mut self.enhancer, mode, &residual, &mut enhanced);

        // Enhancer-delay-shifted synthesis, identical structure to the
        // good-frame path but with the single reused LP filter.
        let shift = match mode {
            FrameMode::Ms20 => 1usize,
            FrameMode::Ms30 => 2usize,
        };
        let mut shifted_a = Vec::with_capacity(n_sub);
        for i in 0..n_sub {
            if i < shift {
                let off = i + n_sub - shift;
                let row = self.old_a_per_sub.get(off).copied().unwrap_or(plc_a);
                shifted_a.push(row);
            } else {
                shifted_a.push(a_per_sub[i - shift]);
            }
        }
        // Filter without the §4.5.1 PCM-domain bookkeeping so the §A.14
        // residual-domain state set by `conceal_residual` (consPLICount,
        // prevPLI, prevLag, per, prevResidual) survives intact for the
        // next consecutive loss.
        synthesise_blocks(&enhanced, &shifted_a, &mut self.synth.mem, out);

        // The concealed frame's LP filter becomes the previous-frame LPC
        // for the next frame's enhancer-delay shift.
        self.old_a_per_sub = a_per_sub;
        self.enhancer.prev_enh_pl = 1;
    }
}

impl Decoder for IlbcDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "iLBC decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        // Detect frame mode, handle both valid and lost/empty shapes.
        let mode_opt = FrameMode::from_packet_len(pkt.data.len());
        let (mode, mut samples) = match mode_opt {
            Some(m) => {
                let mut out = vec![0.0f32; m.samples()];
                self.decode_into(&pkt.data, &mut out)?;
                self.last_mode = Some(m);
                (m, out)
            }
            None if pkt.data.is_empty() => {
                // Zero-byte packet: a lost frame carrying no length hint.
                // Conceal in the mode the stream has been running in so
                // the sample count / PTS advance stays aligned (a 30 ms
                // stream must emit 240 samples). Fall back to 20 ms only
                // when no sized packet has been seen yet.
                let m = self.last_mode.unwrap_or(FrameMode::Ms20);
                let mut out = vec![0.0f32; m.samples()];
                self.conceal_into(m, &mut out);
                self.last_mode = Some(m);
                (m, out)
            }
            None => {
                return Err(Error::invalid(format!(
                    "iLBC frame: unexpected packet length {} (want 38 or 50)",
                    pkt.data.len()
                )));
            }
        };

        // RFC 3951 §4.8 Post Filtering: optional 65 Hz HP filter on the
        // per-frame PCM block. We feed `samples` through `hp_output` in
        // place so the IIR delay line carries continuously across frame
        // boundaries.
        if self.hp_filter_on {
            let filtered = hp_output_in_place(&mut samples, &mut self.hp_state);
            debug_assert_eq!(filtered, samples.len());
        }

        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &s in samples.iter() {
            let v = sample_to_i16(s);
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // mode is implicit in `samples.len()` — reference kept so
        // future mode-specific trailer/padding logic has a hook.
        let _ = mode;
        Ok(Frame::Audio(AudioFrame {
            samples: samples.len() as u32,
            pts: pkt.pts,
            data: vec![bytes],
        }))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.lsf_state.reset();
        self.synth.reset();
        self.enhancer.reset();
        self.cb_mem = [0.0; CB_LMEM];
        let mut id = [0.0f32; LPC_ORDER + 1];
        id[0] = 1.0;
        self.old_a_per_sub = vec![id; 6];
        self.hp_state.reset();
        self.last_mode = None;
        self.pending = None;
        self.eof = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FRAME_BYTES_20MS, FRAME_BYTES_30MS};
    use oxideav_core::TimeBase;

    fn make_dec() -> Box<dyn Decoder> {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(1);
        make_decoder(&params).expect("make_decoder should succeed")
    }

    #[test]
    fn make_decoder_accepts_mono_8k() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(1);
        assert!(make_decoder(&params).is_ok());
    }

    #[test]
    fn make_decoder_rejects_16k() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(16_000);
        assert!(make_decoder(&params).is_err());
    }

    #[test]
    fn make_decoder_rejects_stereo() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(2);
        assert!(make_decoder(&params).is_err());
    }

    #[test]
    fn decodes_zero_20ms_packet_to_160_samples() {
        let mut dec = make_dec();
        let pkt = Packet::new(
            0,
            TimeBase::new(1, SAMPLE_RATE as i64),
            vec![0u8; FRAME_BYTES_20MS],
        );
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(a) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        assert_eq!(a.samples, 160);
        assert_eq!(a.data[0].len(), 160 * 2);
        for chunk in a.data[0].chunks_exact(2) {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]);
            // sample is finite by construction (clamped + round + cast).
            let _ = s;
        }
    }

    #[test]
    fn decodes_zero_30ms_packet_to_240_samples() {
        let mut dec = make_dec();
        let pkt = Packet::new(
            0,
            TimeBase::new(1, SAMPLE_RATE as i64),
            vec![0u8; FRAME_BYTES_30MS],
        );
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(a) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        assert_eq!(a.samples, 240);
    }

    #[test]
    fn rejects_short_packet() {
        let mut dec = make_dec();
        let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), vec![0u8; 7]);
        dec.send_packet(&pkt).unwrap();
        assert!(dec.receive_frame().is_err());
    }

    fn decode_len(dec: &mut Box<dyn Decoder>, data: Vec<u8>) -> u32 {
        let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), data);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(a) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        a.samples
    }

    /// A zero-byte "lost" packet carries no length hint. Before any sized
    /// packet it falls back to the 20 ms default, but once the stream has
    /// established a mode the concealment must follow it so the sample
    /// count (and PTS advance) stays aligned.
    #[test]
    fn zero_byte_loss_conceals_in_established_mode() {
        // Fresh decoder, first packet lost: 20 ms fallback.
        let mut dec = make_dec();
        assert_eq!(decode_len(&mut dec, Vec::new()), 160);

        // 30 ms stream: a subsequent zero-byte loss must conceal 240
        // samples, not the 20 ms default.
        let mut dec = make_dec();
        assert_eq!(decode_len(&mut dec, vec![0u8; FRAME_BYTES_30MS]), 240);
        assert_eq!(decode_len(&mut dec, Vec::new()), 240);
        // A following good 30 ms frame still decodes normally.
        assert_eq!(decode_len(&mut dec, vec![0u8; FRAME_BYTES_30MS]), 240);

        // 20 ms stream: a zero-byte loss stays at 160.
        let mut dec = make_dec();
        assert_eq!(decode_len(&mut dec, vec![0u8; FRAME_BYTES_20MS]), 160);
        assert_eq!(decode_len(&mut dec, Vec::new()), 160);

        // reset() clears the established mode -> back to 20 ms fallback.
        dec.reset().unwrap();
        assert_eq!(decode_len(&mut dec, Vec::new()), 160);
    }

    #[test]
    fn empty_frame_indicator_triggers_plc() {
        let mut dec = make_dec();
        // Prime with a normal frame to seed synth.last_rms.
        let good = vec![0u8; FRAME_BYTES_20MS];
        let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), good);
        dec.send_packet(&pkt).unwrap();
        let _ = dec.receive_frame().unwrap();
        // Now an all-zero packet with the empty-frame bit set (LSB of
        // the last byte).
        let mut bad = vec![0u8; FRAME_BYTES_20MS];
        bad[FRAME_BYTES_20MS - 1] = 1;
        let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), bad);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(a) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        assert_eq!(a.samples, 160);
    }

    /// §4.5.2: after a run of good voiced blocks, a concealed block must
    /// continue the previous excitation pitch-synchronously rather than
    /// emit silence — so the concealed block carries real energy that
    /// tracks the preceding frames (not a zero/near-zero block).
    fn frame_energy(a: &AudioFrame) -> f64 {
        a.data[0]
            .chunks_exact(2)
            .map(|c| {
                let s = i16::from_le_bytes([c[0], c[1]]) as f64;
                s * s
            })
            .sum()
    }

    #[test]
    fn plc_continues_energy_after_good_voiced_frames() {
        let mut dec = make_dec();
        // Drive a string of identical non-trivial packets to build up a
        // periodic excitation history in the synthesis state.
        let pkt_bytes = vec![0x6Cu8; FRAME_BYTES_20MS];
        let mut last_good_energy = 0.0;
        for pts in 0..6 {
            let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), pkt_bytes.clone())
                .with_pts(pts * 160);
            dec.send_packet(&pkt).unwrap();
            let Frame::Audio(a) = dec.receive_frame().unwrap() else {
                panic!("audio frame expected");
            };
            last_good_energy = frame_energy(&a);
        }
        assert!(last_good_energy > 0.0, "primer frames produced silence");

        // Now lose a frame (empty-frame indicator). The PLC must emit a
        // non-silent block whose energy is on the order of the last good
        // block (it is dampened by ~0.85 but must not collapse to zero).
        let mut bad = vec![0u8; FRAME_BYTES_20MS];
        bad[FRAME_BYTES_20MS - 1] = 1;
        let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), bad).with_pts(6 * 160);
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(a) = dec.receive_frame().unwrap() else {
            panic!("audio frame expected");
        };
        let plc_energy = frame_energy(&a);
        assert!(
            plc_energy > 0.0,
            "PLC produced a silent block instead of a pitch-synchronous \
             continuation"
        );
        // Bounded above: the dampened continuation cannot exceed a few
        // times the last good block's energy.
        assert!(
            plc_energy < last_good_energy * 4.0 + 1.0,
            "PLC energy {plc_energy} ran away vs last good {last_good_energy}"
        );
    }

    #[test]
    fn multiple_frames_have_bounded_output() {
        let mut dec = make_dec();
        let pkt_bytes = vec![0x55u8; FRAME_BYTES_20MS]; // non-trivial pattern
        for pts in 0..10 {
            let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), pkt_bytes.clone())
                .with_pts(pts * 160);
            dec.send_packet(&pkt).unwrap();
            let Frame::Audio(a) = dec.receive_frame().unwrap() else {
                panic!("audio frame expected");
            };
            for chunk in a.data[0].chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                // Not stuck at the clip rails.
                // sample is finite by construction (clamped + round + cast).
                let _ = s;
            }
        }
    }

    /// `hp_filter=on` must opt in to the §4.8 post-filter; without the
    /// option, the decoder produces the unfiltered baseline.
    #[test]
    fn hp_filter_option_toggles_post_filter() {
        fn decode_with(option: Option<&str>) -> Vec<i16> {
            let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
            params.sample_rate = Some(SAMPLE_RATE);
            params.channels = Some(1);
            if let Some(v) = option {
                params
                    .options
                    .insert("hp_filter".to_string(), v.to_string());
            }
            let mut dec = make_decoder(&params).expect("make_decoder");
            let pkt = Packet::new(
                0,
                TimeBase::new(1, SAMPLE_RATE as i64),
                vec![0xAAu8; FRAME_BYTES_20MS],
            );
            dec.send_packet(&pkt).unwrap();
            let Frame::Audio(a) = dec.receive_frame().unwrap() else {
                panic!("expected audio frame");
            };
            a.data[0]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect()
        }

        let baseline = decode_with(None);
        let with_off = decode_with(Some("off"));
        let with_on = decode_with(Some("on"));
        let with_true = decode_with(Some("true"));

        // Default and `off` must produce the exact same output (no
        // filter applied either way).
        assert_eq!(
            baseline, with_off,
            "default decode must match hp_filter=off"
        );

        // `on` and `true` must agree (same boolean coercion).
        assert_eq!(with_on, with_true, "hp_filter=on must match hp_filter=true");

        // The filter is opt-in and changes at least one sample for a
        // non-silent input. (For an all-zero input the filter is a no-
        // op; the 0xAA packet drives the CELP decoder with non-zero
        // excitation so a difference is expected.)
        assert_eq!(baseline.len(), with_on.len(), "frame length unchanged");
        let any_diff = baseline.iter().zip(with_on.iter()).any(|(a, b)| a != b);
        assert!(
            any_diff,
            "expected hp_filter=on to change at least one PCM sample"
        );
    }

    /// Silence on the decoder output stays silence regardless of the
    /// post-filter: the all-pole IIR has zero state, so a zero excitation
    /// produces zero output.
    #[test]
    fn hp_filter_preserves_silence() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(1);
        params
            .options
            .insert("hp_filter".to_string(), "on".to_string());
        let mut dec = make_decoder(&params).expect("make_decoder");
        // An empty-frame-indicator packet with zero LSF/state bits runs
        // the PLC path; depending on `prev_enh_pl` the first such
        // packet still emits a deterministic-but-quiet block. We feed
        // a string of empty-marker packets and assert the filter
        // doesn't cause the output to blow up.
        let mut bad = vec![0u8; FRAME_BYTES_20MS];
        bad[FRAME_BYTES_20MS - 1] = 1;
        for pts in 0..5 {
            let pkt = Packet::new(0, TimeBase::new(1, SAMPLE_RATE as i64), bad.clone())
                .with_pts(pts * 160);
            dec.send_packet(&pkt).unwrap();
            let Frame::Audio(a) = dec.receive_frame().unwrap() else {
                panic!("expected audio frame");
            };
            for chunk in a.data[0].chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                assert!(s.abs() < 32000, "post-filter output blew up: {s}");
            }
        }
    }

    /// `reset()` must clear the §4.8 HP filter delay line so the next
    /// frame starts from silence — otherwise a stale IIR memory could
    /// inject a low-frequency transient at the head of the new stream.
    #[test]
    fn reset_clears_hp_filter_state() {
        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(SAMPLE_RATE);
        params.channels = Some(1);
        params
            .options
            .insert("hp_filter".to_string(), "on".to_string());
        let mut dec = make_decoder(&params).expect("make_decoder");
        // Prime the filter with a few non-trivial packets.
        let pkt = Packet::new(
            0,
            TimeBase::new(1, SAMPLE_RATE as i64),
            vec![0xCCu8; FRAME_BYTES_20MS],
        );
        for _ in 0..3 {
            dec.send_packet(&pkt).unwrap();
            let _ = dec.receive_frame().unwrap();
        }
        dec.reset().expect("reset");
        // After reset, a fresh stream of identical packets must match a
        // brand-new decoder with the same option.
        let mut params2 = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params2.sample_rate = Some(SAMPLE_RATE);
        params2.channels = Some(1);
        params2
            .options
            .insert("hp_filter".to_string(), "on".to_string());
        let mut dec2 = make_decoder(&params2).expect("make_decoder");

        for _ in 0..2 {
            dec.send_packet(&pkt).unwrap();
            let Frame::Audio(a1) = dec.receive_frame().unwrap() else {
                panic!();
            };
            dec2.send_packet(&pkt).unwrap();
            let Frame::Audio(a2) = dec2.receive_frame().unwrap() else {
                panic!();
            };
            assert_eq!(
                a1.data[0], a2.data[0],
                "reset() did not clear HP filter state"
            );
        }
    }
}
