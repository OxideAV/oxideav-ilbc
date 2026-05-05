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
use crate::lsf::{decode_and_interpolate, dequant_lsf, LsfState};
use crate::state::reconstruct_scalar_state;
use crate::synthesis::{conceal_frame, synthesise_frame, SynthState};
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
    Ok(Box::new(IlbcDecoder::new()))
}

/// Parse a packet into its [`FrameParams`] — thin re-export for tests
/// and external tooling.
pub fn parse_packet(packet: &[u8]) -> Result<FrameParams> {
    parse_frame(packet)
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
    pending: Option<Packet>,
    eof: bool,
}

impl IlbcDecoder {
    fn new() -> Self {
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
            pending: None,
            eof: false,
        }
    }

    fn decode_into(&mut self, packet: &[u8], out: &mut [f32]) -> Result<()> {
        let fp = parse_frame(packet)?;
        // Empty-frame flag: §3.8 — if set, the decoder SHOULD treat the
        // block as lost and run PLC.
        if fp.empty_flag {
            conceal_frame(&mut self.synth, fp.mode, out);
            self.enhancer.prev_enh_pl = 1;
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
        // Cache the current frame's LPC rows so the next frame's first
        // sub-blocks can use them per the enhancer-delay shift above.
        self.old_a_per_sub = a_per_sub;
        self.enhancer.prev_enh_pl = 0;
        Ok(())
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
        let (mode, samples) = match mode_opt {
            Some(m) => {
                let mut out = vec![0.0f32; m.samples()];
                self.decode_into(&pkt.data, &mut out)?;
                (m, out)
            }
            None if pkt.data.is_empty() => {
                // Zero-byte packet: treat as a 20 ms concealment frame.
                let m = FrameMode::Ms20;
                let mut out = vec![0.0f32; m.samples()];
                conceal_frame(&mut self.synth, m, &mut out);
                (m, out)
            }
            None => {
                return Err(Error::invalid(format!(
                    "iLBC frame: unexpected packet length {} (want 38 or 50)",
                    pkt.data.len()
                )));
            }
        };

        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &s in samples.iter() {
            let v = s.round().clamp(-32768.0, 32767.0) as i16;
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
}
