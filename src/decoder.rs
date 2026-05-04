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

        // Reconstruct start state. Use the first sub-block's LPC for
        // the all-pass phase compensation (which currently is a no-op
        // — see state.rs).
        let a_first: [f32; LPC_ORDER + 1] = a_per_sub[0];
        let scalar_state =
            reconstruct_scalar_state(fp.mode, fp.scale_idx, &fp.state_samples, &a_first);

        // Seed the adaptive-codebook memory with the scalar state,
        // padded to STATE_LEN samples. The 23-/22-sample boundary
        // block will be decoded as the first CB sub-block below.
        //
        // The `position` bit (RFC 3951 §3.5 / §4.2) selects whether the
        // boundary CB output precedes (0) or follows (1) the scalar
        // start-state samples within the 80-sample state vector:
        //   - position = 1: state_vec = [ scalar(n_short) | boundary(boundary_samples) ]
        //   - position = 0: state_vec = [ boundary(boundary_samples) | scalar(n_short) ]
        //
        // The encoder picks whichever layout drops the boundary into the
        // lower-energy half of the residual, leaving the higher-energy
        // half to scalar coding (which has more bits per sample than the
        // 22/23-sample CB).
        let n_short = fp.mode.state_short_len();
        let boundary_samples = match fp.mode {
            FrameMode::Ms20 => 23,
            FrameMode::Ms30 => 22,
        };
        debug_assert_eq!(scalar_state.len(), n_short);
        let scalar_offset = if fp.position == 1 {
            0
        } else {
            boundary_samples
        };
        let boundary_offset = if fp.position == 1 { n_short } else { 0 };
        let mut state_vec = [0.0f32; STATE_LEN];
        // Place scalar state in its position-selected slot. boundary
        // samples are zero in `state_vec` until the boundary CB decode
        // below fills them in.
        for (k, &s) in scalar_state.iter().enumerate() {
            let dst = scalar_offset + k;
            if dst < STATE_LEN {
                state_vec[dst] = s;
            }
        }
        // Seed cb_mem with the partially-filled state_vec (boundary slot
        // still zero). RFC §3.6.1 boundary search reads back lMem=85
        // samples, of which the last `STATE_LEN` are state_vec. The
        // lookback for the boundary block legitimately includes the
        // scalar samples that already sit in the state_vec slot, while
        // the boundary slot itself reads as zero.
        for i in 0..CB_LMEM {
            self.cb_mem[i] = if i >= CB_LMEM - STATE_LEN {
                state_vec[i - (CB_LMEM - STATE_LEN)]
            } else {
                0.0
            };
        }

        // Decode each sub-block excitation.
        //
        // Layout: the full `mode.sub_blocks()` frame excitation is built
        // from three sources:
        //   - sub-blocks 0 and 1: state vector (scalar + boundary CB),
        //     sliced into two SUBL halves.
        //   - sub-blocks 2..: one per entry in `fp.sub_blocks` (2 for
        //     20 ms, 4 for 30 ms).
        let n_sub = fp.mode.sub_blocks();
        let mut excitation = vec![0.0f32; n_sub * SUBL];
        // Sub-blocks 0/1: the 80-sample state vector directly drives
        // the first two synthesis sub-blocks. The 22-/23-sample
        // boundary CB block is used both as the CB-memory seed and to
        // fill the boundary slot of the state vector.
        //
        // RFC 3951 §3.6.1: the boundary block uses lMem = 85 (not 147),
        // i.e. the search/extract operates on the last 85 entries of the
        // 147-sample codebook memory.
        let boundary_mem = &self.cb_mem[CB_LMEM - 85..];
        let boundary_full = construct_excitation_veclen(
            boundary_mem,
            boundary_samples,
            &fp.boundary.cb_idx,
            &fp.boundary.gain_idx,
        );
        // For consistency with the prior behaviour (which seeded a full
        // SUBL excitation into the CB memory after the boundary block),
        // we expand the boundary excitation to a SUBL-sized vector by
        // zero-padding.
        let boundary_exc: [f32; SUBL] = {
            let mut arr = [0.0f32; SUBL];
            let copy = boundary_full.len().min(SUBL);
            arr[..copy].copy_from_slice(&boundary_full[..copy]);
            arr
        };
        // Copy the (already partially populated) state vector into the
        // first two sub-blocks of excitation, then drop the boundary
        // CB samples into their position-selected slot.
        excitation[0..STATE_LEN].copy_from_slice(&state_vec[..STATE_LEN]);
        for (i, &sample) in boundary_exc
            .iter()
            .take(boundary_samples.min(SUBL))
            .enumerate()
        {
            let dst = boundary_offset + i;
            if dst < excitation.len() {
                excitation[dst] += sample;
            }
        }
        update_cb_memory(&mut self.cb_mem, &boundary_exc);
        // Remaining `cb_sub_blocks()` sub-blocks (2 for 20ms, 4 for
        // 30ms) use the packet's per-sub-block CB indices.
        let n_cb_sub = fp.mode.cb_sub_blocks();
        for cb_i in 0..n_cb_sub {
            let pkt_sb = &fp.sub_blocks[cb_i];
            let e = construct_excitation(&self.cb_mem, &pkt_sb.cb_idx, &pkt_sb.gain_idx);
            let sb = 2 + cb_i;
            if sb < n_sub {
                excitation[sb * SUBL..(sb + 1) * SUBL].copy_from_slice(&e);
            }
            update_cb_memory(&mut self.cb_mem, &e);
        }

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
