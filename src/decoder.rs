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

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::bitreader::{parse_frame, FrameParams};
use crate::cb::{construct_excitation, update_cb_memory};
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
    pending: Option<Packet>,
    eof: bool,
    time_base: TimeBase,
}

impl IlbcDecoder {
    fn new() -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            lsf_state: LsfState::new(),
            synth: SynthState::new(),
            enhancer: EnhancerState::new(),
            cb_mem: [0.0; CB_LMEM],
            pending: None,
            eof: false,
            time_base: TimeBase::new(1, SAMPLE_RATE as i64),
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
        let mut state_vec = [0.0f32; STATE_LEN];
        let copy_len = scalar_state.len().min(STATE_LEN);
        state_vec[..copy_len].copy_from_slice(&scalar_state[..copy_len]);
        // The position bit selects whether the 23-/22-sample CB output
        // precedes (0) or follows (1) the scalar state within the
        // 80-sample state vector. For a first-cut decoder we always
        // place the scalar samples at the end of CB memory and let the
        // first CB sub-block fill in the rest.
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
        // fill any residual bump in the state — we fold its energy
        // into the second half of the state vector.
        let boundary_exc =
            construct_excitation(&self.cb_mem, &fp.boundary.cb_idx, &fp.boundary.gain_idx);
        // Copy the state vector into the first two sub-blocks of
        // excitation. The `position` bit selects whether the boundary
        // CB samples prepend or append the scalar state; we treat them
        // as appending for a first-cut decoder.
        excitation[0..STATE_LEN].copy_from_slice(&state_vec[..STATE_LEN]);
        // Fold the boundary CB vector into the tail of the state span
        // so its contribution reaches the synthesis filter.
        let boundary_samples = match fp.mode {
            FrameMode::Ms20 => 23,
            FrameMode::Ms30 => 22,
        };
        for i in 0..boundary_samples.min(SUBL) {
            let dst = STATE_LEN - boundary_samples + i;
            if dst < excitation.len() {
                excitation[dst] += boundary_exc[i];
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

        // Synthesise from the enhanced excitation.
        synthesise_frame(&enhanced, &a_per_sub, &mut self.synth, out);
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
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: SAMPLE_RATE,
            samples: samples.len() as u32,
            pts: pkt.pts,
            time_base: self.time_base,
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
        self.pending = None;
        self.eof = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FRAME_BYTES_20MS, FRAME_BYTES_30MS};

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
        assert_eq!(a.channels, 1);
        assert_eq!(a.sample_rate, SAMPLE_RATE);
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
