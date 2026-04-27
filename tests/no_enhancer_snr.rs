//! Diagnostic: round-trip SNR with a bypassed enhancer.
//!
//! This test re-implements the decode path with the §4.6 enhancer
//! removed (excitation goes straight to the LPC synthesis filter). It
//! is *not* an interop test — it exists only so that the round-trip
//! SNR figures are not muddied by the enhancer's effect on synthesised
//! voiced speech, which the encoder cannot model in lockstep.

use oxideav_core::CodecRegistry;
use oxideav_core::{
    AudioFrame, CodecId, CodecOptions, CodecParameters, Encoder, Frame, SampleFormat,
};

use oxideav_ilbc::bitreader::parse_frame;
use oxideav_ilbc::cb::{construct_excitation, update_cb_memory};
use oxideav_ilbc::lsf::{decode_and_interpolate, dequant_lsf, LsfState};
use oxideav_ilbc::state::reconstruct_scalar_state;
use oxideav_ilbc::synthesis::{synthesise_frame, SynthState};
use oxideav_ilbc::{FrameMode, CB_LMEM, CODEC_ID_STR, LPC_ORDER, SAMPLE_RATE, STATE_LEN, SUBL};

struct BypassDecoder {
    lsf_state: LsfState,
    synth: SynthState,
    cb_mem: [f32; CB_LMEM],
}

impl BypassDecoder {
    fn new() -> Self {
        Self {
            lsf_state: LsfState::new(),
            synth: SynthState::new(),
            cb_mem: [0.0; CB_LMEM],
        }
    }

    fn decode(&mut self, packet: &[u8]) -> Vec<i16> {
        let fp = parse_frame(packet).expect("parse");
        let mut lsf_vectors = Vec::new();
        for idx in &fp.lsf_idx {
            lsf_vectors.push(dequant_lsf(idx));
        }
        let a_per_sub = decode_and_interpolate(&mut self.lsf_state, fp.mode, &lsf_vectors);
        let a_first: [f32; LPC_ORDER + 1] = a_per_sub[0];
        let scalar_state =
            reconstruct_scalar_state(fp.mode, fp.scale_idx, &fp.state_samples, &a_first);
        let mut state_vec = [0.0f32; STATE_LEN];
        let copy_len = scalar_state.len().min(STATE_LEN);
        state_vec[..copy_len].copy_from_slice(&scalar_state[..copy_len]);
        for i in 0..CB_LMEM {
            self.cb_mem[i] = if i >= CB_LMEM - STATE_LEN {
                state_vec[i - (CB_LMEM - STATE_LEN)]
            } else {
                0.0
            };
        }
        let n_sub = fp.mode.sub_blocks();
        let mut excitation = vec![0.0f32; n_sub * SUBL];
        let boundary_exc =
            construct_excitation(&self.cb_mem, &fp.boundary.cb_idx, &fp.boundary.gain_idx);
        excitation[0..STATE_LEN].copy_from_slice(&state_vec[..STATE_LEN]);
        let boundary_samples = match fp.mode {
            FrameMode::Ms20 => 23,
            FrameMode::Ms30 => 22,
        };
        for (i, &sample) in boundary_exc
            .iter()
            .take(boundary_samples.min(SUBL))
            .enumerate()
        {
            let dst = STATE_LEN - boundary_samples + i;
            if dst < excitation.len() {
                excitation[dst] += sample;
            }
        }
        update_cb_memory(&mut self.cb_mem, &boundary_exc);
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
        // BYPASS THE ENHANCER — feed `excitation` directly to synthesis.
        let mut out = vec![0.0f32; n_sub * SUBL];
        synthesise_frame(&excitation, &a_per_sub, &mut self.synth, &mut out);
        out.iter()
            .map(|&v| v.round().clamp(-32768.0, 32767.0) as i16)
            .collect()
    }
}

fn gen_voiced(samples: usize) -> Vec<i16> {
    let f0 = 130.0f32;
    (0..samples)
        .map(|n| {
            let t = n as f32 / SAMPLE_RATE as f32;
            let mut v = 0.0f32;
            for h in 1..5 {
                v +=
                    ((2.0 * core::f32::consts::PI * h as f32 * f0 * t).sin()) * (3000.0 / h as f32);
            }
            v.round().clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

fn pcm_to_audio_frame(pcm: &[i16]) -> Frame {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        samples: pcm.len() as u32,
        pts: Some(0),
        data: vec![bytes],
    })
}

fn round_trip_no_enh(mode: FrameMode, pcm: &[i16]) -> Vec<i16> {
    let mut reg = CodecRegistry::new();
    oxideav_ilbc::register(&mut reg);
    let mut enc_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    enc_params.sample_rate = Some(SAMPLE_RATE);
    enc_params.channels = Some(1);
    enc_params.sample_format = Some(SampleFormat::S16);
    let mut options = CodecOptions::new();
    if mode == FrameMode::Ms30 {
        options = options.set("frame_ms", "30");
    }
    enc_params.options = options;
    let mut enc: Box<dyn Encoder> = reg.make_encoder(&enc_params).expect("encoder");

    let mut dec = BypassDecoder::new();
    enc.send_frame(&pcm_to_audio_frame(pcm)).unwrap();
    enc.flush().unwrap();
    let mut decoded = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        let s = dec.decode(&pkt.data);
        decoded.extend(s);
    }
    decoded
}

fn best_snr_db(reference: &[i16], test: &[i16], max_lag: isize) -> f64 {
    let len = reference.len().min(test.len());
    let mut best = f64::NEG_INFINITY;
    for lag in -max_lag..=max_lag {
        let mut s_sig = 0.0f64;
        let mut s_err = 0.0f64;
        for (i, &r_sample) in reference.iter().take(len).enumerate() {
            let j = i as isize + lag;
            if j < 0 || j as usize >= test.len() {
                continue;
            }
            let r = r_sample as f64;
            let t = test[j as usize] as f64;
            s_sig += r * r;
            s_err += (r - t) * (r - t);
        }
        if s_err < 1e-9 {
            return f64::INFINITY;
        }
        let snr = 10.0 * (s_sig / s_err).log10();
        if snr > best {
            best = snr;
        }
    }
    best
}

fn per_frame_best_snr_avg(
    reference: &[i16],
    test: &[i16],
    frame_len: usize,
    skip_frames: usize,
) -> f64 {
    let n_frames = reference.len() / frame_len;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for f in skip_frames..n_frames {
        let lo = f * frame_len;
        let hi = lo + frame_len;
        if hi > test.len() {
            break;
        }
        let r = &reference[lo..hi];
        let t = &test[lo..hi];
        let snr = best_snr_db(r, t, (frame_len / 2) as isize);
        if snr.is_finite() {
            sum += snr;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

#[test]
fn round_trip_voiced_20ms_no_enhancer() {
    let n_frames = 50;
    let pcm = gen_voiced(n_frames * 160);
    let decoded = round_trip_no_enh(FrameMode::Ms20, &pcm);
    let avg = per_frame_best_snr_avg(&pcm, &decoded, 160, 5);
    println!("BYPASS ENHANCER 20ms voiced: per-frame SNR = {avg:.2} dB");
}

#[test]
fn round_trip_voiced_30ms_no_enhancer() {
    let n_frames = 40;
    let pcm = gen_voiced(n_frames * 240);
    let decoded = round_trip_no_enh(FrameMode::Ms30, &pcm);
    let avg = per_frame_best_snr_avg(&pcm, &decoded, 240, 4);
    println!("BYPASS ENHANCER 30ms voiced: per-frame SNR = {avg:.2} dB");
}
