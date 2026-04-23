//! iLBC codec registration.
//!
//! iLBC is a decode-only codec in this crate. Frame size is inferred
//! from packet length (38 -> 20 ms / 160 samples, 50 -> 30 ms / 240
//! samples). Both modes produce 8 kHz mono `S16` audio.

use oxideav_codec::{CodecInfo, CodecRegistry, Decoder};
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

use crate::{CODEC_ID_STR, SAMPLE_RATE};

/// Register the iLBC decoder on `reg`.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("ilbc_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(1)
        .with_max_sample_rate(SAMPLE_RATE);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    crate::decoder::make_decoder(params)
}
