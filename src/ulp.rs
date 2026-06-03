//! Uneven-level-protection (ULP) bit layout for RFC 3951 §3.8.
//!
//! iLBC frames are not laid out as a flat sequence of parameter fields
//! on the wire. Each parameter is *split* across three bit-error
//! sensitivity classes — class 1 (most sensitive, placed first), class
//! 2, and class 3 (least sensitive) — so the RTP payload format
//! (RFC 3952) can apply uneven channel protection. The split widths
//! per parameter come from `ULP_20msTbl` / `ULP_30msTbl` in RFC 3951
//! Appendix A.41 (`packing.h`). For each parameter the table carries
//! a 5-element row `[c1, c2, c3, 0, 0]` summing to the parameter's
//! full bit width.
//!
//! The encoder visits the parameter list three times, emitting only
//! the class-N high-order bits of each value on pass N. The decoder
//! performs the symmetric reconstruction with `unpack` + `packcombine`
//! (RFC 3951 Appendix A.42 / A.41 `packing.c`).
//!
//! This module holds the ULP tables and a small typed driver. It is
//! independent of the [`crate::bitreader`] / [`crate::bitwriter`]
//! `BitReader` / `BitWriter` primitives — those still provide the raw
//! MSB-first bit IO over a byte slice. ULP just specifies the *order*
//! of fields and the per-pass widths.

use crate::FrameMode;

/// The five-slot ULP row used per parameter. iLBC only ever fills the
/// first three slots; the trailing zeros are kept for fidelity with
/// the RFC tables.
pub(crate) type UlpRow = [u8; 5];

/// Per-mode ULP table. All `*_bits` arrays are indexed `[parameter]
/// [class]`. The `lsf_bits[k]` row applies to the `k`-th LSF split-VQ
/// index (k ∈ 0..LSF_NSPLIT*lpc_n). Per-subblock CB / gain tables are
/// indexed `[sub_block][stage]`. The boundary 22/23-sample block is
/// indexed `[stage]` under the `extra_*` names matching the RFC.
pub(crate) struct UlpTable {
    pub lsf_bits: &'static [UlpRow],
    pub start_bits: UlpRow,
    pub startfirst_bits: UlpRow,
    pub scale_bits: UlpRow,
    pub state_bits: UlpRow,
    pub extra_cb_index: &'static [UlpRow; 3],
    pub extra_cb_gain: &'static [UlpRow; 3],
    pub cb_index: &'static [[UlpRow; 3]],
    pub cb_gain: &'static [[UlpRow; 3]],
}

/// RFC 3951 §A.41 `ULP_20msTbl`.
const ULP_20MS: UlpTable = UlpTable {
    lsf_bits: &[[6, 0, 0, 0, 0], [7, 0, 0, 0, 0], [7, 0, 0, 0, 0]],
    start_bits: [2, 0, 0, 0, 0],
    startfirst_bits: [1, 0, 0, 0, 0],
    scale_bits: [6, 0, 0, 0, 0],
    state_bits: [0, 1, 2, 0, 0],
    extra_cb_index: &[[6, 0, 1, 0, 0], [0, 0, 7, 0, 0], [0, 0, 7, 0, 0]],
    extra_cb_gain: &[[2, 0, 3, 0, 0], [1, 1, 2, 0, 0], [0, 0, 3, 0, 0]],
    cb_index: &[
        [[7, 0, 1, 0, 0], [0, 0, 7, 0, 0], [0, 0, 7, 0, 0]],
        [[0, 0, 8, 0, 0], [0, 0, 8, 0, 0], [0, 0, 8, 0, 0]],
    ],
    cb_gain: &[
        [[1, 2, 2, 0, 0], [1, 1, 2, 0, 0], [0, 0, 3, 0, 0]],
        [[1, 1, 3, 0, 0], [0, 2, 2, 0, 0], [0, 0, 3, 0, 0]],
    ],
};

/// RFC 3951 §A.41 `ULP_30msTbl`.
const ULP_30MS: UlpTable = UlpTable {
    lsf_bits: &[
        [6, 0, 0, 0, 0],
        [7, 0, 0, 0, 0],
        [7, 0, 0, 0, 0],
        [6, 0, 0, 0, 0],
        [7, 0, 0, 0, 0],
        [7, 0, 0, 0, 0],
    ],
    start_bits: [3, 0, 0, 0, 0],
    startfirst_bits: [1, 0, 0, 0, 0],
    scale_bits: [6, 0, 0, 0, 0],
    state_bits: [0, 1, 2, 0, 0],
    extra_cb_index: &[[4, 2, 1, 0, 0], [0, 0, 7, 0, 0], [0, 0, 7, 0, 0]],
    extra_cb_gain: &[[1, 1, 3, 0, 0], [1, 1, 2, 0, 0], [0, 0, 3, 0, 0]],
    cb_index: &[
        [[6, 1, 1, 0, 0], [0, 0, 7, 0, 0], [0, 0, 7, 0, 0]],
        [[0, 7, 1, 0, 0], [0, 0, 8, 0, 0], [0, 0, 8, 0, 0]],
        [[0, 7, 1, 0, 0], [0, 0, 8, 0, 0], [0, 0, 8, 0, 0]],
        [[0, 7, 1, 0, 0], [0, 0, 8, 0, 0], [0, 0, 8, 0, 0]],
    ],
    cb_gain: &[
        [[1, 2, 2, 0, 0], [1, 2, 1, 0, 0], [0, 0, 3, 0, 0]],
        [[0, 2, 3, 0, 0], [0, 2, 2, 0, 0], [0, 0, 3, 0, 0]],
        [[0, 1, 4, 0, 0], [0, 1, 3, 0, 0], [0, 0, 3, 0, 0]],
        [[0, 1, 4, 0, 0], [0, 1, 3, 0, 0], [0, 0, 3, 0, 0]],
    ],
};

/// Return the ULP table for a given mode.
pub(crate) fn table(mode: FrameMode) -> &'static UlpTable {
    match mode {
        FrameMode::Ms20 => &ULP_20MS,
        FrameMode::Ms30 => &ULP_30MS,
    }
}

/// Pre-flight check: every per-parameter ULP row sums to that
/// parameter's documented width. Used at module load to catch
/// transcription typos.
#[cfg(test)]
fn assert_ulp_widths(mode: FrameMode) {
    let t = table(mode);
    // LSF splits — 6, 7, 7 bits each.
    let n_lsf = match mode {
        FrameMode::Ms20 => 3,
        FrameMode::Ms30 => 6,
    };
    assert_eq!(t.lsf_bits.len(), n_lsf);
    for (i, row) in t.lsf_bits.iter().enumerate() {
        let sum: u16 = row.iter().map(|&b| b as u16).sum();
        let want = if i % 3 == 0 { 6 } else { 7 };
        assert_eq!(sum, want, "lsf_bits[{i}] sums to {sum}, expected {want}");
    }
    let start_bits = match mode {
        FrameMode::Ms20 => 2u16,
        FrameMode::Ms30 => 3u16,
    };
    assert_eq!(
        t.start_bits.iter().map(|&b| b as u16).sum::<u16>(),
        start_bits
    );
    assert_eq!(t.startfirst_bits.iter().map(|&b| b as u16).sum::<u16>(), 1);
    assert_eq!(t.scale_bits.iter().map(|&b| b as u16).sum::<u16>(), 6);
    assert_eq!(t.state_bits.iter().map(|&b| b as u16).sum::<u16>(), 3);
    // Boundary block CB: 7,7,7. Gain: 5,4,3.
    let cb_want = [7u16, 7, 7];
    let g_want = [5u16, 4, 3];
    for (k, row) in t.extra_cb_index.iter().enumerate() {
        assert_eq!(
            row.iter().map(|&b| b as u16).sum::<u16>(),
            cb_want[k],
            "extra_cb_index[{k}]"
        );
    }
    for (k, row) in t.extra_cb_gain.iter().enumerate() {
        assert_eq!(
            row.iter().map(|&b| b as u16).sum::<u16>(),
            g_want[k],
            "extra_cb_gain[{k}]"
        );
    }
    // Sub-blocks: first sub-block has stage widths (8,7,7); others (8,8,8).
    let nasub = match mode {
        FrameMode::Ms20 => 2,
        FrameMode::Ms30 => 4,
    };
    assert_eq!(t.cb_index.len(), nasub);
    assert_eq!(t.cb_gain.len(), nasub);
    for (i, stages) in t.cb_index.iter().enumerate() {
        let widths = if i == 0 { [8u16, 7, 7] } else { [8u16, 8, 8] };
        for (k, row) in stages.iter().enumerate() {
            assert_eq!(
                row.iter().map(|&b| b as u16).sum::<u16>(),
                widths[k],
                "cb_index[{i}][{k}]"
            );
        }
    }
    for stages in t.cb_gain.iter() {
        let widths = [5u16, 4, 3];
        for (k, row) in stages.iter().enumerate() {
            assert_eq!(
                row.iter().map(|&b| b as u16).sum::<u16>(),
                widths[k],
                "cb_gain[?][{k}]"
            );
        }
    }
}

/// All parameter values used by one frame, in the **same field order**
/// as RFC 3951 §3.8 / Table 3.2. Total bit width per parameter equals
/// the corresponding row's bit-count sum in the active ULP table.
///
/// This is the per-frame *logical* view — the bit layout (flat vs
/// ULP) is applied around this struct by [`pack_logical`] /
/// [`unpack_logical`].
#[derive(Clone, Debug, Default)]
pub(crate) struct LogicalParams {
    pub lsf_idx: Vec<u32>,
    pub start: u32,
    pub state_first: u32,
    pub idx_for_max: u32,
    pub state_samples: Vec<u32>,
    pub extra_cb_index: [u32; 3],
    pub extra_cb_gain: [u32; 3],
    pub cb_index: Vec<[u32; 3]>,
    pub cb_gain: Vec<[u32; 3]>,
}

/// Split a value into its three ULP class halves. Matches the RFC's
/// `packsplit` semantics: the top `widths[0]` bits go to class 1, the
/// next `widths[1]` to class 2, and the bottom `widths[2]` to class 3.
fn split3(value: u32, widths: &UlpRow) -> [u32; 3] {
    let w1 = widths[0] as u32;
    let w2 = widths[1] as u32;
    let w3 = widths[2] as u32;
    let mask = |w: u32| if w == 0 { 0 } else { (1u32 << w) - 1 };
    let lo = value & mask(w3);
    let mid = (value >> w3) & mask(w2);
    let hi = (value >> (w2 + w3)) & mask(w1);
    [hi, mid, lo]
}

/// Inverse of [`split3`]. Used in unit tests as a round-trip oracle.
#[cfg(test)]
fn combine3(parts: [u32; 3], widths: &UlpRow) -> u32 {
    let w2 = widths[1] as u32;
    let w3 = widths[2] as u32;
    (parts[0] << (w2 + w3)) | (parts[1] << w3) | parts[2]
}

/// Three-pass ULP serialiser. Walks the parameter list in the
/// RFC-mandated order, emitting class-`ulp` bits of each parameter
/// on pass `ulp ∈ {0, 1, 2}`, then appends a trailing zero bit (the
/// empty-frame indicator). Returns the per-pass list of `(value,
/// width)` emissions for the caller's bit-writer to consume.
pub(crate) fn pack_emit_list(
    mode: FrameMode,
    params: &LogicalParams,
    empty_flag: bool,
) -> Vec<(u32, u32)> {
    let t = table(mode);
    let n_lsf = t.lsf_bits.len();
    let n_state = params.state_samples.len();
    let nasub = t.cb_index.len();

    // Pre-split every value into its 3 class chunks. Iteration order
    // mirrors RFC's `iLBC_encode` packing loop.
    let lsf_split: Vec<[u32; 3]> = (0..n_lsf)
        .map(|k| split3(params.lsf_idx[k], &t.lsf_bits[k]))
        .collect();
    let start_split = split3(params.start, &t.start_bits);
    let startfirst_split = split3(params.state_first, &t.startfirst_bits);
    let scale_split = split3(params.idx_for_max, &t.scale_bits);
    let state_split: Vec<[u32; 3]> = (0..n_state)
        .map(|k| split3(params.state_samples[k], &t.state_bits))
        .collect();
    let extra_cb_split: Vec<[u32; 3]> = (0..3)
        .map(|k| split3(params.extra_cb_index[k], &t.extra_cb_index[k]))
        .collect();
    let extra_gain_split: Vec<[u32; 3]> = (0..3)
        .map(|k| split3(params.extra_cb_gain[k], &t.extra_cb_gain[k]))
        .collect();
    let cb_split: Vec<Vec<[u32; 3]>> = (0..nasub)
        .map(|i| {
            (0..3)
                .map(|k| split3(params.cb_index[i][k], &t.cb_index[i][k]))
                .collect()
        })
        .collect();
    let gain_split: Vec<Vec<[u32; 3]>> = (0..nasub)
        .map(|i| {
            (0..3)
                .map(|k| split3(params.cb_gain[i][k], &t.cb_gain[i][k]))
                .collect()
        })
        .collect();

    let mut out: Vec<(u32, u32)> = Vec::with_capacity(mode.bits());
    for ulp in 0..3 {
        for (k, lsf_row) in lsf_split.iter().enumerate().take(n_lsf) {
            push_field(&mut out, lsf_row[ulp], t.lsf_bits[k][ulp]);
        }
        push_field(&mut out, start_split[ulp], t.start_bits[ulp]);
        push_field(&mut out, startfirst_split[ulp], t.startfirst_bits[ulp]);
        push_field(&mut out, scale_split[ulp], t.scale_bits[ulp]);
        for state_row in state_split.iter().take(n_state) {
            push_field(&mut out, state_row[ulp], t.state_bits[ulp]);
        }
        for (k, row) in extra_cb_split.iter().enumerate() {
            push_field(&mut out, row[ulp], t.extra_cb_index[k][ulp]);
        }
        for (k, row) in extra_gain_split.iter().enumerate() {
            push_field(&mut out, row[ulp], t.extra_cb_gain[k][ulp]);
        }
        for (i, stages) in cb_split.iter().enumerate().take(nasub) {
            for (k, row) in stages.iter().enumerate() {
                push_field(&mut out, row[ulp], t.cb_index[i][k][ulp]);
            }
        }
        for (i, stages) in gain_split.iter().enumerate().take(nasub) {
            for (k, row) in stages.iter().enumerate() {
                push_field(&mut out, row[ulp], t.cb_gain[i][k][ulp]);
            }
        }
    }
    // The empty-frame indicator (1 bit) lives outside the ULP loop —
    // RFC 3951 §3.8 appends it as the last bit of the payload.
    out.push((u32::from(empty_flag), 1));
    out
}

fn push_field(out: &mut Vec<(u32, u32)>, value: u32, w: u8) {
    if w > 0 {
        out.push((value, w as u32));
    }
}

/// Three-pass ULP deserialiser. Reads `width` bits per parameter per
/// pass into the per-parameter accumulator. After three passes the
/// accumulator holds the original (joined) value.
///
/// The bit reader is supplied via the `read` closure so we don't lock
/// in a particular `BitReader` type.
pub(crate) fn unpack_logical<F>(
    mode: FrameMode,
    n_state: usize,
    mut read: F,
) -> Result<(LogicalParams, bool), oxideav_core::Error>
where
    F: FnMut(u32) -> Result<u32, oxideav_core::Error>,
{
    let t = table(mode);
    let n_lsf = t.lsf_bits.len();
    let nasub = t.cb_index.len();
    let mut params = LogicalParams {
        lsf_idx: vec![0; n_lsf],
        start: 0,
        state_first: 0,
        idx_for_max: 0,
        state_samples: vec![0; n_state],
        extra_cb_index: [0; 3],
        extra_cb_gain: [0; 3],
        cb_index: vec![[0u32; 3]; nasub],
        cb_gain: vec![[0u32; 3]; nasub],
    };
    for ulp in 0..3 {
        for (acc, row) in params.lsf_idx.iter_mut().zip(t.lsf_bits.iter()).take(n_lsf) {
            combine_in(acc, &mut read, row[ulp])?;
        }
        combine_in(&mut params.start, &mut read, t.start_bits[ulp])?;
        combine_in(&mut params.state_first, &mut read, t.startfirst_bits[ulp])?;
        combine_in(&mut params.idx_for_max, &mut read, t.scale_bits[ulp])?;
        for acc in params.state_samples.iter_mut().take(n_state) {
            combine_in(acc, &mut read, t.state_bits[ulp])?;
        }
        for (acc, row) in params
            .extra_cb_index
            .iter_mut()
            .zip(t.extra_cb_index.iter())
        {
            combine_in(acc, &mut read, row[ulp])?;
        }
        for (acc, row) in params.extra_cb_gain.iter_mut().zip(t.extra_cb_gain.iter()) {
            combine_in(acc, &mut read, row[ulp])?;
        }
        for (sub, table_sub) in params
            .cb_index
            .iter_mut()
            .zip(t.cb_index.iter())
            .take(nasub)
        {
            for (acc, row) in sub.iter_mut().zip(table_sub.iter()) {
                combine_in(acc, &mut read, row[ulp])?;
            }
        }
        for (sub, table_sub) in params.cb_gain.iter_mut().zip(t.cb_gain.iter()).take(nasub) {
            for (acc, row) in sub.iter_mut().zip(table_sub.iter()) {
                combine_in(acc, &mut read, row[ulp])?;
            }
        }
    }
    // Drain padding between the last ULP field and the empty-frame
    // indicator. Total ULP-field bit width equals `total_bits - 1`
    // exactly (the empty-frame indicator is the last bit), so there is
    // no padding in well-formed iLBC frames.
    let empty_flag = read(1)? != 0;
    Ok((params, empty_flag))
}

fn combine_in<F>(acc: &mut u32, read: &mut F, w: u8) -> Result<(), oxideav_core::Error>
where
    F: FnMut(u32) -> Result<u32, oxideav_core::Error>,
{
    if w == 0 {
        return Ok(());
    }
    let chunk = read(w as u32)?;
    *acc = (*acc << w) | chunk;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_widths_sum_to_field_widths() {
        assert_ulp_widths(FrameMode::Ms20);
        assert_ulp_widths(FrameMode::Ms30);
    }

    #[test]
    fn split3_round_trips() {
        // 7-bit value into (4,2,1) chunks (the 30 ms extra_cb_index[0]
        // row): hi 4 bits, mid 2 bits, lo 1 bit.
        let widths: UlpRow = [4, 2, 1, 0, 0];
        for v in 0..(1u32 << 7) {
            let parts = split3(v, &widths);
            assert_eq!(combine3(parts, &widths), v, "v={v}");
        }
    }

    #[test]
    fn split3_handles_zero_first_class() {
        // (0,1,2) — the state_bits row — value 0..=7.
        let widths: UlpRow = [0, 1, 2, 0, 0];
        for v in 0..8u32 {
            let parts = split3(v, &widths);
            assert_eq!(parts[0], 0);
            assert_eq!(combine3(parts, &widths), v);
        }
    }
}
