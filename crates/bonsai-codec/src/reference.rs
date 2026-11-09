//! Portable wire-format oracle. All integer rounding is nearest, ties to even.
//!
//! Payload is `[layer, head, token, channel]`, little endian for BF16; INT4
//! uses low nibble first and two's complement for symmetric codes. Metadata is
//! interleaved f16 `(scale, offset)` (scale only for symmetric), in
//! `[layer,head,token,channel_group]` order for HeadDim and
//! `[layer,head,token_group,channel]` order for Token groups. Groups never cross
//! a head or frame boundary; the final group can be short.
//!
//! Metadata is rounded to f16 *before* computing codes. Nonfinite quantization
//! inputs and metadata that overflow f16 are rejected. Underflowed/constant
//! zero scales produce code zero. BF16 copies preserve all bits.

use bonsai_core::{
    BonsaiError, ChunkDesc, CodecSpec, ElemType, GroupAxis, KvLayout, QuantBits, QuantParams,
    Result,
};
use half::{bf16, f16};

#[derive(Debug, Clone)]
pub struct HostKv {
    pub layout: KvLayout,
    pub seq_len: u32,
    pub k: Vec<Vec<u16>>,
    pub v: Vec<Vec<u16>>,
}

impl HostKv {
    pub fn zeros(layout: KvLayout, seq_len: u32) -> Self {
        let n = layout.elems(1, seq_len as u64) as usize;
        let l = layout.num_layers as usize;
        HostKv {
            layout,
            seq_len,
            k: vec![vec![0; n]; l],
            v: vec![vec![0; n]; l],
        }
    }
}

fn invalid(msg: &str) -> BonsaiError {
    BonsaiError::Codec(msg.into())
}

pub(crate) fn validate_frame(
    spec: &CodecSpec,
    layout: &KvLayout,
    seq_len: u32,
    chunk: &ChunkDesc,
    frame_len: usize,
) -> Result<()> {
    spec.validate(layout)?;
    if layout.dtype != ElemType::Bf16
        || layout.num_kv_heads == 0
        || layout.head_dim == 0
        || layout.num_layers == 0
    {
        return Err(invalid("codec requires a nonempty BF16 layout"));
    }
    if spec != &chunk.codec
        || chunk.layers.start >= chunk.layers.end
        || chunk.layers.end > layout.num_layers
        || chunk.tokens.start >= chunk.tokens.end
        || chunk.tokens.end > seq_len
    {
        return Err(invalid("chunk codec or rectangle does not match cache"));
    }
    if !chunk
        .frame_len
        .is_multiple_of(bonsai_core::wire::FRAME_ALIGN)
        || chunk.frame_len != frame_len as u64
    {
        return Err(invalid("frame length differs from chunk.frame_len"));
    }
    let (k, v) = spec.section_bytes(
        layout,
        (chunk.layers.end - chunk.layers.start) as u64,
        (chunk.tokens.end - chunk.tokens.start) as u64,
    );
    let sections = [
        chunk.sections.k_payload,
        chunk.sections.v_payload,
        chunk.sections.k_meta,
        chunk.sections.v_meta,
    ];
    for (s, expected) in sections.iter().zip([k.payload, v.payload, k.meta, v.meta]) {
        if s.offset % bonsai_core::wire::FRAME_ALIGN != 0
            || s.len != expected
            || s.offset
                .checked_add(s.len)
                .is_none_or(|end| end > chunk.frame_len)
        {
            return Err(invalid("invalid frame section length or bounds"));
        }
    }
    for (i, a) in sections.iter().enumerate() {
        for b in sections.iter().skip(i + 1) {
            if a.len != 0
                && b.len != 0
                && a.offset < b.offset + b.len
                && b.offset < a.offset + a.len
            {
                return Err(invalid("overlapping frame sections"));
            }
        }
    }
    Ok(())
}

fn validate(spec: &CodecSpec, kv: &HostKv, chunk: &ChunkDesc, frame_len: usize) -> Result<()> {
    validate_frame(spec, &kv.layout, kv.seq_len, chunk, frame_len)?;
    let n = kv.layout.elems(1, kv.seq_len as u64) as usize;
    if kv.k.len() != kv.layout.num_layers as usize
        || kv.v.len() != kv.k.len()
        || kv.k.iter().chain(&kv.v).any(|layer| layer.len() != n)
    {
        return Err(invalid("host KV layer buffer does not match layout"));
    }
    Ok(())
}

fn metadata(x: f32) -> Result<f16> {
    let h = f16::from_f32(x);
    if !x.is_finite() || !h.is_finite() {
        Err(invalid("quantization metadata exceeds finite f16 range"))
    } else {
        Ok(h)
    }
}
fn read_half(frame: &[u8], at: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([frame[at], frame[at + 1]])).to_f32()
}

/// Returns the flattened group and position of a payload element.
fn group_of(i: usize, tokens: usize, dim: usize, p: QuantParams) -> usize {
    let d = i % dim;
    let t = (i / dim) % tokens;
    let lh = i / (tokens * dim);
    let g = p.group_size as usize;
    match p.axis {
        GroupAxis::HeadDim => (lh * tokens + t) * dim.div_ceil(g) + d / g,
        GroupAxis::Token => (lh * tokens.div_ceil(g) + t / g) * dim + d,
    }
}

pub fn encode_chunk(
    spec: &CodecSpec,
    src: &HostKv,
    chunk: &ChunkDesc,
    frame: &mut [u8],
) -> Result<()> {
    validate(spec, src, chunk, frame.len())?;
    frame.fill(0);
    let n = src.layout.elems(
        (chunk.layers.end - chunk.layers.start) as u64,
        (chunk.tokens.end - chunk.tokens.start) as u64,
    ) as usize;
    let tokens = (chunk.tokens.end - chunk.tokens.start) as usize;
    let dim = src.layout.head_dim as usize;
    for (source, payload, meta, p) in [
        (
            &src.k,
            chunk.sections.k_payload,
            chunk.sections.k_meta,
            match spec {
                CodecSpec::Quant { key, .. } => Some(*key),
                _ => None,
            },
        ),
        (
            &src.v,
            chunk.sections.v_payload,
            chunk.sections.v_meta,
            match spec {
                CodecSpec::Quant { value, .. } => Some(*value),
                _ => None,
            },
        ),
    ] {
        let mut values = Vec::with_capacity(n);
        for l in chunk.layers.clone() {
            for h in 0..src.layout.num_kv_heads as usize {
                for t in chunk.tokens.clone() {
                    let at = (h * src.seq_len as usize + t as usize) * dim;
                    values.extend_from_slice(&source[l as usize][at..at + dim]);
                }
            }
        }
        let base = payload.offset as usize;
        if let Some(p) = p {
            let bits = match spec {
                CodecSpec::Quant { bits, .. } => bits.bits() as u32,
                _ => unreachable!(),
            };
            let groups = meta.len as usize / p.meta_bytes_per_group() as usize;
            let mut lo = vec![f32::INFINITY; groups];
            let mut hi = vec![f32::NEG_INFINITY; groups];
            for (i, &x) in values.iter().enumerate() {
                let g = group_of(i, tokens, dim, p);
                let x = bf16::from_bits(x).to_f32();
                if !x.is_finite() {
                    return Err(invalid("nonfinite quantization input"));
                }
                lo[g] = lo[g].min(x);
                hi[g] = hi[g].max(x);
            }
            let qmax = if p.symmetric {
                (1u32 << (bits - 1)) - 1
            } else {
                (1u32 << bits) - 1
            };
            for g in 0..groups {
                let scale = metadata(if p.symmetric {
                    lo[g].abs().max(hi[g].abs()) / qmax as f32
                } else {
                    (hi[g] - lo[g]) / qmax as f32
                })?;
                let at = meta.offset as usize + g * p.meta_bytes_per_group() as usize;
                frame[at..at + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
                if !p.symmetric {
                    frame[at + 2..at + 4]
                        .copy_from_slice(&metadata(lo[g])?.to_bits().to_le_bytes());
                }
            }
            for (i, &x) in values.iter().enumerate() {
                let at = meta.offset as usize
                    + group_of(i, tokens, dim, p) * p.meta_bytes_per_group() as usize;
                let s = read_half(frame, at);
                let z = if p.symmetric {
                    0.0
                } else {
                    read_half(frame, at + 2)
                };
                let q = if s == 0.0 {
                    0
                } else {
                    ((bf16::from_bits(x).to_f32() - z) / s)
                        .round_ties_even()
                        .clamp(if p.symmetric { -(qmax as f32) } else { 0.0 }, qmax as f32)
                        as i32
                };
                if bits == 8 {
                    frame[base + i] = q as u8;
                } else {
                    frame[base + i / 2] |= ((q as u8) & 15) << ((i % 2) * 4);
                }
            }
        } else {
            for (i, x) in values.iter().enumerate() {
                frame[base + 2 * i..base + 2 * i + 2].copy_from_slice(&x.to_le_bytes());
            }
        }
    }
    Ok(())
}

pub fn decode_chunk(
    spec: &CodecSpec,
    frame: &[u8],
    chunk: &ChunkDesc,
    dst: &mut HostKv,
) -> Result<()> {
    validate(spec, dst, chunk, frame.len())?;
    // Reject malformed metadata before modifying any destination elements.
    for (m, p) in match spec {
        CodecSpec::Bf16 => Vec::new(),
        CodecSpec::Quant { key, value, .. } => vec![
            (chunk.sections.k_meta, *key),
            (chunk.sections.v_meta, *value),
        ],
    } {
        for at in (m.offset as usize..(m.offset + m.len) as usize)
            .step_by(p.meta_bytes_per_group() as usize)
        {
            if !read_half(frame, at).is_finite()
                || read_half(frame, at) < 0.0
                || (!p.symmetric && !read_half(frame, at + 2).is_finite())
            {
                return Err(invalid("invalid quantization metadata"));
            }
        }
    }
    let tokens = (chunk.tokens.end - chunk.tokens.start) as usize;
    let dim = dst.layout.head_dim as usize;
    for (destination, payload, meta, p) in [
        (
            &mut dst.k,
            chunk.sections.k_payload,
            chunk.sections.k_meta,
            match spec {
                CodecSpec::Quant { key, .. } => Some(*key),
                _ => None,
            },
        ),
        (
            &mut dst.v,
            chunk.sections.v_payload,
            chunk.sections.v_meta,
            match spec {
                CodecSpec::Quant { value, .. } => Some(*value),
                _ => None,
            },
        ),
    ] {
        let mut i = 0;
        for l in chunk.layers.clone() {
            for h in 0..dst.layout.num_kv_heads as usize {
                for t in chunk.tokens.clone() {
                    for d in 0..dim {
                        let base = payload.offset as usize;
                        let x = if let Some(p) = p {
                            let bits = match spec {
                                CodecSpec::Quant { bits, .. } => *bits,
                                _ => unreachable!(),
                            };
                            let raw = if bits == QuantBits::Int8 {
                                frame[base + i]
                            } else {
                                (frame[base + i / 2] >> ((i % 2) * 4)) & 15
                            };
                            let q = if p.symmetric {
                                if bits == QuantBits::Int8 {
                                    raw as i8 as i32
                                } else {
                                    (((raw as i8) << 4) >> 4) as i32
                                }
                            } else {
                                raw as i32
                            };
                            let at = meta.offset as usize
                                + group_of(i, tokens, dim, p) * p.meta_bytes_per_group() as usize;
                            let s = read_half(frame, at);
                            let z = if p.symmetric {
                                0.0
                            } else {
                                read_half(frame, at + 2)
                            };
                            bf16::from_f32((q as f32 * s) + z).to_bits()
                        } else {
                            u16::from_le_bytes([frame[base + 2 * i], frame[base + 2 * i + 1]])
                        };
                        destination[l as usize]
                            [(h * dst.seq_len as usize + t as usize) * dim + d] = x;
                        i += 1;
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ErrorStats {
    pub max_abs: f32,
    pub mse: f64,
    pub sqnr_db: f64,
}

/// Panics for incompatible cache shapes. NaN inputs produce NaN statistics.
pub fn compare(a: &HostKv, b: &HostKv) -> ErrorStats {
    assert_eq!(a.layout, b.layout);
    assert_eq!(a.seq_len, b.seq_len);
    assert_eq!(a.k.len(), b.k.len());
    assert_eq!(a.v.len(), b.v.len());
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut n = 0usize;
    for (aa, bb) in a.k.iter().chain(&a.v).zip(b.k.iter().chain(&b.v)) {
        assert_eq!(aa.len(), bb.len());
        for (&x, &y) in aa.iter().zip(bb) {
            let x = bf16::from_bits(x).to_f32();
            let y = bf16::from_bits(y).to_f32();
            let e = (x as f64) - (y as f64);
            if !e.is_finite() {
                return ErrorStats {
                    max_abs: f32::NAN,
                    mse: f64::NAN,
                    sqnr_db: f64::NAN,
                };
            }
            signal += (x as f64).powi(2);
            noise += e * e;
            max_abs = max_abs.max(e.abs() as f32);
            n += 1;
        }
    }
    ErrorStats {
        max_abs,
        mse: if n == 0 { 0.0 } else { noise / n as f64 },
        sqnr_db: if noise == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (signal / noise).log10()
        },
    }
}
