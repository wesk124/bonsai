//! What encoding a handoff uses, and exactly how many bytes it costs.
//!
//! Byte accounting lives here (not in the codec crate) because the control plane
//! needs it to size receive slots before any GPU work happens.

use serde::{Deserialize, Serialize};

use crate::{
    error::{BonsaiError, Result},
    layout::KvLayout,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantBits {
    Int8,
    /// Two 4-bit values per byte; element `2i` in the low nibble, `2i+1` in the high nibble.
    Int4Packed,
}

impl QuantBits {
    pub const fn bits(self) -> u64 {
        match self {
            QuantBits::Int8 => 8,
            QuantBits::Int4Packed => 4,
        }
    }
}

/// Which axis a quantization group runs along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GroupAxis {
    /// Groups of `group_size` contiguous channels within one (layer, head, token).
    /// "Per-token" quantization. Works with any chunking.
    HeadDim,
    /// Groups of `group_size` consecutive tokens for one (layer, head, channel).
    /// "Per-channel" quantization (KIVI-style, good for K outliers).
    /// Chunk token boundaries must be multiples of `group_size`.
    Token,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuantParams {
    pub group_size: u16,
    pub axis: GroupAxis,
    /// Symmetric stores only a scale; asymmetric also stores a zero point.
    pub symmetric: bool,
}

impl QuantParams {
    /// Bytes of metadata per group. Scales and zero points are stored as f16.
    pub const fn meta_bytes_per_group(&self) -> u64 {
        if self.symmetric { 2 } else { 4 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodecSpec {
    /// Uncompressed baseline. Still packed into frames so it shares the same pipeline.
    Bf16,
    /// Quantized. K and V get separate parameters since their distributions differ.
    Quant { bits: QuantBits, key: QuantParams, value: QuantParams },
}

/// Bytes of one encoded K-or-V section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SectionBytes {
    pub payload: u64,
    pub meta: u64,
}

impl CodecSpec {
    /// INT8, symmetric, one f16 scale per (layer, head, token) over all 128 channels.
    pub fn int8_default(layout: &KvLayout) -> Self {
        let p = QuantParams { group_size: layout.head_dim, axis: GroupAxis::HeadDim, symmetric: true };
        CodecSpec::Quant { bits: QuantBits::Int8, key: p, value: p }
    }

    /// INT4 asymmetric, group 32. K per-channel (Token axis), V per-token (HeadDim axis).
    pub fn int4_default() -> Self {
        CodecSpec::Quant {
            bits: QuantBits::Int4Packed,
            key: QuantParams { group_size: 32, axis: GroupAxis::Token, symmetric: false },
            value: QuantParams { group_size: 32, axis: GroupAxis::HeadDim, symmetric: false },
        }
    }

    pub fn name(&self) -> String {
        match self {
            CodecSpec::Bf16 => "bf16".into(),
            CodecSpec::Quant { bits, key, value } => format!(
                "{}-k{}{}{}-v{}{}{}",
                match bits {
                    QuantBits::Int8 => "int8",
                    QuantBits::Int4Packed => "int4",
                },
                axis_tag(key.axis),
                key.group_size,
                if key.symmetric { "s" } else { "a" },
                axis_tag(value.axis),
                value.group_size,
                if value.symmetric { "s" } else { "a" },
            ),
        }
    }

    pub fn validate(&self, layout: &KvLayout) -> Result<()> {
        if let CodecSpec::Quant { key, value, .. } = self {
            for p in [key, value] {
                if p.group_size == 0 {
                    return Err(BonsaiError::InvalidConfig("group_size must be > 0".into()));
                }
                if p.axis == GroupAxis::HeadDim && layout.head_dim % p.group_size != 0 {
                    return Err(BonsaiError::InvalidConfig(format!(
                        "head_dim {} not divisible by group_size {}",
                        layout.head_dim, p.group_size
                    )));
                }
            }
        }
        Ok(())
    }

    /// Token granularity a chunk boundary must respect.
    pub fn token_alignment(&self) -> u32 {
        match self {
            CodecSpec::Bf16 => 1,
            CodecSpec::Quant { key, value, .. } => {
                let a = |p: &QuantParams| if p.axis == GroupAxis::Token { p.group_size as u32 } else { 1 };
                lcm(a(key), a(value))
            }
        }
    }

    /// Bytes for the K section and V section of a `layers` x `tokens` slice.
    pub fn section_bytes(&self, layout: &KvLayout, layers: u64, tokens: u64) -> (SectionBytes, SectionBytes) {
        match self {
            CodecSpec::Bf16 => {
                let b = SectionBytes { payload: layout.elems(layers, tokens) * layout.dtype.size(), meta: 0 };
                (b, b)
            }
            CodecSpec::Quant { bits, key, value } => {
                let one = |p: &QuantParams| {
                    let elems = layout.elems(layers, tokens);
                    let heads = layers * layout.num_kv_heads as u64;
                    let groups = match p.axis {
                        GroupAxis::HeadDim => heads * tokens * (layout.head_dim as u64).div_ceil(p.group_size as u64),
                        GroupAxis::Token => heads * layout.head_dim as u64 * tokens.div_ceil(p.group_size as u64),
                    };
                    SectionBytes {
                        payload: (elems * bits.bits()).div_ceil(8),
                        meta: groups * p.meta_bytes_per_group(),
                    }
                };
                (one(key), one(value))
            }
        }
    }

    /// Total encoded bytes (payload + metadata, no frame padding) for a whole sequence.
    pub fn encoded_bytes(&self, layout: &KvLayout, seq_len: u64) -> u64 {
        let (k, v) = self.section_bytes(layout, layout.num_layers as u64, seq_len);
        k.payload + k.meta + v.payload + v.meta
    }
}

fn axis_tag(a: GroupAxis) -> &'static str {
    match a {
        GroupAxis::HeadDim => "t",
        GroupAxis::Token => "c",
    }
}

fn lcm(a: u32, b: u32) -> u32 {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    a / gcd(a, b) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: KvLayout = KvLayout::QWEN2_5_1_5B;

    #[test]
    fn qwen_bytes_per_token() {
        // 28 layers * 2 (K,V) * 2 heads * 128 dim * 2 bytes
        assert_eq!(L.raw_bytes_per_token(), 28_672);
        assert_eq!(CodecSpec::Bf16.encoded_bytes(&L, 1), 28_672);
        // INT8 sym g128: (128 B payload + 2 B scale) per head-token
        assert_eq!(CodecSpec::int8_default(&L).encoded_bytes(&L, 1), 28 * 2 * 2 * 130);
    }

    #[test]
    fn int4_ratio_at_8k() {
        let seq = 8192;
        let raw = CodecSpec::Bf16.encoded_bytes(&L, seq);
        let int4 = CodecSpec::int4_default().encoded_bytes(&L, seq);
        // payload is 1/4 of bf16; asym g32 metadata adds 4 B per 32 elems = 1/16 of bf16
        assert_eq!(int4 * 16, raw * 5);
    }

    #[test]
    fn token_alignment() {
        assert_eq!(CodecSpec::Bf16.token_alignment(), 1);
        assert_eq!(CodecSpec::int8_default(&L).token_alignment(), 1);
        assert_eq!(CodecSpec::int4_default().token_alignment(), 32);
    }
}

/// Codec per layer: the unit of **adaptive precision**.
///
/// Layers differ a lot in how much quantization hurts (early layers and
/// layers with large K outliers are the usual sensitive ones). A map keeps
/// sensitive layers at higher precision and pushes the rest lower, for example
/// INT8 on layers 0-1 and 26-27 and INT4 elsewhere. The map is picked per request
/// by a `CodecPolicy` in the control plane (see `docs/HLD.md` §6). The chunk
/// planner never puts two codecs in one chunk.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrecisionMap {
    pub default: CodecSpec,
    /// Later entries win where ranges overlap.
    pub overrides: Vec<LayerOverride>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerOverride {
    pub layers: std::ops::Range<u16>,
    pub codec: CodecSpec,
}

impl From<CodecSpec> for PrecisionMap {
    fn from(c: CodecSpec) -> Self {
        PrecisionMap::uniform(c)
    }
}

impl PrecisionMap {
    pub fn uniform(codec: CodecSpec) -> Self {
        PrecisionMap { default: codec, overrides: Vec::new() }
    }

    /// `hi` on the listed layers, `lo` everywhere else.
    pub fn protect_layers(lo: CodecSpec, hi: CodecSpec, layers: &[std::ops::Range<u16>]) -> Self {
        PrecisionMap {
            default: lo,
            overrides: layers.iter().map(|r| LayerOverride { layers: r.clone(), codec: hi }).collect(),
        }
    }

    pub fn codec_for_layer(&self, layer: u16) -> CodecSpec {
        self.overrides.iter().rev().find(|o| o.layers.contains(&layer)).map(|o| o.codec).unwrap_or(self.default)
    }

    pub fn is_uniform(&self, codec: CodecSpec) -> bool {
        self.default == codec && self.overrides.iter().all(|o| o.codec == codec || o.layers.is_empty())
    }

    /// Maximal runs of consecutive layers that share a codec.
    pub fn segments(&self, num_layers: u16) -> Vec<(std::ops::Range<u16>, CodecSpec)> {
        let mut out: Vec<(std::ops::Range<u16>, CodecSpec)> = Vec::new();
        for l in 0..num_layers {
            let c = self.codec_for_layer(l);
            match out.last_mut() {
                Some((r, prev)) if *prev == c => r.end = l + 1,
                _ => out.push((l..l + 1, c)),
            }
        }
        out
    }

    pub fn validate(&self, layout: &KvLayout) -> Result<()> {
        self.segments(layout.num_layers).iter().try_for_each(|(_, c)| c.validate(layout))
    }

    pub fn token_alignment(&self, layout: &KvLayout) -> u32 {
        self.segments(layout.num_layers).iter().fold(1, |a, (_, c)| lcm(a, c.token_alignment()))
    }

    pub fn encoded_bytes(&self, layout: &KvLayout, seq_len: u64) -> u64 {
        self.segments(layout.num_layers)
            .iter()
            .map(|(r, c)| {
                let (k, v) = c.section_bytes(layout, (r.end - r.start) as u64, seq_len);
                k.payload + k.meta + v.payload + v.meta
            })
            .sum()
    }

    pub fn name(&self, layout: &KvLayout) -> String {
        let segs = self.segments(layout.num_layers);
        if segs.len() == 1 {
            return segs[0].1.name();
        }
        segs.iter().map(|(r, c)| format!("L{}-{}:{}", r.start, r.end - 1, c.name())).collect::<Vec<_>>().join(",")
    }
}

#[cfg(test)]
mod precision_tests {
    use super::*;

    #[test]
    fn mixed_map_segments_and_bytes() {
        let l = KvLayout::QWEN2_5_1_5B;
        let int8 = CodecSpec::int8_default(&l);
        let int4 = CodecSpec::int4_default();
        let m = PrecisionMap::protect_layers(int4, int8, &[0..2, 26..28]);
        let segs = m.segments(28);
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[1], (2..26, int4));
        // 4 INT8 layers + 24 INT4 layers
        let per_layer = |c: CodecSpec| c.encoded_bytes(&l, 1024) / 28;
        assert_eq!(m.encoded_bytes(&l, 1024), 4 * per_layer(int8) + 24 * per_layer(int4));
        assert_eq!(m.token_alignment(&l), 32);
    }
}
