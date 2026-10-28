//! The on-the-wire (and in-memory) format of an encoded handoff.
//!
//! A handoff is a sequence of **chunk frames** followed by a fixed-size
//! **trailer**, laid out back to back in the receiver's leased slot:
//!
//! ```text
//! slot base
//! ├── frame 0  [ K payload | V payload | K meta | V meta ]   (each section 256 B aligned)
//! ├── frame 1  [ ... ]
//! ├── ...
//! └── trailer  [ HandoffTrailer ]                              (fixed 256 B)
//! ```
//!
//! A chunk covers a rectangle of `layers x tokens`. Each frame goes out as one
//! RDMA WRITE-with-immediate, with its scales and zero points next to the payload,
//! so a frame can be decoded on its own once it lands. The trailer carries the
//! first sampled token and sender timings and is also counted by the immediate,
//! so the receiver needs one completion signal and no separate "done" message.
//!
//! Inside a section, data is ordered `[layer][head][token][channel]` (payload) and
//! `[layer][head][group]` (meta), with layer/token local to the chunk.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::{
    codec_spec::{CodecSpec, PrecisionMap},
    error::{BonsaiError, Result},
    ids::{ImmTag, SlotId},
    layout::KvLayout,
};

/// Section and frame alignment. Keeps kernels' vector loads aligned and gives
/// RDMA writes cache-line-friendly boundaries.
pub const FRAME_ALIGN: u64 = 256;
pub const TRAILER_BYTES: u64 = 256;

/// How the sender cuts a handoff into chunks. The main experiment knob for
/// "transfer chunk size".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkSpec {
    /// Tokens per chunk. `0` means the whole sequence.
    pub tokens_per_chunk: u32,
    /// Layers per chunk. `0` means all layers.
    pub layers_per_chunk: u16,
}

impl ChunkSpec {
    pub const WHOLE: ChunkSpec = ChunkSpec { tokens_per_chunk: 0, layers_per_chunk: 0 };
    pub const PER_LAYER: ChunkSpec = ChunkSpec { tokens_per_chunk: 0, layers_per_chunk: 1 };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Section {
    /// Offset relative to the start of the frame.
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct FrameLayout {
    pub k_payload: Section,
    pub v_payload: Section,
    pub k_meta: Section,
    pub v_meta: Section,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkDesc {
    pub index: u32,
    pub layers: Range<u16>,
    pub tokens: Range<u32>,
    /// Offset of the frame from the slot base (and from the staging buffer base
    /// when the sender stages the whole handoff).
    pub frame_offset: u64,
    /// Frame size including alignment padding; the RDMA write length.
    pub frame_len: u64,
    pub sections: FrameLayout,
    /// Codec of every layer in this chunk (chunks never mix codecs).
    pub codec: CodecSpec,
    /// Completion group this chunk counts toward (see [`ImmTag`]).
    pub group: u16,
}

impl ChunkDesc {
    /// Payload + metadata bytes, excluding padding.
    pub fn useful_bytes(&self) -> u64 {
        let s = &self.sections;
        s.k_payload.len + s.v_payload.len + s.k_meta.len + s.v_meta.len
    }
}

/// The full plan for one handoff. Built deterministically from
/// `(layout, precision, seq_len, chunk_spec)`, so sender and receiver can each build
/// it locally and agree byte for byte; only those four inputs travel in messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkPlan {
    pub layout: KvLayout,
    pub precision: PrecisionMap,
    pub seq_len: u32,
    pub spec: ChunkSpec,
    pub chunks: Vec<ChunkDesc>,
    pub trailer_offset: u64,
    /// Bytes the receiver must lease: all frames + trailer.
    pub slot_bytes: u64,
    /// Number of completion groups (one per distinct layer range).
    /// Layer groups never cross a precision boundary.
    pub num_groups: u16,
}

impl ChunkPlan {
    /// Chunks are ordered layer-major (all token ranges of layer group 0, then
    /// group 1, ...), matching the order HF produces layers during prefill and
    /// the order the receiver wants to dequantize them.
    pub fn build(layout: KvLayout, precision: impl Into<PrecisionMap>, seq_len: u32, spec: ChunkSpec) -> Result<Self> {
        let precision = precision.into();
        precision.validate(&layout)?;
        if seq_len == 0 {
            return Err(BonsaiError::InvalidConfig("seq_len must be > 0".into()));
        }
        let tpc = if spec.tokens_per_chunk == 0 { seq_len } else { spec.tokens_per_chunk.min(seq_len) };
        let lpc = if spec.layers_per_chunk == 0 { layout.num_layers } else { spec.layers_per_chunk };
        let align = precision.token_alignment(&layout);
        if spec.tokens_per_chunk != 0 && tpc < seq_len && tpc % align != 0 {
            return Err(BonsaiError::InvalidConfig(format!(
                "tokens_per_chunk {tpc} must be a multiple of {align} for {}",
                precision.name(&layout)
            )));
        }
        // Layer groups: split each same-codec segment into runs of `lpc` layers.
        let groups: Vec<(Range<u16>, CodecSpec)> = precision
            .segments(layout.num_layers)
            .into_iter()
            .flat_map(|(seg, c)| {
                (seg.start..seg.end).step_by(lpc as usize).map(move |s| (s..(s + lpc).min(seg.end), c))
            })
            .collect();
        if groups.len() > ImmTag::MAX_GROUPS as usize {
            return Err(BonsaiError::InvalidConfig("too many layer groups".into()));
        }
        let num_groups = groups.len() as u16;

        let mut chunks = Vec::new();
        let mut offset = 0u64;
        for (g, (layers, codec)) in groups.into_iter().enumerate() {
            let g = g as u16;
            let mut t0 = 0u32;
            while t0 < seq_len {
                let tokens = t0..(t0 + tpc).min(seq_len);
                let (k, v) = codec.section_bytes(
                    &layout,
                    (layers.end - layers.start) as u64,
                    (tokens.end - tokens.start) as u64,
                );
                let mut cursor = 0u64;
                let mut place = |len: u64| {
                    let s = Section { offset: cursor, len };
                    cursor = align_up(cursor + len, FRAME_ALIGN);
                    s
                };
                let sections = FrameLayout {
                    k_payload: place(k.payload),
                    v_payload: place(v.payload),
                    k_meta: place(k.meta),
                    v_meta: place(v.meta),
                };
                chunks.push(ChunkDesc {
                    index: chunks.len() as u32,
                    layers: layers.clone(),
                    tokens: tokens.clone(),
                    frame_offset: offset,
                    frame_len: cursor,
                    sections,
                    codec,
                    group: g,
                });
                offset += cursor;
                t0 = tokens.end;
            }
        }
        Ok(ChunkPlan {
            layout,
            precision,
            seq_len,
            spec,
            chunks,
            trailer_offset: offset,
            slot_bytes: offset + TRAILER_BYTES,
            num_groups,
        })
    }

    /// Bytes that actually cross the network (frames incl. padding + trailer).
    pub fn wire_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// Writes that will land for completion group `g` (the trailer counts toward
    /// the last group).
    pub fn writes_in_group(&self, g: u16) -> u32 {
        let n = self.chunks.iter().filter(|c| c.group == g).count() as u32;
        if g + 1 == self.num_groups { n + 1 } else { n }
    }

    pub fn imm_tag(&self, slot: SlotId, g: u16) -> ImmTag {
        ImmTag::new(slot, g)
    }

    pub fn largest_frame(&self) -> u64 {
        self.chunks.iter().map(|c| c.frame_len).max().unwrap_or(0)
    }
}

/// Written by the sender at `ChunkPlan::trailer_offset` as the last frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandoffTrailer {
    pub magic: u32,
    /// The token the prefill worker sampled from the final prompt position.
    /// The decode worker feeds it as its first input.
    pub first_token: u32,
    /// Sender clock: prefill done -> last frame posted, in ns.
    pub sender_encode_ns: u64,
    pub sender_transfer_ns: u64,
    /// Checksum over frame bytes in debug builds; 0 when disabled.
    pub checksum: u64,
}

impl HandoffTrailer {
    pub const MAGIC: u32 = 0xB0A5_A1D0;

    /// Fixed little-endian encoding, zero-padded to `TRAILER_BYTES`.
    pub fn to_bytes(&self) -> [u8; TRAILER_BYTES as usize] {
        let mut b = [0u8; TRAILER_BYTES as usize];
        b[0..4].copy_from_slice(&self.magic.to_le_bytes());
        b[4..8].copy_from_slice(&self.first_token.to_le_bytes());
        b[8..16].copy_from_slice(&self.sender_encode_ns.to_le_bytes());
        b[16..24].copy_from_slice(&self.sender_transfer_ns.to_le_bytes());
        b[24..32].copy_from_slice(&self.checksum.to_le_bytes());
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() < 32 {
            return Err(BonsaiError::Protocol("trailer too short".into()));
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let t = HandoffTrailer {
            magic: u32_at(0),
            first_token: u32_at(4),
            sender_encode_ns: u64_at(8),
            sender_transfer_ns: u64_at(16),
            checksum: u64_at(24),
        };
        if t.magic != Self::MAGIC {
            return Err(BonsaiError::Protocol(format!("bad trailer magic {:#x}", t.magic)));
        }
        Ok(t)
    }
}

pub const fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: KvLayout = KvLayout::QWEN2_5_1_5B;

    #[test]
    fn whole_plan_is_one_frame() {
        let p = ChunkPlan::build(L, CodecSpec::Bf16, 1000, ChunkSpec::WHOLE).unwrap();
        assert_eq!(p.chunks.len(), 1);
        assert_eq!(p.num_groups, 1);
        assert_eq!(p.chunks[0].useful_bytes(), 28_672 * 1000);
        assert_eq!(p.writes_in_group(0), 2);
    }

    #[test]
    fn per_layer_chunking_covers_everything() {
        let codec = CodecSpec::int8_default(&L);
        let p = ChunkPlan::build(L, codec, 4096, ChunkSpec { tokens_per_chunk: 1024, layers_per_chunk: 1 }).unwrap();
        assert_eq!(p.chunks.len(), 28 * 4);
        assert_eq!(p.num_groups, 28);
        let useful: u64 = p.chunks.iter().map(ChunkDesc::useful_bytes).sum();
        assert_eq!(useful, codec.encoded_bytes(&L, 4096));
        // frames are contiguous and aligned
        for w in p.chunks.windows(2) {
            assert_eq!(w[0].frame_offset + w[0].frame_len, w[1].frame_offset);
            assert_eq!(w[1].frame_offset % FRAME_ALIGN, 0);
        }
    }

    #[test]
    fn mixed_precision_groups_split_at_boundaries() {
        let int8 = CodecSpec::int8_default(&L);
        let m = PrecisionMap::protect_layers(CodecSpec::int4_default(), int8, &[0..2, 26..28]);
        // 4 layers per group: segments 0-1 | 2-25 | 26-27 -> 1 + 6 + 1 groups
        let p = ChunkPlan::build(L, m.clone(), 2048, ChunkSpec { tokens_per_chunk: 1024, layers_per_chunk: 4 }).unwrap();
        assert_eq!(p.num_groups, 8);
        assert!(p.chunks.iter().all(|c| c.codec == m.codec_for_layer(c.layers.start)
            && c.codec == m.codec_for_layer(c.layers.end - 1)));
        let useful: u64 = p.chunks.iter().map(ChunkDesc::useful_bytes).sum();
        assert_eq!(useful, m.encoded_bytes(&L, 2048));
    }

    #[test]
    fn int4_rejects_misaligned_chunks() {
        let spec = ChunkSpec { tokens_per_chunk: 100, layers_per_chunk: 0 };
        assert!(ChunkPlan::build(L, CodecSpec::int4_default(), 4096, spec).is_err());
    }
}
