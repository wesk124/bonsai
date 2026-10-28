//! CPU reference codec. Defines the numerics the CUDA kernels must match.
//!
//! Quantization rules (per group of values `x`):
//!
//! * Symmetric INT-b:  `s = max|x| / qmax`, `q = clamp(round(x / s), -qmax, qmax)`,
//!   `x' = q * s`, with `qmax = 2^(b-1) - 1` (127 for INT8, 7 for INT4).
//! * Asymmetric INT-b: `s = (max - min) / (2^b - 1)`, `z = min`,
//!   `q = clamp(round((x - z) / s), 0, 2^b - 1)`, `x' = q * s + z`.
//! * `s` and `z` are rounded to f16 **before** quantizing, so encoder and decoder
//!   use the same values. A group with `s == 0` stores `q = 0` (or `z` only).
//! * Math is done in f32; inputs/outputs are BF16.
//! * INT4 packs element `2i` in the low nibble and `2i+1` in the high nibble.
//!   Asymmetric INT4 is stored unsigned; symmetric INT4 is stored as two's complement.

use bonsai_core::{ChunkDesc, CodecSpec, KvLayout, Result};

/// Host copy of a KV cache. BF16 values are stored as raw `u16` bits.
/// Per layer, K and V are contiguous `[num_kv_heads, seq_len, head_dim]`.
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
        HostKv { layout, seq_len, k: vec![vec![0; n]; l], v: vec![vec![0; n]; l] }
    }
}

/// Encode `chunk` of `src` into `frame` (`frame.len() == chunk.frame_len`).
pub fn encode_chunk(_spec: &CodecSpec, _src: &HostKv, _chunk: &ChunkDesc, _frame: &mut [u8]) -> Result<()> {
    todo!("CPU reference encode (milestone 2)")
}

/// Decode `frame` into the `chunk` rectangle of `dst`.
pub fn decode_chunk(_spec: &CodecSpec, _frame: &[u8], _chunk: &ChunkDesc, _dst: &mut HostKv) -> Result<()> {
    todo!("CPU reference decode (milestone 2)")
}

/// Error statistics between an original and a round-tripped cache, used to
/// sanity-check codecs before running task-quality evals.
#[derive(Debug, Clone, Copy, Default)]
pub struct ErrorStats {
    pub max_abs: f32,
    pub mse: f64,
    /// Signal-to-quantization-noise ratio in dB.
    pub sqnr_db: f64,
}

pub fn compare(_a: &HostKv, _b: &HostKv) -> ErrorStats {
    todo!()
}
