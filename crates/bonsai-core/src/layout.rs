use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElemType {
    Bf16,
    F16,
    F32,
}

impl ElemType {
    pub const fn size(self) -> u64 {
        match self {
            ElemType::Bf16 | ElemType::F16 => 2,
            ElemType::F32 => 4,
        }
    }
}

/// Shape of a model's KV cache, independent of sequence length.
///
/// Per layer, HF Transformers stores K and V as `[batch=1, num_kv_heads, seq, head_dim]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvLayout {
    pub num_layers: u16,
    pub num_kv_heads: u16,
    pub head_dim: u16,
    pub dtype: ElemType,
}

impl KvLayout {
    /// Qwen/Qwen2.5-1.5B-Instruct: 28 layers, GQA with 2 KV heads, head_dim 128, BF16.
    pub const QWEN2_5_1_5B: KvLayout =
        KvLayout { num_layers: 28, num_kv_heads: 2, head_dim: 128, dtype: ElemType::Bf16 };

    /// Elements in one K (or one V) tensor slice of `layers` x `tokens`.
    pub fn elems(&self, layers: u64, tokens: u64) -> u64 {
        layers * self.num_kv_heads as u64 * tokens * self.head_dim as u64
    }

    /// Uncompressed K+V bytes for one token across all layers.
    pub fn raw_bytes_per_token(&self) -> u64 {
        2 * self.elems(self.num_layers as u64, 1) * self.dtype.size()
    }
}
