//! Thin, FFI-friendly descriptions of GPU memory.
//!
//! These are plain integers on purpose: the Python side hands us
//! `tensor.data_ptr()` and `torch.cuda.current_stream().cuda_stream`, and the
//! codec/dataplane crates turn them back into typed CUDA objects.

use serde::{Deserialize, Serialize};

use crate::layout::KvLayout;

/// A raw device (or pinned host) address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    pub fn offset(self, bytes: u64) -> DevicePtr {
        DevicePtr(self.0 + bytes)
    }
}

/// A raw `cudaStream_t`. `0` means the legacy default stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaStream(pub u64);

/// K and V tensors of one layer, `[1, num_kv_heads, seq, head_dim]`.
///
/// Strides are in elements, like `torch.Tensor.stride()`, so non-contiguous
/// views (e.g. a slice of a pre-allocated static cache) work without a copy.
#[derive(Debug, Clone, Copy)]
pub struct LayerKv {
    pub k: DevicePtr,
    pub v: DevicePtr,
    /// Strides for (head, token, channel).
    pub stride_head: u64,
    pub stride_token: u64,
    pub stride_dim: u64,
}

/// The sender's view of a prefilled KV cache (borrowed from HF `DynamicCache`).
#[derive(Debug, Clone)]
pub struct KvSource {
    pub layout: KvLayout,
    pub seq_len: u32,
    pub device: u8,
    pub layers: Vec<LayerKv>,
}

/// Where the receiver writes reconstructed BF16 K/V. Same shape as [`KvSource`];
/// typically freshly allocated torch tensors that become a new `DynamicCache`.
pub type KvSink = KvSource;
