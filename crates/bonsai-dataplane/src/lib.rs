//! Bonsai data plane.
//!
//! ```text
//!  prefill GPU                                         decode GPU
//!  ───────────                                         ──────────
//!  HF DynamicCache ──encode_chunk──▶ StagingRing ──RDMA WRITE+imm──▶ ReceiveArena slot
//!   (per-layer BF16)   (CUDA stream)   (registered)     (fabric-lib)     (registered)
//!                                                                         │ per-group imm count
//!                                                                         ▼
//!                                                         decode_chunk ──▶ new BF16 tensors
//!                                                                         ──▶ DynamicCache ──▶ generate()
//! ```
//!
//! The data plane never talks to the coordinator. It is driven by the worker
//! process (Python via `bonsai-py`), which gets `HandoffParams` / `SlotGrant`
//! from the control plane and passes them in.

pub mod arena;
pub mod config;
pub mod gpu;
pub mod receiver;
pub mod sender;
pub mod staging;
pub mod throttle;

pub use arena::{ReceiveArena, SlotLease};
pub use config::{DataPlaneConfig, StagingMode};
#[cfg(feature = "cuda")]
pub use gpu::CudaGpu;
pub use gpu::{DeviceBuffer, Gpu, HostGpu};
pub use receiver::{KvReceiver, PreparedHandoff, ReceivedHandoff};
pub use sender::{KvSender, SendJob};
pub use staging::{StagingLease, StagingRing};
pub use throttle::Throttle;

/// Raw views share a single stride tuple between K and V. Callers must verify
/// K/V strides agree before constructing `LayerKv` (the view erases separate strides).
fn validate_view(
    view: &bonsai_core::KvSource,
    layout: bonsai_core::KvLayout,
    seq_len: u32,
    device: u8,
) -> bonsai_core::Result<()> {
    use bonsai_core::BonsaiError;
    if view.layout != layout
        || view.seq_len != seq_len
        || view.device != device
        || view.layers.len() != layout.num_layers as usize
    {
        return Err(BonsaiError::InvalidConfig(
            "KV view shape/device differs from handoff".into(),
        ));
    }
    for layer in &view.layers {
        if layer.stride_dim == 0
            || layer.stride_token == 0
            || layer.stride_head == 0
            || layer.k.0 == 0
            || layer.v.0 == 0
        {
            return Err(BonsaiError::InvalidConfig(
                "KV view has null pointers/zero strides".into(),
            ));
        }
        let span = (layout.num_kv_heads as u64)
            .checked_sub(1)
            .and_then(|n| n.checked_mul(layer.stride_head))
            .and_then(|n| {
                (seq_len as u64)
                    .checked_sub(1)
                    .and_then(|s| s.checked_mul(layer.stride_token))
                    .and_then(|s| n.checked_add(s))
            })
            .and_then(|n| {
                (layout.head_dim as u64)
                    .checked_sub(1)
                    .and_then(|d| d.checked_mul(layer.stride_dim))
                    .and_then(|d| n.checked_add(d))
            })
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(layout.dtype.size()));
        if span.is_none_or(|n| {
            layer.k.0.checked_add(n).is_none() || layer.v.0.checked_add(n).is_none()
        }) {
            return Err(BonsaiError::InvalidConfig(
                "KV view address span overflows".into(),
            ));
        }
    }
    Ok(())
}
