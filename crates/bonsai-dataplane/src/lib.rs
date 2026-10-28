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
pub use gpu::{DeviceBuffer, Gpu};
pub use receiver::{KvReceiver, PreparedHandoff, ReceivedHandoff};
pub use sender::{KvSender, SendJob};
pub use staging::{StagingLease, StagingRing};
pub use throttle::Throttle;
