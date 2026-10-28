//! Shared vocabulary for the Bonsai suite.
//!
//! Nothing in this crate touches CUDA, RDMA, or sockets, so both planes (and the
//! Python bindings) can depend on it and it unit-tests on any machine.

pub mod codec_spec;
pub mod device;
pub mod error;
pub mod ids;
pub mod layout;
pub mod metrics;
pub mod msg;
pub mod wire;

pub use codec_spec::{CodecSpec, GroupAxis, LayerOverride, PrecisionMap, QuantBits, QuantParams};
pub use device::{CudaStream, DevicePtr, KvSink, KvSource, LayerKv};
pub use error::{BonsaiError, Result};
pub use ids::{HandoffId, ImmTag, RequestId, SlotId, WorkerId, WorkerRole};
pub use layout::{ElemType, KvLayout};
pub use metrics::{HandoffReport, ReceiverTimings, SenderTimings};
pub use wire::{ChunkDesc, ChunkPlan, ChunkSpec, FrameLayout, HandoffTrailer, Section};
