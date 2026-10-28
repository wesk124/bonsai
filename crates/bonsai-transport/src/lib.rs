//! The small slice of RDMA that Bonsai needs, behind one trait.
//!
//! Why not use `fabric_lib::TransferEngine` directly everywhere?
//! * fabric-lib only builds on Linux with libfabric/libibverbs/CUDA/GDRCopy;
//!   hiding it behind [`Transport`] lets the dataplane and control plane build
//!   and unit-test on a laptop with [`loopback::LoopbackTransport`].
//! * It pins down the one subtle rule we depend on (see [`Transport::arm_imm`]).
//!
//! Operations used, and what they map to in fabric-lib:
//!
//! | Bonsai                    | fabric-lib                                                    |
//! |---------------------------|---------------------------------------------------------------|
//! | `register_source`         | `RdmaEngine::register_memory_local`                           |
//! | `register_sink`           | `RdmaEngine::register_memory_allow_remote` -> descriptor      |
//! | `write`                   | `TransferRequest::Single` (+ `imm_data`), `submit_transfer`   |
//! | `write_paged`             | `TransferRequest::Paged` (paged-attention caches, future)     |
//! | `arm_imm` / `ImmWait`     | `TransferEngine::set_imm_count_expected` + callback           |
//! | `disarm_imm`              | `TransferEngine::remove_imm_count`                            |

pub mod loopback;

#[cfg(feature = "fabric")]
pub mod fabric;

use std::{
    future::Future,
    num::{NonZeroU8, NonZeroU32},
    pin::Pin,
    task::{Context, Poll},
};

use bonsai_core::{
    BonsaiError, DevicePtr, ImmTag, Result,
    msg::{FabricAddress, RemoteRegion},
};

/// Where a buffer lives, for registration (DMA-BUF for CUDA, plain pin for host).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemDevice {
    Host,
    Cuda(u8),
}

#[derive(Debug, Clone, Copy)]
pub struct WriteOp<L, R> {
    pub src: L,
    pub src_offset: u64,
    pub len: u64,
    pub dst: R,
    pub dst_offset: u64,
    pub imm: Option<ImmTag>,
    /// Spread one write across this many NICs. `None` = all NICs of the GPU.
    /// Also the knob for emulating a bandwidth-constrained link (1 NIC).
    pub shards: Option<NonZeroU8>,
}

/// Bonsai's RDMA interface.
pub trait Transport: Send + Sync + 'static {
    /// A registered local region usable as a write source or sink.
    type Local: Clone + Send + Sync + 'static;
    /// A peer's region, resolved from a [`RemoteRegion`] once and reused for every write.
    type Remote: Clone + Send + Sync + 'static;

    fn local_address(&self) -> FabricAddress;
    fn nets_per_gpu(&self) -> u8;
    /// Aggregated link speed in bits/s.
    fn link_bps(&self) -> u64;

    /// Register memory used as an RDMA source (sender staging ring).
    fn register_source(&self, ptr: DevicePtr, len: u64, dev: MemDevice) -> Result<Self::Local>;

    /// Register memory peers may write into (receiver arena). Returns the
    /// descriptor to ship to senders.
    fn register_sink(&self, ptr: DevicePtr, len: u64, dev: MemDevice) -> Result<(Self::Local, RemoteRegion)>;

    fn unregister(&self, region: &Self::Local) -> Result<()>;

    fn resolve_remote(&self, region: &RemoteRegion) -> Result<Self::Remote>;

    /// One-sided RDMA WRITE (with optional immediate). Resolves when the local
    /// NIC reports completion, i.e. the data is placed in remote memory.
    fn write(&self, op: WriteOp<Self::Local, Self::Remote>) -> impl Future<Output = Result<()>> + Send;

    /// Start counting writes carrying `tag`; the returned future resolves after
    /// `count` of them have landed.
    ///
    /// **Must be called before the peer can send any write with `tag`.**
    /// fabric-lib resets a counter when it is armed, so immediates that arrive
    /// earlier are lost. This is synchronous for that reason: the receiver arms
    /// every group's counter *before* it publishes the slot grant.
    fn arm_imm(&self, tag: ImmTag, count: NonZeroU32) -> Result<ImmWait>;

    /// Stop counting `tag` (abort / timeout). A pending `ImmWait` resolves with an error.
    fn disarm_imm(&self, tag: ImmTag);
}

/// Future returned by [`Transport::arm_imm`].
pub struct ImmWait {
    rx: tokio::sync::oneshot::Receiver<Result<()>>,
}

impl ImmWait {
    pub fn new() -> (tokio::sync::oneshot::Sender<Result<()>>, Self) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (tx, ImmWait { rx })
    }
}

impl Future for ImmWait {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.rx).poll(cx).map(|r| match r {
            Ok(res) => res,
            Err(_) => Err(BonsaiError::Transport("immediate counter dropped (disarmed)".into())),
        })
    }
}
