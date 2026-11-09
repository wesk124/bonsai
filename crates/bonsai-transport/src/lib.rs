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
#[cfg(feature = "fabric")]
mod topology;

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
    ///
    /// The caller retains the allocation and excludes conflicting accesses until
    /// all writes complete and unregister succeeds. DevicePtr borrows memory.
    fn register_source(&self, ptr: DevicePtr, len: u64, dev: MemDevice) -> Result<Self::Local>;

    /// Register memory peers may write into (receiver arena). Returns the
    /// descriptor to ship to senders.
    fn register_sink(
        &self,
        ptr: DevicePtr,
        len: u64,
        dev: MemDevice,
    ) -> Result<(Self::Local, RemoteRegion)>;

    fn unregister(&self, region: &Self::Local) -> Result<()>;

    fn resolve_remote(&self, region: &RemoteRegion) -> Result<Self::Remote>;

    /// One-sided RDMA WRITE (with optional immediate). Resolves when the local
    /// NIC reports completion, i.e. the data is placed in remote memory.
    fn write(
        &self,
        op: WriteOp<Self::Local, Self::Remote>,
    ) -> impl Future<Output = Result<()>> + Send;

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
    cleanup: Option<Box<dyn FnOnce() + Send + Sync>>,
}

impl ImmWait {
    pub fn new() -> (tokio::sync::oneshot::Sender<Result<()>>, Self) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (tx, ImmWait { rx, cleanup: None })
    }
}

impl Drop for ImmWait {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }
    }
}

impl Future for ImmWait {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let result = Pin::new(&mut self.rx).poll(cx);
        if result.is_ready() {
            if let Some(cleanup) = self.cleanup.take() {
                cleanup();
            }
        }
        result.map(|r| match r {
            Ok(res) => res,
            Err(_) => Err(BonsaiError::Transport(
                "immediate counter dropped (disarmed)".into(),
            )),
        })
    }
}

pub(crate) fn check_region(ptr: DevicePtr, len: u64) -> Result<()> {
    if ptr.0 == 0 || len == 0 {
        return Err(BonsaiError::Transport(
            "registration requires non-null pointer and positive length".into(),
        ));
    }
    let end = ptr
        .0
        .checked_add(len)
        .ok_or_else(|| BonsaiError::Transport("registration address overflow".into()))?;
    usize::try_from(end)
        .map_err(|_| BonsaiError::Transport("registration exceeds host address space".into()))?;
    usize::try_from(len)
        .map_err(|_| BonsaiError::Transport("registration length overflow".into()))?;
    Ok(())
}

pub(crate) fn check_range(offset: u64, len: u64, capacity: u64) -> Result<()> {
    if len == 0 || offset.checked_add(len).is_none_or(|end| end > capacity) {
        return Err(BonsaiError::Transport(
            "write range is empty, overflowing, or outside registration".into(),
        ));
    }
    Ok(())
}
