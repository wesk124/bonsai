//! Sender-side staging memory: where encoded frames sit until their RDMA write completes.
//!
//! Registered once at startup (registration costs ms), reused for every handoff.

use std::sync::Arc;

use bonsai_core::{DevicePtr, Result};
use bonsai_transport::Transport;
use tokio::sync::Semaphore;

use crate::{config::StagingMode, gpu::DeviceBuffer};

pub struct StagingRing<T: Transport> {
    buffer: DeviceBuffer,
    region: T::Local,
    mode: StagingMode,
    /// One permit per ring entry (Ring) or per byte-page (Whole).
    permits: Arc<Semaphore>,
}

/// A frame-sized piece of staging memory. Returned to the ring on drop, which
/// the sender does only after the write using it has completed.
pub struct StagingLease {
    pub ptr: DevicePtr,
    /// Offset within the registered region; the RDMA `src_offset`.
    pub offset: u64,
    pub len: u64,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl<T: Transport> StagingRing<T> {
    pub fn new(_transport: &T, _buffer: DeviceBuffer, _mode: StagingMode) -> Result<Self> {
        todo!("register_source(buffer) and create permits")
    }

    pub fn region(&self) -> &T::Local {
        &self.region
    }

    /// Wait for space for one frame of `len` bytes. Errors if `len` exceeds
    /// `frame_cap` (the planner must not produce such chunks).
    pub async fn acquire(&self, _len: u64) -> Result<StagingLease> {
        let _ = (&self.buffer, &self.mode, &self.permits);
        todo!()
    }

    /// Bytes of device memory held by staging (for buffer-usage metrics).
    pub fn footprint(&self) -> u64 {
        self.buffer.len
    }
}
