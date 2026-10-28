//! Decode-side receive arena: one large registered buffer carved into slots.
//!
//! The arena is the system's credit pool. A handoff can start only once it
//! holds a slot, so arena capacity directly bounds concurrent in-flight KV and
//! gives natural backpressure to the coordinator (`SlotDenied`).

use std::time::Instant;

use bonsai_core::{DevicePtr, HandoffId, Result, SlotId, msg::RemoteRegion};
use bonsai_transport::Transport;
use parking_lot::Mutex;

use crate::gpu::DeviceBuffer;

#[derive(Debug, Clone)]
pub struct SlotLease {
    pub handoff: HandoffId,
    pub slot: SlotId,
    pub ptr: DevicePtr,
    pub offset: u64,
    pub len: u64,
    pub leased_at: Instant,
}

struct ArenaState {
    // free list of (offset, len), sorted by offset; first-fit, 256 B aligned
    // slot table: index -> (epoch, Option<lease>, quarantine_until)
}

pub struct ReceiveArena<T: Transport> {
    buffer: DeviceBuffer,
    region: T::Local,
    remote: RemoteRegion,
    state: Mutex<ArenaState>,
}

impl<T: Transport> ReceiveArena<T> {
    pub fn new(_transport: &T, _buffer: DeviceBuffer) -> Result<Self> {
        todo!("register_sink(buffer)")
    }

    /// The descriptor every grant carries.
    pub fn remote(&self) -> &RemoteRegion {
        &self.remote
    }

    pub fn region(&self) -> &T::Local {
        &self.region
    }

    /// Reserve `len` bytes. Bumps the slot epoch.
    /// Fails with `ArenaExhausted` instead of waiting; the coordinator queues.
    pub fn lease(&self, _handoff: HandoffId, _len: u64) -> Result<SlotLease> {
        let _ = (&self.buffer, &self.state);
        todo!()
    }

    /// Return a slot after its contents have been decoded.
    pub fn release(&self, _slot: SlotId) {
        todo!()
    }

    /// Return a slot after an abort; held back for `quarantine` before reuse.
    pub fn release_quarantined(&self, _slot: SlotId) {
        todo!()
    }

    pub fn free_bytes(&self) -> u64 {
        todo!()
    }
}
