//! Receive credits. Aborted slots remain allocated until an explicit DMA drain.
use crate::{
    gpu::DeviceBuffer,
    staging::{aligned, return_range},
};
use bonsai_core::{BonsaiError, DevicePtr, HandoffId, ImmTag, Result, SlotId, msg::RemoteRegion};
use bonsai_transport::Transport;
use parking_lot::Mutex;
use std::{sync::Arc, time::Instant};
#[derive(Debug, Clone)]
pub struct SlotLease {
    pub handoff: HandoffId,
    pub slot: SlotId,
    pub ptr: DevicePtr,
    pub offset: u64,
    pub len: u64,
    pub leased_at: Instant,
}
#[derive(Clone, Copy)]
enum Status {
    Free,
    Leased { offset: u64, len: u64 },
    Quarantined { offset: u64, len: u64 },
}
struct Entry {
    epoch: u8,
    status: Status,
}
struct ArenaState {
    free: Vec<(u64, u64)>,
    slots: Vec<Entry>,
}
pub struct ReceiveArena<T: Transport> {
    buffer: DeviceBuffer,
    transport: Arc<T>,
    region: T::Local,
    remote: RemoteRegion,
    state: Mutex<ArenaState>,
}
impl<T: Transport> Drop for ReceiveArena<T> {
    fn drop(&mut self) {
        let active = self
            .state
            .get_mut()
            .slots
            .iter()
            .any(|e| !matches!(e.status, Status::Free));
        // No duration proves a failed peer has stopped DMA. Retain the complete
        // registration and allocation if a caller drops an undrained arena.
        if active || self.transport.unregister(&self.region).is_err() {
            std::mem::forget(self.buffer.clone());
            std::mem::forget(self.transport.clone());
        }
    }
}
impl<T: Transport> ReceiveArena<T> {
    pub fn new(transport: Arc<T>, buffer: DeviceBuffer) -> Result<Self> {
        buffer.validate()?;
        if buffer.len % 256 != 0 {
            return Err(BonsaiError::InvalidConfig(
                "arena size must be 256-byte aligned".into(),
            ));
        }
        let (region, remote) =
            transport.register_sink(buffer.ptr, buffer.len, buffer.memory_device())?;
        let len = buffer.len;
        Ok(Self {
            buffer,
            transport,
            region,
            remote,
            state: Mutex::new(ArenaState {
                free: vec![(0, len)],
                slots: vec![],
            }),
        })
    }
    pub fn remote(&self) -> &RemoteRegion {
        &self.remote
    }
    pub fn region(&self) -> &T::Local {
        &self.region
    }
    pub fn lease(&self, handoff: HandoffId, len: u64) -> Result<SlotLease> {
        let reserved = aligned(len)
            .filter(|_| len != 0)
            .ok_or_else(|| BonsaiError::InvalidConfig("empty/overflowing slot request".into()))?;
        let mut state = self.state.lock();
        let available = state.free.iter().map(|r| r.1).sum();
        let range =
            state
                .free
                .iter()
                .position(|r| r.1 >= reserved)
                .ok_or(BonsaiError::ArenaExhausted {
                    requested: reserved,
                    available,
                })?;
        // An 8-bit epoch must never wrap: retire exhausted indices permanently.
        let index = if let Some(i) = state
            .slots
            .iter()
            .position(|s| matches!(s.status, Status::Free) && s.epoch != u8::MAX)
        {
            state.slots[i].epoch += 1;
            i
        } else if state.slots.len() < ImmTag::MAX_SLOTS as usize {
            state.slots.push(Entry {
                epoch: 0,
                status: Status::Free,
            });
            state.slots.len() - 1
        } else {
            return Err(BonsaiError::ArenaExhausted {
                requested: reserved,
                available,
            });
        };
        let (offset, capacity) = state.free.remove(range);
        if capacity > reserved {
            state
                .free
                .insert(range, (offset + reserved, capacity - reserved));
        }
        state.slots[index].status = Status::Leased {
            offset,
            len: reserved,
        };
        Ok(SlotLease {
            handoff,
            slot: SlotId {
                index: index as u16,
                epoch: state.slots[index].epoch,
            },
            ptr: DevicePtr(self.buffer.ptr.0 + offset),
            offset,
            len,
            leased_at: Instant::now(),
        })
    }
    pub fn release(&self, slot: SlotId) {
        let mut state = self.state.lock();
        let Some(entry) = state
            .slots
            .get_mut(slot.index as usize)
            .filter(|e| e.epoch == slot.epoch)
        else {
            return;
        };
        if let Status::Leased { offset, len } = entry.status {
            entry.status = Status::Free;
            return_range(&mut state.free, offset, len);
        }
    }
    pub fn release_quarantined(&self, slot: SlotId) {
        let mut state = self.state.lock();
        if let Some(entry) = state
            .slots
            .get_mut(slot.index as usize)
            .filter(|e| e.epoch == slot.epoch)
        {
            if let Status::Leased { offset, len } = entry.status {
                entry.status = Status::Quarantined { offset, len };
            }
        }
    }
    /// Reclaim an aborted slot only after the caller has explicitly drained all
    /// possible peer DMA and GPU reads. A timer or control-plane abort is insufficient.
    ///
    /// # Safety
    /// No operation from the aborted handoff may still access this slot.
    pub unsafe fn release_after_drain(&self, slot: SlotId) {
        let mut state = self.state.lock();
        let Some(entry) = state
            .slots
            .get_mut(slot.index as usize)
            .filter(|e| e.epoch == slot.epoch)
        else {
            return;
        };
        if let Status::Quarantined { offset, len } = entry.status {
            entry.status = Status::Free;
            return_range(&mut state.free, offset, len);
        }
    }
    pub fn free_bytes(&self) -> u64 {
        self.state.lock().free.iter().map(|r| r.1).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{Gpu, HostGpu};
    use bonsai_transport::loopback::{LinkModel, LoopbackFabric};
    #[test]
    fn exhausted_tag_namespace_is_not_recycled() {
        let gpu = HostGpu::default();
        let transport = Arc::new(LoopbackFabric::new(LinkModel::default()).endpoint());
        let arena = ReceiveArena::new(transport, gpu.alloc(256).unwrap()).unwrap();
        arena.state.lock().slots = (0..ImmTag::MAX_SLOTS)
            .map(|_| Entry {
                epoch: u8::MAX,
                status: Status::Free,
            })
            .collect();
        assert_eq!(arena.free_bytes(), 256);
        assert!(matches!(
            arena.lease(HandoffId(1), 1),
            Err(BonsaiError::ArenaExhausted { .. })
        ));
    }
}
