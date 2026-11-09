//! Registered staging with bounded, cancellation-safe allocation leases.
use crate::{config::StagingMode, gpu::DeviceBuffer};
use bonsai_core::{BonsaiError, DevicePtr, Result};
use bonsai_transport::Transport;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::Notify;

struct Inner<T: Transport> {
    transport: Arc<T>,
    buffer: DeviceBuffer,
    region: T::Local,
    mode: StagingMode,
    free: Mutex<Vec<(u64, u64)>>,
    available: Notify,
}
impl<T: Transport> Drop for Inner<T> {
    fn drop(&mut self) {
        if self.transport.unregister(&self.region).is_err() {
            // Never free memory still registered with a NIC.
            std::mem::forget(self.buffer.clone());
            std::mem::forget(self.transport.clone());
        }
    }
}
pub struct StagingRing<T: Transport> {
    inner: Arc<Inner<T>>,
}
/// A lease retains registration and allocation ownership. Drop only after GPU
/// and transport operations have completed; cancellation must drain those first.
pub struct StagingLease {
    pub ptr: DevicePtr,
    pub offset: u64,
    pub len: u64,
    release: Option<Box<dyn FnOnce() + Send + Sync>>,
}
impl Drop for StagingLease {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}
pub(crate) fn aligned(len: u64) -> Option<u64> {
    len.checked_add(255).map(|n| n & !255)
}
pub(crate) fn return_range(free: &mut Vec<(u64, u64)>, offset: u64, len: u64) {
    free.push((offset, len));
    free.sort_unstable_by_key(|r| r.0);
    let mut i = 0;
    while i + 1 < free.len() {
        if free[i].0 + free[i].1 == free[i + 1].0 {
            free[i].1 += free.remove(i + 1).1;
        } else {
            i += 1;
        }
    }
}
impl<T: Transport> StagingRing<T> {
    pub fn new(transport: Arc<T>, buffer: DeviceBuffer, mode: StagingMode) -> Result<Self> {
        buffer.validate()?;
        let free = match mode {
            StagingMode::Ring { depth, frame_cap } => {
                if depth == 0
                    || frame_cap < 256
                    || frame_cap % 256 != 0
                    || frame_cap
                        .checked_mul(depth as u64)
                        .is_none_or(|n| n > buffer.len)
                {
                    return Err(BonsaiError::InvalidConfig(
                        "invalid ring depth/capacity/alignment".into(),
                    ));
                }
                (0..depth)
                    .map(|i| (i as u64 * frame_cap, frame_cap))
                    .collect()
            }
            StagingMode::Whole { capacity } => {
                if capacity == 0 || capacity > buffer.len || capacity % 256 != 0 {
                    return Err(BonsaiError::InvalidConfig(
                        "invalid whole staging capacity/alignment".into(),
                    ));
                }
                vec![(0, capacity)]
            }
        };
        let region = transport.register_source(buffer.ptr, buffer.len, buffer.memory_device())?;
        Ok(Self {
            inner: Arc::new(Inner {
                transport,
                buffer,
                region,
                mode,
                free: Mutex::new(free),
                available: Notify::new(),
            }),
        })
    }
    pub fn region(&self) -> &T::Local {
        &self.inner.region
    }
    pub fn mode(&self) -> StagingMode {
        self.inner.mode
    }
    pub async fn acquire(&self, len: u64) -> Result<StagingLease> {
        let reserved = match self.inner.mode {
            StagingMode::Ring { frame_cap, .. } if len > 0 && len <= frame_cap => frame_cap,
            StagingMode::Whole { capacity } if len > 0 && len <= capacity => {
                aligned(len).ok_or(BonsaiError::StagingExhausted)?
            }
            _ => return Err(BonsaiError::StagingExhausted),
        };
        loop {
            let notified = self.inner.available.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let offset = {
                let mut free = self.inner.free.lock();
                free.iter().position(|r| r.1 >= reserved).map(|i| {
                    let (offset, capacity) = free.remove(i);
                    if capacity > reserved {
                        free.insert(i, (offset + reserved, capacity - reserved));
                    }
                    offset
                })
            };
            if let Some(offset) = offset {
                let inner = self.inner.clone();
                return Ok(StagingLease {
                    ptr: DevicePtr(inner.buffer.ptr.0 + offset),
                    offset,
                    len,
                    release: Some(Box::new(move || {
                        let mut free = inner.free.lock();
                        match inner.mode {
                            StagingMode::Ring { .. } => free.push((offset, reserved)),
                            StagingMode::Whole { .. } => return_range(&mut free, offset, reserved),
                        }
                        drop(free);
                        inner.available.notify_waiters();
                    })),
                });
            }
            notified.await;
        }
    }
    pub fn footprint(&self) -> u64 {
        self.inner.buffer.len
    }
}
