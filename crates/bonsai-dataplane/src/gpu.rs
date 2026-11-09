//! GPU operations and allocation owners. The portable host backend is for
//! lifecycle tests; CUDA codecs require `CudaGpu` and the `cuda` feature.
use bonsai_core::{BonsaiError, CudaStream, DevicePtr, Result};
use bonsai_transport::MemDevice;
use parking_lot::Mutex;
use std::{
    collections::BTreeMap,
    future::Future,
    sync::{Arc, Weak},
    time::Instant,
};

#[derive(Debug)]
struct Allocation {
    ptr: DevicePtr,
    len: u64,
    storage: Storage,
}
#[derive(Debug)]
enum Storage {
    Host(Mutex<Box<[u8]>>),
    #[cfg(feature = "cuda")]
    Cuda(u8),
}
impl Drop for Allocation {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if let Storage::Cuda(device) = self.storage {
            // Destruction must not panic; CUDA context teardown may already have run.
            unsafe {
                cuda::cudaSetDevice(device as i32);
                cuda::cudaFree(self.ptr.0 as *mut _);
            }
        }
    }
}
/// Allocation owner. Leases retain this owner until GPU/NIC use has completed.
#[derive(Debug, Clone)]
pub struct DeviceBuffer {
    pub ptr: DevicePtr,
    pub len: u64,
    pub device: u8,
    owner: Arc<Allocation>,
}
impl DeviceBuffer {
    pub fn memory_device(&self) -> MemDevice {
        match &self.owner.storage {
            Storage::Host(_) => MemDevice::Host,
            #[cfg(feature = "cuda")]
            Storage::Cuda(d) => MemDevice::Cuda(*d),
        }
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if self.ptr != self.owner.ptr || self.len != self.owner.len || self.len == 0 {
            return Err(BonsaiError::InvalidConfig(
                "modified or empty buffer descriptor".into(),
            ));
        }
        #[cfg(feature = "cuda")]
        if let Storage::Cuda(device) = self.owner.storage {
            if self.device != device {
                return Err(BonsaiError::InvalidConfig("modified buffer device".into()));
            }
        }
        self.ptr
            .0
            .checked_add(self.len)
            .ok_or_else(|| BonsaiError::InvalidConfig("buffer address overflow".into()))?;
        Ok(())
    }
}
pub trait Gpu: Send + Sync + 'static {
    type Event: Send + Sync + 'static;
    fn alloc(&self, bytes: u64) -> Result<DeviceBuffer>;
    fn free(&self, buf: DeviceBuffer) -> Result<()>;
    fn upload(&self, dst: DevicePtr, src: &[u8], stream: CudaStream) -> Result<()>;
    fn download(&self, src: DevicePtr, dst: &mut [u8], stream: CudaStream) -> Result<()>;
    fn record(&self, stream: CudaStream) -> Result<Self::Event>;
    fn completed(&self, ev: &Self::Event) -> impl Future<Output = Result<()>> + Send;
    fn elapsed_ns(&self, start: &Self::Event, end: &Self::Event) -> Result<u64>;
}
#[derive(Default)]
pub struct HostGpu {
    allocations: Mutex<BTreeMap<u64, Weak<Allocation>>>,
}
impl HostGpu {
    fn allocation(&self, ptr: DevicePtr, len: usize) -> Result<(Arc<Allocation>, usize)> {
        let mut allocations = self.allocations.lock();
        allocations.retain(|_, a| a.strong_count() != 0);
        let a = allocations
            .range(..=ptr.0)
            .next_back()
            .and_then(|(_, a)| a.upgrade())
            .ok_or_else(|| BonsaiError::Cuda("host pointer is not allocated".into()))?;
        let offset = ptr.0 - a.ptr.0;
        if offset.checked_add(len as u64).is_none_or(|end| end > a.len) {
            return Err(BonsaiError::Cuda("host copy exceeds allocation".into()));
        }
        Ok((a, offset as usize))
    }
}
impl Gpu for HostGpu {
    type Event = Instant;
    fn alloc(&self, bytes: u64) -> Result<DeviceBuffer> {
        let len =
            usize::try_from(bytes).map_err(|_| BonsaiError::Cuda("allocation too large".into()))?;
        if len == 0 {
            return Err(BonsaiError::Cuda("empty allocation".into()));
        }
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(len)
            .map_err(|e| BonsaiError::Cuda(e.to_string()))?;
        storage.resize(len, 0);
        let mut storage = storage.into_boxed_slice();
        let ptr = DevicePtr(storage.as_mut_ptr() as u64);
        let owner = Arc::new(Allocation {
            ptr,
            len: bytes,
            storage: Storage::Host(Mutex::new(storage)),
        });
        self.allocations
            .lock()
            .insert(ptr.0, Arc::downgrade(&owner));
        Ok(DeviceBuffer {
            ptr,
            len: bytes,
            device: 0,
            owner,
        })
    }
    fn free(&self, buf: DeviceBuffer) -> Result<()> {
        drop(buf);
        Ok(())
    }
    fn upload(&self, dst: DevicePtr, src: &[u8], _: CudaStream) -> Result<()> {
        let (a, offset) = self.allocation(dst, src.len())?;
        let storage = match &a.storage {
            Storage::Host(storage) => storage,
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => unreachable!(),
        };
        storage.lock()[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }
    fn download(&self, src: DevicePtr, dst: &mut [u8], _: CudaStream) -> Result<()> {
        let (a, offset) = self.allocation(src, dst.len())?;
        let storage = match &a.storage {
            Storage::Host(storage) => storage,
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => unreachable!(),
        };
        dst.copy_from_slice(&storage.lock()[offset..offset + dst.len()]);
        Ok(())
    }
    fn record(&self, _: CudaStream) -> Result<Instant> {
        Ok(Instant::now())
    }
    async fn completed(&self, _: &Instant) -> Result<()> {
        Ok(())
    }
    fn elapsed_ns(&self, start: &Instant, end: &Instant) -> Result<u64> {
        Ok(end
            .checked_duration_since(*start)
            .ok_or_else(|| BonsaiError::Cuda("events out of order".into()))?
            .as_nanos()
            .min(u64::MAX as u128) as u64)
    }
}
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaGpu;
