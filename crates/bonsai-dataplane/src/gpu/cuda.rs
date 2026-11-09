use super::*;
use std::ffi::{CStr, c_char, c_int, c_void};
#[link(name = "cudart")]
unsafe extern "C" {
    pub(super) fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    pub(super) fn cudaFree(ptr: *mut c_void) -> c_int;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    fn cudaEventCreate(event: *mut *mut c_void) -> c_int;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    fn cudaEventQuery(event: *mut c_void) -> c_int;
    fn cudaEventDestroy(event: *mut c_void) -> c_int;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> c_int;
    fn cudaGetErrorString(code: c_int) -> *const c_char;
}
fn check(code: c_int) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(cudaGetErrorString(code)) }.to_string_lossy();
    Err(BonsaiError::Cuda(format!("CUDA {code}: {message}")))
}
#[derive(Debug)]
pub struct CudaEvent {
    raw: u64,
    device: u8,
}
impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            cudaSetDevice(self.device as i32);
            cudaEventDestroy(self.raw as *mut _);
        }
    }
}
#[derive(Debug)]
pub struct CudaGpu {
    device: u8,
}
impl CudaGpu {
    pub fn new(device: u8) -> Result<Self> {
        check(unsafe { cudaSetDevice(device as i32) })?;
        Ok(Self { device })
    }
    fn select(&self) -> Result<()> {
        check(unsafe { cudaSetDevice(self.device as i32) })
    }
}
impl Gpu for CudaGpu {
    type Event = CudaEvent;
    fn alloc(&self, bytes: u64) -> Result<DeviceBuffer> {
        self.select()?;
        let len = usize::try_from(bytes)
            .map_err(|_| BonsaiError::Cuda("allocation exceeds address space".into()))?;
        if len == 0 {
            return Err(BonsaiError::Cuda("empty CUDA allocation".into()));
        }
        let mut raw = std::ptr::null_mut();
        check(unsafe { cudaMalloc(&mut raw, len) })?;
        let ptr = DevicePtr(raw as u64);
        let owner = Arc::new(Allocation {
            ptr,
            len: bytes,
            storage: Storage::Cuda(self.device),
        });
        Ok(DeviceBuffer {
            ptr,
            len: bytes,
            device: self.device,
            owner,
        })
    }
    fn free(&self, buf: DeviceBuffer) -> Result<()> {
        drop(buf);
        Ok(())
    }
    fn upload(&self, dst: DevicePtr, src: &[u8], stream: CudaStream) -> Result<()> {
        self.select()?;
        check(unsafe {
            cudaMemcpyAsync(
                dst.0 as *mut _,
                src.as_ptr().cast(),
                src.len(),
                1,
                stream.0 as *mut _,
            )
        })?;
        // The host slice may disappear on return. Synchronize before releasing it.
        check(unsafe { cudaStreamSynchronize(stream.0 as *mut _) })
    }
    fn download(&self, src: DevicePtr, dst: &mut [u8], stream: CudaStream) -> Result<()> {
        self.select()?;
        check(unsafe {
            cudaMemcpyAsync(
                dst.as_mut_ptr().cast(),
                src.0 as *const _,
                dst.len(),
                2,
                stream.0 as *mut _,
            )
        })?;
        check(unsafe { cudaStreamSynchronize(stream.0 as *mut _) })
    }
    fn record(&self, stream: CudaStream) -> Result<CudaEvent> {
        self.select()?;
        let mut raw = std::ptr::null_mut();
        check(unsafe { cudaEventCreate(&mut raw) })?;
        let event = CudaEvent {
            raw: raw as u64,
            device: self.device,
        };
        check(unsafe { cudaEventRecord(raw, stream.0 as *mut _) })?;
        Ok(event)
    }
    async fn completed(&self, ev: &CudaEvent) -> Result<()> {
        if ev.device != self.device {
            return Err(BonsaiError::Cuda("event device mismatch".into()));
        }
        loop {
            self.select()?;
            match unsafe { cudaEventQuery(ev.raw as *mut _) } {
                0 => return Ok(()),
                600 => tokio::task::yield_now().await,
                code => return check(code),
            }
        }
    }
    fn elapsed_ns(&self, start: &CudaEvent, end: &CudaEvent) -> Result<u64> {
        self.select()?;
        if start.device != self.device || end.device != self.device {
            return Err(BonsaiError::Cuda("event device mismatch".into()));
        }
        let mut ms = 0.0;
        check(unsafe { cudaEventElapsedTime(&mut ms, start.raw as *mut _, end.raw as *mut _) })?;
        Ok((ms as f64 * 1e6) as u64)
    }
}
