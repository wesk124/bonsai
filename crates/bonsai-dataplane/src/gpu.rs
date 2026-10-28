//! The few CUDA runtime calls the pipeline needs, behind a trait so the
//! pipeline logic can be tested with a fake (host memory, instantly-complete events).

use std::future::Future;

use bonsai_core::{CudaStream, DevicePtr, Result};

/// Device memory owned by Bonsai (staging ring, receive arena, pointer tables).
#[derive(Debug)]
pub struct DeviceBuffer {
    pub ptr: DevicePtr,
    pub len: u64,
    pub device: u8,
}

pub trait Gpu: Send + Sync + 'static {
    type Event: Send + Sync + 'static;

    /// `cudaMalloc` (or `cuMemCreate` for DMA-BUF-exportable memory).
    fn alloc(&self, bytes: u64) -> Result<DeviceBuffer>;
    fn free(&self, buf: DeviceBuffer) -> Result<()>;

    /// Upload a small host array (e.g. per-layer pointer table) on `stream`.
    fn upload(&self, dst: DevicePtr, src: &[u8], stream: CudaStream) -> Result<()>;

    /// Copy device bytes back to the host (synchronous on `stream`). Used for the trailer.
    fn download(&self, src: DevicePtr, dst: &mut [u8], stream: CudaStream) -> Result<()>;

    /// `cudaEventRecord` with timing enabled.
    fn record(&self, stream: CudaStream) -> Result<Self::Event>;

    /// Resolves when the event has completed (polls `cudaEventQuery`, yielding
    /// to tokio between polls so the RDMA path keeps running).
    fn completed(&self, ev: &Self::Event) -> impl Future<Output = Result<()>> + Send;

    /// `cudaEventElapsedTime`, in ns.
    fn elapsed_ns(&self, start: &Self::Event, end: &Self::Event) -> Result<u64>;
}
