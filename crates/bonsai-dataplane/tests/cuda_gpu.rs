//! Requires a real allocated CUDA GPU; intentionally excluded from portable tests.
#![cfg(feature = "cuda")]
use bonsai_core::CudaStream;
use bonsai_dataplane::{CudaGpu, Gpu};
#[tokio::test]
#[ignore = "requires an allocated CUDA GPU"]
async fn cuda_allocation_copy_event_and_retained_owner() {
    let gpu = CudaGpu::new(0).unwrap();
    assert!(gpu.alloc(0).is_err());
    let buffer = gpu.alloc(1024).unwrap();
    let retained = buffer.clone();
    let ptr = buffer.ptr;
    gpu.free(buffer).unwrap();
    let start = gpu.record(CudaStream(0)).unwrap();
    let bytes: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
    gpu.upload(ptr, &bytes, CudaStream(0)).unwrap();
    let end = gpu.record(CudaStream(0)).unwrap();
    gpu.completed(&end).await.unwrap();
    let _elapsed_ns = gpu.elapsed_ns(&start, &end).unwrap();
    let mut downloaded = vec![0; 1024];
    gpu.download(ptr, &mut downloaded, CudaStream(0)).unwrap();
    assert_eq!(downloaded, bytes);
    gpu.free(retained).unwrap();
}
