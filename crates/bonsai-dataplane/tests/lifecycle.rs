use bonsai_core::{CudaStream, DevicePtr, HandoffId, SlotId};
use bonsai_dataplane::{Gpu, HostGpu, ReceiveArena, StagingMode, StagingRing};
use bonsai_transport::{
    MemDevice, Transport, WriteOp,
    loopback::{LinkModel, LoopbackFabric},
};
use std::{sync::Arc, time::Duration};

#[test]
fn arena_alignment_fragmentation_and_stale_release() {
    let gpu = HostGpu::default();
    let transport = Arc::new(LoopbackFabric::new(LinkModel::default()).endpoint());
    let arena = ReceiveArena::new(transport, gpu.alloc(1024).unwrap()).unwrap();
    let a = arena.lease(HandoffId(1), 1).unwrap();
    let b = arena.lease(HandoffId(2), 257).unwrap();
    assert_eq!((a.offset, b.offset, arena.free_bytes()), (0, 256, 256));
    arena.release(a.slot);
    let c = arena.lease(HandoffId(3), 256).unwrap();
    assert_eq!(c.slot.index, a.slot.index);
    assert_ne!(c.slot.epoch, a.slot.epoch);
    arena.release(a.slot);
    assert_eq!(arena.free_bytes(), 256);
    arena.release(c.slot);
    arena.release(b.slot);
    assert_eq!(arena.free_bytes(), 1024);
    let all = arena.lease(HandoffId(4), 1024).unwrap();
    assert_eq!(all.offset, 0);
    arena.release(all.slot);
    assert!(arena.lease(HandoffId(5), u64::MAX).is_err());
    assert!(arena.lease(HandoffId(5), 0).is_err());
}
#[test]
fn aborted_slots_require_explicit_drain() {
    let gpu = HostGpu::default();
    let transport = Arc::new(LoopbackFabric::new(LinkModel::default()).endpoint());
    let arena = ReceiveArena::new(transport, gpu.alloc(256).unwrap()).unwrap();
    let a = arena.lease(HandoffId(1), 256).unwrap();
    arena.release_quarantined(a.slot);
    arena.release(a.slot); // ordinary release cannot override quarantine
    assert_eq!(arena.free_bytes(), 0);
    assert!(arena.lease(HandoffId(2), 256).is_err());
    unsafe {
        arena.release_after_drain(SlotId { epoch: 1, ..a.slot });
    }
    assert_eq!(arena.free_bytes(), 0);
    // No writes were ever submitted, so the drain condition holds.
    unsafe {
        arena.release_after_drain(a.slot);
    }
    assert_eq!(arena.free_bytes(), 256);
}
#[test]
fn epoch_never_wraps() {
    let gpu = HostGpu::default();
    let transport = Arc::new(LoopbackFabric::new(LinkModel::default()).endpoint());
    let arena = ReceiveArena::new(transport, gpu.alloc(256).unwrap()).unwrap();
    for epoch in 0..=255 {
        let a = arena.lease(HandoffId(epoch), 1).unwrap();
        assert_eq!(
            a.slot,
            SlotId {
                index: 0,
                epoch: epoch as u8
            }
        );
        arena.release(a.slot);
    }
    let next = arena.lease(HandoffId(256), 1).unwrap();
    assert_eq!(next.slot, SlotId { index: 1, epoch: 0 });
    arena.release(SlotId { index: 0, epoch: 0 });
    assert_eq!(arena.free_bytes(), 0);
    arena.release(next.slot);
}
#[tokio::test]
async fn ring_backpressure_and_cancelled_waiter() {
    let gpu = HostGpu::default();
    let transport = Arc::new(LoopbackFabric::new(LinkModel::default()).endpoint());
    let ring = Arc::new(
        StagingRing::new(
            transport,
            gpu.alloc(512).unwrap(),
            StagingMode::Ring {
                depth: 2,
                frame_cap: 256,
            },
        )
        .unwrap(),
    );
    let a = ring.acquire(1).await.unwrap();
    let b = ring.acquire(256).await.unwrap();
    assert_ne!(a.offset, b.offset);
    assert!(ring.acquire(257).await.is_err());
    assert!(
        tokio::time::timeout(Duration::from_millis(5), ring.acquire(1))
            .await
            .is_err()
    );
    let task_ring = ring.clone();
    let waiter = tokio::spawn(async move { task_ring.acquire(64).await.unwrap() });
    tokio::task::yield_now().await;
    assert!(!waiter.is_finished());
    drop(a);
    let c = tokio::time::timeout(Duration::from_secs(1), waiter)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(c.offset, 0);
    drop((b, c));
    let _a = ring.acquire(256).await.unwrap();
    let _b = ring.acquire(256).await.unwrap();
}
#[tokio::test]
async fn whole_staging_splits_coalesces_and_preserves_registration() {
    let gpu = HostGpu::default();
    let fabric = LoopbackFabric::new(LinkModel::default());
    let transport = Arc::new(fabric.endpoint());
    let sink_transport = fabric.endpoint();
    let sink = gpu.alloc(1024).unwrap();
    let (_, descriptor) = sink_transport
        .register_sink(sink.ptr, sink.len, MemDevice::Host)
        .unwrap();
    let remote = transport.resolve_remote(&descriptor).unwrap();
    let staging = StagingRing::new(
        transport.clone(),
        gpu.alloc(1024).unwrap(),
        StagingMode::Whole { capacity: 1024 },
    )
    .unwrap();
    let a = staging.acquire(1).await.unwrap();
    let b = staging.acquire(257).await.unwrap();
    assert_eq!((a.offset, b.offset), (0, 256));
    drop((a, b));
    let all = staging.acquire(1024).await.unwrap();
    gpu.upload(all.ptr, &[7; 1024], CudaStream(0)).unwrap();
    let region = staging.region().clone();
    drop(staging); // lease keeps registration and allocation alive
    let op = WriteOp {
        src: region.clone(),
        src_offset: 0,
        len: 1024,
        dst: remote.clone(),
        dst_offset: 0,
        imm: None,
        shards: None,
    };
    transport.write(op).await.unwrap();
    let mut bytes = [0; 1024];
    gpu.download(sink.ptr, &mut bytes, CudaStream(0)).unwrap();
    assert_eq!(bytes, [7; 1024]);
    drop(all);
    assert!(
        transport
            .write(WriteOp {
                src: region,
                src_offset: 0,
                len: 1,
                dst: remote,
                dst_offset: 0,
                imm: None,
                shards: None
            })
            .await
            .is_err()
    );
}
#[test]
fn host_gpu_bounds_and_owner_lifetime() {
    let gpu = HostGpu::default();
    let buf = gpu.alloc(256).unwrap();
    assert!(
        gpu.upload(buf.ptr.offset(255), &[1, 2], CudaStream(0))
            .is_err()
    );
    assert!(
        gpu.download(DevicePtr(u64::MAX), &mut [0], CudaStream(0))
            .is_err()
    );
    let retained = buf.clone();
    let ptr = buf.ptr;
    gpu.free(buf).unwrap();
    gpu.upload(ptr, &[1], CudaStream(0)).unwrap();
    drop(retained);
    assert!(gpu.upload(ptr, &[1], CudaStream(0)).is_err());
}

#[tokio::test]
async fn concurrent_ring_leases_are_exclusive() {
    let gpu = HostGpu::default();
    let transport = Arc::new(LoopbackFabric::new(LinkModel::default()).endpoint());
    let ring = Arc::new(
        StagingRing::new(
            transport,
            gpu.alloc(1024).unwrap(),
            StagingMode::Ring {
                depth: 4,
                frame_cap: 256,
            },
        )
        .unwrap(),
    );
    let live = Arc::new(parking_lot::Mutex::new(std::collections::HashSet::new()));
    let mut tasks = Vec::new();
    for _ in 0..16 {
        let (ring, live) = (ring.clone(), live.clone());
        tasks.push(tokio::spawn(async move {
            for _ in 0..20 {
                let lease = ring.acquire(128).await.unwrap();
                assert!(live.lock().insert(lease.offset));
                tokio::task::yield_now().await;
                assert!(live.lock().remove(&lease.offset));
                drop(lease);
            }
        }));
    }
    for task in tasks {
        task.await.unwrap();
    }
    assert!(live.lock().is_empty());
}
