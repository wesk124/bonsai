use bonsai_core::{DevicePtr, ImmTag, msg::RemoteRegion};
use bonsai_transport::{
    MemDevice, Transport, WriteOp,
    loopback::{LinkModel, LoopbackFabric, LoopbackRegion},
};
use std::{
    num::{NonZeroU8, NonZeroU32},
    time::Duration,
};
fn ptr(bytes: &mut [u8]) -> DevicePtr {
    DevicePtr(bytes.as_mut_ptr() as u64)
}
fn op(
    src: LoopbackRegion,
    dst: LoopbackRegion,
    len: u64,
    tag: Option<ImmTag>,
) -> WriteOp<LoopbackRegion, LoopbackRegion> {
    WriteOp {
        src,
        src_offset: 0,
        len,
        dst,
        dst_offset: 0,
        imm: tag,
        shards: None,
    }
}

#[tokio::test]
async fn copies_selected_ranges_and_completes_counter() {
    let f = LoopbackFabric::new(LinkModel::default());
    let a = f.endpoint();
    let b = f.endpoint();
    assert_ne!(a.local_address().0, b.local_address().0);
    let mut source = [1, 2, 3, 4];
    let mut sink = [0; 6];
    let src = a
        .register_source(ptr(&mut source), 4, MemDevice::Host)
        .unwrap();
    let (local, desc) = b.register_sink(ptr(&mut sink), 6, MemDevice::Host).unwrap();
    let dst = a.resolve_remote(&desc).unwrap();
    let tag = ImmTag(7);
    let wait = b.arm_imm(tag, NonZeroU32::new(2).unwrap()).unwrap();
    let mut request = op(src, dst, 2, Some(tag));
    request.src_offset = 1;
    request.dst_offset = 3;
    a.write(request).await.unwrap();
    assert_eq!(sink, [0, 0, 0, 2, 3, 0]);
    request.src_offset = 0;
    request.dst_offset = 0;
    a.write(request).await.unwrap();
    wait.await.unwrap();
    assert_eq!(sink, [1, 2, 0, 2, 3, 0]);
    a.unregister(&src).unwrap();
    b.unregister(&local).unwrap();
}

#[tokio::test(start_paused = true)]
async fn early_immediates_are_lost_and_disarm_fails_waiter() {
    let f = LoopbackFabric::new(LinkModel::default());
    let a = f.endpoint();
    let b = f.endpoint();
    let mut source = [3];
    let mut sink = [0];
    let src = a
        .register_source(ptr(&mut source), 1, MemDevice::Host)
        .unwrap();
    let (_, desc) = b.register_sink(ptr(&mut sink), 1, MemDevice::Host).unwrap();
    let dst = a.resolve_remote(&desc).unwrap();
    let tag = ImmTag(5);
    a.write(op(src, dst, 1, Some(tag))).await.unwrap();
    let mut wait = b.arm_imm(tag, NonZeroU32::MIN).unwrap();
    assert!(b.arm_imm(tag, NonZeroU32::MIN).is_err());
    assert!(
        tokio::time::timeout(Duration::from_secs(1), &mut wait)
            .await
            .is_err()
    );
    b.disarm_imm(tag);
    assert!(wait.await.is_err());
    let wait = b.arm_imm(tag, NonZeroU32::MIN).unwrap();
    a.write(op(src, dst, 1, Some(tag))).await.unwrap();
    wait.await.unwrap();
    let wait = b.arm_imm(tag, NonZeroU32::MIN).unwrap();
    b.disarm_imm(tag);
    b.disarm_imm(tag);
    assert!(wait.await.is_err());
}

#[tokio::test]
async fn invalid_registration_ranges_and_wrong_devices_are_rejected() {
    let f = LoopbackFabric::new(LinkModel::default());
    let a = f.endpoint();
    let mut data = [0; 4];
    for (p, len, dev) in [
        (DevicePtr(0), 1, MemDevice::Host),
        (ptr(&mut data), 0, MemDevice::Host),
        (DevicePtr(u64::MAX), 2, MemDevice::Host),
        (ptr(&mut data), 4, MemDevice::Cuda(0)),
    ] {
        assert!(a.register_source(p, len, dev).is_err());
    }
    let src = a
        .register_source(ptr(&mut data), 4, MemDevice::Host)
        .unwrap();
    let mut sink = [0; 4];
    let (_, desc) = a.register_sink(ptr(&mut sink), 4, MemDevice::Host).unwrap();
    let dst = a.resolve_remote(&desc).unwrap();
    for (offset, len) in [(0, 0), (0, 5), (u64::MAX, 2), (4, 1)] {
        let mut request = op(src, dst, len, None);
        request.src_offset = offset;
        assert!(a.write(request).await.is_err());
        request.src_offset = 0;
        request.dst_offset = offset;
        assert!(a.write(request).await.is_err());
    }
    let mut request = op(src, dst, 1, None);
    request.shards = NonZeroU8::new(2);
    assert!(a.write(request).await.is_err());
    assert_eq!(sink, [0; 4]);
}

#[tokio::test]
async fn descriptors_cannot_forge_pointers_or_cross_fabrics() {
    let f = LoopbackFabric::new(LinkModel::default());
    let a = f.endpoint();
    let b = f.endpoint();
    let other = LoopbackFabric::new(LinkModel::default()).endpoint();
    let mut sink = [0; 4];
    let (_, desc) = b.register_sink(ptr(&mut sink), 4, MemDevice::Host).unwrap();
    assert!(other.resolve_remote(&desc).is_err());
    for bytes in [vec![], vec![0; 23], vec![255; 24], vec![0; 25]] {
        assert!(a.resolve_remote(&RemoteRegion(bytes.into())).is_err());
    }
    // Guessing a source registration's ID does not grant remote write access.
    let mut source = [1; 4];
    let _src = b
        .register_source(ptr(&mut source), 4, MemDevice::Host)
        .unwrap();
    let mut source_desc = desc.0.to_vec();
    source_desc[16..24].copy_from_slice(&2u64.to_le_bytes());
    assert!(a.resolve_remote(&RemoteRegion(source_desc.into())).is_err());
}

#[tokio::test]
async fn unregister_invalidates_resolved_handles_and_never_resurrects_them() {
    let f = LoopbackFabric::new(LinkModel::default());
    let a = f.endpoint();
    let b = f.endpoint();
    let mut source = [1; 4];
    let mut sink = [0; 4];
    let src = a
        .register_source(ptr(&mut source), 4, MemDevice::Host)
        .unwrap();
    let (local, desc) = b.register_sink(ptr(&mut sink), 4, MemDevice::Host).unwrap();
    let dst = a.resolve_remote(&desc).unwrap();
    assert!(a.unregister(&local).is_err());
    b.unregister(&local).unwrap();
    assert!(b.unregister(&local).is_err());
    assert!(a.resolve_remote(&desc).is_err());
    assert!(a.write(op(src, dst, 4, None)).await.is_err());
    let (_, fresh) = b.register_sink(ptr(&mut sink), 4, MemDevice::Host).unwrap();
    assert_ne!(desc.0, fresh.0);
    assert!(a.write(op(src, dst, 4, None)).await.is_err());
    let dst = a.resolve_remote(&fresh).unwrap();
    a.unregister(&src).unwrap();
    assert!(a.write(op(src, dst, 4, None)).await.is_err());
    assert_eq!(sink, [0; 4]);
}

#[tokio::test(start_paused = true)]
async fn delayed_write_rechecks_registration_before_touching_freed_memory() {
    let f = LoopbackFabric::new(LinkModel {
        latency: Duration::from_secs(1),
        bandwidth_bps: None,
    });
    let a = f.endpoint();
    let b = f.endpoint();
    let mut source = vec![8; 4];
    let mut sink = vec![0; 4];
    let src = a
        .register_source(ptr(&mut source), 4, MemDevice::Host)
        .unwrap();
    let (local, desc) = b.register_sink(ptr(&mut sink), 4, MemDevice::Host).unwrap();
    let dst = a.resolve_remote(&desc).unwrap();
    let write = a.write(op(src, dst, 4, None));
    tokio::pin!(write);
    tokio::select! { result = &mut write => panic!("completed too soon: {result:?}"), _ = tokio::task::yield_now() => {} }
    b.unregister(&local).unwrap();
    drop(sink);
    assert!(write.await.is_err());
}

#[tokio::test(start_paused = true)]
async fn smaller_later_write_can_complete_first_and_counter_follows_placement() {
    let f = LoopbackFabric::new(LinkModel {
        latency: Duration::from_millis(10),
        bandwidth_bps: Some(8),
    });
    let a = f.endpoint();
    let b = f.endpoint();
    let mut source = [9; 4];
    let mut sink = [0; 4];
    let src = a
        .register_source(ptr(&mut source), 4, MemDevice::Host)
        .unwrap();
    let (_, desc) = b.register_sink(ptr(&mut sink), 4, MemDevice::Host).unwrap();
    let dst = a.resolve_remote(&desc).unwrap();
    let long = a.write(op(src, dst, 4, Some(ImmTag(1))));
    tokio::pin!(long);
    let mut short = op(src, dst, 1, Some(ImmTag(2)));
    short.dst_offset = 3;
    let long_wait = b.arm_imm(ImmTag(1), NonZeroU32::MIN).unwrap();
    let short_wait = b.arm_imm(ImmTag(2), NonZeroU32::MIN).unwrap();
    tokio::select! { result = &mut long => panic!("long write completed too soon: {result:?}"), _ = tokio::task::yield_now() => {} }
    a.write(short).await.unwrap();
    short_wait.await.unwrap();
    assert_eq!(sink, [0, 0, 0, 9]);
    long.await.unwrap();
    long_wait.await.unwrap();
    assert_eq!(sink, [9; 4]);
}

#[tokio::test]
async fn dropped_endpoint_revokes_handles_and_pending_counters() {
    let f = LoopbackFabric::new(LinkModel::default());
    let a = f.endpoint();
    let b = f.endpoint();
    let mut sink = [0];
    let (_, desc) = b.register_sink(ptr(&mut sink), 1, MemDevice::Host).unwrap();
    let wait = b.arm_imm(ImmTag(1), NonZeroU32::MIN).unwrap();
    drop(b);
    assert!(wait.await.is_err());
    assert!(a.resolve_remote(&desc).is_err());
}
