//! Decode side: arm counters before publishing credits, then reconstruct.
//!
//! Two steps, because of fabric-lib's counter semantics:
//!
//! 1. [`KvReceiver::prepare`] (sync): lease a slot and **arm one immediate
//!    counter per layer group**. Only then is the `SlotGrant` returned to be
//!    published, so no write can land before its counter exists.
//! 2. [`KvReceiver::receive`] (async): the default Whole mode waits for every
//!    group and the trailer before decoding. Explicit Ring mode permits
//!    progressive reconstruction while later groups are still arriving.

use std::{sync::Arc, time::Instant};

use bonsai_codec::{Backend, CodecSet};
use bonsai_core::{
    BonsaiError, ChunkPlan, CudaStream, HandoffTrailer, KvSink, ReceiverTimings, Result, WorkerId,
    msg::{FabricAddress, HandoffParams, SlotGrant},
    wire::TRAILER_BYTES,
};
use bonsai_transport::{ImmWait, Transport};

use crate::{
    arena::{ReceiveArena, SlotLease},
    config::{DataPlaneConfig, StagingMode},
    gpu::Gpu,
};

pub struct PreparedHandoff {
    /// Send this to the coordinator (`ToCoordinator::SlotGranted`).
    pub grant: SlotGrant,
    lease: SlotLease,
    plan: ChunkPlan,
    waits: Vec<ImmWait>,
    cleanup: Option<ReceiveCleanup>,
}

pub struct ReceivedHandoff {
    pub trailer: HandoffTrailer,
    pub timings: ReceiverTimings,
}

pub struct KvReceiver<T: Transport, G: Gpu> {
    transport: Arc<T>,
    gpu: Arc<G>,
    arena: Arc<ReceiveArena<T>>,
    cfg: DataPlaneConfig,
    worker: WorkerId,
    addr: FabricAddress,
}

impl<T: Transport, G: Gpu> KvReceiver<T, G> {
    pub fn new(
        transport: Arc<T>,
        gpu: Arc<G>,
        arena: Arc<ReceiveArena<T>>,
        cfg: DataPlaneConfig,
        worker: WorkerId,
    ) -> Self {
        let addr = transport.local_address();
        KvReceiver {
            transport,
            gpu,
            arena,
            cfg,
            worker,
            addr,
        }
    }

    pub fn prepare(&self, params: &HandoffParams) -> Result<PreparedHandoff> {
        if self.cfg.verify_checksums {
            return Err(BonsaiError::InvalidConfig(
                "frame checksum verification is not supported".into(),
            ));
        }
        let plan = params.plan()?;
        let lease = self.arena.lease(params.handoff, plan.slot_bytes)?;
        let mut tags = Vec::with_capacity(plan.num_groups as usize);
        let mut waits = Vec::with_capacity(plan.num_groups as usize);
        for g in 0..plan.num_groups {
            let tag = plan.imm_tag(lease.slot, g);
            let count =
                std::num::NonZeroU32::new(plan.writes_in_group(g)).expect("group has writes");
            match self.transport.arm_imm(tag, count) {
                Ok(w) => {
                    tags.push(tag);
                    waits.push(w);
                }
                Err(e) => {
                    tags.iter().for_each(|t| self.transport.disarm_imm(*t));
                    self.arena.release(lease.slot);
                    return Err(e);
                }
            }
        }
        let grant = SlotGrant {
            handoff: params.handoff,
            decode_worker: self.worker,
            decode_addr: self.addr.clone(),
            slot: lease.slot,
            region: self.arena.remote().clone(),
            base_offset: lease.offset,
            len: lease.len,
        };
        let (transport, arena, cleanup_tags, slot) = (
            self.transport.clone(),
            self.arena.clone(),
            tags.clone(),
            lease.slot,
        );
        let cleanup = ReceiveCleanup {
            complete: false,
            action: Some(Box::new(move |complete| {
                for tag in cleanup_tags {
                    transport.disarm_imm(tag);
                }
                if complete {
                    arena.release(slot);
                } else {
                    arena.release_quarantined(slot);
                }
            })),
        };
        Ok(PreparedHandoff {
            grant,
            lease,
            plan,
            waits,
            cleanup: Some(cleanup),
        })
    }

    /// Wait for the KV and write reconstructed BF16 into `sink`.
    /// On timeout/error the slot is quarantined, not reused immediately.
    pub async fn receive(
        &self,
        mut p: PreparedHandoff,
        sink: &KvSink,
        stream: CudaStream,
    ) -> Result<ReceivedHandoff> {
        let handoff = p.grant.handoff;
        let mut guard = p.cleanup.take().expect("prepared handoff cleanup owner");
        match tokio::time::timeout(self.cfg.lease_timeout, self.receive_inner(p, sink, stream))
            .await
        {
            Ok(Ok(r)) => {
                guard.complete = true;
                Ok(r)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Err(BonsaiError::Timeout(handoff, "receiving")),
        }
    }

    /// Coordinator said the sender failed.
    pub fn abort(&self, p: PreparedHandoff) {
        drop(p);
    }

    async fn receive_inner(
        &self,
        p: PreparedHandoff,
        sink: &KvSink,
        stream: CudaStream,
    ) -> Result<ReceivedHandoff> {
        crate::validate_view(sink, p.plan.layout, p.plan.seq_len, self.cfg.cuda_device)?;
        let codecs = CodecSet::for_plan(&p.plan, Backend::Cuda)?;
        let mut t = ReceiverTimings {
            slot_bytes: p.lease.len,
            ..Default::default()
        };
        let mut decode_events = Vec::new();
        let mut decode_started = None;
        let sequential = matches!(self.cfg.staging, StagingMode::Whole { .. });
        if sequential {
            // All frames and the final trailer must arrive before reconstruction.
            await_all_frames(p.waits).await?;
            t.wait_ns = p.lease.leased_at.elapsed().as_nanos() as u64;
            decode_started = Some(Instant::now());
            for chunk in &p.plan.chunks {
                let start = self.gpu.record(stream)?;
                codecs.get(chunk).decode_chunk(
                    p.lease.ptr.offset(chunk.frame_offset),
                    chunk,
                    sink,
                    stream,
                )?;
                decode_events.push((start, self.gpu.record(stream)?));
            }
        } else {
            for (g, wait) in p.waits.into_iter().enumerate() {
                wait.await?;
                t.wait_ns = p.lease.leased_at.elapsed().as_nanos() as u64;
                decode_started.get_or_insert_with(Instant::now);
                for chunk in p.plan.chunks.iter().filter(|c| c.group as usize == g) {
                    let start = self.gpu.record(stream)?;
                    codecs.get(chunk).decode_chunk(
                        p.lease.ptr.offset(chunk.frame_offset),
                        chunk,
                        sink,
                        stream,
                    )?;
                    decode_events.push((start, self.gpu.record(stream)?));
                }
            }
        }
        let end = self.gpu.record(stream)?;
        self.gpu.completed(&end).await?;
        for (start, end) in decode_events {
            t.decode_gpu_ns = t
                .decode_gpu_ns
                .saturating_add(self.gpu.elapsed_ns(&start, &end)?);
        }
        t.decode_wall_ns = decode_started
            .map(|s| s.elapsed().as_nanos() as u64)
            .unwrap_or(0);

        let mut buf = [0u8; TRAILER_BYTES as usize];
        self.gpu
            .download(p.lease.ptr.offset(p.plan.trailer_offset), &mut buf, stream)?;
        let trailer = HandoffTrailer::from_bytes(&buf)?;
        Ok(ReceivedHandoff {
            trailer,
            timings: t,
        })
    }
}

struct ReceiveCleanup {
    action: Option<Box<dyn FnOnce(bool) + Send + Sync>>,
    complete: bool,
}
impl Drop for ReceiveCleanup {
    fn drop(&mut self) {
        if let Some(action) = self.action.take() {
            action(self.complete);
        }
    }
}

async fn await_all_frames(waits: Vec<ImmWait>) -> Result<()> {
    for wait in waits {
        wait.await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::HostGpu;
    use bonsai_core::{CodecSpec, ElemType, HandoffId, KvLayout, RequestId, wire::ChunkSpec};
    use bonsai_transport::{
        MemDevice, WriteOp,
        loopback::{LinkModel, LoopbackFabric},
    };
    #[tokio::test]
    async fn sequential_wait_requires_all_groups_and_final_trailer() {
        let gpu = Arc::new(HostGpu::default());
        let fabric = LoopbackFabric::new(LinkModel::default());
        let tx = fabric.endpoint();
        let rx = Arc::new(fabric.endpoint());
        let arena = Arc::new(ReceiveArena::new(rx.clone(), gpu.alloc(4096).unwrap()).unwrap());
        let receiver = KvReceiver::new(
            rx.clone(),
            gpu.clone(),
            arena.clone(),
            DataPlaneConfig::default(),
            WorkerId(1),
        );
        let params = HandoffParams {
            handoff: HandoffId(1),
            request: RequestId(1),
            layout: KvLayout {
                num_layers: 2,
                num_kv_heads: 1,
                head_dim: 8,
                dtype: ElemType::Bf16,
            },
            precision: CodecSpec::Bf16.into(),
            seq_len: 2,
            chunk: ChunkSpec::PER_LAYER,
        };
        let mut p = receiver.prepare(&params).unwrap();
        for g in 0..p.plan.num_groups {
            assert!(
                rx.arm_imm(
                    p.plan.imm_tag(p.lease.slot, g),
                    std::num::NonZeroU32::new(1).unwrap()
                )
                .is_err()
            );
        }
        let waiter = tokio::spawn(await_all_frames(std::mem::take(&mut p.waits)));
        let source = gpu.alloc(p.plan.slot_bytes).unwrap();
        let region = tx
            .register_source(source.ptr, source.len, MemDevice::Host)
            .unwrap();
        let remote = tx.resolve_remote(&p.grant.region).unwrap();
        // Deliberately deliver frame groups in reverse order. No group order
        // can replace the all-groups barrier, and last-group count includes trailer.
        for chunk in p.plan.chunks.iter().rev() {
            tx.write(WriteOp {
                src: region.clone(),
                src_offset: chunk.frame_offset,
                len: chunk.frame_len,
                dst: remote.clone(),
                dst_offset: p.grant.base_offset + chunk.frame_offset,
                imm: Some(p.plan.imm_tag(p.lease.slot, chunk.group)),
                shards: None,
            })
            .await
            .unwrap();
        }
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        tx.write(WriteOp {
            src: region.clone(),
            src_offset: p.plan.trailer_offset,
            len: TRAILER_BYTES,
            dst: remote,
            dst_offset: p.grant.base_offset + p.plan.trailer_offset,
            imm: Some(p.plan.imm_tag(p.lease.slot, p.plan.num_groups - 1)),
            shards: None,
        })
        .await
        .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        tx.unregister(&region).unwrap();
        let slot = p.lease.slot;
        drop(p); // A dropped published grant quarantines its memory.
        assert!(arena.free_bytes() < 4096);
        unsafe {
            arena.release_after_drain(slot);
        }
        assert_eq!(arena.free_bytes(), 4096);
    }
}
