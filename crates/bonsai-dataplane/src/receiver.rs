//! Decode side: lease a slot, arm completion counters, then decode progressively.
//!
//! Two steps, because of fabric-lib's counter semantics:
//!
//! 1. [`KvReceiver::prepare`] (sync): lease a slot and **arm one immediate
//!    counter per layer group**. Only then is the `SlotGrant` returned to be
//!    published, so no write can land before its counter exists.
//! 2. [`KvReceiver::receive`] (async): await group 0, decode its chunks into the
//!    sink while groups 1.. are still arriving, and so on. The last group also
//!    covers the trailer, which holds the first token.

use std::{sync::Arc, time::Instant};

use bonsai_codec::{Backend, CodecSet};
use bonsai_core::{
    BonsaiError, ChunkPlan, CudaStream, HandoffTrailer, ImmTag, KvSink, ReceiverTimings, Result, WorkerId,
    msg::{FabricAddress, HandoffParams, SlotGrant},
    wire::TRAILER_BYTES,
};
use bonsai_transport::{ImmWait, Transport};

use crate::{
    arena::{ReceiveArena, SlotLease},
    config::DataPlaneConfig,
    gpu::Gpu,
};

pub struct PreparedHandoff {
    /// Send this to the coordinator (`ToCoordinator::SlotGranted`).
    pub grant: SlotGrant,
    lease: SlotLease,
    plan: ChunkPlan,
    tags: Vec<ImmTag>,
    waits: Vec<ImmWait>,
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
    pub fn new(transport: Arc<T>, gpu: Arc<G>, arena: Arc<ReceiveArena<T>>, cfg: DataPlaneConfig, worker: WorkerId) -> Self {
        let addr = transport.local_address();
        KvReceiver { transport, gpu, arena, cfg, worker, addr }
    }

    pub fn prepare(&self, params: &HandoffParams) -> Result<PreparedHandoff> {
        let plan = params.plan()?;
        let lease = self.arena.lease(params.handoff, plan.slot_bytes)?;
        let mut tags = Vec::with_capacity(plan.num_groups as usize);
        let mut waits = Vec::with_capacity(plan.num_groups as usize);
        for g in 0..plan.num_groups {
            let tag = plan.imm_tag(lease.slot, g);
            let count = std::num::NonZeroU32::new(plan.writes_in_group(g)).expect("group has writes");
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
        Ok(PreparedHandoff { grant, lease, plan, tags, waits })
    }

    /// Wait for the KV and write reconstructed BF16 into `sink`.
    /// On timeout/error the slot is quarantined, not reused immediately.
    pub async fn receive(&self, p: PreparedHandoff, sink: &KvSink, stream: CudaStream) -> Result<ReceivedHandoff> {
        let handoff = p.grant.handoff;
        let tags = p.tags.clone();
        let slot = p.lease.slot;
        match tokio::time::timeout(self.cfg.lease_timeout, self.receive_inner(p, sink, stream)).await {
            Ok(Ok(r)) => {
                self.arena.release(slot);
                Ok(r)
            }
            Ok(Err(e)) => {
                self.cleanup(&tags, slot);
                Err(e)
            }
            Err(_) => {
                self.cleanup(&tags, slot);
                Err(BonsaiError::Timeout(handoff, "receiving"))
            }
        }
    }

    /// Coordinator said the sender failed.
    pub fn abort(&self, p: PreparedHandoff) {
        self.cleanup(&p.tags, p.lease.slot);
    }

    fn cleanup(&self, tags: &[ImmTag], slot: bonsai_core::SlotId) {
        tags.iter().for_each(|t| self.transport.disarm_imm(*t));
        self.arena.release_quarantined(slot);
    }

    async fn receive_inner(&self, p: PreparedHandoff, sink: &KvSink, stream: CudaStream) -> Result<ReceivedHandoff> {
        let codecs = CodecSet::for_plan(&p.plan, Backend::Cuda)?;
        let mut t = ReceiverTimings { slot_bytes: p.lease.len, ..Default::default() };
        let mut first_ev = None;
        let mut decode_started = None;
        for (g, wait) in p.waits.into_iter().enumerate() {
            wait.await?;
            if g == 0 {
                t.wait_ns = p.lease.leased_at.elapsed().as_nanos() as u64;
            }
            decode_started.get_or_insert_with(Instant::now);
            for chunk in p.plan.chunks.iter().filter(|c| c.group as usize == g) {
                if first_ev.is_none() {
                    first_ev = Some(self.gpu.record(stream)?);
                }
                codecs.get(chunk).decode_chunk(p.lease.ptr.offset(chunk.frame_offset), chunk, sink, stream)?;
            }
        }
        let end = self.gpu.record(stream)?;
        self.gpu.completed(&end).await?;
        if let Some(first) = first_ev {
            t.decode_gpu_ns = self.gpu.elapsed_ns(&first, &end)?;
        }
        t.decode_wall_ns = decode_started.map(|s| s.elapsed().as_nanos() as u64).unwrap_or(0);

        let mut buf = [0u8; TRAILER_BYTES as usize];
        self.gpu.download(p.lease.ptr.offset(p.plan.trailer_offset), &mut buf, stream)?;
        let trailer = HandoffTrailer::from_bytes(&buf)?;
        Ok(ReceivedHandoff { trailer, timings: t })
    }
}
