//! Prefill side: encode chunks and RDMA-write them into the decode worker's slot.
//!
//! Pipeline for one handoff (Ring staging):
//!
//! ```text
//! stream:   [enc c0][enc c1][enc c2][enc c3] ...          (GPU, in order)
//! writers:        [wait ev0 → WRITE c0]                    (tokio tasks)
//!                         [wait ev1 → WRITE c1]
//!                                 [wait ev2 → WRITE c2] ...
//! ```
//!
//! * Encoding is enqueued as fast as staging space allows; a full ring is the
//!   backpressure that stops the encoder running ahead of the network.
//! * Each write runs in its own spawned task that owns its `StagingLease`. If
//!   writes were polled inline, waiting on `staging.acquire` would stop them
//!   from completing and freeing space: a deadlock.
//! * Writes may complete in any order (EFA SRD is unordered). The receiver
//!   counts immediates per group, so nothing relies on ordering.
//! * The trailer goes last, once every frame write has completed, so its
//!   `sender_transfer_ns` is final. It is counted in the last group.

use std::{sync::Arc, time::Instant};

use bonsai_codec::{Backend, CodecSet};
use bonsai_core::{
    BonsaiError, CudaStream, HandoffTrailer, ImmTag, KvSource, Result, SenderTimings,
    msg::{HandoffParams, SlotGrant},
    wire::TRAILER_BYTES,
};
use bonsai_transport::{Transport, WriteOp};
use tokio::{sync::oneshot, task::JoinSet};

use crate::{config::DataPlaneConfig, gpu::Gpu, staging::StagingRing, throttle::Throttle};

pub struct SendJob {
    pub params: HandoffParams,
    /// Borrowed view of the HF cache. The caller keeps the tensors alive until `send` returns.
    pub source: KvSource,
    /// The stream prefill ran on; encode is ordered after prefill on it.
    pub stream: CudaStream,
    pub first_token: u32,
    /// The decode worker's grant, forwarded by the coordinator. May arrive
    /// before or after prefill finishes.
    pub grant: oneshot::Receiver<SlotGrant>,
    /// Host time when prefill finished (start of handoff latency).
    pub prefill_done: Instant,
    pub prefill_ns: u64,
}

pub struct KvSender<T: Transport, G: Gpu> {
    transport: Arc<T>,
    gpu: Arc<G>,
    staging: Arc<StagingRing<T>>,
    throttle: Option<Arc<Throttle>>,
    cfg: DataPlaneConfig,
}

impl<T: Transport, G: Gpu> KvSender<T, G> {
    pub fn new(transport: Arc<T>, gpu: Arc<G>, staging: Arc<StagingRing<T>>, cfg: DataPlaneConfig) -> Self {
        let throttle = cfg.throttle_bps.map(|bps| Arc::new(Throttle::new(bps, 4 << 20)));
        KvSender { transport, gpu, staging, throttle, cfg }
    }

    pub async fn send(&self, job: SendJob) -> Result<SenderTimings> {
        let plan = job.params.plan()?;
        let codecs = CodecSet::for_plan(&plan, Backend::Cuda)?;
        let mut t = SenderTimings { prefill_ns: job.prefill_ns, ..Default::default() };

        // 1. The only point where the sender waits on the control plane.
        let grant = job.grant.await.map_err(|_| BonsaiError::Aborted(job.params.handoff, "grant dropped".into()))?;
        t.grant_wait_ns = job.prefill_done.elapsed().as_nanos() as u64;
        if grant.len < plan.slot_bytes {
            return Err(BonsaiError::Protocol(format!("slot {} B < plan {} B", grant.len, plan.slot_bytes)));
        }
        let remote = self.transport.resolve_remote(&grant.region)?;

        // 2. Encode + write every chunk.
        let mut writes = JoinSet::new();
        let mut first_ev = None;
        for chunk in &plan.chunks {
            let wait_start = Instant::now();
            let lease = self.staging.acquire(chunk.frame_len).await?;
            t.staging_wait_ns += wait_start.elapsed().as_nanos() as u64;

            codecs.get(chunk).encode_chunk(&job.source, chunk, lease.ptr, job.stream)?;
            let ev = self.gpu.record(job.stream)?;
            if first_ev.is_none() {
                first_ev = Some(self.gpu.record(job.stream)?);
            }

            let op = WriteOp {
                src: self.staging.region().clone(),
                src_offset: lease.offset,
                len: chunk.frame_len,
                dst: remote.clone(),
                dst_offset: grant.base_offset + chunk.frame_offset,
                imm: Some(ImmTag::new(grant.slot, chunk.group)),
                shards: self.cfg.write_shards,
            };
            let (gpu, transport, throttle) = (self.gpu.clone(), self.transport.clone(), self.throttle.clone());
            writes.spawn(async move {
                gpu.completed(&ev).await?;
                if let Some(th) = throttle {
                    th.acquire(op.len).await;
                }
                let r = transport.write(op).await;
                drop(lease); // staging space is free only after the NIC is done with it
                r
            });

            t.wire_bytes += chunk.frame_len;
            t.useful_bytes += chunk.useful_bytes();
            t.metadata_bytes += chunk.sections.k_meta.len + chunk.sections.v_meta.len;
            t.num_writes += 1;
        }
        while let Some(r) = writes.join_next().await {
            r.map_err(|e| BonsaiError::Transport(e.to_string()))??;
        }

        // 3. Trailer.
        t.handoff_send_ns = job.prefill_done.elapsed().as_nanos() as u64;
        let trailer = HandoffTrailer {
            magic: HandoffTrailer::MAGIC,
            first_token: job.first_token,
            sender_encode_ns: t.encode_gpu_ns,
            sender_transfer_ns: t.handoff_send_ns,
            checksum: 0,
        };
        let lease = self.staging.acquire(TRAILER_BYTES).await?;
        self.gpu.upload(lease.ptr, &trailer.to_bytes(), job.stream)?;
        let ev = self.gpu.record(job.stream)?;
        self.gpu.completed(&ev).await?;
        self.transport
            .write(WriteOp {
                src: self.staging.region().clone(),
                src_offset: lease.offset,
                len: TRAILER_BYTES,
                dst: remote,
                dst_offset: grant.base_offset + plan.trailer_offset,
                imm: Some(ImmTag::new(grant.slot, plan.num_groups - 1)),
                shards: self.cfg.write_shards,
            })
            .await?;
        if let Some(first) = first_ev {
            t.encode_gpu_ns = self.gpu.elapsed_ns(&first, &ev)?; // upper bound: includes stream gaps
        }
        t.wire_bytes += TRAILER_BYTES;
        t.num_writes += 1;
        t.handoff_send_ns = job.prefill_done.elapsed().as_nanos() as u64;
        t.staging_peak_bytes = self.staging.footprint();
        Ok(t)
    }
}
