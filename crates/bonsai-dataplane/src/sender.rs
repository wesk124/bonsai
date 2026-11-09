//! Sender: sequential staging is the default; ring overlap is opt-in.
use crate::{
    config::{DataPlaneConfig, StagingMode},
    gpu::Gpu,
    staging::{StagingLease, StagingRing},
    throttle::Throttle,
};
use bonsai_codec::{Backend, CodecSet};
use bonsai_core::{
    BonsaiError, CudaStream, HandoffTrailer, ImmTag, KvSource, Result, SenderTimings,
    msg::{HandoffParams, SlotGrant},
    wire::TRAILER_BYTES,
};
use bonsai_transport::{Transport, WriteOp};
use std::{sync::Arc, time::Instant};
use tokio::{sync::oneshot, task::JoinHandle};
pub struct SendJob {
    pub params: HandoffParams,
    /// Borrowed tensors must remain alive until send completes. If the caller
    /// cancels the future, it must retain tensors until the CUDA stream drains.
    pub source: KvSource,
    pub stream: CudaStream,
    pub first_token: u32,
    pub grant: oneshot::Receiver<SlotGrant>,
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
    pub fn new(
        transport: Arc<T>,
        gpu: Arc<G>,
        staging: Arc<StagingRing<T>>,
        cfg: DataPlaneConfig,
    ) -> Self {
        let throttle = cfg
            .throttle_bps
            .map(|bps| Arc::new(Throttle::new(bps, 4 << 20)));
        Self {
            transport,
            gpu,
            staging,
            throttle,
            cfg,
        }
    }
    pub async fn send(&self, job: SendJob) -> Result<SenderTimings> {
        if self.cfg.verify_checksums {
            return Err(BonsaiError::InvalidConfig(
                "frame checksum verification is not supported".into(),
            ));
        }
        if self.cfg.throttle_bps == Some(0)
            || self.cfg.max_inflight_writes == 0
            || self.cfg.staging != self.staging.mode()
        {
            return Err(BonsaiError::InvalidConfig(
                "staging mode mismatch or zero write concurrency".into(),
            ));
        }
        let sender = Self {
            transport: self.transport.clone(),
            gpu: self.gpu.clone(),
            staging: self.staging.clone(),
            throttle: self.throttle.clone(),
            cfg: self.cfg.clone(),
        };
        // Detached on cancellation: do not release staging while CUDA or NIC
        // operations still use it. This also bounds lifetimes on early errors.
        tokio::spawn(async move { sender.send_inner(job).await })
            .await
            .map_err(|e| BonsaiError::Transport(e.to_string()))?
    }
    async fn send_inner(&self, job: SendJob) -> Result<SenderTimings> {
        let plan = job.params.plan()?;
        crate::validate_view(
            &job.source,
            job.params.layout,
            job.params.seq_len,
            self.cfg.cuda_device,
        )?;
        let codecs = CodecSet::for_plan(&plan, Backend::Cuda)?;
        let mut t = SenderTimings {
            prefill_ns: job.prefill_ns,
            ..Default::default()
        };
        let wait_start = Instant::now();
        let grant = job
            .grant
            .await
            .map_err(|_| BonsaiError::Aborted(job.params.handoff, "grant dropped".into()))?;
        t.grant_wait_ns = wait_start.elapsed().as_nanos() as u64;
        if grant.handoff != job.params.handoff
            || grant.len < plan.slot_bytes
            || grant.base_offset.checked_add(plan.slot_bytes).is_none()
            || grant.slot.index >= ImmTag::MAX_SLOTS
        {
            return Err(BonsaiError::Protocol(
                "grant identity or bounds mismatch".into(),
            ));
        }
        let remote = self.transport.resolve_remote(&grant.region)?;
        let mut whole = if matches!(self.cfg.staging, StagingMode::Whole { .. }) {
            let wait_start = Instant::now();
            let lease = self.staging.acquire(plan.slot_bytes).await?;
            t.staging_wait_ns += wait_start.elapsed().as_nanos() as u64;
            Some(lease)
        } else {
            None
        };
        let permits = Arc::new(tokio::sync::Semaphore::new(
            self.cfg.max_inflight_writes as usize,
        ));
        let mut writes: Vec<JoinHandle<Result<()>>> = Vec::new();
        let transfer_start = Instant::now();
        let mut encode_error = None;
        for chunk in &plan.chunks {
            let lease = if whole.is_none() {
                let start = Instant::now();
                let lease = self.staging.acquire(chunk.frame_len).await?;
                t.staging_wait_ns += start.elapsed().as_nanos() as u64;
                Some(lease)
            } else {
                None
            };
            let (ptr, offset) = match (&whole, &lease) {
                (Some(all), _) => (
                    all.ptr.offset(chunk.frame_offset),
                    all.offset + chunk.frame_offset,
                ),
                (_, Some(one)) => (one.ptr, one.offset),
                _ => unreachable!(),
            };
            let start = self.gpu.record(job.stream)?;
            let encoded = codecs
                .get(chunk)
                .encode_chunk(&job.source, chunk, ptr, job.stream);
            let end = match self.gpu.record(job.stream) {
                Ok(ev) => ev,
                Err(e) => {
                    std::mem::forget(lease);
                    std::mem::forget(whole.take());
                    return Err(e);
                }
            };
            // Complete each encode before observing an encode error or allowing
            // a lease to be dropped. Whole mode does not post any writes yet.
            if let Err(e) = self.gpu.completed(&end).await {
                std::mem::forget(lease);
                std::mem::forget(whole.take());
                return Err(e);
            }
            t.encode_gpu_ns = t
                .encode_gpu_ns
                .saturating_add(self.gpu.elapsed_ns(&start, &end)?);
            if let Err(e) = encoded {
                encode_error = Some(e);
                break;
            }
            if let Some(lease) = lease {
                let permit = permits
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| BonsaiError::StagingExhausted)?;
                let (transport, throttle) = (self.transport.clone(), self.throttle.clone());
                let op = WriteOp {
                    src: self.staging.region().clone(),
                    src_offset: offset,
                    len: chunk.frame_len,
                    dst: remote.clone(),
                    dst_offset: grant.base_offset + chunk.frame_offset,
                    imm: Some(ImmTag::new(grant.slot, chunk.group)),
                    shards: self.cfg.write_shards,
                };
                writes.push(tokio::spawn(async move {
                    let lease = lease;
                    let _permit = permit;
                    if let Some(th) = throttle {
                        th.acquire(op.len).await;
                    }
                    let result = transport.write(op).await;
                    if result.is_err() {
                        std::mem::forget(lease);
                    }
                    result
                }));
            }
            t.wire_bytes += chunk.frame_len;
            t.useful_bytes += chunk.useful_bytes();
            t.metadata_bytes += chunk.sections.k_meta.len + chunk.sections.v_meta.len;
            t.num_writes += 1;
        }
        let mut write_error = None;
        // Join every operation even when an earlier write failed. Detached
        // children retain their leases if this parent task is cancelled.
        for write in writes {
            match write.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    write_error.get_or_insert(e);
                }
                Err(e) => {
                    write_error.get_or_insert(BonsaiError::Transport(e.to_string()));
                }
            }
        }
        if let Some(e) = encode_error.or(write_error) {
            return Err(e);
        }
        let transfer_start = if whole.is_some() {
            Instant::now()
        } else {
            transfer_start
        };
        if let Some(all) = &whole {
            // Sequential baseline: all encoding has completed before the first write.
            for chunk in &plan.chunks {
                if let Some(th) = &self.throttle {
                    th.acquire(chunk.frame_len).await;
                }
                let result = self
                    .transport
                    .write(WriteOp {
                        src: self.staging.region().clone(),
                        src_offset: all.offset + chunk.frame_offset,
                        len: chunk.frame_len,
                        dst: remote.clone(),
                        dst_offset: grant.base_offset + chunk.frame_offset,
                        imm: Some(ImmTag::new(grant.slot, chunk.group)),
                        shards: self.cfg.write_shards,
                    })
                    .await;
                if let Err(e) = result {
                    std::mem::forget(whole.take());
                    return Err(e);
                }
            }
        }
        t.transfer_wall_ns = transfer_start.elapsed().as_nanos() as u64;
        t.handoff_send_ns = job.prefill_done.elapsed().as_nanos() as u64;
        let trailer = HandoffTrailer {
            magic: HandoffTrailer::MAGIC,
            first_token: job.first_token,
            sender_encode_ns: t.encode_gpu_ns,
            sender_transfer_ns: t.transfer_wall_ns,
            checksum: 0,
        };
        let mut trailer_lease: Option<StagingLease> = if whole.is_none() {
            Some(self.staging.acquire(TRAILER_BYTES).await?)
        } else {
            None
        };
        let (ptr, offset) = match (&whole, &trailer_lease) {
            (Some(all), _) => (
                all.ptr.offset(plan.trailer_offset),
                all.offset + plan.trailer_offset,
            ),
            (_, Some(one)) => (one.ptr, one.offset),
            _ => unreachable!(),
        };
        if let Err(e) = self.gpu.upload(ptr, &trailer.to_bytes(), job.stream) {
            std::mem::forget(whole.take());
            std::mem::forget(trailer_lease.take());
            return Err(e);
        }
        let ev = match self.gpu.record(job.stream) {
            Ok(ev) => ev,
            Err(e) => {
                std::mem::forget(whole.take());
                std::mem::forget(trailer_lease.take());
                return Err(e);
            }
        };
        if let Err(e) = self.gpu.completed(&ev).await {
            std::mem::forget(whole.take());
            std::mem::forget(trailer_lease.take());
            return Err(e);
        }
        if let Err(e) = self
            .transport
            .write(WriteOp {
                src: self.staging.region().clone(),
                src_offset: offset,
                len: TRAILER_BYTES,
                dst: remote,
                dst_offset: grant.base_offset + plan.trailer_offset,
                imm: Some(ImmTag::new(grant.slot, plan.num_groups - 1)),
                shards: self.cfg.write_shards,
            })
            .await
        {
            std::mem::forget(whole.take());
            std::mem::forget(trailer_lease.take());
            return Err(e);
        }
        t.wire_bytes += TRAILER_BYTES;
        t.num_writes += 1;
        t.handoff_send_ns = job.prefill_done.elapsed().as_nanos() as u64;
        t.staging_peak_bytes = self.staging.footprint();
        Ok(t)
    }
}
