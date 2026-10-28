//! Control-plane messages.
//!
//! These go over the coordinator's TCP connections as length-prefixed postcard
//! frames. They are small and never carry KV bytes; KV only moves over fabric-lib.

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{
    codec_spec::PrecisionMap,
    error::Result,
    ids::{HandoffId, RequestId, SlotId, WorkerId, WorkerRole},
    layout::KvLayout,
    metrics::{HandoffReport, SenderTimings},
    wire::{ChunkPlan, ChunkSpec},
};

/// Serialized `fabric_lib::api::DomainAddress`. Opaque outside `bonsai-transport`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FabricAddress(pub Bytes);

/// Serialized `fabric_lib::api::MemoryRegionDescriptor` (ptr + per-NIC rkeys).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteRegion(pub Bytes);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub role: WorkerRole,
    pub hostname: String,
    pub cuda_device: u8,
    pub fabric_addr: FabricAddress,
    pub nets_per_gpu: u8,
    /// Aggregated NIC speed from `TransferEngine::aggregated_link_speed`, bits/s.
    pub link_bps: u64,
    pub model: String,
    pub layout: KvLayout,
    /// Decode workers: size of the registered receive arena.
    pub arena_bytes: u64,
    pub max_concurrency: u32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct WorkerLoad {
    pub active: u32,
    pub queued: u32,
    pub arena_free_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplingParams {
    /// 0.0 = greedy. Evaluation always uses greedy so configs are comparable.
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams { temperature: 0.0, top_p: 1.0, seed: 0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: RequestId,
    /// Already tokenized with the chat template, so every config sees identical input.
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: u32,
    pub stop_tokens: Vec<u32>,
    pub sampling: SamplingParams,
    /// Per-request overrides; otherwise the coordinator's policy decides.
    pub precision: Option<PrecisionMap>,
    pub chunk: Option<ChunkSpec>,
    /// Free-form label carried into reports, e.g. "qasper/123".
    pub tag: String,
}

/// Everything needed to rebuild the same [`ChunkPlan`] on both ends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffParams {
    pub handoff: HandoffId,
    pub request: RequestId,
    pub layout: KvLayout,
    pub precision: PrecisionMap,
    pub seq_len: u32,
    pub chunk: ChunkSpec,
}

impl HandoffParams {
    pub fn plan(&self) -> Result<ChunkPlan> {
        ChunkPlan::build(self.layout, self.precision.clone(), self.seq_len, self.chunk)
    }
}

/// A decode worker's answer to `LeaseSlot`: where the sender should write.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotGrant {
    pub handoff: HandoffId,
    pub decode_worker: WorkerId,
    pub decode_addr: FabricAddress,
    pub slot: SlotId,
    /// The whole arena's descriptor; the slot is `[base_offset, base_offset + len)`.
    pub region: RemoteRegion,
    pub base_offset: u64,
    pub len: u64,
}

/// Anything -> coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToCoordinator {
    // Worker lifecycle
    Register(WorkerInfo),
    Heartbeat { worker: WorkerId, load: WorkerLoad },
    Deregister { worker: WorkerId },

    // Client
    Submit(InferenceRequest),

    // Decode worker
    SlotGranted(SlotGrant),
    SlotDenied { handoff: HandoffId, reason: String },
    Finished { handoff: HandoffId, output_tokens: Vec<u32>, report: HandoffReport },

    // Prefill worker
    PrefillDone { handoff: HandoffId, first_token: u32, timings: SenderTimings },

    // Either worker
    HandoffFailed { handoff: HandoffId, reason: String },
}

/// Coordinator -> worker or client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FromCoordinator {
    Registered { worker: WorkerId },

    // To decode worker: reserve a slot, then wait for the KV and continue decoding.
    LeaseSlot { params: HandoffParams, request: InferenceRequest },

    // To prefill worker: run prefill now. The grant may arrive later as `SlotReady`;
    // the sender only blocks on it right before its first RDMA write.
    Prefill { params: HandoffParams, request: InferenceRequest },
    SlotReady(SlotGrant),

    Abort { handoff: HandoffId, reason: String },

    // To client
    Accepted { request: RequestId, handoff: HandoffId },
    Completed { request: RequestId, output_tokens: Vec<u32>, report: HandoffReport },
    Rejected { request: RequestId, reason: String },

    Shutdown,
}

/// Length-prefixed (u32 LE) postcard framing used on every control connection.
pub mod framing {
    use super::*;

    pub const MAX_FRAME: usize = 64 << 20;

    pub fn encode<T: Serialize>(msg: &T) -> Result<Vec<u8>> {
        let body = postcard::to_allocvec(msg)?;
        let mut out = Vec::with_capacity(4 + body.len());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }

    pub fn decode<'a, T: Deserialize<'a>>(body: &'a [u8]) -> Result<T> {
        Ok(postcard::from_bytes(body)?)
    }
}
