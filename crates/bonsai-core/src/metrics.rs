//! Per-handoff measurements, mapped 1:1 to the evaluation metrics (docs/HLD.md §7).
//!
//! Durations are measured on one host's monotonic clock each; cross-host
//! timestamps are never subtracted. End-to-end numbers use the coordinator clock.

use serde::{Deserialize, Serialize};

use crate::ids::{HandoffId, RequestId};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SenderTimings {
    pub prefill_ns: u64,
    /// Sum of GPU time of encode kernels (CUDA events).
    pub encode_gpu_ns: u64,
    /// Prefill done -> slot grant available (0 if the grant arrived first).
    pub grant_wait_ns: u64,
    /// Time spent waiting for a free staging buffer (backpressure).
    pub staging_wait_ns: u64,
    /// Prefill done -> last write completion. The sender half of handoff latency.
    pub handoff_send_ns: u64,
    pub wire_bytes: u64,
    /// Payload + metadata without padding.
    pub useful_bytes: u64,
    pub metadata_bytes: u64,
    pub num_writes: u32,
    pub staging_peak_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ReceiverTimings {
    /// Lease -> all immediates counted.
    pub wait_ns: u64,
    /// Last immediate -> all layers dequantized.
    pub decode_gpu_ns: u64,
    pub decode_wall_ns: u64,
    /// Dequantized tensors -> `DynamicCache` ready for `generate`.
    pub cache_build_ns: u64,
    pub slot_bytes: u64,
    pub scratch_peak_bytes: u64,
    /// First decode-side token latency, from cache ready.
    pub first_decode_token_ns: u64,
    pub decode_tokens: u32,
    pub decode_ns: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HandoffReport {
    pub request: Option<RequestId>,
    pub handoff: Option<HandoffId>,
    pub tag: String,
    pub codec: String,
    pub tokens_per_chunk: u32,
    pub layers_per_chunk: u16,
    pub seq_len: u32,
    pub concurrency: u32,
    pub sender: SenderTimings,
    pub receiver: ReceiverTimings,
    /// Coordinator clock: submit -> prefill's first token reported.
    pub ttft_ns: u64,
    /// Coordinator clock: submit -> decode worker's first generated token.
    /// This is the TTFT that includes the KV handoff.
    pub ttfdt_ns: u64,
    pub e2e_ns: u64,
}

impl HandoffReport {
    /// KV handoff latency: sender (encode + RDMA) + receiver (reconstruct + rebuild cache).
    pub fn handoff_latency_ns(&self) -> u64 {
        self.sender.handoff_send_ns + self.receiver.decode_wall_ns + self.receiver.cache_build_ns
    }

    pub fn decode_tokens_per_sec(&self) -> f64 {
        if self.receiver.decode_ns == 0 {
            return 0.0;
        }
        self.receiver.decode_tokens as f64 * 1e9 / self.receiver.decode_ns as f64
    }
}
