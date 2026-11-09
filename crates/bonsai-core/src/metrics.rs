//! Per-handoff measurements, mapped 1:1 to the evaluation metrics (docs/HLD.md §7).
//!
//! Durations are measured on one host's monotonic clock each; cross-host
//! timestamps are never subtracted. End-to-end numbers use the coordinator clock.

use serde::{Deserialize, Serialize};

use crate::ids::{HandoffId, RequestId};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SenderTimings {
    pub prefill_ns: u64,
    /// Sum of per-chunk CUDA event intervals around encode kernels.
    pub encode_gpu_ns: u64,
    /// Time waiting inside send for the slot grant.
    pub grant_wait_ns: u64,
    /// Time spent waiting for a free staging buffer (backpressure).
    pub staging_wait_ns: u64,
    /// Prefill done -> last write completion. The sender half of handoff latency.
    pub handoff_send_ns: u64,
    /// Whole mode: encode-complete -> frame writes complete, before trailer.
    /// Ring mode: overlapping encode/write wall window, not isolated RDMA time.
    #[serde(default)]
    pub transfer_wall_ns: u64,
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
    /// Sum of per-chunk CUDA reconstruction event intervals.
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
    /// Measured prefill-ready -> receiver-cache-ready ACK on one clock.
    /// Absent until an explicit ACK measurement is supplied.
    #[serde(default)]
    pub measured_handoff_ack_ns: Option<u64>,
}

impl HandoffReport {
    /// Actual ready-ACK latency, when measured on one monotonic clock.
    pub fn handoff_latency_ns(&self) -> Option<u64> {
        self.measured_handoff_ack_ns
    }

    /// Phase sum for the sequential protocol only. This is an estimate, not a
    /// measured ready latency; it must not be used for progressive overlap.
    pub fn sequential_phase_estimate_ns(&self) -> u64 {
        self.sender
            .handoff_send_ns
            .saturating_add(self.receiver.decode_wall_ns)
            .saturating_add(self.receiver.cache_build_ns)
    }

    pub fn decode_tokens_per_sec(&self) -> f64 {
        if self.receiver.decode_ns == 0 {
            return 0.0;
        }
        self.receiver.decode_tokens as f64 * 1e9 / self.receiver.decode_ns as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_estimate_does_not_fabricate_ready_ack() {
        let mut r = HandoffReport::default();
        r.sender.handoff_send_ns = 100;
        r.receiver.decode_wall_ns = 30;
        r.receiver.cache_build_ns = 10;
        assert_eq!(r.handoff_latency_ns(), None);
        assert_eq!(r.sequential_phase_estimate_ns(), 140);
        r.measured_handoff_ack_ns = Some(125);
        assert_eq!(r.handoff_latency_ns(), Some(125));
    }
}
