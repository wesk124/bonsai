use std::{num::NonZeroU8, time::Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagingMode {
    /// Fixed ring of `depth` frame buffers, each `frame_cap` bytes. Bounds
    /// sender memory regardless of context length; encode of chunk i+1
    /// overlaps the write of chunk i. Explicit experimental option.
    Ring { depth: u32, frame_cap: u64 },
    /// Stage the whole encoded handoff, then write. Simplest; used as the
    /// sequential encode-all, transfer-all, reconstruct-all protocol (default).
    Whole { capacity: u64 },
}

#[derive(Debug, Clone)]
pub struct DataPlaneConfig {
    pub cuda_device: u8,
    pub staging: StagingMode,
    /// Receive arena size on decode workers. Bounds concurrent handoffs.
    pub arena_bytes: u64,
    /// Max writes in flight per handoff.
    pub max_inflight_writes: u32,
    /// NICs per write. `Some(1)` + `throttle_bps` emulate a constrained link.
    pub write_shards: Option<NonZeroU8>,
    /// Token-bucket cap on sender egress, bits/s. `None` = line rate.
    pub throttle_bps: Option<u64>,
    /// Receiver gives up on a leased slot after this long without completion.
    pub lease_timeout: Duration,
    /// Legacy observation interval. This duration does not authorize slot reuse;
    /// aborted slots require an explicit transport/GPU drain.
    pub quarantine: Duration,
    /// Debug: checksum frames and verify on receive.
    pub verify_checksums: bool,
}

impl Default for DataPlaneConfig {
    fn default() -> Self {
        DataPlaneConfig {
            cuda_device: 0,
            staging: StagingMode::Whole { capacity: 1 << 30 },
            arena_bytes: 8 << 30,
            max_inflight_writes: 8,
            write_shards: None,
            throttle_bps: None,
            lease_timeout: Duration::from_secs(10),
            quarantine: Duration::from_millis(500),
            verify_checksums: false,
        }
    }
}
