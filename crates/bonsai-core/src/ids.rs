use serde::{Deserialize, Serialize};

/// A client inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RequestId(pub u64);

/// One KV-cache handoff from a prefill worker to a decode worker.
/// Usually 1:1 with a request, but kept separate so a request can be retried
/// (e.g. falling back to BF16) under a new handoff id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HandoffId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkerRole {
    Prefill,
    Decode,
    /// Runs prefill and decode in one process: the "codec round-trip" mode used for
    /// large-scale quality evaluation without RDMA.
    Colocated,
}

/// A leased region of a decode worker's receive arena.
///
/// `epoch` is bumped each time the slot index is reused, so a stale completion
/// from an aborted handoff cannot be counted against a newer lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SlotId {
    pub index: u16,
    pub epoch: u8,
}

/// The 32-bit RDMA immediate value attached to every write of a handoff.
///
/// fabric-lib counts immediates per distinct u32 value
/// (`TransferEngine::set_imm_count_expected`). Bonsai packs:
///
/// ```text
///  31          20 19      12 11          0
/// +--------------+----------+-------------+
/// | slot (12b)   | epoch(8b)| group (12b) |
/// +--------------+----------+-------------+
/// ```
///
/// `group` lets the receiver count completions per layer group, so it can start
/// dequantizing early layers while later ones are still in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImmTag(pub u32);

impl ImmTag {
    pub const MAX_SLOTS: u16 = 1 << 12;
    pub const MAX_GROUPS: u16 = 1 << 12;

    pub fn new(slot: SlotId, group: u16) -> Self {
        assert!(slot.index < Self::MAX_SLOTS, "slot index out of range");
        assert!(group < Self::MAX_GROUPS, "group out of range");
        ImmTag(((slot.index as u32) << 20) | ((slot.epoch as u32) << 12) | group as u32)
    }

    pub fn slot(self) -> SlotId {
        SlotId { index: (self.0 >> 20) as u16, epoch: ((self.0 >> 12) & 0xff) as u8 }
    }

    pub fn group(self) -> u16 {
        (self.0 & 0xfff) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imm_tag_round_trip() {
        let slot = SlotId { index: 4095, epoch: 200 };
        let tag = ImmTag::new(slot, 27);
        assert_eq!(tag.slot(), slot);
        assert_eq!(tag.group(), 27);
    }
}
