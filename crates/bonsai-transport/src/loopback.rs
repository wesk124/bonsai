//! In-process transport over host memory, for tests and macOS development.
//!
//! Several `LoopbackTransport`s created from one [`LoopbackFabric`] can write
//! into each other's registered regions with `memcpy`, and immediates are
//! counted exactly like fabric-lib (including dropping un-armed ones), so
//! dataplane ordering bugs show up without RDMA hardware.
//!
//! An optional bandwidth/latency model (`LinkModel`) delays write completions,
//! for rough what-if numbers before running on the real cluster.

use std::{num::NonZeroU32, sync::Arc, time::Duration};

use bonsai_core::{
    DevicePtr, ImmTag, Result,
    msg::{FabricAddress, RemoteRegion},
};

use crate::{ImmWait, MemDevice, Transport, WriteOp};

#[derive(Debug, Clone, Copy, Default)]
pub struct LinkModel {
    pub bandwidth_bps: Option<u64>,
    pub latency: Duration,
}

/// Shared "switch" connecting loopback endpoints.
#[derive(Default)]
pub struct LoopbackFabric {
    _inner: parking_lot::Mutex<()>,
}

impl LoopbackFabric {
    pub fn new(_link: LinkModel) -> Arc<Self> {
        Arc::new(LoopbackFabric::default())
    }

    pub fn endpoint(self: &Arc<Self>) -> LoopbackTransport {
        todo!("allocate an address and per-endpoint imm counter map")
    }
}

pub struct LoopbackTransport {
    _fabric: Arc<LoopbackFabric>,
}

#[derive(Debug, Clone, Copy)]
pub struct LoopbackRegion {
    pub endpoint: u32,
    pub ptr: u64,
    pub len: u64,
}

impl Transport for LoopbackTransport {
    type Local = LoopbackRegion;
    type Remote = LoopbackRegion;

    fn local_address(&self) -> FabricAddress {
        todo!()
    }
    fn nets_per_gpu(&self) -> u8 {
        1
    }
    fn link_bps(&self) -> u64 {
        todo!()
    }
    fn register_source(&self, _ptr: DevicePtr, _len: u64, _dev: MemDevice) -> Result<LoopbackRegion> {
        todo!()
    }
    fn register_sink(&self, _ptr: DevicePtr, _len: u64, _dev: MemDevice) -> Result<(LoopbackRegion, RemoteRegion)> {
        todo!()
    }
    fn unregister(&self, _region: &LoopbackRegion) -> Result<()> {
        todo!()
    }
    fn resolve_remote(&self, _region: &RemoteRegion) -> Result<LoopbackRegion> {
        todo!()
    }
    async fn write(&self, _op: WriteOp<LoopbackRegion, LoopbackRegion>) -> Result<()> {
        todo!("bounds-check, memcpy, apply LinkModel delay, bump peer's imm counter")
    }
    fn arm_imm(&self, _tag: ImmTag, _count: NonZeroU32) -> Result<ImmWait> {
        todo!()
    }
    fn disarm_imm(&self, _tag: ImmTag) {
        todo!()
    }
}
