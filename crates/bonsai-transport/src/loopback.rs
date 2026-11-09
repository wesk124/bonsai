//! In-process host-memory transport. Registrations are borrowed: callers must
//! keep their allocation alive and avoid conflicting accesses until unregister
//! and all writes have completed. Wire descriptors contain registration IDs,
//! never pointers supplied by an untrusted peer.
use crate::{ImmWait, MemDevice, Transport, WriteOp};
use bonsai_core::{
    BonsaiError, DevicePtr, ImmTag, Result,
    msg::{FabricAddress, RemoteRegion},
};
use std::{
    collections::HashMap,
    num::NonZeroU32,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct LinkModel {
    pub bandwidth_bps: Option<u64>,
    pub latency: Duration,
}

static NEXT_FABRIC: AtomicU64 = AtomicU64::new(1);
struct Counter {
    remaining: u32,
    tx: tokio::sync::oneshot::Sender<Result<()>>,
}
#[derive(Default)]
struct Endpoint {
    regions: HashMap<u64, Registration>,
    counters: HashMap<u32, Counter>,
}
struct Registration {
    ptr: u64,
    len: u64,
    sink: bool,
}
#[derive(Default)]
struct State {
    next_endpoint: u64,
    next_region: u64,
    endpoints: HashMap<u64, Endpoint>,
}

pub struct LoopbackFabric {
    id: u64,
    link: LinkModel,
    state: parking_lot::Mutex<State>,
}
impl Default for LoopbackFabric {
    fn default() -> Self {
        Self {
            id: NEXT_FABRIC.fetch_add(1, Ordering::Relaxed),
            link: LinkModel::default(),
            state: Default::default(),
        }
    }
}
impl LoopbackFabric {
    pub fn new(link: LinkModel) -> Arc<Self> {
        Arc::new(Self {
            link,
            ..Self::default()
        })
    }
    pub fn endpoint(self: &Arc<Self>) -> LoopbackTransport {
        let mut state = self.state.lock();
        state.next_endpoint = state
            .next_endpoint
            .checked_add(1)
            .expect("endpoint ID exhausted");
        let endpoint = state.next_endpoint;
        state.endpoints.insert(endpoint, Endpoint::default());
        LoopbackTransport {
            fabric: self.clone(),
            endpoint,
        }
    }
}
pub struct LoopbackTransport {
    fabric: Arc<LoopbackFabric>,
    endpoint: u64,
}
/// An opaque registration capability. Clones do not keep a registration active
/// after explicit unregister, or after its owning endpoint is dropped.
#[derive(Debug, Clone, Copy)]
pub struct LoopbackRegion {
    fabric: u64,
    endpoint: u64,
    id: u64,
}
fn err(msg: impl Into<String>) -> BonsaiError {
    BonsaiError::Transport(msg.into())
}
impl LoopbackTransport {
    fn register(
        &self,
        ptr: DevicePtr,
        len: u64,
        dev: MemDevice,
        sink: bool,
    ) -> Result<LoopbackRegion> {
        if dev != MemDevice::Host {
            return Err(err("loopback requires host memory"));
        }
        crate::check_region(ptr, len)?;
        let mut state = self.fabric.state.lock();
        state.next_region = state
            .next_region
            .checked_add(1)
            .ok_or_else(|| err("registration ID exhausted"))?;
        let id = state.next_region;
        state
            .endpoints
            .get_mut(&self.endpoint)
            .ok_or_else(|| err("endpoint closed"))?
            .regions
            .insert(
                id,
                Registration {
                    ptr: ptr.0,
                    len,
                    sink,
                },
            );
        Ok(LoopbackRegion {
            fabric: self.fabric.id,
            endpoint: self.endpoint,
            id,
        })
    }
    fn lookup<'a>(
        &self,
        state: &'a State,
        region: &LoopbackRegion,
        sink: bool,
    ) -> Result<&'a Registration> {
        if region.fabric != self.fabric.id {
            return Err(err("region belongs to another fabric"));
        }
        let reg = state
            .endpoints
            .get(&region.endpoint)
            .and_then(|e| e.regions.get(&region.id))
            .ok_or_else(|| err("stale registration"))?;
        if sink && !reg.sink {
            return Err(err("region does not allow remote writes"));
        }
        Ok(reg)
    }
}
impl Drop for LoopbackTransport {
    fn drop(&mut self) {
        self.fabric.state.lock().endpoints.remove(&self.endpoint);
    }
}
impl Transport for LoopbackTransport {
    type Local = LoopbackRegion;
    type Remote = LoopbackRegion;
    fn local_address(&self) -> FabricAddress {
        FabricAddress(
            [self.fabric.id.to_le_bytes(), self.endpoint.to_le_bytes()]
                .concat()
                .into(),
        )
    }
    fn nets_per_gpu(&self) -> u8 {
        1
    }
    fn link_bps(&self) -> u64 {
        self.fabric.link.bandwidth_bps.unwrap_or(u64::MAX)
    }
    fn register_source(&self, ptr: DevicePtr, len: u64, dev: MemDevice) -> Result<LoopbackRegion> {
        self.register(ptr, len, dev, false)
    }
    fn register_sink(
        &self,
        ptr: DevicePtr,
        len: u64,
        dev: MemDevice,
    ) -> Result<(LoopbackRegion, RemoteRegion)> {
        let r = self.register(ptr, len, dev, true)?;
        Ok((
            r,
            RemoteRegion(
                [
                    r.fabric.to_le_bytes(),
                    r.endpoint.to_le_bytes(),
                    r.id.to_le_bytes(),
                ]
                .concat()
                .into(),
            ),
        ))
    }
    fn unregister(&self, region: &LoopbackRegion) -> Result<()> {
        if region.endpoint != self.endpoint {
            return Err(err("cannot unregister another endpoint's region"));
        }
        let mut state = self.fabric.state.lock();
        self.lookup(&state, region, false)?;
        state
            .endpoints
            .get_mut(&self.endpoint)
            .unwrap()
            .regions
            .remove(&region.id);
        Ok(())
    }
    fn resolve_remote(&self, region: &RemoteRegion) -> Result<LoopbackRegion> {
        if region.0.len() != 24 {
            return Err(err("invalid loopback descriptor"));
        }
        let word = |i| u64::from_le_bytes(region.0[i..i + 8].try_into().unwrap());
        let r = LoopbackRegion {
            fabric: word(0),
            endpoint: word(8),
            id: word(16),
        };
        self.lookup(&self.fabric.state.lock(), &r, true)?;
        Ok(r)
    }
    async fn write(&self, op: WriteOp<LoopbackRegion, LoopbackRegion>) -> Result<()> {
        if op.src.endpoint != self.endpoint {
            return Err(err("source belongs to another endpoint"));
        }
        if op.shards.is_some_and(|n| n.get() != 1) {
            return Err(err("loopback has only one NIC"));
        }
        // Validate before the delay, then revalidate under the lock before
        // accessing memory, so unregister can safely cancel a delayed write.
        {
            let state = self.fabric.state.lock();
            crate::check_range(
                op.src_offset,
                op.len,
                self.lookup(&state, &op.src, false)?.len,
            )?;
            crate::check_range(
                op.dst_offset,
                op.len,
                self.lookup(&state, &op.dst, true)?.len,
            )?;
        }
        let mut delay = self.fabric.link.latency;
        if let Some(bps) = self.fabric.link.bandwidth_bps {
            if bps == 0 {
                return Err(err("link bandwidth must be positive"));
            }
            let nanos = (u128::from(op.len) * 8 * 1_000_000_000).div_ceil(u128::from(bps));
            let nanos = u64::try_from(nanos).map_err(|_| err("link delay overflow"))?;
            delay = delay
                .checked_add(Duration::from_nanos(nanos))
                .ok_or_else(|| err("link delay overflow"))?;
        }
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let mut state = self.fabric.state.lock();
        let src = self.lookup(&state, &op.src, false)?;
        let dst = self.lookup(&state, &op.dst, true)?;
        crate::check_range(op.src_offset, op.len, src.len)?;
        crate::check_range(op.dst_offset, op.len, dst.len)?;
        // Registration validates pointer arithmetic; the caller owns allocation
        // lifetime. Holding the registry lock excludes unregister during copy.
        unsafe {
            std::ptr::copy(
                (src.ptr + op.src_offset) as *const u8,
                (dst.ptr + op.dst_offset) as *mut u8,
                op.len as usize,
            );
        }
        if let Some(tag) = op.imm {
            let counters = &mut state.endpoints.get_mut(&op.dst.endpoint).unwrap().counters;
            if let Some(c) = counters.get_mut(&tag.0) {
                c.remaining -= 1;
                if c.remaining == 0 {
                    let c = counters.remove(&tag.0).unwrap();
                    let _ = c.tx.send(Ok(()));
                }
            }
        }
        Ok(())
    }
    fn arm_imm(&self, tag: ImmTag, count: NonZeroU32) -> Result<ImmWait> {
        let mut state = self.fabric.state.lock();
        let counters = &mut state.endpoints.get_mut(&self.endpoint).unwrap().counters;
        if counters.contains_key(&tag.0) {
            return Err(err("immediate tag already armed"));
        }
        let (tx, wait) = ImmWait::new();
        counters.insert(
            tag.0,
            Counter {
                remaining: count.get(),
                tx,
            },
        );
        Ok(wait)
    }
    fn disarm_imm(&self, tag: ImmTag) {
        self.fabric
            .state
            .lock()
            .endpoints
            .get_mut(&self.endpoint)
            .unwrap()
            .counters
            .remove(&tag.0);
    }
}
