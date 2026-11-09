//! GPU RDMA transport using the pinned Perplexity fabric-lib TransferEngine.
//! Requires a multi-threaded Tokio runtime. Registration borrows allocations;
//! callers retain them until unregister succeeds and all writes complete.
use crate::{ImmWait, MemDevice, Transport, WriteOp};
use bonsai_core::{
    BonsaiError, DevicePtr, ImmTag, Result,
    msg::{FabricAddress, RemoteRegion},
};
use cuda_lib::{CudaDeviceId, Device};
use fabric_lib::{
    RdmaEngine, TransferCallback, TransferEngine, Worker,
    api::{
        DomainAddress, DomainGroupRouting, MemoryRegionDescriptor, MemoryRegionHandle,
        SingleTransferRequest, TransferRequest,
    },
};
use std::{
    collections::HashMap,
    ffi::c_void,
    num::{NonZeroU8, NonZeroU32},
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

pub struct FabricTransportConfig {
    pub cuda_device: u8,
    pub nets_per_gpu: u8,
    pub pin_worker_cpu: u16,
    pub pin_uvm_cpu: u16,
}
static NEXT_ENGINE: AtomicU64 = AtomicU64::new(1);
#[derive(Clone)]
pub struct FabricRegion(Arc<LocalRegion>);
struct LocalRegion {
    owner: u64,
    native: MemoryRegionHandle,
    len: u64,
    active: Arc<tokio::sync::RwLock<bool>>,
}
pub struct FabricRemoteRegion {
    native: MemoryRegionDescriptor,
    len: u64,
}
pub struct FabricTransport {
    engine: Arc<TransferEngine>,
    cuda_device: u8,
    nets: u8,
    id: u64,
    regions: parking_lot::Mutex<HashMap<u64, FabricRegion>>,
    armed: Arc<parking_lot::Mutex<HashMap<u32, Arc<()>>>>,
}
pub(crate) fn err(e: impl std::fmt::Display) -> BonsaiError {
    BonsaiError::Transport(e.to_string())
}

impl FabricTransport {
    pub fn auto(cuda_device: i32, nets_per_gpu: usize) -> Result<Self> {
        let cuda_device = u8::try_from(cuda_device).map_err(err)?;
        let nets_per_gpu = u8::try_from(nets_per_gpu).map_err(err)?;
        let selected = crate::topology::select(cuda_device, nets_per_gpu)?;
        let worker_cpu = selected.cpus[0];
        let uvm_cpu = *selected.cpus.get(1).unwrap_or(&worker_cpu);
        Self::build(cuda_device, nets_per_gpu, worker_cpu, uvm_cpu, selected)
    }
    pub fn new(cfg: FabricTransportConfig) -> Result<Self> {
        let selected = crate::topology::select(cfg.cuda_device, cfg.nets_per_gpu)?;
        let allowed = crate::topology::allowed_cpus()?;
        if !allowed.contains(&cfg.pin_worker_cpu) || !allowed.contains(&cfg.pin_uvm_cpu) {
            return Err(err("worker CPUs outside scheduler affinity"));
        }
        Self::build(
            cfg.cuda_device,
            cfg.nets_per_gpu,
            cfg.pin_worker_cpu,
            cfg.pin_uvm_cpu,
            selected,
        )
    }
    fn build(
        cuda_device: u8,
        nets: u8,
        worker_cpu: u16,
        uvm_cpu: u16,
        selected: crate::topology::Selection,
    ) -> Result<Self> {
        eprintln!(
            "Bonsai fabric topology: {}; worker_cpu={worker_cpu} uvm_cpu={uvm_cpu}",
            selected.description
        );
        let engine = TransferEngine::new(vec![(
            cuda_device,
            Worker {
                domain_list: selected.domains,
                pin_worker_cpu: Some(worker_cpu),
                pin_uvm_cpu: Some(uvm_cpu),
            },
        )])
        .map_err(err)?;
        Ok(Self {
            engine: Arc::new(engine),
            cuda_device,
            nets,
            id: NEXT_ENGINE.fetch_add(1, Ordering::Relaxed),
            regions: Default::default(),
            armed: Default::default(),
        })
    }
    pub fn engine(&self) -> &Arc<TransferEngine> {
        &self.engine
    }
    fn device(&self, dev: MemDevice) -> Result<Device> {
        match dev {
            MemDevice::Host => Ok(Device::Host),
            MemDevice::Cuda(id) if id == self.cuda_device => Ok(Device::Cuda(CudaDeviceId(id))),
            _ => Err(err("registration GPU differs from transport GPU")),
        }
    }
    fn register(
        &self,
        ptr: DevicePtr,
        len: u64,
        dev: MemDevice,
        sink: bool,
    ) -> Result<(FabricRegion, Option<RemoteRegion>)> {
        crate::check_region(ptr, len)?;
        let device = self.device(dev)?;
        let ptr_native = NonNull::new(ptr.0 as *mut c_void).ok_or_else(|| err("null pointer"))?;
        let mut regions = self.regions.lock();
        if regions.contains_key(&ptr.0) {
            return Err(err("pointer already registered"));
        }
        let (native, descriptor) = if sink {
            let (native, desc) = self
                .engine
                .register_memory_allow_remote(ptr_native, len as usize, device)
                .map_err(err)?;
            // Include length: the native descriptor itself carries only ptr/rkeys.
            match postcard::to_allocvec(&(1u8, len, desc)) {
                Ok(bytes) => (native, Some(RemoteRegion(bytes.into()))),
                Err(e) => {
                    let _ = self.engine.unregister_memory(ptr_native);
                    return Err(err(e));
                }
            }
        } else {
            (
                self.engine
                    .register_memory_local(ptr_native, len as usize, device)
                    .map_err(err)?,
                None,
            )
        };
        let region = FabricRegion(Arc::new(LocalRegion {
            owner: self.id,
            native,
            len,
            active: Arc::new(tokio::sync::RwLock::new(true)),
        }));
        regions.insert(ptr.0, region.clone());
        Ok((region, descriptor))
    }
}
impl Transport for FabricTransport {
    type Local = FabricRegion;
    type Remote = Arc<FabricRemoteRegion>;
    fn local_address(&self) -> FabricAddress {
        let DomainAddress(bytes) = self.engine.main_address();
        FabricAddress(bytes)
    }
    fn nets_per_gpu(&self) -> u8 {
        self.nets
    }
    fn link_bps(&self) -> u64 {
        self.engine.aggregated_link_speed()
    }
    fn register_source(&self, ptr: DevicePtr, len: u64, dev: MemDevice) -> Result<FabricRegion> {
        self.register(ptr, len, dev, false).map(|r| r.0)
    }
    fn register_sink(
        &self,
        ptr: DevicePtr,
        len: u64,
        dev: MemDevice,
    ) -> Result<(FabricRegion, RemoteRegion)> {
        let (region, desc) = self.register(ptr, len, dev, true)?;
        Ok((region, desc.expect("sink registration has descriptor")))
    }
    fn unregister(&self, region: &FabricRegion) -> Result<()> {
        if region.0.owner != self.id {
            return Err(err("registration belongs to another transport"));
        }
        let mut regions = self.regions.lock();
        let mut active = region
            .0
            .active
            .try_write()
            .map_err(|_| err("registration has in-flight writes"))?;
        if !*active {
            return Err(err("stale registration"));
        }
        self.engine
            .unregister_memory(region.0.native.ptr)
            .map_err(err)?;
        *active = false;
        regions.remove(&(region.0.native.ptr.as_ptr() as u64));
        Ok(())
    }
    fn resolve_remote(&self, region: &RemoteRegion) -> Result<Arc<FabricRemoteRegion>> {
        let (version, len, native): (u8, u64, MemoryRegionDescriptor) =
            postcard::from_bytes(&region.0).map_err(err)?;
        if version != 1 || native.addr_rkey_list.is_empty() {
            return Err(err("invalid fabric descriptor"));
        }
        crate::check_region(DevicePtr(native.ptr), len)?;
        // Remote revocation is enforced by the NIC's rkey, not by this cache.
        Ok(Arc::new(FabricRemoteRegion { native, len }))
    }
    async fn write(&self, op: WriteOp<FabricRegion, Arc<FabricRemoteRegion>>) -> Result<()> {
        if op.src.0.owner != self.id {
            return Err(err("source belongs to another transport"));
        }
        let active = op.src.0.active.clone().read_owned().await;
        if !*active {
            return Err(err("stale source registration"));
        }
        crate::check_range(op.src_offset, op.len, op.src.0.len)?;
        crate::check_range(op.dst_offset, op.len, op.dst.len)?;
        let shards = op
            .shards
            .unwrap_or_else(|| NonZeroU8::new(self.nets).unwrap());
        if shards.get() > self.nets {
            return Err(err("write shard count exceeds available NICs"));
        }
        // Verbs providers can impose a <=2 GiB maximum message size even
        // though the memory region and the public API use 64-bit lengths.
        // Bound every native Single request; never truncate into ibv_sge.u32.
        // Completing each chunk before submitting the next also makes the
        // last immediate a readiness barrier for the entire logical write.
        const MAX_WRITE_BYTES: u64 = 256 * 1024 * 1024;
        let lease = Arc::new(active);
        let mut completed = 0;
        while completed < op.len {
            let length = (op.len - completed).min(MAX_WRITE_BYTES);
            let is_last = completed + length == op.len;
            let req = TransferRequest::Single(SingleTransferRequest {
                src_mr: op.src.0.native,
                src_offset: op.src_offset + completed,
                length,
                imm_data: if is_last { op.imm.map(|t| t.0) } else { None },
                dst_mr: op.dst.native.clone(),
                dst_offset: op.dst_offset + completed,
                domain: DomainGroupRouting::RoundRobinSharded { num_shards: shards },
            });
            // Native callbacks retain the registration lease even if this
            // future is cancelled while the NIC is still reading a chunk.
            let (tx, rx) = tokio::sync::oneshot::channel();
            let completion = Arc::new(parking_lot::Mutex::new(Some((lease.clone(), tx))));
            let done = completion.clone();
            let failed = completion.clone();
            let callback = TransferCallback {
                on_done: Box::new(move || {
                    if let Some((lease, tx)) = done.lock().take() {
                        drop(lease);
                        let _ = tx.send(Ok(()));
                    }
                    Ok(())
                }),
                on_error: Box::new(move |e| {
                    if let Some((lease, tx)) = failed.lock().take() {
                        drop(lease);
                        let _ = tx.send(Err(err(e)));
                    }
                    Ok(())
                }),
            };
            tokio::task::block_in_place(|| self.engine.submit_transfer(req, callback))
                .map_err(err)?;
            rx.await.map_err(err)??;
            completed += length;
        }
        Ok(())
    }

    fn arm_imm(&self, tag: ImmTag, count: NonZeroU32) -> Result<ImmWait> {
        let mut armed = self.armed.lock();
        if armed.contains_key(&tag.0) {
            return Err(err("immediate tag already armed"));
        }
        let generation = Arc::new(());
        armed.insert(tag.0, generation.clone());
        let (tx, mut wait) = ImmWait::new();
        let tx = parking_lot::Mutex::new(Some(tx));
        self.engine.set_imm_count_expected(
            tag.0,
            count,
            Box::new(move || {
                if let Some(tx) = tx.lock().take() {
                    let _ = tx.send(Ok(()));
                }
                // Polling/dropping ImmWait removes this callback after it returns.
                // Returning false here races with rearming the same immediate tag.
                Ok(true)
            }),
        );
        let tags = self.armed.clone();
        let engine = self.engine.clone();
        wait.cleanup = Some(Box::new(move || {
            let mut armed = tags.lock();
            if armed
                .get(&tag.0)
                .is_some_and(|g| Arc::ptr_eq(g, &generation))
            {
                engine.remove_imm_count(tag.0);
                armed.remove(&tag.0);
            }
        }));
        Ok(wait)
    }
    fn disarm_imm(&self, tag: ImmTag) {
        let mut armed = self.armed.lock();
        self.engine.remove_imm_count(tag.0);
        armed.remove(&tag.0);
    }
}
