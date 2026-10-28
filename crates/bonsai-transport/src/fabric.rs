//! [`Transport`] on Perplexity fabric-lib's `TransferEngine`.
//!
//! Build with `--features fabric` on a Linux host that has the pplx-garden
//! system requirements (libfabric, libibverbs, CUDA >= 12.8, GDRCopy, DMA-BUF).
//! The tokio runtime must be multi-threaded: fabric-lib's async helpers use
//! `tokio::task::block_in_place`.

use std::{
    ffi::c_void,
    num::{NonZeroU8, NonZeroU32},
    ptr::NonNull,
    sync::Arc,
};

use cuda_lib::{CudaDeviceId, Device};
use bonsai_core::{
    BonsaiError, DevicePtr, ImmTag, Result,
    msg::{FabricAddress, RemoteRegion},
};
use fabric_lib::{
    AsyncTransferEngine, RdmaEngine, TransferEngine, TransferEngineBuilder,
    api::{
        DomainAddress, DomainGroupRouting, MemoryRegionDescriptor, MemoryRegionHandle,
        SingleTransferRequest, TransferRequest,
    },
};

use crate::{ImmWait, MemDevice, Transport, WriteOp};

pub struct FabricTransportConfig {
    pub cuda_device: u8,
    /// NICs to aggregate for this GPU (taken from `detect_topology`).
    pub nets_per_gpu: u8,
    pub pin_worker_cpu: u16,
    pub pin_uvm_cpu: u16,
}

pub struct FabricTransport {
    engine: Arc<TransferEngine>,
    cuda_device: u8,
}

impl FabricTransport {
    /// Pick the GPU's NICs from `fabric_lib::detect_topology()` and build an engine.
    pub fn new(cfg: FabricTransportConfig) -> Result<Self> {
        let topo = fabric_lib::detect_topology().map_err(err)?;
        let group = topo
            .iter()
            .find(|t| t.cuda_device == cfg.cuda_device)
            .ok_or_else(|| err(format!("cuda:{} not in topology", cfg.cuda_device)))?;
        let nics = group.domains.iter().take(cfg.nets_per_gpu as usize).cloned().collect();
        let mut builder = TransferEngineBuilder::default();
        builder.add_gpu_domains(cfg.cuda_device, nics, cfg.pin_worker_cpu, cfg.pin_uvm_cpu);
        let engine = builder.build().map_err(err)?;
        Ok(FabricTransport { engine: Arc::new(engine), cuda_device: cfg.cuda_device })
    }

    pub fn engine(&self) -> &Arc<TransferEngine> {
        &self.engine
    }

    fn device(&self, dev: MemDevice) -> Device {
        match dev {
            MemDevice::Host => Device::Host,
            MemDevice::Cuda(id) => Device::Cuda(CudaDeviceId(id)),
        }
    }
}

fn err(e: impl std::fmt::Display) -> BonsaiError {
    BonsaiError::Transport(e.to_string())
}

fn nonnull(ptr: DevicePtr) -> Result<NonNull<c_void>> {
    NonNull::new(ptr.0 as *mut c_void).ok_or_else(|| err("null pointer"))
}

impl Transport for FabricTransport {
    type Local = MemoryRegionHandle;
    type Remote = Arc<MemoryRegionDescriptor>;

    fn local_address(&self) -> FabricAddress {
        let DomainAddress(bytes) = self.engine.main_address();
        FabricAddress(bytes)
    }

    fn nets_per_gpu(&self) -> u8 {
        self.engine.nets_per_gpu().get()
    }

    fn link_bps(&self) -> u64 {
        self.engine.aggregated_link_speed()
    }

    fn register_source(&self, ptr: DevicePtr, len: u64, dev: MemDevice) -> Result<MemoryRegionHandle> {
        self.engine.register_memory_local(nonnull(ptr)?, len as usize, self.device(dev)).map_err(err)
    }

    fn register_sink(&self, ptr: DevicePtr, len: u64, dev: MemDevice) -> Result<(MemoryRegionHandle, RemoteRegion)> {
        let (handle, desc) =
            self.engine.register_memory_allow_remote(nonnull(ptr)?, len as usize, self.device(dev)).map_err(err)?;
        Ok((handle, RemoteRegion(postcard::to_allocvec(&desc)?.into())))
    }

    fn unregister(&self, region: &MemoryRegionHandle) -> Result<()> {
        self.engine.unregister_memory(region.ptr).map_err(err)
    }

    fn resolve_remote(&self, region: &RemoteRegion) -> Result<Arc<MemoryRegionDescriptor>> {
        Ok(Arc::new(postcard::from_bytes(&region.0)?))
    }

    async fn write(&self, op: WriteOp<MemoryRegionHandle, Arc<MemoryRegionDescriptor>>) -> Result<()> {
        let shards = op
            .shards
            .unwrap_or_else(|| NonZeroU8::new(self.nets_per_gpu()).unwrap_or(NonZeroU8::MIN));
        let req = TransferRequest::Single(SingleTransferRequest {
            src_mr: op.src,
            src_offset: op.src_offset,
            length: op.len,
            imm_data: op.imm.map(|t| t.0),
            dst_mr: (*op.dst).clone(),
            dst_offset: op.dst_offset,
            domain: DomainGroupRouting::RoundRobinSharded { num_shards: shards },
        });
        self.engine.submit_transfer_async(req).await.map_err(err)
    }

    fn arm_imm(&self, tag: ImmTag, count: NonZeroU32) -> Result<ImmWait> {
        let (tx, wait) = ImmWait::new();
        let tx = parking_lot::Mutex::new(Some(tx));
        // Arm synchronously (unlike AsyncTransferEngine::wait_for_imm_count,
        // which only arms when first polled).
        self.engine.set_imm_count_expected(
            tag.0,
            count,
            Box::new(move || {
                if let Some(tx) = tx.lock().take() {
                    let _ = tx.send(Ok(()));
                }
                Ok(false) // one-shot: stop counting after it is reached
            }),
        );
        Ok(wait)
    }

    fn disarm_imm(&self, tag: ImmTag) {
        // Dropping the callback drops the sender, which fails the pending ImmWait.
        self.engine.remove_imm_count(tag.0);
    }
}
