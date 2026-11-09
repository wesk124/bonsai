//! Minimal synchronous Python access to Bonsai's real fabric-lib transport.
//! Python owns CUDA tensors and must retain storage until completion.
use pyo3::{exceptions::PyRuntimeError, prelude::*};

fn pyerr(e: impl std::fmt::Display) -> PyErr { PyRuntimeError::new_err(e.to_string()) }

#[cfg(feature = "fabric")]
mod endpoint {
    use super::*;
    use std::{collections::HashMap, num::{NonZeroU8, NonZeroU32}, sync::{Arc, Mutex}, time::Duration};
    use bonsai_core::{DevicePtr, ImmTag, msg::RemoteRegion};
    use bonsai_transport::{fabric::FabricTransport, ImmWait, MemDevice, Transport, WriteOp};
    type Local = <FabricTransport as Transport>::Local;

    #[pyclass]
    pub struct FabricEndpoint {
        transport: Arc<FabricTransport>,
        runtime: Arc<tokio::runtime::Runtime>,
        regions: Mutex<HashMap<u64, Local>>,
        waits: Mutex<HashMap<u32, ImmWait>>,
        cuda_device: u8,
    }

    #[pymethods]
    impl FabricEndpoint {
        #[new]
        #[pyo3(signature=(cuda_device, nets_per_gpu=1, pin_worker_cpu=0, pin_uvm_cpu=0))]
        fn new(cuda_device: u8, nets_per_gpu: u8, pin_worker_cpu: u16, pin_uvm_cpu: u16) -> PyResult<Self> {
            let _ = (pin_worker_cpu, pin_uvm_cpu);
            if nets_per_gpu == 0 { return Err(pyerr("nets_per_gpu must be positive")); }
            let runtime = Arc::new(tokio::runtime::Builder::new_multi_thread().enable_all().build().map_err(pyerr)?);
            let _guard = runtime.enter();
            let transport = Arc::new(FabricTransport::auto(cuda_device as i32, nets_per_gpu as usize).map_err(pyerr)?);
            drop(_guard);
            Ok(Self { transport, runtime, regions: Mutex::new(HashMap::new()), waits: Mutex::new(HashMap::new()), cuda_device })
        }

        fn register_source(&self, ptr: u64, length: u64) -> PyResult<u64> {
            let mut regions = self.regions.lock().map_err(pyerr)?;
            if regions.contains_key(&ptr) { return Err(pyerr("pointer already registered")); }
            let region = self.transport.register_source(DevicePtr(ptr), length, MemDevice::Cuda(self.cuda_device)).map_err(pyerr)?;
            regions.insert(ptr, region);
            Ok(ptr)
        }

        fn register_sink(&self, ptr: u64, length: u64) -> PyResult<(u64, Vec<u8>)> {
            let mut regions = self.regions.lock().map_err(pyerr)?;
            if regions.contains_key(&ptr) { return Err(pyerr("pointer already registered")); }
            let (region, descriptor) = self.transport.register_sink(DevicePtr(ptr), length, MemDevice::Cuda(self.cuda_device)).map_err(pyerr)?;
            regions.insert(ptr, region);
            Ok((ptr, descriptor.0.to_vec()))
        }

        fn unregister(&self, handle: u64) -> PyResult<()> {
            let mut regions = self.regions.lock().map_err(pyerr)?;
            let region = regions.get(&handle).ok_or_else(|| pyerr("unknown registration"))?;
            self.transport.unregister(region).map_err(pyerr)?;
            regions.remove(&handle);
            Ok(())
        }

        fn arm_imm(&self, tag: u32) -> PyResult<()> {
            let mut waits = self.waits.lock().map_err(pyerr)?;
            if waits.contains_key(&tag) { return Err(pyerr("immediate already armed")); }
            waits.insert(tag, self.transport.arm_imm(ImmTag(tag), NonZeroU32::MIN).map_err(pyerr)?);
            Ok(())
        }

        fn wait_imm(&self, py: Python<'_>, tag: u32, timeout: f64) -> PyResult<()> {
            if !timeout.is_finite() || timeout <= 0.0 { return Err(pyerr("timeout must be positive and finite")); }
            let wait = self.waits.lock().map_err(pyerr)?.remove(&tag).ok_or_else(|| pyerr("immediate not armed"))?;
            let runtime = self.runtime.clone();
            let transport = self.transport.clone();
            py.detach(move || runtime.block_on(async move {
                match tokio::time::timeout(Duration::from_secs_f64(timeout), wait).await {
                    Ok(result) => result.map_err(pyerr),
                    Err(_) => { transport.disarm_imm(ImmTag(tag)); Err(pyerr("fabric receive timed out")) }
                }
            }))
        }

        fn write(&self, py: Python<'_>, handle: u64, descriptor: Vec<u8>, length: u64, tag: u32, timeout: f64) -> PyResult<()> {
            if !timeout.is_finite() || timeout <= 0.0 { return Err(pyerr("timeout must be positive and finite")); }
            let src = self.regions.lock().map_err(pyerr)?.get(&handle).cloned().ok_or_else(|| pyerr("unknown source registration"))?;
            let dst = self.transport.resolve_remote(&RemoteRegion(descriptor.into())).map_err(pyerr)?;
            let runtime = self.runtime.clone();
            let transport = self.transport.clone();
            py.detach(move || runtime.block_on(async move {
                let op = WriteOp { src, src_offset: 0, len: length, dst, dst_offset: 0,
                                   imm: Some(ImmTag(tag)), shards: NonZeroU8::new(transport.nets_per_gpu()) };
                tokio::time::timeout(Duration::from_secs_f64(timeout), transport.write(op)).await
                    .map_err(|_| pyerr("fabric write timed out; worker must terminate before storage is reused"))?.map_err(pyerr)
            }))
        }

        #[getter]
        fn link_bps(&self) -> u64 { self.transport.link_bps() }
        #[getter]
        fn nets_per_gpu(&self) -> u8 { self.transport.nets_per_gpu() }
    }
}

#[pymodule]
fn bonsai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "fabric")]
    m.add_class::<endpoint::FabricEndpoint>()?;
    Ok(())
}
