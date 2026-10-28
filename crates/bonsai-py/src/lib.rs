//! `import bonsai` — Python side of a Bonsai worker.
//!
//! The model runs in Python (HF Transformers); everything else runs here on a
//! background tokio runtime: the control connection, the transport, staging
//! and arena memory, and the codec launches. Python only sees blocking calls
//! that release the GIL.
//!
//! ```python
//! # prefill worker
//! w = bonsai.Worker(role="prefill", coordinator="10.0.0.1:7000", cuda_device=0)
//! while (job := w.next_job()) is not None:          # Prefill assignment
//!     out = model(input_ids=job.input_ids, use_cache=True)
//!     first = sample(out.logits[:, -1])
//!     report = w.send_kv(job, out.past_key_values, first)   # blocks until all writes complete
//!
//! # decode worker
//! w = bonsai.Worker(role="decode", coordinator="10.0.0.1:7000", cuda_device=0)
//! while (job := w.next_job()) is not None:          # LeaseSlot: slot is leased + armed already
//!     cache, first = w.recv_kv(job)                  # DynamicCache of fresh BF16 tensors
//!     toks = model.generate(..., past_key_values=cache, ...)
//!     w.finish(job, toks)
//! ```

use pyo3::prelude::*;

/// One worker process = one GPU = one fabric-lib `TransferEngine`.
#[pyclass]
pub struct Worker {}

#[pymethods]
impl Worker {
    #[new]
    #[pyo3(signature = (role, coordinator, cuda_device, nets_per_gpu=None, arena_bytes=None, staging_depth=4, throttle_gbps=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        role: &str,
        coordinator: &str,
        cuda_device: u8,
        nets_per_gpu: Option<u8>,
        arena_bytes: Option<u64>,
        staging_depth: u32,
        throttle_gbps: Option<f64>,
    ) -> PyResult<Self> {
        let _ = (role, coordinator, cuda_device, nets_per_gpu, arena_bytes, staging_depth, throttle_gbps);
        todo!("build FabricTransport, allocate staging/arena, connect + Register")
    }

    /// Next assignment (`Prefill` or `LeaseSlot`), or `None` on shutdown.
    /// Decode workers get the job only after the slot is leased and immediates armed.
    fn next_job(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let _ = py;
        todo!()
    }

    /// Prefill side. Reads `DynamicCache` layer tensors (`key_cache[i]`, `value_cache[i]`)
    /// by pointer and stride, on the current torch CUDA stream.
    fn send_kv(&self, py: Python<'_>, job: Py<PyAny>, past_key_values: Py<PyAny>, first_token: u32) -> PyResult<Py<PyAny>> {
        let _ = (py, job, past_key_values, first_token);
        todo!()
    }

    /// Decode side. Allocates BF16 tensors via torch, decodes into them, and returns
    /// `(DynamicCache, first_token)`.
    fn recv_kv(&self, py: Python<'_>, job: Py<PyAny>) -> PyResult<(Py<PyAny>, u32)> {
        let _ = (py, job);
        todo!()
    }

    fn finish(&self, job: Py<PyAny>, output_tokens: Vec<u32>) -> PyResult<()> {
        let _ = (job, output_tokens);
        todo!()
    }
}

/// Same-process codec round trip: encode -> decode on one GPU, no RDMA.
/// Numerically identical to a real handoff, so the quality eval (LongBench,
/// RULER) runs at full speed on any single GPU.
#[pyfunction]
fn roundtrip_kv(py: Python<'_>, past_key_values: Py<PyAny>, precision: &str) -> PyResult<Py<PyAny>> {
    let _ = (py, past_key_values, precision);
    todo!()
}

#[pymodule]
fn bonsai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Worker>()?;
    m.add_function(wrap_pyfunction!(roundtrip_kv, m)?)?;
    Ok(())
}
