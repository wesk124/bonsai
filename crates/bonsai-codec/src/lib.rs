//! KV-cache codecs.
//!
//! A codec turns one chunk of the live BF16 cache into one wire frame (see
//! `bonsai_core::wire`) and back. Every codec, including the BF16 baseline, goes
//! through the same frame format, so the pipeline is identical across configs
//! and only the codec changes between experiments.
//!
//! Two backends:
//! * [`cuda::CudaCodec`] runs kernels asynchronously on a caller-provided stream.
//!   It reads straight from HF's per-layer tensors and writes into a registered
//!   staging buffer, so encoding needs no extra full-size temporary buffer.
//! * [`reference`] is a slow, obviously-correct CPU implementation. It is the
//!   oracle for kernel tests and defines the numerics.

pub mod reference;

#[cfg(feature = "cuda")]
pub mod cuda;

use bonsai_core::{ChunkDesc, CodecSpec, CudaStream, DevicePtr, KvSink, KvSource, Result};

/// Encode/decode one chunk. Calls are asynchronous with respect to the host:
/// they only enqueue work on `stream`. The caller records an event after
/// `encode_chunk` and waits on it before posting the RDMA write of that frame.
pub trait KvCodec: Send + Sync {
    fn spec(&self) -> CodecSpec;

    /// Read `chunk.layers x chunk.tokens` from `src` and write the frame at `dst`
    /// (`dst` points at the frame start; section offsets come from `chunk.sections`).
    fn encode_chunk(&self, src: &KvSource, chunk: &ChunkDesc, dst: DevicePtr, stream: CudaStream) -> Result<()>;

    /// Read the frame at `frame` and write BF16 values into the matching
    /// rectangle of `dst`.
    fn decode_chunk(&self, frame: DevicePtr, chunk: &ChunkDesc, dst: &KvSink, stream: CudaStream) -> Result<()>;

    /// Device scratch the codec needs per in-flight chunk, beyond the frame
    /// itself. Reported as part of "temporary buffer usage".
    fn scratch_bytes(&self, _chunk: &ChunkDesc) -> u64 {
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda,
}

/// Build the codec for `spec`.
pub fn make_codec(spec: CodecSpec, backend: Backend) -> Result<Box<dyn KvCodec>> {
    match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(Box::new(cuda::CudaCodec::new(spec)?)),
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => {
            let _ = spec;
            Err(bonsai_core::BonsaiError::Codec("built without the `cuda` feature".into()))
        }
    }
}

/// One codec instance per distinct `CodecSpec` in a plan (mixed precision may use several).
pub struct CodecSet {
    codecs: Vec<(CodecSpec, Box<dyn KvCodec>)>,
}

impl CodecSet {
    pub fn for_plan(plan: &bonsai_core::ChunkPlan, backend: Backend) -> Result<Self> {
        let mut codecs: Vec<(CodecSpec, Box<dyn KvCodec>)> = Vec::new();
        for c in &plan.chunks {
            if !codecs.iter().any(|(s, _)| *s == c.codec) {
                codecs.push((c.codec, make_codec(c.codec, backend)?));
            }
        }
        Ok(CodecSet { codecs })
    }

    /// Codec for `chunk`. Panics if the chunk is not from the plan this set was built for.
    pub fn get(&self, chunk: &ChunkDesc) -> &dyn KvCodec {
        self.codecs.iter().find(|(s, _)| *s == chunk.codec).map(|(_, c)| c.as_ref()).expect("chunk codec not in set")
    }
}
