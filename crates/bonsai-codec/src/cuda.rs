//! CUDA backend: FFI to `kernels/bonsai_codec.cu`.
//!
//! Kernel mapping (one CUDA block per (layer, head, token-tile)):
//! * BF16 pack: strided copy into the contiguous frame layout.
//! * HeadDim groups: each warp owns one (layer, head, token) row of 128 channels,
//!   reduces max/min with warp shuffles, writes scale (+zero) and codes. Single pass.
//! * Token groups (per-channel K): each thread owns one channel and walks
//!   `group_size` tokens; two passes over a tile in shared memory.
//! Decode is the inverse and writes into the sink's strided BF16 tensors.

use bonsai_core::{ChunkDesc, CodecSpec, CudaStream, DevicePtr, KvSink, KvSource, Result};

use crate::KvCodec;

/// Arguments shared by encode and decode launches. `#[repr(C)]` mirror of
/// `BonsaiChunkArgs` in `kernels/bonsai_codec.cuh`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChunkArgs {
    /// Device array of `num_layers` K pointers / V pointers (BF16).
    pub k_ptrs: u64,
    pub v_ptrs: u64,
    pub stride_head: u64,
    pub stride_token: u64,
    pub stride_dim: u64,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub layer_begin: u32,
    pub layer_end: u32,
    pub token_begin: u32,
    pub token_end: u32,
    /// Frame base + section offsets.
    pub frame: u64,
    pub k_payload: u64,
    pub v_payload: u64,
    pub k_meta: u64,
    pub v_meta: u64,
    /// 16, 8 or 4.
    pub bits: u32,
    pub k_group: u32,
    pub v_group: u32,
    /// 0 = HeadDim, 1 = Token.
    pub k_axis: u32,
    pub v_axis: u32,
    pub k_symmetric: u32,
    pub v_symmetric: u32,
}

unsafe extern "C" {
    fn bonsai_encode_chunk(args: *const ChunkArgs, stream: u64) -> i32;
    fn bonsai_decode_chunk(args: *const ChunkArgs, stream: u64) -> i32;
}

pub struct CudaCodec {
    spec: CodecSpec,
}

impl CudaCodec {
    pub fn new(spec: CodecSpec) -> Result<Self> {
        Ok(CudaCodec { spec })
    }

    fn args(&self, _kv: &KvSource, _chunk: &ChunkDesc, _frame: DevicePtr) -> ChunkArgs {
        todo!("fill ChunkArgs; layer pointer arrays are cached per KvSource in device memory")
    }
}

impl KvCodec for CudaCodec {
    fn spec(&self) -> CodecSpec {
        self.spec
    }

    fn encode_chunk(&self, src: &KvSource, chunk: &ChunkDesc, dst: DevicePtr, stream: CudaStream) -> Result<()> {
        let args = self.args(src, chunk, dst);
        check(unsafe { bonsai_encode_chunk(&args, stream.0) })
    }

    fn decode_chunk(&self, frame: DevicePtr, chunk: &ChunkDesc, dst: &KvSink, stream: CudaStream) -> Result<()> {
        let args = self.args(dst, chunk, frame);
        check(unsafe { bonsai_decode_chunk(&args, stream.0) })
    }
}

fn check(code: i32) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(bonsai_core::BonsaiError::Cuda(format!("codec kernel failed: {code}"))) }
}
