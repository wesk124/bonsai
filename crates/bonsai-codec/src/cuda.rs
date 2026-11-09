//! CUDA backend. Each layer launches with its own pointers and strides.
//!
//! Encode validates numerical ranges on the caller's stream before packing;
//! that validation synchronizes the stream so nonfinite inputs and f16 metadata
//! overflow can be returned as errors. Decode similarly validates metadata
//! before enqueuing reconstruction. These checks synchronize the caller stream.
use crate::KvCodec;
use bonsai_core::{
    BonsaiError, ChunkDesc, CodecSpec, CudaStream, DevicePtr, GroupAxis, KvSink, KvSource, Result,
};

/// C ABI mirror of `kernels/bonsai_codec.cuh`. Pointers refer to a single layer;
/// `layer_begin` is its frame-local index. Strides are in BF16 elements.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChunkArgs {
    pub k_ptr: u64,
    pub v_ptr: u64,
    pub stride_head: u64,
    pub stride_token: u64,
    pub stride_dim: u64,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub layer_begin: u32,
    pub layer_end: u32,
    pub token_begin: u32,
    pub token_end: u32,
    pub frame: u64,
    pub k_payload: u64,
    pub v_payload: u64,
    pub k_meta: u64,
    pub v_meta: u64,
    pub bits: u32,
    pub k_group: u32,
    pub v_group: u32,
    pub k_axis: u32,
    pub v_axis: u32,
    pub k_symmetric: u32,
    pub v_symmetric: u32,
}
unsafe extern "C" {
    fn bonsai_validate_chunk(args: *const ChunkArgs, stream: u64) -> i32;
    fn bonsai_validate_metadata(args: *const ChunkArgs, stream: u64) -> i32;
    fn bonsai_zero_frame(frame: u64, bytes: u64, stream: u64) -> i32;
    fn bonsai_encode_chunk(args: *const ChunkArgs, stream: u64) -> i32;
    fn bonsai_decode_chunk(args: *const ChunkArgs, stream: u64) -> i32;
}
pub struct CudaCodec {
    spec: CodecSpec,
}
impl CudaCodec {
    pub fn new(spec: CodecSpec) -> Result<Self> {
        Ok(Self { spec })
    }
    fn args(&self, kv: &KvSource, chunk: &ChunkDesc, frame: DevicePtr) -> Result<Vec<ChunkArgs>> {
        crate::reference::validate_frame(
            &self.spec,
            &kv.layout,
            kv.seq_len,
            chunk,
            usize::try_from(chunk.frame_len)
                .map_err(|_| BonsaiError::Codec("frame too large".into()))?,
        )?;
        if frame.0 == 0
            || !frame.0.is_multiple_of(4)
            || kv.layers.len() != kv.layout.num_layers as usize
        {
            return Err(BonsaiError::Codec(
                "unaligned/null frame or invalid layer count".into(),
            ));
        }
        if frame.0.checked_add(chunk.frame_len).is_none() {
            return Err(BonsaiError::Codec("frame address overflow".into()));
        }
        let (bits, kg, vg, ka, va, ks, vs) = match self.spec {
            CodecSpec::Bf16 => (16, 1, 1, 0, 0, 1, 1),
            CodecSpec::Quant { bits, key, value } => (
                bits.bits() as u32,
                key.group_size as u32,
                value.group_size as u32,
                axis(key.axis),
                axis(value.axis),
                key.symmetric as u32,
                value.symmetric as u32,
            ),
        };
        let mut out = Vec::new();
        for (i, l) in chunk.layers.clone().enumerate() {
            let layer = &kv.layers[l as usize];
            if layer.k.0 == 0
                || layer.v.0 == 0
                || !layer.k.0.is_multiple_of(2)
                || !layer.v.0.is_multiple_of(2)
                || layer.stride_dim == 0
                || layer.stride_token == 0
                || layer.stride_head == 0
            {
                return Err(BonsaiError::Codec(
                    "null/unaligned tensor or zero stride".into(),
                ));
            }
            let maxoff = layer
                .stride_head
                .checked_mul((kv.layout.num_kv_heads - 1) as u64)
                .and_then(|h| {
                    layer
                        .stride_token
                        .checked_mul((chunk.tokens.end - 1) as u64)
                        .and_then(|t| h.checked_add(t))
                })
                .and_then(|ht| {
                    layer
                        .stride_dim
                        .checked_mul((kv.layout.head_dim - 1) as u64)
                        .and_then(|d| ht.checked_add(d))
                })
                .and_then(|x| x.checked_mul(2));
            if maxoff.is_none_or(|off| {
                layer.k.0.checked_add(off).is_none() || layer.v.0.checked_add(off).is_none()
            }) {
                return Err(BonsaiError::Codec("tensor stride address overflow".into()));
            }
            out.push(ChunkArgs {
                k_ptr: layer.k.0,
                v_ptr: layer.v.0,
                stride_head: layer.stride_head,
                stride_token: layer.stride_token,
                stride_dim: layer.stride_dim,
                num_kv_heads: kv.layout.num_kv_heads as u32,
                head_dim: kv.layout.head_dim as u32,
                layer_begin: i as u32,
                layer_end: i as u32 + 1,
                token_begin: chunk.tokens.start,
                token_end: chunk.tokens.end,
                frame: frame.0,
                k_payload: chunk.sections.k_payload.offset,
                v_payload: chunk.sections.v_payload.offset,
                k_meta: chunk.sections.k_meta.offset,
                v_meta: chunk.sections.v_meta.offset,
                bits,
                k_group: kg,
                v_group: vg,
                k_axis: ka,
                v_axis: va,
                k_symmetric: ks,
                v_symmetric: vs,
            });
        }
        Ok(out)
    }
}
fn axis(a: GroupAxis) -> u32 {
    match a {
        GroupAxis::HeadDim => 0,
        GroupAxis::Token => 1,
    }
}
impl KvCodec for CudaCodec {
    fn spec(&self) -> CodecSpec {
        self.spec
    }
    fn encode_chunk(
        &self,
        src: &KvSource,
        chunk: &ChunkDesc,
        dst: DevicePtr,
        stream: CudaStream,
    ) -> Result<()> {
        let args = self.args(src, chunk, dst)?;
        for a in &args {
            check(unsafe { bonsai_validate_chunk(a, stream.0) })?;
        }
        check(unsafe { bonsai_zero_frame(dst.0, chunk.frame_len, stream.0) })?;
        for a in &args {
            check(unsafe { bonsai_encode_chunk(a, stream.0) })?;
        }
        Ok(())
    }
    fn decode_chunk(
        &self,
        frame: DevicePtr,
        chunk: &ChunkDesc,
        dst: &KvSink,
        stream: CudaStream,
    ) -> Result<()> {
        let args = self.args(dst, chunk, frame)?;
        for a in &args {
            check(unsafe { bonsai_validate_metadata(a, stream.0) })?;
        }
        for a in &args {
            check(unsafe { bonsai_decode_chunk(a, stream.0) })?;
        }
        Ok(())
    }
}
fn check(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        -1 => Err(BonsaiError::Codec("nonfinite quantization input".into())),
        -2 => Err(BonsaiError::Codec(
            "quantization metadata exceeds finite f16 range".into(),
        )),
        -3 => Err(BonsaiError::Codec("invalid quantization metadata".into())),
        _ => Err(BonsaiError::Cuda(format!("codec CUDA status {code}"))),
    }
}
