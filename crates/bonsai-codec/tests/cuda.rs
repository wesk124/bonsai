//! Run explicitly on a GPU: cargo test -p bonsai-codec --features cuda --test cuda -- --ignored
#![cfg(feature = "cuda")]
use bonsai_codec::{
    KvCodec,
    cuda::CudaCodec,
    reference::{self, HostKv},
};
use bonsai_core::{
    ChunkPlan, ChunkSpec, CodecSpec, CudaStream, DevicePtr, ElemType, GroupAxis, KvLayout,
    KvSource, LayerKv, QuantBits, QuantParams,
};
use half::bf16;
use std::ffi::c_void;
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
}
fn ok(status: i32) {
    assert_eq!(status, 0, "CUDA runtime status {status}");
}
struct Buffer {
    ptr: *mut c_void,
    bytes: usize,
}
impl Buffer {
    fn new(bytes: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        ok(unsafe { cudaMalloc(&mut ptr, bytes) });
        Self { ptr, bytes }
    }
    fn upload<T>(&self, data: &[T]) {
        assert_eq!(std::mem::size_of_val(data), self.bytes);
        ok(unsafe { cudaMemcpy(self.ptr, data.as_ptr().cast(), self.bytes, 1) });
    }
    fn read<T: Default + Clone>(&self) -> Vec<T> {
        assert_eq!(self.bytes % std::mem::size_of::<T>(), 0);
        let mut out = vec![T::default(); self.bytes / std::mem::size_of::<T>()];
        ok(unsafe { cudaMemcpy(out.as_mut_ptr().cast(), self.ptr, self.bytes, 2) });
        out
    }
    fn addr(&self) -> DevicePtr {
        DevicePtr(self.ptr as u64)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        ok(unsafe { cudaFree(self.ptr) });
    }
}
struct Stream(*mut c_void);
impl Stream {
    fn new() -> Self {
        let mut s = std::ptr::null_mut();
        ok(unsafe { cudaStreamCreateWithFlags(&mut s, 1) });
        Self(s)
    }
    fn id(&self) -> CudaStream {
        CudaStream(self.0 as u64)
    }
    fn sync(&self) {
        ok(unsafe { cudaStreamSynchronize(self.0) });
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        ok(unsafe { cudaStreamDestroy(self.0) });
    }
}
struct DeviceKv {
    view: KvSource,
    buffers: Vec<(Buffer, Buffer)>,
    original: Vec<(Vec<u16>, Vec<u16>)>,
}
impl DeviceKv {
    fn from_host(src: &HostKv) -> Self {
        let mut layers = Vec::new();
        let mut buffers = Vec::new();
        let mut original = Vec::new();
        for l in 0..src.layout.num_layers as usize {
            // Different noncontiguous strides on every layer, channel gaps,
            // head padding, and a trailing allocation guard.
            let sd = 2 + l;
            let st = src.layout.head_dim as usize * sd + 3;
            let sh = src.seq_len as usize * st + 7;
            let n = src.layout.num_kv_heads as usize * sh + 13;
            let mut k = vec![0xa55a; n];
            let mut v = k.clone();
            for h in 0..src.layout.num_kv_heads as usize {
                for t in 0..src.seq_len as usize {
                    for d in 0..src.layout.head_dim as usize {
                        let flat =
                            (h * src.seq_len as usize + t) * src.layout.head_dim as usize + d;
                        let at = h * sh + t * st + d * sd;
                        k[at] = src.k[l][flat];
                        v[at] = src.v[l][flat];
                    }
                }
            }
            let kb = Buffer::new(n * 2);
            let vb = Buffer::new(n * 2);
            kb.upload(&k);
            vb.upload(&v);
            layers.push(LayerKv {
                k: kb.addr(),
                v: vb.addr(),
                stride_head: sh as u64,
                stride_token: st as u64,
                stride_dim: sd as u64,
            });
            buffers.push((kb, vb));
            original.push((k, v));
        }
        Self {
            view: KvSource {
                layout: src.layout,
                seq_len: src.seq_len,
                device: 0,
                layers,
            },
            buffers,
            original,
        }
    }
    fn check_matches(&self, expected: &HostKv) {
        for (l, ((k, v), (initial_k, initial_v))) in
            self.buffers.iter().zip(&self.original).enumerate()
        {
            let mut ek = initial_k.clone();
            let mut ev = initial_v.clone();
            let strides = &self.view.layers[l];
            for h in 0..expected.layout.num_kv_heads as usize {
                for t in 0..expected.seq_len as usize {
                    for d in 0..expected.layout.head_dim as usize {
                        let flat = (h * expected.seq_len as usize + t)
                            * expected.layout.head_dim as usize
                            + d;
                        let at = h * strides.stride_head as usize
                            + t * strides.stride_token as usize
                            + d * strides.stride_dim as usize;
                        ek[at] = expected.k[l][flat];
                        ev[at] = expected.v[l][flat];
                    }
                }
            }
            assert_eq!(k.read::<u16>(), ek, "layer {l} K/padding differs");
            assert_eq!(v.read::<u16>(), ev, "layer {l} V/padding differs");
        }
    }
}
fn params(axis: GroupAxis, symmetric: bool) -> QuantParams {
    QuantParams {
        axis,
        symmetric,
        group_size: 4,
    }
}
#[test]
#[ignore = "requires a CUDA GPU; invoked by tests/run_cuda.sbatch"]
fn native_wire_and_roundtrip_match_cpu_on_strided_tails() {
    let layout = KvLayout {
        num_layers: 3,
        num_kv_heads: 3,
        head_dim: 5,
        dtype: ElemType::Bf16,
    };
    let mut src = HostKv::zeros(layout, 11);
    for (l, x) in src.k.iter_mut().chain(&mut src.v).enumerate() {
        for (i, x) in x.iter_mut().enumerate() {
            *x = bf16::from_f32((((l * 11 + i) % 51) as f32 - 25.0) * 0.125).to_bits();
        }
    }
    let input = DeviceKv::from_host(&src);
    let stream = Stream::new();
    let mut specs = vec![CodecSpec::Bf16];
    for bits in [QuantBits::Int8, QuantBits::Int4Packed] {
        for axis in [GroupAxis::HeadDim, GroupAxis::Token] {
            for symmetric in [false, true] {
                specs.push(CodecSpec::Quant {
                    bits,
                    key: params(axis, symmetric),
                    value: params(axis, symmetric),
                });
            }
        }
    }
    specs.push(CodecSpec::Quant {
        bits: QuantBits::Int4Packed,
        key: params(GroupAxis::Token, false),
        value: params(GroupAxis::HeadDim, false),
    });
    for spec in specs {
        let mut expected = HostKv::zeros(layout, 11);
        let output = DeviceKv::from_host(&expected);
        let codec = CudaCodec::new(spec).unwrap();
        // Only first 9 of 11 tokens are transferred: logical trailing tokens,
        // stride gaps, and allocation guards must remain untouched.
        let plan = ChunkPlan::build(
            layout,
            spec,
            9,
            ChunkSpec {
                tokens_per_chunk: 4,
                layers_per_chunk: 2,
            },
        )
        .unwrap();
        for c in &plan.chunks {
            let mut frame = vec![0; c.frame_len as usize];
            reference::encode_chunk(&spec, &src, c, &mut frame).unwrap();
            let device_frame = Buffer::new(frame.len());
            codec
                .encode_chunk(&input.view, c, device_frame.addr(), stream.id())
                .unwrap();
            stream.sync();
            assert_eq!(
                device_frame.read::<u8>(),
                frame,
                "wire differs for {spec:?}, chunk {}",
                c.index
            );
            codec
                .decode_chunk(device_frame.addr(), c, &output.view, stream.id())
                .unwrap();
            stream.sync();
            reference::decode_chunk(&spec, &frame, c, &mut expected).unwrap();
            output.check_matches(&expected);
        }
        input.check_matches(&src);
    }
}
#[test]
#[ignore = "requires a CUDA GPU; invoked by tests/run_cuda.sbatch"]
fn native_rejects_invalid_numeric_ranges_before_writes() {
    let layout = KvLayout {
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: 5,
        dtype: ElemType::Bf16,
    };
    let spec = CodecSpec::Quant {
        bits: QuantBits::Int8,
        key: params(GroupAxis::HeadDim, false),
        value: params(GroupAxis::HeadDim, false),
    };
    let plan = ChunkPlan::build(layout, spec, 3, ChunkSpec::WHOLE).unwrap();
    let c = &plan.chunks[0];
    let frame = Buffer::new(c.frame_len as usize);
    let guard = vec![0x5au8; c.frame_len as usize];
    let stream = Stream::new();
    let codec = CudaCodec::new(spec).unwrap();
    for x in [f32::NAN, f32::INFINITY, 1e8] {
        let mut src = HostKv::zeros(layout, 3);
        src.k[0][0] = bf16::from_f32(x).to_bits();
        let input = DeviceKv::from_host(&src);
        frame.upload(&guard);
        assert!(
            codec
                .encode_chunk(&input.view, c, frame.addr(), stream.id())
                .is_err()
        );
        stream.sync();
        assert_eq!(frame.read::<u8>(), guard);
    }
    let src = HostKv::zeros(layout, 3);
    let input = DeviceKv::from_host(&src);
    codec
        .encode_chunk(&input.view, c, frame.addr(), stream.id())
        .unwrap();
    stream.sync();
    let mut malformed = frame.read::<u8>();
    let at = c.sections.k_meta.offset as usize;
    malformed[at..at + 2].copy_from_slice(&half::f16::NAN.to_bits().to_le_bytes());
    frame.upload(&malformed);
    let output = DeviceKv::from_host(&src);
    assert!(
        codec
            .decode_chunk(frame.addr(), c, &output.view, stream.id())
            .is_err()
    );
    stream.sync();
    output.check_matches(&src);
}
