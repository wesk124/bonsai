"""Optional GPU-only Triton codec, byte compatible with bonsai-torch-packed-v1.

One destination wire allocation, one fused kernel per tensor, one error-flag
read per quantized cache. Shapes/strides/section offsets stay runtime arguments,
so varying prompt lengths do not cause one compilation per prompt. Raw 16-bit
codec retains actual input dtype, like packed.py. No CPU or reference fallback.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .codecs import HEAD_DIM, TOKEN, Precision
from .kv import _layers
from .packed import PackedCache, _DTYPES, _validate_buffer

BACKEND = "triton_packed_v1"
_RUNTIME = ["T", "H", "S0", "S1", "S2", "S3", "NG", "SC", "QB", "OFF", "N"]


@triton.jit
def _store_half(B, index, value):
    bits = value.to(tl.float16, fp_downcast_rounding="rtne").to(tl.uint16, bitcast=True)
    tl.store(B + index, bits.to(tl.uint8))
    tl.store(B + index + 1, (bits >> 8).to(tl.uint8))


@triton.jit
def _load_half(B, index):
    low = tl.load(B + index).to(tl.uint16)
    high = tl.load(B + index + 1).to(tl.uint16)
    return (low | (high << 8)).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit(do_not_specialize=_RUNTIME)
def _encode(X, B, Error, T, H, S0, S1, S2, S3, NG, SC, QB, OFF, N,
            D: tl.constexpr, G: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
            BITS: tl.constexpr, SYM: tl.constexpr, TOKEN_AXIS: tl.constexpr):
    rr = tl.arange(0, R)
    cc = tl.arange(0, C)
    if TOKEN_AXIS and G % 2 == 0:
        # Coalesced [token, channel] loads; wire groups are [channel, token].
        tiles = tl.cdiv(D, R)
        channel = (tl.program_id(0) % tiles) * R + rr
        rest = tl.program_id(0) // tiles
        tg, bh = rest % NG, rest // NG
        gi = (bh * D + channel) * NG + tg
        valid = channel < D
        token = tl.minimum(tg * G + cc, T - 1)
        offset = (bh // H) * S0 + (bh % H) * S1 + token[:, None] * S2 + channel[None, :] * S3
        x = tl.load(X + offset, (cc[:, None] < G) & valid[None, :], other=0).to(tl.float32).trans()
    else:
        gi = tl.program_id(0) * R + rr
        valid = gi < SC
        if TOKEN_AXIS:
            bh = gi // (D * NG)
            channel = (gi // NG) % D
            token = tl.minimum((gi % NG)[:, None] * G + cc[None, :], T - 1)
            offset = (bh // H)[:, None] * S0 + (bh % H)[:, None] * S1 + token * S2 + channel[:, None] * S3
        else:
            bh = gi // (T * NG)
            token = (gi // NG) % T
            channel = tl.minimum((gi % NG)[:, None] * G + cc[None, :], D - 1)
            offset = (bh // H)[:, None] * S0 + (bh % H)[:, None] * S1 + token[:, None] * S2 + channel * S3
        x = tl.load(X + offset, valid[:, None] & (cc[None, :] < G), other=0).to(tl.float32)
    active = valid[:, None] & (cc[None, :] < G)
    bad_input = tl.sum(tl.sum((active & ~(tl.abs(x) < float("inf"))).to(tl.int32), 1), 0)
    if SYM:
        maxq: tl.constexpr = (1 << (BITS - 1)) - 1
        bound = tl.max(tl.where(cc[None, :] < G, tl.abs(x), 0.0), 1)
        # PyTorch tensor/scalar division precomputes an f32 reciprocal.
        # Exact division differs at f16 rounding midpoints and changes wire bytes.
        inverse = tl.full((R,), 1.0 / maxq, tl.float32)
        scale = (bound * inverse).to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
        z = tl.full((R,), 0.0, tl.float32)
    else:
        maxq: tl.constexpr = (1 << BITS) - 1
        low = tl.min(tl.where(cc[None, :] < G, x, float("inf")), 1)
        high = tl.max(tl.where(cc[None, :] < G, x, -float("inf")), 1)
        inverse = tl.full((R,), 1.0 / maxq, tl.float32)
        scale = ((high - low) * inverse).to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
        z = low.to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
    bad_meta = tl.sum((valid & (~(tl.abs(scale) < float("inf")) | ~(tl.abs(z) < float("inf")))).to(tl.int32), 0)
    if bad_input | bad_meta:
        tl.atomic_or(Error, tl.where(bad_input != 0, 1, 0) | tl.where(bad_meta != 0, 2, 0))
    safe = tl.where(scale == 0, 1.0, scale)
    if SYM:
        q = libdevice.nearbyint(tl.div_rn(x, safe[:, None]))
        q = tl.minimum(tl.maximum(q, -maxq), maxq)
    else:
        q = libdevice.nearbyint(tl.div_rn(x - z[:, None], safe[:, None]))
        q = tl.minimum(tl.maximum(q, 0), maxq)
    q = tl.where(active & (scale[:, None] != 0), q, 0).to(tl.int32)
    if BITS == 8:
        tl.store(B + OFF + gi[:, None] * G + cc[None, :], q.to(tl.uint8), active)
    elif TOKEN_AXIS and G % 2 == 0:
        pair = tl.arange(0, C // 2)
        lowq = tl.gather(q, tl.broadcast_to(pair[None, :] * 2, (R, C // 2)), 1)
        highq = tl.gather(q, tl.broadcast_to(pair[None, :] * 2 + 1, (R, C // 2)), 1)
        packed = (lowq & 15) | ((highq & 15) << 4)
        tl.store(B + OFF + gi[:, None] * (G // 2) + pair[None, :], packed.to(tl.uint8), valid[:, None] & (pair[None, :] < G // 2))
    else:
        # Flatten actual group elements, not the power-of-two reduction padding.
        pair = tl.arange(0, triton.next_power_of_2(R * G // 2))
        e = pair * 2
        flat = q.reshape((R * C,))
        i0 = tl.minimum((e // G) * C + e % G, R * C - 1)
        i1 = tl.minimum(((e + 1) // G) * C + (e + 1) % G, R * C - 1)
        a, b = tl.gather(flat, i0, 0), tl.gather(flat, i1, 0)
        packed = (a & 15) | ((b & 15) << 4)
        byte = tl.program_id(0) * (R * G // 2) + pair
        tl.store(B + OFF + byte, packed.to(tl.uint8), (pair < R * G // 2) & (byte < QB))
    # Mask invalid rows before unaligned byte stores to metadata.
    sbits = scale.to(tl.float16).to(tl.uint16, bitcast=True)
    tl.store(B + OFF + QB + gi * 2, sbits.to(tl.uint8), valid)
    tl.store(B + OFF + QB + gi * 2 + 1, (sbits >> 8).to(tl.uint8), valid)
    if not SYM:
        zbits = z.to(tl.float16).to(tl.uint16, bitcast=True)
        tl.store(B + OFF + QB + SC * 2 + gi * 2, zbits.to(tl.uint8), valid)
        tl.store(B + OFF + QB + SC * 2 + gi * 2 + 1, (zbits >> 8).to(tl.uint8), valid)


@triton.jit(do_not_specialize=_RUNTIME)
def _decode(B, Y, Error, T, H, NG, SC, QB, OFF, N,
            D: tl.constexpr, G: tl.constexpr, BITS: tl.constexpr,
            SYM: tl.constexpr, TOKEN_AXIS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    d, t, bh = i % D, (i // D) % T, i // (D * T)
    if TOKEN_AXIS:
        qindex = ((bh * D + d) * NG + t // G) * G + t % G
    else:
        qindex = ((bh * T + t) * NG + d // G) * G + d % G
    group = qindex // G
    # Invalid lanes use group zero to keep byte loads within frame bounds.
    group = tl.where(valid, group, 0)
    scale = _load_half(B, OFF + QB + group * 2)
    if SYM:
        z = tl.full((BLOCK,), 0.0, tl.float32)
    else:
        z = _load_half(B, OFF + QB + SC * 2 + group * 2)
    bad = valid & (~(tl.abs(scale) < float("inf")) | ~(tl.abs(z) < float("inf")))
    if tl.sum(bad.to(tl.int32), 0):
        tl.atomic_or(Error, 2)
    if BITS == 4:
        code = tl.load(B + OFF + qindex // 2, valid, other=0).to(tl.int32)
        code = (code >> ((qindex % 2) * 4)) & 15
    else:
        code = tl.load(B + OFF + qindex, valid, other=0).to(tl.int32)
    if SYM:
        code = tl.where(code >= (1 << (BITS - 1)), code - (1 << BITS), code)
    out = code.to(tl.float32) * scale
    if not SYM:
        out = out + z
    tl.store(Y + i, out, valid)


@triton.jit(do_not_specialize=_RUNTIME)
def _raw_encode(X, B, T, H, S0, S1, S2, S3, OFF, N,
                D: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr, CONTIG: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CONTIG:
        # Layout-class specialization only: no runtime sequence-length divisions.
        value = tl.load(X + i, i < N, other=0)
    else:
        offset = (i // (H * T * D)) * S0 + ((i // (T * D)) % H) * S1 + ((i // D) % T) * S2 + (i % D) * S3
        value = tl.load(X + offset, i < N, other=0)
    if OFF % E == 0:
        out = (B + OFF).to(X.dtype)
        tl.store(out + i, value, i < N)
    else:
        if E == 2:
            raw = value.to(tl.uint16, bitcast=True)
        elif E == 4:
            raw = value.to(tl.uint32, bitcast=True)
        else:
            raw = value.to(tl.uint64, bitcast=True)
        for byte in tl.static_range(E):
            tl.store(B + OFF + i * E + byte, (raw >> (byte * 8)).to(tl.uint8), i < N)


@triton.jit(do_not_specialize=_RUNTIME)
def _raw_decode(B, Y, OFF, N, E: tl.constexpr, BLOCK: tl.constexpr, BASE_ALIGNED: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # A contiguous uint8 view can itself start at an unaligned byte address.
    if BASE_ALIGNED and OFF % E == 0:
        source = (B + OFF).to(Y.dtype)
        tl.store(Y + i, tl.load(source + i, i < N, other=0), i < N)
    else:
        if E == 2:
            raw = tl.full((BLOCK,), 0, tl.uint16)
        elif E == 4:
            raw = tl.full((BLOCK,), 0, tl.uint32)
        else:
            raw = tl.full((BLOCK,), 0, tl.uint64)
        for byte in tl.static_range(E):
            raw |= tl.load(B + OFF + i * E + byte, i < N, other=0).to(raw.dtype) << (byte * 8)
        tl.store(Y + i, raw.to(Y.dtype.element_ty, bitcast=True), i < N)


def _description(shape, dtype, bits, params=None):
    shape = list(shape)
    if len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("KV tensor must be nonempty [batch, heads, tokens, channels]")
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported KV dtype {dtype}")
    meta = {"version": 1, "shape": shape, "dtype": dtype, "bits": bits}
    if bits == 16:
        return meta, math.prod(shape) * torch.empty((), dtype=_DTYPES[dtype]).element_size()
    if bits not in (4, 8) or params is None or params.group_size <= 0 or params.axis not in (HEAD_DIM, TOKEN):
        raise ValueError("invalid quantized codec or grouping parameters")
    b, h, t, d = shape
    padded_shape = [b, h, d, triton.cdiv(t, params.group_size) * params.group_size] if params.axis == TOKEN else [b, h, t, triton.cdiv(d, params.group_size) * params.group_size]
    count = math.prod(padded_shape)
    qb, sc = triton.cdiv(count * bits, 8), count // params.group_size
    meta.update(axis=params.axis, group_size=params.group_size, symmetric=params.symmetric,
                padded_shape=padded_shape, quant_count=count, quant_bytes=qb, scale_count=sc)
    return meta, qb + sc * (2 if params.symmetric else 4)


def _cuda(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("triton_packed_v1 requires CUDA tensors; no CPU fallback")
    return device


def _check(flag):
    code = flag.item()  # Exactly one device-to-host flag read per quantized cache.
    if code:
        raise ValueError("quantized KV contains nonfinite values or f16 metadata overflows/is nonfinite")


def encode_cache(cache: Any, precision: Precision) -> PackedCache:
    work, layers, cursor, device = [], [], 0, None
    for i, pair in enumerate(_layers(cache)):
        codec = precision.codec_for_layer(i)
        if codec.atom_exact:
            raise ValueError("atom-kv4-exact is not a wire codec")
        row = []
        for x, params in zip(pair, (codec.key, codec.value)):
            current = _cuda(x.device)
            if device is not None and device != current:
                raise ValueError("all KV tensors must be on the same CUDA device")
            device = current
            meta, length = _description(x.shape, str(x.dtype).removeprefix("torch."), codec.bits, params)
            row.append({"offset": cursor, "length": length, "tensor": meta})
            work.append((x, meta, cursor))
            cursor += length
        layers.append(row)
    if not work:
        raise ValueError("cannot encode an empty cache")
    with torch.cuda.device(device):
        buffer = torch.empty(cursor, dtype=torch.uint8, device=device)
        quantized = any(m["bits"] != 16 for _, m, _ in work)
        error = torch.zeros((), dtype=torch.int32, device=device) if quantized else None
        for x, m, offset in work:
            b, h, t, d = m["shape"]
            if m["bits"] == 16:
                _raw_encode[(triton.cdiv(x.numel(), 1024),)](x, buffer, t, h, *x.stride(), offset, x.numel(), d, x.element_size(), 1024, x.is_contiguous())
            else:
                g, sc = m["group_size"], m["scale_count"]
                ng = triton.cdiv(t if m["axis"] == TOKEN else d, g)
                r = 8
                programs = b * h * ng * triton.cdiv(d, r) if m["axis"] == TOKEN and g % 2 == 0 else triton.cdiv(sc, r)
                _encode[(programs,)](x, buffer, error, t, h, *x.stride(), ng, sc, m["quant_bytes"], offset, x.numel(),
                                      d, g, triton.next_power_of_2(g), r, m["bits"], m["symmetric"], m["axis"] == TOKEN,
                                      enable_fp_fusion=False)
        if quantized:
            _check(error)
    return PackedCache(buffer, {"version": 1, "format": "bonsai-torch-packed-v1", "precision": precision.name, "layers": layers})


def decode_cache(packet: PackedCache, device: Any = None, as_legacy: bool = False):
    _validate_buffer(packet)
    if packet.metadata.get("format") != "bonsai-torch-packed-v1":
        raise ValueError("unsupported cache packet format")
    target = _cuda(packet.buffer.device if device is None else device)
    work, cursor = [], 0
    for layer in packet.metadata["layers"]:
        if len(layer) != 2:
            raise ValueError("cache layer must contain K and V")
        for section in layer:
            m = section["tensor"]
            from .codecs import QuantParams
            params = None if m["bits"] == 16 else QuantParams(m["group_size"], m["axis"], m["symmetric"])
            expected, length = _description(m["shape"], m["dtype"], m["bits"], params)
            if m != expected or section["offset"] != cursor or section["length"] != length or cursor + length > packet.buffer.numel():
                raise ValueError("invalid cache section metadata or bounds")
            work.append((m, cursor))
            cursor += length
    if cursor != packet.buffer.numel() or not work:
        raise ValueError("invalid empty cache or trailing payload bytes")
    with torch.cuda.device(target):
        buffer = packet.buffer.to(target)
        quantized = any(m["bits"] != 16 for m, _ in work)
        error = torch.zeros((), dtype=torch.int32, device=target) if quantized else None
        tensors = []
        for m, offset in work:
            out = torch.empty(m["shape"], dtype=_DTYPES[m["dtype"]], device=target)
            b, h, t, d = m["shape"]
            if m["bits"] == 16:
                _raw_decode[(triton.cdiv(out.numel(), 1024),)](buffer, out, offset, out.numel(), out.element_size(), 1024, buffer.data_ptr() % out.element_size() == 0)
            else:
                g = m["group_size"]
                ng = triton.cdiv(t if m["axis"] == TOKEN else d, g)
                _decode[(triton.cdiv(out.numel(), 1024),)](buffer, out, error, t, h, ng, m["scale_count"], m["quant_bytes"], offset, out.numel(),
                                                          d, g, m["bits"], m["symmetric"], m["axis"] == TOKEN, 1024,
                                                          enable_fp_fusion=False)
            tensors.append(out)
        if quantized:
            _check(error)
    pairs = tuple(zip(tensors[::2], tensors[1::2]))
    if as_legacy:
        return pairs
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer
    # In pinned 4.56.2 even ddp_cache_data invokes update -> cat(empty, data).
    # Adopt our already-owned outputs directly, setting the lazy-init fields.
    cache = DynamicCache()
    for k, v in pairs:
        layer = DynamicLayer()
        layer.keys, layer.values = k, v
        layer.dtype, layer.device = k.dtype, k.device
        cache.layers.append(layer)
    return cache
