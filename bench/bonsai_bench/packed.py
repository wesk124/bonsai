"""Portable packed KV frames implemented with device-local PyTorch operations.

The contiguous uint8 ``buffer`` can be sent by any transport; canonical JSON
``metadata_bytes()`` describes its slices. No tensor values are serialized in
JSON. Both payload padding and the JSON header count towards ``wire_nbytes``.
INT4 uses low nibble first, f16 scales/offsets, K token groups and V channel
groups as specified by the selected Codec. This is a PyTorch backend, not a
native extension and not the Rust frame ABI.
"""
from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import torch

from .codecs import HEAD_DIM, TOKEN, Precision, QuantParams

_DTYPES = {str(x).removeprefix("torch."): x for x in
           (torch.bfloat16, torch.float16, torch.float32, torch.float64)}


def _json(meta: dict) -> bytes:
    return json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclasses.dataclass
class PackedTensor:
    buffer: torch.Tensor
    metadata: dict

    @property
    def payload(self) -> torch.Tensor:
        return self.buffer

    def metadata_bytes(self) -> bytes:
        return _json(self.metadata)

    @property
    def wire_nbytes(self) -> int:
        return self.buffer.numel() + len(self.metadata_bytes())

    @property
    def payload_nbytes(self) -> int:
        return self.buffer.numel()

    @classmethod
    def from_wire(cls, buffer: torch.Tensor, metadata: bytes | dict):
        if isinstance(metadata, bytes):
            metadata = json.loads(metadata)
        return cls(buffer, metadata)


@dataclasses.dataclass
class PackedCache(PackedTensor):
    pass


def _bytes(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().reshape(-1).view(torch.uint8)


def _group(x: torch.Tensor, p: QuantParams):
    if p.group_size <= 0 or p.axis not in (HEAD_DIM, TOKEN):
        raise ValueError("invalid grouping parameters")
    y = x.float().transpose(-1, -2) if p.axis == TOKEN else x.float()
    n = y.shape[-1]
    pad = (-n) % p.group_size
    if pad:
        y = torch.cat((y, y[..., -1:].expand(*y.shape[:-1], pad)), dim=-1)
    return y.reshape(*y.shape[:-1], -1, p.group_size), list(y.shape)


def encode_tensor(x: torch.Tensor, bits: int, params: QuantParams | None = None) -> PackedTensor:
    """Encode [batch, heads, tokens, channels] without moving values to CPU.

    16-bit codec is raw passthrough (normally BF16); a non-BF16 input retains
    its actual dtype and actual byte count, which is useful for reference tests.
    """
    if x.ndim != 4 or any(n == 0 for n in x.shape):
        raise ValueError("KV tensor must be a nonempty [batch, heads, tokens, channels] tensor")
    dtype = str(x.dtype).removeprefix("torch.")
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported KV dtype {x.dtype}")
    meta = {"version": 1, "shape": list(x.shape), "dtype": dtype, "bits": bits}
    if bits == 16:
        return PackedTensor(_bytes(x).clone(), meta)
    if bits not in (4, 8) or params is None:
        raise ValueError("quantized codecs require bits=4/8 and grouping parameters")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("quantized KV contains nonfinite values")
    g, padded_shape = _group(x, params)
    if params.symmetric:
        qmax = (1 << (bits - 1)) - 1
        scale = (g.abs().amax(-1, keepdim=True) / qmax).half()
        offset = None
        sf = scale.float()
        q = (g / torch.where(sf == 0, 1.0, sf)).round().clamp(-qmax, qmax)
        q = torch.where(sf == 0, 0.0, q).to(torch.int8)
    else:
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        scale, offset = ((hi - lo) / ((1 << bits) - 1)).half(), lo.half()
        sf = scale.float()
        q = ((g - offset.float()) / torch.where(sf == 0, 1.0, sf)).round().clamp(0, (1 << bits) - 1)
        q = torch.where(sf == 0, 0.0, q).to(torch.uint8)
    if not bool(torch.isfinite(scale).all()) or (offset is not None and not bool(torch.isfinite(offset).all())):
        raise ValueError("quantization metadata overflows f16")
    q = q.reshape(-1).to(torch.uint8)
    quant_count = q.numel()
    if bits == 4:
        if q.numel() % 2:
            q = torch.cat((q, q.new_zeros(1)))
        q = (q[::2] & 15) | ((q[1::2] & 15) << 4)
    parts = [q, _bytes(scale)]
    if offset is not None:
        parts.append(_bytes(offset))
    meta.update({"axis": params.axis, "group_size": params.group_size,
                 "symmetric": params.symmetric, "padded_shape": padded_shape,
                 "quant_count": quant_count, "quant_bytes": q.numel(), "scale_count": scale.numel()})
    return PackedTensor(torch.cat(parts), meta)


def _validate_buffer(packet: PackedTensor):
    if packet.buffer.dtype != torch.uint8 or packet.buffer.ndim != 1 or not packet.buffer.is_contiguous():
        raise ValueError("wire buffer must be contiguous, one-dimensional uint8")
    if packet.metadata.get("version") != 1:
        raise ValueError("unsupported packet version")


def decode_tensor(packet: PackedTensor, device: Any = None) -> torch.Tensor:
    _validate_buffer(packet)
    m = packet.metadata
    buf = packet.buffer if device is None else packet.buffer.to(device)
    shape = m["shape"]
    dtype = _DTYPES[m["dtype"]]
    if m["bits"] == 16:
        expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if buf.numel() != expected:
            raise ValueError("raw tensor payload length mismatch")
        return buf.clone().view(dtype).reshape(shape)
    n, qb, sc = m["quant_count"], m["quant_bytes"], m["scale_count"]
    if buf.numel() != qb + sc * (2 if m["symmetric"] else 4):
        raise ValueError("quantized tensor payload length mismatch")
    q = buf[:qb]
    if m["bits"] == 4:
        q = torch.stack((q & 15, q >> 4), dim=-1).reshape(-1)[:n]
    if m["symmetric"]:
        q = q.to(torch.int16)
        cutoff = 1 << (m["bits"] - 1)
        q = torch.where(q >= cutoff, q - (1 << m["bits"]), q)
    grouped_shape = [*m["padded_shape"][:-1], -1, m["group_size"]]
    # clone before view: concatenated K/V slices need not have f16 alignment.
    scale = buf[qb:qb + sc * 2].clone().view(torch.float16).float().reshape(*grouped_shape[:-1], 1)
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("nonfinite quantization scale in packet")
    out = q.float().reshape(grouped_shape) * scale
    if not m["symmetric"]:
        offset = buf[qb + sc * 2:].clone().view(torch.float16).float().reshape(*grouped_shape[:-1], 1)
        if not bool(torch.isfinite(offset).all()):
            raise ValueError("nonfinite quantization offset in packet")
        out = out + offset
    out = out.reshape(m["padded_shape"])
    if m["axis"] == TOKEN:
        out = out[..., :shape[-2]].transpose(-1, -2)
    else:
        out = out[..., :shape[-1]]
    return out.contiguous().to(dtype)


def encode_cache(cache: Any, precision: Precision) -> PackedCache:
    from .kv import _layers
    parts, layers, cursor = [], [], 0
    for i, pair in enumerate(_layers(cache)):
        codec = precision.codec_for_layer(i)
        if codec.atom_exact:
            raise ValueError("atom-kv4-exact is a fake-quantization comparison, not a wire codec")
        row = []
        for x, p in zip(pair, (codec.key, codec.value)):
            packet = encode_tensor(x, codec.bits, p)
            parts.append(packet.buffer)
            row.append({"offset": cursor, "length": packet.buffer.numel(), "tensor": packet.metadata})
            cursor += packet.buffer.numel()
        layers.append(row)
    if not parts:
        raise ValueError("cannot encode an empty cache")
    return PackedCache(torch.cat(parts), {"version": 1, "format": "bonsai-torch-packed-v1",
                                        "precision": precision.name, "layers": layers})


def decode_cache(packet: PackedCache, device: Any = None, as_legacy: bool = False):
    """Return a fresh cache, never aliasing or extending the sender's cache."""
    _validate_buffer(packet)
    if packet.metadata.get("format") != "bonsai-torch-packed-v1":
        raise ValueError("unsupported cache packet format")
    buf = packet.buffer if device is None else packet.buffer.to(device)
    pairs, cursor = [], 0
    for layer in packet.metadata["layers"]:
        pair = []
        if len(layer) != 2:
            raise ValueError("cache layer must contain K and V")
        for section in layer:
            offset, length = section["offset"], section["length"]
            if offset != cursor or length < 0 or offset + length > buf.numel():
                raise ValueError("invalid cache section bounds")
            pair.append(decode_tensor(PackedTensor(buf[offset:offset + length], section["tensor"])))
            cursor += length
        pairs.append(tuple(pair))
    if cursor != buf.numel():
        raise ValueError("unreferenced trailing payload bytes")
    if as_legacy:
        return tuple(pairs)
    from transformers import DynamicCache
    cache = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        cache.update(k, v, i)
    return cache
