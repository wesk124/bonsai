"""Measure actual packed PyTorch encode and decode separately.

Timings are synchronized wall time, including Python dispatch, validation,
packing, metadata serialization/parsing, and allocation. Backend is explicitly
``torch-packed`` regardless of whether a package named bonsai is importable.
The emitted cost-model rates use raw uncompressed bytes divided by each measured
stage time. These are measurements of the reference backend, not CUDA kernels.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .codecs import load_precision
from .packed import PackedCache, encode_cache, decode_cache


def _timeit(fn, device: str, iters: int) -> float:
    if iters < 1:
        raise ValueError("iters must be positive")
    fn()
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precisions", default="bf16,int8,int4,atom-kv4")
    ap.add_argument("--seq-lens", default="1024,8192")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if min(a.layers, a.kv_heads, a.head_dim, a.iters) < 1:
        ap.error("layers, kv-heads, head-dim and iters must be positive")
    rates = {}
    for seq in (int(s) for s in a.seq_lens.split(",")):
        k = torch.randn(1, a.kv_heads, seq, a.head_dim, dtype=torch.bfloat16, device=a.device)
        v = torch.randn_like(k)
        cache = tuple((k, v) for _ in range(a.layers))
        raw = 2 * k.numel() * k.element_size() * a.layers
        source = torch.empty(raw, dtype=torch.uint8, device=a.device)
        destination = torch.empty_like(source)
        copy_s = _timeit(lambda: destination.copy_(source), a.device, a.iters)
        del source, destination
        for pname in a.precisions.split(","):
            prec = load_precision(pname, a.config, a.head_dim)
            def encode():
                packet = encode_cache(cache, prec)
                return packet, packet.metadata_bytes()
            packet, header = encode()
            def decode():
                return decode_cache(PackedCache.from_wire(packet.buffer, header), as_legacy=True)
            encode_s = _timeit(encode, a.device, a.iters)
            decode_s = _timeit(decode, a.device, a.iters)
            rates[pname] = (raw / encode_s / 1e9, raw / decode_s / 1e9)
            print(json.dumps({"bench": "codec", "backend": "torch-packed", "format": packet.metadata["format"],
                              "timing": "synchronized-wall", "precision": pname, "seq_len": seq,
                              "raw_bytes": raw, "payload_bytes": packet.payload_nbytes,
                              "metadata_bytes": len(header), "wire_bytes": packet.wire_nbytes,
                              "compression_ratio": raw / packet.wire_nbytes,
                              "encode_ms": encode_s * 1e3, "decode_ms": decode_s * 1e3,
                              "encode_gbps": rates[pname][0], "decode_gbps": rates[pname][1],
                              "roundtrip_ms": (encode_s + decode_s) * 1e3,
                              "roundtrip_gbps": raw / (encode_s + decode_s) / 1e9,
                              "memcpy_gbps": raw / copy_s / 1e9}), flush=True)
    print("\n# [codec.adaptive.cost_model] measured torch-packed rates; last seq_len")
    for fam in ("bf16", "int8", "int4"):
        if fam in rates:
            enc, dec = rates[fam]
            print(f"{fam} = {{ encode_gbps = {enc:.6g}, decode_gbps = {dec:.6g} }}")


if __name__ == "__main__":
    main()
