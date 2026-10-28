"""Codec microbenchmark (Atom's kernel-benchmark methodology, applied to KV codecs).

Times encode+decode of a full Qwen2.5-1.5B KV cache at several context lengths
and reports throughput over raw BF16 bytes, alongside a device-to-device
memcpy of the same size (the roofline). The output is the
`[codec.adaptive.cost_model]` table for bonsai-ctl.

Until the CUDA kernels exist (milestone 2) this times the PyTorch reference,
which is much slower than the kernels will be. Use those numbers only to
exercise the pipeline, not to calibrate the cost model.

    python -m bonsai_bench.codec_bench --seq-lens 1024,8192,32768
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from .codecs import load_precision, roundtrip_layer


def _timeit(fn, device: str, iters: int) -> float:
    fn()  # warm-up
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1e3 / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precisions", default="bf16,int8,int4,atom-kv4")
    ap.add_argument("--seq-lens", default="1024,8192")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    try:
        import bonsai  # noqa: F401
        backend = "native"
    except ImportError:
        backend = "torch-reference"

    rates: dict[str, float] = {}
    for seq in (int(s) for s in a.seq_lens.split(",")):
        k = torch.randn(1, a.kv_heads, seq, a.head_dim, dtype=torch.bfloat16, device=a.device)
        v = torch.randn_like(k)
        raw = 2 * k.numel() * k.element_size() * a.layers
        dst = torch.empty_like(k)
        t_copy = _timeit(lambda: (dst.copy_(k), dst.copy_(v)), a.device, a.iters) * a.layers
        for pname in a.precisions.split(","):
            prec = load_precision(pname, head_dim=a.head_dim)
            if prec.default.bits >= 16:   # BF16 "codec" = strided pack into the frame (+ unpack): two copies
                fn = lambda: (dst.copy_(k), dst.copy_(v), dst.copy_(k), dst.copy_(v))  # noqa: E731
            else:
                fn = lambda: roundtrip_layer(k, v, prec.default)  # noqa: E731
            t = _timeit(fn, a.device, a.iters) * a.layers
            gbps = raw / t / 1e9
            rates[pname] = gbps
            print(json.dumps({"bench": "codec", "backend": backend, "precision": pname, "seq_len": seq,
                              "raw_mib": raw / 2**20, "roundtrip_ms": t * 1e3, "roundtrip_gbps": gbps,
                              "memcpy_gbps": raw / t_copy / 1e9}))

    # Encode and decode are timed together here; split evenly (2x each) until the native API times them separately.
    print(f"\n# [codec.adaptive.cost_model] rates from backend={backend}, last seq_len")
    for fam in ("bf16", "int8", "int4"):
        if fam in rates:
            g = 2 * rates[fam]
            print(f"{fam} = {{ encode_gbps = {g:.1f}, decode_gbps = {g:.1f} }}")


if __name__ == "__main__":
    main()
