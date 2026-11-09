#!/usr/bin/env python3
"""Exact-wire, real-Qwen-cache codec comparison; never generates continuations.

Each prompt is prefetched once with the pinned Qwen model. The two backends
encode the same distinct per-layer tensors. Full packets and all cross-decoded
BF16 tensors must match before warmed, synchronized wall-clock timing begins.
A Triton JIT cache miss during timing fails the run before compilation/loading.
Outputs require a fresh directory; frozen experiments are never overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
# Record only operational settings. Authentication variables are never copied.
PROVENANCE_ENV = (
    "TRITON_CACHE_DIR", "TRITON_INTERPRET", "TRITON_ALWAYS_COMPILE",
    "TRITON_DEBUG", "TRITON_OVERRIDE_ARCH", "TRITON_F32_DEFAULT",
    "TRITON_DEFAULT_FP_FUSION", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
    "HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_CACHE", "TRANSFORMERS_OFFLINE",
    "CUBLAS_WORKSPACE_CONFIG", "CUDA_HOME", "CUDA_PATH", "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "LD_LIBRARY_PATH",
)


class GateFailure(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def emit(path, value):
    with path.open("a") as stream:
        stream.write(canonical(value).decode() + "\n")
        stream.flush()
    print(canonical(value).decode(), flush=True)


def command_output(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"argv": args, "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def source_files():
    paths = list((ROOT / "bench/bonsai_bench").glob("*.py"))
    paths += [Path(__file__).resolve(), Path(__file__).resolve().with_suffix(".sbatch"), ROOT / "scripts/native_env.sh"]
    return {str(p.relative_to(ROOT)): p.read_bytes() for p in sorted(set(paths))}


def source_hashes(files):
    return {name: sha(data) for name, data in files.items()}


class JitMonitor:
    """Preserve existing Triton hooks and permit disk-cache loads during warmup.

    Triton invokes jit_cache_hook on an in-process specialization miss, even
    when its compiler can load an existing disk artifact. Both are forbidden
    during measured intervals. Hooks do not run on steady-state cache hits.
    """
    def __init__(self, triton):
        self.runtime = triton.knobs.runtime
        self.previous = self.runtime.jit_cache_hook
        self.phase = "setup"
        self.events = []
        self.runtime.jit_cache_hook = self.hook

    def hook(self, **info):
        event = {"phase": self.phase, "signature_sha256": sha(str(info.get("key")).encode()),
                 "kernel": str(info.get("repr")), "already_compiled": info.get("already_compiled")}
        self.events.append(event)
        if self.phase.startswith("timing:"):
            raise GateFailure("Triton specialization/cache load requested inside timed interval: " + event["kernel"])
        if self.previous is not None:
            return self.previous(**info)
        return None

    @contextmanager
    def during(self, phase):
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def close(self):
        self.runtime.jit_cache_hook = self.previous


def cache_hashes(cache, layers):
    """Hash every raw BF16 byte in canonical layer/K/V/logical tensor order."""
    digest = hashlib.sha256()
    result = []
    total = 0
    for index, pair in enumerate(layers(cache)):
        row = []
        for name, tensor in zip(("k", "v"), pair):
            raw = tensor.detach().contiguous().view(__import__("torch").uint8).cpu().numpy()
            data = memoryview(raw).cast("B")
            digest.update(data)
            total += len(data)
            row.append({"name": name, "shape": list(tensor.shape), "stride": list(tensor.stride()),
                        "dtype": str(tensor.dtype), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        result.append({"layer": index, "tensors": row})
    return {"sha256": digest.hexdigest(), "bytes": total, "layers": result}


def tensor_hash(tensor):
    raw = tensor.detach().contiguous().view(__import__("torch").uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


def exact_cache(actual, expected, layers, torch, label):
    a, b = layers(actual), layers(expected)
    if len(a) != len(b):
        raise GateFailure(label + ": layer count differs")
    for index, (ap, bp) in enumerate(zip(a, b)):
        if len(ap) != 2 or len(bp) != 2:
            raise GateFailure(label + ": cache layer does not contain exactly K and V")
        for name, x, y in zip(("k", "v"), ap, bp):
            if x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16 or x.shape != y.shape:
                raise GateFailure(f"{label}: layer {index} {name} shape/dtype differs")
            if not torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)):
                raise GateFailure(f"{label}: layer {index} {name} BF16 bits differ")


def validate_actual_cache(cache, layers, model, sample, torch):
    pairs = layers(cache)
    if len(pairs) != model.config.num_hidden_layers:
        raise GateFailure("actual model cache layer count differs from config")
    seen = {}
    for index, pair in enumerate(pairs):
        for name, x in zip(("k", "v"), pair):
            if x.dtype != torch.bfloat16 or x.ndim != 4 or x.shape[0] != 1 or x.shape[-2] != sample["input_length"]:
                raise GateFailure(f"invalid actual BF16 KV tensor at layer {index} {name}")
            address = x.untyped_storage().data_ptr()
            if address in seen and seen[address] != index:
                raise GateFailure(f"source cache aliases storage across layers {seen[address]} and {index}")
            seen[address] = index
    return {"layers": len(pairs), "tensor_count": len(pairs) * 2,
            "distinct_layer_storage_verified": True,
            "raw_bytes": sum(x.numel() * x.element_size() for pair in pairs for x in pair)}


def verify_wire(cache, precision, reference, fused, layers, torch):
    packets = {"reference": reference.encode_cache(cache, precision),
               "fused": fused.encode_cache(cache, precision)}
    ref, fast = packets["reference"], packets["fused"]
    for label, packet in packets.items():
        if packet.buffer.dtype != torch.uint8 or packet.buffer.ndim != 1 or not packet.buffer.is_contiguous():
            raise GateFailure(label + ": packet is not contiguous one-dimensional uint8")
    rh, fh = ref.metadata_bytes(), fast.metadata_bytes()
    if rh != fh:
        raise GateFailure("canonical packet metadata differs")
    if ref.buffer.shape != fast.buffer.shape or not torch.equal(ref.buffer, fast.buffer):
        raise GateFailure("complete encoded payload differs")
    hashes = {"reference_payload_sha256": tensor_hash(ref.buffer),
              "fused_payload_sha256": tensor_hash(fast.buffer),
              "canonical_metadata_sha256": sha(rh), "metadata_bytes": len(rh),
              "payload_bytes": ref.payload_nbytes, "wire_bytes": ref.wire_nbytes}
    baseline = reference.decode_cache(ref, as_legacy=True)
    decoded = cache_hashes(baseline, layers)
    for decoder_name, decoder, packet_name, packet in [
        ("fused", fused, "reference", ref),
        ("reference", reference, "fused", fast),
        ("fused", fused, "fused", fast),
    ]:
        actual = decoder.decode_cache(packet, as_legacy=True)
        exact_cache(actual, baseline, layers, torch, decoder_name + " decode " + packet_name)
        del actual
    is_bf16 = precision.default.bits == 16 and not precision.overrides
    if is_bf16:
        exact_cache(baseline, cache, layers, torch, "BF16 versus original model cache")
    del baseline, fast, packets
    return ref, {**hashes, "decoded_bf16_sha256": decoded["sha256"],
                 "decoded_bf16_bytes": decoded["bytes"],
                 "canonical_metadata_exact": True, "complete_payload_exact": True,
                 "all_four_cross_decodes_bitwise_equal": True,
                 "bf16_matches_original": True if is_bf16 else None}


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def measure(fn, repeats, device, torch, monitor, label):
    measurements = []
    gc.collect()
    for index in range(repeats):
        torch.cuda.synchronize(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        with monitor.during("timing:" + label):
            start = time.perf_counter_ns()
            result = fn()
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        measurements.append({"repeat": index, "elapsed_ms": elapsed_ms,
                             "baseline_allocated_bytes": baseline_allocated,
                             "baseline_reserved_bytes": baseline_reserved,
                             "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                             "peak_allocated_increment_bytes": max(0, peak_allocated - baseline_allocated),
                             "peak_reserved_increment_bytes": max(0, peak_reserved - baseline_reserved)})
        del result
    times = [x["elapsed_ms"] for x in measurements]
    return {"timing": "synchronized_wall_clock", "median_ms": statistics.median(times),
            "p95_ms": percentile(times, .95), "repeats": repeats,
            "peak_allocated_increment_bytes": max(x["peak_allocated_increment_bytes"] for x in measurements),
            "peak_reserved_increment_bytes": max(x["peak_reserved_increment_bytes"] for x in measurements),
            "measurements": measurements}


def run(args, out):
    import torch
    import transformers
    import triton
    from bonsai_bench import fused, packed as reference
    from bonsai_bench.codecs import load_precision
    from bonsai_bench.continuation import prefill
    from bonsai_bench.hf import head_dim
    from bonsai_bench.kv import _layers
    from bonsai_bench.runner import MODEL, load_pinned_model, read_samples
    from bonsai_bench.ruler import MODEL_REVISION

    if not torch.cuda.is_available():
        raise GateFailure("CUDA is required; no CPU fallback")
    if os.environ.get("TRITON_INTERPRET", "0") != "0" or os.environ.get("TRITON_ALWAYS_COMPILE", "0") != "0":
        raise GateFailure("Triton interpreter/always-compile modes cannot measure compiled steady state")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise GateFailure("benchmark requires a CUDA device")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    if "H200" not in torch.cuda.get_device_name(device):
        raise GateFailure("this benchmark protocol requires an H200 GPU")
    samples = read_samples([args.manifest])
    if len(samples) != args.expected_prompts:
        raise GateFailure(f"expected {args.expected_prompts} prompts, found {len(samples)}")
    sources = source_files()
    hashes = source_hashes(sources)
    for relative, data in sources.items():
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    props = torch.cuda.get_device_properties(device)
    provenance = {"schema": "bonsai.fused-codec-bench.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "argv": sys.argv, "model": MODEL, "model_revision": MODEL_REVISION,
                  "manifest": str(args.manifest.resolve()), "manifest_sha256": sha(args.manifest.read_bytes()),
                  "prompts": len(samples), "precisions": args.precisions.split(","),
                  "warmup_iterations_per_backend_and_stage": args.warmup, "timing_repeats": args.repeats,
                  "source_sha256": hashes, "source_snapshot": "source/", "git_head": command_output(["git", "rev-parse", "HEAD"]),
                  "git_status": command_output(["git", "status", "--short"]),
                  "software": {"python": sys.version, "python_executable": sys.executable,
                               "torch": torch.__version__, "transformers": transformers.__version__,
                               "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
                               "torch_config": torch.__config__.show()},
                  "device": {"requested": str(device), "name": props.name, "total_memory": props.total_memory,
                             "compute_capability": [props.major, props.minor], "multiprocessors": props.multi_processor_count},
                  "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]),
                  "environment": {k: os.environ[k] for k in PROVENANCE_ENV if k in os.environ},
                  "backends": {"reference": "pytorch_packed_v1", "fused": fused.BACKEND},
                  "timing_scope": "encode_cache/decode_cache including Python dispatch, allocations, validation and terminal CUDA synchronization; excludes prefill, correctness hashing, warmup and continuations",
                  "decode_returns": "fresh DynamicCache", "percentile_method": "linear interpolation at (N-1)*p",
                  "triton_cache_policy": "existing disk files are retained; warm each actual shape; JIT cache misses during timing are fatal"}
    write_json(out / "provenance.json", provenance)
    fingerprint = sha(canonical(provenance))
    monitor = JitMonitor(triton)
    verification_count = timing_count = 0
    try:
        model, tokenizer = load_pinned_model(MODEL, MODEL_REVISION, str(device))
        if getattr(model.config, "_commit_hash", MODEL_REVISION) != MODEL_REVISION:
            raise GateFailure("loaded model commit differs from pinned revision")
        write_json(out / "model-config.json", model.config.to_dict())
        with torch.inference_mode():
            for sample in samples:
                print(canonical({"event": "prefill_start", "sample_id": sample["sample_id"], "input_length": sample["input_length"]}).decode(), flush=True)
                ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
                state = prefill(model, ids)
                torch.cuda.synchronize(device)
                actual = validate_actual_cache(state.cache, _layers, model, sample, torch)
                source = cache_hashes(state.cache, _layers)
                identity = {"sample_id": sample["sample_id"], "task": sample["task"],
                            "context_tokens": sample["context_tokens"], "input_length": sample["input_length"],
                            "sample_sha256": sha(canonical(sample)), "sample_hashes": sample["hashes"],
                            "source_cache_sha256": source["sha256"], "run_fingerprint": fingerprint}
                emit(out / "samples.jsonl", {"kind": "actual_model_cache", **identity, **actual, "source_cache": source})
                for pname in args.precisions.split(","):
                    spec = load_precision(pname, head_dim=head_dim(model))
                    with monitor.during("verification:" + pname):
                        packet, checks = verify_wire(state.cache, spec, reference, fused, _layers, torch)
                    emit(out / "verification.jsonl", {"kind": "verification", "gate": "PASS", **identity, "precision": pname, **checks})
                    verification_count += 1
                    backends = [("reference", reference), ("fused", fused)]
                    # Alternate order by sample, without changing backend identity.
                    if verification_count % 2 == 0:
                        backends.reverse()
                    for name, backend in backends:
                        with monitor.during("warmup:" + name + ":" + pname):
                            for _ in range(args.warmup):
                                warm_packet = backend.encode_cache(state.cache, spec)
                                warm_cache = backend.decode_cache(warm_packet, device=device)
                                torch.cuda.synchronize(device)
                                del warm_cache, warm_packet
                        gc.collect()
                        encode = measure(lambda: backend.encode_cache(state.cache, spec), args.repeats, device, torch, monitor, name + ":" + pname + ":encode")
                        decode = measure(lambda: backend.decode_cache(packet, device=device), args.repeats, device, torch, monitor, name + ":" + pname + ":decode")
                        emit(out / "timings.jsonl", {"kind": "codec_timing", **identity, "precision": pname,
                                                     "backend": provenance["backends"][name], "backend_role": name,
                                                     "payload_bytes": packet.payload_nbytes, "metadata_bytes": len(packet.metadata_bytes()),
                                                     "raw_bytes": actual["raw_bytes"], "encode": encode, "decode": decode,
                                                     "sum_stage_medians_ms": encode["median_ms"] + decode["median_ms"],
                                                     "jit_cache_misses_during_timing": 0})
                        timing_count += 1
                    del packet
                    gc.collect()
                if cache_hashes(state.cache, _layers)["sha256"] != source["sha256"]:
                    raise GateFailure("source model cache changed during codec evaluation")
                del state, ids, source
                gc.collect()
        if source_hashes(source_files()) != hashes:
            raise GateFailure("benchmark sources changed during execution")
        summary = {"gate": "PASS", "prompts": len(samples), "verified_precision_cells": verification_count,
                   "backend_timing_rows": timing_count, "expected_precision_cells": len(samples) * len(args.precisions.split(",")),
                   "jit_cache_misses_during_timing": 0, "run_fingerprint": fingerprint,
                   "model": MODEL, "model_revision": MODEL_REVISION, "continuations_generated": 0}
        write_json(out / "summary.json", summary)
        print(canonical({"event": "FUSED_CODEC_BENCH_PASS", **summary}).decode(), flush=True)
    finally:
        write_json(out / "triton-jit-events.json", monitor.events)
        monitor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/ruler/pilot-smoke.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True, help="fresh output directory; existing paths are rejected")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precisions", default="bf16,int8,int4")
    parser.add_argument("--expected-prompts", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if min(args.repeats, args.warmup, args.expected_prompts) < 1:
        parser.error("repeats, warmup and expected-prompts must be positive")
    if args.precisions.split(",") != ["bf16", "int8", "int4"]:
        parser.error("this comparison protocol requires --precisions bf16,int8,int4")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "summary.json", {"gate": "RUNNING", "continuations_generated": 0})
    try:
        run(args, out)
    except Exception as exc:
        failure = {"gate": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(), "continuations_generated": 0}
        write_json(out / "summary.json", failure)
        print(canonical({"event": "FUSED_CODEC_BENCH_FAIL", **failure}).decode(), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
