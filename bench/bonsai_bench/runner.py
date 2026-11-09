"""Sequential, resumable one/two GPU RULER benchmark. No implicit transport fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import time
import traceback

import torch

from .codecs import load_precision
from .continuation import prefill, continue_from_prefill
from .hf import head_dim
from .codec_backend import BACKENDS, DEFAULT_BACKEND, get_codec_backend
from .ruler import MODEL_REVISION, score_prediction
from .transport import FramedSocket, FabricPayload, TcpDiagnosticPayload, allocator_environment

SCHEMA_VERSION = 1
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CONDITIONS = ("local_bf16", "local_roundtrip", "transfer", "text_refill")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, function):
    synchronize(device)
    start = time.perf_counter_ns()
    value = function()
    synchronize(device)
    return value, (time.perf_counter_ns() - start) / 1e6


def memory(device, registered=0):
    if torch.device(device).type != "cuda":
        return {"allocated_bytes": 0, "reserved_bytes": 0, "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0, "native_registered_bytes": registered}
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "native_registered_bytes": registered}


def load_pinned_model(model, revision, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if revision != MODEL_REVISION:
        raise ValueError("benchmark requires the frozen model revision " + MODEL_REVISION)
    loaded = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", trust_remote_code=False).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    return loaded, tokenizer


def read_samples(paths):
    samples = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                sample = json.loads(line)
                if not sample.get("input_ids") or len(sample["input_ids"]) != sample["input_length"]:
                    raise ValueError("manifest input IDs and length disagree")
                if sample.get("model_revision") != MODEL_REVISION:
                    raise ValueError("manifest model revision differs from frozen protocol")
                if digest(sample["input_ids"]) != sample["hashes"]["input_ids_sha256"]:
                    raise ValueError("manifest input IDs hash mismatch")
                if "prompt" in sample and hashlib.sha256(sample["prompt"].encode()).hexdigest() != sample["hashes"]["prompt_sha256"]:
                    raise ValueError("manifest prompt hash mismatch")
                samples.append(sample)
    keys = [sample_key(s) for s in samples]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate sample identities in manifests")
    return samples


def sample_key(sample):
    return canonical([sample.get("split"), sample["task"], sample["context_tokens"], sample["sample_id"]])


def row_key(sample, condition, precision):
    return sample_key(sample) + ":" + condition + ":" + precision


def parse_conditions(values):
    result = []
    for value in values:
        condition, _, precision = value.partition(":")
        precision = precision or "bf16"
        if condition not in CONDITIONS or precision not in ("bf16", "int8", "int4"):
            raise ValueError("unknown condition/precision: " + value)
        if condition in ("local_bf16", "text_refill") and precision != "bf16":
            raise ValueError(condition + " only accepts bf16")
        if (condition, precision) in result:
            raise ValueError("duplicate condition: " + value)
        result.append((condition, precision))
    return result


def initialize_output(path, config, *, retry_failures=False):
    config = json.loads(canonical(config))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = path.with_suffix(path.suffix + ".provenance.json")
    fingerprint = digest(config)
    provenance = {"schema_version": SCHEMA_VERSION, "run_fingerprint": fingerprint, "config": config}
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("resume provenance differs; choose a fresh output file")
    else:
        if path.exists() and path.stat().st_size:
            raise ValueError("nonempty results lack provenance; refusing unsafe resume")
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    complete = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("run_fingerprint") != fingerprint:
                raise ValueError("result row fingerprint mismatch")
            if row["kind"] == "prediction":
                complete.add(row["row_key"])
            elif row["kind"] == "failure" and not retry_failures:
                raise ValueError("results contain previous failures; use --retry-failures to explicitly append retry attempts")
    return fingerprint, complete


def generate_prediction(model, tokenizer, state, sample):
    ids = continue_from_prefill(model, state, sample["max_new_tokens"],
                               eos_token_id=model.generation_config.eos_token_id,
                               pad_token_id=model.generation_config.pad_token_id)
    generated = ids[0].tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {"generated_ids": generated, "prediction": text}


def prediction(model, tokenizer, state, sample):
    result = generate_prediction(model, tokenizer, state, sample)
    return {**result, "outputs": sample["outputs"], "score": score_prediction(result["prediction"], sample["outputs"])}


def local_evaluate(model, tokenizer, state, sample, condition, precision, device, *,
                   codec_backend=DEFAULT_BACKEND):
    codec = get_codec_backend(codec_backend)
    baseline = memory(device)
    metrics = {"encode_ms": 0.0, "decode_ms": 0.0, "handoff_ack_ms": None,
               "payload_bytes": 0, "metadata_bytes": 0, "control_bytes": 0, "wire_bytes": 0}
    if condition == "local_roundtrip":
        spec = load_precision(precision, head_dim=head_dim(model))
        packet, metrics["encode_ms"] = timed(device, lambda: codec.encode_cache(state.cache, spec))
        cache, metrics["decode_ms"] = timed(device, lambda: codec.decode_cache(packet, device=device))
        metrics.update(payload_bytes=packet.payload_nbytes, metadata_bytes=len(packet.metadata_bytes()),
                       wire_bytes=packet.wire_nbytes)
        state = state.replace_cache(cache)
    handoff_memory = memory(device)
    handoff_memory["baseline_allocated_bytes"] = baseline["allocated_bytes"]
    handoff_memory["temporary_peak_allocated_bytes"] = max(0, handoff_memory["peak_allocated_bytes"] - baseline["allocated_bytes"])
    result, continuation_ms = timed(device, lambda: prediction(model, tokenizer, state, sample))
    return {**result, **metrics, "continuation_ms": continuation_ms, "transport": "none",
            "memory": memory(device), "handoff_memory": handoff_memory, "codec_backend": codec_backend,
            "wire_bytes_scope": "encoded_frame_size_only_no_network_transfer"}


def remote_evaluate(channel, payload, model, state, sample, condition, precision, device, tag, *,
                    codec_backend=DEFAULT_BACKEND):
    codec = get_codec_backend(codec_backend)
    baseline = memory(device)
    start_bytes = channel.total_bytes
    synchronize(device)
    start = time.perf_counter_ns()  # sender cache-ready through receiver-ready ACK
    packet = None
    encode_ms = 0.0
    if condition == "transfer":
        spec = load_precision(precision, head_dim=head_dim(model))
        packet, encode_ms = timed(device, lambda: codec.encode_cache(state.cache, spec))
    inference_sample = {k: sample[k] for k in ("sample_id", "input_length", "max_new_tokens")}
    inference_sample["input_ids_sha256"] = sample["hashes"]["input_ids_sha256"]
    if condition == "text_refill":
        if "prompt" not in sample:
            raise ValueError("text_refill requires the original rendered prompt")
        inference_sample["prompt"] = sample["prompt"]
    request = {"kind": "request", "condition": condition, "precision": precision,
               "codec_backend": codec_backend,
               "sample": inference_sample, "first_token": int(state.next_token_logits.argmax(-1).item()), "tag": tag,
               "payload_bytes": packet.payload_nbytes if packet else 0,
               "metadata": packet.metadata if packet else None}
    channel.send_json(request)
    if condition == "transfer":
        grant = channel.recv_json()
        if grant.get("kind") != "grant":
            raise ValueError("expected receiver buffer grant")
        if payload.name == "fabric-lib":
            payload.send(packet.buffer, grant)
        else:
            payload.send(packet.buffer, channel)
    ack = channel.recv_json()
    ack_ms = (time.perf_counter_ns() - start) / 1e6
    if ack.get("kind") != "ready":
        raise ValueError("expected receiver-ready acknowledgement")
    ready_bytes = channel.total_bytes
    handoff_memory = memory(device)
    handoff_memory["baseline_allocated_bytes"] = baseline["allocated_bytes"]
    handoff_memory["temporary_peak_allocated_bytes"] = max(0, handoff_memory["peak_allocated_bytes"] - baseline["allocated_bytes"])
    result = channel.recv_json()
    if result.get("kind") != "result":
        raise ValueError("expected receiver prediction")
    # Handoff bytes stop at receiver-ready ACK, matching the latency interval.
    # Continuation response is measured separately. Network headers excluded.
    handoff_control = ready_bytes - start_bytes
    payload_bytes = packet.payload_nbytes if packet else len(sample["prompt"].encode())
    metadata_bytes = len(packet.metadata_bytes()) if packet else 0
    tcp_payload = payload_bytes if packet is not None and payload.name == "tcp_diagnostic" else 0
    if condition == "text_refill":
        tcp_payload = payload_bytes  # text is carried in the counted JSON frame
    wire_bytes = handoff_control + (payload_bytes if packet is not None and payload.name == "fabric-lib" else 0)
    return {**result["result"], "outputs": sample["outputs"],
            "score": score_prediction(result["result"]["prediction"], sample["outputs"]),
            "encode_ms": encode_ms, "decode_ms": ack["decode_ms"],
            "receiver_prefill_ms": ack.get("prefill_ms", 0.0), "handoff_ack_ms": ack_ms,
            "payload_bytes": payload_bytes, "metadata_bytes": metadata_bytes,
            "control_bytes": handoff_control - tcp_payload - metadata_bytes,
            "wire_bytes": wire_bytes, "prediction_response_bytes": channel.total_bytes - ready_bytes,
            "transport": payload.name, "memory": memory(device), "handoff_memory": handoff_memory, "receiver_memory": ack["memory"],
            "codec_backend": codec_backend, "wire_bytes_scope": "sender_cache_ready_to_receiver_ready_ack_application_frames_excludes_network_headers"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--codec-backend", choices=BACKENDS, default=DEFAULT_BACKEND)
    ap.add_argument("--condition", action="append", default=None,
                    help="repeat CONDITION:PRECISION; local_bf16/local_roundtrip/transfer/text_refill")
    ap.add_argument("--receiver", help="HOST:PORT for a separately launched bonsai_bench.worker")
    ap.add_argument("--transport", choices=["fabric-lib", "tcp_diagnostic"], default="fabric-lib")
    ap.add_argument("--nets-per-gpu", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--warmup-samples", type=int, default=1,
                    help="unmeasured complete condition passes before collecting results")
    ap.add_argument("--retry-failures", action="store_true", help="explicitly retry failed rows, retaining all prior attempts")
    args = ap.parse_args(argv)
    if args.warmup_samples < 0:
        ap.error("--warmup-samples must be nonnegative")
    conditions = parse_conditions(args.condition or ["local_bf16:bf16", "local_roundtrip:int8", "local_roundtrip:int4"])
    samples = read_samples(args.manifest)
    if args.limit is not None:
        if any(s["split"] == "final" for s in samples):
            ap.error("--limit is restricted to pilot manifests; final evaluation must remain complete")
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("no samples selected")
    if any(s["model"] != args.model for s in samples):
        raise ValueError("model differs from manifest")
    remote = any(c in ("transfer", "text_refill") for c, _ in conditions)
    if remote and not args.receiver:
        ap.error("remote conditions require --receiver")
    allocator_env = allocator_environment(require_fabric_compatible=remote and args.transport == "fabric-lib")
    print(canonical({"startup_allocator_environment": allocator_env}), flush=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        git_commit = "unavailable"
    import transformers
    source_hashes = {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    get_codec_backend(args.codec_backend)  # Fail before loading a model if unavailable.
    config = {"codec_backend": args.codec_backend, "warmup_samples": args.warmup_samples, "allocator_environment": allocator_env, "source_sha256": source_hashes, "condition_order": "per_sample_seeded_shuffle_v1", "model": args.model, "model_revision": args.revision, "conditions": conditions,
              "transport": args.transport if remote else "none", "nets_per_gpu": args.nets_per_gpu,
              "sample_hashes": [digest(s) for s in samples], "seed": 0, "git_commit": git_commit,
              "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
              "device": args.device, "attn_implementation": "sdpa", "dtype": "bfloat16"}
    fingerprint, completed = initialize_output(args.out, config, retry_failures=args.retry_failures)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    model, tokenizer = load_pinned_model(args.model, args.revision, args.device)
    channel = payload = sock = None
    if remote:
        host, port = args.receiver.rsplit(":", 1)
        sock = socket.create_connection((host, int(port)), args.timeout)
        sock.settimeout(args.timeout)
        channel = FramedSocket(sock)
        payload = FabricPayload(torch.device(args.device).index or 0, args.nets_per_gpu, args.timeout) if args.transport == "fabric-lib" else TcpDiagnosticPayload()
        channel.send_json({"kind": "hello", "model": args.model, "revision": args.revision,
                           "codec_backend": args.codec_backend,
                           "transport": args.transport, "run_fingerprint": fingerprint})
        hello = channel.recv_json()
        if hello.get("kind") != "hello" or hello.get("revision") != args.revision or hello.get("model") != args.model:
            raise ValueError("receiver model provenance differs")
        if hello.get("codec_backend", DEFAULT_BACKEND) != args.codec_backend:
            raise ValueError("receiver codec backend differs")
    try:
        with Path(args.out).open("a", buffering=1) as output:
            tag = 0
            pending = [s for s in samples if any(row_key(s, c, p) not in completed for c, p in conditions)]
            for warmup_index in range(args.warmup_samples if pending else 0):
                warmup_sample = pending[warmup_index % len(pending)]
                warmup_ids = torch.tensor([warmup_sample["input_ids"]], dtype=torch.long, device=args.device)
                warmup_state = prefill(model, warmup_ids)
                for condition, precision in conditions:
                    tag += 1
                    if condition in ("local_bf16", "local_roundtrip"):
                        local_evaluate(model, tokenizer, warmup_state, warmup_sample, condition, precision, args.device,
                                       codec_backend=args.codec_backend)
                    else:
                        remote_evaluate(channel, payload, model, warmup_state, warmup_sample, condition, precision, args.device, tag,
                                        codec_backend=args.codec_backend)
                del warmup_state, warmup_ids
                print(canonical({"kind": "warmup_complete", "index": warmup_index,
                                 "sample_id": warmup_sample["sample_id"]}), flush=True)
            for sample in samples:
                needed = [(c, p) for c, p in conditions if row_key(sample, c, p) not in completed]
                if not needed:
                    continue
                random.Random(int(digest(sample)[:16], 16)).shuffle(needed)
                if torch.device(args.device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(args.device)
                ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=args.device)
                state = None
                prefill_error = None
                try:
                    state, prefill_ms = timed(args.device, lambda: prefill(model, ids))
                except Exception:
                    prefill_error = traceback.format_exc()
                for condition, precision in needed:
                    if torch.device(args.device).type == "cuda":
                        torch.cuda.reset_peak_memory_stats(args.device)
                    tag += 1
                    row = {"schema_version": SCHEMA_VERSION, "run_fingerprint": fingerprint,
                           "row_key": row_key(sample, condition, precision), "condition": condition, "precision": precision,
                           **{k: sample[k] for k in ("sample_id", "task", "context_tokens", "split", "seed", "model", "model_revision", "input_length")},
                           "sample_hash": digest(sample), "hostname": socket.gethostname()}
                    try:
                        if prefill_error:
                            raise RuntimeError(prefill_error)
                        if condition in ("local_bf16", "local_roundtrip"):
                            result = local_evaluate(model, tokenizer, state, sample, condition, precision, args.device,
                                                    codec_backend=args.codec_backend)
                        else:
                            result = remote_evaluate(channel, payload, model, state, sample, condition, precision, args.device, tag,
                                                     codec_backend=args.codec_backend)
                        row.update(kind="prediction", prefill_ms=prefill_ms, **result)
                    except Exception:
                        row.update(kind="failure", error=traceback.format_exc(), memory=memory(args.device))
                    output.write(canonical(row) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                    print(canonical({k: row[k] for k in ("kind", "sample_id", "condition", "precision")}), flush=True)
                    if row["kind"] == "failure" and (prefill_error or condition in ("transfer", "text_refill")):
                        raise RuntimeError("prefill or remote condition failed; retained failure row and terminating")
                del state, ids
    finally:
        if sock:
            sock.close()


if __name__ == "__main__":
    main()
