"""One-model, one-GPU sequential decode worker for the benchmark runner.

fabric-lib moves CUDA uint8 buffers via RDMA. TCP carries counted metadata and
completion ACKs. --transport tcp_diagnostic stages payloads through host memory
and is an explicitly diagnostic baseline, never an RDMA result.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import traceback

import torch

from .continuation import PrefillState, prefill
from .packed import PackedCache
from .codec_backend import BACKENDS, DEFAULT_BACKEND, get_codec_backend
from .runner import (MODEL, MODEL_REVISION, digest, load_pinned_model, memory,
                     generate_prediction, synchronize, timed)
from .transport import FabricPayload, FramedSocket, TcpDiagnosticPayload, allocator_environment


def receive_request(channel, payload, model, tokenizer, request, device, *, codec_backend=DEFAULT_BACKEND):
    if request.get("codec_backend", DEFAULT_BACKEND) != codec_backend:
        raise ValueError("request codec backend differs from negotiated backend")
    codec = get_codec_backend(codec_backend)
    baseline = memory(device)
    sample = request["sample"]
    condition = request["condition"]
    prefill_ms = decode_ms = 0.0
    if condition == "text_refill":
        # Real text-transfer baseline: receiver repeats tokenizer and prefill.
        ids = tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"]
        if digest(ids) != sample["input_ids_sha256"]:
            raise ValueError("receiver tokenization differs from frozen input IDs")
        sample = {**sample, "input_ids": ids}
        tensor_ids = torch.tensor([ids], dtype=torch.long, device=device)
        state, prefill_ms = timed(device, lambda: prefill(model, tensor_ids))
        if int(state.next_token_logits.argmax(-1).item()) != request["first_token"]:
            raise ValueError("text receiver's first token differs from sender")
    elif condition == "transfer":
        length = request["payload_bytes"]
        if not isinstance(length, int) or length <= 0:
            raise ValueError("invalid receive payload size")
        grant = None
        if payload.name == "fabric-lib":
            buffer = torch.empty(length, dtype=torch.uint8, device=device)
            grant = payload.prepare_receive(buffer, request["tag"])
            channel.send_json({"kind": "grant", "descriptor": grant["descriptor"], "tag": grant["tag"]})
            try:
                payload.wait_receive(grant)
            except Exception:
                # A peer may still be DMA-writing. Keep the registration and
                # storage alive until this failing worker process terminates.
                payload.quarantine.append((buffer, grant))
                raise
            synchronize(device)
        else:
            channel.send_json({"kind": "grant", "tag": request["tag"]})
            buffer = payload.receive(length, channel, device)
            synchronize(device)
        try:
            packet = PackedCache.from_wire(buffer, request["metadata"])
            cache, decode_ms = timed(device, lambda: codec.decode_cache(packet, device=device))
            ids = torch.zeros((1, sample["input_length"]), dtype=torch.long, device=device)
            # Keep exact sender prompt-boundary first-token choice. Later
            # generated KV comes from ordinary BF16 model forwards.
            logits = torch.full((1, model.config.vocab_size), -float("inf"), device=device)
            logits[0, request["first_token"]] = 0
            state = PrefillState(ids, torch.ones_like(ids), logits, cache)
            receiver_memory = memory(device, buffer.numel() if grant else 0)
        finally:
            if grant:
                payload.release_receive(grant, length)
    else:
        raise ValueError("worker only accepts transfer/text_refill")
    if condition == "text_refill":
        receiver_memory = memory(device)
    # ACK is sent only after dequantization and device synchronization. Its
    # receipt closes the sender monotonic timer before continuation starts.
    synchronize(device)
    receiver_memory["baseline_allocated_bytes"] = baseline["allocated_bytes"]
    receiver_memory["temporary_peak_allocated_bytes"] = max(0, receiver_memory["peak_allocated_bytes"] - baseline["allocated_bytes"])
    channel.send_json({"kind": "ready", "decode_ms": decode_ms, "prefill_ms": prefill_ms,
                       "memory": receiver_memory})
    result, continuation_ms = timed(device, lambda: generate_prediction(model, tokenizer, state, sample))
    result.update(continuation_ms=continuation_ms)
    channel.send_json({"kind": "result", "result": result})


def serve_connection(sock, model, tokenizer, *, model_name, revision, device,
                     transport="fabric-lib", nets_per_gpu=1, timeout=300, payload=None,
                     codec_backend=DEFAULT_BACKEND):
    sock.settimeout(timeout)
    channel = FramedSocket(sock)
    hello = channel.recv_json()
    if (hello.get("kind"), hello.get("model"), hello.get("revision"), hello.get("transport")) != (
            "hello", model_name, revision, transport):
        raise ValueError("sender model revision or requested transport differs")
    if hello.get("codec_backend", DEFAULT_BACKEND) != codec_backend:
        raise ValueError("sender codec backend differs")
    get_codec_backend(codec_backend)
    if payload is None:
        payload = FabricPayload(torch.device(device).index or 0, nets_per_gpu, timeout) if transport == "fabric-lib" else TcpDiagnosticPayload()
    channel.send_json({"kind": "hello", "model": model_name, "revision": revision,
                       "codec_backend": codec_backend,
                       "transport": transport, "hostname": socket.gethostname()})
    while True:
        try:
            request = channel.recv_json()
        except EOFError:
            return
        if request.get("kind") != "request":
            raise ValueError("expected a benchmark request")
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            receive_request(channel, payload, model, tokenizer, request, device, codec_backend=codec_backend)
        except Exception:
            channel.send_json({"kind": "error", "error": traceback.format_exc()})
            raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen", default="0.0.0.0:7007")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--codec-backend", choices=BACKENDS, default=DEFAULT_BACKEND)
    ap.add_argument("--transport", choices=["fabric-lib", "tcp_diagnostic"], default="fabric-lib")
    ap.add_argument("--nets-per-gpu", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--once", action="store_true", help="exit after the first runner disconnects")
    args = ap.parse_args(argv)
    allocator_env = allocator_environment(require_fabric_compatible=args.transport == "fabric-lib")
    print(json.dumps({"startup_allocator_environment": allocator_env}, sort_keys=True), flush=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    get_codec_backend(args.codec_backend)
    model, tokenizer = load_pinned_model(args.model, args.revision, args.device)
    payload = FabricPayload(torch.device(args.device).index or 0, args.nets_per_gpu, args.timeout) if args.transport == "fabric-lib" else TcpDiagnosticPayload()
    host, port = args.listen.rsplit(":", 1)
    with socket.create_server((host, int(port)), backlog=1) as server:
        print(f"READY {args.listen} transport={args.transport} model_revision={args.revision}", flush=True)
        while True:
            sock, _ = server.accept()
            with sock:
                serve_connection(sock, model, tokenizer, model_name=args.model, revision=args.revision,
                                 device=args.device, transport=args.transport, nets_per_gpu=args.nets_per_gpu,
                                 timeout=args.timeout, payload=payload, codec_backend=args.codec_backend)
            if args.once:
                break


if __name__ == "__main__":
    main()
