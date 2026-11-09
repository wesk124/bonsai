#!/usr/bin/env python3
"""Untimed actual-Qwen KV integrity diagnostic over native GPU RDMA.

TCP carries JSON metadata, grants and SHA256 results; packet buffers travel
exclusively through FabricEndpoint. Uses frozen benchmark modules from PYTHONPATH.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import torch
import bonsai
from bonsai_bench.codecs import load_precision
from bonsai_bench.continuation import prefill
from bonsai_bench.hf import head_dim
from bonsai_bench.packed import PackedCache, encode_cache, decode_cache
from bonsai_bench.runner import MODEL, load_pinned_model, read_samples
from bonsai_bench.ruler import MODEL_REVISION


def send(stream, record):
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    stream.flush()


def receive(stream):
    line = stream.readline()
    if not line:
        raise RuntimeError("control peer disconnected")
    return json.loads(line)


def sha_tensor(buffer):
    # Entire actual packet, with no sampling or host pattern regeneration.
    host = buffer.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(host)).hexdigest()


def reconstructed_hashes(packet):
    decoded = decode_cache(packet)
    canonical = encode_cache(decoded, load_precision("bf16"))
    result = {"reconstructed_bf16_sha256": sha_tensor(canonical.buffer),
              "reconstructed_bf16_bytes": canonical.payload_nbytes,
              "reconstructed_metadata_sha256": hashlib.sha256(canonical.metadata_bytes()).hexdigest()}
    del decoded, canonical
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["SLURM_PROCID"])
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    assert "H200" in torch.cuda.get_device_name(), torch.cuda.get_device_name()
    endpoint = bonsai.FabricEndpoint(0, 1)
    print(json.dumps({"event": "start", "rank": rank, "host": socket.gethostname(),
                      "gpu": torch.cuda.get_device_name(), "native": bonsai.__file__,
                      "model": MODEL, "model_revision": MODEL_REVISION,
                      "allocator": os.environ.get("PYTORCH_ALLOC_CONF"),
                      "diagnostic_only": True}), flush=True)
    if rank == 0:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", args.port))
        listener.listen(1)
        listener.settimeout(120)
        connection, _ = listener.accept()
        listener.close()
    else:
        deadline = time.monotonic() + 120
        while True:
            try:
                connection = socket.create_connection((args.host, args.port), timeout=5)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
    connection.settimeout(480)
    stream = connection.makefile("rwb")
    if rank == 1:
        sample = read_samples([args.run_dir / "data/ruler/pilot/16384/niah_multikey_1.jsonl"])[0]
        model, tokenizer = load_pinned_model(MODEL, MODEL_REVISION, "cuda:0")
        ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            state = prefill(model, ids)
        hd = head_dim(model)
        print(json.dumps({"event": "prefill_complete", "sample_id": sample["sample_id"],
                          "input_length": sample["input_length"], "input_ids_sha256": sample["hashes"]["input_ids_sha256"],
                          "layers": len(state.cache.layers), "head_dim": hd}), flush=True)
        # Releasing model weights does not affect its already constructed cache.
        del model, tokenizer
        gc.collect()
    for tag, precision in enumerate(("bf16", "int8", "int4"), 1):
        if rank == 1:
            packet = encode_cache(state.cache, load_precision(precision, head_dim=hd))
            torch.cuda.synchronize()
            header = packet.metadata_bytes()
            expected = {"payload_sha256": sha_tensor(packet.buffer),
                        "metadata_sha256": hashlib.sha256(header).hexdigest(),
                        **reconstructed_hashes(packet)}
            send(stream, {"precision": precision, "tag": tag, "size": packet.payload_nbytes,
                          "metadata": packet.metadata, "metadata_bytes": len(header),
                          "wire_bytes": packet.wire_nbytes, "expected": expected})
            grant = receive(stream)
            assert grant["tag"] == tag
            handle = endpoint.register_source(packet.buffer.data_ptr(), packet.payload_nbytes)
            try:
                # Frozen native implementation chunks at 256MiB and sends the
                # IMM on the final chunk. Keep packet alive through completion.
                endpoint.write(handle, grant["descriptor"], packet.payload_nbytes, tag, 120.0)
                response = receive(stream)
                assert response["verified"] is True and response["actual"] == expected, response
            finally:
                endpoint.unregister(handle)
            print(json.dumps({"event": "source_verified", "precision": precision,
                              "payload_bytes": packet.payload_nbytes, **expected}), flush=True)
            del packet
        else:
            request = receive(stream)
            assert request["tag"] == tag and request["precision"] == precision
            buffer = torch.empty(request["size"], dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            handle, descriptor = endpoint.register_sink(buffer.data_ptr(), buffer.numel())
            try:
                endpoint.arm_imm(tag)
                send(stream, {"tag": tag, "descriptor": list(descriptor)})
                endpoint.wait_imm(tag, 120.0)
                torch.cuda.synchronize()
                packet = PackedCache(buffer, request["metadata"])
                actual = {"payload_sha256": sha_tensor(buffer),
                          "metadata_sha256": hashlib.sha256(packet.metadata_bytes()).hexdigest(),
                          **reconstructed_hashes(packet)}
                assert actual == request["expected"], (actual, request["expected"])
                assert len(packet.metadata_bytes()) == request["metadata_bytes"]
                assert packet.wire_nbytes == request["wire_bytes"]
            finally:
                endpoint.unregister(handle)
            send(stream, {"verified": True, "actual": actual})
            print(json.dumps({"event": "MODEL_CACHE_INTEGRITY_PASS", "precision": precision,
                              "payload_bytes": buffer.numel(), "metadata_bytes": request["metadata_bytes"],
                              "wire_bytes": packet.wire_nbytes, **actual}), flush=True)
            del packet, buffer
        gc.collect()
        torch.cuda.empty_cache()
    send(stream, {"done": rank})
    assert "done" in receive(stream)
    print(f"MODEL_CACHE_INTEGRITY_ALL_PASS rank={rank}", flush=True)
    stream.close()
    connection.close()


if __name__ == "__main__":
    main()
