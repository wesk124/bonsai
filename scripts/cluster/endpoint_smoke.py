#!/usr/bin/env python3
"""Two-node GPU RDMA smoke; TCP carries descriptors and ACKs only."""
import argparse
import json
import os
import socket
import time

import torch
import bonsai


def send(stream, data):
    stream.write(json.dumps(data).encode() + b"\n")
    stream.flush()


def receive(stream):
    line = stream.readline()
    if not line:
        raise RuntimeError("control peer disconnected")
    return json.loads(line)


def pattern(size, iteration):
    return ((torch.arange(size, device="cuda", dtype=torch.int64) * 17 + iteration * 31) % 251).to(torch.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=29874)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 65536, 1048576, 16777216])
    args = parser.parse_args()
    rank = int(os.environ["SLURM_PROCID"])
    torch.cuda.set_device(0)
    endpoint = bonsai.FabricEndpoint(0, 1)
    print(json.dumps({"rank": rank, "hostname": socket.gethostname(), "gpu": torch.cuda.get_device_name(), "link_bps": endpoint.link_bps, "nets": endpoint.nets_per_gpu}), flush=True)
    if rank == 0:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", args.port))
        listener.listen(1)
        listener.settimeout(60)
        connection, _ = listener.accept()
    else:
        deadline = time.monotonic() + 60
        while True:
            try:
                connection = socket.create_connection((args.host, args.port), timeout=5)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
    connection.settimeout(60)
    stream = connection.makefile("rwb")
    for iteration, size in enumerate(args.sizes, 1):
        if rank == 0:
            tensor = torch.full((size,), 255, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            handle, descriptor = endpoint.register_sink(tensor.data_ptr(), size)
            endpoint.arm_imm(iteration)
            send(stream, {"descriptor": list(descriptor), "size": size, "tag": iteration})
            endpoint.wait_imm(iteration, 30.0)
            torch.cuda.synchronize()
            if not torch.equal(tensor, pattern(size, iteration)):
                raise AssertionError(f"GPU RDMA payload mismatch: {size} bytes")
            endpoint.unregister(handle)
            send(stream, {"verified": size})
            print(f"ENDPOINT_GPU_RDMA_VERIFIED bytes={size} tag={iteration}", flush=True)
        else:
            request = receive(stream)
            assert request["size"] == size
            tensor = pattern(size, iteration)
            torch.cuda.synchronize()
            handle = endpoint.register_source(tensor.data_ptr(), size)
            start = time.perf_counter()
            endpoint.write(handle, request["descriptor"], size, request["tag"], 30.0)
            elapsed = time.perf_counter() - start
            print(json.dumps({"measurement": "single_write_smoke", "bytes": size, "seconds": elapsed, "gbps": size * 8 / elapsed / 1e9}), flush=True)
            assert receive(stream)["verified"] == size
            endpoint.unregister(handle)
    send(stream, {"done": rank})
    assert "done" in receive(stream)
    print(f"ENDPOINT_SMOKE_PASS rank={rank}", flush=True)
    stream.close()
    connection.close()


if __name__ == "__main__":
    main()
