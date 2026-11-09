"""Counted TCP control and real fabric-lib GPU payload transfers.

TCP payload transport is diagnostic only. Selecting fabric-lib never falls
back to TCP, host staging, or a simulated completion.
"""
from __future__ import annotations

import json
import os
import re
import socket
import struct
import torch


def allocator_environment(*, require_fabric_compatible=False):
    """Read before CUDA initialization; reject the reproduced large-MR failure."""
    values = {key: os.environ.get(key) for key in
              ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_MODULE_LOADING")}
    if require_fabric_compatible:
        for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            if re.search(r"(?:^|,)\s*expandable_segments\s*:\s*True\s*(?:,|$)", values[key] or "", re.I):
                raise RuntimeError(f"{key} enables expandable_segments, which failed large GPU MR "
                                   "registration on this fabric; source scripts/native_env.sh "
                                   "before starting Python")
    return values


class FramedSocket:
    def __init__(self, sock: socket.socket, max_frame_bytes: int = 1 << 34):
        self.sock = sock
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.max_frame_bytes = max_frame_bytes
        self.sent_bytes = self.received_bytes = 0

    def send_blob(self, data: bytes) -> None:
        if len(data) > self.max_frame_bytes:
            raise ValueError("frame exceeds configured size limit")
        self.sock.sendall(struct.pack("!Q", len(data)) + data)
        self.sent_bytes += 8 + len(data)

    def _read(self, count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            chunk = self.sock.recv(min(count - len(chunks), 8 << 20))
            if not chunk:
                raise EOFError("peer closed an incomplete frame")
            chunks.extend(chunk)
            self.received_bytes += len(chunk)
        return bytes(chunks)

    def recv_blob(self) -> bytes:
        length, = struct.unpack("!Q", self._read(8))
        if length > self.max_frame_bytes:
            raise ValueError("peer frame exceeds configured size limit")
        return self._read(length)

    def send_json(self, value: dict) -> None:
        self.send_blob(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())

    def recv_json(self) -> dict:
        value = json.loads(self.recv_blob())
        if not isinstance(value, dict):
            raise ValueError("control frame must be a JSON object")
        if value.get("kind") == "error":
            raise RuntimeError("receiver: " + value["error"])
        return value

    @property
    def total_bytes(self):
        return self.sent_bytes + self.received_bytes


def validate_buffer(buffer: torch.Tensor, *, cuda: bool = False):
    if buffer.dtype != torch.uint8 or buffer.ndim != 1 or not buffer.is_contiguous():
        raise ValueError("transport requires contiguous one-dimensional uint8")
    if cuda and buffer.device.type != "cuda":
        raise ValueError("fabric-lib benchmark requires a CUDA buffer; no host fallback")
    if not buffer.numel():
        raise ValueError("cannot register an empty buffer")


class FabricPayload:
    """A thin wrapper around Bonsai's Rust fabric-lib adapter.

    All tensor storage stays alive through completion. The receiver arms its
    immediate before publishing the remote memory descriptor.
    """
    name = "fabric-lib"

    def __init__(self, cuda_device: int, nets_per_gpu: int = 1, timeout: float = 120):
        import os
        try:
            import bonsai
            endpoint = bonsai.FabricEndpoint
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("fabric-lib requested but Bonsai FabricEndpoint is unavailable; "
                               "build crates/bonsai-py with --features fabric") from exc
        cpus = sorted(os.sched_getaffinity(0))
        self.endpoint = endpoint(cuda_device, nets_per_gpu, cpus[0], cpus[-1])
        self.timeout = timeout
        self.registered_bytes = 0
        self.quarantine = []

    def prepare_receive(self, buffer: torch.Tensor, tag: int) -> dict:
        validate_buffer(buffer, cuda=True)
        handle, descriptor = self.endpoint.register_sink(buffer.data_ptr(), buffer.numel())
        try:
            self.endpoint.arm_imm(tag)
        except Exception:
            self.endpoint.unregister(handle)
            raise
        self.registered_bytes += buffer.numel()
        return {"handle": handle, "descriptor": bytes(descriptor).hex(), "tag": tag}

    def wait_receive(self, grant: dict):
        self.endpoint.wait_imm(grant["tag"], self.timeout)

    def release_receive(self, grant: dict, length: int):
        self.endpoint.unregister(grant["handle"])
        self.registered_bytes -= length

    def send(self, buffer: torch.Tensor, grant: dict):
        validate_buffer(buffer, cuda=True)
        handle = self.endpoint.register_source(buffer.data_ptr(), buffer.numel())
        self.registered_bytes += buffer.numel()
        try:
            self.endpoint.write(handle, bytes.fromhex(grant["descriptor"]), buffer.numel(),
                                grant["tag"], self.timeout)
        except Exception:
            self.quarantine.append((buffer, handle))
            raise
        else:
            self.endpoint.unregister(handle)
            self.registered_bytes -= buffer.numel()


class TcpDiagnosticPayload:
    name = "tcp_diagnostic"
    registered_bytes = 0

    def send(self, buffer: torch.Tensor, channel: FramedSocket):
        validate_buffer(buffer)
        channel.send_blob(buffer.cpu().numpy().tobytes())

    def receive(self, length: int, channel: FramedSocket, device):
        data = channel.recv_blob()
        if len(data) != length:
            raise ValueError("TCP diagnostic payload length mismatch")
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone().to(device)
