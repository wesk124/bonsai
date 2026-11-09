import socket
import struct
import threading

import pytest
import torch

from bonsai_bench.transport import FramedSocket, TcpDiagnosticPayload, FabricPayload, validate_buffer


def test_frames_count_both_directions_and_preserve_payload():
    a, b = socket.socketpair()
    try:
        left, right = FramedSocket(a), FramedSocket(b)
        left.send_json({"sample": 4})
        assert right.recv_json() == {"sample": 4}
        payload = torch.arange(255, dtype=torch.uint8)
        TcpDiagnosticPayload().send(payload, left)
        assert torch.equal(TcpDiagnosticPayload().receive(255, right, "cpu"), payload)
        right.send_json({"kind": "ready"})
        assert left.recv_json()["kind"] == "ready"
        assert left.total_bytes == right.total_bytes
        assert left.sent_bytes == right.received_bytes
        assert left.received_bytes == right.sent_bytes
    finally:
        a.close()
        b.close()


def test_reject_oversized_and_truncated_frames():
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack("!Q", 100))
        with pytest.raises(ValueError, match="size limit"):
            FramedSocket(b, max_frame_bytes=5).recv_blob()
        a.sendall(struct.pack("!Q", 3) + b"x")
        a.close()
        with pytest.raises(EOFError):
            FramedSocket(b).recv_blob()
    finally:
        a.close()
        b.close()


def test_transport_rejects_host_fabric_and_invalid_storage():
    with pytest.raises(ValueError, match="CUDA"):
        validate_buffer(torch.ones(4, dtype=torch.uint8), cuda=True)
    with pytest.raises(ValueError, match="contiguous"):
        validate_buffer(torch.ones(4))
    with pytest.raises(ValueError, match="contiguous"):
        validate_buffer(torch.ones(8, dtype=torch.uint8)[::2])


def test_fabric_has_no_simulated_fallback(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "bonsai", types.ModuleType("bonsai"))
    with pytest.raises(RuntimeError, match="FabricEndpoint is unavailable"):
        FabricPayload(0)


@pytest.mark.parametrize("key", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
def test_fabric_rejects_reproduced_expandable_allocator_failure(monkeypatch, key):
    from bonsai_bench.transport import allocator_environment
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(key, "max_split_size_mb:128,expandable_segments:True")
    with pytest.raises(RuntimeError, match="large GPU MR"):
        allocator_environment(require_fabric_compatible=True)
    assert allocator_environment()[key].endswith("True")
    monkeypatch.setenv(key, "expandable_segments:False")
    assert allocator_environment(require_fabric_compatible=True)[key] == "expandable_segments:False"
