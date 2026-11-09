import json
import math

import pytest
import torch

from bonsai_bench.codecs import QuantParams, HEAD_DIM, TOKEN, fake_quant, load_precision
from bonsai_bench.packed import PackedCache, encode_tensor, decode_tensor, encode_cache, decode_cache

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bits,symmetric,axis", [(16, False, HEAD_DIM), (8, True, HEAD_DIM),
                                               (4, False, HEAD_DIM), (4, False, TOKEN),
                                               (4, True, TOKEN), (8, False, TOKEN)])
def test_packed_matches_reference(device, dtype, bits, symmetric, axis):
    x = torch.randn(2, 2, 35, 64, device=device).to(dtype)
    params = QuantParams(32, axis, symmetric)
    packet = encode_tensor(x, bits, params)
    assert packet.buffer.device == x.device and packet.buffer.dtype == torch.uint8
    actual = decode_tensor(packet)
    expected = fake_quant(x, bits, params)
    assert torch.equal(actual, expected)
    assert packet.wire_nbytes == packet.buffer.numel() + len(packet.metadata_bytes())
    assert json.loads(packet.metadata_bytes())["shape"] == list(x.shape)
    if bits == 4:
        padded = 64 if axis == TOKEN else 64
        values = 2 * 2 * (padded * 64 if axis == TOKEN else 35 * padded)
        assert packet.metadata["quant_bytes"] == math.ceil(values / 2)


def test_nibble_order_and_constant_group():
    x = torch.arange(16, dtype=torch.float32).view(1, 1, 1, 16)
    p = encode_tensor(x, 4, QuantParams(16, HEAD_DIM, False))
    assert p.buffer[:8].tolist() == [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]
    assert torch.equal(decode_tensor(p), x)
    for v in [0.0, 0.5, -0.5]:
        x = torch.full((1, 1, 3, 32), v, dtype=torch.bfloat16)
        assert torch.equal(decode_tensor(encode_tensor(x, 4, QuantParams(32, TOKEN, False))), x)


def test_partial_channels_roundtrip():
    x = torch.randn(1, 1, 5, 35).bfloat16()
    p = encode_tensor(x, 4, QuantParams(32, HEAD_DIM, False))
    y = decode_tensor(p)
    assert y.shape == x.shape
    assert torch.equal(y[..., :32], fake_quant(x[..., :32], 4, QuantParams(32, HEAD_DIM, False)))
    assert torch.equal(y[..., 32:], fake_quant(x[..., 32:], 4, QuantParams(3, HEAD_DIM, False)))


@pytest.mark.parametrize("precision", ["bf16", "int8", "int4"])
def test_cache_serialization_and_independence(precision):
    x = torch.randn(1, 2, 35, 32).bfloat16()
    original = ((x, x * 2), (x + 1, x - 1))
    packet = encode_cache(original, load_precision(precision, head_dim=32))
    wire = PackedCache.from_wire(packet.buffer.clone(), packet.metadata_bytes())
    decoded = decode_cache(wire)
    legacy = decode_cache(wire, as_legacy=True)
    for layer, (k, v) in zip(decoded.layers, legacy):
        assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
        assert layer.keys.data_ptr() != x.data_ptr()
    assert packet.wire_nbytes == len(packet.metadata_bytes()) + packet.payload_nbytes
    assert packet.metadata["format"] == "bonsai-torch-packed-v1"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e10])
def test_invalid_quantized_input_rejected(value):
    x = torch.full((1, 1, 1, 32), value)
    with pytest.raises(ValueError):
        encode_tensor(x, 4, QuantParams(32, HEAD_DIM, False))


def test_false_native_module_rejected(monkeypatch):
    import importlib
    import sys
    import types
    import bonsai_bench.kv as kv
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "bonsai", types.ModuleType("bonsai"))
        importlib.reload(kv)
        assert not kv.native_available()
        with pytest.raises(RuntimeError, match="unavailable"):
            kv.roundtrip((), load_precision("bf16"), use_native=True)
    importlib.reload(kv)
