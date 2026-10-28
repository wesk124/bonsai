from pathlib import Path

import pytest
import torch

from bonsai_bench.codecs import (HEAD_DIM, TOKEN, QuantParams, encoded_bytes_per_token, fake_quant,
                                 load_precision)

CONFIG = Path(__file__).resolve().parents[2] / "config" / "bonsai-ctl.example.toml"


def kv(t=100, d=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 2, t, d, generator=g).to(torch.bfloat16)


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def test_bf16_is_identity():
    x = kv()
    assert torch.equal(fake_quant(x, 16, None), x)


@pytest.mark.parametrize("bits,limit", [(8, 0.01), (4, 0.15)])
def test_error_shrinks_with_bits(bits, limit):
    x = kv()
    for p in [QuantParams(128, HEAD_DIM, True), QuantParams(32, HEAD_DIM, False), QuantParams(32, TOKEN, False)]:
        y = fake_quant(x, bits, p)
        assert y.shape == x.shape and y.dtype == x.dtype
        assert rel_err(y, x) < limit, (bits, p)


def test_token_axis_handles_partial_group():
    # 100 tokens = 3 full groups of 32 + 4; the partial group must not be distorted by padding
    x = kv(t=100)
    y = fake_quant(x, 4, QuantParams(32, TOKEN, False))
    tail = slice(96, 100)
    ref = fake_quant(x[:, :, tail].contiguous(), 4, QuantParams(4, TOKEN, False))
    assert torch.allclose(y[:, :, tail].float(), ref.float())


def test_constant_group_is_exact():
    x = torch.full((1, 1, 4, 128), 0.5, dtype=torch.bfloat16)
    for p in [QuantParams(128, HEAD_DIM, True), QuantParams(32, HEAD_DIM, False)]:
        assert torch.allclose(fake_quant(x, 4, p).float(), x.float(), atol=1e-3)


def test_atom_kv4_close_to_exact_atom_quantizer():
    x = kv()
    ours = load_precision("atom-kv4").default
    atom = load_precision("atom-kv4-exact").default
    a = fake_quant(x, 4, ours.key)
    b = fake_quant(x, 4, atom.key, atom_exact=True)
    # float vs integer zero point: same error level, small difference
    assert abs(rel_err(a, x) - rel_err(b, x)) < 0.02


def test_config_precisions_and_bytes_match_rust():
    # Values printed by `bonsai-ctl --check` (bonsai-core byte accounting).
    expect = {"bf16": 28672, "int8": 14560, "int4": 8960, "atom-kv4": 7616, "int8-edges": 10560, "int4-g64": 8064}
    for name, b in expect.items():
        p = load_precision(name, CONFIG)
        assert encoded_bytes_per_token(p, 28, 2, 128) == pytest.approx(b), name
    p = load_precision("int8-edges", CONFIG)
    assert p.codec_for_layer(0).name == "int8" and p.codec_for_layer(10).name == "int4"
    assert p.codec_for_layer(27).name == "int8"
