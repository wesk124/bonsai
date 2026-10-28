import pytest
import torch

transformers = pytest.importorskip("transformers")

from bonsai_bench.codecs import load_precision  # noqa: E402
from bonsai_bench.ppl import handoff_nll, reference_nll  # noqa: E402


@pytest.fixture(scope="module")
def tiny_qwen():
    cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).eval()   # head_dim = 32


def test_bf16_handoff_equals_plain_forward(tiny_qwen):
    window = torch.randint(0, 256, (128,), generator=torch.Generator().manual_seed(1))
    a, n = handoff_nll(tiny_qwen, window, 64, load_precision("bf16", head_dim=32))
    b, m = reference_nll(tiny_qwen, window, 64)
    assert n == m == 63
    assert a == pytest.approx(b, rel=1e-4)


def test_quantized_handoff_changes_but_stays_close(tiny_qwen):
    window = torch.randint(0, 256, (128,), generator=torch.Generator().manual_seed(2))
    ref, _ = reference_nll(tiny_qwen, window, 64)
    i8, _ = handoff_nll(tiny_qwen, window, 64, load_precision("int8", head_dim=32))
    i4, _ = handoff_nll(tiny_qwen, window, 64, load_precision("int4-pertoken", head_dim=32))
    assert i8 != ref and abs(i8 - ref) / ref < 0.01
    assert abs(i4 - ref) >= abs(i8 - ref) * 0.5   # int4 noisier than int8 (loose)


def test_layer_sensitivity_precisions():
    import argparse

    from bonsai_bench.ppl import _precisions

    a = argparse.Namespace(layer_sensitivity="int4-pertoken", config=None, precisions="")
    ps = list(_precisions(a, num_layers=4, hd=32))
    assert [n for n, _ in ps] == ["bf16"] + [f"int4-pertoken@L{i}" for i in range(4)]
    p = ps[2][1]
    assert p.codec_for_layer(1).bits == 4 and p.codec_for_layer(0).bits == 16
