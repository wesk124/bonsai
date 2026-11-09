import pytest
import torch

transformers = pytest.importorskip("transformers")

from bonsai_bench.codecs import load_precision
from bonsai_bench.continuation import prefill, continue_from_prefill
from bonsai_bench.packed import encode_cache, decode_cache


@pytest.fixture(scope="module")
def tiny_qwen3():
    torch.manual_seed(43)
    config = transformers.Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=96,
                                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                     head_dim=32, max_position_embeddings=128,
                                     attention_dropout=0, bos_token_id=1, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "eager"
    return transformers.Qwen3ForCausalLM(config).to(torch.bfloat16).eval()


@pytest.mark.parametrize("padded", [False, True])
def test_bf16_handoff_matches_monolithic_greedy(tiny_qwen3, padded):
    ids = torch.tensor([[5, 11, 7, 43, 8, 6], [3, 4, 19, 9, 20, 10]])
    mask = torch.ones_like(ids)
    if padded:
        ids[0, :2] = 0
        mask[0, :2] = 0
    state = prefill(tiny_qwen3, ids, mask)
    original_keys = [layer.keys.clone() for layer in state.cache.layers]
    packet = encode_cache(state.cache, load_precision("bf16", head_dim=32))
    received = state.replace_cache(decode_cache(packet))
    generated = continue_from_prefill(tiny_qwen3, received, 6)
    monolithic = tiny_qwen3.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6,
                                    do_sample=False, eos_token_id=None, pad_token_id=0)
    assert torch.equal(generated, monolithic[:, ids.shape[1]:])
    assert torch.equal(generated, continue_from_prefill(tiny_qwen3, received, 6))
    assert torch.equal(generated[:, 0], state.next_token_logits.argmax(-1))
    for layer, key in zip(state.cache.layers, original_keys):
        assert torch.equal(layer.keys, key)
    assert received.cache.get_seq_length() == ids.shape[1]


def test_quantization_only_affects_received_prefix(tiny_qwen3):
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    state = prefill(tiny_qwen3, ids)
    baseline = state.cache.layers[0].keys.clone()
    for name in ["int8", "int4", "bf16"]:
        received = state.replace_cache(decode_cache(encode_cache(state.cache, load_precision(name, head_dim=32))))
        first = continue_from_prefill(tiny_qwen3, received, 1)
        assert torch.equal(first[:, 0], state.next_token_logits.argmax(-1))
        assert received.cache.layers[0].keys.dtype == torch.bfloat16
        assert received.cache.get_seq_length() == ids.shape[1]
    assert torch.equal(state.cache.layers[0].keys, baseline)
