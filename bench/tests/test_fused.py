"""GPU wire equality tests; CPU rejection stays testable on login nodes."""
import copy

import pytest
import torch

pytest.importorskip("triton")
from bonsai_bench import fused, packed
from bonsai_bench.codecs import Codec, Precision, QuantParams, HEAD_DIM, TOKEN, load_precision

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def test_cpu_is_explicitly_unsupported():
    x = torch.zeros(1, 1, 1, 32)
    with pytest.raises(ValueError, match="requires CUDA"):
        fused.encode_cache(((x, x),), load_precision("bf16"))
    packet = packed.encode_cache(((x, x),), load_precision("bf16"))
    with pytest.raises(ValueError, match="requires CUDA"):
        fused.decode_cache(packet)


def _assert_same(cache, precision):
    reference = packed.encode_cache(cache, precision)
    actual = fused.encode_cache(cache, precision)
    assert actual.metadata_bytes() == reference.metadata_bytes()
    assert actual.wire_nbytes == reference.wire_nbytes
    assert torch.equal(actual.buffer, reference.buffer), (precision.name, torch.nonzero(actual.buffer != reference.buffer)[:10].cpu())
    expected = packed.decode_cache(reference, as_legacy=True)
    for source in (reference, actual):
        outputs = fused.decode_cache(source, as_legacy=True)
        for layer, ((k, v), (rk, rv)) in enumerate(zip(outputs, expected)):
            assert torch.equal(k, rk) and torch.equal(v, rv), precision.name
            assert k.is_contiguous() and v.is_contiguous()
            assert k.data_ptr() != cache[layer][0].data_ptr()
            assert v.data_ptr() != cache[layer][1].data_ptr()
            assert not torch._C._is_alias_of(k, source.buffer)
    return actual


@GPU
@pytest.mark.parametrize("precision", ["bf16", "int8", "int4", "int4-pertoken", "atom-kv4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tail,strided", [(False, False), (True, True)])
def test_full_frame_equality(precision, dtype, tail, strided):
    torch.manual_seed(81)
    t, d = (35, 35) if tail else (64, 128)
    if strided:
        # Distinct K/V strides, including channel and token strides > 1.
        k = torch.randn(2, t * 2, 3, d * 2, device="cuda", dtype=dtype)[:, ::2, :, ::2].permute(0, 2, 1, 3)
        v = torch.randn(2, 3, t * 2, d * 3, device="cuda", dtype=dtype)[:, :, ::2, ::3]
    else:
        k = torch.randn(2, 3, t, d, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
    _assert_same(((k, v), (v, k)), load_precision(precision, head_dim=d))


@GPU
@pytest.mark.parametrize("g,axis,bits,symmetric", [(1, HEAD_DIM, 4, False), (3, HEAD_DIM, 4, True),
                                                  (3, TOKEN, 4, False), (6, TOKEN, 4, False),
                                                  (3, TOKEN, 8, True), (32, TOKEN, 8, False)])
def test_odd_groups_unaligned_frames(g, axis, bits, symmetric):
    torch.manual_seed(19)
    p = QuantParams(g, axis, symmetric)
    q = Codec("custom", bits, p, p)
    bf16 = load_precision("bf16").default
    precision = Precision("mixed", q, ((1, 2, bf16),))
    x = torch.randn(1, 1, 5, 7, device="cuda").bfloat16()
    _assert_same(((x, x.transpose(-1, -2)), (x, x)), precision)


@GPU
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_raw_dtype_bits_and_fresh_cache(dtype):
    x = torch.tensor([0., -0., float("nan"), float("inf"), -float("inf"), 1., -1.], device="cuda", dtype=dtype).reshape(1, 1, 1, 7)
    cache = ((x, x),)
    a = packed.encode_cache(cache, load_precision("bf16"))
    b = fused.encode_cache(cache, load_precision("bf16"))
    assert torch.equal(a.buffer, b.buffer)
    decoded = fused.decode_cache(b)
    repacked = fused.encode_cache(decoded, load_precision("bf16"))
    assert torch.equal(repacked.buffer, a.buffer)
    assert decoded.layers[0].keys.data_ptr() != x.data_ptr()


@GPU
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e10])
def test_invalid_input_is_rejected(value):
    good = torch.zeros((1, 1, 3, 32), device="cuda")
    bad = good.clone()
    bad[..., 0] = value
    with pytest.raises(ValueError, match="nonfinite|overflows"):
        fused.encode_cache(((good, good), (bad, good)), load_precision("int4"))


@GPU
def test_invalid_received_metadata_rejected():
    x = torch.ones((1, 1, 5, 32), device="cuda").bfloat16()
    a = fused.encode_cache(((x, x),), load_precision("int4"))
    for index in (0, 1):
        b = packed.PackedCache(a.buffer.clone(), copy.deepcopy(a.metadata))
        section = b.metadata["layers"][0][index]
        offset = section["offset"] + section["tensor"]["quant_bytes"]
        b.buffer[offset:offset + 2] = torch.tensor([0, 124], device="cuda", dtype=torch.uint8)
        with pytest.raises(ValueError, match="nonfinite"):
            fused.decode_cache(b)
    b = packed.PackedCache(a.buffer, copy.deepcopy(a.metadata))
    b.metadata["layers"][0][0]["tensor"]["quant_count"] += 1
    with pytest.raises(ValueError, match="metadata"):
        fused.decode_cache(b)


@GPU
def test_one_error_check_per_cache_and_constant_groups(monkeypatch):
    checks = []
    original = fused._check
    def check(flag):
        checks.append(flag)
        original(flag)
    monkeypatch.setattr(fused, "_check", check)
    for name in ("int8", "int4"):
        x = torch.full((1, 2, 35, 32), -0.5, device="cuda", dtype=torch.bfloat16)
        packet = fused.encode_cache(((x, x), (x, x)), load_precision(name, head_dim=32))
        assert len(checks) == 1
        fused.decode_cache(packet)
        assert len(checks) == 2
        checks.clear()
    packet = fused.encode_cache(((x, x),), load_precision("bf16"))
    fused.decode_cache(packet)
    assert not checks


@GPU
def test_round_to_nearest_even_and_underflow():
    x = torch.tensor([-127., 127., -2.5, -1.5, -0.5, 0.5, 1.5, 2.5] * 4, device="cuda").reshape(1, 1, 1, 32)
    _assert_same(((x, x),), load_precision("int8", head_dim=32))
    x = torch.tensor([0., 15., 0.5, 1.5, 2.5, 3.5, 4.5, 5.5] * 4, device="cuda").reshape(1, 1, 1, 32)
    _assert_same(((x, x),), load_precision("int4-pertoken"))
    x = torch.linspace(1e-10, 2e-10, 35, device="cuda").reshape(1, 1, 5, 7)
    _assert_same(((x, x),), load_precision("int4", head_dim=7))


@GPU
def test_varying_prompt_lengths_reuse_kernel_specializations():
    # Every T and tensor stride must remain runtime, including section offsets.
    # The JIT argument annotation contract is independent of cache internals.
    for function in (fused._encode, fused._decode, fused._raw_encode, fused._raw_decode):
        for parameter in function.params:
            if parameter.name in ("T", "H", "S0", "S1", "S2", "S3", "NG", "SC", "QB", "OFF", "N"):
                assert not parameter.is_constexpr
                assert parameter.do_not_specialize
    for t in (33, 34, 47):
        x = torch.randn(1, 2, t, 32, device="cuda", dtype=torch.bfloat16)
        _assert_same(((x, x),), load_precision("int4", head_dim=32))


@GPU
@pytest.mark.parametrize("bits,symmetric", [(8, True), (4, True), (4, False)])
def test_half_scale_midpoint_rounding(bits, symmetric):
    maxq = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1
    scales = torch.tensor([2.0 ** e * 0.75 for e in (-20, -10, -1, 0, 4, 13)], device="cuda", dtype=torch.float16)
    next_scales = torch.nextafter(scales, torch.full_like(scales, float("inf")))
    midpoint = (scales.float() + next_scales.float()) * 0.5
    bounds = midpoint * maxq
    bounds = torch.stack((torch.nextafter(bounds, torch.full_like(bounds, -float("inf"))),
                          bounds, torch.nextafter(bounds, torch.full_like(bounds, float("inf")))), dim=1).reshape(-1)
    x = torch.zeros((1, 1, bounds.numel(), 32), device="cuda")
    x[..., -1] = bounds
    p = QuantParams(32, HEAD_DIM, symmetric)
    _assert_same(((x, x),), Precision("midpoint", Codec("midpoint", bits, p, p)))


@GPU
def test_dynamic_cache_adopts_outputs_without_hidden_cat(monkeypatch):
    x = torch.randn(1, 2, 35, 32, device="cuda", dtype=torch.bfloat16)
    packet = fused.encode_cache(((x, x), (x, x)), load_precision("int4", head_dim=32))
    reference = packed.decode_cache(packet, as_legacy=True)
    def forbidden_cat(*args, **kwargs):
        raise AssertionError("fused decode must adopt outputs without concatenating a copy")
    with monkeypatch.context() as patch:
        patch.setattr(torch, "cat", forbidden_cat)
        cache = fused.decode_cache(packet)
    assert cache.get_seq_length() == 35
    for layer, (k, v) in zip(cache.layers, reference):
        assert torch.equal(layer.keys, k) and torch.equal(layer.values, v)
        assert layer.dtype == x.dtype and layer.device == x.device
    # The adopted layers still behave as normal mutable HF dynamic layers.
    cache.update(x[:, :, :1], x[:, :, :1], 0)
    assert cache.get_seq_length() == 36


@GPU
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("base_offset", [0, 1, 3])
def test_raw_decode_offset_buffer_view(dtype, mixed, base_offset):
    # Raw dtype bits include signed zero, NaN and infinities. Comparing float
    # equality would miss bit preservation and cannot compare NaNs directly.
    values = torch.tensor([0., -0., float("nan"), float("inf"), -float("inf"), 0.5, -3.25],
                          device="cuda", dtype=dtype).reshape(1, 1, 1, 7)
    cache = ((values, values),)
    precision = load_precision("bf16")
    if mixed:
        # First quantized layer occupies 5+6=11 bytes. Following raw sections
        # have odd offsets, exercising section and base alignment independently.
        first = torch.full((1, 1, 1, 1), 0.5, device="cuda")
        quant = Codec("odd-sections", 4, QuantParams(1, HEAD_DIM, False), QuantParams(3, HEAD_DIM, False))
        precision = Precision("mixed-offset", precision.default, ((0, 1, quant),))
        cache = ((first, first), *cache)
    reference_packet = packed.encode_cache(cache, precision)
    backing = torch.empty(reference_packet.payload_nbytes + base_offset, dtype=torch.uint8, device="cuda")
    buffer = backing[base_offset:]
    buffer.copy_(reference_packet.buffer)
    packet = packed.PackedCache(buffer, copy.deepcopy(reference_packet.metadata))
    assert buffer.is_contiguous()
    assert buffer.data_ptr() == backing.data_ptr() + base_offset
    expected = packed.decode_cache(packet, as_legacy=True)
    for as_legacy in (True, False):
        decoded = fused.decode_cache(packet, as_legacy=as_legacy)
        pairs = decoded if as_legacy else tuple((layer.keys, layer.values) for layer in decoded.layers)
        for actual_pair, reference_pair in zip(pairs, expected):
            for actual, reference in zip(actual_pair, reference_pair):
                assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
                assert actual.is_contiguous()
                assert not torch._C._is_alias_of(actual, buffer)
                assert not torch._C._is_alias_of(actual, values)
        canonical = load_precision("bf16")
        assert torch.equal(fused.encode_cache(decoded, canonical).buffer, packed.encode_cache(expected, canonical).buffer)
    assert torch.equal(buffer, reference_packet.buffer)
