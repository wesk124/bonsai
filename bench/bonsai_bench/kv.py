"""Apply a Bonsai handoff to a Hugging Face KV cache.

`roundtrip(cache, precision)` returns the cache the decode worker would end up
with: every layer encoded and decoded with that layer's codec. It uses the
`bonsai` extension module (CUDA kernels) when available, otherwise the PyTorch
reference in `bonsai_bench.codecs`. The network moves bytes unchanged, so this
is numerically identical to a real two-node handoff.
"""

from __future__ import annotations

from typing import Any

from .codecs import Precision, roundtrip_layer

try:  # built from crates/bonsai-py (milestone 2)
    import bonsai as _native  # type: ignore
except ImportError:  # pragma: no cover
    _native = None


def _layers(cache: Any) -> list[tuple[Any, Any]]:
    """(K, V) per layer for transformers 4.x DynamicCache, 5.x DynamicCache, or legacy tuples."""
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):  # transformers >= 4.56 / 5.x
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):                                                       # transformers 4.36-4.55
        return list(zip(cache.key_cache, cache.value_cache))
    return [tuple(kv) for kv in cache]                                                     # legacy tuple-of-tuples


def _set_layer(cache: Any, i: int, k: Any, v: Any) -> Any:
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):
        cache.layers[i].keys, cache.layers[i].values = k, v
    elif hasattr(cache, "key_cache"):
        cache.key_cache[i], cache.value_cache[i] = k, v
    else:
        raise TypeError("tuple caches are immutable; use roundtrip()'s return value")
    return cache


def roundtrip(cache: Any, precision: Precision, use_native: bool | None = None) -> Any:
    """In-place for DynamicCache (returned for convenience); new tuple for legacy caches."""
    native = _native is not None if use_native is None else use_native
    if native:
        if _native is None:
            raise RuntimeError("bonsai extension module not built")
        return _native.roundtrip_kv(cache, precision.name)
    layers = _layers(cache)
    if isinstance(cache, tuple):
        return tuple(roundtrip_layer(k, v, precision.codec_for_layer(i)) for i, (k, v) in enumerate(layers))
    for i, (k, v) in enumerate(layers):
        nk, nv = roundtrip_layer(k, v, precision.codec_for_layer(i))
        _set_layer(cache, i, nk, nv)
    return cache
