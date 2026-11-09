"""Apply a real packed Bonsai PyTorch handoff to a Hugging Face KV cache."""
from __future__ import annotations

from typing import Any

from .codecs import Precision, roundtrip_layer

try:
    import bonsai as _candidate  # type: ignore
except ImportError:
    _candidate = None
# A repository directory can import as the namespace package ``bonsai``. It is
# not a native extension and must never cause a native backend label.
_native = _candidate if callable(getattr(_candidate, "roundtrip_kv", None)) else None


def native_available() -> bool:
    return _native is not None


def _layers(cache: Any) -> list[tuple[Any, Any]]:
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [tuple(kv) for kv in cache]


def _set_layer(cache: Any, i: int, k: Any, v: Any) -> Any:
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):
        cache.layers[i].keys, cache.layers[i].values = k, v
    elif hasattr(cache, "key_cache"):
        cache.key_cache[i], cache.value_cache[i] = k, v
    else:
        raise TypeError("tuple caches are immutable; use roundtrip()'s return value")
    return cache


def roundtrip(cache: Any, precision: Precision, use_native: bool | None = None) -> Any:
    """Independent returned cache; callers can compare precisions without aliasing.

    Native execution is explicit. The default always runs this module's actual
    packed PyTorch backend. Atom-exact is a separately named fake-quant baseline.
    """
    if use_native:
        if _native is None:
            raise RuntimeError("native bonsai.roundtrip_kv callable is unavailable")
        return _native.roundtrip_kv(cache, precision.name)
    if any(precision.codec_for_layer(i).atom_exact for i in range(len(_layers(cache)))):
        pairs = tuple(roundtrip_layer(k, v, precision.codec_for_layer(i))
                      for i, (k, v) in enumerate(_layers(cache)))
        if isinstance(cache, tuple):
            return tuple((k.clone(), v.clone()) for k, v in pairs)
        from transformers import DynamicCache
        fresh = DynamicCache()
        for i, (k, v) in enumerate(pairs):
            fresh.update(k.clone(), v.clone(), i)
        return fresh
    from .packed import encode_cache, decode_cache
    return decode_cache(encode_cache(cache, precision), as_legacy=isinstance(cache, tuple))
