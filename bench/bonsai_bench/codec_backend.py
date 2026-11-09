"""Explicit, lazy selection of wire-compatible cache implementations."""
from importlib import import_module

DEFAULT_BACKEND = "pytorch_packed_v1"
BACKENDS = (DEFAULT_BACKEND, "triton_packed_v1")


def get_codec_backend(name=DEFAULT_BACKEND):
    """Never silently substitute an implementation in a measured run."""
    if name not in BACKENDS:
        raise ValueError(f"unknown codec backend: {name}")
    module = "packed" if name == DEFAULT_BACKEND else "fused"
    return import_module(f"bonsai_bench.{module}")
