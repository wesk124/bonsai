"""Bonsai benchmarks: KV-handoff quality evals, workload generators, codec microbenchmarks.

Quality evals use `bonsai_bench.codecs`, a PyTorch reference of Bonsai's codecs
(same numerics as crates/bonsai-codec/src/reference.rs). Once the `bonsai`
extension module is built, `bonsai_bench.kv.roundtrip` uses the CUDA kernels
instead, with no change to the eval scripts.
"""
