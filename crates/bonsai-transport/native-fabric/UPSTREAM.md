Vendored from https://github.com/perplexityai/pplx-garden at
`1ed972ed3f0bd5616c997c9507c25616c63394fc`, fabric-lib/src only.
The original Apache-2.0 license is included in LICENSE.

Only source change: expose the existing NIC discovery helper as
`fabric_lib::visible_domains` (rename its internal call site).
`visible-domains.patch` reproduces that change. Cargo.toml replaces inherited
workspace dependencies with the same version specifications and keeps cuda-lib,
thread-lib and both native sys crates at the exact upstream git revision.

Reason: the pinned topology scanner reads NUMA data from a fabricated PCI
function (device/function zero). H200 hardware observed in Slurm job 1294320
fails with `Failed to read NUMA node`. Its private discovery types prevent an
external caller from constructing a DomainInfo without successful topology
scan. Bonsai uses the exported discovery helper and performs PCI/NUMA selection
in its own topology.rs, then uses the original Worker and TransferEngine.
No RDMA, registration, CUDA, completion or transport code is changed here.
