"""PyTorch reference of Bonsai's KV codecs (fake quantization: quantize -> dequantize).

Numerics follow crates/bonsai-codec/src/reference.rs:
  symmetric : s = max|x| / qmax,            q = clamp(round(x/s), -qmax, qmax), x' = q*s
  asymmetric: s = (max-min) / (2^b - 1), z = min,  q = clamp(round((x-z)/s), 0, 2^b-1), x' = q*s + z
  s and z are rounded to f16 before use; math in f32; output cast back to the input dtype.

Codec and precision names match bonsai-ctl's config (`bonsai_ctl` built-ins plus
any [codecs.*] / [precisions.*] tables in a config file), so the precision you
evaluate here is exactly the one the control plane will use.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import torch

HEAD_DIM = "head_dim"   # groups of channels inside one (layer, head, token): per-token
TOKEN = "token"         # groups of tokens inside one (layer, head, channel): per-channel


@dataclasses.dataclass(frozen=True)
class QuantParams:
    group_size: int
    axis: str
    symmetric: bool


@dataclasses.dataclass(frozen=True)
class Codec:
    """bits = 16 means BF16 passthrough."""
    name: str
    bits: int
    key: QuantParams | None = None
    value: QuantParams | None = None
    # Python-only: reproduce Atom's exact quantizer (integer zero point, unrounded
    # scale). Used to check `atom-kv4` against the paper's scheme; not transferable.
    atom_exact: bool = False


@dataclasses.dataclass(frozen=True)
class Precision:
    """Codec per layer. `overrides` are (start, end, codec) half-open ranges; later wins."""
    name: str
    default: Codec
    overrides: tuple[tuple[int, int, Codec], ...] = ()

    def codec_for_layer(self, layer: int) -> Codec:
        for start, end, codec in reversed(self.overrides):
            if start <= layer < end:
                return codec
        return self.default


def builtin_codecs(head_dim: int = 128) -> dict[str, Codec]:
    asym = lambda g, axis: QuantParams(g, axis, False)  # noqa: E731
    return {
        "bf16": Codec("bf16", 16),
        "int8": Codec("int8", 8, QuantParams(head_dim, HEAD_DIM, True), QuantParams(head_dim, HEAD_DIM, True)),
        "int4": Codec("int4", 4, asym(32, TOKEN), asym(32, HEAD_DIM)),
        "int4-pertoken": Codec("int4-pertoken", 4, asym(32, HEAD_DIM), asym(32, HEAD_DIM)),
        "atom-kv4": Codec("atom-kv4", 4, asym(head_dim, HEAD_DIM), asym(head_dim, HEAD_DIM)),
        "atom-kv4-exact": Codec("atom-kv4-exact", 4, asym(head_dim, HEAD_DIM), asym(head_dim, HEAD_DIM), atom_exact=True),
    }


def load_precision(name: str, config: str | Path | None = None, head_dim: int = 128) -> Precision:
    """Resolve a codec or precision name, optionally using a bonsai-ctl TOML config."""
    codecs = builtin_codecs(head_dim)
    precisions: dict = {}
    if config is not None:
        cfg = tomllib.loads(Path(config).read_text())
        for cname, d in cfg.get("codecs", {}).items():
            if d["bits"] == 16:
                codecs[cname] = Codec(cname, 16)
            else:
                q = lambda t: QuantParams(t["group_size"], t["axis"], t["symmetric"])  # noqa: E731
                codecs[cname] = Codec(cname, d["bits"], q(d["key"]), q(d["value"]))
        precisions = cfg.get("precisions", {})

    def codec(n: str) -> Codec:
        if n not in codecs:
            raise KeyError(f"unknown codec {n!r}; known: {sorted(codecs)}")
        return codecs[n]

    if name in precisions:
        p = precisions[name]
        return Precision(
            name,
            codec(p["default"]),
            tuple((o["layers"][0], o["layers"][1], codec(o["codec"])) for o in p.get("overrides", [])),
        )
    return Precision(name, codec(name))


def _f16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16).to(torch.float32)


def _quant_groups(g: torch.Tensor, bits: int, symmetric: bool, atom_exact: bool) -> torch.Tensor:
    """Fake-quantize f32 groups laid out on the last dim."""
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        s = _f16(g.abs().amax(dim=-1, keepdim=True) / qmax)
        safe = torch.where(s == 0, torch.ones_like(s), s)
        q = torch.clamp(torch.round(g / safe), -qmax, qmax)
        return torch.where(s == 0, torch.zeros_like(g), q * s)
    qmax = 2**bits - 1
    lo, hi = g.amin(dim=-1, keepdim=True), g.amax(dim=-1, keepdim=True)
    if atom_exact:  # Atom (model/quant.py): integer zero point, scale clamped, not f16-rounded
        s = (hi - lo).clamp(min=1e-5) / qmax
        base = torch.round(-lo / s).clamp(0, qmax)
        return (torch.clamp(torch.round(g / s) + base, 0, qmax) - base) * s
    s, z = _f16((hi - lo) / qmax), _f16(lo)
    safe = torch.where(s == 0, torch.ones_like(s), s)
    q = torch.clamp(torch.round((g - z) / safe), 0, qmax)
    return torch.where(s == 0, z.expand_as(g), q * s + z)


def fake_quant(x: torch.Tensor, bits: int, p: QuantParams, atom_exact: bool = False) -> torch.Tensor:
    """x: [batch, heads, seq, head_dim] (HF layout). Returns same shape and dtype."""
    if bits >= 16:
        return x
    dtype, (b, h, t, d) = x.dtype, x.shape
    xf = x.to(torch.float32)
    if p.axis == HEAD_DIM:
        if d % p.group_size:
            raise ValueError(f"head_dim {d} not divisible by group_size {p.group_size}")
        out = _quant_groups(xf.reshape(b, h, t, d // p.group_size, p.group_size), bits, p.symmetric, atom_exact)
        return out.reshape(b, h, t, d).to(dtype)
    if p.axis == TOKEN:
        g = p.group_size
        xt = xf.transpose(2, 3)                     # [b, h, d, t]
        pad = (-t) % g
        if pad:                                     # edge-pad: keeps each group's min/max unchanged
            xt = torch.cat([xt, xt[..., -1:].expand(b, h, d, pad)], dim=-1)
        out = _quant_groups(xt.reshape(b, h, d, -1, g), bits, p.symmetric, atom_exact)
        out = out.reshape(b, h, d, t + pad)[..., :t].transpose(2, 3)
        return out.contiguous().to(dtype)
    raise ValueError(f"unknown axis {p.axis!r}")


def roundtrip_layer(k: torch.Tensor, v: torch.Tensor, codec: Codec) -> tuple[torch.Tensor, torch.Tensor]:
    if codec.bits >= 16:
        return k, v
    return (fake_quant(k, codec.bits, codec.key, codec.atom_exact),
            fake_quant(v, codec.bits, codec.value, codec.atom_exact))


def encoded_bytes_per_token(precision: Precision, num_layers: int, num_kv_heads: int, head_dim: int) -> float:
    """Amortized wire bytes/token (payload + metadata, long-sequence limit); mirrors bonsai-core."""
    total = 0.0
    for layer in range(num_layers):
        c = precision.codec_for_layer(layer)
        for p in (c.key, c.value):
            elems = num_kv_heads * head_dim
            if c.bits >= 16:
                total += elems * 2
                continue
            meta = 2 if p.symmetric else 4
            total += elems * c.bits / 8 + elems / p.group_size * meta
    return total
