"""Prefix-handoff perplexity on WikiText2 / PTB / C4 (Atom's accuracy benchmark,
adapted to KV transfer).

Atom fake-quantizes the KV cache inside every attention call, which measures a
quantized-KV *model*. Bonsai only quantizes KV *in transit*: the decode worker
attends over a dequantized copy of the prefill cache, while its own new tokens
stay BF16. So each window of `seqlen` tokens is split:

    prefix  = window[:P]     -> prefill (BF16), cache -> roundtrip(precision)
    suffix  = window[P:]     -> teacher-forced decode over the round-tripped cache
    score   = NLL of window[P+1:]  (window[P] comes from prefill logits, which the
              handoff cannot change)

With precision = bf16 this equals the plain model's perplexity on those
positions (test_ppl.py checks it). It is a cheap, continuous quality signal,
good for sweeping many configs and for the per-layer sensitivity sweep that
picks protected layers for mixed precision.

    python -m bonsai_bench.ppl --datasets wikitext2,c4 --precisions bf16,int8,int4,atom-kv4

Layer sensitivity (input for mixed-precision `[precisions.*]`): quantize one
layer at a time with CODEC, keep every other layer BF16, and report each
layer's perplexity increase. The most sensitive layers are the ones to protect:

    python -m bonsai_bench.ppl --datasets wikitext2 --layer-sensitivity int4 --limit 40
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .codecs import Precision, load_precision, builtin_codecs
from .hf import head_dim, load_model
from .kv import roundtrip


def load_tokens(name: str, tokenizer, max_tokens: int | None = None) -> torch.Tensor:
    """1-D tensor of evaluation tokens (test/validation splits, as in GPTQ/Atom)."""
    from datasets import load_dataset

    if name == "wikitext2":
        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    elif name == "ptb":
        text = "\n\n".join(load_dataset("ptb_text_only", "penn_treebank", split="validation", trust_remote_code=True)["sentence"])
    elif name == "c4":
        ds = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
        text = " ".join(ds.select(range(1100))["text"])
    else:
        raise ValueError(f"unknown dataset {name!r} (wikitext2 | ptb | c4)")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return ids[:max_tokens] if max_tokens else ids


@torch.no_grad()
def handoff_nll(model, window: torch.Tensor, prefix_len: int, precision: Precision) -> tuple[float, int]:
    """Sum of NLL over window[prefix_len+1:] after a KV handoff at prefix_len."""
    ids = window.unsqueeze(0).to(model.device)
    out = model(input_ids=ids[:, :prefix_len], use_cache=True)
    cache = roundtrip(out.past_key_values, precision)
    out2 = model(input_ids=ids[:, prefix_len:], past_key_values=cache, use_cache=True)
    logits = out2.logits[0, :-1].float()
    targets = ids[0, prefix_len + 1:]
    return F.cross_entropy(logits, targets, reduction="sum").item(), targets.numel()


@torch.no_grad()
def reference_nll(model, window: torch.Tensor, prefix_len: int) -> tuple[float, int]:
    """Same positions, single forward pass, no handoff (sanity baseline)."""
    ids = window.unsqueeze(0).to(model.device)
    logits = model(input_ids=ids).logits[0, prefix_len:-1].float()
    targets = ids[0, prefix_len + 1:]
    return F.cross_entropy(logits, targets, reduction="sum").item(), targets.numel()


def evaluate(model, tokens: torch.Tensor, precision: Precision, seqlen: int, prefix_len: int,
             limit: int = 0) -> dict:
    n = tokens.numel() // seqlen
    if limit:
        n = min(n, limit)
    nll, count = 0.0, 0
    for i in range(n):
        s, c = handoff_nll(model, tokens[i * seqlen:(i + 1) * seqlen], prefix_len, precision)
        nll, count = nll + s, count + c
    return {"ppl": math.exp(nll / max(count, 1)), "windows": n, "scored_tokens": count}


def _precisions(a, num_layers: int, hd: int):
    if not a.layer_sensitivity:
        for pname in a.precisions.split(","):
            yield pname, load_precision(pname, a.config, hd)
        return
    bf16 = builtin_codecs(hd)["bf16"]
    codec = load_precision(a.layer_sensitivity, a.config, hd).default
    yield "bf16", Precision("bf16", bf16)
    for layer in range(num_layers):
        name = f"{codec.name}@L{layer}"
        yield name, Precision(name, bf16, ((layer, layer + 1, codec),))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--datasets", default="wikitext2,c4")
    ap.add_argument("--precisions", default="bf16,int8,int4,atom-kv4")
    ap.add_argument("--config", help="bonsai-ctl TOML, for custom [codecs]/[precisions]")
    ap.add_argument("--seqlen", type=int, default=2048, help="window length (Atom uses 2048)")
    ap.add_argument("--prefix-frac", type=float, default=0.5, help="handoff point inside each window")
    ap.add_argument("--limit", type=int, default=0, help="max windows per dataset (0 = all)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layer-sensitivity", metavar="CODEC",
                    help="instead of --precisions: quantize one layer at a time with CODEC")
    ap.add_argument("--out", default="results/ppl.jsonl")
    a = ap.parse_args()

    model, tok = load_model(a.model, a.dtype, a.device)
    hd = head_dim(model)
    prefix_len = int(a.seqlen * a.prefix_frac)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for ds in a.datasets.split(","):
            tokens = load_tokens(ds, tok)
            for pname, prec in _precisions(a, model.config.num_hidden_layers, hd):
                t0 = time.time()
                r = evaluate(model, tokens, prec, a.seqlen, prefix_len, a.limit)
                rec = {"bench": "ppl", "model": a.model, "dataset": ds, "precision": pname,
                       "seqlen": a.seqlen, "prefix_len": prefix_len, **r, "secs": round(time.time() - t0, 1)}
                print(json.dumps(rec), flush=True)
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
