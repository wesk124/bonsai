"""Zero-shot commonsense accuracy under KV handoff (Atom's second accuracy benchmark).

Atom reports zero-shot accuracy on PIQA, ARC-easy, ARC-challenge, BoolQ,
HellaSwag and WinoGrande with lm-evaluation-harness. This module plugs a
handoff-aware model into lm-eval >= 0.4:

    context       -> prefill (BF16) -> logits for continuation[0]   (unaffected)
    KV cache      -> roundtrip(precision)
    continuation  -> teacher-forced decode over the round-tripped cache

Contexts are short (tens to hundreds of tokens), so this is a regression
check that short-context quality is unchanged, not where compression pays off.
LongBench and RULER remain the main quality benchmarks (see docs/BENCHMARKS.md).

    python -m bonsai_bench.zeroshot --precisions bf16,int8,int4 --limit 500
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import torch

from .codecs import Precision, load_precision
from .hf import head_dim, load_model
from .kv import roundtrip

ATOM_TASKS = ["piqa", "arc_easy", "arc_challenge", "boolq", "hellaswag", "winogrande"]


def make_lm(model, tokenizer, precision: Precision):
    from lm_eval.api.model import LM

    class BonsaiHandoffLM(LM):
        def __init__(self):
            super().__init__()
            self.model, self.tok, self.precision = model, tokenizer, precision
            fallback = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            self.prefix_for_empty = [fallback]

        def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
            # lm-eval convention: trailing context whitespace belongs to the continuation.
            n = len(context) - len(context.rstrip())
            if n:
                continuation, context = context[-n:] + continuation, context[:-n]
            whole = self.tok(context + continuation, add_special_tokens=False).input_ids
            ctx = self.tok(context, add_special_tokens=False).input_ids
            cont = whole[len(ctx):]
            return (ctx or self.prefix_for_empty), cont

        @torch.no_grad()
        def _prefill(self, ctx: list[int]):
            ids = torch.tensor([ctx], device=self.model.device)
            out = self.model(input_ids=ids, use_cache=True)
            return out.logits[0, -1].float(), roundtrip(out.past_key_values, self.precision)

        @torch.no_grad()
        def _score(self, first_logits, cache, cont: list[int]) -> tuple[float, bool]:
            logits = first_logits.unsqueeze(0)
            if len(cont) > 1:
                ids = torch.tensor([cont[:-1]], device=self.model.device)
                out = self.model(input_ids=ids, past_key_values=copy.deepcopy(cache), use_cache=True)
                logits = torch.cat([logits, out.logits[0].float()], dim=0)
            logp = torch.log_softmax(logits, dim=-1)
            tgt = torch.tensor(cont, device=logp.device)
            ll = logp.gather(1, tgt.unsqueeze(1)).sum().item()
            return ll, bool((logp.argmax(-1) == tgt).all())

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            # Multiple-choice requests share a context: prefill + round-trip once per context.
            encoded = [self._encode_pair(*r.args) for r in requests]
            by_ctx = defaultdict(list)
            for i, (ctx, _) in enumerate(encoded):
                by_ctx[tuple(ctx)].append(i)
            results = [None] * len(requests)
            for ctx, idxs in by_ctx.items():
                first, cache = self._prefill(list(ctx))
                for i in idxs:
                    results[i] = self._score(first, cache, encoded[i][1])
            return results

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError("use bonsai_bench.ppl for perplexity")

        def generate_until(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError("generation tasks are evaluated with the LongBench harness")

    return BonsaiHandoffLM()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tasks", default=",".join(ATOM_TASKS))
    ap.add_argument("--precisions", default="bf16,int8,int4")
    ap.add_argument("--config")
    ap.add_argument("--limit", type=int, default=None, help="examples per task (None = all)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/zeroshot.jsonl")
    a = ap.parse_args()

    import lm_eval

    model, tok = load_model(a.model, a.dtype)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for pname in a.precisions.split(","):
        lm = make_lm(model, tok, load_precision(pname, a.config, head_dim(model)))
        res = lm_eval.simple_evaluate(model=lm, tasks=a.tasks.split(","), num_fewshot=0, limit=a.limit)
        for task, metrics in res["results"].items():
            rec = {"bench": "zeroshot", "model": a.model, "precision": pname, "task": task,
                   "acc": metrics.get("acc,none"), "acc_norm": metrics.get("acc_norm,none")}
            print(json.dumps(rec), flush=True)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
