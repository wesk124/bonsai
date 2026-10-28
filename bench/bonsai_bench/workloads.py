"""Request workloads for latency/throughput benchmarks.

* `atom_lognormal`: the synthetic request distribution used in Atom's (MLSys'24)
  end-to-end serving benchmark: prompt length ~ lognormal(s=0.8, loc=-1, scale=18),
  total length ~ uniform[0, maxlen), seed 0xabcdabcd987. Reimplemented from
  the paper's description and parameters (the Atom repo has no license file,
  so no code is copied).
* `sharegpt`: prompt/output lengths (and prompt text) sampled from ShareGPT
  conversations, Atom's "real production" trace.
* `fixed`: controlled context-length sweep, the main axis for KV-handoff
  experiments (Atom-style workloads have short prompts, where the KV cache is small).

All return a `RequestSet` that can be saved as JSONL for the load generator.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import numpy as np

ATOM_SEED = 0xABCDABCD987


@dataclasses.dataclass
class RequestSet:
    name: str
    prompt_lens: np.ndarray
    output_lens: np.ndarray
    prompts: list[str] | None = None   # text, when the workload has real prompts

    def __len__(self) -> int:
        return len(self.prompt_lens)

    def summary(self) -> dict:
        p, o = self.prompt_lens, self.output_lens
        return {
            "name": self.name, "requests": len(self),
            "prompt_mean": float(p.mean()), "prompt_p50": float(np.median(p)), "prompt_p99": float(np.percentile(p, 99)),
            "output_mean": float(o.mean()), "output_p50": float(np.median(o)),
        }

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for i in range(len(self)):
                rec = {"id": i, "workload": self.name,
                       "prompt_len": int(self.prompt_lens[i]), "max_new_tokens": int(self.output_lens[i])}
                if self.prompts is not None:
                    rec["prompt"] = self.prompts[i]
                f.write(json.dumps(rec) + "\n")


def atom_lognormal(num_requests: int, maxlen: int = 2048, seed: int = ATOM_SEED) -> RequestSet:
    import scipy.stats

    rng = np.random.Generator(np.random.PCG64(seed=seed))
    prompt_dist = scipy.stats.lognorm(0.8, -1.0, 18.0)
    total_dist = scipy.stats.randint(0, maxlen)
    prompts, outputs = [], []
    for _ in range(num_requests):
        p = int(min(max(1, prompt_dist.rvs(random_state=rng)), maxlen - 2))
        t = int(max(p + 2, total_dist.rvs(random_state=rng)))
        prompts.append(p)
        outputs.append(t - p)
    return RequestSet("atom-lognormal", np.array(prompts, np.int32), np.array(outputs, np.int32))


def sharegpt(path: str | Path, tokenizer, num_requests: int, seed: int = 0,
             max_prompt: int = 32768, max_output: int = 2048, min_len: int = 4) -> RequestSet:
    """Sample (first human turn, first assistant reply length) pairs from
    ShareGPT_V3_unfiltered_cleaned_split.json."""
    data = json.loads(Path(path).read_text())
    convs = [c["conversations"] for c in data if len(c.get("conversations", [])) >= 2]
    random.Random(seed).shuffle(convs)
    prompts, plens, olens = [], [], []
    for conv in convs:
        if conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
            continue
        prompt, reply = conv[0]["value"], conv[1]["value"]
        p = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        o = len(tokenizer(reply, add_special_tokens=False).input_ids)
        if p < min_len or o < min_len or p > max_prompt:
            continue
        prompts.append(prompt)
        plens.append(p)
        olens.append(min(o, max_output))
        if len(prompts) == num_requests:
            break
    return RequestSet("sharegpt", np.array(plens, np.int32), np.array(olens, np.int32), prompts)


def fixed(context_lens: list[int], output_len: int = 32, per_len: int = 8) -> RequestSet:
    plens = np.repeat(np.array(context_lens, np.int32), per_len)
    return RequestSet("fixed", plens, np.full_like(plens, output_len))


def synthetic_prompt_ids(prompt_len: int, vocab_size: int, seed: int) -> list[int]:
    """Random token ids for length-only workloads (latency doesn't depend on content)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(100, vocab_size - 100, size=prompt_len).tolist()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate a workload JSONL for the Bonsai load generator")
    ap.add_argument("kind", choices=["atom-lognormal", "sharegpt", "fixed"])
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--sharegpt-path")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--context-lens", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.kind == "atom-lognormal":
        rs = atom_lognormal(a.num, a.maxlen)
    elif a.kind == "sharegpt":
        from transformers import AutoTokenizer
        rs = sharegpt(a.sharegpt_path, AutoTokenizer.from_pretrained(a.tokenizer), a.num)
    else:
        rs = fixed([int(x) for x in a.context_lens.split(",")], a.output_len, max(1, a.num // 6))
    rs.save_jsonl(a.out)
    print(json.dumps(rs.summary()))


if __name__ == "__main__":
    main()
