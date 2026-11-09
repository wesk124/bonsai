"""Validate final/pilot completeness and immutable identities, then write index."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    index = {}
    seen_ids, seen_hashes = set(), set()
    tasks = {"niah_single_2", "niah_multikey_1", "niah_multiquery"}
    for split, per_cell in (("pilot", 5), ("final", 100)):
        entries = []
        for cap in (4096, 8192, 16384):
            for task in sorted(tasks):
                path = root / split / str(cap) / (task + ".jsonl")
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if len(rows) != per_cell:
                    raise ValueError(f"{path}: expected {per_cell}, got {len(rows)}")
                for row in rows:
                    assert (row["task"], row["context_tokens"], row["split"]) == (task, cap, split)
                    assert row["input_length"] == len(row["input_ids"])
                    assert row["input_length"] + row["max_new_tokens"] <= cap
                    assert row["sample_id"] not in seen_ids
                    assert row["hashes"]["prompt_sha256"] not in seen_hashes
                    assert hashlib.sha256(row["prompt"].encode()).hexdigest() == row["hashes"]["prompt_sha256"]
                    assert row["outputs"] and all(any(f["target"] and f["value"] == v for f in row["facts"]) for v in row["outputs"])
                    seen_ids.add(row["sample_id"])
                    seen_hashes.add(row["hashes"]["prompt_sha256"])
                entries.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    count=len(rows), seed=rows[0]["seed"],
                                    input_length_min=min(r["input_length"] for r in rows),
                                    input_length_max=max(r["input_length"] for r in rows)))
        index[split] = dict(count=sum(e["count"] for e in entries), expected_count=9 * per_cell,
                            complete=True, files=entries)
    (root / "suite_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("count", "expected_count", "complete")} for k, v in index.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/ruler"))
    audit(parser.parse_args().root)
