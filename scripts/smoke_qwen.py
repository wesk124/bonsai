"""Real-checkpoint integration smoke; this is not a RULER measurement."""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bonsai_bench.codecs import load_precision
from bonsai_bench.continuation import prefill, continue_from_prefill
from bonsai_bench.packed import encode_cache, decode_cache

REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REV, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
).cuda().eval()
text = tokenizer.apply_chat_template([
    {"role": "user", "content": "The archive access code is 58274913. What is the archive access code? Return only the code."}
], tokenize=False, add_generation_prompt=True)
ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
with torch.inference_mode():
    state = prefill(model, ids)
    direct = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=32,
                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)[:, ids.shape[1]:]
    rows = []
    for precision in ("bf16", "int8", "int4"):
        torch.cuda.synchronize()
        start = time.perf_counter()
        packet = encode_cache(state.cache, load_precision(precision, head_dim=model.config.head_dim))
        decoded = decode_cache(packet)
        torch.cuda.synchronize()
        codec_ms = (time.perf_counter() - start) * 1000
        output = continue_from_prefill(model, state.replace_cache(decoded), 32, tokenizer.eos_token_id)
        if precision == "bf16":
            assert torch.equal(direct, output), (direct, output)
        row = dict(kind="non_ruler_smoke", model=MODEL, revision=REV, precision=precision,
                   prompt_tokens=ids.shape[1], prediction=tokenizer.decode(output[0], skip_special_tokens=True),
                   output_ids=output[0].tolist(), payload_bytes=packet.payload_nbytes,
                   wire_bytes=packet.wire_nbytes, codec_roundtrip_ms=codec_ms,
                   equals_monolithic=torch.equal(direct, output))
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path("results/smoke").mkdir(parents=True, exist_ok=True)
    Path("results/smoke/qwen-local.json").write_text(json.dumps(rows, indent=2))
