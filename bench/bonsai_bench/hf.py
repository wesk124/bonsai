"""Small Hugging Face helpers shared by the eval scripts."""

from __future__ import annotations

import torch


def load_model(name: str, dtype: str = "bfloat16", device: str | None = None):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    # `dtype=` replaced `torch_dtype=` in transformers 4.56; older versions would
    # silently ignore `dtype` and load fp32.
    kw = {"dtype": getattr(torch, dtype)} if major_minor >= (4, 56) else {"torch_dtype": getattr(torch, dtype)}
    model = AutoModelForCausalLM.from_pretrained(name, **kw).to(device).eval()
    return model, AutoTokenizer.from_pretrained(name)


def head_dim(model) -> int:
    c = model.config
    return getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
