"""Exact full-prompt handoff semantics for transformers 4.56 DynamicCache.

The first generated token is selected from the sender's last prompt logits.
Only subsequent forwards attend to the received KV. New decode KV is produced
in the model's dtype (BF16 for BF16 models), never requantized by this helper.
"""
from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import torch

from .kv import _layers


@dataclasses.dataclass(frozen=True)
class PrefillState:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    next_token_logits: torch.Tensor
    cache: Any

    def replace_cache(self, cache: Any) -> "PrefillState":
        return dataclasses.replace(self, cache=cache)


def clone_cache(cache):
    """Fresh DynamicCache for every continuation comparison."""
    from transformers import DynamicCache
    fresh = DynamicCache()
    for i, (k, v) in enumerate(_layers(cache)):
        fresh.update(k.clone(), v.clone(), i)
    return fresh


@torch.inference_mode()
def prefill(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> PrefillState:
    if input_ids.ndim != 2 or input_ids.shape[1] < 1:
        raise ValueError("prefill requires nonempty [batch, sequence] input_ids")
    mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
    if mask.shape != input_ids.shape or not bool((mask[:, -1] == 1).all()):
        raise ValueError("use unpadded or left-padded prompts with matching attention masks")
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    # Avoid allocating [sequence, vocabulary] logits for a 16K-token prompt.
    # Cache construction still processes the entire prompt.
    logits_kw = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    out = model(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                cache_position=torch.arange(input_ids.shape[1], device=input_ids.device), use_cache=True, **logits_kw)
    return PrefillState(input_ids.clone(), mask.clone(), out.logits[:, -1].clone(), out.past_key_values)


@torch.inference_mode()
def continue_from_prefill(model, state: PrefillState, max_new_tokens: int,
                          eos_token_id: int | list[int] | None = None,
                          pad_token_id: int | None = None) -> torch.Tensor:
    """Greedy generated IDs only, without modifying ``state`` or its KV cache.

    For batched EOS behavior pass model generation_config's EOS and PAD IDs.
    With eos_token_id=None decode exactly max_new_tokens, as in the wire bench.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if max_new_tokens == 0:
        return state.input_ids[:, :0].clone()
    device = next(model.parameters()).device
    ids, mask = state.input_ids.to(device), state.attention_mask.to(device)
    logits = state.next_token_logits.to(device)
    cache = clone_cache(state.cache)
    # Receiver placement can differ from the sender's device.
    for layer in cache.layers:
        layer.keys, layer.values = layer.keys.to(device), layer.values.to(device)
    eos = [] if eos_token_id is None else ([eos_token_id] if isinstance(eos_token_id, int) else eos_token_id)
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=device)
    generated = []
    for step in range(max_new_tokens):
        token = logits.argmax(-1)
        if eos:
            pad = eos[0] if pad_token_id is None else pad_token_id
            token = torch.where(finished, pad, token)
        generated.append(token)
        if eos:
            finished |= (token[:, None] == torch.tensor(eos, device=device)).any(-1)
            if bool(finished.all()):
                break
        if step + 1 == max_new_tokens:
            break
        # token is at absolute cache position prompt_len + step. Position IDs
        # count real tokens; attention_mask still covers left-padded columns.
        positions = mask.long().sum(-1, keepdim=True)
        mask = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=-1)
        out = model(input_ids=token[:, None], attention_mask=mask, position_ids=positions,
                    cache_position=torch.tensor([ids.shape[1] + step], device=device),
                    past_key_values=cache, use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1]
    return torch.stack(generated, dim=1)
