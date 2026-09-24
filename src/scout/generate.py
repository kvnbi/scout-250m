from __future__ import annotations

from typing import Callable

import torch

from scout.cache import KVCache
from scout.model import Scout

LogitsProcessor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def generate(
    model: Scout,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    stop_token: int | None = None,
    logits_processor: LogitsProcessor | None = None,
    cache_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if prompt.dim() != 2 or prompt.shape[1] < 1:
        raise ValueError(f"expected a prompt of shape (batch, seq), got {tuple(prompt.shape)}")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must not be negative")
    batch, length = prompt.shape
    dtype = cache_dtype if cache_dtype is not None else model.lm_head.weight.dtype
    cache = KVCache(model.config, batch, length + max_new_tokens, dtype, prompt.device)
    logits = model.forward_cached(prompt, cache, last_only=True)[:, -1]
    generated = prompt.new_empty((batch, 0))
    finished = torch.zeros(batch, dtype=torch.bool, device=prompt.device)
    for step in range(max_new_tokens):
        if logits_processor is not None:
            logits = logits_processor(logits, generated)
        if temperature == 0:
            chosen = logits.argmax(dim=-1)
        else:
            probabilities = torch.softmax(logits.float() / temperature, dim=-1)
            chosen = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        if stop_token is not None:
            chosen = torch.where(finished, stop_token, chosen)
            finished = finished | (chosen == stop_token)
        generated = torch.cat((generated, chosen[:, None]), dim=1)
        if step == max_new_tokens - 1 or bool(finished.all()):
            break
        logits = model.forward_cached(chosen[:, None], cache, last_only=True)[:, -1]
    return generated
