from __future__ import annotations

import torch

from scout.config import ModelConfig


class LayerCache:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, start: int, end: int, mask: torch.Tensor) -> None:
        self.keys = keys
        self.values = values
        self.start = start
        self.end = end
        self.mask = mask

    def store(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if k.dtype != self.keys.dtype or v.dtype != self.values.dtype:
            raise TypeError(f"cache holds {self.keys.dtype}, got keys {k.dtype} and values {v.dtype}")
        self.keys[:, :, self.start : self.end] = k
        self.values[:, :, self.start : self.end] = v
        return self.keys[:, :, : self.end], self.values[:, :, : self.end]


class KVCache:
    def __init__(
        self,
        config: ModelConfig,
        batch: int,
        max_length: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        if batch < 1 or max_length < 1:
            raise ValueError("batch and max_length must be positive")
        shape = (config.n_layers, batch, config.n_kv_heads, max_length, config.head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)
        self.valid = torch.zeros(batch, max_length, dtype=torch.bool, device=device)
        self.batch = batch
        self.max_length = max_length
        self.length = 0
        self._pending: tuple[int, int, torch.Tensor] | None = None

    def begin(self, new_valid: torch.Tensor) -> torch.Tensor:
        if new_valid.dim() != 2 or new_valid.shape[0] != self.batch or new_valid.dtype != torch.bool:
            raise ValueError(f"expected a boolean mask of shape ({self.batch}, new tokens)")
        start = self.length
        end = start + new_valid.shape[1]
        if end > self.max_length:
            raise ValueError(f"cache holds {self.max_length} tokens, {end} requested")
        self.valid[:, start:end] = new_valid
        keys = torch.arange(end, device=self.valid.device)
        queries = torch.arange(start, end, device=self.valid.device)
        causal = keys[None, :] <= queries[:, None]
        itself = keys[None, :] == queries[:, None]
        mask = (self.valid[:, None, None, :end] & causal) | itself
        self._pending = (start, end, mask)
        return (self.valid[:, :end].cumsum(-1) - 1).clamp_min(0)[:, start:end]

    def layer(self, index: int) -> LayerCache:
        if self._pending is None:
            raise RuntimeError("begin must be called before using the cache")
        start, end, mask = self._pending
        return LayerCache(self.keys[index], self.values[index], start, end, mask)

    def commit(self) -> None:
        if self._pending is None:
            raise RuntimeError("begin must be called before commit")
        self.length = self._pending[1]
        self._pending = None
