from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim < 2 or head_dim % 2:
            raise ValueError("head_dim must be a positive even number")
        if base <= 1:
            raise ValueError("base must be greater than 1")
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("inv_freq", self._inverse_frequencies(), persistent=False)

    def _inverse_frequencies(self) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim))

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.inv_freq = self._inverse_frequencies().to(self.inv_freq.device)
        return self

    @torch.no_grad()
    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() != 2:
            raise ValueError("positions must have shape (batch, seq)")
        with torch.autocast(device_type=positions.device.type, enabled=False):
            freqs = positions.float()[:, :, None] * self.inv_freq.float()[None, None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype), sin.to(dtype)

    def extra_repr(self) -> str:
        return f"{self.head_dim}, base={self.base}"


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return x * cos + rotate_half(x) * sin
