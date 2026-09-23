from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"expected a floating point tensor, got {x.dtype}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")
        h = x.to(torch.promote_types(x.dtype, torch.float32))
        h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (h * self.weight.to(h.dtype)).to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}"
