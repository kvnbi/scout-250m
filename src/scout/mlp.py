from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        if d_model < 1 or hidden < 1:
            raise ValueError("d_model and hidden must be positive")
        self.d_model = d_model
        self.hidden = hidden
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, got shape {tuple(x.shape)}")
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, hidden={self.hidden}"
