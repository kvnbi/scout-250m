from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from scout.layers import RMSNorm
from scout.rotary import apply_rotary


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        head_dim: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(d_model, n_query_heads, n_kv_heads, head_dim) < 1:
            raise ValueError("d_model, head counts and head_dim must be positive")
        if n_query_heads % n_kv_heads:
            raise ValueError("n_query_heads must be a multiple of n_kv_heads")
        if head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        self.d_model = d_model
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(d_model, n_query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_query_heads * head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(head_dim, eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim, eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected input of shape (batch, seq, {self.d_model}), got {tuple(x.shape)}")
        batch, seq, _ = x.shape
        for name, table in (("cos", cos), ("sin", sin)):
            if table.dim() != 3 or table.shape[0] not in (1, batch) or table.shape[1:] != (seq, self.head_dim):
                raise ValueError(f"expected {name} of shape (batch, {seq}, {self.head_dim}), got {tuple(table.shape)}")
            if table.dtype != x.dtype:
                raise TypeError(f"expected {name} in {x.dtype}, got {table.dtype}")
        q = self.q_norm(self.q_proj(x).view(batch, seq, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(batch, seq, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.scaling,
            enable_gqa=self.n_query_heads != self.n_kv_heads,
        )
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq, -1))

    def extra_repr(self) -> str:
        return f"query_heads={self.n_query_heads}, kv_heads={self.n_kv_heads}, head_dim={self.head_dim}"
