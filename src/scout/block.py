from __future__ import annotations

import torch
from torch import nn

from scout.attention import Attention
from scout.layers import RMSNorm
from scout.mlp import SwiGLU


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        head_dim: int,
        ffn_hidden: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(d_model, eps)
        self.self_attn = Attention(d_model, n_query_heads, n_kv_heads, head_dim, qk_norm, eps)
        self.post_attention_layernorm = RMSNorm(d_model, eps)
        self.mlp = SwiGLU(d_model, ffn_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))
