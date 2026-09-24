from __future__ import annotations

import torch
from torch import nn

from scout.block import Block
from scout.config import ModelConfig
from scout.layers import RMSNorm
from scout.rotary import RotaryEmbedding


class Backbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            Block(
                config.d_model,
                config.n_query_heads,
                config.n_kv_heads,
                config.head_dim,
                config.ffn_hidden,
                config.qk_norm,
                config.norm_eps,
            )
            for _ in range(config.n_layers)
        )
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions, h.dtype)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.norm(h)


class Scout(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        self.model = Backbone(self.config)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"expected input_ids of shape (batch, seq), got {tuple(input_ids.shape)}")
        if input_ids.is_floating_point() or input_ids.is_complex() or input_ids.dtype == torch.bool:
            raise TypeError(f"expected integer token ids, got {input_ids.dtype}")
        batch, seq = input_ids.shape
        if positions is None:
            positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        elif positions.dim() != 2 or positions.shape[0] not in (1, batch) or positions.shape[1] != seq:
            raise ValueError(f"expected positions of shape (batch, {seq}), got {tuple(positions.shape)}")
        return self.lm_head(self.model(input_ids, positions))
