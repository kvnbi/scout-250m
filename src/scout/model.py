from __future__ import annotations

import math

import torch
from torch import nn

from scout.block import Block
from scout.cache import KVCache
from scout.config import ModelConfig
from scout.layers import RMSNorm
from scout.loss import IGNORE_INDEX, LossTotals, lm_loss, lm_loss_totals
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

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions, h.dtype)
        for index, layer in enumerate(self.layers):
            h = layer(h, cos, sin, None if cache is None else cache.layer(index))
        return self.norm(h)


class Scout(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        self.model = Backbone(self.config)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.config.init_std
        residual_std = std / math.sqrt(2 * self.config.n_layers) if self.config.depth_scaled_init else std
        for name, module in self.named_modules():
            if module is self.lm_head and self.config.tie_embeddings:
                continue
            if isinstance(module, nn.Linear):
                scale = residual_std if name.endswith(("o_proj", "down_proj")) else std
                nn.init.trunc_normal_(module.weight, std=scale, a=-3 * scale, b=3 * scale)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=std, a=-3 * std, b=3 * std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.hidden_states(input_ids, positions))

    def loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        positions: torch.Tensor | None = None,
        chunk_size: int = 4096,
        z_loss_weight: float = 0.0,
    ) -> torch.Tensor:
        hidden = self.hidden_states(input_ids, positions)
        return lm_loss(hidden, self.lm_head.weight, targets, chunk_size, IGNORE_INDEX, z_loss_weight)

    def loss_totals(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        positions: torch.Tensor | None = None,
        chunk_size: int = 4096,
        z_loss_weight: float = 0.0,
    ) -> LossTotals:
        hidden = self.hidden_states(input_ids, positions)
        return lm_loss_totals(hidden, self.lm_head.weight, targets, chunk_size, IGNORE_INDEX, z_loss_weight)

    def hidden_states(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq = self._check_input_ids(input_ids)
        if positions is None:
            positions = torch.arange(seq, device=input_ids.device).unsqueeze(0)
        else:
            self._check_positions(positions, batch, seq)
        return self.model(input_ids, positions)

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        cache: KVCache,
        valid: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        last_only: bool = False,
    ) -> torch.Tensor:
        batch, seq = self._check_input_ids(input_ids)
        if batch != cache.batch:
            raise ValueError(f"cache holds {cache.batch} rows, got {batch}")
        if valid is None:
            valid = torch.ones(batch, seq, dtype=torch.bool, device=input_ids.device)
        counted = cache.begin(valid)
        if positions is None:
            positions = counted
        else:
            self._check_positions(positions, batch, seq)
        hidden = self.model(input_ids, positions, cache)
        cache.commit()
        if last_only:
            hidden = hidden[:, -1:]
        return self.lm_head(hidden)

    @staticmethod
    def _check_input_ids(input_ids: torch.Tensor) -> tuple[int, int]:
        if input_ids.dim() != 2:
            raise ValueError(f"expected input_ids of shape (batch, seq), got {tuple(input_ids.shape)}")
        if input_ids.is_floating_point() or input_ids.is_complex() or input_ids.dtype == torch.bool:
            raise TypeError(f"expected integer token ids, got {input_ids.dtype}")
        return input_ids.shape[0], input_ids.shape[1]

    @staticmethod
    def _check_positions(positions: torch.Tensor, batch: int, seq: int) -> None:
        if positions.dim() != 2 or positions.shape[0] not in (1, batch) or positions.shape[1] != seq:
            raise ValueError(f"expected positions of shape (batch, {seq}), got {tuple(positions.shape)}")
