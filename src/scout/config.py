from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    reserved_slots: int = 32
    d_model: int = 896
    n_layers: int = 26
    n_query_heads: int = 14
    n_kv_heads: int = 2
    head_dim: int = 64
    ffn_hidden: int = 2432
    context: int = 2048
    tie_embeddings: bool = True
    qk_norm: bool = True
    norm_eps: float = 1e-6
    rope_theta: float = 100000.0
    init_std: float = 0.02
    depth_scaled_init: bool = True

    def __post_init__(self) -> None:
        sizes = (
            self.vocab_size,
            self.d_model,
            self.n_layers,
            self.n_query_heads,
            self.n_kv_heads,
            self.head_dim,
            self.ffn_hidden,
            self.context,
        )
        if min(sizes) < 1:
            raise ValueError("sizes and counts must be positive")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.rope_theta <= 1:
            raise ValueError("rope_theta must be greater than 1")
        if self.init_std <= 0:
            raise ValueError("init_std must be positive")
        if self.n_query_heads % self.n_kv_heads:
            raise ValueError("n_query_heads must be a multiple of n_kv_heads")
        if self.n_query_heads * self.head_dim != self.d_model:
            raise ValueError("n_query_heads * head_dim must equal d_model")
        if not 0 <= self.reserved_slots < self.vocab_size:
            raise ValueError("reserved_slots must be at least 0 and smaller than vocab_size")

    @property
    def reserved_token_ids(self) -> range:
        return range(self.vocab_size - self.reserved_slots, self.vocab_size)

    @property
    def embedding_parameters(self) -> int:
        tables = 1 if self.tie_embeddings else 2
        return tables * self.vocab_size * self.d_model

    @property
    def layer_parameters(self) -> int:
        d = self.d_model
        kv = self.n_kv_heads * self.head_dim
        attention = 2 * d * d + 2 * d * kv
        if self.qk_norm:
            attention += 2 * self.head_dim
        ffn = 3 * d * self.ffn_hidden
        return attention + ffn + 2 * d

    @property
    def parameter_count(self) -> int:
        return self.embedding_parameters + self.n_layers * self.layer_parameters + self.d_model

    @property
    def non_embedding_parameters(self) -> int:
        return self.parameter_count - self.embedding_parameters

    def kv_cache_bytes_per_token(self, bytes_per_value: int = 2) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * bytes_per_value

    def qwen3_config(self) -> dict:
        if not self.qk_norm:
            raise ValueError("the Qwen3 format requires qk_norm")
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.d_model,
            "intermediate_size": self.ffn_hidden,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_query_heads,
            "num_key_value_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "hidden_act": "silu",
            "max_position_embeddings": self.context,
            "rms_norm_eps": self.norm_eps,
            "rope_theta": self.rope_theta,
            "initializer_range": self.init_std,
            "tie_word_embeddings": self.tie_embeddings,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "use_sliding_window": False,
        }
