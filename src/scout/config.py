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

    def __post_init__(self) -> None:
        if self.n_query_heads % self.n_kv_heads:
            raise ValueError("n_query_heads must be a multiple of n_kv_heads")
        if self.n_query_heads * self.head_dim != self.d_model:
            raise ValueError("n_query_heads * head_dim must equal d_model")
        if self.reserved_slots >= self.vocab_size:
            raise ValueError("reserved_slots must be smaller than vocab_size")

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
