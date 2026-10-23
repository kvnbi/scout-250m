from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from scout.retrieval.weights import read_safetensors

IGNORED = re.compile(r"^(pooler\.|cls\.|embeddings\.position_ids$)")
RENAMES = (
    (r"^embeddings\.", ""),
    (r"^encoder\.layer\.(\d+)\.attention\.self\.(query|key|value)\.", r"layers.\1.\2."),
    (r"^encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"layers.\1.attention_output."),
    (r"^encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.", r"layers.\1.attention_norm."),
    (r"^encoder\.layer\.(\d+)\.intermediate\.dense\.", r"layers.\1.intermediate."),
    (r"^encoder\.layer\.(\d+)\.output\.dense\.", r"layers.\1.output."),
    (r"^encoder\.layer\.(\d+)\.output\.LayerNorm\.", r"layers.\1.output_norm."),
    (r"\.gamma$", ".weight"),
    (r"\.beta$", ".bias"),
)


@dataclass(frozen=True)
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12

    def __post_init__(self) -> None:
        sizes = (
            self.vocab_size,
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.intermediate_size,
            self.max_position_embeddings,
            self.type_vocab_size,
        )
        if min(sizes) < 1 or self.layer_norm_eps <= 0:
            raise ValueError("sizes must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be a multiple of num_attention_heads")

    @classmethod
    def from_file(cls, path: Path) -> BertConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("hidden_act", "gelu") != "gelu":
            raise ValueError(f"unsupported activation {raw['hidden_act']!r}")
        if raw.get("position_embedding_type", "absolute") != "absolute":
            raise ValueError(f"unsupported position embeddings {raw['position_embedding_type']!r}")
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{name: value for name, value in raw.items() if name in known})


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.heads = config.num_attention_heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.attention_output = nn.Linear(hidden, hidden)
        self.attention_norm = nn.LayerNorm(hidden, eps=config.layer_norm_eps)
        self.intermediate = nn.Linear(hidden, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, hidden)
        self.output_norm = nn.LayerNorm(hidden, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.heads, hidden // self.heads).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = self.attention_norm(self.attention_output(context.transpose(1, 2).reshape(batch, length, hidden)) + x)
        return self.output_norm(self.output(F.gelu(self.intermediate(x))) + x)


class Bert(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(BertLayer(config) for _ in range(config.num_hidden_layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = input_ids.shape
        if length > self.config.max_position_embeddings:
            raise ValueError(f"{length} tokens exceed the {self.config.max_position_embeddings} positions of the model")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        positions = torch.arange(length, device=input_ids.device)
        x = self.word_embeddings(input_ids) + self.token_type_embeddings(token_type_ids)
        x = self.LayerNorm(x + self.position_embeddings(positions))
        mask = attention_mask.bool()[:, None, None, :]
        for layer in self.layers:
            x = layer(x, mask)
        return x


def convert_key(key: str) -> str | None:
    if key.startswith("bert."):
        key = key[len("bert.") :]
    if IGNORED.match(key):
        return None
    for pattern, replacement in RENAMES:
        key = re.sub(pattern, replacement, key)
    return key


def load_bert(directory: Path, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> Bert:
    directory = Path(directory)
    config = BertConfig.from_file(directory / "config.json")
    state = {}
    for key, tensor in read_safetensors(directory / "model.safetensors").items():
        name = convert_key(key)
        if name is not None:
            state[name] = tensor.to(dtype=dtype, device=device) if tensor.is_floating_point() else tensor.to(device)
    with torch.device("meta"):
        model = Bert(config)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().requires_grad_(False)
