from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from scout.retrieval.bert import Bert, load_bert

PREFIXES = {"query": "query: ", "passage": "passage: "}


class Embedder:
    def __init__(
        self,
        directory: Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        max_length: int = 512,
    ) -> None:
        directory = Path(directory)
        self.model: Bert = load_bert(directory, dtype, device)
        if not 3 <= max_length <= self.model.config.max_position_embeddings:
            raise ValueError(f"max_length must be between 3 and {self.model.config.max_position_embeddings}")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.max_length = max_length
        self.tokenizer.enable_truncation(max_length=max_length)
        self.pad = self.tokenizer.token_to_id("[PAD]")
        self.device = torch.device(device)
        self.dimension = self.model.config.hidden_size

    def encode(self, texts: Sequence[str], kind: str = "passage", batch_size: int = 32) -> np.ndarray:
        if kind not in PREFIXES:
            raise ValueError(f"kind must be one of {sorted(PREFIXES)}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        prefix = PREFIXES[kind]
        encodings = self.tokenizer.encode_batch([prefix + text for text in texts])
        order = sorted(range(len(texts)), key=lambda i: len(encodings[i].ids), reverse=True)
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                chosen = order[start : start + batch_size]
                width = max(len(encodings[i].ids) for i in chosen)
                ids = torch.full((len(chosen), width), self.pad, dtype=torch.long)
                mask = torch.zeros((len(chosen), width), dtype=torch.long)
                for row, i in enumerate(chosen):
                    length = len(encodings[i].ids)
                    ids[row, :length] = torch.tensor(encodings[i].ids)
                    mask[row, :length] = 1
                ids, mask = ids.to(self.device), mask.to(self.device)
                hidden = self.model(ids, mask).float()
                weights = mask.unsqueeze(-1).float()
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
                result[chosen] = F.normalize(pooled, p=2, dim=1).cpu().numpy()
        return result
