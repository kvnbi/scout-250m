from __future__ import annotations

import torch


def document_ids(input_ids: torch.Tensor, end_of_text_id: int) -> torch.Tensor:
    if input_ids.dim() != 2:
        raise ValueError(f"expected input_ids of shape (batch, seq), got {tuple(input_ids.shape)}")
    ends = (input_ids == end_of_text_id).long()
    return ends.cumsum(dim=1) - ends


def document_mask(ids: torch.Tensor) -> torch.Tensor:
    if ids.dim() != 2:
        raise ValueError(f"expected document ids of shape (batch, seq), got {tuple(ids.shape)}")
    if ids.is_floating_point() or ids.is_complex() or ids.dtype == torch.bool:
        raise TypeError(f"expected integer document ids, got {ids.dtype}")
    seq = ids.shape[1]
    causal = torch.ones(seq, seq, dtype=torch.bool, device=ids.device).tril()
    return (ids[:, :, None] == ids[:, None, :]) & causal
