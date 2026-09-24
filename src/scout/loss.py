from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


class LossTotals(NamedTuple):
    cross_entropy_sum: torch.Tensor
    z_loss_sum: torch.Tensor
    tokens: torch.Tensor


def _chunk_totals(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden, weight)
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    log_normaliser = torch.logsumexp(logits, dim=-1)
    valid = targets != ignore_index
    target_logits = logits.gather(-1, torch.where(valid, targets, 0).unsqueeze(-1)).squeeze(-1)
    cross_entropy = torch.where(valid, log_normaliser - target_logits, 0.0).sum()
    z_loss = torch.where(valid, log_normaliser.square(), 0.0).sum()
    return cross_entropy, z_loss


def lm_loss_totals(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 4096,
    ignore_index: int = IGNORE_INDEX,
) -> LossTotals:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if hidden.shape[:-1] != targets.shape:
        raise ValueError(f"hidden {tuple(hidden.shape)} and targets {tuple(targets.shape)} do not line up")
    if weight.dim() != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError(f"weight {tuple(weight.shape)} does not match hidden width {hidden.shape[-1]}")
    if targets.is_floating_point() or targets.is_complex() or targets.dtype == torch.bool:
        raise TypeError(f"expected integer targets, got {targets.dtype}")
    valid = targets != ignore_index
    if (valid & ((targets < 0) | (targets >= weight.shape[0]))).any():
        raise ValueError(f"targets must be {ignore_index} or in [0, {weight.shape[0]})")
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    accumulator = torch.promote_types(torch.promote_types(hidden.dtype, weight.dtype), torch.float32)
    cross_entropy = hidden.new_zeros((), dtype=accumulator)
    z_loss = hidden.new_zeros((), dtype=accumulator)
    for start in range(0, targets.numel(), chunk_size):
        end = start + chunk_size
        chunk_ce, chunk_z = checkpoint(
            _chunk_totals,
            hidden[start:end],
            weight,
            targets[start:end],
            ignore_index,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        cross_entropy = cross_entropy + chunk_ce
        z_loss = z_loss + chunk_z
    return LossTotals(cross_entropy, z_loss, valid.sum())


def lm_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 4096,
    ignore_index: int = IGNORE_INDEX,
    z_loss_weight: float = 0.0,
) -> torch.Tensor:
    if z_loss_weight < 0:
        raise ValueError("z_loss_weight must not be negative")
    totals = lm_loss_totals(hidden, weight, targets, chunk_size, ignore_index)
    return (totals.cross_entropy_sum + z_loss_weight * totals.z_loss_sum) / totals.tokens.clamp_min(1)
