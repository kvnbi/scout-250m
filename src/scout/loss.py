from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


class LossTotals(NamedTuple):
    loss_sum: torch.Tensor
    cross_entropy_sum: torch.Tensor
    z_loss_sum: torch.Tensor
    tokens: torch.Tensor


def _sums_and_gradients(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    chunk_size: int,
    z_loss_weight: float,
    need_hidden: bool,
    need_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    accumulator = torch.promote_types(torch.promote_types(hidden.dtype, weight.dtype), torch.float32)
    cross_entropy = hidden.new_zeros((), dtype=accumulator)
    z_loss = hidden.new_zeros((), dtype=accumulator)
    grad_hidden = torch.empty_like(hidden) if need_hidden else None
    grad_weight = torch.zeros(weight.shape, dtype=accumulator, device=weight.device) if need_weight else None
    for start in range(0, targets.numel(), chunk_size):
        chunk = slice(start, start + chunk_size)
        keep = valid[chunk].unsqueeze(-1)
        index = torch.where(keep, targets[chunk].unsqueeze(-1), 0)
        logits = F.linear(hidden[chunk], weight)
        compute = logits.dtype
        logits = logits.to(torch.promote_types(compute, torch.float32))
        log_normaliser = torch.logsumexp(logits, dim=-1, keepdim=True)
        cross_entropy += torch.where(keep, log_normaliser - logits.gather(-1, index), 0.0).sum()
        z_loss += torch.where(keep, log_normaliser.square(), 0.0).sum()
        if need_hidden or need_weight:
            scale = torch.where(keep, 1.0 + 2.0 * z_loss_weight * log_normaliser, 0.0)
            grad = torch.exp(logits - log_normaliser) * scale
            grad.scatter_add_(-1, index, -keep.to(grad.dtype))
            grad = grad.to(compute)
            if need_hidden:
                grad_hidden[chunk] = grad @ weight
            if need_weight:
                grad_weight += grad.T @ hidden[chunk]
    return cross_entropy, z_loss, grad_hidden, grad_weight


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, valid, chunk_size, z_loss_weight):
        need_hidden, need_weight = ctx.needs_input_grad[:2]
        cross_entropy, z_loss, grad_hidden, grad_weight = _sums_and_gradients(
            hidden, weight, targets, valid, chunk_size, z_loss_weight, need_hidden, need_weight
        )
        ctx.save_for_backward(grad_hidden, grad_weight)
        ctx.mark_non_differentiable(cross_entropy, z_loss)
        return cross_entropy + z_loss_weight * z_loss, cross_entropy, z_loss

    @staticmethod
    def backward(ctx, grad_loss, grad_cross_entropy, grad_z_loss):
        grad_hidden, grad_weight = ctx.saved_tensors
        if grad_hidden is not None:
            grad_hidden = (grad_hidden * grad_loss).to(grad_hidden.dtype)
        if grad_weight is not None:
            grad_weight = grad_weight * grad_loss
        return grad_hidden, grad_weight, None, None, None, None


def lm_loss_totals(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 4096,
    ignore_index: int = IGNORE_INDEX,
    z_loss_weight: float = 0.0,
) -> LossTotals:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if z_loss_weight < 0:
        raise ValueError("z_loss_weight must not be negative")
    if hidden.shape[:-1] != targets.shape:
        raise ValueError(f"hidden {tuple(hidden.shape)} and targets {tuple(targets.shape)} do not line up")
    if weight.dim() != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError(f"weight {tuple(weight.shape)} does not match hidden width {hidden.shape[-1]}")
    if targets.is_floating_point() or targets.is_complex() or targets.dtype == torch.bool:
        raise TypeError(f"expected integer targets, got {targets.dtype}")
    hidden = hidden.reshape(-1, hidden.shape[-1])
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
        loss_sum, cross_entropy, z_loss = _LinearCrossEntropy.apply(
            hidden, weight, targets, valid, chunk_size, z_loss_weight
        )
    else:
        cross_entropy, z_loss, _, _ = _sums_and_gradients(
            hidden, weight, targets, valid, chunk_size, z_loss_weight, False, False
        )
        loss_sum = cross_entropy + z_loss_weight * z_loss
    return LossTotals(loss_sum, cross_entropy, z_loss, valid.sum())


def lm_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 4096,
    ignore_index: int = IGNORE_INDEX,
    z_loss_weight: float = 0.0,
) -> torch.Tensor:
    totals = lm_loss_totals(hidden, weight, targets, chunk_size, ignore_index, z_loss_weight)
    return totals.loss_sum / totals.tokens.clamp_min(1)
