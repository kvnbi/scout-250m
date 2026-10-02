from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch

from scout.checkpoint import save_checkpoint
from scout.loss import IGNORE_INDEX
from scout.model import Scout
from scout.optim.schedule import set_learning_rates, wsd_factor


class Batch(NamedTuple):
    input_ids: torch.Tensor
    targets: torch.Tensor
    document_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class TrainSettings:
    total_steps: int
    warmup_steps: int
    decay_steps: int
    accumulate: int = 1
    max_grad_norm: float | None = 1.0
    autocast: bool = False
    z_loss_weight: float = 0.0
    chunk_size: int = 4096

    def __post_init__(self) -> None:
        if self.total_steps < 1 or self.accumulate < 1 or self.chunk_size < 1:
            raise ValueError("total_steps, accumulate and chunk_size must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.z_loss_weight < 0:
            raise ValueError("z_loss_weight must not be negative")
        wsd_factor(0, self.total_steps, self.warmup_steps, self.decay_steps)


def train_step(
    model: Scout, optimizers: Sequence[torch.optim.Optimizer], micro_batches: Sequence[Batch], settings: TrainSettings
) -> dict[str, torch.Tensor | int]:
    tokens = sum(int((batch.targets != IGNORE_INDEX).sum()) for batch in micro_batches)
    if tokens == 0:
        raise ValueError("a step needs at least one scored token")
    device = next(model.parameters()).device
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    cross_entropy = torch.zeros((), device=device)
    for batch in micro_batches:
        document_ids = None if batch.document_ids is None else batch.document_ids.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=settings.autocast):
            totals = model.loss_totals(
                batch.input_ids.to(device),
                batch.targets.to(device),
                chunk_size=settings.chunk_size,
                z_loss_weight=settings.z_loss_weight,
                document_ids=document_ids,
            )
        (totals.loss_sum / tokens).backward()
        cross_entropy = cross_entropy + totals.cross_entropy_sum.detach()
    grad_norm = torch.zeros((), device=device)
    if settings.max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings.max_grad_norm, error_if_nonfinite=True
        )
    for optimizer in optimizers:
        optimizer.step()
    return {"loss": cross_entropy / tokens, "grad_norm": grad_norm, "tokens": tokens}


def train(
    model: Scout,
    optimizers: Sequence[torch.optim.Optimizer],
    batches: Iterator[Batch],
    settings: TrainSettings,
    start_step: int = 0,
    stop_step: int | None = None,
    on_step: Callable[[int, dict[str, float | int | list[float]]], None] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
) -> int:
    stop = settings.total_steps if stop_step is None else stop_step
    if not 0 <= start_step <= stop <= settings.total_steps:
        raise ValueError(f"need 0 <= start_step <= stop_step <= {settings.total_steps}")
    model.train()
    for step in range(start_step, stop):
        factor = wsd_factor(step, settings.total_steps, settings.warmup_steps, settings.decay_steps)
        set_learning_rates(optimizers, factor)
        micro_batches = [next(batches) for _ in range(settings.accumulate)]
        result = train_step(model, optimizers, micro_batches, settings)
        done = step + 1
        if on_step is not None:
            on_step(
                done,
                {
                    "loss": result["loss"].item(),
                    "grad_norm": result["grad_norm"].item(),
                    "tokens": result["tokens"],
                    "lrs": [optimizer.param_groups[0]["lr"] for optimizer in optimizers],
                },
            )
        if checkpoint_path is not None and checkpoint_every and (done % checkpoint_every == 0 or done == stop):
            save_checkpoint(checkpoint_path, model, optimizers, done)
    return stop
