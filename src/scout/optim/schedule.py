from __future__ import annotations

from collections.abc import Iterable

import torch


def wsd_factor(step: int, total: int, warmup: int, decay: int, final: float = 0.0) -> float:
    if total < 1 or warmup < 0 or decay < 0 or warmup + decay > total:
        raise ValueError("need total >= 1, warmup >= 0, decay >= 0 and warmup + decay <= total")
    if not 0 <= final <= 1:
        raise ValueError("final must be between 0 and 1")
    if not 0 <= step < total:
        raise ValueError(f"step must be in [0, {total}), got {step}")
    if step < warmup:
        return (step + 1) / warmup
    start = total - decay
    if step < start:
        return 1.0
    return final + (1 - final) * (1 - ((step - start) / decay) ** 0.5)


def set_learning_rates(optimizers: Iterable[torch.optim.Optimizer], factor: float) -> None:
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group.setdefault("base_lr", group["lr"]) * factor
