from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path

import torch

FORMAT = 1


def random_state() -> dict[str, object]:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": cuda}


def restore_random_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizers: Sequence[torch.optim.Optimizer],
    step: int,
    extra: dict[str, object] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": FORMAT,
        "step": step,
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "random": random_state(),
        "extra": extra or {},
    }
    partial = path.with_name(path.name + ".tmp")
    try:
        torch.save(checkpoint, partial)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path, model: torch.nn.Module, optimizers: Sequence[torch.optim.Optimizer]
) -> tuple[int, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format"] != FORMAT:
        raise ValueError(f"unknown checkpoint format {checkpoint['format']}")
    if len(checkpoint["optimizers"]) != len(optimizers):
        raise ValueError(f"checkpoint holds {len(checkpoint['optimizers'])} optimizers, got {len(optimizers)}")
    model.load_state_dict(checkpoint["model"])
    for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
        optimizer.load_state_dict(state)
    restore_random_state(checkpoint["random"])
    return checkpoint["step"], checkpoint["extra"]
