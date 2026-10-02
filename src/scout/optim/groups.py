from __future__ import annotations

import torch

from scout.optim.adamw import CautiousAdamW
from scout.optim.normuon import NorMuon


def split_parameters(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    matrices, others = [], []
    for name, parameter in model.named_parameters():
        hidden = name.startswith("model.layers.") and parameter.dim() == 2
        (matrices if hidden else others).append(parameter)
    return matrices, others


def build_optimizers(
    model: torch.nn.Module,
    matrix_lr: float,
    other_lr: float,
    weight_decay: float = 0.1,
    cautious: bool = True,
) -> list[torch.optim.Optimizer]:
    matrices, others = split_parameters(model)
    return [
        NorMuon(matrices, lr=matrix_lr, weight_decay=weight_decay),
        CautiousAdamW(others, lr=other_lr, weight_decay=0.0, cautious=cautious),
    ]
