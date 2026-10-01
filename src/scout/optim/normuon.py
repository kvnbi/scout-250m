from __future__ import annotations

from collections.abc import Iterable

import torch

COEFFICIENTS = (3.4445, -4.7750, 2.0315)
UPDATE_RMS = 0.2


def orthogonalize(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if matrix.dim() != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(matrix.shape)}")
    a, b, c = COEFFICIENTS
    x = matrix.to(torch.promote_types(matrix.dtype, torch.float32))
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if tall else x


class NorMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0.95,
        beta2: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        steps: int = 5,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0 or weight_decay < 0 or eps <= 0:
            raise ValueError("lr and weight_decay must not be negative and eps must be positive")
        if not 0 <= momentum < 1 or not 0 <= beta2 < 1:
            raise ValueError("momentum and beta2 must be in [0, 1)")
        if steps < 1:
            raise ValueError("steps must be positive")
        defaults = dict(
            lr=lr, momentum=momentum, beta2=beta2, weight_decay=weight_decay, nesterov=nesterov, steps=steps, eps=eps
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError(f"NorMuon only takes matrices, got shape {tuple(p.shape)}")

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    dtype = torch.promote_types(p.dtype, torch.float32)
                    state["second_moment"] = torch.zeros(p.shape[0], 1, dtype=dtype, device=p.device)
                buffer = state["momentum_buffer"]
                buffer.lerp_(p.grad, 1 - group["momentum"])
                update = p.grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                update = orthogonalize(update, group["steps"])
                second_moment = state["second_moment"]
                second_moment.lerp_(update.square().mean(dim=-1, keepdim=True), 1 - group["beta2"])
                update = update / (second_moment.sqrt() + group["eps"])
                update = update * (UPDATE_RMS * p.numel() ** 0.5 / (update.norm() + group["eps"]))
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"])
