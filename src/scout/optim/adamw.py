from __future__ import annotations

from collections.abc import Iterable

import torch


class CautiousAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious: bool = True,
    ) -> None:
        if lr < 0 or weight_decay < 0 or eps <= 0:
            raise ValueError("lr and weight_decay must not be negative and eps must be positive")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas must be in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, cautious=cautious))

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                grad, exp_avg, exp_avg_sq = p.grad, state["exp_avg"], state["exp_avg_sq"]
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denominator = (exp_avg_sq / (1 - beta2 ** state["step"])).sqrt() + group["eps"]
                update = exp_avg / (1 - beta1 ** state["step"]) / denominator
                if group["cautious"]:
                    keep = (update * grad > 0).to(update.dtype)
                    update = update * keep * (p.numel() / (keep.sum() + 1))
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
