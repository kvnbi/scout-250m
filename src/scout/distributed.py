from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist

BUCKET_BYTES = 64 << 20


def world_size(group: dist.ProcessGroup | None = None) -> int:
    return dist.get_world_size(group) if dist.is_available() and dist.is_initialized() else 1


def rank(group: dist.ProcessGroup | None = None) -> int:
    return dist.get_rank(group) if dist.is_available() and dist.is_initialized() else 0


def broadcast_parameters(model: torch.nn.Module, group: dist.ProcessGroup | None = None) -> None:
    if world_size(group) == 1:
        return
    source = dist.get_global_rank(group, 0) if group is not None else 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.data, src=source, group=group)


def reduce_gradients(
    parameters: Iterable[torch.nn.Parameter], group: dist.ProcessGroup | None = None, bucket_bytes: int = BUCKET_BYTES
) -> None:
    parameters = [p for p in parameters if p.requires_grad]
    world = world_size(group)
    if world == 1 or not parameters:
        return
    if len({p.dtype for p in parameters}) != 1:
        raise ValueError("every parameter must share one dtype")
    for p in parameters:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    limit = max(1, bucket_bytes // parameters[0].element_size())
    bucket: list[torch.nn.Parameter] = []
    size = 0
    for p in parameters + [None]:
        if p is not None and (not bucket or size + p.numel() <= limit):
            bucket.append(p)
            size += p.numel()
            continue
        _sum_in_rank_order(bucket, world, group)
        bucket, size = ([p], p.numel()) if p is not None else ([], 0)


def _sum_in_rank_order(bucket: list[torch.nn.Parameter], world: int, group: dist.ProcessGroup | None) -> None:
    flat = torch.cat([p.grad.reshape(-1) for p in bucket])
    gathered = [torch.empty_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat, group=group)
    total = gathered[0]
    for other in gathered[1:]:
        total = total + other
    offset = 0
    for p in bucket:
        p.grad.copy_(total[offset : offset + p.numel()].view_as(p))
        offset += p.numel()
