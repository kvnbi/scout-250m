from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import NamedTuple

import torch
import torch.nn.functional as F

from scout.model import Scout


class Choices(NamedTuple):
    prompt: list[int]
    options: list[list[int]]
    correct: int


@contextmanager
def evaluating(model: Scout):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def evaluate_loss(
    model: Scout, batches: Iterable, chunk_size: int = 4096, autocast: bool = False
) -> dict[str, float | int]:
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    tokens = 0
    with evaluating(model):
        for batch in batches:
            document_ids = None if batch.document_ids is None else batch.document_ids.to(device)
            inputs, targets = batch.input_ids.to(device), batch.targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                totals = model.loss_totals(inputs, targets, chunk_size=chunk_size, document_ids=document_ids)
            total = total + totals.cross_entropy_sum
            tokens += int(totals.tokens)
    if tokens == 0:
        raise ValueError("there is nothing to score")
    return {"loss": total.item() / tokens, "tokens": tokens}


def _rows(model: Scout, items: Sequence[Choices]) -> list[tuple[int, int, list[int], int]]:
    rows = []
    for index, item in enumerate(items):
        if not item.prompt or not item.options or not 0 <= item.correct < len(item.options):
            raise ValueError(f"item {index} needs a prompt, options and a valid correct index")
        for number, option in enumerate(item.options):
            if not option:
                raise ValueError(f"item {index} has an empty option")
            if len(item.prompt) + len(option) > model.config.context:
                raise ValueError(f"item {index} is longer than the context of {model.config.context} tokens")
            rows.append((index, number, item.prompt + option, len(item.prompt)))
    return rows


def option_log_probs(
    model: Scout, items: Sequence[Choices], batch_size: int = 16, autocast: bool = False
) -> list[list[float]]:
    rows = _rows(model, items)
    order = sorted(range(len(rows)), key=lambda k: len(rows[k][2]))
    scores = [[0.0] * len(item.options) for item in items]
    device = next(model.parameters()).device
    with evaluating(model):
        for start in range(0, len(order), batch_size):
            chunk = [rows[k] for k in order[start : start + batch_size]]
            ids = torch.zeros(len(chunk), max(len(row[2]) for row in chunk), dtype=torch.long)
            owner, position, target = [], [], []
            for b, (_, _, sequence, prompt_length) in enumerate(chunk):
                ids[b, : len(sequence)] = torch.tensor(sequence)
                for t in range(prompt_length, len(sequence)):
                    owner.append(b)
                    position.append(t - 1)
                    target.append(sequence[t])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                hidden = model.hidden_states(ids.to(device))
            states = hidden[torch.tensor(owner, device=device), torch.tensor(position, device=device)]
            logits = F.linear(states, model.lm_head.weight)
            logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
            picked = logits.log_softmax(-1).gather(-1, torch.tensor(target, device=device)[:, None]).squeeze(-1)
            sums = torch.zeros(len(chunk), dtype=picked.dtype, device=device)
            sums.index_add_(0, torch.tensor(owner, device=device), picked)
            for (index, number, _, _), value in zip(chunk, sums.tolist()):
                scores[index][number] = value
    return scores


def evaluate_choices(
    model: Scout, items: Sequence[Choices], batch_size: int = 16, normalize: bool = False, autocast: bool = False
) -> dict[str, float | int]:
    if not items:
        raise ValueError("there are no items to score")
    scores = option_log_probs(model, items, batch_size, autocast)
    probability = accuracy = answer = 0.0
    for item, row in zip(items, scores):
        values = [s / len(o) if normalize else s for s, o in zip(row, item.options)]
        top = max(values)
        weights = [math.exp(v - top) for v in values]
        probability += weights[item.correct] / sum(weights)
        accuracy += values.index(top) == item.correct
        answer += row[item.correct] / len(item.options[item.correct])
    count = len(items)
    return {
        "correct_probability": probability / count,
        "accuracy": accuracy / count,
        "answer_log_prob": answer / count,
        "items": count,
    }
