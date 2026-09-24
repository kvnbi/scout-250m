import math

import pytest
import torch

from scout.config import ModelConfig
from scout.loss import IGNORE_INDEX
from scout.model import Scout

TINY = ModelConfig(
    vocab_size=64,
    reserved_slots=8,
    d_model=64,
    n_layers=2,
    n_query_heads=4,
    n_kv_heads=2,
    head_dim=16,
    ffn_hidden=128,
    context=32,
)


def batch(seed):
    ids = torch.randint(0, 64, (4, 32), generator=torch.Generator().manual_seed(100 + seed))
    ids[:, 0] = torch.tensor([1, 2, 3, 4])
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = IGNORE_INDEX
    return ids, targets


def train(seed, steps, autocast=False):
    torch.manual_seed(seed)
    model = Scout(TINY)
    ids, targets = batch(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    losses = []
    for _ in range(steps):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
            loss = model.loss(ids, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses, ids


def accuracy(model, ids):
    with torch.no_grad():
        predictions = model(ids).argmax(-1)[:, :-1]
    return (predictions == ids[:, 1:]).float().mean().item()


@pytest.mark.parametrize("seed", [0, 1, 4])
def test_overfits_a_tiny_batch(seed):
    model, losses, ids = train(seed, 200)
    assert abs(losses[0] - math.log(64)) < 0.3
    assert losses[-1] < 0.05
    assert accuracy(model, ids) >= 0.98


def test_overfits_under_bfloat16_autocast():
    model, losses, ids = train(0, 200, autocast=True)
    assert losses[-1] < 0.05
    assert accuracy(model, ids) >= 0.98


def test_every_parameter_learns():
    torch.manual_seed(0)
    before = {name: parameter.detach().clone() for name, parameter in Scout(TINY).named_parameters()}
    model, _, _ = train(0, 5)
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter, before[name]), name


def test_training_is_bitwise_reproducible():
    first_model, first_losses, _ = train(3, 20)
    second_model, second_losses, _ = train(3, 20)
    assert first_losses == second_losses
    second_state = second_model.state_dict()
    assert all(torch.equal(value, second_state[key]) for key, value in first_model.state_dict().items())
