import random

import pytest
import torch

from scout.checkpoint import load_checkpoint, save_checkpoint
from scout.config import ModelConfig
from scout.loss import IGNORE_INDEX
from scout.masking import document_ids
from scout.model import Scout
from scout.optim.groups import build_optimizers
from scout.optim.schedule import set_learning_rates, wsd_factor

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
EOT = TINY.reserved_token_ids.start
TOTAL, WARMUP, DECAY = 12, 4, 4


def batch():
    ids = torch.randint(0, 50, (4, 24))
    for row in range(4):
        ids[row, random.randrange(4, 20)] = EOT
    targets = torch.roll(ids, -1, 1)
    targets[ids == EOT] = IGNORE_INDEX
    targets[:, -1] = IGNORE_INDEX
    return ids, targets


def build(seed):
    torch.manual_seed(seed)
    model = Scout(TINY)
    return model, build_optimizers(model, matrix_lr=0.02, other_lr=0.01, weight_decay=0.1)


def train(model, optimizers, start, stop, autocast=False, accumulate=1, skip_schedule=False):
    losses = []
    for step in range(start, stop):
        if not skip_schedule:
            set_learning_rates(optimizers, wsd_factor(step, TOTAL, WARMUP, DECAY))
        for optimizer in optimizers:
            optimizer.zero_grad()
        total = 0.0
        for _ in range(accumulate):
            ids, targets = batch()
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=autocast):
                loss = model.loss(ids, targets, document_ids=document_ids(ids, EOT)) / accumulate
            loss.backward()
            total += loss.item()
        for optimizer in optimizers:
            optimizer.step()
        losses.append(total)
    return losses


def seed_streams(value):
    random.seed(value)
    torch.manual_seed(value)


def snapshot(model, optimizers):
    state = {f"model.{k}": v.clone() for k, v in model.state_dict().items()}
    for index, optimizer in enumerate(optimizers):
        saved = optimizer.state_dict()
        for key, value in saved["state"].items():
            for name, tensor in value.items():
                state[f"optimizer{index}.{key}.{name}"] = torch.as_tensor(tensor).clone()
        state[f"optimizer{index}.lrs"] = torch.tensor([group["lr"] for group in saved["param_groups"]])
    return state


def identical(first, second):
    return first.keys() == second.keys() and all(torch.equal(first[key], second[key]) for key in first)


def straight(autocast=False, accumulate=1):
    seed_streams(7)
    model, optimizers = build(0)
    losses = train(model, optimizers, 0, TOTAL, autocast, accumulate)
    return losses, snapshot(model, optimizers)


def interrupted(stop_at, path, autocast=False, accumulate=1):
    seed_streams(7)
    model, optimizers = build(0)
    first = train(model, optimizers, 0, stop_at, autocast, accumulate)
    save_checkpoint(path, model, optimizers, stop_at)
    seed_streams(12345)
    resumed, resumed_optimizers = build(99)
    step, _ = load_checkpoint(path, resumed, resumed_optimizers)
    second = train(resumed, resumed_optimizers, step, TOTAL, autocast, accumulate)
    return first + second, snapshot(resumed, resumed_optimizers), step


@pytest.mark.parametrize("stop_at", [1, 3, 4, 8, 11])
def test_resuming_at_any_point_matches_the_straight_run_bit_for_bit(tmp_path, stop_at):
    losses, state = straight()
    resumed_losses, resumed_state, step = interrupted(stop_at, tmp_path / "c.pt")
    assert step == stop_at
    assert resumed_losses == losses
    assert identical(state, resumed_state)


@pytest.mark.parametrize("autocast, accumulate", [(True, 1), (False, 3), (True, 2)])
def test_it_holds_under_bfloat16_autocast_and_gradient_accumulation(tmp_path, autocast, accumulate):
    losses, state = straight(autocast, accumulate)
    resumed_losses, resumed_state, _ = interrupted(6, tmp_path / "c.pt", autocast, accumulate)
    assert resumed_losses == losses
    assert identical(state, resumed_state)


def test_two_interruptions_in_a_row_still_match(tmp_path):
    losses, state = straight()
    seed_streams(7)
    model, optimizers = build(0)
    seen = train(model, optimizers, 0, 3)
    for number, (stop, path) in enumerate(((3, tmp_path / "a.pt"), (9, tmp_path / "b.pt"))):
        if number:
            seen += train(model, optimizers, 3, stop)
        save_checkpoint(path, model, optimizers, stop)
        seed_streams(1000 + number)
        model, optimizers = build(50 + number)
        load_checkpoint(path, model, optimizers)
    seen += train(model, optimizers, 9, TOTAL)
    assert seen == losses
    assert identical(state, snapshot(model, optimizers))


def test_the_comparison_can_fail_two_runs_differ_when_the_data_differs():
    first, _ = straight()
    seed_streams(8)
    model, optimizers = build(0)
    assert train(model, optimizers, 0, TOTAL) != first


def test_two_straight_runs_are_identical():
    first, first_state = straight()
    second, second_state = straight()
    assert first == second and identical(first_state, second_state)


def test_without_the_saved_random_state_the_resume_would_drift(tmp_path, monkeypatch):
    losses, state = straight()
    import scout.checkpoint as checkpoint

    monkeypatch.setattr(checkpoint, "restore_random_state", lambda saved: None)
    resumed_losses, resumed_state, _ = interrupted(6, tmp_path / "c.pt")
    assert resumed_losses != losses
    assert not identical(state, resumed_state)


def test_resuming_at_the_wrong_step_changes_the_learning_rates_and_the_result(tmp_path):
    seed_streams(7)
    model, optimizers = build(0)
    train(model, optimizers, 0, 6)
    save_checkpoint(tmp_path / "c.pt", model, optimizers, 6)
    _, state = straight()
    seed_streams(0)
    resumed, resumed_optimizers = build(99)
    load_checkpoint(tmp_path / "c.pt", resumed, resumed_optimizers)
    train(resumed, resumed_optimizers, 2, TOTAL)
    assert not identical(state, snapshot(resumed, resumed_optimizers))
