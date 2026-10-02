import math

import pytest
import torch
import torch.nn.functional as F

from scout.checkpoint import load_checkpoint
from scout.config import ModelConfig
from scout.loss import IGNORE_INDEX
from scout.masking import document_ids
from scout.model import Scout
from scout.optim.groups import build_optimizers
from scout.optim.schedule import wsd_factor
from scout.train import Batch, TrainSettings, train, train_step

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


def build(dtype=torch.float32, seed=0, weight_decay=0.0, lr=0.01):
    torch.manual_seed(seed)
    model = Scout(TINY).to(dtype)
    return model, build_optimizers(model, lr, lr, weight_decay)


def make_batch(rows, seed, ignore=()):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 50, (rows, 16), generator=generator)
    ids[:, 7] = EOT
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = IGNORE_INDEX
    targets[:, 7] = IGNORE_INDEX
    for row, column in ignore:
        targets[row, column] = IGNORE_INDEX
    return Batch(ids, targets, document_ids(ids, EOT))


def repeat(batch):
    while True:
        yield batch


def settings(**kwargs):
    defaults = dict(total_steps=10, warmup_steps=2, decay_steps=3)
    return TrainSettings(**{**defaults, **kwargs})


def grads(model):
    return {name: p.grad.clone() for name, p in model.named_parameters()}


def test_gradient_checkpointing_changes_nothing_but_memory():
    batch = make_batch(2, 1)
    results = []
    for enabled in (False, True):
        model, _ = build(torch.float64)
        model.set_gradient_checkpointing(enabled)
        loss = model.loss(batch.input_ids, batch.targets, document_ids=batch.document_ids)
        loss.backward()
        results.append((loss.detach(), grads(model)))
    assert torch.equal(results[0][0], results[1][0])
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=1e-12, atol=1e-14)


def test_gradient_checkpointing_stores_far_fewer_activations():
    batch = make_batch(2, 1)
    stored = {}
    for enabled in (False, True):
        model, _ = build()
        model.set_gradient_checkpointing(enabled)
        count = []

        def pack(tensor):
            count.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            model.hidden_states(batch.input_ids, document_ids=batch.document_ids)
        stored[enabled] = sum(count)
    assert stored[True] < stored[False] / 2


def test_gradient_checkpointing_leaves_cached_decoding_alone():
    model, _ = build(torch.float64)
    ids = make_batch(1, 3).input_ids[:, :8]
    expected = model(ids)
    model.set_gradient_checkpointing(True)
    from scout.cache import KVCache

    cache = KVCache(model.config, 1, 8, torch.float64)
    pieces = [model.forward_cached(ids[:, :3], cache), model.forward_cached(ids[:, 3:], cache)]
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model(ids), expected, rtol=1e-12, atol=1e-12)


def test_micro_batches_are_weighted_by_their_scored_tokens():
    first, second = make_batch(2, 1, ignore=[(0, c) for c in range(8, 14)]), make_batch(2, 2)
    whole = Batch(*(torch.cat((a, b)) for a, b in zip(first, second)))
    accumulated, optimizers = build(torch.float64)
    train_step(accumulated, optimizers, [first, second], settings(max_grad_norm=None))
    single, single_optimizers = build(torch.float64)
    train_step(single, single_optimizers, [whole], settings(max_grad_norm=None))
    for (name, a), (_, b) in zip(accumulated.named_parameters(), single.named_parameters()):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


def test_unscored_tokens_add_nothing_to_the_loss_or_the_count():
    batch = make_batch(2, 1, ignore=[(0, 2), (1, 3), (1, 4)])
    model, optimizers = build()
    logits = model(batch.input_ids, document_ids=batch.document_ids)
    expected = F.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=IGNORE_INDEX)
    result = train_step(*(model, optimizers, [batch], settings()))
    scored = int((batch.targets != IGNORE_INDEX).sum())
    assert result["tokens"] == scored == 2 * 16 - 2 - 2 - 3
    torch.testing.assert_close(result["loss"], expected, rtol=1e-6, atol=1e-6)


def test_a_step_with_nothing_to_score_is_refused():
    batch = make_batch(2, 1)
    empty = Batch(batch.input_ids, torch.full_like(batch.targets, IGNORE_INDEX), batch.document_ids)
    model, optimizers = build()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="scored token"):
        train_step(model, optimizers, [empty], settings())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_gradient_clipping_reports_the_raw_norm_and_enforces_the_limit():
    batch = make_batch(2, 1)
    model, optimizers = build()
    raw = train_step(model, optimizers, [batch], settings(max_grad_norm=1e6))["grad_norm"].item()
    model, optimizers = build()
    result = train_step(model, optimizers, [batch], settings(max_grad_norm=0.05))
    assert result["grad_norm"].item() == pytest.approx(raw, rel=1e-5)
    clipped = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters()))
    assert clipped.item() == pytest.approx(0.05, rel=1e-4)


def test_no_clipping_leaves_gradients_alone():
    batch = make_batch(2, 1)
    model, optimizers = build()
    result = train_step(model, optimizers, [batch], settings(max_grad_norm=None))
    assert result["grad_norm"].item() == 0.0
    assert torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters())).item() > 0.05


def test_a_non_finite_gradient_stops_training_before_the_update():
    batch = make_batch(2, 1)
    model, optimizers = build()
    with torch.no_grad():
        model.model.layers[0].mlp.up_proj.weight[0, 0] = float("nan")
    before = model.model.norm.weight.clone()
    with pytest.raises(RuntimeError, match="non-finite"):
        train_step(model, optimizers, [batch], settings())
    assert torch.equal(model.model.norm.weight, before)


def test_bfloat16_autocast_keeps_float32_weights_and_stays_close():
    batch = make_batch(2, 1)
    results = {}
    for autocast in (False, True):
        model, optimizers = build()
        results[autocast] = train_step(model, optimizers, [batch], settings(autocast=autocast))
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert all(torch.isfinite(p).all() for p in model.parameters())
    assert results[True]["loss"].item() == pytest.approx(results[False]["loss"].item(), rel=0.02)


def test_learning_rates_follow_the_schedule_and_the_callback_sees_every_step():
    model, optimizers = build(lr=0.02)
    seen = []
    config = settings(total_steps=10, warmup_steps=2, decay_steps=3)
    assert train(model, optimizers, repeat(make_batch(2, 1)), config, on_step=lambda s, m: seen.append((s, m))) == 10
    assert [step for step, _ in seen] == list(range(1, 11))
    for step, metrics in seen:
        expected = 0.02 * wsd_factor(step - 1, 10, 2, 3)
        assert metrics["lrs"] == pytest.approx([expected, expected])
        assert metrics["tokens"] == int((make_batch(2, 1).targets != IGNORE_INDEX).sum())
        assert math.isfinite(metrics["loss"]) and metrics["grad_norm"] > 0


def test_training_in_two_pieces_equals_training_in_one():
    batch = make_batch(2, 1)
    one, one_optimizers = build()
    train(one, one_optimizers, repeat(batch), settings())
    two, two_optimizers = build()
    train(two, two_optimizers, repeat(batch), settings(), stop_step=4)
    train(two, two_optimizers, repeat(batch), settings(), start_step=4)
    assert all(torch.equal(a, b) for a, b in zip(one.parameters(), two.parameters()))


def test_training_is_bitwise_reproducible():
    results = []
    for _ in range(2):
        model, optimizers = build()
        train(model, optimizers, repeat(make_batch(2, 1)), settings(accumulate=2, autocast=True))
        results.append([p.detach().clone() for p in model.parameters()])
    assert all(torch.equal(a, b) for a, b in zip(*results))


def test_checkpoints_are_written_on_schedule_and_at_the_end(tmp_path):
    model, optimizers = build()
    path = tmp_path / "ck" / "latest.pt"
    steps = []
    train(model, optimizers, repeat(make_batch(2, 1)), settings(), stop_step=7, checkpoint_path=path, checkpoint_every=3,
          on_step=lambda s, m: steps.append((s, path.exists() and load_checkpoint(path, *build())[0])))
    assert [saved for _, saved in steps] == [False, False, False, 3, 3, 3, 6]
    assert load_checkpoint(path, *build())[0] == 7
    other, other_optimizers = build(seed=1)
    train(other, other_optimizers, repeat(make_batch(2, 1)), settings(), stop_step=2, checkpoint_path=tmp_path / "none.pt")
    assert not (tmp_path / "none.pt").exists()


def test_the_tiny_model_overfits_with_every_feature_switched_on():
    model, optimizers = build(lr=0.01)
    model.set_gradient_checkpointing(True)
    log = []
    batch = make_batch(4, 5)
    train(model, optimizers, repeat(batch), settings(total_steps=200, warmup_steps=10, decay_steps=40, accumulate=2,
                                                      autocast=True), on_step=lambda s, m: log.append(m["loss"]))
    assert log[0] > 3.5 and log[-1] < 0.1


@pytest.mark.parametrize(
    "kwargs",
    [dict(total_steps=0), dict(accumulate=0), dict(chunk_size=0), dict(max_grad_norm=0.0), dict(z_loss_weight=-1.0)]
    + [dict(warmup_steps=8, decay_steps=8), dict(warmup_steps=-1), dict(decay_steps=-1)],
)
def test_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        settings(**kwargs)


@pytest.mark.parametrize("start, stop", [(-1, 3), (5, 3), (0, 11)])
def test_rejects_an_invalid_step_range(start, stop):
    model, optimizers = build()
    with pytest.raises(ValueError):
        train(model, optimizers, repeat(make_batch(2, 1)), settings(), start_step=start, stop_step=stop)


def test_a_z_loss_weight_changes_the_update_but_not_the_reported_loss():
    batch = make_batch(2, 1)
    outcomes = []
    for weight in (0.0, 0.1):
        model, optimizers = build()
        result = train_step(model, optimizers, [batch], settings(z_loss_weight=weight, max_grad_norm=None))
        outcomes.append((result["loss"].item(), [p.detach().clone() for p in model.parameters()]))
    assert outcomes[0][0] == pytest.approx(outcomes[1][0], rel=1e-6)
    assert not all(torch.equal(a, b) for a, b in zip(outcomes[0][1], outcomes[1][1]))


def test_gradients_are_those_of_the_mean_loss_over_every_scored_token():
    first, second = make_batch(2, 1, ignore=[(0, c) for c in range(8, 14)]), make_batch(2, 2)
    model, optimizers = build(torch.float64)
    reference, _ = build(torch.float64)
    train_step(model, optimizers, [first, second], settings(max_grad_norm=None))
    whole = Batch(*(torch.cat((a, b)) for a, b in zip(first, second)))
    reference.loss(whole.input_ids, whole.targets, document_ids=whole.document_ids).backward()
    for (name, p), (_, r) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(p.grad, r.grad, rtol=1e-9, atol=1e-12)


def test_autocast_really_runs_the_matrix_multiplications_in_bfloat16():
    batch = make_batch(2, 1)
    for autocast, expected in ((False, torch.float32), (True, torch.bfloat16)):
        model, optimizers = build()
        seen = []
        model.model.layers[0].self_attn.q_proj.register_forward_hook(lambda module, args, output: seen.append(output.dtype))
        train_step(model, optimizers, [batch], settings(autocast=autocast))
        assert seen == [expected]


def test_settings_and_batch_tensors_reach_the_loss(monkeypatch):
    batch = make_batch(2, 1)
    model, optimizers = build()
    calls = []
    real = model.loss_totals

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(model, "loss_totals", spy)
    train_step(model, optimizers, [batch, batch], settings(chunk_size=7, z_loss_weight=0.25))
    assert len(calls) == 2
    assert all(call["chunk_size"] == 7 and call["z_loss_weight"] == 0.25 for call in calls)
    assert all(torch.equal(call["document_ids"], batch.document_ids) for call in calls)


def test_each_step_draws_exactly_accumulate_batches():
    drawn = []

    def counting():
        while True:
            drawn.append(1)
            yield make_batch(2, len(drawn))

    model, optimizers = build()
    train(model, optimizers, counting(), settings(accumulate=3), stop_step=4)
    assert len(drawn) == 12


def test_the_loop_puts_the_model_in_training_mode():
    model, optimizers = build()
    model.eval()
    train(model, optimizers, repeat(make_batch(2, 1)), settings(), stop_step=1)
    assert model.training
