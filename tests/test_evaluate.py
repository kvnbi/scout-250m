import math
import random

import pytest
import torch

from scout.config import ModelConfig
import scout.evaluate
from scout.evaluate import Choices, evaluate_choices, evaluate_loss, evaluating, option_log_probs
from scout.loss import IGNORE_INDEX
from scout.model import Scout
from scout.optim.groups import build_optimizers
from scout.train import Batch, TrainSettings, train

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


def build(dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    model = Scout(TINY).to(dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 1:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
    return model


def items(count=6, seed=1):
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        prompt = [rng.randrange(50) for _ in range(rng.randint(1, 9))]
        options = [[rng.randrange(50) for _ in range(rng.randint(1, 6))] for _ in range(rng.randint(2, 4))]
        result.append(Choices(prompt, options, rng.randrange(len(options))))
    return result


def naive(model, item):
    scores = []
    for option in item.options:
        ids = torch.tensor([item.prompt + option])
        logits = model(ids)[0].double().log_softmax(-1)
        total = 0.0
        for t in range(len(item.prompt), len(item.prompt) + len(option)):
            total += logits[t - 1, ids[0, t]].item()
        scores.append(total)
    return scores


def test_option_scores_equal_a_naive_full_logit_computation():
    model = build()
    data = items()
    got = option_log_probs(model, data)
    for item, row in zip(data, got):
        assert row == pytest.approx(naive(model, item), rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 100])
def test_batching_padding_and_ordering_do_not_change_the_scores(batch_size):
    model = build()
    data = items(9, seed=4)
    reference = option_log_probs(model, data, batch_size=1)
    got = option_log_probs(model, data, batch_size=batch_size)
    for a, b in zip(reference, got):
        assert a == pytest.approx(b, rel=1e-10, abs=1e-10)
    reordered = option_log_probs(model, list(reversed(data)), batch_size=batch_size)
    for a, b in zip(reference, reversed(reordered)):
        assert a == pytest.approx(b, rel=1e-10, abs=1e-10)


def test_each_scored_token_adds_a_negative_log_probability():
    model = build()
    short = option_log_probs(model, [Choices([1, 2, 3], [[4], [4, 5]], 0)])[0]
    assert short[0] < 0 and short[1] < short[0]


def test_correct_probability_is_a_softmax_over_the_options():
    model = build()
    item = items(1)[0]
    row = option_log_probs(model, [item])[0]
    weights = [math.exp(v - max(row)) for v in row]
    result = evaluate_choices(model, [item])
    assert result["correct_probability"] == pytest.approx(weights[item.correct] / sum(weights), rel=1e-10)
    probabilities = [
        evaluate_choices(model, [Choices(item.prompt, item.options, k)])["correct_probability"]
        for k in range(len(item.options))
    ]
    assert sum(probabilities) == pytest.approx(1.0)
    assert result["items"] == 1 and result["accuracy"] in (0.0, 1.0)


def test_length_normalisation_changes_the_ranking_of_short_and_long_options():
    model = build()
    item = Choices([1, 2, 3], [[4], [4, 5, 6, 7, 8]], 0)
    plain = evaluate_choices(model, [item])["correct_probability"]
    normalised = evaluate_choices(model, [item], normalize=True)["correct_probability"]
    assert plain != pytest.approx(normalised)
    row = option_log_probs(model, [item])[0]
    expected = [row[0] / 1, row[1] / 5]
    weights = [math.exp(v - max(expected)) for v in expected]
    assert normalised == pytest.approx(weights[0] / sum(weights), rel=1e-10)


def test_answer_log_prob_is_the_mean_per_token_of_the_correct_option():
    model = build()
    item = Choices([1, 2, 3], [[4, 5], [6]], 0)
    row = option_log_probs(model, [item])[0]
    assert evaluate_choices(model, [item])["answer_log_prob"] == pytest.approx(row[0] / 2, rel=1e-10)


def test_an_untrained_model_is_near_chance_and_a_trained_one_is_not():
    config = TINY
    torch.manual_seed(0)
    model = Scout(config)
    prompts = [[1 + i, 20 + i, 30 + i] for i in range(8)]
    data = [Choices(p, [[40 + i % 5, 41], [50 + (i + 1) % 5, 51], [45, 46 + i % 3]], 0) for i, p in enumerate(prompts)]
    before = evaluate_choices(model, data)
    assert before["correct_probability"] < 0.7
    optimizers = build_optimizers(model, 0.02, 0.02, weight_decay=0.0)
    ids = torch.tensor([item.prompt + item.options[0] for item in data])
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = IGNORE_INDEX
    targets[:, :2] = IGNORE_INDEX

    def forever():
        while True:
            yield Batch(ids, targets)

    train(model, optimizers, forever(), TrainSettings(total_steps=150, warmup_steps=5, decay_steps=30))
    after = evaluate_choices(model, data)
    assert after["correct_probability"] > 0.95 and after["accuracy"] == 1.0
    assert after["answer_log_prob"] > before["answer_log_prob"]


def loss_batches(count=3):
    generator = torch.Generator().manual_seed(7)
    for number in range(count):
        ids = torch.randint(0, 50, (2, 12), generator=generator)
        targets = torch.roll(ids, -1, 1)
        targets[:, -1] = IGNORE_INDEX
        targets[0, number] = IGNORE_INDEX
        yield Batch(ids, targets)


def test_held_out_loss_is_the_mean_over_every_scored_token():
    model = build()
    result = evaluate_loss(model, loss_batches(), chunk_size=5)
    total = count = 0
    for batch in loss_batches():
        logits = model(batch.input_ids).log_softmax(-1)
        for row in range(batch.targets.shape[0]):
            for t in range(batch.targets.shape[1]):
                if batch.targets[row, t] != IGNORE_INDEX:
                    total -= logits[row, t, batch.targets[row, t]].item()
                    count += 1
    assert result["tokens"] == count
    assert result["loss"] == pytest.approx(total / count, rel=1e-10)


def test_evaluation_leaves_weights_gradients_modes_and_random_streams_alone():
    model = build(torch.float32)
    model.train()
    before = [p.detach().clone() for p in model.parameters()]
    torch.manual_seed(3)
    random.seed(3)
    state = (torch.get_rng_state(), random.getstate())
    evaluate_loss(model, loss_batches())
    evaluate_choices(model, items())
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert torch.equal(state[0], torch.get_rng_state()) and state[1] == random.getstate()
    model.eval()
    evaluate_loss(model, loss_batches())
    assert not model.training


def test_bfloat16_autocast_stays_close_to_full_precision():
    model = build(torch.float32)
    data = items(5)
    full = evaluate_choices(model, data)
    low = evaluate_choices(model, data, autocast=True)
    assert low["correct_probability"] == pytest.approx(full["correct_probability"], abs=0.05)
    assert evaluate_loss(model, loss_batches(), autocast=True)["loss"] == pytest.approx(
        evaluate_loss(model, loss_batches())["loss"], rel=0.02
    )


@pytest.mark.parametrize(
    "item",
    [
        Choices([], [[1]], 0),
        Choices([1], [], 0),
        Choices([1], [[2], []], 0),
        Choices([1], [[2]], 1),
        Choices([1], [[2]], -1),
        Choices([1] * 30, [[2, 3, 4]], 0),
    ],
)
def test_rejects_malformed_items(item):
    with pytest.raises(ValueError):
        option_log_probs(build(), [item])


def test_rejects_empty_inputs():
    with pytest.raises(ValueError):
        evaluate_choices(build(), [])
    with pytest.raises(ValueError):
        evaluate_loss(build(), [])


def batches_forever():
    generator = torch.Generator().manual_seed(11)
    while True:
        ids = torch.randint(0, 50, (2, 12), generator=generator)
        targets = torch.roll(ids, -1, 1)
        targets[:, -1] = IGNORE_INDEX
        yield Batch(ids, targets)


def run(evaluate=None, **kwargs):
    model = build(torch.float32)
    optimizers = build_optimizers(model, 0.02, 0.01)
    train(model, optimizers, batches_forever(), TrainSettings(total_steps=7, warmup_steps=1, decay_steps=2),
          evaluate=evaluate, **kwargs)
    return [p.detach().clone() for p in model.parameters()]


def test_the_loop_evaluates_on_schedule_and_at_the_end():
    seen, modes = [], []

    def evaluate(model):
        modes.append(model.training)
        return {"value": len(seen)}

    run(evaluate, evaluate_every=3, on_evaluate=lambda step, results: seen.append((step, results)))
    assert seen == [(3, {"value": 0}), (6, {"value": 1}), (7, {"value": 2})]
    assert modes == [True, True, True]


def test_evaluating_during_training_changes_nothing_about_training():
    plain = run()
    watched = run(lambda model: evaluate_loss(model, loss_batches()), evaluate_every=2)
    assert all(torch.equal(a, b) for a, b in zip(plain, watched))


def test_no_evaluation_without_a_schedule_or_a_function():
    calls = []
    run(lambda model: calls.append(1) or {}, evaluate_every=0)
    run(None, evaluate_every=2)
    assert calls == []


def test_the_softmax_survives_very_negative_scores(monkeypatch):
    monkeypatch.setattr(scout.evaluate, "option_log_probs", lambda *args, **kwargs: [[-1000.0, -1001.0]])
    result = evaluate_choices(build(), [Choices([1], [[2], [3]], 0)])
    assert result["correct_probability"] == pytest.approx(1 / (1 + math.exp(-1)))


def test_the_evaluation_context_is_eval_mode_without_gradients_and_restores_the_mode():
    model = build(torch.float32)
    model.train()
    with evaluating(model):
        assert not model.training
        assert not model(torch.tensor([[1, 2, 3]])).requires_grad
    assert model.training
    model.eval()
    with evaluating(model):
        pass
    assert not model.training


def test_autocast_really_runs_evaluation_in_bfloat16():
    for function, arguments in ((evaluate_loss, (list(loss_batches()),)), (option_log_probs, (items(3),))):
        for autocast, expected in ((False, torch.float32), (True, torch.bfloat16)):
            model = build(torch.float32)
            seen = []
            model.model.layers[0].self_attn.q_proj.register_forward_hook(lambda module, args, output: seen.append(output.dtype))
            function(model, *arguments, autocast=autocast)
            assert set(seen) == {expected}
