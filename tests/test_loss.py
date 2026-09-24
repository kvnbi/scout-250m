import math

import pytest
import torch
import torch.nn.functional as F

from scout.config import ModelConfig
from scout.loss import IGNORE_INDEX, lm_loss, lm_loss_totals
from scout.model import Scout


def inputs(tokens=37, width=16, vocab=50, dtype=torch.float64, seed=0, ignored=(3, 10, 11, 30)):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(tokens, width, dtype=dtype, generator=generator)
    weight = torch.randn(vocab, width, dtype=dtype, generator=generator) * 0.5
    targets = torch.randint(0, vocab, (tokens,), generator=generator)
    targets[list(ignored)] = IGNORE_INDEX
    return hidden.requires_grad_(True), weight.requires_grad_(True), targets


def reference(hidden, weight, targets, z_loss_weight=0.0):
    logits = (hidden @ weight.T).double()
    valid = targets != IGNORE_INDEX
    cross_entropy = F.cross_entropy(logits[valid], targets[valid], reduction="sum")
    z_loss = torch.logsumexp(logits[valid], dim=-1).square().sum()
    return (cross_entropy + z_loss_weight * z_loss) / valid.sum().clamp_min(1)


@pytest.mark.parametrize("chunk_size", [1, 5, 7, 37, 4096])
def test_matches_reference_cross_entropy(chunk_size):
    hidden, weight, targets = inputs()
    expected = reference(hidden, weight, targets)
    torch.testing.assert_close(lm_loss(hidden, weight, targets, chunk_size), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("chunk_size", [5, 4096])
def test_gradients_match_reference(chunk_size):
    hidden, weight, targets = inputs()
    lm_loss(hidden, weight, targets, chunk_size).backward()
    ours = (hidden.grad.clone(), weight.grad.clone())
    hidden.grad = None
    weight.grad = None
    reference(hidden, weight, targets).backward()
    torch.testing.assert_close(ours[0], hidden.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ours[1], weight.grad, rtol=1e-12, atol=1e-12)


def test_z_loss_matches_reference():
    hidden, weight, targets = inputs()
    torch.testing.assert_close(
        lm_loss(hidden, weight, targets, 5, z_loss_weight=1e-4),
        reference(hidden, weight, targets, z_loss_weight=1e-4),
        rtol=1e-12,
        atol=1e-12,
    )


def test_totals_count_only_valid_tokens():
    hidden, weight, targets = inputs()
    totals = lm_loss_totals(hidden, weight, targets, 5)
    assert totals.tokens.item() == 33
    assert totals.cross_entropy_sum.dtype == totals.z_loss_sum.dtype == torch.float64
    low = lm_loss_totals(hidden.bfloat16(), weight.bfloat16(), targets, 5)
    assert low.cross_entropy_sum.dtype == low.z_loss_sum.dtype == torch.float32


def test_totals_combine_exactly_across_micro_batches():
    hidden, weight, targets = inputs(tokens=40)
    first = lm_loss_totals(hidden[:9], weight, targets[:9], 4)
    second = lm_loss_totals(hidden[9:], weight, targets[9:], 4)
    combined = (first.cross_entropy_sum + second.cross_entropy_sum) / (first.tokens + second.tokens)
    torch.testing.assert_close(combined, reference(hidden, weight, targets), rtol=1e-12, atol=1e-12)


def test_accepts_batch_and_sequence_dimensions():
    hidden, weight, targets = inputs(tokens=24, ignored=(3, 10, 11))
    batched = lm_loss(hidden.reshape(2, 12, 16), weight, targets.reshape(2, 12), 5)
    torch.testing.assert_close(batched, lm_loss(hidden, weight, targets, 5), rtol=1e-12, atol=1e-12)


def test_all_ignored_gives_zero_loss_and_zero_gradients():
    hidden, weight, targets = inputs(tokens=6, ignored=range(6))
    loss = lm_loss(hidden, weight, targets, 4)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(hidden.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0


def test_is_bitwise_deterministic():
    hidden, weight, targets = inputs(dtype=torch.float32)
    assert torch.equal(lm_loss(hidden, weight, targets, 5), lm_loss(hidden, weight, targets, 5))


def test_full_logits_are_never_stored_for_backward():
    tokens, vocab = 64, 1000
    hidden, weight, targets = inputs(tokens=tokens, width=8, vocab=vocab, dtype=torch.float32, ignored=())
    stored = []

    def pack(tensor):
        stored.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = lm_loss(hidden, weight, targets, chunk_size=8)
    loss.backward()
    assert max(stored) < tokens * vocab // 4
    stored.clear()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        F.cross_entropy(hidden @ weight.T, targets)
    assert max(stored) >= tokens * vocab


def test_bfloat16_inputs_stay_close_to_float64():
    hidden, weight, targets = inputs(dtype=torch.float32)
    loss = lm_loss(hidden.bfloat16(), weight.bfloat16(), targets, 5)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss.double(), reference(hidden, weight, targets), rtol=0.003, atol=0.003)


@pytest.mark.parametrize("bad", [-1, 50])
def test_rejects_out_of_range_targets(bad):
    hidden, weight, targets = inputs()
    targets[0] = bad
    with pytest.raises(ValueError):
        lm_loss(hidden, weight, targets)


@pytest.mark.parametrize("ignore_index", [50, 999, -1])
def test_custom_ignore_index(ignore_index):
    hidden, weight, targets = inputs()
    custom = torch.where(targets == IGNORE_INDEX, ignore_index, targets)
    expected = reference(hidden, weight, targets)
    torch.testing.assert_close(lm_loss(hidden, weight, custom, 5, ignore_index), expected, rtol=1e-12, atol=1e-12)


def test_works_without_gradients():
    hidden, weight, targets = inputs()
    with torch.no_grad():
        loss = lm_loss(hidden, weight, targets, 5)
    assert not loss.requires_grad
    torch.testing.assert_close(loss, reference(hidden, weight, targets).detach(), rtol=1e-12, atol=1e-12)


def test_leaves_random_state_untouched():
    hidden, weight, targets = inputs()
    before = torch.get_rng_state()
    lm_loss(hidden, weight, targets, 5).backward()
    assert torch.equal(before, torch.get_rng_state())


def test_rejects_misaligned_shapes():
    hidden, weight, targets = inputs()
    with pytest.raises(ValueError):
        lm_loss(hidden, weight, targets[:-1])
    with pytest.raises(ValueError):
        lm_loss(hidden, weight[:, :-1], targets)


def test_rejects_float_targets():
    hidden, weight, targets = inputs()
    with pytest.raises(TypeError):
        lm_loss(hidden, weight, targets.float())


@pytest.mark.parametrize("kwargs", [{"chunk_size": 0}, {"z_loss_weight": -1.0}])
def test_rejects_invalid_settings(kwargs):
    hidden, weight, targets = inputs()
    with pytest.raises(ValueError):
        lm_loss(hidden, weight, targets, **kwargs)


def small_model(seed=0):
    torch.manual_seed(seed)
    return Scout(ModelConfig(vocab_size=512, n_layers=2)).double()


def test_forward_equals_head_of_hidden_states():
    model = small_model()
    ids = torch.randint(0, 512, (2, 12))
    assert torch.equal(model(ids), model.lm_head(model.hidden_states(ids)))


def test_model_loss_and_tied_gradient_match_full_logits():
    model = small_model()
    ids = torch.randint(0, 512, (2, 12))
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = IGNORE_INDEX
    model.loss(ids, targets, chunk_size=5).backward()
    ours = model.lm_head.weight.grad.clone()
    model.zero_grad()
    logits = model(ids)
    F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE_INDEX).backward()
    torch.testing.assert_close(ours, model.lm_head.weight.grad, rtol=1e-10, atol=1e-12)


def test_model_loss_totals_match_model_loss():
    model = small_model()
    ids = torch.randint(0, 512, (2, 12))
    targets = torch.roll(ids, -1, 1)
    totals = model.loss_totals(ids, targets, chunk_size=7)
    mean = totals.cross_entropy_sum / totals.tokens
    torch.testing.assert_close(mean, model.loss(ids, targets, chunk_size=7), rtol=1e-12, atol=1e-12)


def test_untrained_model_loss_is_close_to_uniform():
    torch.manual_seed(0)
    model = Scout(ModelConfig(n_layers=2))
    ids = torch.randint(0, 32768, (2, 128))
    with torch.no_grad():
        loss = model.loss(ids, torch.roll(ids, -1, 1)).item()
    assert math.log(32768) <= loss <= math.log(32768) + 0.4
