import pytest
import torch

from scout.optim.adamw import CautiousAdamW


def grads(shape, count, seed=0, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=generator, dtype=dtype) for _ in range(count)]


def run(optimizer, p, sequence):
    for grad in sequence:
        p.grad = grad.clone()
        optimizer.step()
    return p.detach().clone()


def reference(start, sequence, lr, betas, eps, weight_decay, cautious):
    p, m, v = start.clone(), torch.zeros_like(start), torch.zeros_like(start)
    for t, g in enumerate(sequence, start=1):
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g * g
        u = (m / (1 - betas[0] ** t)) / ((v / (1 - betas[1] ** t)).sqrt() + eps)
        if cautious:
            keep = (u * g > 0).double()
            u = u * keep * (u.numel() / (keep.sum() + 1))
        p = p * (1 - lr * weight_decay) - lr * u
    return p


@pytest.mark.parametrize("shape", [(8, 6), (10,), (3, 4, 5)])
def test_without_caution_it_is_torch_adamw(shape):
    start = torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    sequence = grads(shape, 12)
    ours, theirs = torch.nn.Parameter(start.clone()), torch.nn.Parameter(start.clone())
    ours_result = run(CautiousAdamW([ours], lr=0.01, weight_decay=0.1, cautious=False), ours, sequence)
    theirs_result = run(torch.optim.AdamW([theirs], lr=0.01, betas=(0.9, 0.95), weight_decay=0.1), theirs, sequence)
    torch.testing.assert_close(ours_result, theirs_result, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_matches_an_explicit_reference(weight_decay):
    start = torch.randn(7, 9, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    sequence = grads((7, 9), 15, seed=3)
    p = torch.nn.Parameter(start.clone())
    result = run(CautiousAdamW([p], lr=0.02, weight_decay=weight_decay), p, sequence)
    expected = reference(start, sequence, 0.02, (0.9, 0.95), 1e-8, weight_decay, True)
    torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-12)
    assert not torch.allclose(result, reference(start, sequence, 0.02, (0.9, 0.95), 1e-8, weight_decay, False))


def test_never_moves_against_the_gradient_it_just_saw():
    p = torch.nn.Parameter(torch.zeros(50, dtype=torch.float64))
    optimizer = CautiousAdamW([p], lr=0.01)
    for grad in grads((50,), 30, seed=4):
        before = p.detach().clone()
        p.grad = grad
        optimizer.step()
        assert ((p.detach() - before) * grad <= 0).all()


def test_coordinates_whose_momentum_disagrees_with_the_gradient_stay_put():
    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    optimizer = CautiousAdamW([p], lr=0.1, betas=(0.99, 0.99))
    p.grad = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    optimizer.step()
    before = p.detach().clone()
    p.grad = torch.tensor([-0.01, 1.0, -0.01, 1.0], dtype=torch.float64)
    optimizer.step()
    moved = (p.detach() != before).tolist()
    assert moved == [False, True, False, True]


def test_a_zero_gradient_does_not_carry_the_momentum_forward():
    p = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = CautiousAdamW([p], lr=0.1)
    p.grad = torch.ones(3, dtype=torch.float64)
    optimizer.step()
    before = p.detach().clone()
    p.grad = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    optimizer.step()
    assert (p.detach() != before).tolist() == [False, True, False]


def test_the_step_is_rescaled_for_the_coordinates_that_were_masked():
    p = torch.nn.Parameter(torch.zeros(10, dtype=torch.float64))
    p.grad = torch.ones(10, dtype=torch.float64)
    CautiousAdamW([p], lr=0.1, eps=1e-30).step()
    assert torch.allclose(p.detach(), torch.full((10,), -0.1 * 10 / 11, dtype=torch.float64))


def test_weight_decay_uses_the_plain_learning_rate_even_when_everything_is_masked():
    p = torch.nn.Parameter(torch.ones(6, 4))
    p.grad = torch.zeros(6, 4)
    CautiousAdamW([p], lr=0.1, weight_decay=0.5).step()
    assert torch.allclose(p.detach(), torch.full((6, 4), 0.95))


def test_skips_parameters_without_gradients_and_reads_the_learning_rate_each_step():
    used, unused = torch.nn.Parameter(torch.zeros(5)), torch.nn.Parameter(torch.ones(5))
    optimizer = CautiousAdamW([used, unused], lr=0.1)
    used.grad = torch.ones(5)
    optimizer.param_groups[0]["lr"] = 0.0
    optimizer.step()
    assert torch.count_nonzero(used) == 0
    assert torch.equal(unused, torch.ones(5))
    assert unused not in optimizer.state


def test_state_dict_round_trip_continues_exactly():
    sequence = grads((12, 8), 8, seed=5)
    straight = torch.nn.Parameter(torch.zeros(12, 8, dtype=torch.float64))
    expected = run(CautiousAdamW([straight], lr=0.05, weight_decay=0.1), straight, sequence)
    resumed = torch.nn.Parameter(torch.zeros(12, 8, dtype=torch.float64))
    first = CautiousAdamW([resumed], lr=0.05, weight_decay=0.1)
    run(first, resumed, sequence[:4])
    second = CautiousAdamW([resumed], lr=0.05, weight_decay=0.1)
    second.load_state_dict(first.state_dict())
    assert torch.equal(run(second, resumed, sequence[4:]), expected)


def test_training_is_bitwise_reproducible():
    results = []
    for _ in range(2):
        p = torch.nn.Parameter(torch.zeros(12, 8))
        results.append(run(CautiousAdamW([p], lr=0.05), p, grads((12, 8), 6, dtype=torch.float32)))
    assert torch.equal(*results)


def test_learns_a_linear_map():
    generator = torch.Generator().manual_seed(0)
    target = torch.randn(16, 32, generator=generator)
    x = torch.randn(256, 32, generator=generator)
    weight = torch.nn.Parameter(torch.zeros(16, 32))
    optimizer = CautiousAdamW([weight], lr=0.02)
    losses = []
    for _ in range(300):
        loss = ((x @ weight.T - x @ target.T) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.01 * losses[0]


def test_works_on_vectors_and_bfloat16_parameters():
    vector = torch.nn.Parameter(torch.zeros(9, dtype=torch.bfloat16))
    vector.grad = torch.ones(9, dtype=torch.bfloat16)
    CautiousAdamW([vector], lr=0.1).step()
    assert vector.dtype == torch.bfloat16 and torch.isfinite(vector.float()).all() and (vector < 0).all()


@pytest.mark.parametrize(
    "kwargs",
    [{"lr": -1.0}, {"lr": 0.1, "betas": (1.0, 0.9)}, {"lr": 0.1, "betas": (0.9, -0.1)}]
    + [{"lr": 0.1, "weight_decay": -1.0}, {"lr": 0.1, "eps": 0.0}],
)
def test_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        CautiousAdamW([torch.nn.Parameter(torch.zeros(3))], **kwargs)
