import pytest
import torch

from scout.optim.normuon import COEFFICIENTS, NorMuon, orthogonalize


def matrix(rows, cols, seed=0, dtype=torch.float32):
    return torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def param(rows, cols, seed=0, dtype=torch.float32):
    p = torch.nn.Parameter(torch.zeros(rows, cols, dtype=dtype))
    p.grad = matrix(rows, cols, seed, dtype)
    return p


@pytest.mark.parametrize("shape", [(64, 64), (32, 128), (128, 32), (896, 2432), (5, 1)])
def test_orthogonalize_pushes_singular_values_towards_one(shape):
    values = torch.linalg.svdvals(orthogonalize(matrix(*shape)))
    assert values.max() < 1.25
    assert 0.6 < values.median() < 1.2


def test_newton_schulz_uses_the_published_quintic():
    assert COEFFICIENTS == (3.4445, -4.7750, 2.0315)


def test_orthogonalize_ignores_scale_and_follows_transposes():
    x = matrix(48, 96)
    assert torch.allclose(orthogonalize(x), orthogonalize(1000 * x), atol=1e-5)
    assert torch.equal(orthogonalize(x.T), orthogonalize(x).T)


def test_orthogonalize_keeps_float64_and_rejects_non_matrices():
    assert orthogonalize(matrix(8, 16, dtype=torch.float64)).dtype == torch.float64
    assert orthogonalize(matrix(8, 16).bfloat16()).dtype == torch.float32
    with pytest.raises(ValueError):
        orthogonalize(torch.ones(4))
    assert torch.count_nonzero(orthogonalize(torch.zeros(4, 6))) == 0


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
def test_first_step_has_the_target_rms_from_zero_weights(shape):
    p = param(*shape)
    NorMuon([p], lr=0.1).step()
    assert p.pow(2).mean().sqrt().item() == pytest.approx(0.1 * 0.2, rel=1e-4)


def test_second_moment_is_the_per_neuron_mean_square_of_the_orthogonal_update():
    p = param(40, 24)
    optimizer = NorMuon([p], lr=0.1, beta2=0.9)
    optimizer.step()
    expected = 0.1 * orthogonalize(matrix(40, 24)).square().mean(dim=-1, keepdim=True)
    assert optimizer.state[p]["second_moment"].shape == (40, 1)
    torch.testing.assert_close(optimizer.state[p]["second_moment"], expected, rtol=1e-5, atol=1e-8)


def test_rows_with_a_larger_orthogonal_update_are_scaled_down():
    p = param(40, 24)
    optimizer = NorMuon([p], lr=1.0)
    optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p)
    optimizer.state[p]["second_moment"] = torch.ones(40, 1)
    optimizer.state[p]["second_moment"][:20] = 100.0
    optimizer.step()
    rows = p.detach().pow(2).mean(dim=-1).sqrt()
    assert rows[:20].mean() < rows[20:].mean() / 3


def test_weight_decay_is_decoupled_and_zero_gradients_only_decay():
    p = torch.nn.Parameter(torch.ones(6, 4))
    p.grad = torch.zeros(6, 4)
    NorMuon([p], lr=0.1, weight_decay=0.5).step()
    assert torch.allclose(p, torch.full((6, 4), 0.95))
    assert torch.isfinite(p).all()


def test_plain_momentum_is_the_default_as_in_the_paper():
    assert NorMuon([torch.nn.Parameter(torch.zeros(4, 4))], lr=0.1).defaults["nesterov"] is False


def test_nesterov_flag_changes_the_step():
    results = []
    for nesterov in (True, False):
        p = param(16, 16)
        optimizer = NorMuon([p], lr=0.1, nesterov=nesterov)
        optimizer.step()
        p.grad = matrix(16, 16, seed=5)
        optimizer.step()
        results.append(p.detach().clone())
    assert not torch.allclose(results[0], results[1])


def test_skips_parameters_without_gradients_and_reads_the_learning_rate_each_step():
    used, unused = param(8, 8), torch.nn.Parameter(torch.ones(8, 8))
    optimizer = NorMuon([used, unused], lr=0.1)
    optimizer.param_groups[0]["lr"] = 0.0
    optimizer.step()
    assert torch.count_nonzero(used) == 0
    assert torch.equal(unused, torch.ones(8, 8))
    assert unused not in optimizer.state


def test_state_dict_round_trip_continues_exactly():
    def run(p, optimizer, seeds):
        for seed in seeds:
            p.grad = matrix(24, 16, seed)
            optimizer.step()

    straight = param(24, 16, seed=9)
    straight_optimizer = NorMuon([straight], lr=0.05, weight_decay=0.1)
    run(straight, straight_optimizer, range(6))
    resumed = param(24, 16, seed=9)
    first = NorMuon([resumed], lr=0.05, weight_decay=0.1)
    run(resumed, first, range(3))
    second = NorMuon([resumed], lr=0.05, weight_decay=0.1)
    second.load_state_dict(first.state_dict())
    run(resumed, second, range(3, 6))
    assert torch.equal(straight, resumed)


def test_training_is_bitwise_reproducible():
    results = []
    for _ in range(2):
        p = param(24, 16, seed=2)
        optimizer = NorMuon([p], lr=0.05)
        for seed in range(5):
            p.grad = matrix(24, 16, seed)
            optimizer.step()
        results.append(p.detach().clone())
    assert torch.equal(*results)


def test_learns_a_linear_map():
    generator = torch.Generator().manual_seed(0)
    target = torch.randn(16, 32, generator=generator)
    x = torch.randn(256, 32, generator=generator)
    weight = torch.nn.Parameter(torch.zeros(16, 32))
    optimizer = NorMuon([weight], lr=0.02)
    losses = []
    for _ in range(300):
        loss = ((x @ weight.T - x @ target.T) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.05 * losses[0]


def test_float64_parameters_keep_float64_state():
    p = param(8, 8, dtype=torch.float64)
    optimizer = NorMuon([p], lr=0.1)
    optimizer.step()
    assert p.dtype == optimizer.state[p]["second_moment"].dtype == torch.float64


@pytest.mark.parametrize(
    "kwargs",
    [{"lr": -1.0}, {"lr": 0.1, "momentum": 1.0}, {"lr": 0.1, "beta2": -0.1}, {"lr": 0.1, "weight_decay": -1.0}]
    + [{"lr": 0.1, "steps": 0}, {"lr": 0.1, "eps": 0.0}],
)
def test_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        NorMuon([torch.nn.Parameter(torch.zeros(4, 4))], **kwargs)


@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
def test_rejects_parameters_that_are_not_matrices(shape):
    with pytest.raises(ValueError):
        NorMuon([torch.nn.Parameter(torch.zeros(*shape))], lr=0.1)
