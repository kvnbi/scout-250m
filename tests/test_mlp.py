import pytest
import torch

from scout.config import ModelConfig
from scout.mlp import SwiGLU


def make(dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    return SwiGLU(896, 2432).to(dtype)


def reference(mlp, x):
    x = x.double()
    gate = x @ mlp.gate_proj.weight.double().T
    up = x @ mlp.up_proj.weight.double().T
    return (gate / (1.0 + torch.exp(-gate)) * up) @ mlp.down_proj.weight.double().T


def test_parameter_names_and_no_biases():
    names = {name for name, _ in make().named_parameters()}
    assert names == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}


def test_parameter_shapes():
    mlp = make()
    assert mlp.gate_proj.weight.shape == (2432, 896)
    assert mlp.up_proj.weight.shape == (2432, 896)
    assert mlp.down_proj.weight.shape == (896, 2432)


def test_parameter_count_matches_config():
    config = ModelConfig()
    expected = 3 * config.d_model * config.ffn_hidden
    assert sum(p.numel() for p in make().parameters()) == expected


def test_output_shape_and_dtype():
    y = make()(torch.randn(2, 16, 896))
    assert y.shape == (2, 16, 896)
    assert y.dtype == torch.float32


def test_matches_explicit_reference():
    mlp = make()
    x = torch.randn(3, 8, 896)
    torch.testing.assert_close(mlp(x).double(), reference(mlp, x), rtol=1e-4, atol=1e-5)


def test_matches_reference_in_float64():
    mlp = make(torch.float64)
    x = torch.randn(3, 8, 896, dtype=torch.float64)
    torch.testing.assert_close(mlp(x), reference(mlp, x), rtol=1e-12, atol=1e-12)


def test_gate_and_up_are_not_interchangeable():
    mlp = make(torch.float64)
    x = torch.randn(2, 4, 896, dtype=torch.float64)
    swapped = SwiGLU(896, 2432).double()
    with torch.no_grad():
        swapped.gate_proj.weight.copy_(mlp.up_proj.weight)
        swapped.up_proj.weight.copy_(mlp.gate_proj.weight)
        swapped.down_proj.weight.copy_(mlp.down_proj.weight)
    assert not torch.allclose(mlp(x), swapped(x))


def test_acts_on_each_token_independently():
    mlp = make(torch.float64)
    x = torch.randn(1, 6, 896, dtype=torch.float64)
    changed = x.clone()
    changed[:, 3] = torch.randn(896, dtype=torch.float64)
    before, after = mlp(x), mlp(changed)
    assert torch.equal(before[:, :3], after[:, :3])
    assert torch.equal(before[:, 4:], after[:, 4:])
    assert not torch.allclose(before[:, 3], after[:, 3])


def test_zero_input_gives_zero_output():
    assert torch.equal(make()(torch.zeros(2, 896)), torch.zeros(2, 896))


@pytest.mark.parametrize("seed", range(5))
def test_bfloat16_error_stays_within_rounding(seed):
    mlp = make(seed=seed)
    x = torch.randn(2, 16, 896)
    expected = reference(mlp, x)
    y = mlp.to(torch.bfloat16)(x.bfloat16())
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y.double(), expected, rtol=0.0, atol=0.01)


def test_gradients_are_finite():
    mlp = make()
    x = torch.randn(2, 8, 896, requires_grad=True)
    mlp(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, parameter in mlp.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_passes_gradcheck():
    torch.manual_seed(1)
    mlp = SwiGLU(8, 12).double()
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(mlp, (x,))


@pytest.mark.parametrize("d_model, hidden", [(0, 2432), (896, 0)])
def test_rejects_invalid_arguments(d_model, hidden):
    with pytest.raises(ValueError):
        SwiGLU(d_model, hidden)


def test_rejects_wrong_input_width():
    with pytest.raises(ValueError):
        make()(torch.randn(2, 512))


def test_rejects_scalar_input():
    with pytest.raises(ValueError):
        make()(torch.tensor(1.0))
