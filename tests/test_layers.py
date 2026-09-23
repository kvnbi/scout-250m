import pytest
import torch
import torch.nn.functional as F

from scout.layers import RMSNorm


def reference_rmsnorm(x, weight, eps):
    x = x.double()
    return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.double()


def test_rmsnorm_parameters():
    norm = RMSNorm(896)
    params = list(norm.parameters())
    assert len(params) == 1
    assert params[0].shape == (896,)
    assert torch.equal(norm.weight, torch.ones(896))


def test_rmsnorm_matches_reference():
    torch.manual_seed(0)
    norm = RMSNorm(64)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64))
    x = torch.randn(2, 5, 64)
    expected = reference_rmsnorm(x, norm.weight, norm.eps)
    torch.testing.assert_close(norm(x).double(), expected, rtol=1e-5, atol=1e-6)


def test_rmsnorm_matches_torch_functional():
    torch.manual_seed(1)
    norm = RMSNorm(896)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(896))
    x = torch.randn(3, 7, 896)
    expected = F.rms_norm(x, (896,), norm.weight, norm.eps)
    torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-6)


def test_rmsnorm_output_has_unit_rms():
    torch.manual_seed(2)
    x = torch.randn(4, 16, 896) * 7.0 + 3.0
    y = RMSNorm(896)(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-4, atol=1e-4)


def test_rmsnorm_is_scale_invariant():
    torch.manual_seed(3)
    norm = RMSNorm(64)
    x = torch.randn(8, 64)
    torch.testing.assert_close(norm(x * 25.0), norm(x), rtol=1e-5, atol=1e-5)


def test_rmsnorm_preserves_bfloat16():
    torch.manual_seed(4)
    norm = RMSNorm(896)
    x = torch.randn(2, 8, 896).bfloat16()
    y = norm(x)
    assert y.dtype == torch.bfloat16
    expected = reference_rmsnorm(x, norm.weight, norm.eps)
    torch.testing.assert_close(y.double(), expected, rtol=1e-2, atol=1e-2)


def test_rmsnorm_keeps_float64_precision():
    torch.manual_seed(6)
    norm = RMSNorm(64).double()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(64, dtype=torch.float64))
    x = torch.randn(3, 64, dtype=torch.float64)
    y = norm(x)
    assert y.dtype == torch.float64
    torch.testing.assert_close(y, reference_rmsnorm(x, norm.weight, norm.eps), rtol=1e-12, atol=1e-12)


def test_rmsnorm_passes_gradcheck():
    torch.manual_seed(7)
    norm = RMSNorm(16).double()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16, dtype=torch.float64))
    x = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(norm, (x,))


def test_rmsnorm_handles_zero_input():
    y = RMSNorm(64)(torch.zeros(3, 64))
    assert torch.isfinite(y).all()
    assert torch.equal(y, torch.zeros(3, 64))


def test_rmsnorm_gradients_are_finite():
    torch.manual_seed(5)
    norm = RMSNorm(64)
    x = torch.randn(4, 64, requires_grad=True)
    norm(x).square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()
    assert norm.weight.grad.abs().sum() > 0


def test_rmsnorm_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        RMSNorm(64)(torch.randn(2, 32))


def test_rmsnorm_rejects_integer_input():
    with pytest.raises(TypeError):
        RMSNorm(4)(torch.ones(2, 4, dtype=torch.int64))


@pytest.mark.parametrize("dim, eps", [(0, 1e-6), (64, 0.0), (64, -1e-6)])
def test_rmsnorm_rejects_invalid_arguments(dim, eps):
    with pytest.raises(ValueError):
        RMSNorm(dim, eps)
