import math

import pytest
import torch

from scout.rotary import RotaryEmbedding, apply_rotary, rotate_half


def qwen3_cos_sin(inv_freq, positions, dtype):
    inv_freq_expanded = inv_freq[None, :, None].float().expand(positions.shape[0], -1, 1)
    positions_expanded = positions[:, None, :].float()
    freqs = (inv_freq_expanded @ positions_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def complex_reference(x, positions, head_dim, base):
    half = head_dim // 2
    x = x.double()
    inv_freq = torch.tensor([base ** (-2 * i / head_dim) for i in range(half)], dtype=torch.float64)
    angles = positions.double()[:, None, :, None] * inv_freq
    z = torch.complex(x[..., :half], x[..., half:]) * torch.polar(torch.ones_like(angles), angles)
    return torch.cat((z.real, z.imag), dim=-1)


def test_inverse_frequencies():
    rotary = RotaryEmbedding(64, base=10000.0)
    expected = torch.tensor([10000.0 ** (-2 * i / 64) for i in range(32)], dtype=torch.float64)
    torch.testing.assert_close(rotary.inv_freq.double(), expected, rtol=1e-6, atol=0.0)


def test_inverse_frequencies_are_not_saved():
    assert "inv_freq" not in RotaryEmbedding(64).state_dict()


def test_inverse_frequencies_survive_dtype_casts():
    reference = RotaryEmbedding(64).inv_freq.clone()
    for dtype in (torch.bfloat16, torch.float16, torch.float64):
        rotary = RotaryEmbedding(64).to(dtype)
        assert rotary.inv_freq.dtype == torch.float32
        assert torch.equal(rotary.inv_freq, reference)


def test_inverse_frequencies_survive_module_casts():
    parent = torch.nn.Module()
    parent.rotary = RotaryEmbedding(64)
    parent.bfloat16()
    assert parent.rotary.inv_freq.dtype == torch.float32
    assert torch.equal(parent.rotary.inv_freq, RotaryEmbedding(64).inv_freq)


def test_cos_sin_ignore_autocast():
    rotary = RotaryEmbedding(64)
    positions = torch.arange(512).repeat(1, 1)
    expected = rotary(positions, torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        cos, sin = rotary(positions, torch.float32)
    assert torch.equal(cos, expected[0])
    assert torch.equal(sin, expected[1])


def test_cos_sin_match_qwen3_exactly():
    rotary = RotaryEmbedding(64)
    positions = torch.arange(2048).repeat(2, 1)
    cos, sin = rotary(positions, torch.float32)
    ref_cos, ref_sin = qwen3_cos_sin(rotary.inv_freq, positions, torch.float32)
    assert torch.equal(cos, ref_cos)
    assert torch.equal(sin, ref_sin)


def test_cos_sin_shape_and_dtype():
    rotary = RotaryEmbedding(64)
    cos, sin = rotary(torch.arange(10).repeat(3, 1), torch.bfloat16)
    assert cos.shape == sin.shape == (3, 10, 64)
    assert cos.dtype == sin.dtype == torch.bfloat16


def test_rotate_half_layout():
    x = torch.arange(8.0)
    assert torch.equal(rotate_half(x), torch.tensor([-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0]))


def test_matches_complex_rotation():
    torch.manual_seed(0)
    rotary = RotaryEmbedding(64, base=10000.0)
    positions = torch.stack((torch.arange(16), torch.arange(100, 116)))
    x = torch.randn(2, 14, 16, 64)
    cos, sin = rotary(positions, torch.float32)
    expected = complex_reference(x, positions, 64, 10000.0)
    torch.testing.assert_close(apply_rotary(x, cos, sin).double(), expected, rtol=1e-4, atol=1e-4)


def test_position_zero_is_identity():
    torch.manual_seed(1)
    x = torch.randn(1, 2, 1, 64)
    cos, sin = RotaryEmbedding(64)(torch.zeros(1, 1, dtype=torch.long), torch.float32)
    assert torch.equal(apply_rotary(x, cos, sin), x)


def test_rotation_preserves_norm():
    torch.manual_seed(2)
    x = torch.randn(2, 4, 32, 64)
    cos, sin = RotaryEmbedding(64)(torch.arange(32).repeat(2, 1), torch.float32)
    torch.testing.assert_close(apply_rotary(x, cos, sin).norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_scores_depend_only_on_relative_position():
    torch.manual_seed(3)
    rotary = RotaryEmbedding(64)
    q = torch.randn(1, 1, 1, 64, dtype=torch.float64)
    k = torch.randn(1, 1, 1, 64, dtype=torch.float64)

    def score(m, n):
        cos_q, sin_q = rotary(torch.tensor([[m]]), torch.float64)
        cos_k, sin_k = rotary(torch.tensor([[n]]), torch.float64)
        return (apply_rotary(q, cos_q, sin_q) * apply_rotary(k, cos_k, sin_k)).sum()

    assert math.isclose(score(5, 2), score(1003, 1000), rel_tol=1e-5, abs_tol=1e-6)
    assert not math.isclose(score(5, 2), score(9, 2), rel_tol=1e-3)


def test_unsqueeze_dim_supports_heads_last_layout():
    torch.manual_seed(4)
    x = torch.randn(2, 14, 8, 64)
    cos, sin = RotaryEmbedding(64)(torch.arange(8).repeat(2, 1), torch.float32)
    heads_first = apply_rotary(x, cos, sin, unsqueeze_dim=1)
    heads_last = apply_rotary(x.transpose(1, 2), cos, sin, unsqueeze_dim=2)
    assert torch.equal(heads_first, heads_last.transpose(1, 2))


def test_cos_sin_carry_no_gradient():
    cos, sin = RotaryEmbedding(64)(torch.arange(4).repeat(1, 1), torch.float32)
    assert not cos.requires_grad
    assert not sin.requires_grad


@pytest.mark.parametrize("head_dim, base", [(0, 10000.0), (63, 10000.0), (64, 1.0), (64, 0.5)])
def test_rejects_invalid_arguments(head_dim, base):
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim, base)


def test_rejects_flat_positions():
    with pytest.raises(ValueError):
        RotaryEmbedding(64)(torch.arange(8), torch.float32)
