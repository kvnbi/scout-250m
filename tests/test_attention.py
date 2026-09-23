import pytest
import torch

from scout.attention import Attention
from scout.config import ModelConfig
from scout.rotary import RotaryEmbedding, apply_rotary


def make(n_kv_heads=2, qk_norm=True, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    attention = Attention(896, 14, n_kv_heads, 64, qk_norm=qk_norm).to(dtype)
    if qk_norm:
        with torch.no_grad():
            attention.q_norm.weight.copy_(1.0 + 0.1 * torch.randn(64))
            attention.k_norm.weight.copy_(1.0 + 0.1 * torch.randn(64))
    return attention


def cos_sin(batch, seq, dtype=torch.float32, offset=0):
    positions = torch.arange(offset, offset + seq).repeat(batch, 1)
    return RotaryEmbedding(64)(positions, dtype)


def reference(attention, x, cos, sin):
    batch, seq, _ = x.shape
    groups = attention.n_query_heads // attention.n_kv_heads
    q = attention.q_norm(attention.q_proj(x).view(batch, seq, -1, 64)).transpose(1, 2)
    k = attention.k_norm(attention.k_proj(x).view(batch, seq, -1, 64)).transpose(1, 2)
    v = attention.v_proj(x).view(batch, seq, -1, 64).transpose(1, 2)
    q = apply_rotary(q, cos, sin)
    k = apply_rotary(k, cos, sin).repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = (q @ k.transpose(-2, -1)) * 64**-0.5
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores.to(torch.promote_types(q.dtype, torch.float32)), dim=-1).to(q.dtype)
    out = (weights @ v).transpose(1, 2).reshape(batch, seq, -1)
    return attention.o_proj(out)


def test_parameter_names_and_no_biases():
    names = {name for name, _ in make().named_parameters()}
    assert names == {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
    }


@pytest.mark.parametrize("n_kv_heads", [1, 2, 7, 14])
def test_parameter_count_matches_config(n_kv_heads):
    config = ModelConfig(n_kv_heads=n_kv_heads)
    kv = n_kv_heads * 64
    expected = 2 * 896 * 896 + 2 * 896 * kv + 2 * 64
    assert sum(p.numel() for p in make(n_kv_heads).parameters()) == expected
    assert config.layer_parameters - 3 * 896 * 2432 - 2 * 896 == expected


def test_output_shape_and_dtype():
    attention = make()
    cos, sin = cos_sin(2, 16)
    y = attention(torch.randn(2, 16, 896), cos, sin)
    assert y.shape == (2, 16, 896)
    assert y.dtype == torch.float32


@pytest.mark.parametrize("n_kv_heads", [1, 2, 7, 14])
def test_matches_explicit_reference_in_float64(n_kv_heads):
    attention = make(n_kv_heads, dtype=torch.float64)
    x = torch.randn(2, 24, 896, dtype=torch.float64)
    cos, sin = cos_sin(2, 24, torch.float64)
    torch.testing.assert_close(attention(x, cos, sin), reference(attention, x, cos, sin), rtol=1e-10, atol=1e-10)


def test_matches_explicit_reference_in_float32():
    attention = make()
    x = torch.randn(2, 64, 896)
    cos, sin = cos_sin(2, 64)
    torch.testing.assert_close(attention(x, cos, sin), reference(attention, x, cos, sin), rtol=1e-4, atol=1e-5)


def test_without_qk_norm():
    attention = make(qk_norm=False, dtype=torch.float64)
    assert not any("norm" in name for name, _ in attention.named_parameters())
    x = torch.randn(1, 8, 896, dtype=torch.float64)
    cos, sin = cos_sin(1, 8, torch.float64)
    torch.testing.assert_close(attention(x, cos, sin), reference(attention, x, cos, sin), rtol=1e-10, atol=1e-10)


def test_is_causal():
    attention = make(dtype=torch.float64)
    x = torch.randn(1, 20, 896, dtype=torch.float64)
    changed = x.clone()
    changed[:, 12:] = torch.randn(1, 8, 896, dtype=torch.float64)
    cos, sin = cos_sin(1, 20, torch.float64)
    before = attention(x, cos, sin)
    after = attention(changed, cos, sin)
    torch.testing.assert_close(before[:, :12], after[:, :12], rtol=0.0, atol=1e-12)
    assert not torch.allclose(before[:, 12:], after[:, 12:])


def test_query_heads_share_kv_heads_in_contiguous_groups():
    attention = make(n_kv_heads=2, dtype=torch.float64)
    with torch.no_grad():
        attention.v_proj.weight[64:].zero_()
        attention.o_proj.weight.zero_()
        attention.o_proj.weight[:, :7 * 64] = torch.randn(896, 7 * 64, dtype=torch.float64)
    cos, sin = cos_sin(1, 6, torch.float64)
    assert attention(torch.randn(1, 6, 896, dtype=torch.float64), cos, sin).abs().sum() > 0
    with torch.no_grad():
        attention.o_proj.weight.zero_()
        attention.o_proj.weight[:, 7 * 64:] = torch.randn(896, 7 * 64, dtype=torch.float64)
    assert attention(torch.randn(1, 6, 896, dtype=torch.float64), cos, sin).abs().max() == 0


def test_bfloat16_close_to_float32():
    attention = make()
    x = torch.randn(2, 32, 896)
    cos, sin = cos_sin(2, 32)
    expected = attention(x, cos, sin)
    low = attention.to(torch.bfloat16)
    cos16, sin16 = cos_sin(2, 32, torch.bfloat16)
    y = low(x.bfloat16(), cos16, sin16)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y.float(), expected, rtol=0.1, atol=0.05)


def test_gradients_are_finite():
    attention = make()
    x = torch.randn(2, 16, 896, requires_grad=True)
    cos, sin = cos_sin(2, 16)
    attention(x, cos, sin).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize(
    "args",
    [(896, 14, 3, 64), (896, 14, 2, 63), (0, 14, 2, 64), (896, 14, 0, 64)],
)
def test_rejects_invalid_arguments(args):
    with pytest.raises(ValueError):
        Attention(*args)


def test_rejects_wrong_input_shape():
    attention = make()
    cos, sin = cos_sin(2, 8)
    with pytest.raises(ValueError):
        attention(torch.randn(2, 8, 512), cos, sin)


def test_output_depends_only_on_relative_positions():
    attention = make(dtype=torch.float64)
    x = torch.randn(2, 32, 896, dtype=torch.float64)
    near = attention(x, *cos_sin(2, 32, torch.float64))
    far = attention(x, *cos_sin(2, 32, torch.float64, offset=1000))
    torch.testing.assert_close(near, far, rtol=1e-5, atol=1e-5)


def test_accepts_one_cos_sin_row_for_the_whole_batch():
    attention = make(dtype=torch.float64)
    x = torch.randn(3, 10, 896, dtype=torch.float64)
    shared = attention(x, *cos_sin(1, 10, torch.float64))
    expanded = attention(x, *cos_sin(3, 10, torch.float64))
    assert torch.equal(shared, expanded)


@pytest.mark.parametrize("batch, seq", [(2, 1), (2, 7), (3, 8)])
def test_rejects_cos_sin_of_the_wrong_shape(batch, seq):
    attention = make()
    with pytest.raises(ValueError):
        attention(torch.randn(2, 8, 896), *cos_sin(batch, seq))


def test_rejects_cos_sin_of_the_wrong_dtype():
    attention = make().to(torch.bfloat16)
    with pytest.raises(TypeError):
        attention(torch.randn(2, 8, 896).bfloat16(), *cos_sin(2, 8, torch.float32))
