import pytest
import torch

from scout.block import Block
from scout.config import ModelConfig
from scout.layers import RMSNorm
from scout.rotary import RotaryEmbedding

QWEN3_LAYER_NAMES = {
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}


def make(n_kv_heads=2, dtype=torch.float32, seed=0, eps=1e-6):
    torch.manual_seed(seed)
    block = Block(896, 14, n_kv_heads, 64, 2432, eps=eps).to(dtype)
    with torch.no_grad():
        for norm in (block.input_layernorm, block.post_attention_layernorm):
            norm.weight.copy_(1.0 + 0.1 * torch.randn(896))
        for norm in (block.self_attn.q_norm, block.self_attn.k_norm):
            norm.weight.copy_(1.0 + 0.1 * torch.randn(64))
    return block


def cos_sin(batch, seq, dtype=torch.float32, offset=0):
    positions = torch.arange(offset, offset + seq).repeat(batch, 1)
    return RotaryEmbedding(64)(positions, dtype)


def test_parameter_names_match_qwen3():
    assert {name for name, _ in make().named_parameters()} == QWEN3_LAYER_NAMES


@pytest.mark.parametrize("n_kv_heads", [2, 7])
def test_parameter_count_matches_config(n_kv_heads):
    config = ModelConfig(n_kv_heads=n_kv_heads)
    assert sum(p.numel() for p in make(n_kv_heads).parameters()) == config.layer_parameters


def test_output_shape_and_dtype():
    y = make()(torch.randn(2, 16, 896), *cos_sin(2, 16))
    assert y.shape == (2, 16, 896)
    assert y.dtype == torch.float32


def test_matches_pre_norm_residual_structure():
    block = make(dtype=torch.float64)
    x = torch.randn(2, 12, 896, dtype=torch.float64)
    cos, sin = cos_sin(2, 12, torch.float64)
    h = x + block.self_attn(block.input_layernorm(x), cos, sin)
    expected = h + block.mlp(block.post_attention_layernorm(h))
    assert torch.equal(block(x, cos, sin), expected)


def test_is_identity_when_output_projections_are_zero():
    block = make(dtype=torch.float64)
    with torch.no_grad():
        block.self_attn.o_proj.weight.zero_()
        block.mlp.down_proj.weight.zero_()
    x = torch.randn(2, 10, 896, dtype=torch.float64)
    assert torch.equal(block(x, *cos_sin(2, 10, torch.float64)), x)


def test_attention_sees_normalised_input():
    block = make(dtype=torch.float64, eps=1e-30)
    with torch.no_grad():
        block.mlp.down_proj.weight.zero_()
    x = torch.randn(1, 8, 896, dtype=torch.float64)
    cos, sin = cos_sin(1, 8, torch.float64)
    torch.testing.assert_close(block(x * 50.0, cos, sin) - x * 50.0, block(x, cos, sin) - x, rtol=1e-10, atol=1e-10)


def test_mlp_sees_normalised_input():
    block = make(dtype=torch.float64, eps=1e-30)
    with torch.no_grad():
        block.self_attn.o_proj.weight.zero_()
    x = torch.randn(1, 8, 896, dtype=torch.float64)
    cos, sin = cos_sin(1, 8, torch.float64)
    torch.testing.assert_close(block(x * 50.0, cos, sin) - x * 50.0, block(x, cos, sin) - x, rtol=1e-10, atol=1e-10)


def test_is_causal():
    block = make(dtype=torch.float64)
    x = torch.randn(1, 20, 896, dtype=torch.float64)
    changed = x.clone()
    changed[:, 12:] = torch.randn(1, 8, 896, dtype=torch.float64)
    cos, sin = cos_sin(1, 20, torch.float64)
    before, after = block(x, cos, sin), block(changed, cos, sin)
    torch.testing.assert_close(before[:, :12], after[:, :12], rtol=0.0, atol=1e-12)
    assert not torch.allclose(before[:, 12:], after[:, 12:])


def test_output_depends_only_on_relative_positions():
    block = make(dtype=torch.float64)
    x = torch.randn(2, 32, 896, dtype=torch.float64)
    near = block(x, *cos_sin(2, 32, torch.float64))
    far = block(x, *cos_sin(2, 32, torch.float64, offset=1000))
    torch.testing.assert_close(near, far, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_bfloat16_error_stays_within_rounding(seed):
    block = make(seed=seed)
    x = torch.randn(2, 32, 896)
    expected = block.double()(x.double(), *cos_sin(2, 32, torch.float64))
    y = block.to(torch.bfloat16)(x.bfloat16(), *cos_sin(2, 32, torch.bfloat16))
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y.double(), expected, rtol=0.02, atol=0.02)


def test_gradients_are_finite():
    block = make()
    x = torch.randn(2, 16, 896, requires_grad=True)
    block(x, *cos_sin(2, 16)).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_passes_gradcheck():
    torch.manual_seed(1)
    block = Block(16, 4, 2, 8, 24).double()
    x = torch.randn(1, 5, 16, dtype=torch.float64, requires_grad=True)
    positions = torch.arange(5)[None]
    cos, sin = RotaryEmbedding(8)(positions, torch.float64)
    assert torch.autograd.gradcheck(lambda t: block(t, cos, sin), (x,))


def test_eps_reaches_every_norm():
    block = Block(896, 14, 2, 64, 2432, eps=3e-5)
    norms = [module for module in block.modules() if isinstance(module, RMSNorm)]
    assert len(norms) == 4
    assert all(norm.eps == 3e-5 for norm in norms)


def test_without_qk_norm_matches_config():
    block = Block(896, 14, 2, 64, 2432, qk_norm=False)
    config = ModelConfig(qk_norm=False)
    assert sum(p.numel() for p in block.parameters()) == config.layer_parameters
    assert "self_attn.q_norm.weight" not in dict(block.named_parameters())


def test_train_and_eval_modes_agree():
    block = make()
    x = torch.randn(2, 16, 896)
    cos, sin = cos_sin(2, 16)
    block.train()
    first, second = block(x, cos, sin), block(x, cos, sin)
    block.eval()
    assert torch.equal(first, second)
    assert torch.equal(first, block(x, cos, sin))


def test_residual_stream_stays_float32_under_autocast():
    block = make()
    x = torch.randn(2, 16, 896)
    cos, sin = cos_sin(2, 16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = block(x, cos, sin)
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, block(x, cos, sin), rtol=0.02, atol=0.02)


def test_rejects_wrong_input_width():
    with pytest.raises(ValueError):
        make()(torch.randn(2, 8, 512), *cos_sin(2, 8))
