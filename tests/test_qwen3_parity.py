import pytest
import torch

from scout.layers import RMSNorm
from scout.rotary import RotaryEmbedding, apply_rotary

qwen3 = pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
Qwen3Config = pytest.importorskip("transformers").Qwen3Config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_matches_transformers(dtype):
    torch.manual_seed(0)
    ours = RMSNorm(896).to(dtype)
    theirs = qwen3.Qwen3RMSNorm(896).to(dtype)
    weight = torch.randn(896).to(dtype)
    with torch.no_grad():
        ours.weight.copy_(weight)
        theirs.weight.copy_(weight)
    x = torch.randn(4, 64, 896).to(dtype)
    assert torch.equal(ours(x), theirs(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_matches_transformers(dtype):
    torch.manual_seed(1)
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        head_dim=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    theirs = qwen3.Qwen3RotaryEmbedding(config)
    ours = RotaryEmbedding(64, 10000.0)
    positions = torch.stack((torch.arange(2048), torch.arange(2048) + 3))
    q = torch.randn(2, 14, 2048, 64).to(dtype)
    k = torch.randn(2, 2, 2048, 64).to(dtype)
    their_cos, their_sin = theirs(q, positions)
    our_cos, our_sin = ours(positions, dtype)
    their_q, their_k = qwen3.apply_rotary_pos_emb(q, k, their_cos, their_sin)
    assert torch.equal(our_cos, their_cos)
    assert torch.equal(our_sin, their_sin)
    assert torch.equal(apply_rotary(q, our_cos, our_sin), their_q)
    assert torch.equal(apply_rotary(k, our_cos, our_sin), their_k)
