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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_kv_heads", [2, 7])
def test_attention_matches_transformers(dtype, n_kv_heads):
    from scout.attention import Attention

    torch.manual_seed(2)
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=n_kv_heads,
        head_dim=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "sdpa"
    ours = Attention(896, 14, n_kv_heads, 64).to(dtype)
    with torch.no_grad():
        ours.q_norm.weight.copy_(1.0 + 0.1 * torch.randn(64))
        ours.k_norm.weight.copy_(1.0 + 0.1 * torch.randn(64))
    theirs = qwen3.Qwen3Attention(config, layer_idx=0).to(dtype)
    theirs.load_state_dict(ours.state_dict())
    positions = torch.arange(256).repeat(2, 1)
    x = torch.randn(2, 256, 896).to(dtype)
    cos, sin = RotaryEmbedding(64)(positions, dtype)
    their_out, _ = theirs(x, (cos, sin), attention_mask=None)
    tolerance = {torch.float32: 1e-5, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(ours(x, cos, sin), their_out, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mlp_matches_transformers(dtype):
    from scout.mlp import SwiGLU

    torch.manual_seed(3)
    config = Qwen3Config(hidden_size=896, intermediate_size=2432, hidden_act="silu")
    ours = SwiGLU(896, 2432).to(dtype)
    theirs = qwen3.Qwen3MLP(config).to(dtype)
    theirs.load_state_dict(ours.state_dict())
    x = torch.randn(2, 64, 896).to(dtype)
    assert torch.equal(ours(x), theirs(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_kv_heads", [2, 7])
def test_block_matches_transformers(dtype, n_kv_heads):
    from scout.block import Block

    torch.manual_seed(4)
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=2432,
        num_attention_heads=14,
        num_key_value_heads=n_kv_heads,
        head_dim=64,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "sdpa"
    ours = Block(896, 14, n_kv_heads, 64, 2432).to(dtype)
    with torch.no_grad():
        for parameter in ours.parameters():
            if parameter.dim() == 1:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
    theirs = qwen3.Qwen3DecoderLayer(config, layer_idx=0).to(dtype)
    theirs.load_state_dict(ours.state_dict())
    positions = torch.arange(128).repeat(2, 1)
    x = torch.randn(2, 128, 896).to(dtype)
    cos, sin = RotaryEmbedding(64)(positions, dtype)
    their_out = theirs(x, attention_mask=None, position_embeddings=(cos, sin))
    if isinstance(their_out, tuple):
        their_out = their_out[0]
    assert torch.equal(ours(x, cos, sin), their_out)
