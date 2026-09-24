import pytest
import torch

from scout.config import ModelConfig
from scout.layers import RMSNorm
from scout.model import Scout

SMALL = ModelConfig(vocab_size=512, n_layers=2)

LAYER_NAMES = (
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
)


def make(config=SMALL, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    model = Scout(config).to(dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 1:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
    return model


def tokens(batch, seq, vocab=512, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=generator)


def test_state_dict_names_match_qwen3():
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    expected |= {f"model.layers.{i}.{name}" for i in range(2) for name in LAYER_NAMES}
    assert set(make().state_dict()) == expected


def test_parameter_count_matches_config_at_full_size():
    model = Scout()
    assert sum(p.numel() for p in model.parameters()) == 247_088_768 == model.config.parameter_count


@pytest.mark.parametrize(
    "config",
    [
        SMALL,
        ModelConfig(vocab_size=512, n_layers=2, n_kv_heads=7),
        ModelConfig(vocab_size=512, n_layers=2, tie_embeddings=False),
    ],
)
def test_parameter_count_matches_config(config):
    assert sum(p.numel() for p in Scout(config).parameters()) == config.parameter_count


def test_embeddings_are_tied_and_stay_tied():
    model = make()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    for dtype in (torch.bfloat16, torch.float64):
        model = model.to(dtype)
        assert model.lm_head.weight is model.model.embed_tokens.weight
        assert model.lm_head.weight.dtype == dtype


def test_untied_embeddings_are_separate():
    model = Scout(ModelConfig(vocab_size=512, n_layers=2, tie_embeddings=False))
    assert model.lm_head.weight is not model.model.embed_tokens.weight


def test_tied_weight_learns_from_input_and_output():
    model = make()
    ids = torch.zeros(1, 4, dtype=torch.long)
    model(ids)[0, -1, 7].backward()
    grad = model.lm_head.weight.grad
    assert grad[0].abs().sum() > 0
    assert grad[7].abs().sum() > 0
    assert grad[100].abs().sum() == 0


def test_logits_shape_and_dtype():
    logits = make()(tokens(2, 16))
    assert logits.shape == (2, 16, 512)
    assert logits.dtype == torch.float32


def test_matches_manual_composition():
    model = make(dtype=torch.float64)
    ids = tokens(2, 12)
    positions = torch.arange(12).unsqueeze(0)
    h = model.model.embed_tokens(ids)
    cos, sin = model.model.rotary_emb(positions, torch.float64)
    for layer in model.model.layers:
        h = layer(h, cos, sin)
    expected = model.lm_head(model.model.norm(h))
    assert torch.equal(model(ids), expected)


def test_default_positions_match_explicit_positions():
    model = make()
    ids = tokens(2, 10)
    explicit = model(ids, torch.arange(10).repeat(2, 1))
    assert torch.equal(model(ids), explicit)


def test_shared_positions_row_matches_per_row_positions():
    model = make()
    ids = tokens(3, 10)
    shared = model(ids, torch.arange(5, 15).unsqueeze(0))
    per_row = model(ids, torch.arange(5, 15).repeat(3, 1))
    assert torch.equal(shared, per_row)


def test_is_causal():
    model = make(dtype=torch.float64)
    ids = tokens(1, 20)
    changed = ids.clone()
    changed[:, 12:] = tokens(1, 8, seed=9)
    before, after = model(ids), model(changed)
    torch.testing.assert_close(before[:, :12], after[:, :12], rtol=0.0, atol=1e-10)
    assert not torch.allclose(before[:, 12:], after[:, 12:])


def test_logits_depend_only_on_relative_positions():
    model = make(dtype=torch.float64)
    ids = tokens(2, 24)
    near = model(ids)
    far = model(ids, torch.arange(1000, 1024).unsqueeze(0))
    scale = near.abs().max().item()
    torch.testing.assert_close(near, far, rtol=0.0, atol=1e-6 * scale)


def test_config_values_reach_every_module():
    config = ModelConfig(vocab_size=512, n_layers=3, norm_eps=3e-5, rope_theta=50000.0)
    model = Scout(config)
    norms = [module for module in model.modules() if isinstance(module, RMSNorm)]
    assert len(norms) == 4 * 3 + 1
    assert all(norm.eps == 3e-5 for norm in norms)
    assert model.model.rotary_emb.base == 50000.0
    assert len(model.model.layers) == 3


def test_default_config_is_used_when_none_given():
    assert Scout().config == ModelConfig()


def test_bfloat16_runs_and_stays_finite():
    logits = make().to(torch.bfloat16)(tokens(2, 16))
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()


def test_autocast_forward_is_finite():
    model = make()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(tokens(2, 16))
    assert torch.isfinite(logits).all()


def test_every_parameter_receives_a_finite_gradient():
    model = make()
    model(tokens(2, 16)).float().logsumexp(-1).mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("ids", [torch.zeros(8, dtype=torch.long), torch.zeros(1, 2, 3, dtype=torch.long)])
def test_rejects_input_ids_of_the_wrong_rank(ids):
    with pytest.raises(ValueError):
        make()(ids)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bool])
def test_rejects_non_integer_input_ids(dtype):
    with pytest.raises(TypeError):
        make()(torch.zeros(1, 4, dtype=dtype))


@pytest.mark.parametrize("positions", [torch.arange(4), torch.arange(5).unsqueeze(0), torch.arange(4).repeat(3, 1)])
def test_rejects_positions_of_the_wrong_shape(positions):
    with pytest.raises(ValueError):
        make()(tokens(2, 4), positions)


def test_rejects_out_of_range_token_ids():
    with pytest.raises(IndexError):
        make()(torch.full((1, 4), 512, dtype=torch.long))
