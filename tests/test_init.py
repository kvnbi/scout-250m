import math

import pytest
import torch
import torch.nn.functional as F

from scout.config import ModelConfig
from scout.model import Scout

TRUNCATED_STD_FACTOR = 0.98658


@pytest.fixture
def full(full_model):
    return full_model


def measured_std(weight):
    return weight.detach().double().std().item()


@pytest.mark.parametrize("name", ["q_proj", "k_proj", "v_proj"])
def test_attention_input_projections_use_base_std(full, name):
    weight = getattr(full.model.layers[5].self_attn, name).weight
    assert measured_std(weight) == pytest.approx(0.02 * TRUNCATED_STD_FACTOR, rel=0.03)


@pytest.mark.parametrize("name", ["gate_proj", "up_proj"])
def test_mlp_input_projections_use_base_std(full, name):
    weight = getattr(full.model.layers[5].mlp, name).weight
    assert measured_std(weight) == pytest.approx(0.02 * TRUNCATED_STD_FACTOR, rel=0.03)


@pytest.mark.parametrize("path", ["self_attn.o_proj", "mlp.down_proj"])
def test_residual_output_projections_are_scaled_by_depth(full, path):
    weight = full.model.layers[5].get_submodule(path).weight
    expected = 0.02 / math.sqrt(2 * 26) * TRUNCATED_STD_FACTOR
    assert measured_std(weight) == pytest.approx(expected, rel=0.03)


def test_embeddings_use_base_std_and_stay_tied(full):
    assert full.lm_head.weight is full.model.embed_tokens.weight
    assert measured_std(full.model.embed_tokens.weight) == pytest.approx(0.02 * TRUNCATED_STD_FACTOR, rel=0.03)


def test_every_weight_is_truncated_at_three_std(full):
    residual_std = 0.02 / math.sqrt(2 * 26)
    for name, parameter in full.named_parameters():
        if parameter.dim() == 2:
            limit = 3 * (residual_std if name.endswith(("o_proj.weight", "down_proj.weight")) else 0.02)
            assert parameter.abs().max().item() <= limit, name


def test_every_norm_starts_at_one(full):
    for name, parameter in full.named_parameters():
        if parameter.dim() == 1:
            assert torch.equal(parameter, torch.ones_like(parameter)), name


def test_weights_have_zero_mean(full):
    for name, parameter in full.named_parameters():
        if parameter.dim() == 2:
            values = parameter.detach().double()
            standard_error = values.std().item() / math.sqrt(values.numel())
            assert abs(values.mean().item()) < 5 * standard_error, name


def test_untied_head_is_initialised_separately():
    torch.manual_seed(0)
    model = Scout(ModelConfig(vocab_size=4096, n_layers=2, tie_embeddings=False))
    head = model.lm_head.weight
    assert measured_std(head) == pytest.approx(0.02 * TRUNCATED_STD_FACTOR, rel=0.03)
    assert not torch.equal(head, model.model.embed_tokens.weight)


def test_init_std_is_configurable():
    torch.manual_seed(0)
    model = Scout(ModelConfig(vocab_size=4096, n_layers=2, init_std=0.01))
    assert measured_std(model.model.layers[0].self_attn.q_proj.weight) == pytest.approx(0.01 * TRUNCATED_STD_FACTOR, rel=0.03)
    assert measured_std(model.model.layers[0].mlp.down_proj.weight) == pytest.approx(0.01 / 2 * TRUNCATED_STD_FACTOR, rel=0.03)


def test_plain_init_skips_depth_scaling():
    torch.manual_seed(0)
    model = Scout(ModelConfig(vocab_size=4096, n_layers=4, depth_scaled_init=False))
    for path in ("self_attn.o_proj", "mlp.down_proj"):
        weight = model.model.layers[1].get_submodule(path).weight
        assert measured_std(weight) == pytest.approx(0.02 * TRUNCATED_STD_FACTOR, rel=0.03), path


def test_same_seed_gives_identical_weights():
    config = ModelConfig(vocab_size=1024, n_layers=2)
    torch.manual_seed(7)
    first = Scout(config).state_dict()
    torch.manual_seed(7)
    second = Scout(config).state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_different_seeds_give_different_weights():
    config = ModelConfig(vocab_size=1024, n_layers=2)
    torch.manual_seed(7)
    first = Scout(config).model.layers[0].self_attn.q_proj.weight
    torch.manual_seed(8)
    second = Scout(config).model.layers[0].self_attn.q_proj.weight
    assert not torch.equal(first, second)


def test_reset_parameters_reinitialises_everything():
    torch.manual_seed(3)
    model = Scout(ModelConfig(vocab_size=1024, n_layers=2))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(5.0)
    model.reset_parameters()
    for name, parameter in model.named_parameters():
        if parameter.dim() == 1:
            assert torch.equal(parameter, torch.ones_like(parameter)), name
        else:
            assert parameter.abs().max().item() <= 0.06, name


def test_initial_loss_is_close_to_uniform():
    torch.manual_seed(0)
    model = Scout(ModelConfig(n_layers=2))
    ids = torch.randint(0, 32768, (2, 128))
    with torch.no_grad():
        logits = model(ids)
    loss = F.cross_entropy(logits.flatten(0, 1), torch.roll(ids, -1, 1).flatten()).item()
    assert math.log(32768) <= loss <= math.log(32768) + 0.4


def test_residual_stream_stays_small_through_depth(full):
    ids = torch.randint(0, 32768, (1, 128), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        h = full.model.embed_tokens(ids)
        cos, sin = full.model.rotary_emb(torch.arange(128)[None], h.dtype)
        for layer in full.model.layers:
            h = layer(h, cos, sin)
    assert h.pow(2).mean().sqrt().item() < 0.5
