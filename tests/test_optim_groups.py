import pytest
import torch

from scout.config import ModelConfig
from scout.loss import IGNORE_INDEX
from scout.model import Scout
from scout.optim.adamw import CautiousAdamW
from scout.optim.groups import build_optimizers, split_parameters
from scout.optim.normuon import NorMuon

TINY = ModelConfig(
    vocab_size=64,
    reserved_slots=8,
    d_model=64,
    n_layers=2,
    n_query_heads=4,
    n_kv_heads=2,
    head_dim=16,
    ffn_hidden=128,
    context=32,
)
MATRIX_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def names(model, parameters):
    wanted = {id(p) for p in parameters}
    return [name for name, p in model.named_parameters() if id(p) in wanted]


def test_every_parameter_lands_in_exactly_one_group_at_full_size(full_model):
    matrices, others = split_parameters(full_model)
    everything = list(full_model.parameters())
    assert len(matrices) + len(others) == len(everything)
    assert {id(p) for p in matrices} | {id(p) for p in others} == {id(p) for p in everything}
    assert not {id(p) for p in matrices} & {id(p) for p in others}
    assert sum(p.numel() for p in matrices + others) == full_model.config.parameter_count


def test_matrices_are_exactly_the_seven_projections_of_every_layer(full_model):
    matrices, others = split_parameters(full_model)
    assert len(matrices) == 7 * 26
    assert all(name.split(".")[-2] in MATRIX_NAMES for name in names(full_model, matrices))
    assert all(p.dim() == 2 for p in matrices)
    assert sum(p.numel() for p in matrices) == 26 * (2 * 896 * 896 + 2 * 896 * 128 + 3 * 896 * 2432)


def test_embedding_and_norms_go_to_adamw_and_the_tied_head_is_not_duplicated(full_model):
    _, others = split_parameters(full_model)
    other_names = names(full_model, others)
    assert len(others) == 1 + 26 * 4 + 1
    assert other_names[0] == "model.embed_tokens.weight"
    assert sum("norm" in name for name in other_names) == 26 * 4 + 1
    assert "lm_head.weight" not in other_names


def test_an_untied_head_goes_to_adamw():
    model = Scout(ModelConfig(vocab_size=64, reserved_slots=8, n_layers=2, tie_embeddings=False))
    _, others = split_parameters(model)
    assert {"model.embed_tokens.weight", "lm_head.weight"} <= set(names(model, others))


def test_builds_normuon_for_matrices_and_cautious_adamw_for_the_rest():
    model = Scout(TINY)
    muon, adam = build_optimizers(model, matrix_lr=0.02, other_lr=0.003, weight_decay=0.1)
    assert isinstance(muon, NorMuon) and isinstance(adam, CautiousAdamW)
    assert (muon.param_groups[0]["lr"], adam.param_groups[0]["lr"]) == (0.02, 0.003)
    assert (muon.param_groups[0]["weight_decay"], adam.param_groups[0]["weight_decay"]) == (0.1, 0.0)
    assert adam.param_groups[0]["cautious"] is True
    assert build_optimizers(model, 0.02, 0.003, cautious=False)[1].param_groups[0]["cautious"] is False
    assert build_optimizers(model, 0.02, 0.003)[0].param_groups[0]["weight_decay"] == 0.1


def test_one_step_updates_every_parameter_and_only_through_its_own_optimizer():
    torch.manual_seed(0)
    model = Scout(TINY)
    optimizers = build_optimizers(model, matrix_lr=0.02, other_lr=0.01)
    ids = torch.randint(0, 64, (2, 16))
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = IGNORE_INDEX
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.loss(ids, targets).backward()
    for optimizer in optimizers:
        optimizer.step()
    assert all(not torch.equal(p, before[name]) for name, p in model.named_parameters())
    assert len(optimizers[0].state) == 7 * 2
    assert len(optimizers[1].state) == 1 + 2 * 4 + 1


def test_trains_the_tiny_model_to_overfit_a_batch():
    torch.manual_seed(0)
    model = Scout(TINY)
    optimizers = build_optimizers(model, matrix_lr=0.01, other_lr=0.01, weight_decay=0.0)
    ids = torch.randint(0, 64, (4, 32), generator=torch.Generator().manual_seed(100))
    ids[:, 0] = torch.tensor([1, 2, 3, 4])
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = IGNORE_INDEX
    for _ in range(200):
        loss = model.loss(ids, targets)
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
    assert loss.item() < 0.05


def test_weight_decay_shrinks_only_the_matrices():
    model = Scout(TINY)
    embedding, norm = model.model.embed_tokens.weight.detach().clone(), model.model.norm.weight.detach().clone()
    layer = model.model.layers[0].mlp.down_proj.weight.detach().clone()
    optimizers = build_optimizers(model, matrix_lr=0.1, other_lr=0.1, weight_decay=0.5)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    for optimizer in optimizers:
        optimizer.step()
    assert torch.allclose(model.model.layers[0].mlp.down_proj.weight, layer * 0.95)
    assert torch.equal(model.model.embed_tokens.weight, embedding)
    assert torch.equal(model.model.norm.weight, norm)


@pytest.mark.parametrize("bad", [{"matrix_lr": -1.0, "other_lr": 0.1}, {"matrix_lr": 0.1, "other_lr": -1.0}])
def test_rejects_negative_learning_rates(bad):
    with pytest.raises(ValueError):
        build_optimizers(Scout(TINY), **bad)
