from dataclasses import asdict

import pytest
import torch

from scout.config import PROXY_CONFIG, ModelConfig
from scout.generate import generate
from scout.loss import IGNORE_INDEX
from scout.model import Scout
from scout.optim.groups import build_optimizers, split_parameters
from scout.train import Batch, TrainSettings, train

HALVED = ("d_model", "n_query_heads", "n_kv_heads", "ffn_hidden")


def test_parameter_counts():
    assert PROXY_CONFIG.parameter_count == 69_126_592
    assert PROXY_CONFIG.embedding_parameters == 14_680_064
    assert PROXY_CONFIG.non_embedding_parameters == 54_446_528


def test_the_proxy_is_the_full_model_at_exactly_half_the_width():
    full, proxy = asdict(ModelConfig()), asdict(PROXY_CONFIG)
    for name in HALVED:
        assert proxy[name] * 2 == full[name], name
    assert {k: v for k, v in proxy.items() if k not in HALVED} == {k: v for k, v in full.items() if k not in HALVED}


def test_attention_grouping_and_feed_forward_ratio_are_preserved():
    full = ModelConfig()
    assert PROXY_CONFIG.n_query_heads // PROXY_CONFIG.n_kv_heads == full.n_query_heads // full.n_kv_heads == 7
    assert PROXY_CONFIG.ffn_hidden / PROXY_CONFIG.d_model == full.ffn_hidden / full.d_model
    assert PROXY_CONFIG.head_dim == full.head_dim and PROXY_CONFIG.n_layers == full.n_layers


def test_non_embedding_parameters_are_a_quarter_of_the_full_models():
    full = ModelConfig()
    ratio = PROXY_CONFIG.non_embedding_parameters / full.non_embedding_parameters
    assert ratio == pytest.approx(0.25, abs=1e-3)


@pytest.fixture(scope="module")
def proxy():
    torch.manual_seed(0)
    return Scout(PROXY_CONFIG)


def test_built_model_matches_the_config_and_the_full_models_structure(proxy):
    assert sum(p.numel() for p in proxy.parameters()) == PROXY_CONFIG.parameter_count
    with torch.device("meta"):
        full = Scout(ModelConfig())
    assert list(proxy.state_dict()) == list(full.state_dict())
    for name, value in full.state_dict().items():
        assert proxy.state_dict()[name].dim() == value.dim(), name
    matrices, others = split_parameters(proxy)
    full_matrices, full_others = split_parameters(full)
    assert (len(matrices), len(others)) == (len(full_matrices), len(full_others)) == (182, 106)


def test_forward_and_backward_run_with_a_single_key_value_head(proxy):
    torch.manual_seed(1)
    ids = torch.randint(0, PROXY_CONFIG.reserved_token_ids.start, (2, 48))
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = IGNORE_INDEX
    proxy.zero_grad()
    totals = proxy.loss_totals(ids, targets)
    (totals.loss_sum / totals.tokens).backward()
    assert torch.isfinite(totals.loss_sum)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in proxy.parameters())
    assert totals.cross_entropy_sum / totals.tokens == pytest.approx(torch.log(torch.tensor(32768.0)).item(), abs=0.5)
    proxy.zero_grad(set_to_none=True)


def test_cached_generation_equals_recomputing_the_whole_sequence(proxy):
    torch.manual_seed(2)
    prompt = torch.randint(0, PROXY_CONFIG.reserved_token_ids.start, (2, 12))
    new = generate(proxy, prompt, 10)
    sequence = prompt
    with torch.no_grad():
        for _ in range(10):
            sequence = torch.cat([sequence, proxy(sequence)[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(new, sequence[:, 12:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_proxy_matches_transformers_qwen3_bit_for_bit(dtype):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
    torch.manual_seed(3)
    ours = Scout(PROXY_CONFIG).to(dtype)
    with torch.no_grad():
        for parameter in ours.parameters():
            if parameter.dim() == 1:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
    their_config = transformers.Qwen3Config(**PROXY_CONFIG.qwen3_config())
    their_config._attn_implementation = "sdpa"
    theirs = transformers.AutoModelForCausalLM.from_config(their_config, dtype=dtype)
    theirs.load_state_dict(ours.state_dict())
    ids = torch.randint(0, PROXY_CONFIG.reserved_token_ids.start, (2, 64))
    with torch.no_grad():
        assert torch.equal(ours(ids), theirs(input_ids=ids).logits)


def test_a_few_optimizer_steps_at_the_proxy_shape_lower_the_loss_and_stay_finite():
    torch.manual_seed(4)
    model = Scout(PROXY_CONFIG)
    ids = torch.randint(0, PROXY_CONFIG.reserved_token_ids.start, (2, 32))
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = IGNORE_INDEX
    losses = []

    def batches():
        while True:
            yield Batch(ids, targets)

    train(
        model,
        build_optimizers(model, 0.02, 0.004),
        batches(),
        TrainSettings(total_steps=6, warmup_steps=1, decay_steps=2),
        on_step=lambda step, metrics: losses.append(metrics["loss"]),
    )
    assert all(loss == loss and abs(loss) < 1e3 for loss in losses)
    assert losses[-1] < losses[0] - 0.3
    assert all(torch.isfinite(p).all() for p in model.parameters())
