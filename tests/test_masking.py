import pytest
import torch
from conftest import unchanged

from scout.attention import Attention
from scout.cache import KVCache
from scout.config import ModelConfig
from scout.loss import IGNORE_INDEX
from scout.masking import document_ids, document_mask
from scout.model import Scout
from scout.rotary import RotaryEmbedding

SMALL = ModelConfig(vocab_size=512, n_layers=2)
EOT = 480


def make_model(dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    model = Scout(SMALL).to(dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 1:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
    return model


@pytest.fixture(scope="module")
def model():
    yield from unchanged(make_model())


def tokens(batch, seq, seed=1):
    return torch.randint(0, 400, (batch, seq), generator=torch.Generator().manual_seed(seed))


def test_document_ids_count_the_end_tokens_before_each_position():
    ids = torch.tensor([[5, 6, EOT, 7, EOT, EOT, 8], [1, 2, 3, 4, 5, 6, 7], [EOT, 1, 2, EOT, 3, 4, 5]])
    assert document_ids(ids, EOT).tolist() == [[0, 0, 0, 1, 1, 2, 3], [0] * 7, [0, 1, 1, 1, 2, 2, 2]]


def test_document_ids_reject_the_wrong_rank():
    with pytest.raises(ValueError):
        document_ids(torch.zeros(5, dtype=torch.long), EOT)


def test_document_mask_is_causal_and_block_diagonal():
    ids = torch.tensor([[0, 0, 0, 1, 1, 2], [0, 0, 0, 0, 0, 0]])
    mask = document_mask(ids)
    assert mask.shape == (2, 6, 6) and mask.dtype == torch.bool
    for b in range(2):
        for i in range(6):
            for j in range(6):
                assert mask[b, i, j] == (j <= i and ids[b, i] == ids[b, j])
    assert mask[:, range(6), range(6)].all()
    assert torch.equal(mask[1], torch.ones(6, 6, dtype=torch.bool).tril())


@pytest.mark.parametrize("bad", [torch.zeros(5, dtype=torch.long), torch.zeros(2, 3, dtype=torch.float32)])
def test_document_mask_rejects_bad_ids(bad):
    with pytest.raises((ValueError, TypeError)):
        document_mask(bad)


def packed(lengths, seed=1):
    pieces = []
    for index, length in enumerate(lengths):
        pieces.append(tokens(1, length - 1, seed + index))
        pieces.append(torch.tensor([[EOT]]))
    return torch.cat(pieces, dim=1)


def restarting_positions(ids):
    positions = torch.zeros_like(ids)
    for row in range(ids.shape[0]):
        start = 0
        for index in range(ids.shape[1]):
            positions[row, index] = index - start
            if ids[row, index] == EOT:
                start = index + 1
    return positions


def test_a_packed_batch_equals_every_document_run_alone(model):
    lengths = (9, 14, 6)
    ids = packed(lengths)
    doc = document_ids(ids, EOT)
    logits = model(ids, restarting_positions(ids), doc)
    start = 0
    for length in lengths:
        alone = model(ids[:, start : start + length])
        torch.testing.assert_close(logits[:, start : start + length], alone, rtol=1e-9, atol=1e-9)
        start += length


def test_default_positions_give_the_same_answers_up_to_rotary_precision(model):
    lengths = (11, 17)
    ids = packed(lengths)
    logits = model(ids, document_ids=document_ids(ids, EOT))
    scale = logits.abs().max().item()
    start = 0
    for length in lengths:
        torch.testing.assert_close(
            logits[:, start : start + length], model(ids[:, start : start + length]), rtol=0.0, atol=1e-4 * scale
        )
        start += length


def test_documents_cannot_see_earlier_documents(model):
    ids = packed((8, 8))
    doc = document_ids(ids, EOT)
    changed = ids.clone()
    changed[:, :8] = tokens(1, 8, seed=50)
    before, after = model(ids, restarting_positions(ids), doc), model(changed, restarting_positions(ids), doc)
    torch.testing.assert_close(before[:, 8:], after[:, 8:], rtol=0.0, atol=1e-10)
    assert not torch.allclose(before[:, :8], after[:, :8])
    unmasked_before, unmasked_after = model(ids), model(changed)
    assert not torch.allclose(unmasked_before[:, 8:], unmasked_after[:, 8:])


def test_one_document_per_row_is_the_ordinary_causal_model(model):
    ids = tokens(3, 20)
    zeros = torch.zeros(3, 20, dtype=torch.long)
    torch.testing.assert_close(model(ids, document_ids=zeros), model(ids), rtol=1e-10, atol=1e-10)


def test_rows_with_different_document_layouts_in_one_batch(model):
    ids = torch.cat((packed((6, 10)), packed((4, 4, 8))))
    doc = document_ids(ids, EOT)
    logits = model(ids, restarting_positions(ids), doc)
    torch.testing.assert_close(logits[0:1, 6:16], model(ids[0:1, 6:16]), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(logits[1:2, 4:8], model(ids[1:2, 4:8]), rtol=1e-9, atol=1e-9)


def test_loss_and_totals_accept_document_ids_and_match_separate_documents(model):
    lengths = (10, 12)
    ids = packed(lengths)
    positions = restarting_positions(ids)
    doc = document_ids(ids, EOT)
    targets = torch.roll(ids, -1, 1)
    targets[:, 9] = IGNORE_INDEX
    targets[:, -1] = IGNORE_INDEX
    totals = model.loss_totals(ids, targets, positions, document_ids=doc)
    separate = [model.loss_totals(ids[:, a:b], targets[:, a:b]) for a, b in ((0, 10), (10, 22))]
    torch.testing.assert_close(totals.loss_sum, sum(t.loss_sum for t in separate), rtol=1e-9, atol=1e-9)
    assert totals.tokens == sum(t.tokens for t in separate)
    torch.testing.assert_close(
        model.loss(ids, targets, positions, document_ids=doc), totals.loss_sum / totals.tokens, rtol=1e-12, atol=1e-12
    )


def test_gradients_flow_and_stay_inside_each_document():
    model = make_model()
    ids = packed((8, 8))
    doc = document_ids(ids, EOT)
    embeddings = model.model.embed_tokens(ids).detach().requires_grad_(True)
    positions = restarting_positions(ids)
    cos, sin = model.model.rotary_emb(positions, embeddings.dtype)
    mask = document_mask(doc)
    h = embeddings
    for layer in model.model.layers:
        h = layer(h, cos, sin, None, mask)
    h[:, 8:].sum().backward()
    assert embeddings.grad[:, :8].abs().sum() == 0
    assert embeddings.grad[:, 8:].abs().sum() > 0


def test_a_tiny_model_overfits_packed_documents():
    config = ModelConfig(
        vocab_size=64, reserved_slots=8, d_model=64, n_layers=2, n_query_heads=4, n_kv_heads=2, head_dim=16, ffn_hidden=128
    )
    torch.manual_seed(0)
    model = Scout(config)
    ids = torch.randint(0, 40, (4, 32), generator=torch.Generator().manual_seed(7))
    ids[:, 0] = torch.tensor([1, 2, 3, 4])
    ids[:, 15] = 56
    ids[:, 16] = torch.tensor([5, 6, 7, 8])
    doc = document_ids(ids, 56)
    targets = torch.roll(ids, -1, 1)
    targets[:, 15] = IGNORE_INDEX
    targets[:, -1] = IGNORE_INDEX
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(200):
        loss = model.loss(ids, targets, document_ids=doc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05


def test_grouped_and_ungrouped_heads_both_respect_the_mask():
    for n_kv_heads in (2, 14):
        torch.manual_seed(0)
        attention = Attention(896, 14, n_kv_heads, 64).double()
        x = torch.randn(1, 12, 896, dtype=torch.float64)
        positions = torch.cat((torch.arange(6), torch.arange(6)))[None]
        cos, sin = RotaryEmbedding(64)(positions, torch.float64)
        out = attention(x, cos, sin, mask=document_mask(torch.tensor([[0] * 6 + [1] * 6])))
        alone = attention(x[:, 6:], *RotaryEmbedding(64)(torch.arange(6)[None], torch.float64))
        torch.testing.assert_close(out[:, 6:], alone, rtol=1e-10, atol=1e-10)


def test_a_mask_cannot_be_combined_with_a_cache(model):
    cache = KVCache(model.config, 1, 8, torch.float64)
    cache.begin(torch.ones(1, 4, dtype=torch.bool))
    attention = model.model.layers[0].self_attn
    cos, sin = model.model.rotary_emb(torch.arange(4)[None], torch.float64)
    with pytest.raises(ValueError, match="cache"):
        attention(torch.randn(1, 4, 896, dtype=torch.float64), cos, sin, cache.layer(0), document_mask(torch.zeros(1, 4, dtype=torch.long)))


@pytest.mark.parametrize("shape", [(2, 9), (1, 8), (8,)])
def test_rejects_document_ids_of_the_wrong_shape(model, shape):
    with pytest.raises(ValueError):
        model(tokens(2, 8), document_ids=torch.zeros(*shape, dtype=torch.long))
