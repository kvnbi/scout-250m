import pytest
import torch
from conftest import unchanged

from scout.cache import KVCache
from scout.config import ModelConfig
from scout.generate import generate
from scout.model import Scout

SMALL = ModelConfig(vocab_size=512, n_layers=3)



@pytest.fixture(scope="module")
def model64():
    yield from unchanged(make())


@pytest.fixture(scope="module")
def model32():
    yield from unchanged(make(torch.float32))

def make(dtype=torch.float64, seed=0, config=SMALL):
    torch.manual_seed(seed)
    model = Scout(config).to(dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 1:
                parameter.copy_(1.0 + 0.1 * torch.randn_like(parameter))
    return model


def tokens(batch, seq, seed=1):
    return torch.randint(0, 512, (batch, seq), generator=torch.Generator().manual_seed(seed))


def cache_for(model, batch, max_length):
    return KVCache(model.config, batch, max_length, model.lm_head.weight.dtype)


def test_prefill_matches_uncached_forward(model64):
    model = model64
    ids = tokens(2, 24)
    cache = cache_for(model, 2, 24)
    torch.testing.assert_close(model.forward_cached(ids, cache), model(ids), rtol=1e-12, atol=1e-12)
    assert cache.length == 24


def test_token_by_token_matches_uncached_forward(model64):
    model = model64
    ids = tokens(2, 20)
    cache = cache_for(model, 2, 20)
    stepped = torch.cat([model.forward_cached(ids[:, t : t + 1], cache) for t in range(20)], dim=1)
    torch.testing.assert_close(stepped, model(ids), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("chunks", [(5, 3, 7, 1, 4), (1, 19), (19, 1)])
def test_chunked_prefill_matches_uncached_forward(chunks, model64):
    model = model64
    ids = tokens(2, 20)
    cache = cache_for(model, 2, 20)
    pieces, start = [], 0
    for size in chunks:
        pieces.append(model.forward_cached(ids[:, start : start + size], cache))
        start += size
    torch.testing.assert_close(torch.cat(pieces, dim=1), model(ids), rtol=1e-12, atol=1e-12)


def test_float32_decoding_stays_within_rounding(model32):
    model = model32
    ids = tokens(2, 32)
    cache = cache_for(model, 2, 32)
    stepped = torch.cat([model.forward_cached(ids[:, t : t + 1], cache) for t in range(32)], dim=1)
    torch.testing.assert_close(stepped, model(ids), rtol=1e-4, atol=1e-5)


def test_left_padding_matches_unpadded_rows(model64):
    model = model64
    short, long = tokens(1, 9, seed=2), tokens(1, 14, seed=3)
    padded = torch.cat((torch.cat((torch.zeros(1, 5, dtype=torch.long), short), dim=1), long))
    valid = torch.ones(2, 14, dtype=torch.bool)
    valid[0, :5] = False
    cache = cache_for(model, 2, 14)
    logits = model.forward_cached(padded, cache, valid=valid)
    torch.testing.assert_close(logits[0, 5:], model(short)[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(logits[1], model(long)[0], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_padding_positions_stay_finite(dtype):
    model = make(dtype)
    ids = torch.cat((torch.zeros(1, 5, dtype=torch.long), tokens(1, 9, seed=2)), dim=1)
    valid = torch.ones(1, 14, dtype=torch.bool)
    valid[0, :5] = False
    logits = model.forward_cached(ids, cache_for(model, 1, 14), valid=valid)
    assert torch.isfinite(logits).all()


def test_ragged_appends_match_each_row_alone(model64):
    model = model64
    first = [tokens(1, 6, seed=4), tokens(1, 6, seed=5)]
    second = [tokens(1, 3, seed=6), tokens(1, 7, seed=7)]
    third = [tokens(1, 2, seed=8), tokens(1, 2, seed=9)]
    cache = cache_for(model, 2, 6 + 7 + 2)
    model.forward_cached(torch.cat(first), cache)
    appended = torch.zeros(2, 7, dtype=torch.long)
    appended[0, 4:] = second[0][0]
    appended[1] = second[1][0]
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[0, :4] = False
    model.forward_cached(appended, cache, valid=valid)
    logits = model.forward_cached(torch.cat(third), cache)
    for row in range(2):
        alone = model(torch.cat((first[row], second[row], third[row]), dim=1))
        torch.testing.assert_close(logits[row], alone[0, -2:], rtol=1e-12, atol=1e-12)


def test_last_only_returns_final_position(model64):
    model = model64
    ids = tokens(2, 10)
    cache = cache_for(model, 2, 10)
    last = model.forward_cached(ids, cache, last_only=True)
    assert last.shape == (2, 1, 512)
    torch.testing.assert_close(last[:, 0], model(ids)[:, -1], rtol=1e-12, atol=1e-12)


def test_cache_size_matches_config():
    config = ModelConfig()
    cache = KVCache(config, 2, 100, torch.bfloat16)
    stored = (cache.keys.numel() + cache.values.numel()) * cache.keys.element_size()
    assert stored == 2 * 100 * config.kv_cache_bytes_per_token(2)


def test_rejects_overflow(model64):
    model = model64
    cache = cache_for(model, 1, 8)
    model.forward_cached(tokens(1, 6), cache)
    with pytest.raises(ValueError):
        model.forward_cached(tokens(1, 3), cache)


def test_rejects_wrong_batch_and_mask(model64):
    model = model64
    with pytest.raises(ValueError):
        model.forward_cached(tokens(2, 4), cache_for(model, 3, 8))
    with pytest.raises(ValueError):
        model.forward_cached(tokens(2, 4), cache_for(model, 2, 8), valid=torch.ones(2, 3, dtype=torch.bool))


def test_rejects_mismatched_cache_dtype(model32):
    model = model32
    with pytest.raises(TypeError):
        model.forward_cached(tokens(1, 4), KVCache(model.config, 1, 8, torch.float64))


def test_cache_must_begin_before_use():
    cache = KVCache(SMALL, 1, 8)
    with pytest.raises(RuntimeError):
        cache.layer(0)
    with pytest.raises(RuntimeError):
        cache.commit()


def naive_greedy(model, prompt, steps):
    ids = prompt
    for _ in range(steps):
        ids = torch.cat((ids, model(ids)[:, -1].argmax(-1, keepdim=True)), dim=1)
    return ids[:, prompt.shape[1] :]


def test_greedy_generation_matches_uncached_decoding(model32):
    model = model32
    prompt = tokens(3, 12)
    assert torch.equal(generate(model, prompt, 16), naive_greedy(model, prompt, 16))


def test_sampling_is_reproducible_with_a_generator(model32):
    model = model32
    prompt = tokens(2, 8)
    first = generate(model, prompt, 12, temperature=1.0, generator=torch.Generator().manual_seed(5))
    second = generate(model, prompt, 12, temperature=1.0, generator=torch.Generator().manual_seed(5))
    other = generate(model, prompt, 12, temperature=1.0, generator=torch.Generator().manual_seed(6))
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_stop_token_ends_rows_and_generation(model32):
    model = model32
    prompt = tokens(2, 8)
    free = generate(model, prompt, 10)
    stop = int(free[0, 2])
    stopped = generate(model, prompt, 10, stop_token=stop)
    first_stop = int((free[0] == stop).nonzero()[0])
    assert torch.equal(stopped[0, : first_stop + 1], free[0, : first_stop + 1])
    assert (stopped[0, first_stop:] == stop).all()


def test_generation_stops_early_when_every_row_finishes(model32):
    model = model32
    prompt = tokens(1, 8)
    stop = int(generate(model, prompt, 1)[0, 0])
    assert generate(model, prompt, 10, stop_token=stop).shape == (1, 1)


def test_logits_processor_is_applied_with_history(model32):
    model = model32
    prompt = tokens(2, 8)
    seen = []

    def forbid_repeats(logits, generated):
        seen.append(generated.shape[1])
        logits = logits.clone()
        logits.scatter_(1, generated, float("-inf"))
        return logits

    out = generate(model, prompt, 20, logits_processor=forbid_repeats)
    assert seen == list(range(20))
    for row in out:
        assert len(set(row.tolist())) == 20


@pytest.mark.parametrize(
    "kwargs",
    [{"max_new_tokens": 0}, {"max_new_tokens": 4, "temperature": -1.0}],
)
def test_generate_rejects_invalid_settings(kwargs, model32):
    with pytest.raises(ValueError):
        generate(model32, tokens(1, 4), **kwargs)
