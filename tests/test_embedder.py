import numpy as np
import pytest
import torch
from bert_files import TINY, WORDS, write_model_directory

from scout.retrieval.embedder import PREFIXES, Embedder


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    directory = tmp_path_factory.mktemp("model")
    model = write_model_directory(directory)
    return directory, model, Embedder(directory, max_length=12)


def reference(model, embedder, text, kind):
    ids = embedder.tokenizer.encode(PREFIXES[kind] + text).ids
    with torch.no_grad():
        hidden = model(torch.tensor([ids]))[0]
    return torch.nn.functional.normalize(hidden.mean(0), dim=0).numpy()


TEXTS = ["the cat sat on the mat", "dog ran far away", "hello world", "cafe", "the cat sat on the mat and the dog ran far away hello world"]


@pytest.mark.parametrize("kind", ["query", "passage"])
def test_an_embedding_is_the_prefixed_text_mean_pooled_and_normalised(setup, kind):
    _, model, embedder = setup
    result = embedder.encode(TEXTS, kind)
    assert result.shape == (len(TEXTS), TINY.hidden_size) and result.dtype == np.float32
    for row, text in zip(result, TEXTS):
        np.testing.assert_allclose(row, reference(model, embedder, text, kind), atol=1e-6)
        assert np.linalg.norm(row) == pytest.approx(1.0, abs=1e-6)


def test_the_two_kinds_use_different_prefixes(setup):
    _, _, embedder = setup
    assert PREFIXES == {"query": "query: ", "passage": "passage: "}
    assert not np.allclose(embedder.encode(["the cat"], "query"), embedder.encode(["the cat"], "passage"))
    assert np.array_equal(embedder.encode(["the cat"]), embedder.encode(["the cat"], "passage"))


def test_rows_stay_in_input_order_whatever_the_batch_size(setup):
    _, _, embedder = setup
    alone = np.stack([embedder.encode([text], "query")[0] for text in TEXTS])
    for batch_size in (1, 2, 3, 100):
        batched = embedder.encode(TEXTS, "query", batch_size=batch_size)
        np.testing.assert_allclose(batched, alone, atol=1e-5)
    shuffled = [TEXTS[i] for i in (3, 0, 4, 2, 1)]
    np.testing.assert_allclose(embedder.encode(shuffled, "query", batch_size=2), alone[[3, 0, 4, 2, 1]], atol=1e-5)


def test_repeated_texts_get_identical_rows_and_calls_are_reproducible(setup):
    _, _, embedder = setup
    rows = embedder.encode(["hello world"] * 3 + ["dog"], "query", batch_size=2)
    np.testing.assert_allclose(rows[0], rows[1], atol=1e-6)
    np.testing.assert_allclose(rows[0], rows[2], atol=1e-6)
    assert np.array_equal(embedder.encode(TEXTS), embedder.encode(TEXTS))


def test_no_texts_give_an_empty_matrix_and_an_empty_text_still_embeds(setup):
    _, _, embedder = setup
    assert embedder.encode([], "query").shape == (0, TINY.hidden_size)
    row = embedder.encode([""], "query")[0]
    assert np.linalg.norm(row) == pytest.approx(1.0, abs=1e-6)


def test_long_texts_are_cut_to_the_maximum_length(setup):
    directory, model, _ = setup
    long_text = " ".join(["cat"] * 40)
    ids = [2, WORDS.index("query"), WORDS.index(":")] + [WORDS.index("cat")] * 40 + [3]
    for limit in (5, 9, 16):
        embedder = Embedder(directory, max_length=limit)
        cut = ids[: limit - 1] + [3]
        with torch.no_grad():
            expected = torch.nn.functional.normalize(model(torch.tensor([cut]))[0].mean(0), dim=0).numpy()
        np.testing.assert_allclose(embedder.encode([long_text], "query")[0], expected, atol=1e-6)


def test_unknown_words_and_case_and_accents_are_handled_by_the_tokenizer(setup):
    _, _, embedder = setup
    np.testing.assert_allclose(embedder.encode(["THE Cat"]), embedder.encode(["the cat"]), atol=1e-6)
    np.testing.assert_allclose(embedder.encode(["caf\u00e9"]), embedder.encode(["cafe"]), atol=1e-6)
    assert not np.allclose(embedder.encode(["zzzz"]), embedder.encode(["the cat"]))


def test_padding_does_not_leak_into_the_pooled_average(setup):
    _, _, embedder = setup
    pair = embedder.encode(["cat", "the cat sat on the mat and the dog ran far away"], "query")
    np.testing.assert_allclose(pair[0], embedder.encode(["cat"], "query")[0], atol=1e-5)


def test_half_precision_stays_close_to_full_precision(setup):
    directory, _, embedder = setup
    low = Embedder(directory, dtype=torch.bfloat16, max_length=12)
    assert low.model.word_embeddings.weight.dtype == torch.bfloat16
    full, half = embedder.encode(TEXTS, "query"), low.encode(TEXTS, "query")
    assert half.dtype == np.float32 and np.all(np.sum(full * half, axis=1) > 0.98)


def test_invalid_arguments_are_refused(setup):
    directory, _, embedder = setup
    with pytest.raises(ValueError, match="kind"):
        embedder.encode(["x"], "document")
    for size in (0, -1):
        with pytest.raises(ValueError, match="batch_size"):
            embedder.encode(["x"], "query", batch_size=size)
    for limit in (2, 0, 17, 513):
        with pytest.raises(ValueError, match="max_length"):
            Embedder(directory, max_length=limit)


def test_the_dimension_follows_the_model(setup):
    assert setup[2].dimension == TINY.hidden_size
