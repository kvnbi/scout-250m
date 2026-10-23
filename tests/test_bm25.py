import json
import math
import random

import numpy as np
import pytest

import scout.retrieval.bm25 as bm25
from scout.retrieval.bm25 import BM25Index, build_index, tokenize
from scout.retrieval.passages import Passage
from scout.retrieval.search import SearchBackend, check_backend


def passage(number, text, title="Title", url="https://example.org/x", date="2018-12-20"):
    return Passage(f"{number}:0", title, "", text, url, date)


def naive_scores(passages, query, k1, b, include_title=True):
    documents = [tokenize(f"{p.title}\n{p.text}" if include_title else p.text) for p in passages]
    average = sum(len(d) for d in documents) / len(documents) or 1.0
    counts = [{} for _ in documents]
    for number, tokens in enumerate(documents):
        for token in tokens:
            counts[number][token] = counts[number].get(token, 0) + 1
    scores = [0.0] * len(documents)
    for term in set(tokenize(query)):
        df = sum(1 for c in counts if term in c)
        if not df:
            continue
        idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
        for number, c in enumerate(counts):
            if term in c:
                tf = c[term]
                scores[number] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(documents[number]) / average))
    return scores


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World!", ["hello", "world"]),
        ("Caf\u00e9 na\u00efve R\u00c9SUM\u00c9", ["cafe", "naive", "resume"]),
        ("snake_case and kebab-case", ["snake", "case", "kebab", "case"]),
        ("born 14 March 1879 in Ulm", ["born", "14", "march", "1879", "ulm"]),
        ("The quick brown fox is on the mat", ["quick", "brown", "fox", "mat"]),
        ("don't stop", ["don", "t", "stop"]),
        ("Stra\u00dfe", ["strasse"]),
        ("x" * 64 + " " + "y" * 65, ["x" * 64]),
        ("", []),
        ("   ...  ", []),
        ("the of and", []),
    ],
)
def test_tokenizer(text, expected):
    assert tokenize(text) == expected


VOCABULARY = [f"w{i}" for i in range(60)]


def random_passages(rng, count=150):
    weights = [1 / (rank + 1) for rank in range(len(VOCABULARY))]
    return [
        passage(i, " ".join(rng.choices(VOCABULARY, weights, k=rng.randint(0, 40))), title=" ".join(rng.choices(VOCABULARY, weights, k=rng.randint(0, 3))))
        for i in range(count)
    ]


@pytest.mark.parametrize("k1, b, include_title", [(0.9, 0.4, True), (1.2, 0.75, True), (2.0, 0.0, False), (0.5, 1.0, True)])
def test_scores_equal_a_naive_implementation_of_the_formula(tmp_path, k1, b, include_title):
    rng = random.Random(7)
    passages = random_passages(rng)
    build_index(passages, tmp_path / "idx", k1=k1, b=b, include_title=include_title)
    index = BM25Index(tmp_path / "idx")
    for _ in range(60):
        query = " ".join(rng.choices(VOCABULARY, k=rng.randint(1, 5)))
        expected = naive_scores(passages, query, k1, b, include_title)
        top = sorted(expected, reverse=True)[:10]
        top = [score for score in top if score > 0]
        hits = index.search(query, 10)
        assert [h.score for h in hits] == pytest.approx(top, rel=1e-5)
        by_id = {p.id: n for n, p in enumerate(passages)}
        for hit in hits:
            assert hit.score == pytest.approx(expected[by_id[hit.passage.id]], rel=1e-5)
        if hits:
            returned = {h.passage.id for h in hits}
            ceiling = hits[-1].score * (1 + 1e-5)
            assert all(score <= ceiling for n, score in enumerate(expected) if passages[n].id not in returned)


def test_ties_come_out_in_document_order_whatever_k_is(tmp_path):
    passages = [passage(i, "alpha beta") for i in range(20)]
    build_index(passages, tmp_path / "idx")
    index = BM25Index(tmp_path / "idx")
    assert [h.passage.id for h in index.search("alpha", 20)] == [p.id for p in passages]
    assert [h.passage.id for h in index.search("alpha", 5)] == [p.id for p in passages[:5]]
    assert [h.passage.id for h in index.search("alpha", 1)] == [passages[0].id]


def test_rarer_terms_count_more_and_frequency_saturates(tmp_path):
    passages = [passage(0, "common rare"), passage(1, "common"), passage(2, "common common common common"), passage(3, "other")]
    passages += [passage(10 + i, "common filler") for i in range(10)]
    build_index(passages, tmp_path / "idx", include_title=False)
    index = BM25Index(tmp_path / "idx")
    scores = {h.passage.id: h.score for h in index.search("common rare", 20)}
    assert scores["0:0"] > scores["2:0"] > scores["1:0"]
    four, one = scores["2:0"], scores["1:0"]
    assert four < 4 * one


def test_longer_passages_score_lower_unless_b_is_zero(tmp_path):
    passages = [passage(0, "target"), passage(1, "target " + "pad " * 30), passage(2, "pad pad")]
    build_index(passages, tmp_path / "normalised", include_title=False, b=0.75)
    build_index(passages, tmp_path / "flat", include_title=False, b=0.0)
    short, long = (h.score for h in sorted(BM25Index(tmp_path / "normalised").search("target", 5), key=lambda h: h.passage.id))
    assert short > long
    short, long = (h.score for h in sorted(BM25Index(tmp_path / "flat").search("target", 5), key=lambda h: h.passage.id))
    assert short == pytest.approx(long)


def test_repeating_a_query_word_changes_nothing_and_case_and_accents_do_not_matter(tmp_path):
    build_index([passage(0, "The Caf\u00e9 serves coffee"), passage(1, "nothing relevant")], tmp_path / "idx")
    index = BM25Index(tmp_path / "idx")
    once = index.search("coffee", 5)
    assert [h.score for h in index.search("coffee COFFEE coffee", 5)] == [h.score for h in once]
    assert [h.passage.id for h in index.search("CAFE", 5)] == ["0:0"]
    assert [h.passage.id for h in index.search("caf\u00e9", 5)] == ["0:0"]


def test_blank_unknown_and_stopword_only_queries_find_nothing_and_k_must_be_positive(tmp_path):
    build_index([passage(0, "alpha beta")], tmp_path / "idx")
    index = BM25Index(tmp_path / "idx")
    for query in ("", "   ", "zzzz", "the of and", "!!!"):
        assert index.search(query, 5) == []
    for k in (0, -3):
        with pytest.raises(ValueError):
            index.search("alpha", k)
    assert len(index.search("alpha beta zzzz", 100)) == 1


def test_titles_are_searchable_only_when_requested(tmp_path):
    passages = [passage(0, "plain body text", title="Quokka"), passage(1, "another body")]
    build_index(passages, tmp_path / "with", include_title=True)
    build_index(passages, tmp_path / "without", include_title=False)
    assert [h.passage.id for h in BM25Index(tmp_path / "with").search("quokka", 5)] == ["0:0"]
    assert BM25Index(tmp_path / "without").search("quokka", 5) == []


def test_the_index_returns_the_stored_passages_exactly(tmp_path):
    passages = [
        Passage("1:0", "Z\u00fcrich \u4e2d\u6587", "Sec > Sub", "line one\nline two with \"quotes\" and \\ backslash", "https://x.org/a?b=c", "2018-12-20"),
        passage(2, "second passage about zurich"),
    ]
    build_index(passages, tmp_path / "idx")
    index = BM25Index(tmp_path / "idx")
    assert [index.passage(i) for i in range(2)] == passages
    assert index.search("backslash", 5)[0].passage == passages[0]
    for bad in (-1, 2, 10):
        with pytest.raises(IndexError):
            index.passage(bad)


def test_documents_made_only_of_stopwords_are_stored_but_never_found(tmp_path):
    passages = [passage(0, "the of and", title="The"), passage(1, "alpha")]
    assert build_index(passages, tmp_path / "idx") == 2
    index = BM25Index(tmp_path / "idx")
    assert index.documents == 2 and index.passage(0) == passages[0]
    assert [h.passage.id for h in index.search("alpha", 5)] == ["1:0"]


def test_an_index_of_only_stopwords_still_loads_and_searches(tmp_path):
    build_index([passage(0, "the of", title="a")], tmp_path / "idx")
    index = BM25Index(tmp_path / "idx")
    assert index.search("anything", 5) == [] and index.meta["terms"] == 0


def test_a_term_that_occurs_a_huge_number_of_times_is_capped_not_wrapped(tmp_path):
    build_index([passage(0, "x " * 70000), passage(1, "y")], tmp_path / "idx", include_title=False)
    hit = BM25Index(tmp_path / "idx").search("x", 5)[0]
    k1, b = 0.9, 0.4
    average = (70000 + 1) / 2
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 65535 * (k1 + 1) / (65535 + k1 * (1 - b + b * 70000 / average))
    assert hit.score == pytest.approx(expected, rel=1e-5)


def test_postings_of_every_term_are_in_ascending_document_order(tmp_path):
    build_index(random_passages(random.Random(4), count=120), tmp_path / "idx")
    index = BM25Index(tmp_path / "idx")
    offsets, docs = np.asarray(index._offsets), np.asarray(index._doc_ids)
    for term in range(len(offsets) - 1):
        ids = docs[offsets[term] : offsets[term + 1]]
        assert len(ids) > 0 and np.all(np.diff(ids) > 0)


def test_the_block_size_used_to_compute_impacts_changes_nothing(tmp_path, monkeypatch):
    passages = random_passages(random.Random(6), count=80)
    build_index(passages, tmp_path / "whole")
    monkeypatch.setattr(bm25, "BLOCK_POSTINGS", 7)
    build_index(passages, tmp_path / "blocks")
    assert (tmp_path / "whole" / "impacts.npy").read_bytes() == (tmp_path / "blocks" / "impacts.npy").read_bytes()


def test_the_defaults_are_the_pyserini_parameters(tmp_path):
    rng = random.Random(8)
    passages = random_passages(rng, count=60)
    build_index(passages, tmp_path / "idx")
    index = BM25Index(tmp_path / "idx")
    assert (index.meta["k1"], index.meta["b"], index.meta["include_title"]) == (0.9, 0.4, True)
    expected = naive_scores(passages, "w0 w2", 0.9, 0.4)
    assert [h.score for h in index.search("w0 w2", 3)] == pytest.approx(sorted(expected, reverse=True)[:3], rel=1e-5)


def test_chunking_the_build_changes_nothing(tmp_path):
    passages = random_passages(random.Random(3), count=40)
    for size in (1, 3, 1000):
        build_index(passages, tmp_path / f"c{size}", chunk_documents=size)
    reference = {f.name: f.read_bytes() for f in (tmp_path / "c1000").iterdir()}
    for size in (1, 3):
        assert {f.name: f.read_bytes() for f in (tmp_path / f"c{size}").iterdir()} == reference


def test_metadata_describes_the_index(tmp_path):
    passages = [passage(0, "alpha beta alpha"), passage(1, "beta gamma")]
    build_index(passages, tmp_path / "idx", k1=1.1, b=0.6, include_title=False)
    meta = json.loads((tmp_path / "idx" / "meta.json").read_text())
    assert meta == {
        "format": 1,
        "documents": 2,
        "terms": 3,
        "postings": 4,
        "k1": 1.1,
        "b": 0.6,
        "include_title": False,
        "average_length": 2.5,
    }
    assert (tmp_path / "idx" / "terms.txt").read_text().split("\n") == ["alpha", "beta", "gamma"]


def test_arrays_are_memory_mapped_and_read_only(tmp_path):
    build_index([passage(0, "alpha")], tmp_path / "idx")
    index = BM25Index(tmp_path / "idx")
    for array in (index._offsets, index._doc_ids, index._impacts, index.store._starts):
        assert isinstance(array, np.memmap)
        with pytest.raises(ValueError):
            array[0] = 0


def test_refusals_leave_nothing_behind(tmp_path):
    build_index([passage(0, "alpha")], tmp_path / "idx")
    with pytest.raises(FileExistsError):
        build_index([passage(1, "beta")], tmp_path / "idx")
    assert BM25Index(tmp_path / "idx").search("alpha", 1)
    with pytest.raises(ValueError, match="no passages"):
        build_index([], tmp_path / "empty")
    assert not (tmp_path / "empty").exists() and not (tmp_path / "empty.part").exists()
    for kwargs in (dict(k1=0), dict(k1=-1), dict(b=-0.1), dict(b=1.5), dict(chunk_documents=0)):
        with pytest.raises(ValueError):
            build_index([passage(0, "alpha")], tmp_path / "bad", **kwargs)
    assert not (tmp_path / "bad").exists()


def test_a_stale_staging_folder_from_a_crashed_build_is_replaced(tmp_path):
    (tmp_path / "idx.part").mkdir()
    (tmp_path / "idx.part" / "junk").write_text("x")
    build_index([passage(0, "alpha")], tmp_path / "idx")
    assert sorted(f.name for f in (tmp_path / "idx").iterdir()) == [
        "doc_ids.npy", "impacts.npy", "meta.json", "offsets.npy", "passages.jsonl", "starts.npy", "terms.txt"
    ]
    assert not (tmp_path / "idx.part").exists()


def test_an_index_in_another_format_is_refused(tmp_path):
    build_index([passage(0, "alpha")], tmp_path / "idx")
    meta = json.loads((tmp_path / "idx" / "meta.json").read_text())
    (tmp_path / "idx" / "meta.json").write_text(json.dumps({**meta, "format": 2}))
    with pytest.raises(ValueError, match="format 1"):
        BM25Index(tmp_path / "idx")


def test_a_failing_source_leaves_no_index(tmp_path):
    def source():
        yield passage(0, "alpha")
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError):
        build_index(source(), tmp_path / "idx")
    assert not (tmp_path / "idx").exists()


def test_the_index_honours_the_backend_contract(tmp_path):
    passages = random_passages(random.Random(5), count=60)
    build_index(passages, tmp_path / "idx")
    index = BM25Index(tmp_path / "idx", name="wikipedia-2018-bm25")
    assert isinstance(index, SearchBackend) and index.name == "wikipedia-2018-bm25"
    check_backend(index, ["w0", "w1 w2 w3", "unknown words only", "w5 " * 20], k=7)
