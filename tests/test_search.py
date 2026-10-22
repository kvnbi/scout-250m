import math

import pytest

from scout.retrieval.passages import Passage
from scout.retrieval.search import Hit, SearchBackend, SearchError, check_backend, domain, render_hit, render_hits, validate_hits
from scout.tokenizer import markers
from test_tokenizer_markers import TRANSCRIPT


def passage(identifier="1:0", text="The bridge opened in 1932.", title="Sydney Harbour Bridge", url="https://en.wikipedia.org/wiki/X", date="2018-12-20"):
    return Passage(identifier, title, "", text, url, date)


def hits(*scores, prefix="p"):
    return [Hit(passage(f"{prefix}:{i}", f"text {i}"), score) for i, score in enumerate(scores)]


def test_scores_must_not_increase_down_the_list():
    validate_hits(hits(3.0, 2.0, 2.0, -1.0), 5)
    validate_hits([], 3)
    with pytest.raises(ValueError, match="higher"):
        validate_hits(hits(1.0, 2.0), 5)


def test_at_most_k_hits_and_k_must_be_positive():
    validate_hits(hits(3.0, 2.0), 2)
    with pytest.raises(ValueError, match="3 hits for k=2"):
        validate_hits(hits(3.0, 2.0, 1.0), 2)
    for k in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            validate_hits([], k)


def test_passage_ids_must_be_unique_within_a_result():
    repeated = [Hit(passage("a:0"), 2.0), Hit(passage("a:0", "other"), 1.0)]
    with pytest.raises(ValueError, match="repeats"):
        validate_hits(repeated, 5)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_scores_must_be_finite(score):
    with pytest.raises(ValueError, match="finite"):
        validate_hits([Hit(passage(), score)], 5)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_a_hit_needs_text(text):
    with pytest.raises(ValueError, match="no text"):
        validate_hits([Hit(passage(text=text), 1.0)], 5)


@pytest.mark.parametrize("item", [None, "text", (passage(), 1.0), Hit("passage", 1.0)])
def test_only_hits_of_passages_are_accepted(item):
    with pytest.raises(ValueError, match="not a Hit"):
        validate_hits([item], 5)


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://en.wikipedia.org/wiki/Example", "en.wikipedia.org"),
        ("HTTP://WWW.Example.COM:8080/path?q=1", "www.example.com"),
        ("https://user:pass@docs.python.org/3/", "docs.python.org"),
        ("file:///home/user/notes/todo.txt", "local"),
        ("FILE:///x", "local"),
        ("", ""),
        ("not a url", ""),
        ("/just/a/path", ""),
    ],
)
def test_domain(url, expected):
    assert domain(url) == expected


def test_a_rendered_hit_matches_the_transcript_format_the_markers_were_tested_with():
    hit = Hit(passage(text="The bridge opened in 1932. It was designed by John Bradfield."), 1.0)
    rendered = render_hit(1, hit)
    assert rendered == (
        "[1] Sydney Harbour Bridge (en.wikipedia.org, 2018-12-20)\n"
        "The bridge opened in 1932. It was designed by John Bradfield."
    )
    assert f"{markers.PASSAGE}{rendered}{markers.PASSAGE_END}" in TRANSCRIPT


@pytest.mark.parametrize(
    "url, date, header",
    [
        ("https://a.org/x", "2020-01-02", "[3] T (a.org, 2020-01-02)"),
        ("https://a.org/x", "", "[3] T (a.org)"),
        ("", "2020-01-02", "[3] T (2020-01-02)"),
        ("", "", "[3] T"),
        ("file:///notes.txt", "2024-05-06", "[3] T (local, 2024-05-06)"),
    ],
)
def test_missing_details_leave_no_stray_punctuation(url, date, header):
    assert render_hit(3, Hit(passage(title="T", text="Body.", url=url, date=date), 1.0)) == header + "\nBody."


def test_hits_are_numbered_from_one_and_separated_by_a_blank_line():
    result = render_hits([Hit(passage("a:0", "First.", "A"), 2.0), Hit(passage("b:0", "Second.\nMore.", "B"), 1.0)])
    assert result == "[1] A (en.wikipedia.org, 2018-12-20)\nFirst.\n\n[2] B (en.wikipedia.org, 2018-12-20)\nSecond.\nMore."
    assert render_hits([]) == ""


class Good:
    name = "good"

    def search(self, query, k):
        if k < 1:
            raise ValueError("k")
        if not query.strip():
            return []
        return hits(2.0, 1.0)[:k]


def test_structural_typing_needs_a_name_and_a_search_method():
    assert isinstance(Good(), SearchBackend)

    class NoName:
        def search(self, query, k):
            return []

    class NoSearch:
        name = "x"

    assert not isinstance(NoName(), SearchBackend)
    assert not isinstance(NoSearch(), SearchBackend)


def test_a_conforming_backend_passes_the_contract_check():
    check_backend(Good(), ["bridge", "anything at all"])
    check_backend(Good(), [], k=1)


def broken(**overrides):
    class Broken(Good):
        def search(self, query, k):
            return overrides["search"](self, query, k)

    return Broken()


@pytest.mark.parametrize(
    "search, message",
    [
        (lambda self, q, k: hits(2.0, 1.0)[:k], "blank query"),
        (lambda self, q, k: [] if not q.strip() else hits(2.0, 1.0)[:k], "k below 1"),
        (lambda self, q, k: [] if not q.strip() else (_ for _ in ()).throw(ValueError("k")) if k < 1 else hits(1.0, 2.0), "higher"),
        (lambda self, q, k: [] if not q.strip() else (_ for _ in ()).throw(ValueError("k")) if k < 1 else hits(*range(k + 2, 0, -1)), "hits for"),
    ],
)
def test_each_contract_violation_is_reported(search, message):
    with pytest.raises(ValueError, match=message):
        check_backend(broken(search=search), ["bridge"])


def test_a_blank_query_is_checked_for_every_kind_of_blank():
    def search(self, query, k):
        if k < 1:
            raise ValueError("k")
        return [] if query in ("", "   ") else hits(1.0)

    with pytest.raises(ValueError, match="blank query"):
        check_backend(broken(search=search), [])


def test_an_object_that_is_not_a_backend_is_a_type_error():
    with pytest.raises(TypeError):
        check_backend(object(), ["q"])


def test_search_errors_are_runtime_errors_a_caller_can_tell_from_no_results():
    assert issubclass(SearchError, RuntimeError)
    assert not issubclass(SearchError, ValueError)
