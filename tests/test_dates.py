import datetime
import random

import pytest

from scout.retrieval.bm25 import BM25Index, build_index
from scout.retrieval.dates import AsOf, Moment, compare_dates, newest, normalize_date, parse_date, render_today
from scout.retrieval.passages import Passage
from scout.retrieval.search import Hit, SearchBackend, check_backend


def d(text):
    return datetime.date.fromisoformat(text)


@pytest.mark.parametrize(
    "value, text, first, last",
    [
        ("2018", "2018", "2018-01-01", "2018-12-31"),
        ("2018-12", "2018-12", "2018-12-01", "2018-12-31"),
        ("2018-12-20", "2018-12-20", "2018-12-20", "2018-12-20"),
        ("20181220", "2018-12-20", "2018-12-20", "2018-12-20"),
        ("201812", "2018-12", "2018-12-01", "2018-12-31"),
        ("  2018-12-20  ", "2018-12-20", "2018-12-20", "2018-12-20"),
        ("2018-12-20 14:01:37", "2018-12-20", "2018-12-20", "2018-12-20"),
        ("2026-05-12T14:01:37Z", "2026-05-12", "2026-05-12", "2026-05-12"),
        ("2026-05-12T14:01:37.123+02:00", "2026-05-12", "2026-05-12", "2026-05-12"),
        ("2026-05-12T14:01-0530", "2026-05-12", "2026-05-12", "2026-05-12"),
        ("2024-02", "2024-02", "2024-02-01", "2024-02-29"),
        ("2023-02", "2023-02", "2023-02-01", "2023-02-28"),
        ("2000-02-29", "2000-02-29", "2000-02-29", "2000-02-29"),
        ("2018-04", "2018-04", "2018-04-01", "2018-04-30"),
        ("0001-01-01", "0001-01-01", "0001-01-01", "0001-01-01"),
        ("9999-12-31", "9999-12-31", "9999-12-31", "9999-12-31"),
    ],
)
def test_valid_dates_become_a_canonical_text_and_a_span(value, text, first, last):
    moment = parse_date(value)
    assert moment == Moment(d(first), d(last), text)
    assert moment.exact == (first == last)
    assert normalize_date(value) == text


@pytest.mark.parametrize(
    "value",
    [
        "", "   ", "yesterday", "12/20/2018", "20 December 2018", "2018 AD", "2018-13", "2018-00", "2018-02-30", "1900-02-29",
        "0000", "0000-01-01", "2018122", "2018-1-5", "18-12-20", "2018-12-20 whatever", "2018-12-20T25:00", "2018-12T10:00",
        "2018-12-20 14", "-2018", "2018-", "2018-12-", "".join(chr(code) for code in (0x662, 0x660, 0x661, 0x668)), "2018\n12", "20181", "2018-12-20T23:60", "2018-12-20T24:00", "2018-12-20T10:00:61", "2018-12-20T10:00+24:00", "2018-12-20T10:00+0060",
    ],
)
def test_anything_else_is_not_a_date(value):
    assert parse_date(value) is None
    assert normalize_date(value) == ""


def test_other_types_are_not_dates():
    for value in (None, 2018, datetime.date(2018, 12, 20), ["2018"]):
        assert parse_date(value) is None


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("2026-05-12", "2018-12-20", 1),
        ("2018-12-20", "2026-05-12", -1),
        ("2018-12-20", "2018-12-20", 0),
        ("2018-12-21", "2018-12-20", 1),
        ("2026-05", "2026-05-12", 0),
        ("2026-05-12", "2026-05", 0),
        ("2026", "2026-05-12", 0),
        ("2026-05", "2026-04-30", 1),
        ("2026-05", "2026-06-01", -1),
        ("2026-05", "2026-05", 0),
        ("2025", "2026-01-01", -1),
        ("2026-01-01", "2025", 1),
        ("2025-12-31", "2025", 0),
        ("2026", "2025", 1),
        ("2026-01", "2025", 1),
        ("", "2026-05-12", 0),
        ("2026-05-12", "", 0),
        ("nonsense", "2026-05-12", 0),
        ("", "", 0),
        ("20181221", "2018-12-20", 1),
        ("2026-05-12T10:00:00Z", "2026-05-11", 1),
    ],
)
def test_one_date_is_later_only_when_it_is_certainly_later(a, b, expected):
    assert compare_dates(a, b) == expected


def test_comparison_is_antisymmetric_on_random_dates_of_every_precision():
    rng = random.Random(0)

    def random_date():
        year, month, day = rng.randint(2015, 2027), rng.randint(1, 12), rng.randint(1, 28)
        return rng.choice([f"{year}", f"{year}-{month:02d}", f"{year}-{month:02d}-{day:02d}", "", "junk"])

    for _ in range(2000):
        a, b = random_date(), random_date()
        assert compare_dates(a, b) == -compare_dates(b, a)
        assert compare_dates(a, a) == 0


def hit(identifier, date="", score=1.0):
    return Hit(Passage(identifier, "T", "", f"text {identifier}", "https://x.org", date), score)


def test_newest_picks_the_certainly_latest_and_keeps_the_first_when_unsure():
    old, new = hit("a", "2018-12-20"), hit("b", "2026-05-12")
    assert newest([old, new]) is new and newest([new, old]) is new
    assert newest([]) is None
    assert newest([hit("a", "")]).passage.id == "a"
    assert newest([hit("a", ""), hit("b", "2018-12-20")]).passage.id == "a"
    assert newest([hit("a", "2026"), hit("b", "2026-05-12"), hit("c", "2025-01-01")]).passage.id == "a"
    assert newest([hit("c", "2025-01-01"), hit("a", "2026"), hit("b", "2026-05-12")]).passage.id == "a"
    assert newest([hit("a", "2025-01"), hit("b", "2026-05-12"), hit("c", "2026-05-13")]).passage.id == "c"


def test_the_phrase_that_tells_an_episode_the_date():
    assert render_today("2026-10-23") == "Today is 2026-10-23."
    assert render_today(datetime.date(2026, 1, 5)) == "Today is 2026-01-05."
    assert render_today("20261023") == "Today is 2026-10-23."
    assert render_today("2026-10-23T08:00:00Z") == "Today is 2026-10-23."


@pytest.mark.parametrize("value", ["2026-10", "2026", "", "tomorrow", "2026-02-30"])
def test_only_a_full_date_can_be_today(value):
    with pytest.raises(ValueError, match="full date"):
        render_today(value)


class Corpus:
    name = "corpus"

    def __init__(self, hits):
        self.hits = hits
        self.requests = []

    def search(self, query, k):
        if k < 1:
            raise ValueError("k")
        self.requests.append(k)
        return [] if not query.strip() else self.hits[:k]


def ranked(dates):
    return [hit(f"p{i}", date, score=100.0 - i) for i, date in enumerate(dates)]


def test_sources_certainly_dated_after_today_are_dropped_and_order_is_kept():
    corpus = Corpus(ranked(["2018-12-20", "2026-11-01", "2026-10-23", "2026-10-24", "2020-01-01"]))
    found = AsOf(corpus, "2026-10-23").search("q", 5)
    assert [h.passage.id for h in found] == ["p0", "p2", "p4"]
    assert [h.score for h in found] == [100.0, 98.0, 96.0]


def test_partial_dates_that_could_be_today_or_earlier_are_kept():
    corpus = Corpus(ranked(["2026", "2026-10", "2026-11", "2027", "2026-10-23", "20261023", "2026-10-23T23:59:59Z"]))
    found = AsOf(corpus, d("2026-10-23")).search("q", 7)
    assert [h.passage.id for h in found] == ["p0", "p1", "p4", "p5", "p6"]


def test_undated_and_unreadable_dates_follow_the_keep_undated_choice():
    corpus = Corpus(ranked(["", "junk", "2018-01-01", "2030-01-01"]))
    assert [h.passage.id for h in AsOf(corpus, "2026-10-23").search("q", 4)] == ["p0", "p1", "p2"]
    assert [h.passage.id for h in AsOf(corpus, "2026-10-23", keep_undated=False).search("q", 4)] == ["p2"]


def test_the_wrapper_looks_further_down_the_ranking_when_the_top_is_in_the_future():
    corpus = Corpus(ranked(["2030-01-01"] * 25 + ["2018-01-01"] * 10))
    found = AsOf(corpus, "2026-10-23").search("q", 3)
    assert [h.passage.id for h in found] == ["p25", "p26", "p27"]
    assert corpus.requests == [3, 12, 48]


def test_it_stops_when_the_backend_is_exhausted_or_the_limit_is_reached():
    short = Corpus(ranked(["2030-01-01"] * 5 + ["2018-01-01"] * 2))
    assert [h.passage.id for h in AsOf(short, "2026-10-23").search("q", 4)] == ["p5", "p6"]
    assert short.requests == [4, 16]
    endless = Corpus(ranked(["2030-01-01"] * 100))
    assert AsOf(endless, "2026-10-23", limit=10).search("q", 3) == []
    assert endless.requests == [3, 10]
    exact = Corpus(ranked(["2018-01-01"] * 5))
    assert len(AsOf(exact, "2026-10-23").search("q", 5)) == 5 and exact.requests == [5]


def test_it_never_returns_more_than_k():
    corpus = Corpus(ranked(["2018-01-01"] * 30))
    assert len(AsOf(corpus, "2026-10-23").search("q", 7)) == 7


def test_the_wrapper_is_a_backend_that_names_its_date_and_passes_the_contract_check():
    wrapped = AsOf(Corpus(ranked(["2018-01-01", "2030-01-01", "2020-01-01"])), "2026-10-23")
    assert isinstance(wrapped, SearchBackend) and wrapped.name == "corpus@2026-10-23"
    check_backend(wrapped, ["anything", "else"], k=3)
    assert wrapped.search("   ", 3) == []


def test_the_wrapper_refuses_a_bad_k_whatever_the_backend_does():
    class Lenient:
        name = "lenient"

        def search(self, query, k):
            return []

    for k in (0, -2):
        with pytest.raises(ValueError, match="positive"):
            AsOf(Lenient(), "2026-10-23").search("q", k)


def test_invalid_settings_are_refused():
    corpus = Corpus([])
    for today in ("2026-10", "x", ""):
        with pytest.raises(ValueError, match="full date"):
            AsOf(corpus, today)
    with pytest.raises(ValueError, match="limit"):
        AsOf(corpus, "2026-10-23", limit=0)
    for k in (0, -1):
        with pytest.raises(ValueError):
            AsOf(corpus, "2026-10-23").search("q", k)


def test_an_episode_date_hides_newer_versions_of_a_fact_from_a_real_index(tmp_path):
    passages = [
        Passage("1:0", "Mayor", "", "The mayor of Springfield is Alice Jones.", "https://x.org/a", "2018-12-20"),
        Passage("2:0", "Mayor", "", "The mayor of Springfield is Bob Smith.", "https://x.org/a", "2026-05-12"),
        Passage("3:0", "Other", "", "Unrelated text about rivers.", "https://x.org/b", "2018-12-20"),
    ]
    build_index(passages, tmp_path / "idx")
    index = BM25Index(tmp_path / "idx")
    assert {h.passage.id for h in index.search("mayor of Springfield", 5)} == {"1:0", "2:0"}
    assert [h.passage.id for h in AsOf(index, "2019-06-01").search("mayor of Springfield", 5)] == ["1:0"]
    both = AsOf(index, "2027-01-01").search("mayor of Springfield", 5)
    assert {h.passage.id for h in both} == {"1:0", "2:0"} and newest(both).passage.id == "2:0"
