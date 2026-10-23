import json
import random

import pytest

from scout.data.fetch import sha256_of
from scout.retrieval.bm25 import BM25Index, build_index
from scout.retrieval.bm25 import main as bm25_main
from scout.retrieval.passages import Passage, iter_passages, write_passages
import scout.retrieval.subset as subset_module
from scout.retrieval.subset import _rank, article_of, build_subset, main, read_article_list, select_articles


def passages_for(article, count, date="2018-12-20"):
    return [Passage(f"{article}:{n}", f"Article {article}", "", f"text of {article} part {n} unique{article}x{n}", "https://x.org", date) for n in range(count)]


def corpus(sizes):
    return [p for article, count in sizes.items() for p in passages_for(article, count)]


@pytest.mark.parametrize(
    "passage_id, expected",
    [("12:0", "12"), ("12:345", "12"), ("uuid-a-b:7", "uuid-a-b"), ("a:b:3", "a:b"), ("plain", "plain"), (":3", ":3"), ("", "")],
)
def test_the_article_is_everything_before_the_last_colon(passage_id, expected):
    assert article_of(passage_id) == expected


COUNTS = {f"a{i}": count for i, count in enumerate([5, 3, 8, 2, 7, 1, 4, 6, 9, 2, 3, 5, 1, 8, 4, 2, 6, 3, 7, 5])}


def test_required_articles_come_first_then_filler_fills_the_budget_without_exceeding_it():
    chosen, missing = select_articles(COUNTS, 40, ["a2", "a8"])
    assert {"a2", "a8"} <= chosen and missing == []
    total = sum(COUNTS[a] for a in chosen)
    assert total <= 40 and total >= 38


def test_the_budget_is_filled_as_tightly_as_whole_articles_allow():
    chosen, _ = select_articles({"big": 10, "small": 1, "tiny": 1}, 11, seed=3)
    assert sum({"big": 10, "small": 1, "tiny": 1}[a] for a in chosen) == 11
    chosen, _ = select_articles({"a": 6, "b": 6, "c": 1}, 7, seed=0)
    assert sum({"a": 6, "b": 6, "c": 1}[a] for a in chosen) == 7


def test_an_article_too_big_to_fit_never_stops_the_search_for_ones_that_do():
    counts = {"a": 6, "b": 6, "c": 1}
    for seed in range(40):
        chosen, _ = select_articles(counts, 7, seed=seed)
        assert sum(counts[a] for a in chosen) == 7 and "c" in chosen


def test_ranks_differ_for_seeds_and_names_that_would_run_together_as_text():
    assert _rank(1, "23")[0] != _rank(12, "3")[0]
    assert _rank(0, "a")[0] != _rank(1, "a")[0] and _rank(0, "a")[0] != _rank(0, "b")[0]


def test_required_articles_may_exceed_the_budget_and_then_no_filler_is_added():
    chosen, missing = select_articles(COUNTS, 5, ["a2", "a8"])
    assert chosen == {"a2", "a8"} and missing == []


def test_missing_required_articles_are_reported_once_in_order_and_nothing_is_invented():
    chosen, missing = select_articles({"a": 2, "b": 3}, 10, ["zz", "a", "yy", "zz", "a"])
    assert missing == ["zz", "yy"] and chosen == {"a", "b"}


def test_the_selection_does_not_depend_on_the_order_the_articles_are_listed():
    reference, _ = select_articles(COUNTS, 40, ["a3"], seed=1)
    items = list(COUNTS.items())
    for trial in range(20):
        random.Random(trial).shuffle(items)
        assert select_articles(dict(items), 40, ["a3"], seed=1)[0] == reference


def test_the_seed_changes_the_filler_and_is_otherwise_stable():
    first = select_articles(COUNTS, 30, seed=0)[0]
    assert select_articles(COUNTS, 30, seed=0)[0] == first
    assert any(select_articles(COUNTS, 30, seed=s)[0] != first for s in range(1, 8))


def test_degenerate_inputs():
    assert select_articles({}, 10) == (set(), [])
    assert select_articles({}, 10, ["a"]) == (set(), ["a"])
    assert select_articles({"a": 3}, 100)[0] == {"a"}
    assert select_articles({"a": 3}, 1)[0] == set()


def sources(tmp_path, sizes=None):
    path = tmp_path / "all.jsonl"
    write_passages(path, corpus(sizes or COUNTS))
    return path


def ids_of(path):
    return [p.id for p in iter_passages(path)]


def test_a_subset_holds_whole_articles_in_source_order_unchanged(tmp_path):
    source = sources(tmp_path)
    manifest = build_subset(source, tmp_path / "sub", 40, ["a2", "a8"], seed=5)
    original = list(iter_passages(source))
    subset = list(iter_passages(tmp_path / "sub" / "passages.jsonl"))
    kept = {article_of(p.id) for p in subset}
    assert [p for p in original if article_of(p.id) in kept] == subset
    assert {"a2", "a8"} <= kept
    for article in kept:
        assert sum(1 for p in subset if article_of(p.id) == article) == COUNTS[article]
    assert manifest["selected"]["passages"] == len(subset) <= 40


def test_the_manifest_accounts_for_everything(tmp_path):
    source = sources(tmp_path)
    manifest = build_subset(source, tmp_path / "sub", 40, ["a2", "a8", "nope"], seed=5)
    assert json.loads((tmp_path / "sub" / "manifest.json").read_text()) == manifest
    assert manifest["format"] == 1
    assert manifest["source"] == {"name": "all.jsonl", "bytes": source.stat().st_size, "sha256": sha256_of(source)}
    assert manifest["parameters"] == {"size": 40, "seed": 5, "strict": False}
    assert manifest["source_totals"] == {"articles": 20, "passages": sum(COUNTS.values())}
    assert manifest["required"] == {"requested": 3, "found": 2, "missing_count": 1, "missing": ["nope"]}
    selected = manifest["selected"]
    assert selected["required_passages"] == COUNTS["a2"] + COUNTS["a8"] == 17
    assert selected["filler_passages"] == selected["passages"] - 17 and selected["over_budget"] is False
    assert manifest["passages_sha256"] == sha256_of(tmp_path / "sub" / "passages.jsonl")
    assert sorted(f.name for f in (tmp_path / "sub").iterdir()) == ["manifest.json", "passages.jsonl"]


def test_a_repeated_required_article_counts_once(tmp_path):
    manifest = build_subset(sources(tmp_path), tmp_path / "sub", 40, ["a1", "a1", "ghost", "ghost", "a1"])
    assert manifest["required"] == {"requested": 2, "found": 1, "missing_count": 1, "missing": ["ghost"]}
    assert manifest["selected"]["required_passages"] == COUNTS["a1"]


def test_a_budget_smaller_than_the_required_articles_is_flagged(tmp_path):
    manifest = build_subset(sources(tmp_path), tmp_path / "sub", 5, ["a2", "a8"])
    assert manifest["selected"]["over_budget"] is True and manifest["selected"]["passages"] == 17
    assert manifest["selected"]["filler_passages"] == 0


def test_a_subset_is_the_same_whatever_the_order_of_the_source_file(tmp_path):
    first = sources(tmp_path)
    shuffled = tmp_path / "shuffled.jsonl"
    rows = list(iter_passages(first))
    random.Random(0).shuffle(rows)
    write_passages(shuffled, rows)
    build_subset(first, tmp_path / "one", 40, ["a1"], seed=2)
    build_subset(shuffled, tmp_path / "two", 40, ["a1"], seed=2)
    assert set(ids_of(tmp_path / "one" / "passages.jsonl")) == set(ids_of(tmp_path / "two" / "passages.jsonl"))


def test_building_twice_gives_identical_bytes_and_a_new_seed_a_new_subset(tmp_path):
    source = sources(tmp_path)
    build_subset(source, tmp_path / "one", 40, seed=1)
    build_subset(source, tmp_path / "two", 40, seed=1)
    build_subset(source, tmp_path / "three", 40, seed=2)
    for name in ("passages.jsonl", "manifest.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    assert (tmp_path / "one" / "passages.jsonl").read_bytes() != (tmp_path / "three" / "passages.jsonl").read_bytes()


def test_a_budget_larger_than_the_corpus_keeps_everything(tmp_path):
    source = sources(tmp_path)
    build_subset(source, tmp_path / "sub", 10_000)
    assert ids_of(tmp_path / "sub" / "passages.jsonl") == ids_of(source)


def test_strict_mode_refuses_missing_required_articles_and_leaves_nothing(tmp_path):
    source = sources(tmp_path)
    with pytest.raises(ValueError, match="1 required articles"):
        build_subset(source, tmp_path / "sub", 40, ["a1", "ghost"], strict=True)
    assert not (tmp_path / "sub").exists() and not (tmp_path / "sub.part").exists()
    build_subset(source, tmp_path / "ok", 40, ["a1"], strict=True)


def test_many_missing_ids_are_counted_but_only_some_are_listed(tmp_path):
    ghosts = [f"ghost{i}" for i in range(80)]
    manifest = build_subset(sources(tmp_path), tmp_path / "sub", 40, ghosts)
    assert manifest["required"]["missing_count"] == 80 and manifest["required"]["missing"] == ghosts[:50]


def test_refusals_and_failures_leave_nothing_behind(tmp_path):
    source = sources(tmp_path)
    build_subset(source, tmp_path / "sub", 40)
    with pytest.raises(FileExistsError):
        build_subset(source, tmp_path / "sub", 40)
    for size in (0, -3):
        with pytest.raises(ValueError, match="size"):
            build_subset(source, tmp_path / "bad", size)
    with pytest.raises(FileNotFoundError):
        build_subset(tmp_path / "absent.jsonl", tmp_path / "missing", 40)
    assert not (tmp_path / "bad").exists() and not (tmp_path / "missing").exists() and not (tmp_path / "missing.part").exists()


def test_a_failure_while_writing_removes_the_half_built_subset(tmp_path, monkeypatch):
    def broken(path, passages):
        next(iter(passages))
        raise OSError("disk full")

    monkeypatch.setattr(subset_module, "write_passages", broken)
    with pytest.raises(OSError, match="disk full"):
        build_subset(sources(tmp_path), tmp_path / "sub", 40)
    assert not (tmp_path / "sub").exists() and not (tmp_path / "sub.part").exists()


def test_a_stale_staging_folder_is_replaced(tmp_path):
    source = sources(tmp_path)
    (tmp_path / "sub.part").mkdir()
    (tmp_path / "sub.part" / "junk").write_text("x")
    build_subset(source, tmp_path / "sub", 40)
    assert sorted(f.name for f in (tmp_path / "sub").iterdir()) == ["manifest.json", "passages.jsonl"]


def test_a_source_that_breaks_midway_leaves_no_subset(tmp_path):
    source = sources(tmp_path)
    with open(source, "ab") as stream:
        stream.write(b"this is not a passage\n")
    with pytest.raises(ValueError, match="not a passage"):
        build_subset(source, tmp_path / "sub", 40)
    assert not (tmp_path / "sub").exists() and not (tmp_path / "sub.part").exists()


def test_article_lists_skip_blank_lines_and_trim(tmp_path):
    (tmp_path / "ids.txt").write_text("a1\n\n  a2  \n\t\na3")
    assert read_article_list(tmp_path / "ids.txt") == ["a1", "a2", "a3"]


def test_command_line_builds_a_subset(tmp_path, capsys):
    source = sources(tmp_path)
    (tmp_path / "ids.txt").write_text("a2\n\na5\n")
    arguments = [str(source), "--out", str(tmp_path / "sub"), "--size", "40", "--seed", "3"]
    assert main(arguments + ["--article", "a8", "--must-include", str(tmp_path / "ids.txt")]) == 0
    assert f"of {sum(COUNTS.values())} passages" in capsys.readouterr().out
    manifest = json.loads((tmp_path / "sub" / "manifest.json").read_text())
    assert manifest["required"]["requested"] == 3 and manifest["required"]["found"] == 3
    assert manifest["parameters"] == {"size": 40, "seed": 3, "strict": False}


def test_the_command_line_seed_defaults_to_zero(tmp_path):
    source = sources(tmp_path)
    main([str(source), "--out", str(tmp_path / "default"), "--size", "40"])
    main([str(source), "--out", str(tmp_path / "zero"), "--size", "40", "--seed", "0"])
    assert json.loads((tmp_path / "default" / "manifest.json").read_text())["parameters"]["seed"] == 0
    assert (tmp_path / "default" / "passages.jsonl").read_bytes() == (tmp_path / "zero" / "passages.jsonl").read_bytes()


def test_command_line_refuses_bad_input(tmp_path):
    source = sources(tmp_path)
    base = [str(source), "--out", str(tmp_path / "bad")]
    for extra in (["--size", "0"], ["--size", "5", "--article", "ghost", "--strict"], []):
        with pytest.raises(SystemExit):
            main(base + extra)
    assert not (tmp_path / "bad").exists()


def test_a_required_article_is_searchable_in_the_subset_index(tmp_path):
    source = sources(tmp_path)
    build_subset(source, tmp_path / "sub", 20, ["a2"], seed=9)
    build_index(iter_passages(tmp_path / "sub" / "passages.jsonl"), tmp_path / "idx")
    hits = BM25Index(tmp_path / "idx").search("uniquea2x3", 3)
    assert hits[0].passage.id == "a2:3"


def test_the_bm25_command_line_builds_an_index_from_a_passages_file(tmp_path, capsys):
    source = sources(tmp_path)
    assert bm25_main([str(source), "--out", str(tmp_path / "idx"), "--k1", "1.2", "--b", "0.75", "--no-title"]) == 0
    assert f"{sum(COUNTS.values())} passages ->" in capsys.readouterr().out
    meta = json.loads((tmp_path / "idx" / "meta.json").read_text())
    assert (meta["k1"], meta["b"], meta["include_title"]) == (1.2, 0.75, False)
    assert BM25Index(tmp_path / "idx").search("uniquea4x1", 1)[0].passage.id == "a4:1"
    assert bm25_main([str(source), "--out", str(tmp_path / "idx2")]) == 0
    default = json.loads((tmp_path / "idx2" / "meta.json").read_text())
    assert (default["k1"], default["b"], default["include_title"]) == (0.9, 0.4, True)


@pytest.mark.parametrize("arguments", [["--k1", "0"], ["--b", "2"], []])
def test_the_bm25_command_line_refuses_bad_arguments(tmp_path, arguments):
    source = sources(tmp_path)
    with pytest.raises(SystemExit):
        bm25_main(([str(source), "--out", str(tmp_path / "bad")] + arguments) if arguments else [])
    assert not (tmp_path / "bad").exists()
