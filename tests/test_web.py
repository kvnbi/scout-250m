import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scout.data.fetch import sha256_of
from scout.retrieval import web
from scout.retrieval.bm25 import BM25Index, build_index
from scout.retrieval.dates import AsOf
from scout.retrieval.passages import iter_passages
from scout.retrieval.search import render_hits
from scout.retrieval.web import (
    Page,
    build_web_corpus,
    clean_id,
    crawl_date,
    excluded,
    hostname,
    iter_fineweb_pages,
    looks_like_title,
    main,
    page_article,
    page_date,
    url_date,
    url_title,
)

WARC = "s3://commoncrawl/crawl-data/CC-MAIN-2023-06/segments/1674764495012.84/warc/CC-MAIN-20230127195946-20230127225946-00606.warc.gz"


@pytest.mark.parametrize(
    "file_path, dump, expected",
    [
        (WARC, "CC-MAIN-2023-06", "2023-01-27"),
        (WARC, "CC-MAIN-2019-51", "2023-01-27"),
        (WARC, None, "2023-01-27"),
        ("s3://x/CC-MAIN-20230127.warc.gz", "CC-MAIN-2023-06", "2023-02"),
        ("", "CC-MAIN-2023-06-extra", ""),
        ("", "xCC-MAIN-2023-06", ""),
        ("s3://x/CC-MAIN-20231327195946-1.warc.gz", "CC-MAIN-2023-06", "2023-02"),
        ("s3://x/CC-MAIN-20230230195946-1.warc.gz", "CC-MAIN-2023-06", "2023-02"),
        ("", "CC-MAIN-2023-06", "2023-02"),
        (None, "CC-MAIN-2019-51", "2019-12"),
        ("", "CC-MAIN-2020-01", "2020-01"),
        ("", "CC-MAIN-2025-01", "2025-01"),
        ("", "CC-MAIN-2019-01", "2019-01"),
        ("", "CC-MAIN-2020-53", "2020-12"),
        ("", "CC-MAIN-2020-54", ""),
        ("", "CC-MAIN-2020-00", ""),
        ("", "CC-MAIN-2020-5", ""),
        ("", "nonsense", ""),
        ("", "", ""),
        (None, None, ""),
    ],
)
def test_the_crawl_date_comes_from_the_warc_stamp_else_the_iso_week_of_the_dump(file_path, dump, expected):
    assert crawl_date(file_path, dump) == expected


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://uwstaffnews.wordpress.com/2020/08/06/lockdown-puzzle-word-ladder/", "2020-08-06"),
        ("https://x.com/2019/03/some-post", "2019-03"),
        ("https://x.com/archive/1999/12/31/", "1999-12-31"),
        ("https://x.com/2020/08/06", "2020-08-06"),
        ("https://x.com/2020/08/06?utm=1#top", "2020-08-06"),
        ("https://x.com/blog/2021/13/05/x", ""),
        ("https://x.com/2020/02/30/z", ""),
        ("https://x.com/2020/08/00/z", "2020-08"),
        ("https://x.com/2020/08/32/z", "2020-08"),
        ("https://x.com/2020/00/10/z", ""),
        ("https://x.com/1899/05/05/z", ""),
        ("https://x.com/2100/05/05/z", ""),
        ("https://x.com/20/08/06/", ""),
        ("https://x.com/?p=2020/08/06", ""),
        ("https://x.com/page/2020", ""),
        ("https://x.com/2020/08/06x/", "2020-08"),
        ("", ""),
    ],
)
def test_blog_style_urls_carry_their_publication_date(url, expected):
    assert url_date(url) == expected


@pytest.mark.parametrize(
    "url, crawled, expected",
    [
        ("https://x.com/2020/08/06/p", "2023-01-27", "2020-08-06"),
        ("https://x.com/2023/01/27/p", "2023-01-27", "2023-01-27"),
        ("https://x.com/2025/08/06/p", "2023-01-27", "2023-01-27"),
        ("https://x.com/2020/08/p", "", "2020-08"),
        ("https://x.com/p", "2023-01-27", "2023-01-27"),
        ("https://x.com/p", "", ""),
        ("https://x.com/2023/01/p", "2023-01-27", "2023-01"),
        ("https://x.com/2023/02/p", "2023-01-27", "2023-01-27"),
    ],
)
def test_a_page_is_dated_by_its_url_unless_that_is_later_than_the_crawl(url, crawled, expected):
    assert page_date(url, crawled) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("<urn:uuid:7dac48da-7e8b-4c52-9c1b-3f5e8a1d2b4c>", "7dac48da-7e8b-4c52-9c1b-3f5e8a1d2b4c"),
        ("plain-id", "plain-id"),
        ("  padded  ", "padded"),
        ("a:b c:d", "a-b-c-d"),
        ("<urn:uuid:unclosed", "<urn-uuid-unclosed"),
        ("", ""),
        (None, ""),
    ],
)
def test_page_ids_never_contain_colons_or_spaces(raw, expected):
    assert clean_id(raw) == expected


def test_hosts_and_excluded_domains():
    domains = ("wikipedia.org", "example.com")
    assert hostname("https://EN.Wikipedia.org:8080/wiki/X") == "en.wikipedia.org" and hostname("") == ""
    for url in ("https://wikipedia.org/x", "https://en.wikipedia.org/wiki/X", "http://en.m.wikipedia.org/", "https://a.b.example.com/p"):
        assert excluded(url, domains)
    for url in ("https://notwikipedia.org/x", "https://wikipedia.org.evil.com/x", "https://example.org/", "", "not a url"):
        assert not excluded(url, domains)
    assert not excluded("https://en.wikipedia.org/x", ())


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://garbagewonk.com/composting-ratio/", "Composting ratio"),
        ("https://x.com/rafting/floating-down-the-allegheny-river/", "Floating down the allegheny river"),
        ("https://x.com/a/how_to_bake_bread.html", "How to bake bread"),
        ("https://x.com/a/how+to+bake+bread.php", "How to bake bread"),
        ("https://x.com/a/how%20to%20bake%20bread", "How to bake bread"),
        ("https://x.com/guides/bread-baking/12345", "Bread baking"),
        ("https://x.com/guides/bread-baking/12345.html", "Bread baking"),
        ("https://x.com/articles/ab", "Articles"),
        ("https://x.com/support/faq", "Faq"),
        ("https://x.com/support/ab/", "Support"),
        ("https://x.com/", "x.com"),
        ("https://sub.example.org", "sub.example.org"),
        ("https://x.com/p?id=7", "x.com"),
        ("https://x.com/" + "very-long-title-" * 20, ("Very long title " + "very long title " * 19).strip()[:100]),
        ("", ""),
    ],
)
def test_titles_fall_back_to_a_readable_form_of_the_url(url, expected):
    assert url_title(url) == expected


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Measles", True),
        ("How a Memory Foam Mattress Is Made", True),
        ("Is this a title?", True),
        ("Stop!", True),
        ("This is a full sentence.", False),
        ("Ends with a colon:", False),
        ("Ends with a comma,", False),
        ("Ends with a semicolon;", False),
        (" ".join(["word"] * 12), True),
        (" ".join(["word"] * 13), False),
        ("x" * 100, True),
        ("x" * 101, False),
        ("", False),
        ("   ", False),
    ],
)
def test_a_short_line_without_closing_punctuation_is_a_title(line, expected):
    assert looks_like_title(line) is expected


BODY = " ".join(f"word{i}" for i in range(80))


def test_a_leading_headline_becomes_the_title_and_leaves_the_body():
    article = page_article(Page("p1", "https://x.com/a/slug-here", f"Real Title\n{BODY}\nSecond paragraph here.", "2023-01-27"))
    assert (article.id, article.title, article.url, article.date) == ("p1", "Real Title", "https://x.com/a/slug-here", "2023-01-27")
    assert [b.text for b in article.sections[0].blocks] == [BODY, "Second paragraph here."]
    assert [s.name for s in article.sections] == [""]


def test_without_a_headline_the_title_comes_from_the_url_and_every_line_is_body():
    article = page_article(Page("p1", "https://x.com/a/slug-here-now", f"{BODY}.\nAnother line.", "2023-01-27"))
    assert article.title == "Slug here now"
    assert [b.text for b in article.sections[0].blocks] == [f"{BODY}.", "Another line."]


def test_whitespace_and_blank_lines_are_cleaned():
    text = f"  Title  here \n\n   \n{BODY}\n\t\nsecond   line  "
    article = page_article(Page("p", "https://x.com/z", text, ""))
    assert article.title == "Title here"
    assert [b.text for b in article.sections[0].blocks] == [BODY, "second line"]


def test_a_page_of_one_line_is_body_not_title():
    article = page_article(Page("p", "https://x.com/some-topic-page", BODY, ""))
    assert article.title == "Some topic page" and [b.text for b in article.sections[0].blocks] == [BODY]


def test_short_pages_are_dropped_by_the_minimum():
    page = Page("p", "https://x.com/a", "Title\n" + " ".join(["w"] * 48), "")
    assert page_article(page) is None
    assert page_article(Page("p", "https://x.com/a", "Title\n" + " ".join(["w"] * 49), "")) is not None
    assert page_article(page, min_page_words=40) is not None
    assert page_article(page, min_page_words=50) is None


def test_a_single_short_line_is_the_body_when_the_page_is_allowed_to_be_tiny():
    article = page_article(Page("p", "https://x.com/some-topic-page", "Short Title", ""), min_page_words=0)
    assert article.title == "Some topic page" and [b.text for b in article.sections[0].blocks] == ["Short Title"]


def write_parquet(path, rows):
    columns = ("text", "id", "dump", "url", "file_path", "language")
    table = pa.table(
        {
            **{c: [r.get(c) for r in rows] for c in columns},
            "language_score": [0.99] * len(rows),
            "token_count": [100] * len(rows),
            "score": [3.1] * len(rows),
            "int_score": [3] * len(rows),
        }
    )
    pq.write_table(table, path, row_group_size=2)
    return path


def row(name, url="", text=None, dump="CC-MAIN-2023-06", file_path=WARC):
    return {
        "text": text if text is not None else f"{name} Title\n{BODY} {name}.",
        "id": f"<urn:uuid:{name}>",
        "url": url,
        "dump": dump,
        "file_path": file_path,
        "language": "en",
    }


@pytest.fixture
def sources(tmp_path):
    first = write_parquet(
        tmp_path / "a.parquet",
        [
            row("a1", "https://alpha.org/post/one"),
            row("a2", "https://en.wikipedia.org/wiki/Two"),
            row("a3", "https://short.org/s", text="Tiny\ntoo short"),
            row("a4", "https://alpha.org/post/one"),
            row("a5", "https://beta.net/2020/08/06/dated-post"),
            row("a6", ""),
            row("a7", "https://gamma.io/undated", dump=None, file_path=None),
        ],
    )
    second = write_parquet(
        tmp_path / "b.parquet",
        [
            row("b1", "https://alpha.org/post/one"),
            row("b2", "https://de.wikipedia.org/wiki/Zwei"),
            row("b3", "https://alpha.org/post/three"),
        ],
    )
    return [first, second]


def test_pages_are_read_with_cleaned_ids_and_dates(sources):
    pages = list(iter_fineweb_pages(sources[0]))
    assert [p.id for p in pages][:2] == ["a1", "a2"]
    assert (pages[0].date, pages[0].crawled) == ("2023-01-27", "2023-01-27")
    assert pages[4].date == "2020-08-06" and pages[4].crawled == "2023-01-27"
    assert pages[5].url == "" and (pages[6].date, pages[6].crawled) == ("", "")


def test_a_corpus_keeps_distinct_real_pages_and_accounts_for_every_other_one(sources, tmp_path):
    manifest = build_web_corpus(sources, tmp_path / "web")
    assert manifest["pages"] == {
        "read": 10,
        "kept": 4,
        "dropped": {"duplicate_url": 2, "excluded_domain": 2, "no_url": 1, "too_short": 1},
    }
    passages = list(iter_passages(tmp_path / "web" / "passages.jsonl"))
    assert manifest["passages"] == len(passages)
    assert {p.id.split(":")[0] for p in passages} == {"a1", "a5", "a7", "b3"}
    assert len({p.id for p in passages}) == len(passages)
    by_page = {p.id.split(":")[0]: p for p in passages}
    assert by_page["a1"].title == "a1 Title" and by_page["a1"].url == "https://alpha.org/post/one"
    assert by_page["a5"].date == "2020-08-06" and by_page["a1"].date == "2023-01-27" and by_page["a7"].date == ""
    assert manifest["domains"] == 3
    assert manifest["dates"] == {"first": "2020-08-06", "last": "2023-01-27", "undated": 1, "from_url": 1}


def test_the_manifest_describes_inputs_parameters_and_the_output_exactly(sources, tmp_path):
    manifest = build_web_corpus(sources, tmp_path / "web", max_words=50, min_words=10, min_page_words=60)
    assert json.loads((tmp_path / "web" / "manifest.json").read_text()) == manifest
    assert manifest["format"] == 1 and manifest["source"] == "fineweb-edu"
    assert [(i["name"], i["bytes"], i["sha256"]) for i in manifest["inputs"]] == [
        (p.name, p.stat().st_size, sha256_of(p)) for p in sources
    ]
    assert manifest["parameters"] == {
        "exclude_domains": ["wikipedia.org"], "min_page_words": 60, "max_pages": None, "max_words": 50, "min_words": 10,
    }
    assert manifest["passages_sha256"] == sha256_of(tmp_path / "web" / "passages.jsonl")
    assert sorted(f.name for f in (tmp_path / "web").iterdir()) == ["manifest.json", "passages.jsonl"]


def test_building_twice_gives_identical_bytes(sources, tmp_path):
    build_web_corpus(sources, tmp_path / "one")
    build_web_corpus(sources, tmp_path / "two")
    for name in ("passages.jsonl", "manifest.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_passages_respect_the_word_limits(sources, tmp_path):
    build_web_corpus(sources, tmp_path / "web", max_words=30, min_words=10)
    lengths = [len(p.text.split()) for p in iter_passages(tmp_path / "web" / "passages.jsonl")]
    assert max(lengths) <= 30 + 10 - 1 and len(lengths) > 4


@pytest.mark.parametrize("kwargs", [dict(max_words=0, min_words=0), dict(min_words=-1), dict(max_words=10, min_words=11)])
def test_word_limits_are_checked_even_when_no_page_survives(tmp_path, kwargs):
    only_wikipedia = write_parquet(tmp_path / "w.parquet", [row("w1", "https://en.wikipedia.org/wiki/X")])
    with pytest.raises(ValueError):
        build_web_corpus([only_wikipedia], tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.part").exists()


def test_a_page_dated_only_by_its_url_counts_as_dated_from_the_url(tmp_path):
    path = write_parquet(tmp_path / "u.parquet", [row("u1", "https://x.com/2020/08/06/post", dump=None, file_path=None)])
    manifest = build_web_corpus([path], tmp_path / "out")
    assert manifest["dates"] == {"first": "2020-08-06", "last": "2020-08-06", "undated": 0, "from_url": 1}


def test_the_page_limit_stops_right_after_the_last_kept_page(sources, tmp_path):
    manifest = build_web_corpus(sources, tmp_path / "web", max_pages=2)
    assert manifest["pages"]["kept"] == 2 and manifest["pages"]["read"] == 5
    assert manifest["pages"]["dropped"] == {"duplicate_url": 1, "excluded_domain": 1, "too_short": 1}
    assert {p.id.split(":")[0] for p in iter_passages(tmp_path / "web" / "passages.jsonl")} == {"a1", "a5"}


def test_excluded_domains_can_be_changed_or_emptied(sources, tmp_path):
    kept = build_web_corpus(sources, tmp_path / "all", exclude_domains=())
    assert kept["pages"]["kept"] == 6 and "excluded_domain" not in kept["pages"]["dropped"]
    only = build_web_corpus(sources, tmp_path / "beta", exclude_domains=("beta.net", "gamma.io"))
    assert only["pages"]["kept"] == 4
    assert only["pages"]["dropped"] == {"duplicate_url": 2, "excluded_domain": 2, "no_url": 1, "too_short": 1}
    ids = {p.id.split(":")[0] for p in iter_passages(tmp_path / "beta" / "passages.jsonl")}
    assert "a5" not in ids and "a7" not in ids and {"a1", "a2", "b2", "b3"} <= ids


def test_refusals_leave_nothing_behind(sources, tmp_path):
    build_web_corpus(sources, tmp_path / "web")
    with pytest.raises(FileExistsError):
        build_web_corpus(sources, tmp_path / "web")
    for kwargs in (
        dict(source="nope"), dict(min_page_words=-1), dict(max_pages=0), dict(max_words=0), dict(min_words=-1), dict(min_words=200),
    ):
        with pytest.raises(ValueError):
            build_web_corpus(sources, tmp_path / "bad", **kwargs)
        assert not (tmp_path / "bad").exists() and not (tmp_path / "bad.part").exists()


def test_a_failing_source_leaves_no_corpus_and_a_stale_folder_is_replaced(sources, tmp_path, monkeypatch):
    def broken(path):
        yield Page("x", "https://x.com/a", BODY, "")
        raise OSError("disk went away")

    monkeypatch.setitem(web.READERS, "broken", broken)
    with pytest.raises(OSError, match="disk went away"):
        build_web_corpus(sources, tmp_path / "web", source="broken")
    assert not (tmp_path / "web").exists() and not (tmp_path / "web.part").exists()
    (tmp_path / "again.part").mkdir()
    (tmp_path / "again.part" / "junk").write_text("x")
    build_web_corpus(sources, tmp_path / "again")
    assert sorted(f.name for f in (tmp_path / "again").iterdir()) == ["manifest.json", "passages.jsonl"]


def test_command_line_builds_a_corpus(sources, tmp_path, capsys):
    arguments = [str(p) for p in sources] + ["--out", str(tmp_path / "web")]
    assert main(arguments) == 0
    assert capsys.readouterr().out.startswith("4 of 10 pages,")
    assert main(arguments[:-1] + [str(tmp_path / "limited"), "--max-pages", "1", "--exclude-domain", "beta.net"]) == 0
    assert json.loads((tmp_path / "limited" / "manifest.json").read_text())["parameters"]["exclude_domains"] == ["beta.net"]
    assert json.loads((tmp_path / "limited" / "manifest.json").read_text())["pages"]["kept"] == 1


@pytest.mark.parametrize("arguments", [["--max-pages", "0"], ["--max-words", "0"], ["--min-words", "500"], ["--min-page-words", "-1"], ["--source", "x"]])
def test_command_line_refuses_bad_arguments(sources, tmp_path, arguments):
    with pytest.raises(SystemExit):
        main([str(sources[0]), "--out", str(tmp_path / "bad")] + arguments)
    assert not (tmp_path / "bad").exists()


def test_a_frozen_web_corpus_is_searchable_and_dated_like_a_real_source(sources, tmp_path):
    build_web_corpus(sources, tmp_path / "web")
    build_index(iter_passages(tmp_path / "web" / "passages.jsonl"), tmp_path / "idx")
    index = BM25Index(tmp_path / "idx", name="web")
    hits = index.search("a5 Title", 3)
    assert hits[0].passage.url.startswith("https://beta.net/")
    assert render_hits(hits[:1]).startswith("[1] a5 Title (beta.net, 2020-08-06)\n")
    assert {h.passage.id.split(":")[0] for h in AsOf(index, "2021-01-01").search(BODY, 10)} == {"a5", "a7"}
    assert {h.passage.id.split(":")[0] for h in AsOf(index, "2021-01-01", keep_undated=False).search(BODY, 10)} == {"a5"}
