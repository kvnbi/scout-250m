import datetime
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scout.retrieval import build
from scout.retrieval.passages import Block, Section, iter_passages
from scout.retrieval.structured import iter_articles, parse_sections


def paragraph(value, *identifiers):
    return {"type": "paragraph", "value": value, "citations": [{"identifier": i, "text": "[x]"} for i in identifiers]}


def section(name, *parts):
    return {"type": "section", "name": name, "has_parts": list(parts)}


def item(value, *children):
    return {"type": "list_item", "value": value, "has_parts": list(children)}


SECTIONS = [
    section(
        "Abstract",
        {"type": "image", "images": []},
        paragraph("Lead sentence.", "cite_note-a"),
        paragraph("Second lead paragraph."),
    ),
    section(
        "History",
        paragraph("Early days."),
        section("Origins", paragraph("Roots."), section("Deep", paragraph("Deepest."))),
        paragraph("Back in history."),
        {"type": "table", "table_references": []},
    ),
    section(
        "Lists",
        {"type": "list", "has_parts": [item("apple"), item("pear", {"type": "list", "has_parts": [item("conference pear")]})]},
        {"type": "ordered_list", "has_parts": [item("first"), item("second")]},
        {"type": "definition_list", "has_parts": [{"type": "definition_term", "value": "Term"}, {"type": "definition", "value": "Meaning."}]},
        {"type": "gallery", "images": []},
        {"type": "paragraph", "citations": [{"identifier": "c", "text": "[3]"}]},
    ),
    section("References", paragraph("Hidden reference text.")),
    section("External links", section("Sub", paragraph("Hidden link text."))),
    section("Empty section"),
]


def names(parsed):
    return [(s.name, [(b.text, b.item) for b in s.blocks]) for s in parsed]


def test_sections_flatten_into_paths_with_blocks_in_reading_order():
    assert names(parse_sections(SECTIONS)) == [
        ("", [("Lead sentence.", False), ("Second lead paragraph.", False)]),
        ("History", [("Early days.", False)]),
        ("History > Origins", [("Roots.", False)]),
        ("History > Origins > Deep", [("Deepest.", False)]),
        ("History", [("Back in history.", False)]),
        (
            "Lists",
            [("apple", True), ("pear", True), ("conference pear", True), ("first", True), ("second", True), ("Term", True), ("Meaning.", True)],
        ),
    ]


def test_nested_lists_are_flattened_wherever_they_sit():
    direct = {"type": "list", "has_parts": [item("a"), {"type": "list", "has_parts": [item("b")]}]}
    assert names(parse_sections([section("Abstract", direct)])) == [("", [("a", True), ("b", True)])]


def test_only_the_leading_abstract_loses_its_name():
    nodes = [section("Abstract", paragraph("Lead.")), section("History", section("Abstract", paragraph("Deep.")))]
    assert names(parse_sections(nodes)) == [("", [("Lead.", False)]), ("History > Abstract", [("Deep.", False)])]


def test_skipped_node_types_are_not_searched_for_text():
    nodes = [section("Abstract", {"type": "table", "has_parts": [paragraph("cell text")]}, paragraph("Real."))]
    assert names(parse_sections(nodes)) == [("", [("Real.", False)])]


def test_titles_are_trimmed(tmp_path):
    path = write(tmp_path, "f.parquet", [dict(row(1, "T."), name="  Spaced  ")])
    assert [a.title for a in iter_articles(path)] == ["Spaced"]


@pytest.mark.parametrize(
    "residue",
    ["{{Infobox company | name = X}}", "{{Infobox company | name = X", "text }} more", "{| class=wikitable", "cell |}", "see <ref name=a>x</ref>", "close </REF>"],
)
def test_blocks_with_unparsed_markup_are_dropped(residue):
    nodes = [section("Abstract", paragraph("Good text."), paragraph(residue), {"type": "list", "has_parts": [item(residue), item("Good item")]})]
    assert names(parse_sections(nodes)) == [("", [("Good text.", False), ("Good item", True)])]


def test_values_that_are_not_text_are_ignored():
    nodes = [section("Abstract", {"type": "paragraph", "value": 5}, {"type": "list", "has_parts": [{"type": "list_item", "value": 7}]}, paragraph("Real."))]
    assert names(parse_sections(nodes)) == [("", [("Real.", False)])]


def test_ordinary_punctuation_is_not_mistaken_for_markup():
    text = "A set {x} or a pipe | in text, a [1] note, a < b, and a reference to ref points."
    assert names(parse_sections([section("Abstract", paragraph(text))])) == [("", [(text, False)])]


def test_odd_inputs_yield_nothing():
    assert parse_sections(None) == []
    assert parse_sections("not a list") == []
    assert parse_sections([None, 3, {"type": "paragraph"}, section("Abstract", {"type": "paragraph", "value": "  "})]) == []


def test_unknown_container_types_are_searched_for_text():
    nodes = [section("Abstract", {"type": "infobox_like", "has_parts": [paragraph("Inside a wrapper.")]})]
    assert names(parse_sections(nodes)) == [("", [("Inside a wrapper.", False)])]


def parquet_bytes(rows, modified_type=pa.timestamp("us")):
    stamps = [r.get("modified", "2024-09-16T10:00:00") for r in rows]
    if modified_type == pa.timestamp("us"):
        stamps = [None if x is None else datetime.datetime.fromisoformat(x) for x in stamps]
    table = pa.table(
        {
            "name": [r["name"] for r in rows],
            "identifier": pa.array([r["identifier"] for r in rows], pa.int64()),
            "url": [r["url"] for r in rows],
            "sections": [json.dumps(r["sections"]) for r in rows],
            "date_modified": pa.array(stamps, modified_type),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def write(tmp_path, name, rows, **kwargs):
    path = tmp_path / name
    path.write_bytes(parquet_bytes(rows, **kwargs))
    return path


def row(identifier, text, modified="2024-09-16T10:00:00", name="Example", url="u"):
    return {"name": name, "identifier": identifier, "url": url, "modified": modified, "sections": [section("Abstract", paragraph(text))]}


ROWS = [
    {"name": "Example", "identifier": 42, "url": "https://en.wikipedia.org/wiki/Example", "sections": SECTIONS, "modified": "2024-09-16T10:30:00"},
    {"name": "Picture only", "identifier": 7, "url": None, "sections": [section("Abstract", {"type": "image"})]},
    {"name": "Old title", "identifier": 8, "url": "u", "sections": [section("Abstract", paragraph("#REDIRECT [[Example]]"))]},
    {"name": "Mercury", "identifier": 9, "url": "u", "sections": [section("Abstract", paragraph("Mercury may refer to:"))]},
    {"name": "Mercury (disambiguation)", "identifier": 10, "url": "u", "sections": [section("Abstract", paragraph("x."))]},
    {"name": "No url", "identifier": 11, "url": None, "sections": [section("Abstract", paragraph("Text."))], "modified": "2025-01-02T03:04:05"},
]


def test_articles_come_from_parquet_with_ids_urls_and_their_own_revision_dates(tmp_path):
    path = write(tmp_path, "f.parquet", ROWS)
    articles = list(iter_articles(path))
    assert [(a.id, a.title, a.url, a.date) for a in articles] == [
        ("42", "Example", "https://en.wikipedia.org/wiki/Example", "2024-09-16"),
        ("11", "No url", "", "2025-01-02"),
    ]
    assert articles[0].sections[0] == Section("", (Block("Lead sentence."), Block("Second lead paragraph.")))


def test_a_given_date_overrides_the_revision_dates(tmp_path):
    path = write(tmp_path, "f.parquet", ROWS)
    assert {a.date for a in iter_articles(path, "2020-01-01")} == {"2020-01-01"}


def test_each_article_comes_from_its_latest_revision_only(tmp_path):
    rows = [
        row(1, "Old text.", "2026-01-01T00:00:00"),
        row(1, "Newest text.", "2026-03-01T12:00:00"),
        row(1, "Middle text.", "2026-02-01T00:00:00"),
        row(2, "Other article.", "2026-01-01T00:00:00", name="Other"),
    ]
    articles = list(iter_articles(write(tmp_path, "f.parquet", rows)))
    assert [(a.id, a.date, a.sections[0].blocks[0].text) for a in articles] == [
        ("1", "2026-03-01", "Newest text."),
        ("2", "2026-01-01", "Other article."),
    ]


def test_the_latest_revision_wins_across_files_whatever_their_order(tmp_path):
    older = write(tmp_path, "a.parquet", [row(1, "Older text.", "2025-05-01T00:00:00"), row(3, "Only in a.", "2025-05-01T00:00:00", name="A")])
    newer = write(tmp_path, "b.parquet", [row(1, "Newer text.", "2026-05-01T00:00:00"), row(4, "Only in b.", "2025-05-01T00:00:00", name="B")])
    for order in ([older, newer], [newer, older]):
        found = {a.id: a.sections[0].blocks[0].text for a in iter_articles(order)}
        assert found == {"1": "Newer text.", "3": "Only in a.", "4": "Only in b."}


def test_equal_timestamps_go_to_the_later_row_and_the_later_file(tmp_path):
    same = "2026-01-01T00:00:00"
    within = write(tmp_path, "w.parquet", [row(1, "First.", same), row(1, "Second.", same)])
    assert [a.sections[0].blocks[0].text for a in iter_articles(within)] == ["Second."]
    first = write(tmp_path, "x.parquet", [row(1, "From x.", same)])
    second = write(tmp_path, "y.parquet", [row(1, "From y.", same)])
    assert [a.sections[0].blocks[0].text for a in iter_articles([first, second])] == ["From y."]
    assert [a.sections[0].blocks[0].text for a in iter_articles([second, first])] == ["From x."]


def test_ties_across_files_go_to_the_later_file_even_when_its_row_comes_earlier(tmp_path):
    same = "2026-01-01T00:00:00"
    first = write(tmp_path, "x.parquet", [row(5, "Filler.", same, name="Filler"), row(1, "From x.", same)])
    second = write(tmp_path, "y.parquet", [row(1, "From y.", same)])
    found = {a.id: a.sections[0].blocks[0].text for a in iter_articles([first, second])}
    assert found == {"5": "Filler.", "1": "From y."}


def test_identifiers_stored_as_text_are_understood(tmp_path):
    table = pa.table(
        {
            "name": ["A", "A", "B"],
            "identifier": pa.array(["10", "10", "9"], pa.string()),
            "url": ["u", "u", "u"],
            "sections": [json.dumps(r["sections"]) for r in (row(0, "Old."), row(0, "New."), row(0, "Other."))],
            "date_modified": pa.array([datetime.datetime(2026, 1, 1), datetime.datetime(2026, 2, 1), datetime.datetime(2026, 1, 1)], pa.timestamp("us")),
        }
    )
    path = tmp_path / "f.parquet"
    pq.write_table(table, path)
    assert sorted((a.id, a.sections[0].blocks[0].text) for a in iter_articles(path)) == [("10", "New."), ("9", "Other.")]


def test_a_newer_redirect_removes_the_article_instead_of_resurrecting_an_old_revision(tmp_path):
    rows = [row(1, "Real article text.", "2026-01-01T00:00:00"), row(1, "#REDIRECT [[Elsewhere]]", "2026-02-01T00:00:00")]
    assert list(iter_articles(write(tmp_path, "f.parquet", rows))) == []


def test_rows_without_an_identifier_are_ignored_and_missing_dates_lose(tmp_path):
    rows = [row(None, "No id.", "2026-01-01T00:00:00"), row(1, "Undated.", None), row(1, "Dated.", "2020-01-01T00:00:00"), row(2, "Alone.", None, name="B")]
    articles = list(iter_articles(write(tmp_path, "f.parquet", rows)))
    assert [(a.id, a.date, a.sections[0].blocks[0].text) for a in articles] == [("1", "2020-01-01", "Dated."), ("2", "", "Alone.")]


def test_text_timestamps_are_understood_too(tmp_path):
    rows = [row(1, "Old.", "2026-01-01T00:00:00"), row(1, "New.", "2026-06-01T00:00:00")]
    path = write(tmp_path, "f.parquet", rows, modified_type=pa.string())
    assert [(a.date, a.sections[0].blocks[0].text) for a in iter_articles(path)] == [("2026-06-01", "New.")]


def test_no_files_no_articles():
    assert list(iter_articles([])) == []


def test_command_line_builds_passages_from_structured_files(tmp_path, capsys):
    path = write(tmp_path, "f.parquet", ROWS)
    other = write(tmp_path, "g.parquet", [row(500, "Extra article text.", "2030-02-03T00:00:00", name="Extra")])
    out = tmp_path / "passages.jsonl"
    assert build.main(["--format", "structured", "--out", str(out), str(path), str(other)]) == 0
    assert capsys.readouterr().out.startswith("3 articles, 8 passages")
    passages = list(iter_passages(out))
    assert passages[0].id == "42:0" and passages[0].section == "" and passages[0].date == "2024-09-16"
    assert passages[0].text == "Lead sentence.\nSecond lead paragraph."
    assert passages[-1].date == "2030-02-03"
    assert "Hidden" not in "".join(p.text for p in passages)
    assert len({p.id for p in passages}) == len(passages)


def test_a_date_argument_belongs_to_wikitext_only(tmp_path):
    path = write(tmp_path, "f.parquet", ROWS)
    with pytest.raises(SystemExit):
        build.main(["--format", "structured", "--date", "2024-09-16", "--out", str(tmp_path / "o.jsonl"), str(path)])
    assert not (tmp_path / "o.jsonl").exists()


def test_a_file_without_the_expected_columns_is_refused(tmp_path):
    path = tmp_path / "bad.parquet"
    buffer = io.BytesIO()
    pq.write_table(pa.table({"name": ["x"]}), buffer)
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError):
        list(iter_articles(path))
