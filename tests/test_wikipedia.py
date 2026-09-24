import hashlib
import io
import json
import random
import re

import pyarrow as pa
import pyarrow.parquet as pq

from scout.data.wikipedia import DEFAULT_FOLDER, REPO, iter_records, main, reference_texts, render_article

NEL = chr(0x85)
LINE_SEPARATOR = chr(0x2028)


def reference(identifier, value):
    return {"identifier": identifier, "metadata": "{}", "source": None, "text": {"links": [], "value": value}, "type": "web"}


def paragraph(value, *identifiers):
    return {"type": "paragraph", "value": value, "citations": [{"identifier": i, "text": "[x]"} for i in identifiers]}


def section(name, *parts):
    return {"type": "section", "name": name, "has_parts": list(parts)}


REFERENCES = [
    reference("cite_note-a", "Alpha source."),
    reference("cite_note-b", "Beta" + chr(10) + "  continued" + NEL + "and" + LINE_SEPARATOR + "ended."),
    reference("cite_note-c", "Gamma source."),
    reference("cite_note-empty", "   "),
    reference("cite_note-group-1", "First group source."),
    reference("cite_note-other-group-1", "Second group source."),
]

SECTIONS = [
    section(
        "Abstract",
        {"type": "image", "images": []},
        paragraph("Lead sentence.", "cite_note-b", "cite_note-a", "cite_note-b"),
        paragraph("No sources here."),
    ),
    section(
        "History",
        paragraph("Early days.", "cite_note-a", "cite_note-cnote_a", "cite_note-empty"),
        section(
            "Origins",
            paragraph("Roots.", "cite_note-c"),
            section("Deep", paragraph("Deepest.")),
        ),
        {"type": "table", "table_references": []},
    ),
    section(
        "Lists",
        {"type": "list", "has_parts": [
            {"type": "list_item", "value": "apple"},
            {"type": "list_item", "value": "pear", "has_parts": [
                {"type": "list", "has_parts": [{"type": "list_item", "value": "conference pear"}]},
            ]},
        ]},
        {"type": "ordered_list", "has_parts": [
            {"type": "list_item", "value": "first"},
            {"type": "list_item", "value": "second"},
        ]},
        paragraph("After lists.", "cite_note-group-1", "cite_note-other-group-1"),
        {"type": "definition_list", "has_parts": [
            {"type": "definition_term", "value": "Term"},
            {"type": "definition", "value": "Meaning."},
        ]},
        {"type": "paragraph", "citations": [{"identifier": "cite_note-c", "text": "[3]"}]},
        {"type": "gallery", "images": []},
    ),
    section("References", {"type": "list", "has_parts": [{"type": "list_item", "value": "A hand written source list."}]}),
]

EXPECTED = "\n\n".join(
    [
        "# Example",
        "Lead sentence.[1][2]",
        "No sources here.",
        "## History",
        "Early days.[2]",
        "### Origins",
        "Roots.[3]",
        "#### Deep",
        "Deepest.",
        "## Lists",
        "- apple\n- pear\n  - conference pear\n1. first\n2. second",
        "After lists.[4][5]",
        "Term\n: Meaning.",
        "## References",
        "[1] Beta continued and ended.\n[2] Alpha source.\n[3] Gamma source.\n[4] First group source.\n[5] Second group source.",
    ]
)


def test_renders_structure_citations_and_resolved_references():
    text, citations = render_article("Example", SECTIONS, REFERENCES)
    assert text == EXPECTED
    assert citations == 5


def test_reference_texts_keep_one_line_each_and_drop_empty_ones():
    texts = reference_texts(REFERENCES)
    assert texts["cite_note-b"] == "Beta continued and ended."
    assert "cite_note-empty" not in texts
    assert reference_texts(None) == {}


def test_article_without_sources_has_no_reference_list():
    text, citations = render_article("Plain", [section("Abstract", paragraph("Just text.", "cite_note-missing"))], [])
    assert text == "# Plain\n\nJust text."
    assert citations == 0


def test_sections_without_text_leave_no_heading():
    sections = [
        section("Abstract", paragraph("Lead.")),
        section("Charts", {"type": "table"}, section("Weekly", {"type": "table"})),
        section("Career", section("Early", paragraph("Began.")), section("Empty", {"type": "gallery"})),
        section("Footnotes"),
    ]
    text, _ = render_article("Headings", sections, [])
    assert text == "# Headings\n\nLead.\n\n## Career\n\n### Early\n\nBegan."


def test_headings_never_exceed_six_levels():
    node = paragraph("Bottom.")
    for level in range(8):
        node = section(f"Level {level}", node)
    text, _ = render_article("Deep", [section("Abstract", paragraph("Lead.")), node], [])
    assert max(len(line.split(" ", 1)[0]) for line in text.splitlines() if line.startswith("#")) == 6


def test_unexpected_sections_value_renders_nothing():
    assert render_article("Odd", None, []) == ("", 0)


def test_redirect_pages_are_skipped_but_mentions_are_kept():
    redirect = [section("Abstract", paragraph("Notice text."), paragraph("#REDIRECT Amnesia"))]
    assert render_article("Memory problems", redirect, []) == ("", 0)
    assert render_article("Lower case", [section("Abstract", paragraph("# redirect Elsewhere"))], []) == ("", 0)
    mention = [section("Abstract", paragraph("A page starting with #REDIRECT points elsewhere."))]
    assert render_article("MediaWiki", mention, [])[0] == "# MediaWiki\n\nA page starting with #REDIRECT points elsewhere."


def test_article_without_text_renders_nothing():
    assert render_article("Empty", [section("Abstract", {"type": "image", "images": []})], REFERENCES) == ("", 0)


def random_article(rng):
    identifiers = [f"cite_note-{i}" for i in range(rng.randint(0, 12))]
    references = [reference(i, rng.choice(["Source", "Line one" + chr(10) + "line two", " "])) for i in identifiers]
    unresolved = ["cite_note-cnote_a", "cite_note-page"]

    def build(depth):
        parts = []
        for _ in range(rng.randint(0, 4)):
            roll = rng.random()
            if roll < 0.5:
                cited = rng.sample(identifiers + unresolved, rng.randint(0, min(4, len(identifiers) + 2)))
                parts.append(paragraph(rng.choice(["Text.", "More text.", ""]), *cited))
            elif roll < 0.7 and depth < 4:
                parts.append(section(rng.choice(["History", "References", "Notes"]), *build(depth + 1)))
            elif roll < 0.85:
                parts.append({"type": "list", "has_parts": [{"type": "list_item", "value": "item"}]})
            else:
                parts.append({"type": rng.choice(["image", "table", "gallery"])})
        return parts

    return [section("Abstract", *build(0))] + [section("Part", *build(1)) for _ in range(rng.randint(0, 3))], references


def test_every_marker_resolves_on_random_articles():
    rng = random.Random(0)
    for _ in range(500):
        sections, references = random_article(rng)
        text, citations = render_article("Random", sections, references)
        if not text:
            continue
        body, _, listing = text.partition("\n\n## References\n\n")
        assert text.count("\n\n## References\n\n") <= 1
        assert "\n\n\n" not in text
        labels = [line.split(" ", 1)[0] for line in listing.splitlines()] if listing else []
        assert labels == [f"[{i}]" for i in range(1, citations + 1)]
        used = set(re.findall(r"\[\d+\]", body))
        assert used == set(labels)
        blocks = body.split("\n\n")
        for current, following in zip(blocks, blocks[1:] + [None]):
            if current.startswith("#"):
                level = len(current.split(" ", 1)[0])
                assert following is not None
                assert not (following.startswith("#") and len(following.split(" ", 1)[0]) <= level)


def parquet_bytes(articles):
    reference_type = pa.list_(
        pa.struct(
            [
                ("identifier", pa.string()),
                ("metadata", pa.string()),
                ("source", pa.struct([("links", pa.list_(pa.struct([("text", pa.string()), ("url", pa.string())]))), ("value", pa.string())])),
                ("text", pa.struct([("links", pa.list_(pa.struct([("text", pa.string()), ("url", pa.string())]))), ("value", pa.string())])),
                ("type", pa.string()),
            ]
        )
    )
    table = pa.table(
        {
            "name": [a["name"] for a in articles],
            "identifier": pa.array([a["identifier"] for a in articles], pa.int64()),
            "url": [a["url"] for a in articles],
            "sections": [json.dumps(a["sections"]) for a in articles],
            "references": pa.array([a["references"] for a in articles], reference_type),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


ARTICLES = [
    {"name": "Example", "identifier": 42, "url": "https://en.wikipedia.org/wiki/Example", "sections": SECTIONS, "references": REFERENCES},
    {"name": "Picture only", "identifier": 7, "url": None, "sections": [section("Abstract", {"type": "image"})], "references": None},
]


def test_iter_records_reads_the_real_schema_and_skips_empty_articles(tmp_path):
    path = tmp_path / "f.parquet"
    path.write_bytes(parquet_bytes(ARTICLES))
    records = list(iter_records(path))
    assert records == [
        {"text": EXPECTED, "url": "https://en.wikipedia.org/wiki/Example", "id": "42", "title": "Example", "citations": 5}
    ]


def test_targets_english_structured_wikipedia():
    assert REPO == "wikimedia/structured-wikipedia"
    assert DEFAULT_FOLDER == "enwiki/data"


def test_command_line_downloads_a_verified_shard(serve, tmp_path):
    body = parquet_bytes(ARTICLES)
    revision = "f" * 40
    path = f"{DEFAULT_FOLDER}/enwiki_namespace_0_0.parquet"
    listing = [{"type": "file", "path": path, "size": len(body), "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)}}]
    server = serve(
        {
            f"api/datasets/{REPO}": json.dumps({"sha": revision}).encode(),
            f"api/datasets/{REPO}/tree/{revision}/{DEFAULT_FOLDER}": json.dumps(listing).encode(),
            f"datasets/{REPO}/resolve/{revision}/{path}": body,
        }
    )
    assert main(["--out", str(tmp_path), "--base-url", server.url]) == 0
    assert [r["title"] for r in iter_records(tmp_path / path)] == ["Example"]
