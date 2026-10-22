import bz2

import pytest

from scout.retrieval import build
from scout.retrieval.passages import Block, iter_passages
from scout.retrieval.wikitext import iter_articles, iter_pages, parse_page, parse_text, strip_markup


def clean(text):
    return " ".join(strip_markup(text).split())


GREEK = "".join(chr(code) for code in (0x391, 0x3C1, 0x3B9, 0x3C3))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A {{cite web|url=x|title=y}} B", "A B"),
        ("A {{outer|{{inner|x}}|y}} B", "A B"),
        ("A {{unknown}} B", "A B"),
        ("{{lang|fr|Bonjour}} monde", "Bonjour monde"),
        ("{{lang-de|Hallo}} Welt", "Hallo Welt"),
        ("{{nowrap|a b}} c", "a b c"),
        ("{{small|tiny}}", "tiny"),
        ("It is {{convert|5|km|mi}} away", "It is 5 km away"),
        ("It is {{convert|5|to|10|km}} away", "It is 5 to 10 km away"),
        ("It is {{convert|5|km|abbr=on}} away", "It is 5 km away"),
        ("{{lang-el|[[X|Y]]|Aristotle}} wrote", "Y wrote"),
        ("{{lang|fr}} x", "x"),
        ("{{}} x", "x"),
        ("Before <!-- hidden --> after", "Before after"),
        ("a <!--x--> b <!--y--> c", "a b c"),
        ("a <!-- line one\nline two --> b", "a b"),
        ("A<REF>drop</REF>B", "AB"),
        ("x <reference>kept</ref> y", "x kept y"),
        ("{{Lang|fr|Bonjour}} and {{Convert|5|km}}", "Bonjour and 5 km"),
        ("{{convert|abbr=on|5|km}}", "5 km"),
        ("{{convert|5|km|mi|0}}", "5 km"),
        ("{{convert|5|and|10|km}} {{convert|5|or|10|km}} {{convert|5|by|10|m}}", "5 and 10 km 5 or 10 km 5 by 10 m"),
        ("{{lang-el|[[A=B|label]]}} x", "label x"),
        ("{{lang|fr|oops]]|more}} x", "oops]] x"),
        ("See [[Image]] and [[File]] for details", "See Image and File for details"),
        ("[[Mercury (planet)| ]] is small", "Mercury is small"),
        ("see [//example.org label] now", "see label now"),
        ("a http://x.org/p|b", "a |b"),
        ("Fact<ref>Source one</ref> here<ref name=a/> and<ref name=b>Source two</ref>.", "Fact here and."),
        ("Pre <gallery>\nFile:a.jpg\n</gallery> post", "Pre post"),
        ("Area is <math>x^2</math> units", "Area is units"),
        ("One<br>two<br />three <small>four</small>", "One two three four"),
        ("[[Paris]] and [[Paris|the capital]] and [[:fr:Paris|Paris]]", "Paris and the capital and Paris"),
        ("[[Mercury (planet)|]] is small", "Mercury is small"),
        ("[[:Category:Foo]] stays", "Category:Foo stays"),
        ("Text [[Category:Foo]] [[File:a.jpg|thumb|A [[link]] caption]] end", "Text end"),
        ("[[File:a.jpg|thumb|See [above] and [http://x.org site]]] end", "end"),
        ("[http://x.org label here] and [http://x.org] and http://bare.org/path x", "label here and and x"),
        ("'''bold''' and ''italic'' and '''''both''''' don't", "bold and italic and both don't"),
        ("Tom &amp; Jerry&nbsp;show", "Tom & Jerry show"),
        ("Before {| class=wikitable\n|-\n| cell\n|} after", "Before after"),
        ("A {| outer\n{| inner\n|x\n|}\n|} B", "A B"),
        ("__NOTOC__ text", "text"),
        ("Aristotle ({{IPA|x}}; 384 BC) was", "Aristotle (384 BC) was"),
        ("Foo ({{IPA|x}}) bar", "Foo bar"),
        ("Stray }} and {{ braces", "Stray and braces"),
        ("array[0] and [x] stay", "array[0] and [x] stay"),
        ("Unicode " + GREEK + " [[" + GREEK + "]]", "Unicode " + GREEK + " " + GREEK),
    ],
)
def test_markup_removal(text, expected):
    assert clean(text) == expected


def test_pronunciation_leftovers_collapse_without_extra_spaces():
    assert strip_markup("Foo ({{IPA|x}}) bar") == "Foo bar"
    assert strip_markup("Foo ({{IPA|x}}; born 1900) bar") == "Foo (born 1900) bar"


def sections(text):
    return [(s.name, [(b.text, b.item) for b in s.blocks]) for s in parse_text(text)]


def test_headings_make_nested_section_paths():
    text = "Lead one.\n\n== History ==\nHistory text.\n=== Origins ===\nOrigin text.\n== Geography ==\nGeo text.\n"
    assert sections(text) == [
        ("", [("Lead one.", False)]),
        ("History", [("History text.", False)]),
        ("History > Origins", [("Origin text.", False)]),
        ("Geography", [("Geo text.", False)]),
    ]


def test_a_deeper_heading_after_a_shallower_one_pops_the_path():
    text = "== A ==\na\n=== B ===\nb\n==== C ====\nc\n=== D ===\nd\n== E ==\ne\n"
    assert [name for name, _ in sections(text)] == ["A", "A > B", "A > B > C", "A > D", "E"]


def test_reference_like_sections_and_their_subsections_are_dropped():
    text = "== Body ==\nkeep\n== References ==\ndrop\n=== Sub ===\ndrop too\n== Notes ==\nn\n== More ==\nkeep two\n== See also ==\nx\n"
    assert sections(text) == [("Body", [("keep", False)]), ("More", [("keep two", False)])]


def test_lines_join_into_paragraphs_and_blank_lines_split_them():
    assert sections("First line\nsecond line\n\nNext paragraph\n") == [
        ("", [("First line second line", False), ("Next paragraph", False)])
    ]


def test_list_items_are_kept_without_their_markers_and_hatnotes_are_dropped():
    text = "Intro\n* apple\n** nested pear\n# first\n: indented note\n:''Main article: [[History of X]]''\n; Term\nAfter list\n"
    assert sections(text) == [
        (
            "",
            [
                ("Intro", False),
                ("apple", True),
                ("nested pear", True),
                ("first", True),
                ("indented note", True),
                ("Term", True),
                ("After list", False),
            ],
        )
    ]


def test_whitespace_inside_paragraphs_and_headings_is_collapsed():
    assert sections("==  Early   life  ==\nfirst   line\tof\n  text  \n") == [("Early life", [("first line of text", False)])]


def test_a_subsection_of_a_skipped_section_stays_skipped_whatever_its_own_name():
    text = "== References ==\nx\n==== Notes ====\ny\n=== Other ===\nz\n== Keep ==\nk\n"
    assert sections(text) == [("Keep", [("k", False)])]


def test_four_dashes_are_a_rule_but_fewer_are_ordinary_text():
    assert sections("One\n----\nTwo\n") == [("", [("One", False), ("Two", False)])]
    assert sections("One\n---\nTwo\n") == [("", [("One --- Two", False)])]


def test_list_items_without_letters_are_dropped():
    assert sections("* 1990\n* 2000\n* real item\n") == [("", [("real item", True)])]


def test_an_unterminated_table_leaves_no_markup_behind():
    assert sections("{| class=x\n|-\n! Header\n| cell\nReal text\n") == [("", [("Real text", False)])]


def test_only_whole_lines_that_open_and_close_with_equals_are_headings():
    assert sections("== A == extra\nbody\n") == [("", [("== A == extra body", False)])]
    assert sections("== Broken\nbody\n") == [("", [("== Broken body", False)])]
    assert sections("= Top =\ntext\n") == [("Top", [("text", False)])]


def test_table_leftovers_and_blocks_without_letters_are_dropped():
    text = "{| class=x\n|-\n| 12\n! Header\n|}\n----\n123 456\n\nReal text\n----\nMore text\n"
    assert sections(text) == [("", [("Real text", False), ("More text", False)])]


def test_a_heading_with_markup_and_an_empty_section():
    text = "== [[Early]] ''life'' ==\nfirst\n== Empty ==\n== Full ==\ntext\n"
    assert sections(text) == [("Early life", [("first", False)]), ("Full", [("text", False)])]


def test_nothing_survives_in_a_page_that_is_only_markup():
    assert parse_text("{{Infobox\n|a=b\n}}\n[[Category:X]]\n") == []


def test_redirects_disambiguations_and_empty_pages_are_not_articles():
    assert parse_page("1", "Old", "#REDIRECT [[New]]", "2018-12-20") is None
    assert parse_page("1", "Old", "  #redirect [[New]]", "2018-12-20") is None
    assert parse_page("2", "Mercury", "'''Mercury''' may refer to:\n* [[Mercury (planet)]]\n", "2018-12-20") is None
    assert parse_page("3", "Mercury (disambiguation)", "Some text.", "2018-12-20") is None
    assert parse_page("4", "Empty", "{{stub}}", "2018-12-20") is None


def test_article_fields_and_urls():
    article = parse_page("12", "Zurich (city)", "Text here.", "2018-12-20")
    assert (article.id, article.title, article.date) == ("12", "Zurich (city)", "2018-12-20")
    assert article.url == "https://en.wikipedia.org/wiki/Zurich_(city)"
    assert [s.name for s in article.sections] == [""]
    accent = parse_page("13", "Z" + chr(0xFC) + "rich", "Text.", "2018-12-20")
    assert accent.url == "https://en.wikipedia.org/wiki/Z%C3%BCrich"
    assert parse_page("14", "AC/DC", "Text.", "d").url.endswith("/wiki/AC/DC")


def dump(pages, namespace='xmlns="http://www.mediawiki.org/xml/export-0.10/"'):
    body = []
    for title, ns, page_id, text, redirect in pages:
        body.append(
            f"<page><title>{title}</title><ns>{ns}</ns><id>{page_id}</id>"
            + ('<redirect title="X" />' if redirect else "")
            + f"<revision><id>9{page_id}</id><timestamp>2018-12-01T00:00:00Z</timestamp>"
            f'<text xml:space="preserve">{text}</text></revision></page>'
        )
    header = f"<mediawiki {namespace}><siteinfo><sitename>Wikipedia</sitename></siteinfo>"
    return (header + "".join(body) + "</mediawiki>").encode()


PAGES = [
    ("Alpha", 0, 10, "Alpha is the first letter. It starts things.", False),
    ("Talk:Alpha", 1, 11, "Talk page text.", False),
    ("Beta", 0, 12, "#REDIRECT [[Alpha]]", True),
    ("Gamma &amp; Delta", 0, 13, "Gamma is the third letter &amp; Delta the fourth.", False),
    ("Epsilon", 0, 14, "", False),
    ("Zeta (disambiguation)", 0, 15, "Zeta may refer to:\n* a letter", False),
]


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("namespace", ['xmlns="http://www.mediawiki.org/xml/export-0.10/"', ""])
def test_reading_a_dump_keeps_only_real_articles_with_page_ids(tmp_path, compressed, namespace):
    data = dump(PAGES, namespace)
    path = tmp_path / ("dump.xml.bz2" if compressed else "dump.xml")
    path.write_bytes(bz2.compress(data) if compressed else data)
    assert [(i, t) for i, t, _, _ in iter_pages(path)] == [
        ("10", "Alpha"),
        ("13", "Gamma & Delta"),
        ("14", "Epsilon"),
        ("15", "Zeta (disambiguation)"),
    ]
    articles = list(iter_articles(path, "2018-12-20"))
    assert [(a.id, a.title, a.date) for a in articles] == [("10", "Alpha", "2018-12-20"), ("13", "Gamma & Delta", "2018-12-20")]
    assert articles[1].sections[0].blocks == (Block("Gamma is the third letter & Delta the fourth."),)


def test_titles_are_trimmed(tmp_path):
    path = tmp_path / "d.xml"
    path.write_bytes(dump([("  Padded  ", 0, 1, "Text.", False)]))
    assert [t for _, t, _, _ in iter_pages(path)] == ["Padded"]


def test_a_multistream_dump_reads_as_one(tmp_path):
    first = dump(PAGES[:1])
    second = b"".join(dump(PAGES[3:4]).split(b"</siteinfo>")[1:2]).replace(b"</mediawiki>", b"")
    stream = bz2.compress(first.replace(b"</mediawiki>", b"")) + bz2.compress(second + b"</mediawiki>")
    path = tmp_path / "multi.xml.bz2"
    path.write_bytes(stream)
    assert [t for _, t, _, _ in iter_pages(path)] == ["Alpha", "Gamma & Delta"]


def test_a_page_without_a_revision_has_no_text(tmp_path):
    path = tmp_path / "d.xml"
    path.write_bytes(b"<mediawiki><page><title>Bare</title><ns>0</ns><id>1</id></page></mediawiki>")
    assert list(iter_pages(path)) == [("1", "Bare", "", "0")]
    assert list(iter_articles(path, "d")) == []


def test_memory_stays_bounded_by_clearing_finished_pages(tmp_path):
    pages = [(f"Page {i}", 0, i, f"Page {i} is a page about number {i}.", False) for i in range(1, 400)]
    path = tmp_path / "many.xml"
    path.write_bytes(dump(pages))
    assert len(list(iter_articles(path, "d"))) == 399


def test_command_line_builds_passages_from_a_dump(tmp_path, capsys):
    path = tmp_path / "dump.xml.bz2"
    path.write_bytes(bz2.compress(dump(PAGES)))
    out = tmp_path / "out" / "passages.jsonl"
    assert build.main(["--format", "wikitext", "--date", "2018-12-20", "--out", str(out), str(path)]) == 0
    assert capsys.readouterr().out.startswith("2 articles, 2 passages")
    passages = list(iter_passages(out))
    assert [(p.id, p.title, p.date) for p in passages] == [("10:0", "Alpha", "2018-12-20"), ("13:0", "Gamma & Delta", "2018-12-20")]
    assert passages[0].text == "Alpha is the first letter. It starts things."


@pytest.mark.parametrize(
    "arguments",
    [
        ["--date", "2018-12"],
        ["--date", "yesterday"],
        ["--date", "2018-13-40"],
        ["--date", "2018-12-20", "--max-words", "0"],
        ["--date", "2018-12-20", "--max-words", "0", "--min-words", "0"],
        ["--date", "2018-12-20", "--min-words", "200"],
        ["--date", "2018-12-20", "--format", "html"],
    ],
)
def test_command_line_refuses_bad_arguments(tmp_path, arguments):
    base = ["--format", "wikitext", "--out", str(tmp_path / "o.jsonl"), str(tmp_path / "in.xml")]
    with pytest.raises(SystemExit):
        build.main(arguments + base if "--format" not in arguments else arguments + base[2:])
    assert not (tmp_path / "o.jsonl").exists()


def test_a_wikitext_dump_needs_its_date(tmp_path):
    path = tmp_path / "dump.xml"
    path.write_bytes(dump(PAGES[:1]))
    with pytest.raises(SystemExit):
        build.main(["--format", "wikitext", "--out", str(tmp_path / "o.jsonl"), str(path)])
    assert not (tmp_path / "o.jsonl").exists()


def test_word_limits_default_to_100_and_20_and_can_be_changed(tmp_path):
    path = tmp_path / "dump.xml"
    path.write_bytes(dump([("Long", 0, 1, " ".join(f"w{i}" for i in range(215)), False)]))

    def lengths(*options):
        out = tmp_path / "o.jsonl"
        assert build.main(["--format", "wikitext", "--date", "2018-12-20", "--out", str(out), *options, str(path)]) == 0
        return [len(p.text.split()) for p in iter_passages(out)]

    assert lengths() == [100, 115]
    assert lengths("--max-words", "60", "--min-words", "10") == [60, 60, 60, 35]
    assert lengths("--max-words", "60", "--min-words", "36") == [60, 60, 95]


def test_a_compact_date_is_normalised(tmp_path):
    path = tmp_path / "dump.xml"
    path.write_bytes(dump(PAGES[:1]))
    out = tmp_path / "o.jsonl"
    assert build.main(["--format", "wikitext", "--date", "20181220", "--out", str(out), str(path)]) == 0
    assert {p.date for p in iter_passages(out)} == {"2018-12-20"}
