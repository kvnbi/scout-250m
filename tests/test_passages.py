import json
import random

import pytest

from scout.retrieval.passages import (
    Article,
    Block,
    Passage,
    Section,
    build_passages,
    chunk_blocks,
    is_disambiguation,
    iter_passages,
    skipped_section,
    split_sentences,
    write_passages,
)


def words(count, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(count))


def sentence(count, prefix="w"):
    return words(count, prefix) + "."


@pytest.mark.parametrize(
    "text, expected",
    [
        ("He met Dr. Smith in Paris. They talked.", ["He met Dr. Smith in Paris.", "They talked."]),
        ("J. K. Rowling wrote it. It sold well.", ["J. K. Rowling wrote it.", "It sold well."]),
        ("The U.S. Army won. It left.", ["The U.S. Army won.", "It left."]),
        ("Why? Because. Yes!", ["Why?", "Because.", "Yes!"]),
        ("It ended in 1990. 5 people died.", ["It ended in 1990.", "5 people died."]),
        ("Who is Dr? Nobody.", ["Who is Dr?", "Nobody."]),
        ('He said "Dr. Smith" left. Next.', ['He said "Dr. Smith" left.', "Next."]),
        ("It was (J. Smith) who left. Next.", ["It was (J. Smith) who left.", "Next."]),
        ("Use e.g. this thing. Next one.", ["Use e.g. this thing.", "Next one."]),
        ('He said "Go." Then he left.', ['He said "Go."', "Then he left."]),
        ("Pi is 3.14 exactly. Next.", ["Pi is 3.14 exactly.", "Next."]),
        ("No. 5 was first. Done.", ["No. 5 was first.", "Done."]),
        ("It ended... and then began again. Fine.", ["It ended... and then began again.", "Fine."]),
        ("no capital letters here. so no split", ["no capital letters here. so no split"]),
        ("Just one sentence", ["Just one sentence"]),
        ("", []),
    ],
)
def test_sentence_splitting(text, expected):
    assert split_sentences(text) == expected


def test_sentences_always_rejoin_into_the_original_text():
    rng = random.Random(0)
    pieces = ["Dr.", "Mr.", "A.", "U.S.", "end.", "End.", "what?", "Yes!", '"quoted."', "3.14", "word", "Word", "(Note)", "[1]"]
    for _ in range(300):
        text = " ".join(rng.choice(pieces) for _ in range(rng.randint(0, 25)))
        assert " ".join(split_sentences(text)) == text


def test_a_short_paragraph_is_one_passage():
    assert chunk_blocks([Block("Hello there. General Kenobi.")]) == ["Hello there. General Kenobi."]


def test_sentences_fill_a_passage_greedily_up_to_the_word_limit():
    blocks = [Block(" ".join([sentence(40, "A"), sentence(40, "B"), sentence(40, "C")]))]
    passages = chunk_blocks(blocks, max_words=100, min_words=20)
    assert [len(p.split()) for p in passages] == [80, 40]
    assert passages[0].startswith("A0") and passages[1].startswith("C0")


def test_the_limit_is_inclusive():
    exact = chunk_blocks([Block(sentence(50, "A") + " " + sentence(50, "B"))], max_words=100)
    over = chunk_blocks([Block(sentence(50, "A") + " " + sentence(51, "B"))], max_words=100, min_words=0)
    assert [len(p.split()) for p in exact] == [100]
    assert [len(p.split()) for p in over] == [50, 51]


def test_a_short_last_passage_is_merged_into_the_one_before_it():
    blocks = [Block(" ".join([sentence(60, "A"), sentence(30, "B"), sentence(15, "C")]))]
    assert [len(p.split()) for p in chunk_blocks(blocks, max_words=100, min_words=20)] == [105]
    assert [len(p.split()) for p in chunk_blocks(blocks, max_words=100, min_words=0)] == [90, 15]
    assert [len(p.split()) for p in chunk_blocks(blocks, max_words=100, min_words=15)] == [90, 15]
    assert [len(p.split()) for p in chunk_blocks(blocks, max_words=100, min_words=16)] == [105]


def test_a_lone_short_passage_is_kept():
    assert chunk_blocks([Block("Tiny.")], max_words=100, min_words=50) == ["Tiny."]


def test_an_oversized_sentence_is_cut_at_word_boundaries():
    text = words(250)
    passages = chunk_blocks([Block(text)], max_words=100, min_words=20)
    assert [len(p.split()) for p in passages] == [100, 100, 50]
    assert " ".join(passages) == text


def test_paragraphs_join_with_a_newline_and_sentences_with_a_space():
    blocks = [Block("One. Two."), Block("Three."), Block("Four item", item=True), Block("Five item", item=True)]
    assert chunk_blocks(blocks) == ["One. Two.\nThree.\nFour item\nFive item"]


def test_list_items_are_never_cut_at_a_full_stop():
    blocks = [Block("Item one. With a stop. Inside.", item=True), Block("Second.", item=True)]
    assert chunk_blocks(blocks, max_words=6, min_words=0) == ["Item one. With a stop. Inside.", "Second."]


def test_list_items_are_cut_by_words_not_sentences_when_too_long():
    blocks = [Block("Item one. With a stop. Inside.", item=True)]
    assert chunk_blocks(blocks, max_words=4, min_words=0) == ["Item one. With a", "stop. Inside."]
    assert chunk_blocks([Block("Item one. With a stop. Inside.")], max_words=4, min_words=0) == [
        "Item one.",
        "With a stop. Inside.",
    ]


def test_the_pieces_of_an_oversized_sentence_rejoin_with_a_space_even_when_merged():
    assert chunk_blocks([Block(words(150))], max_words=100, min_words=60) == [words(150)]


def test_the_default_limits_follow_the_wiki18_convention_of_about_100_words():
    assert [len(p.split()) for p in chunk_blocks([Block(words(250))])] == [100, 100, 50]
    blocks = [Block(" ".join([sentence(60, "A"), sentence(30, "B"), sentence(19, "C")]))]
    assert [len(p.split()) for p in chunk_blocks(blocks)] == [109]
    blocks = [Block(" ".join([sentence(60, "A"), sentence(30, "B"), sentence(20, "C")]))]
    assert [len(p.split()) for p in chunk_blocks(blocks)] == [90, 20]


def test_whitespace_is_normalised_and_empty_blocks_vanish():
    nbsp = chr(0xA0)
    blocks = [Block("  a\tb\n c" + nbsp + "d  "), Block("   "), Block(""), Block("e")]
    assert chunk_blocks(blocks) == ["a b c d\ne"]
    assert chunk_blocks([]) == []
    assert chunk_blocks([Block("  ")]) == []


@pytest.mark.parametrize("max_words, min_words", [(0, 0), (-1, 0), (10, -1), (10, 11)])
def test_invalid_limits_are_refused(max_words, min_words):
    with pytest.raises(ValueError):
        chunk_blocks([Block("x")], max_words, min_words)


def test_no_word_is_lost_duplicated_or_reordered_and_every_passage_respects_the_bound():
    rng = random.Random(1)
    vocabulary = ["alpha", "Beta", "gamma.", "Delta?", "eps", "Zeta!", "Dr.", "Smith", "3.5", "(x)", "end."]
    for trial in range(400):
        blocks = [
            Block(" ".join(rng.choice(vocabulary) for _ in range(rng.randint(0, 120))), item=rng.random() < 0.3)
            for _ in range(rng.randint(0, 6))
        ]
        max_words = rng.randint(1, 40)
        min_words = rng.randint(0, max_words)
        passages = chunk_blocks(blocks, max_words, min_words)
        assert " ".join(passages).split() == " ".join(b.text for b in blocks).split(), trial
        assert all(0 < len(p.split()) <= max_words + min_words - (1 if min_words else 0) for p in passages), trial
        assert all(p == p.strip() for p in passages)


def test_passages_carry_the_article_fields_and_sequential_ids_across_sections():
    article = Article(
        "77",
        "Example",
        "https://example.org/Example",
        "2018-12-20",
        [
            Section("", [Block(sentence(70, "A") + " " + sentence(70, "B"))]),
            Section("History", [Block("History text.")]),
            Section("History > Origins", [Block("Origin text.")]),
        ],
    )
    passages = list(build_passages(article))
    assert [(p.id, p.section, p.text.split()[0]) for p in passages] == [
        ("77:0", "", "A0"),
        ("77:1", "", "B0"),
        ("77:2", "History", "History"),
        ("77:3", "History > Origins", "Origin"),
    ]
    assert all((p.title, p.url, p.date) == ("Example", "https://example.org/Example", "2018-12-20") for p in passages)


def test_passages_never_cross_a_section_boundary():
    article = Article("1", "T", "u", "d", [Section("A", [Block("First.")]), Section("B", [Block("Second.")])])
    assert [(p.section, p.text) for p in build_passages(article)] == [("A", "First."), ("B", "Second.")]


def test_sections_to_skip_and_disambiguation_pages():
    assert skipped_section(" References ") and skipped_section("EXTERNAL LINKS") and not skipped_section("History")
    lead = [Section("", [Block("Mercury may refer to:"), Block("The planet", item=True)])]
    assert is_disambiguation("Mercury", lead)
    assert is_disambiguation("Mercury (disambiguation)", [Section("", [Block("Anything.")])])
    assert not is_disambiguation("Mercury", [Section("", [Block("Mercury is the closest planet to the Sun.")])])
    assert is_disambiguation("Mercury", [Section("", [Block("apple", item=True), Block("Mercury may refer to:")])])
    assert not is_disambiguation("List", [Section("", [Block("x", item=True), Block("A long first paragraph. " * 40 + "may refer to")])])


def passages(count=3):
    return [Passage(f"1:{i}", "Caf\u00e9 \u4e2d\u6587", "Sec > Sub", f"text {i}\nsecond line", f"https://x.org/{i}", "2018-12-20") for i in range(count)]


def test_passages_round_trip_through_jsonl_with_unicode(tmp_path):
    path = tmp_path / "deep" / "out.jsonl"
    assert write_passages(path, iter(passages())) == 3
    assert list(iter_passages(path)) == passages()
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["title"] == "Caf\u00e9 \u4e2d\u6587"
    assert [p.name for p in path.parent.iterdir()] == ["out.jsonl"]


def test_an_empty_corpus_is_an_empty_file(tmp_path):
    assert write_passages(tmp_path / "e.jsonl", []) == 0
    assert list(iter_passages(tmp_path / "e.jsonl")) == []


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[]",
        '{"id": "1"}',
        json.dumps({**passages(1)[0].__dict__, "extra": "x"}),
        json.dumps({**passages(1)[0].__dict__, "date": 2018}),
    ],
)
def test_malformed_lines_are_reported_with_their_number(tmp_path, line):
    good = json.dumps(passages(1)[0].__dict__)
    path = tmp_path / "bad.jsonl"
    path.write_text(good + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        list(iter_passages(path))


def test_a_failed_write_leaves_no_partial_output_or_stale_file(tmp_path):
    path = tmp_path / "out.jsonl"
    write_passages(path, passages(2))

    def broken():
        yield passages(1)[0]
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError):
        write_passages(path, broken())
    assert list(iter_passages(path)) == passages(2)
