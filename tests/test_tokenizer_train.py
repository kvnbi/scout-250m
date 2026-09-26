import json
import random
import string

import pytest
from compression import zstd
from tokenizers import Tokenizer, pre_tokenizers

import scout.tokenizer.train
from scout.config import ModelConfig
from scout.tokenizer.train import (
    END_OF_TEXT,
    TOKENIZER_FILE,
    bytes_per_token,
    main,
    new_tokenizer,
    special_tokens,
    train_tokenizer,
)

SMALL = ModelConfig(vocab_size=512)

TRICKY = [
    "",
    " ",
    "Plain text.",
    "Line one\nline two\r\nwindows\n\n\nthree blank\tand tab",
    "unicode separators\u2028and\u2029next\x85line",
    "NUL \x00 and bell \x07 controls",
    "em \u2014 en \u2013 dashes stay raw",
    "emoji \U0001f600 and a flag \U0001f1f3\U0001f1f4",
    "\u4e2d\u6587 \u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u0939\u093f\u0928\u094d\u0926\u0940",
    "    indented code\n\tdef f(x):\n\t\treturn x  \n",
    "trailing spaces   ",
    "e\u0301 decomposed and \u00e9 composed",
    "digits 1234567890 and 3.14159",
    "<|endoftext|> written inside a document",
    "\ufeffbyte order mark",
]


def words(count, seed=0):
    rng = random.Random(seed)
    return ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 7))) for _ in range(count)]


def corpus():
    pool = words(4000)
    return [" ".join(pool[i : i + 20]) for i in range(0, len(pool), 20)] + TRICKY


@pytest.fixture(scope="module")
def trained():
    return train_tokenizer(corpus(), SMALL).to_str()


@pytest.fixture
def small(trained):
    return Tokenizer.from_str(trained)


def test_learns_the_vocabulary_and_puts_special_tokens_on_the_reserved_ids(small):
    assert small.get_vocab_size() == 512
    assert [small.token_to_id(token) for token in special_tokens(32)] == list(SMALL.reserved_token_ids)
    assert small.token_to_id(END_OF_TEXT) == 480
    assert set(pre_tokenizers.ByteLevel.alphabet()) <= set(small.get_vocab())
    assert all(small.id_to_token(index) is not None for index in range(512))


@pytest.mark.parametrize("text", TRICKY)
def test_round_trips_any_text_exactly(small, text):
    small.encode_special_tokens = True
    assert small.decode(small.encode(text).ids, skip_special_tokens=False) == text


def test_round_trips_random_unicode(small):
    rng = random.Random(3)
    points = [p for p in range(0x110000) if not 0xD800 <= p <= 0xDFFF]
    texts = ["".join(chr(rng.choice(points)) for _ in range(rng.randint(1, 60))) for _ in range(300)]
    texts += ["".join(rng.choice(" \n\t\ra\u00e9\u4e2d") for _ in range(200)) for _ in range(100)]
    small.encode_special_tokens = True
    assert [small.decode(encoding.ids, skip_special_tokens=False) for encoding in small.encode_batch(texts)] == texts


def test_raw_text_needs_special_tokens_encoded_as_text(small):
    text = "before <|endoftext|> after"
    assert 480 in small.encode(text).ids
    small.encode_special_tokens = True
    assert 480 not in small.encode(text).ids


def test_splits_text_like_qwen2():
    pieces = new_tokenizer().pre_tokenizer.pre_tokenize_str("I'd say DON'T: 2024 costs $1,250.\n\n  x")
    assert [piece for piece, _ in pieces] == [
        "I",
        "'d",
        "\u0120say",
        "\u0120DON",
        "'T",
        ":",
        "\u0120",
        "2",
        "0",
        "2",
        "4",
        "\u0120costs",
        "\u0120$",
        "1",
        ",",
        "2",
        "5",
        "0",
        ".\u010a\u010a",
        "\u0120",
        "\u0120x",
    ]
    pieces = new_tokenizer().pre_tokenizer.pre_tokenize_str("WE'REALLY we'really")
    assert [piece for piece, _ in pieces] == ["WE", "'RE", "ALLY", "\u0120we", "'re", "ally"]


def test_keeps_text_unnormalised(small):
    assert new_tokenizer().normalizer is None
    assert small.encode("\u00e9").ids != small.encode("e\u0301").ids


def test_training_is_deterministic(trained):
    assert train_tokenizer(corpus(), SMALL).to_str() == trained


def test_rejects_a_sample_too_small_for_the_vocabulary():
    with pytest.raises(ValueError, match="training produced"):
        train_tokenizer(["tiny text"], SMALL)


def test_special_tokens_must_land_on_the_reserved_ids(monkeypatch):
    monkeypatch.setattr(scout.tokenizer.train, "special_tokens", lambda count: ["a"] + [f"<|x{i}|>" for i in range(1, count)])
    with pytest.raises(ValueError, match="reserved ids"):
        train_tokenizer(corpus(), SMALL)


def test_special_token_names():
    assert special_tokens(3) == ["<|endoftext|>", "<|reserved_1|>", "<|reserved_2|>"]
    with pytest.raises(ValueError):
        special_tokens(0)


def test_bytes_per_token_counts_utf8_bytes(small):
    texts = ["abc def", "\u00e9\u00e9"]
    tokens = sum(len(small.encode(text).ids) for text in texts)
    assert bytes_per_token(small, texts) == 11 / tokens


def test_command_line_trains_from_a_sample(tmp_path):
    pool = words(40000)
    records = [{"source": "s", "id": str(i), "part": 0, "text": " ".join(pool[i : i + 20])} for i in range(0, 40000, 20)]
    sample = tmp_path / "sample.jsonl.zst"
    sample.write_bytes(zstd.compress("".join(json.dumps(record) + "\n" for record in records).encode()))
    assert main(["--sample", str(sample), "--out", str(tmp_path / "out")]) == 0
    tokenizer = Tokenizer.from_file(str(tmp_path / "out" / TOKENIZER_FILE))
    assert tokenizer.get_vocab_size() == 32768
    assert tokenizer.token_to_id(END_OF_TEXT) == 32736
    assert tokenizer.decode(tokenizer.encode(records[0]["text"]).ids) == records[0]["text"]


def test_loads_in_transformers(small, tmp_path):
    transformers = pytest.importorskip("transformers")
    small.save(str(tmp_path / TOKENIZER_FILE))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_file=str(tmp_path / TOKENIZER_FILE), eos_token=END_OF_TEXT)
    assert fast.eos_token_id == 480
    assert len(fast) == 512
    for text in TRICKY[:-2]:
        ids = small.encode(text).ids
        assert fast(text, add_special_tokens=False)["input_ids"] == ids
        assert fast.decode(ids) == text
