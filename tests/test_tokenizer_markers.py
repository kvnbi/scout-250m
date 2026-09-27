import pytest

from scout.config import ModelConfig
from scout.tokenizer import markers
from scout.tokenizer.frozen import TOKENIZER_PATH, load_tokenizer
from scout.tokenizer.markers import NAMED, special_tokens

TRANSCRIPT = (
    f"{markers.QUESTION}Who designed the bridge that opened in 1932?{markers.QUESTION_END}"
    f"{markers.PLAN}Find the bridge, then its designer.{markers.PLAN_END}"
    f"{markers.SEARCH}bridge opened 1932{markers.SEARCH_END}"
    f"{markers.PASSAGE}[1] Sydney Harbour Bridge (en.wikipedia.org, 2018-12-20)\n"
    f"The bridge opened in 1932. It was designed by John Bradfield.{markers.PASSAGE_END}"
    f"{markers.NOTE}[1] It was designed by John Bradfield.{markers.NOTE_END}"
    f"{markers.ANSWER}John Bradfield{markers.ANSWER_END}{markers.END_OF_TEXT}"
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def test_named_tokens_are_distinct_and_fill_the_first_reserved_slots(tokenizer):
    assert len(set(NAMED)) == len(NAMED) == 15
    first = ModelConfig().reserved_token_ids.start
    assert [tokenizer.token_to_id(name) for name in NAMED] == list(range(first, first + len(NAMED)))
    assert [tokenizer.token_to_id(name) for name in special_tokens(32)] == list(ModelConfig().reserved_token_ids)


def test_every_marker_is_one_token_in_a_transcript(tokenizer):
    ids = tokenizer.encode(TRANSCRIPT).ids
    reserved = set(ModelConfig().reserved_token_ids)
    used = [NAMED.index(tokenizer.id_to_token(token)) for token in ids if token in reserved]
    assert [NAMED[index] for index in used] == [
        markers.QUESTION,
        markers.QUESTION_END,
        markers.PLAN,
        markers.PLAN_END,
        markers.SEARCH,
        markers.SEARCH_END,
        markers.PASSAGE,
        markers.PASSAGE_END,
        markers.NOTE,
        markers.NOTE_END,
        markers.ANSWER,
        markers.ANSWER_END,
        markers.END_OF_TEXT,
    ]
    assert tokenizer.decode(ids, skip_special_tokens=False) == TRANSCRIPT


def test_marker_text_inside_a_raw_document_stays_text(tokenizer):
    tokenizer.encode_special_tokens = True
    try:
        ids = tokenizer.encode(TRANSCRIPT + markers.DECLINE).ids
        assert not set(ids) & set(ModelConfig().reserved_token_ids)
        assert tokenizer.decode(ids) == TRANSCRIPT + markers.DECLINE
    finally:
        tokenizer.encode_special_tokens = False


def test_no_learned_token_holds_a_marker(tokenizer):
    learned = [tokenizer.decode([index]) for index in range(ModelConfig().reserved_token_ids.start)]
    assert not any("<|" in text and "|>" in text for text in learned)


def test_transformers_reads_the_markers_the_same_way(tokenizer):
    transformers = pytest.importorskip("transformers")
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_PATH), eos_token=markers.END_OF_TEXT, pad_token=markers.PAD
    )
    ids = tokenizer.encode(TRANSCRIPT).ids
    assert fast(TRANSCRIPT, add_special_tokens=False)["input_ids"] == ids
    assert fast.decode(ids) == TRANSCRIPT
    assert (fast.eos_token_id, fast.pad_token_id) == (32736, 32737)
