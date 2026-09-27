import hashlib
import json

from test_tokenizer_train import TRICKY

from scout.config import ModelConfig
from scout.tokenizer.frozen import CORE_SHA256, FILE_SHA256, TOKENIZER_PATH, core_sha256, load_tokenizer
from scout.tokenizer.train import END_OF_TEXT, new_tokenizer, special_tokens


def test_file_is_unchanged():
    assert hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest() == FILE_SHA256


def test_learned_vocabulary_and_rules_are_unchanged():
    assert core_sha256() == CORE_SHA256


def test_core_hash_ignores_only_the_special_tokens(tmp_path):
    content = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    content["added_tokens"][1]["content"] = "<|renamed|>"
    renamed = tmp_path / "renamed.json"
    renamed.write_text(json.dumps(content), encoding="utf-8")
    assert core_sha256(renamed) == CORE_SHA256
    content["model"]["merges"][0] = content["model"]["merges"][1]
    renamed.write_text(json.dumps(content), encoding="utf-8")
    assert core_sha256(renamed) != CORE_SHA256


def test_fits_the_model_vocabulary():
    tokenizer = load_tokenizer()
    config = ModelConfig()
    assert tokenizer.get_vocab_size() == config.vocab_size
    names = special_tokens(config.reserved_slots)
    assert [tokenizer.token_to_id(name) for name in names] == list(config.reserved_token_ids)
    assert tokenizer.token_to_id(END_OF_TEXT) == 32736
    assert max(json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))["model"]["vocab"].values()) == 32735


def test_uses_the_training_rules():
    frozen = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    fresh = json.loads(new_tokenizer().to_str())
    for part in ("normalizer", "pre_tokenizer", "decoder", "post_processor"):
        assert frozen[part] == fresh[part], part


def test_round_trips_text_exactly():
    tokenizer = load_tokenizer()
    tokenizer.encode_special_tokens = True
    for text in TRICKY:
        assert tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=False) == text
