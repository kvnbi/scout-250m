from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from scout.config import ModelConfig
from scout.tokenizer.sample import iter_sample

SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
END_OF_TEXT = "<|endoftext|>"
TOKENIZER_FILE = "tokenizer.json"


def special_tokens(count: int) -> list[str]:
    if count < 1:
        raise ValueError("at least one reserved slot is needed for the end of text token")
    return [END_OF_TEXT] + [f"<|reserved_{index}|>" for index in range(1, count)]


def new_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(texts: Iterable[str], config: ModelConfig = ModelConfig()) -> Tokenizer:
    learned = config.vocab_size - config.reserved_slots
    tokenizer = new_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=learned,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    if tokenizer.get_vocab_size() != learned:
        raise ValueError(f"training produced {tokenizer.get_vocab_size()} tokens, {learned} are needed")
    tokenizer.add_special_tokens(special_tokens(config.reserved_slots))
    reserved = [tokenizer.token_to_id(token) for token in special_tokens(config.reserved_slots)]
    if reserved != list(config.reserved_token_ids):
        raise ValueError("special tokens did not land on the reserved ids")
    return tokenizer


def bytes_per_token(tokenizer: Tokenizer, texts: Sequence[str]) -> float:
    tokens = sum(len(encoding.ids) for encoding in tokenizer.encode_batch(list(texts), add_special_tokens=False))
    return sum(len(text.encode()) for text in texts) / max(tokens, 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the byte level BPE tokenizer on a tokenizer sample.")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    tokenizer = train_tokenizer(record["text"] for record in iter_sample(args.sample))
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out / TOKENIZER_FILE))
    print(f"trained {tokenizer.get_vocab_size()} tokens in {time.perf_counter() - start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
