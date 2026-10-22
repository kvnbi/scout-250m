from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path

from scout.retrieval import structured, wikitext
from scout.retrieval.passages import MAX_WORDS, MIN_WORDS, Passage, build_passages, write_passages

READERS = {"structured": structured.iter_articles, "wikitext": wikitext.iter_articles}


def iter_passages_from(
    format: str, inputs: Sequence[Path], snapshot: str | None, max_words: int, min_words: int, counts: dict[str, int]
) -> Iterator[Passage]:
    for article in READERS[format](inputs, snapshot):
        counts["articles"] += 1
        yield from build_passages(article, max_words, min_words)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build retrieval passages from a Wikipedia dump.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--format", choices=sorted(READERS), required=True)
    parser.add_argument("--date", help="dump date, YYYY-MM-DD (wikitext only; structured rows carry their own)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    args = parser.parse_args(argv)
    snapshot = None
    if args.format == "structured" and args.date is not None:
        parser.error("--date applies only to --format wikitext")
    if args.format == "wikitext":
        try:
            snapshot = date.fromisoformat(args.date or "").isoformat()
        except ValueError:
            parser.error("--format wikitext needs --date like 2018-12-20")
    if args.max_words < 1 or not 0 <= args.min_words <= args.max_words:
        parser.error("need 1 <= --max-words and 0 <= --min-words <= --max-words")
    counts = {"articles": 0}
    passages = write_passages(
        args.out, iter_passages_from(args.format, args.inputs, snapshot, args.max_words, args.min_words, counts)
    )
    print(f"{counts['articles']} articles, {passages} passages -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
