from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from scout.data.fetch import run_hub_fetch
from scout.data.parquet import iter_parquet

REPO = "HuggingFaceFW/finepdfs-edu"
DEFAULT_FOLDER = "data/eng_Latn/train"
COLUMNS = ("text", "id", "url", "token_count", "extractor", "is_truncated")


def iter_records(path: Path, batch_size: int = 256) -> Iterator[dict[str, object]]:
    for row in iter_parquet(path, COLUMNS, batch_size):
        yield {
            "text": row["text"],
            "url": row["url"] or "",
            "id": row["id"] or "",
            "tokens": row["token_count"],
            "extractor": row["extractor"] or "",
            "refetched": bool(row["is_truncated"]),
        }


def main(argv: Sequence[str] | None = None) -> int:
    description = "Download English FinePDFs Edu parquet files at a pinned revision."
    return run_hub_fetch(REPO, DEFAULT_FOLDER, description, argv)


if __name__ == "__main__":
    sys.exit(main())
