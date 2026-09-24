from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from scout.data.fetch import run_hub_fetch
from scout.data.parquet import iter_parquet

REPO = "HuggingFaceFW/fineweb-edu"
DEFAULT_FOLDER = "sample/100BT"
COLUMNS = ("text", "id", "url", "int_score")


def iter_records(path: Path, batch_size: int = 1024) -> Iterator[dict[str, object]]:
    for row in iter_parquet(path, COLUMNS, batch_size):
        yield {"text": row["text"], "url": row["url"] or "", "id": row["id"] or "", "score": row["int_score"]}


def main(argv: Sequence[str] | None = None) -> int:
    return run_hub_fetch(REPO, DEFAULT_FOLDER, "Download FineWeb Edu parquet files at a pinned revision.", argv)


if __name__ == "__main__":
    sys.exit(main())
