from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from scout.data.fetch import run_hub_fetch
from scout.data.parquet import iter_parquet

REPO = "HuggingFaceTB/finemath"
DEFAULT_FOLDER = "finemath-4plus"
COLUMNS = ("text", "url", "warc_filename", "warc_record_offset", "int_score")


def iter_records(path: Path, batch_size: int = 1024) -> Iterator[dict[str, object]]:
    for row in iter_parquet(path, COLUMNS, batch_size):
        yield {
            "text": row["text"],
            "url": row["url"] or "",
            "id": f"{row['warc_filename']}:{row['warc_record_offset']}",
            "score": row["int_score"],
        }


def main(argv: Sequence[str] | None = None) -> int:
    return run_hub_fetch(REPO, DEFAULT_FOLDER, "Download FineMath 4+ parquet files at a pinned revision.", argv)


if __name__ == "__main__":
    sys.exit(main())
