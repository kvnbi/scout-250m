from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from scout.data.fetch import HUB_URL, download_with_retries, hub_file_url, hub_files, hub_revision, sample_files

REPO = "HuggingFaceFW/fineweb-edu"
DEFAULT_FOLDER = "sample/100BT"
COLUMNS = ("text", "id", "url", "int_score")


def iter_records(path: Path, batch_size: int = 1024) -> Iterator[dict[str, object]]:
    parquet = pq.ParquetFile(path)
    missing = [column for column in COLUMNS if column not in parquet.schema_arrow.names]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(COLUMNS)):
        for row in batch.to_pylist():
            if not isinstance(row["text"], str):
                raise ValueError(f"{path} has a row without text: {row['id']!r}")
            yield {"text": row["text"], "url": row["url"] or "", "id": row["id"] or "", "score": row["int_score"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download FineWeb Edu parquet files at a pinned revision.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    parser.add_argument("--files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--base-url", default=HUB_URL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.retries < 0 or args.retry_wait < 0:
        parser.error("retries and retry wait must not be negative")
    revision = args.revision or hub_revision(REPO, args.base_url)
    chosen = sample_files(hub_files(REPO, revision, args.folder, ".parquet", args.base_url), args.files, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    selection = {
        "repo": REPO,
        "revision": revision,
        "folder": args.folder,
        "files_requested": args.files,
        "seed": args.seed,
        "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in chosen],
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"selected {len(chosen)} files, {sum(f.size for f in chosen) / 2**30:.1f} GiB at {revision}")
    if args.dry_run:
        return 0
    for number, file in enumerate(chosen, start=1):
        url = hub_file_url(REPO, revision, file.path, args.base_url)
        download_with_retries(url, args.out / file.path, args.retries, args.retry_wait, file.sha256)
        print(f"{number}/{len(chosen)} {file.path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
