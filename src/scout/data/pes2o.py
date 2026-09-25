from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from scout.data.common_pile import DEFAULT_FOLDER, SUFFIX, iter_documents
from scout.data.fetch import run_hub_fetch

REPO = "common-pile/peS2o_filtered"


def iter_records(path: Path) -> Iterator[dict[str, object]]:
    for document in iter_documents(path):
        metadata = document.get("metadata") or {}
        yield {
            "text": document["text"],
            "url": metadata.get("oa_url") or "",
            "id": document.get("id") or "",
            "license": metadata.get("oa_license") or "",
        }


def main(argv: Sequence[str] | None = None) -> int:
    description = "Download openly licensed peS2o papers from the Common Pile at a pinned revision."
    return run_hub_fetch(REPO, DEFAULT_FOLDER, description, argv, suffix=SUFFIX)


if __name__ == "__main__":
    sys.exit(main())
