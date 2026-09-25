from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from scout.data.common_pile import DEFAULT_FOLDER, SUFFIX, iter_documents
from scout.data.fetch import run_hub_fetch

REPO = "common-pile/stackv2_edu_filtered"


def iter_records(path: Path) -> Iterator[dict[str, object]]:
    for document in iter_documents(path):
        metadata = document.get("metadata") or {}
        if metadata.get("is_generated") or metadata.get("is_vendor"):
            continue
        yield {
            "text": document["text"],
            "url": metadata.get("url") or "",
            "id": document.get("id") or "",
            "repo": metadata.get("repo_name") or "",
            "language": metadata.get("language") or "",
            "score": document.get("int_score"),
            "license": metadata.get("license") or metadata.get("gha_license_id") or "",
        }


def main(argv: Sequence[str] | None = None) -> int:
    description = "Download openly licensed educational code from the Common Pile Stack v2 at a pinned revision."
    return run_hub_fetch(REPO, DEFAULT_FOLDER, description, argv, suffix=SUFFIX)


if __name__ == "__main__":
    sys.exit(main())
