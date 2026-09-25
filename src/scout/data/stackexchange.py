from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from scout.data.common_pile import DEFAULT_FOLDER, SUFFIX, iter_documents
from scout.data.fetch import run_hub_fetch

REPO = "common-pile/stackexchange_filtered"
NON_ENGLISH_SITES = frozenset(
    {
        "es.stackoverflow.com",
        "es.meta.stackoverflow.com",
        "ja.stackoverflow.com",
        "ja.meta.stackoverflow.com",
        "pt.stackoverflow.com",
        "pt.meta.stackoverflow.com",
        "ru.stackoverflow.com",
        "ru.meta.stackoverflow.com",
        "rus.stackexchange.com",
        "rus.meta.stackexchange.com",
        "italian.stackexchange.com",
        "portuguese.stackexchange.com",
        "ukrainian.stackexchange.com",
    }
)


def iter_records(path: Path) -> Iterator[dict[str, object]]:
    for document in iter_documents(path):
        metadata = document.get("metadata") or {}
        site = metadata.get("site") or ""
        if site in NON_ENGLISH_SITES:
            continue
        yield {
            "text": document["text"],
            "url": metadata.get("url") or "",
            "id": f"{site}/{document.get('id') or ''}",
            "site": site,
            "license": metadata.get("license") or "",
        }


def main(argv: Sequence[str] | None = None) -> int:
    description = "Download openly licensed Stack Exchange threads from the Common Pile at a pinned revision."
    return run_hub_fetch(REPO, DEFAULT_FOLDER, description, argv, suffix=SUFFIX)


if __name__ == "__main__":
    sys.exit(main())
