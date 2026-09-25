from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

DEFAULT_FOLDER = ""
SUFFIX = ".json.gz"


def iter_documents(path: Path) -> Iterator[dict[str, object]]:
    with gzip.open(path) as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            document = json.loads(line)
            if not isinstance(document, dict) or not isinstance(document.get("text"), str):
                raise ValueError(f"{path} line {number} has no text")
            yield document
