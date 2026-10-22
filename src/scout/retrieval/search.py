from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from scout.retrieval.passages import Passage


class SearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


@runtime_checkable
class SearchBackend(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


def validate_hits(hits: Sequence[Hit], k: int) -> None:
    if k < 1:
        raise ValueError("k must be positive")
    if len(hits) > k:
        raise ValueError(f"{len(hits)} hits for k={k}")
    seen: set[str] = set()
    previous = math.inf
    for number, hit in enumerate(hits):
        if not isinstance(hit, Hit) or not isinstance(hit.passage, Passage):
            raise ValueError(f"hit {number} is not a Hit")
        if not hit.passage.text.strip():
            raise ValueError(f"hit {number} has no text")
        if hit.passage.id in seen:
            raise ValueError(f"hit {number} repeats passage {hit.passage.id}")
        seen.add(hit.passage.id)
        if not math.isfinite(hit.score):
            raise ValueError(f"hit {number} has a score that is not finite")
        if hit.score > previous:
            raise ValueError(f"hit {number} scores higher than the hit before it")
        previous = hit.score


def check_backend(backend: SearchBackend, queries: Sequence[str], k: int = 5) -> None:
    if not isinstance(backend, SearchBackend):
        raise TypeError("a backend needs a name and a search(query, k) method")
    for blank in ("", "   ", "\n"):
        if backend.search(blank, k):
            raise ValueError("a blank query must return no hits")
    try:
        backend.search("test", 0)
    except ValueError:
        pass
    else:
        raise ValueError("k below 1 must be refused with a ValueError")
    for query in queries:
        validate_hits(backend.search(query, k), k)


def domain(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme == "file":
        return "local"
    return parts.hostname or ""


def render_hit(number: int, hit: Hit) -> str:
    passage = hit.passage
    details = ", ".join(part for part in (domain(passage.url), passage.date) if part)
    header = f"[{number}] {passage.title}" + (f" ({details})" if details else "")
    return f"{header}\n{passage.text}"


def render_hits(hits: Sequence[Hit]) -> str:
    return "\n\n".join(render_hit(number, hit) for number, hit in enumerate(hits, start=1))
