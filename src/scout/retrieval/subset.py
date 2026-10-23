from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from scout.data.fetch import sha256_of
from scout.retrieval.passages import Passage, iter_passages, write_passages

FORMAT = 1
MISSING_LISTED = 50


def article_of(passage_id: str) -> str:
    return passage_id.rpartition(":")[0] or passage_id


def _rank(seed: int, article: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{seed}\0{article}".encode()).digest()
    return int.from_bytes(digest[:8], "big"), article


def select_articles(
    counts: dict[str, int], size: int, must_include: Iterable[str] = (), seed: int = 0
) -> tuple[set[str], list[str]]:
    chosen: set[str] = set()
    missing: list[str] = []
    total = 0
    for article in dict.fromkeys(must_include):
        if article in counts:
            chosen.add(article)
            total += counts[article]
        else:
            missing.append(article)
    for article in sorted((a for a in counts if a not in chosen), key=lambda a: _rank(seed, a)):
        if total >= size:
            break
        if total + counts[article] <= size:
            chosen.add(article)
            total += counts[article]
    return chosen, missing


def build_subset(
    source: Path,
    out: Path,
    size: int,
    must_include: Iterable[str] = (),
    seed: int = 0,
    strict: bool = False,
) -> dict:
    if size < 1:
        raise ValueError("size must be positive")
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(out)
    required = list(dict.fromkeys(must_include))
    counts: Counter[str] = Counter(article_of(p.id) for p in iter_passages(source))
    chosen, missing = select_articles(counts, size, required, seed)
    if strict and missing:
        raise ValueError(f"{len(missing)} required articles are not in {source}: {missing[:5]}")
    staging = out.with_name(out.name + ".part")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    def selected() -> Iterator[Passage]:
        for passage in iter_passages(source):
            if article_of(passage.id) in chosen:
                yield passage

    try:
        written = write_passages(staging / "passages.jsonl", selected())
        found = [a for a in required if a in counts]
        required_passages = sum(counts[a] for a in found)
        manifest = {
            "format": FORMAT,
            "source": {"name": source.name, "bytes": source.stat().st_size, "sha256": sha256_of(source)},
            "parameters": {"size": size, "seed": seed, "strict": strict},
            "source_totals": {"articles": len(counts), "passages": sum(counts.values())},
            "required": {
                "requested": len(required),
                "found": len(found),
                "missing_count": len(missing),
                "missing": missing[:MISSING_LISTED],
            },
            "selected": {
                "articles": len(chosen),
                "passages": written,
                "required_passages": required_passages,
                "filler_passages": written - required_passages,
                "over_budget": written > size,
            },
            "passages_sha256": sha256_of(staging / "passages.jsonl"),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    os.replace(staging, out)
    return manifest


def read_article_list(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cut a deterministic subset of whole articles from a passages file.")
    parser.add_argument("passages", type=Path, help="a passages.jsonl file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True, help="target number of passages")
    parser.add_argument("--article", action="append", default=[], help="an article id that must be included")
    parser.add_argument("--must-include", type=Path, default=None, help="a file with one article id per line")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="fail when a required article is absent")
    args = parser.parse_args(argv)
    required = list(args.article) + (read_article_list(args.must_include) if args.must_include else [])
    try:
        manifest = build_subset(args.passages, args.out, args.size, required, args.seed, args.strict)
    except ValueError as error:
        parser.error(str(error))
    chosen, totals = manifest["selected"], manifest["source_totals"]
    print(f"{chosen['passages']} of {totals['passages']} passages ({chosen['articles']} articles) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
