from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from compression import zstd

from scout.data import finemath, finepdfs, fineweb_edu, nemotron, pes2o, stack_v2, stackexchange, wikipedia

DEFAULT_BYTES = 2_000_000_000
CHUNK_CHARACTERS = 8192
SAMPLE_FILE = "sample.jsonl.zst"
MANIFEST_FILE = "sample.json"
ORIGIN_KEYS = ("repo", "revision", "manifest_sha256")


@dataclass(frozen=True)
class Source:
    name: str
    share: float
    folder: str
    read: Callable[[Path], Iterator[dict[str, object]]]
    synthetic: bool = False


SOURCES = (
    Source("nemotron", 0.20, "nemotron", nemotron.iter_records),
    Source("nemotron_synthetic", 0.10, "nemotron", nemotron.iter_records, synthetic=True),
    Source("fineweb_edu", 0.20, "fineweb_edu", fineweb_edu.iter_records),
    Source("finepdfs", 0.15, "finepdfs", finepdfs.iter_records),
    Source("wikipedia", 0.10, "wikipedia", wikipedia.iter_records),
    Source("pes2o", 0.07, "pes2o", pes2o.iter_records),
    Source("stack_v2", 0.07, "stack_v2", stack_v2.iter_records),
    Source("stackexchange", 0.06, "stackexchange", stackexchange.iter_records),
    Source("finemath", 0.05, "finemath", finemath.iter_records),
)


def selected_files(folder: Path, synthetic: bool = False) -> tuple[dict[str, object], list[Path]]:
    listing = folder / "selection.json"
    if not listing.exists():
        raise FileNotFoundError(f"{listing} is missing, run the fetch for {folder.name} first")
    selection = json.loads(listing.read_text(encoding="utf-8"))
    if "repo" in selection:
        paths = [folder / file["path"] for file in selection["files"]]
    else:
        chosen = [nemotron.parse_path(path) for path in selection["files"]]
        paths = [folder / file.relative_path for file in chosen if file.synthetic == synthetic]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected files are not downloaded yet, the first is {missing[0]}")
    return selection, sorted(paths)


def chunk_text(text: str, limit: int = CHUNK_CHARACTERS) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = start + limit
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut > start:
                end = cut + 1
        yield text[start:end]
        start = end


def _documents(source: Source, paths: Sequence[Path]) -> Iterator[tuple[str, str]]:
    for path in paths:
        for record in source.read(path):
            yield record["id"], record["text"]


def build_sample(
    data: Path, out: Path, total_bytes: int = DEFAULT_BYTES, seed: int = 0, sources: Sequence[Source] = SOURCES
) -> dict[str, object]:
    if total_bytes < 1:
        raise ValueError("total_bytes must be positive")
    if len({source.name for source in sources}) != len(sources):
        raise ValueError("source names must be unique")
    if not math.isclose(sum(source.share for source in sources), 1.0):
        raise ValueError("source shares must add up to 1")
    located = []
    for source in sources:
        selection, paths = selected_files(data / source.folder, source.synthetic)
        if not paths:
            raise ValueError(f"{data / source.folder} has no files selected for {source.name}")
        located.append((source, selection, paths))
    targets = {source.name: round(total_bytes * source.share) for source in sources}
    available = {}
    for source, _, paths in located:
        available[source.name] = sum(len(text.encode()) for _, text in _documents(source, paths))
        print(f"{source.name}: {available[source.name] / 1e6:,.1f} MB available", flush=True)
    short = [name for name in targets if available[name] < targets[name]]
    if short:
        details = ", ".join(f"{name} has {available[name]:,} of {targets[name]:,} bytes" for name in short)
        raise ValueError(f"not enough text downloaded: {details}")
    out.mkdir(parents=True, exist_ok=True)
    partial = out / (SAMPLE_FILE + ".part")
    report = {}
    with zstd.ZstdFile(partial, "w") as stream:
        for source, selection, paths in located:
            keep = targets[source.name] / available[source.name]
            rng = random.Random(f"{seed}/{source.name}")
            sampled = chunks = 0
            for identifier, text in _documents(source, paths):
                for part, piece in enumerate(chunk_text(text)):
                    if rng.random() < keep:
                        record = {"source": source.name, "id": identifier, "part": part, "text": piece}
                        stream.write(json.dumps(record, ensure_ascii=False).encode() + b"\n")
                        sampled += len(piece.encode())
                        chunks += 1
            report[source.name] = {
                "share": source.share,
                "target_bytes": targets[source.name],
                "available_bytes": available[source.name],
                "keep_probability": keep,
                "sampled_bytes": sampled,
                "chunks": chunks,
                "origin": {key: selection[key] for key in ORIGIN_KEYS if key in selection},
                "files": [path.relative_to(data / source.folder).as_posix() for path in paths],
            }
            print(f"{source.name}: kept {chunks:,} chunks, {sampled / 1e6:,.1f} MB", flush=True)
    partial.replace(out / SAMPLE_FILE)
    manifest = {"seed": seed, "total_bytes": total_bytes, "chunk_characters": CHUNK_CHARACTERS, "sources": report}
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def iter_sample(path: Path) -> Iterator[dict[str, str]]:
    with zstd.ZstdFile(path) as lines:
        for line in lines:
            yield json.loads(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample every source in fixed byte shares for tokenizer training.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bytes", type=int, default=DEFAULT_BYTES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    build_sample(args.data, args.out, args.bytes, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
