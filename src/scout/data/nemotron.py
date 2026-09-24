from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import io
import json
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from compression import zstd

BASE_URL = "https://data.commoncrawl.org/"
PREFIX = "contrib/Nemotron/Nemotron-CC/data-jsonl/"
MANIFEST_PATH = "contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"
QUALITIES = ("high", "medium-high", "medium", "medium-low", "low")
DEFAULT_PARTITIONS = (
    ("high", "actual"),
    ("high", "distill"),
    ("high", "diverse_qa_pairs"),
    ("high", "extract_knowledge"),
    ("high", "knowledge_list"),
    ("high", "wrap_medium"),
)
_PATH = re.compile(
    r"^contrib/Nemotron/Nemotron-CC/data-jsonl/quality=(?P<quality>[a-z-]+)/kind=(?P<kind>actual|synthetic)/"
    r"kind2=(?P<kind2>[a-z_]+)/(?P<crawl>CC-MAIN-\d{4}-\d{2})-part-(?P<part>\d{5})\.jsonl\.zstd$"
)
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_UNSATISFIED_RANGE = re.compile(r"^bytes \*/(\d+)$")


@dataclass(frozen=True)
class NemotronFile:
    path: str
    quality: str
    kind: str
    kind2: str
    crawl: str
    part: int

    @property
    def synthetic(self) -> bool:
        return self.kind == "synthetic"

    @property
    def partition(self) -> tuple[str, str]:
        return (self.quality, self.kind2)

    @property
    def relative_path(self) -> Path:
        return Path(self.path.removeprefix(PREFIX))


def parse_path(path: str) -> NemotronFile:
    path = path.strip()
    match = _PATH.match(path)
    if match is None or match["quality"] not in QUALITIES:
        raise ValueError(f"not a Nemotron CC data path: {path!r}")
    if (match["kind"] == "actual") != (match["kind2"] == "actual"):
        raise ValueError(f"kind and kind2 disagree: {path!r}")
    return NemotronFile(path, match["quality"], match["kind"], match["kind2"], match["crawl"], int(match["part"]))


def parse_manifest(lines: Iterable[str]) -> list[NemotronFile]:
    return [parse_path(line) for line in lines if line.strip()]


def select_files(
    files: Sequence[NemotronFile],
    partitions: Iterable[tuple[str, str]],
    per_partition: int | None = None,
    seed: int = 0,
) -> list[NemotronFile]:
    if per_partition is not None and per_partition < 1:
        raise ValueError("per_partition must be positive")
    chosen = []
    for partition in sorted(set(partitions)):
        group = sorted((f for f in files if f.partition == partition), key=lambda f: f.path)
        if not group:
            raise ValueError(f"no files for partition {partition}")
        if per_partition is not None and per_partition < len(group):
            rng = random.Random(f"{seed}/{partition[0]}/{partition[1]}")
            group = sorted(rng.sample(group, per_partition), key=lambda f: f.path)
        chosen.extend(group)
    return chosen


def iter_records(stream: BinaryIO) -> Iterator[dict[str, str]]:
    with zstd.ZstdFile(stream) as decompressed:
        for number, line in enumerate(io.TextIOWrapper(decompressed, encoding="utf-8", newline="\n"), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"record {number} has no text")
            yield {"text": text, "url": record.get("url") or "", "id": record.get("warc_record_id") or ""}


def download(url: str, destination: Path, timeout: float = 60.0) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
    except urllib.error.HTTPError as error:
        match = _UNSATISFIED_RANGE.match(error.headers.get("Content-Range", ""))
        if error.code != 416 or match is None:
            raise
        if start != int(match[1]):
            partial.unlink()
            raise OSError(f"{partial} did not match the {match[1]} byte file, rerun to download afresh") from error
        partial.replace(destination)
        return destination
    with response:
        content_range = response.headers.get("Content-Range", "")
        if start and response.status == 206:
            match = _CONTENT_RANGE.match(content_range)
            if match is None or int(match[1]) != start:
                raise OSError(f"unexpected Content-Range {content_range!r} for {url}")
            total = int(match[3])
        else:
            start = 0
            length = response.headers.get("Content-Length")
            total = int(length) if length is not None else None
        with open(partial, "ab" if start else "wb") as out:
            try:
                shutil.copyfileobj(response, out, 1 << 20)
            except http.client.IncompleteRead as error:
                raise OSError(f"{url} was cut off, rerun to resume") from error
    size = partial.stat().st_size
    if total is not None and size != total:
        raise OSError(f"{url} stopped at {size} of {total} bytes, rerun to resume")
    partial.replace(destination)
    return destination


def download_with_retries(url: str, destination: Path, retries: int, wait: float) -> Path:
    for attempt in range(retries):
        try:
            return download(url, destination)
        except OSError as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code not in (408, 429):
                raise
            print(f"retrying {url} after: {error}", file=sys.stderr, flush=True)
            time.sleep(wait * 2**attempt)
    return download(url, destination)


def parse_partitions(text: str) -> list[tuple[str, str]]:
    partitions = []
    for item in text.split(","):
        quality, _, kind2 = item.strip().partition("/")
        if quality not in QUALITIES or not kind2:
            raise ValueError(f"expected quality/kind2, got {item!r}")
        partitions.append((quality, kind2))
    return partitions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Nemotron CC v1 files, real and synthetic kept apart.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partitions", default=",".join(f"{q}/{k}" for q, k in DEFAULT_PARTITIONS))
    parser.add_argument("--files-per-partition", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.retries < 0 or args.retry_wait < 0:
        parser.error("retries and retry wait must not be negative")
    manifest_url = args.base_url + MANIFEST_PATH
    manifest = download_with_retries(manifest_url, args.out / "data-jsonl.paths.gz", args.retries, args.retry_wait)
    with gzip.open(manifest, "rt", encoding="utf-8") as lines:
        files = parse_manifest(lines)
    chosen = select_files(files, parse_partitions(args.partitions), args.files_per_partition, args.seed)
    selection = {
        "partitions": args.partitions,
        "files_per_partition": args.files_per_partition,
        "seed": args.seed,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "files": [f.path for f in chosen],
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"selected {len(chosen)} of {len(files)} files")
    if args.dry_run:
        return 0
    for number, file in enumerate(chosen, start=1):
        download_with_retries(args.base_url + file.path, args.out / file.relative_path, args.retries, args.retry_wait)
        print(f"{number}/{len(chosen)} {file.relative_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
