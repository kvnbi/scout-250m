from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

HUB_URL = "https://huggingface.co/"
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_UNSATISFIED_RANGE = re.compile(r"^bytes \*/(\d+)$")
_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, sha256: str | None = None, timeout: float = 60.0) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    total = None
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code != 416 or not start:
            raise
        match = _UNSATISFIED_RANGE.match(error.headers.get("Content-Range") or "")
        if match is not None:
            complete = start == int(match[1])
        else:
            complete = sha256 is not None and sha256_of(partial) == sha256
        if not complete:
            partial.unlink()
            raise OSError(f"{partial} could not be confirmed complete, rerun to download afresh") from error
        response = None
    if response is not None:
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
    if sha256 is not None and sha256_of(partial) != sha256:
        partial.unlink()
        raise OSError(f"{url} failed its SHA-256 check, rerun to download afresh")
    partial.replace(destination)
    return destination


def download_with_retries(
    url: str, destination: Path, retries: int, wait: float, sha256: str | None = None
) -> Path:
    for attempt in range(retries):
        try:
            return download(url, destination, sha256)
        except OSError as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code not in (408, 429):
                raise
            print(f"retrying {url} after: {error}", file=sys.stderr, flush=True)
            time.sleep(wait * 2**attempt)
    return download(url, destination, sha256)


@dataclass(frozen=True)
class HubFile:
    path: str
    size: int
    sha256: str


def _get_json(url: str, timeout: float = 60.0) -> tuple[object, str | None]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        link = response.headers.get("Link", "")
        match = _NEXT_LINK.search(link)
        return json.load(response), (match[1] if match else None)


def hub_revision(repo: str, base_url: str = HUB_URL) -> str:
    info, _ = _get_json(f"{base_url}api/datasets/{repo}")
    revision = info.get("sha") if isinstance(info, dict) else None
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"no commit hash for {repo}")
    return revision


def hub_files(repo: str, revision: str, folder: str, suffix: str, base_url: str = HUB_URL) -> list[HubFile]:
    folder = folder.strip("/")
    url = f"{base_url}api/datasets/{repo}/tree/{revision}"
    if folder:
        url += f"/{urllib.parse.quote(folder)}"
    files = []
    while url is not None:
        entries, url = _get_json(url)
        for entry in entries:
            if entry.get("type") != "file" or not entry["path"].endswith(suffix):
                continue
            path = PurePosixPath(entry["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe path in the listing: {entry['path']!r}")
            lfs = entry.get("lfs") or {}
            if not re.fullmatch(r"[0-9a-f]{64}", lfs.get("oid", "")):
                raise ValueError(f"{entry['path']} has no SHA-256 in the listing")
            files.append(HubFile(entry["path"], int(lfs.get("size", entry["size"])), lfs["oid"]))
    if not files:
        raise ValueError(f"no {suffix} files in {repo}/{folder} at {revision}")
    return sorted(files, key=lambda f: f.path)


def hub_file_url(repo: str, revision: str, path: str, base_url: str = HUB_URL) -> str:
    return f"{base_url}datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def sample_files(files: Sequence[HubFile], count: int | None, seed: int) -> list[HubFile]:
    if count is not None and count < 1:
        raise ValueError("count must be positive")
    if count is None or count >= len(files):
        return sorted(files, key=lambda f: f.path)
    population = sorted(files, key=lambda f: f.path)
    return sorted(random.Random(f"{seed}").sample(population, count), key=lambda f: f.path)


def run_hub_fetch(
    repo: str, default_folder: str, description: str, argv: Sequence[str] | None = None, suffix: str = ".parquet"
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folder", default=default_folder)
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
    revision = args.revision or hub_revision(repo, args.base_url)
    chosen = sample_files(hub_files(repo, revision, args.folder, suffix, args.base_url), args.files, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    selection = {
        "repo": repo,
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
        url = hub_file_url(repo, revision, file.path, args.base_url)
        download_with_retries(url, args.out / file.path, args.retries, args.retry_wait, file.sha256)
        print(f"{number}/{len(chosen)} {file.path}", flush=True)
    return 0
