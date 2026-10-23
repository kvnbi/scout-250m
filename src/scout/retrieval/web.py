from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.parse
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from scout.data.fetch import sha256_of
from scout.data.parquet import iter_parquet
from scout.retrieval.dates import compare_dates, normalize_date
from scout.retrieval.passages import (
    MAX_WORDS,
    MIN_WORDS,
    Article,
    Block,
    Passage,
    Section,
    build_passages,
    write_passages,
)

FORMAT = 1
MIN_PAGE_WORDS = 50
EXCLUDED_DOMAINS = ("wikipedia.org",)
TITLE_WORDS = 12
TITLE_CHARACTERS = 100
FINEWEB_COLUMNS = ("text", "id", "url", "dump", "file_path")
_CRAWL_STAMP = re.compile(r"CC-MAIN-(\d{4})(\d{2})(\d{2})\d{6}")
_DUMP = re.compile(r"CC-MAIN-(\d{4})-(\d{2})")
_URL_DATE = re.compile(r"/((?:19|20)[0-9]{2})/(0[1-9]|1[0-2])(?:/(0[1-9]|[12][0-9]|3[01]))?(?=/|$)")
_EXTENSION = re.compile(r"\.(?:html?|php\d?|aspx?|jsp|shtml|cfm|cgi)$", re.IGNORECASE)


@dataclass(frozen=True)
class Page:
    id: str
    url: str
    text: str
    date: str
    crawled: str = ""


def url_date(url: str) -> str:
    match = _URL_DATE.search(urllib.parse.urlsplit(url).path)
    if match is None:
        return ""
    return normalize_date("-".join(part for part in match.groups() if part))


def page_date(url: str, crawled: str) -> str:
    published = url_date(url)
    if published and (not crawled or compare_dates(published, crawled) <= 0):
        return published
    return crawled


def crawl_date(file_path: str | None, dump: str | None) -> str:
    stamp = _CRAWL_STAMP.search(file_path or "")
    if stamp:
        found = normalize_date("".join(stamp.groups()))
        if found:
            return found
    week = _DUMP.fullmatch(dump or "")
    if week:
        try:
            return normalize_date(date.fromisocalendar(int(week[1]), int(week[2]), 4).isoformat()[:7])
        except ValueError:
            return ""
    return ""


def clean_id(raw: str | None) -> str:
    text = (raw or "").strip()
    if text.startswith("<urn:uuid:") and text.endswith(">"):
        text = text[len("<urn:uuid:") : -1]
    return text.replace(":", "-").replace(" ", "-")


def hostname(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or ""


def excluded(url: str, domains: Sequence[str]) -> bool:
    host = hostname(url)
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def url_title(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    segments = [urllib.parse.unquote(s) for s in parts.path.split("/") if s]
    for segment in reversed(segments):
        words = " ".join(re.sub(r"[-_+]+", " ", _EXTENSION.sub("", segment)).split())
        if len(words) >= 3 and not words.replace(" ", "").isdigit():
            return (words[0].upper() + words[1:])[:TITLE_CHARACTERS]
    return (parts.hostname or "")[:TITLE_CHARACTERS]


def looks_like_title(line: str) -> bool:
    words = line.split()
    return 1 <= len(words) <= TITLE_WORDS and len(line) <= TITLE_CHARACTERS and line[-1] not in ".;:,"


def page_article(page: Page, min_page_words: int = MIN_PAGE_WORDS) -> Article | None:
    lines = [" ".join(line.split()) for line in page.text.split("\n")]
    lines = [line for line in lines if line]
    if len(" ".join(lines).split()) < min_page_words:
        return None
    if len(lines) > 1 and looks_like_title(lines[0]):
        title, body = lines[0], lines[1:]
    else:
        title, body = url_title(page.url), lines
    return Article(page.id, title, page.url, page.date, [Section("", [Block(line) for line in body])])


def iter_fineweb_pages(path: Path) -> Iterator[Page]:
    for row in iter_parquet(path, FINEWEB_COLUMNS, 1024):
        url, crawled = row["url"] or "", crawl_date(row["file_path"], row["dump"])
        yield Page(clean_id(row["id"]), url, row["text"], page_date(url, crawled), crawled)


READERS: dict[str, Callable[[Path], Iterable[Page]]] = {"fineweb-edu": iter_fineweb_pages}


def build_web_corpus(
    inputs: Sequence[Path],
    out: Path,
    source: str = "fineweb-edu",
    exclude_domains: Sequence[str] = EXCLUDED_DOMAINS,
    min_page_words: int = MIN_PAGE_WORDS,
    max_pages: int | None = None,
    max_words: int = MAX_WORDS,
    min_words: int = MIN_WORDS,
) -> dict:
    if source not in READERS:
        raise ValueError(f"source must be one of {sorted(READERS)}")
    if min_page_words < 0 or (max_pages is not None and max_pages < 1):
        raise ValueError("need min_page_words >= 0 and max_pages >= 1")
    if max_words < 1 or not 0 <= min_words <= max_words:
        raise ValueError("need 1 <= max_words and 0 <= min_words <= max_words")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    staging = out.with_name(out.name + ".part")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    seen: set[str] = set()
    domains: set[str] = set()
    stamps: list[str] = []
    counts_from_url = {"pages": 0}
    dropped: Counter[str] = Counter()
    counts = {"read": 0, "kept": 0}

    def passages() -> Iterator[Passage]:
        for path in inputs:
            for page in READERS[source](path):
                counts["read"] += 1
                if not page.url:
                    dropped["no_url"] += 1
                elif page.url in seen:
                    dropped["duplicate_url"] += 1
                elif excluded(page.url, exclude_domains):
                    dropped["excluded_domain"] += 1
                else:
                    article = page_article(page, min_page_words)
                    if article is None:
                        dropped["too_short"] += 1
                        continue
                    seen.add(page.url)
                    domains.add(hostname(page.url))
                    stamps.append(page.date)
                    counts_from_url["pages"] += int(page.date != page.crawled)
                    counts["kept"] += 1
                    yield from build_passages(article, max_words, min_words)
                    if max_pages is not None and counts["kept"] >= max_pages:
                        return

    try:
        written = write_passages(staging / "passages.jsonl", passages())
        dated = [s for s in stamps if s]
        manifest = {
            "format": FORMAT,
            "source": source,
            "inputs": [_describe(Path(path)) for path in inputs],
            "parameters": {
                "exclude_domains": list(exclude_domains),
                "min_page_words": min_page_words,
                "max_pages": max_pages,
                "max_words": max_words,
                "min_words": min_words,
            },
            "pages": {"read": counts["read"], "kept": counts["kept"], "dropped": dict(sorted(dropped.items()))},
            "passages": written,
            "domains": len(domains),
            "dates": {
                "first": _extreme(dated, -1),
                "last": _extreme(dated, 1),
                "undated": len(stamps) - len(dated),
                "from_url": counts_from_url["pages"],
            },
            "passages_sha256": sha256_of(staging / "passages.jsonl"),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    os.replace(staging, out)
    return manifest


def _describe(path: Path) -> dict:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_of(path)}


def _extreme(stamps: list[str], direction: int) -> str:
    best = ""
    for stamp in stamps:
        if not best or compare_dates(stamp, best) == direction:
            best = stamp
    return best


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze web pages into a dated passage corpus.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--source", choices=sorted(READERS), default="fineweb-edu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-domain", action="append", default=None, dest="exclude")
    parser.add_argument("--min-page-words", type=int, default=MIN_PAGE_WORDS)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    args = parser.parse_args(argv)
    try:
        manifest = build_web_corpus(
            args.inputs,
            args.out,
            args.source,
            tuple(args.exclude or EXCLUDED_DOMAINS),
            args.min_page_words,
            args.max_pages,
            args.max_words,
            args.min_words,
        )
    except ValueError as error:
        parser.error(str(error))
    pages = manifest["pages"]
    print(f"{pages['kept']} of {pages['read']} pages, {manifest['passages']} passages -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
