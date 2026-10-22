from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_WORDS = 100
MIN_WORDS = 20
SKIPPED_SECTIONS = frozenset(
    {
        "references",
        "external links",
        "see also",
        "further reading",
        "notes",
        "bibliography",
        "footnotes",
        "sources",
        "citations",
    }
)
ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof sr jr st mt ft gen col lt capt sgt cmdr adm maj rev hon gov sen rep pres vs etc inc ltd co corp "
    "no vol pp ed eds fig approx ca cf viz jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
_BOUNDARY = re.compile(r"([.!?][\"')\]]*)\s+(?=[\"'(\[]?[A-Z0-9])")
_INITIALS = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")
_DISAMBIGUATION = re.compile(r"\bmay (?:also )?refer to\b", re.IGNORECASE)
FIELDS = ("id", "title", "section", "text", "url", "date")


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    section: str
    text: str
    url: str
    date: str


@dataclass(frozen=True)
class Block:
    text: str
    item: bool = False


@dataclass(frozen=True)
class Section:
    name: str
    blocks: Sequence[Block]


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    url: str
    date: str
    sections: Sequence[Section]


def skipped_section(name: str) -> bool:
    return name.strip().casefold() in SKIPPED_SECTIONS


def is_disambiguation(title: str, sections: Sequence[Section]) -> bool:
    if title.endswith("(disambiguation)"):
        return True
    lead = next((block.text for section in sections for block in section.blocks if not block.item), "")
    return bool(_DISAMBIGUATION.search(lead[:300]))


def _abbreviated(prefix: str) -> bool:
    words = prefix.split()
    token = words[-1].lstrip("\"'([") if words else ""
    return token.lower() in ABBREVIATIONS or (len(token) == 1 and token.isupper()) or bool(_INITIALS.fullmatch(token))


def split_sentences(text: str) -> list[str]:
    sentences, start = [], 0
    for match in _BOUNDARY.finditer(text):
        if match.group(1)[0] == "." and _abbreviated(text[start : match.start()]):
            continue
        sentences.append(text[start : match.end(1)])
        start = match.end()
    sentences.append(text[start:])
    return [sentence for sentence in sentences if sentence]


def _units(blocks: Iterable[Block], max_words: int) -> list[tuple[str, int, str]]:
    units = []
    for block in blocks:
        for number, piece in enumerate([block.text] if block.item else split_sentences(block.text)):
            lead = "\n" if number == 0 else " "
            words = piece.split()
            for offset in range(0, len(words), max_words):
                chunk = words[offset : offset + max_words]
                units.append((" ".join(chunk), len(chunk), lead if offset == 0 else " "))
    return units


def _join(units: Sequence[tuple[str, int, str]]) -> str:
    return "".join((lead if number else "") + text for number, (text, _, lead) in enumerate(units))


def chunk_blocks(blocks: Iterable[Block], max_words: int = MAX_WORDS, min_words: int = MIN_WORDS) -> list[str]:
    if max_words < 1 or min_words < 0 or min_words > max_words:
        raise ValueError("need 1 <= max_words and 0 <= min_words <= max_words")
    groups: list[list[tuple[str, int, str]]] = []
    current: list[tuple[str, int, str]] = []
    count = 0
    for unit in _units(blocks, max_words):
        if current and count + unit[1] > max_words:
            groups.append(current)
            current, count = [], 0
        current.append(unit)
        count += unit[1]
    if current:
        groups.append(current)
    if len(groups) > 1 and sum(unit[1] for unit in groups[-1]) < min_words:
        groups[-2].extend(groups.pop())
    return [_join(group) for group in groups]


def build_passages(article: Article, max_words: int = MAX_WORDS, min_words: int = MIN_WORDS) -> Iterator[Passage]:
    number = 0
    for section in article.sections:
        for text in chunk_blocks(section.blocks, max_words, min_words):
            yield Passage(f"{article.id}:{number}", article.title, section.name, text, article.url, article.date)
            number += 1


def write_passages(path: Path, passages: Iterable[Passage]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    count = 0
    with open(partial, "w", encoding="utf-8") as stream:
        for passage in passages:
            stream.write(json.dumps(asdict(passage), ensure_ascii=False) + "\n")
            count += 1
    os.replace(partial, path)
    return count


def iter_passages(path: Path) -> Iterator[Passage]:
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
                if set(record) != set(FIELDS) or not all(isinstance(record[name], str) for name in FIELDS):
                    raise ValueError
            except ValueError:
                raise ValueError(f"{path} line {number} is not a passage") from None
            yield Passage(**record)
