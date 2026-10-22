from __future__ import annotations

import bz2
import html
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from urllib.parse import quote

from scout.retrieval.passages import Article, Block, Section, is_disambiguation, skipped_section

BASE_URL = "https://en.wikipedia.org/wiki/"
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_DROPPED = re.compile(
    r"<(ref|gallery|math|timeline|imagemap|syntaxhighlight|source|score|hiero|mapframe|maplink|references|chem)\b"
    r"[^>]*?(?:/>|>.*?</\1\s*>)",
    re.DOTALL | re.IGNORECASE,
)
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_TEMPLATE = re.compile(r"\{\{([^{}]*)\}\}")
_TABLE = re.compile(r"\{\|(?:(?!\{\|).)*?\|\}", re.DOTALL)
_LINK = re.compile(r"\[\[((?:[^\[\]]|\[(?!\[)|\](?!\]))*)\]\]")
_EXTERNAL = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:[ \t]+([^\]]*))?\]")
_URL = re.compile(r"https?://[^\s<>\]\[|]+")
_QUOTES = re.compile(r"'{2,}")
_MAGIC = re.compile(r"__[A-Z]+__")
_HEADING = re.compile(r"(={1,6})\s*(.*?)\s*\1")
_HATNOTE = re.compile(
    r"[:*]+\s*(?:the\s+)?(?:main article|see also|further information|for other|not to be confused|this article)",
    re.IGNORECASE,
)
_FILE_NAMESPACES = frozenset({"file", "image", "category", "media"})
_RANGE_WORDS = frozenset({"to", "and", "or", "by"})
_PAREN = re.compile(r"\s*\([^)]*\)$")
_OPENING = re.compile(r"\(\s*[;,]\s*")
_EMPTY = re.compile(r"[ \t]*\(\s*\)")


def _arguments(body: str) -> list[str]:
    parts, depth, start, index = [], 0, 0, 0
    while index < len(body):
        pair = body[index : index + 2]
        if pair in ("[[", "]]"):
            depth = depth + 1 if pair == "[[" else max(0, depth - 1)
            index += 2
            continue
        if body[index] == "|" and depth == 0:
            parts.append(body[start:index])
            start = index + 1
        index += 1
    parts.append(body[start:])
    return parts


def _positional(body: str) -> list[str]:
    return [part.strip() for part in _arguments(body) if "=" not in re.split(r"[\[{]", part, maxsplit=1)[0]]


def _template(match: re.Match[str]) -> str:
    parts = _positional(match.group(1))
    if not parts:
        return ""
    name = parts[0].lower()
    if name in ("lang", "langx"):
        return parts[2] if len(parts) > 2 else ""
    if name.startswith("lang-"):
        return parts[1] if len(parts) > 1 else ""
    if name in ("nowrap", "small", "smaller", "larger"):
        return parts[1] if len(parts) > 1 else ""
    if name == "convert" and len(parts) > 2:
        if parts[2] in _RANGE_WORDS and len(parts) > 4:
            return f"{parts[1]} {parts[2]} {parts[3]} {parts[4]}"
        return f"{parts[1]} {parts[2]}"
    return ""


def _link(match: re.Match[str]) -> str:
    target, pipe, label = match.group(1).partition("|")
    target = target.strip()
    namespace, colon, _ = target.partition(":")
    if colon and namespace.strip().casefold() in _FILE_NAMESPACES:
        return ""
    if not pipe:
        return target.lstrip(":")
    if not label.strip():
        return _PAREN.sub("", target.lstrip(":"))
    return label


def _external(match: re.Match[str]) -> str:
    return match.group(1) or ""


def _repeat(steps: Sequence[tuple[re.Pattern[str], Callable[[re.Match[str]], str] | str]], text: str) -> str:
    while True:
        updated = text
        for pattern, replacement in steps:
            updated = pattern.sub(replacement, updated)
        if updated == text:
            return text
        text = updated


def strip_markup(text: str) -> str:
    text = _COMMENT.sub("", text)
    text = _DROPPED.sub("", text)
    text = _BREAK.sub(" ", text)
    text = _TAG.sub("", text)
    text = _repeat([(_TEMPLATE, _template)], text)
    text = _repeat([(_TABLE, "")], text)
    text = _repeat([(_EXTERNAL, _external), (_LINK, _link)], text)
    text = _URL.sub("", text)
    text = _QUOTES.sub("", text)
    text = _MAGIC.sub("", text)
    text = html.unescape(text).replace("{{", "").replace("}}", "")
    text = _OPENING.sub("(", text)
    return _EMPTY.sub("", text)


def parse_text(text: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    blocks: list[Block] = []
    paragraph: list[str] = []
    skipping = 0

    def flush_paragraph() -> None:
        joined = " ".join(" ".join(paragraph).split())
        paragraph.clear()
        if any(character.isalpha() for character in joined):
            blocks.append(Block(joined))

    def flush_section() -> None:
        flush_paragraph()
        if blocks and not skipping:
            sections.append(Section(" > ".join(name for _, name in stack), tuple(blocks)))
        blocks.clear()

    for line in strip_markup(text).splitlines():
        line = line.strip()
        heading = _HEADING.fullmatch(line)
        if heading:
            flush_section()
            level, name = len(heading.group(1)), " ".join(heading.group(2).split())
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, name))
            if skipping and level <= skipping:
                skipping = 0
            if not skipping and skipped_section(name):
                skipping = level
        elif not line or (len(line) >= 4 and set(line) == {"-"}):
            flush_paragraph()
        elif line[0] in "*#:;":
            flush_paragraph()
            item = line.lstrip("*#:; ")
            if not _HATNOTE.match(line) and any(character.isalpha() for character in item):
                blocks.append(Block(" ".join(item.split()), item=True))
        elif line[0] in "|!{":
            continue
        else:
            paragraph.append(line)
    flush_section()
    return sections


def parse_page(page_id: str, title: str, text: str, date: str) -> Article | None:
    if text.lstrip().casefold().startswith("#redirect"):
        return None
    sections = parse_text(text)
    if not sections or is_disambiguation(title, sections):
        return None
    url = BASE_URL + quote(title.replace(" ", "_"), safe="/:@!$&'()*+,;=~")
    return Article(page_id, title, url, date, tuple(sections))


def _local(tag: str) -> str:
    return tag.rpartition("}")[2]


def iter_pages(path: Path) -> Iterator[tuple[str, str, str, str]]:
    path = Path(path)
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rb") as stream:
        root = None
        for event, element in ElementTree.iterparse(stream, events=("start", "end")):
            if root is None:
                root = element
            if event != "end" or _local(element.tag) != "page":
                continue
            fields = {_local(child.tag): child for child in element}
            revision = fields.get("revision")
            text = ""
            if revision is not None:
                body = next((child for child in revision if _local(child.tag) == "text"), None)
                text = (body.text or "") if body is not None else ""
            namespace = fields["ns"].text if "ns" in fields else ""
            if namespace == "0" and "redirect" not in fields:
                yield (fields["id"].text or "", (fields["title"].text or "").strip(), text, namespace)
            root.clear()


def iter_articles(paths: Sequence[Path] | Path, date: str) -> Iterator[Article]:
    for path in [Path(paths)] if isinstance(paths, (str, Path)) else paths:
        for page_id, title, text, _ in iter_pages(path):
            article = parse_page(page_id, title, text, date)
            if article is not None:
                yield article
