from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from scout.data.fetch import run_hub_fetch
from scout.data.parquet import iter_parquet

REPO = "wikimedia/structured-wikipedia"
DEFAULT_FOLDER = "enwiki/data"
COLUMNS = ("name", "identifier", "url", "sections", "references")
LEAD = "Abstract"
SKIPPED = frozenset({"image", "gallery", "table"})
REFERENCES_HEADING = "References"
_REDIRECT = re.compile(r"#\s*redirect\b", re.IGNORECASE)
_LINE_BREAK = re.compile(r"\s*[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]\s*")


def reference_texts(references: Sequence[Mapping] | None) -> dict[str, str]:
    texts = {}
    for reference in references or []:
        identifier = reference.get("identifier")
        value = _LINE_BREAK.sub(" ", ((reference.get("text") or {}).get("value") or "").strip())
        if identifier and value:
            texts[identifier] = value
    return texts


class _Renderer:
    def __init__(self, references: dict[str, str]) -> None:
        self.references = references
        self.labels: dict[str, str] = {}
        self.blocks: list[str] = []

    def markers(self, citations: Sequence[Mapping] | None) -> str:
        found = []
        for citation in citations or []:
            identifier = citation.get("identifier")
            if identifier not in self.references:
                continue
            label = self.labels.setdefault(identifier, f"[{len(self.labels) + 1}]")
            if label not in found:
                found.append(label)
        return "".join(found)

    def section(self, node: Mapping, depth: int) -> None:
        name = (node.get("name") or "").strip()
        if name.casefold() == REFERENCES_HEADING.casefold():
            return
        outer, self.blocks = self.blocks, []
        self.parts(node.get("has_parts"), depth)
        inner, self.blocks = self.blocks, outer
        if not inner:
            return
        if name and not (depth == 0 and name == LEAD):
            self.blocks.append("#" * min(depth + 2, 6) + " " + name)
        self.blocks.extend(inner)

    def parts(self, parts: Sequence | None, depth: int) -> None:
        lines: list[str] = []
        for part in parts or []:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind in ("list", "ordered_list", "definition_list"):
                self.list_lines(part, 0, lines)
                continue
            if lines:
                self.blocks.append("\n".join(lines))
                lines = []
            if kind == "section":
                self.section(part, depth + 1)
            elif kind == "paragraph":
                value = (part.get("value") or "").strip()
                if value:
                    self.blocks.append(value + self.markers(part.get("citations")))
            elif kind not in SKIPPED:
                self.parts(part.get("has_parts"), depth)
        if lines:
            self.blocks.append("\n".join(lines))

    def list_lines(self, node: Mapping, indent: int, lines: list[str]) -> None:
        ordered = node.get("type") == "ordered_list"
        number = 0
        for item in node.get("has_parts") or []:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("type")
            value = (item.get("value") or "").strip()
            if kind in ("list", "ordered_list", "definition_list"):
                self.list_lines(item, indent + 1, lines)
                continue
            if kind == "definition_term":
                prefix = ""
            elif kind == "definition":
                prefix = ": "
            else:
                number += 1
                prefix = f"{number}. " if ordered else "- "
            if value:
                lines.append("  " * indent + prefix + value)
            for child in item.get("has_parts") or []:
                if isinstance(child, Mapping) and child.get("type") in ("list", "ordered_list", "definition_list"):
                    self.list_lines(child, indent + 1, lines)


def render_article(name: str, sections: Sequence[Mapping], references: Sequence[Mapping] | None) -> tuple[str, int]:
    renderer = _Renderer(reference_texts(references))
    for section in sections if isinstance(sections, list) else []:
        if isinstance(section, Mapping) and section.get("type") == "section":
            renderer.section(section, 0)
    if not renderer.blocks or any(_REDIRECT.match(block) for block in renderer.blocks):
        return "", 0
    blocks = [f"# {name.strip()}", *renderer.blocks]
    if renderer.labels:
        entries = [f"{label} {renderer.references[identifier]}" for identifier, label in renderer.labels.items()]
        blocks += [f"## {REFERENCES_HEADING}", "\n".join(entries)]
    return "\n\n".join(blocks), len(renderer.labels)


def iter_records(path: Path, batch_size: int = 64) -> Iterator[dict[str, object]]:
    for row in iter_parquet(path, COLUMNS, batch_size, text_column="sections"):
        text, citations = render_article(row["name"] or "", json.loads(row["sections"]), row["references"])
        if text:
            yield {
                "text": text,
                "url": row["url"] or "",
                "id": str(row["identifier"]),
                "title": row["name"] or "",
                "citations": citations,
            }


def main(argv: Sequence[str] | None = None) -> int:
    description = "Download English structured Wikipedia parquet files at a pinned revision."
    return run_hub_fetch(REPO, DEFAULT_FOLDER, description, argv)


if __name__ == "__main__":
    sys.exit(main())
