from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scout.data.parquet import iter_parquet
from scout.data.wikipedia import SKIPPED
from scout.retrieval.passages import Article, Block, Section, is_disambiguation, skipped_section

COLUMNS = ("name", "identifier", "url", "sections", "date_modified")
LISTS = ("list", "ordered_list", "definition_list")
LEAD = "Abstract"
_RESIDUE = re.compile(r"\{\{|\}\}|\{\||\|\}|</?ref\b", re.IGNORECASE)


def _clean(value: object) -> str:
    text = value.strip() if isinstance(value, str) else ""
    return "" if _RESIDUE.search(text) else text


def _items(node: Mapping, blocks: list[Block]) -> None:
    for item in node.get("has_parts") or []:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") in LISTS:
            _items(item, blocks)
            continue
        value = _clean(item.get("value"))
        if value:
            blocks.append(Block(value, item=True))
        for child in item.get("has_parts") or []:
            if isinstance(child, Mapping) and child.get("type") in LISTS:
                _items(child, blocks)


def parse_sections(nodes: Sequence | None) -> list[Section]:
    sections: list[Section] = []

    def walk(parts: Sequence | None, path: tuple[str, ...]) -> None:
        blocks: list[Block] = []

        def flush() -> None:
            if blocks:
                sections.append(Section(" > ".join(path), tuple(blocks)))
                blocks.clear()

        for part in parts or []:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind == "section":
                name = (part.get("name") or "").strip()
                if skipped_section(name):
                    continue
                flush()
                walk(part.get("has_parts"), path if not path and name == LEAD else path + (name,))
            elif kind == "paragraph":
                value = _clean(part.get("value"))
                if value:
                    blocks.append(Block(value))
            elif kind in LISTS:
                _items(part, blocks)
            elif kind not in SKIPPED:
                walk(part.get("has_parts"), path)
        flush()

    walk(nodes, ())
    return sections


def latest_rows(paths: Sequence[Path]) -> list[np.ndarray]:
    identifiers, stamps, files, rows, sizes = [], [], [], [], []
    for index, path in enumerate(paths):
        table = pq.read_table(path, columns=["identifier", "date_modified"])
        identifier = table.column("identifier").cast(pa.int64())
        stamp = table.column("date_modified").cast(pa.timestamp("us")).cast(pa.int64())
        identifiers.append(pc.fill_null(identifier, -1).to_numpy())
        stamps.append(pc.fill_null(stamp, 0).to_numpy())
        files.append(np.full(table.num_rows, index, dtype=np.int64))
        rows.append(np.arange(table.num_rows, dtype=np.int64))
        sizes.append(table.num_rows)
    keep = [np.zeros(size, dtype=bool) for size in sizes]
    if not sizes:
        return keep
    identifier, stamp, file, row = (np.concatenate(columns) for columns in (identifiers, stamps, files, rows))
    order = np.lexsort((row, file, stamp, identifier))
    last = np.append(identifier[order][1:] != identifier[order][:-1], True)
    for position in order[last & (identifier[order] >= 0)]:
        keep[file[position]][row[position]] = True
    return keep


def _day(value: object) -> str:
    return str(value)[:10] if value is not None else ""


def iter_articles(paths: Sequence[Path] | Path, date: str | None = None, batch_size: int = 64) -> Iterator[Article]:
    paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(path) for path in paths]
    keep = latest_rows(paths)
    for index, path in enumerate(paths):
        rows = iter_parquet(path, COLUMNS, batch_size, text_column="sections")
        for position, row in enumerate(rows):
            if not keep[index][position]:
                continue
            title = (row["name"] or "").strip()
            sections = parse_sections(json.loads(row["sections"]))
            if not sections or is_disambiguation(title, sections):
                continue
            if sections[0].blocks[0].text.lstrip().casefold().startswith("#redirect"):
                continue
            stamp = date or _day(row["date_modified"])
            yield Article(str(row["identifier"]), title, row["url"] or "", stamp, tuple(sections))
