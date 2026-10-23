from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from scout.retrieval.passages import Passage
from scout.retrieval.search import Hit
from scout.retrieval.store import PassageStore, PassageWriter, write_starts

FORMAT = 1
K1 = 0.9
B = 0.4
CHUNK_DOCUMENTS = 50_000
BLOCK_POSTINGS = 8_000_000
MAX_TOKEN_LENGTH = 64
MAX_FREQUENCY = 65535
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their then there these they this "
    "to was will with".split()
)
_TOKEN = re.compile(r"[^\W_]+")


def tokenize(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(character for character in folded if not unicodedata.combining(character))
    return [token for token in _TOKEN.findall(plain) if len(token) <= MAX_TOKEN_LENGTH and token not in STOPWORDS]


def _document_text(passage: Passage, include_title: bool) -> str:
    return f"{passage.title}\n{passage.text}" if include_title else passage.text


def build_index(
    passages: Iterable[Passage],
    directory: Path,
    k1: float = K1,
    b: float = B,
    include_title: bool = True,
    chunk_documents: int = CHUNK_DOCUMENTS,
) -> int:
    if k1 <= 0 or not 0 <= b <= 1 or chunk_documents < 1:
        raise ValueError("need k1 > 0, 0 <= b <= 1 and chunk_documents >= 1")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    staging = directory.with_name(directory.name + ".part")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    vocabulary: dict[str, int] = {}
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    pending: tuple[list[int], list[int], list[int]] = ([], [], [])
    lengths: list[int] = []

    def flush() -> None:
        if pending[0]:
            chunks.append(
                (
                    np.array(pending[0], dtype=np.int32),
                    np.array(pending[1], dtype=np.int32),
                    np.array(pending[2], dtype=np.uint16),
                )
            )
            for column in pending:
                column.clear()

    writer = PassageWriter(staging)
    try:
        for number, passage in enumerate(passages):
            writer.write(passage)
            counts = Counter(tokenize(_document_text(passage, include_title)))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                pending[0].append(vocabulary.setdefault(term, len(vocabulary)))
                pending[1].append(number)
                pending[2].append(min(frequency, MAX_FREQUENCY))
            if (number + 1) % chunk_documents == 0:
                flush()
    finally:
        writer.close()
    flush()
    write_starts(staging)
    documents = len(lengths)
    if documents == 0:
        shutil.rmtree(staging)
        raise ValueError("there are no passages to index")
    terms = np.concatenate([chunk[0] for chunk in chunks]) if chunks else np.zeros(0, dtype=np.int32)
    docs = np.concatenate([chunk[1] for chunk in chunks]) if chunks else np.zeros(0, dtype=np.int32)
    frequencies = np.concatenate([chunk[2] for chunk in chunks]) if chunks else np.zeros(0, dtype=np.uint16)
    chunks.clear()
    order = np.argsort(terms, kind="stable")
    docs, frequencies = docs[order], frequencies[order]
    document_frequency = np.bincount(terms, minlength=len(vocabulary)).astype(np.int64)
    del terms, order
    offsets = np.concatenate([[0], np.cumsum(document_frequency)]).astype(np.int64)
    lengths_array = np.array(lengths, dtype=np.float64)
    average = float(lengths_array.mean())
    idf = np.log1p((documents - document_frequency + 0.5) / (document_frequency + 0.5))
    impacts = np.zeros(len(docs), dtype=np.float32)
    owner = np.repeat(np.arange(len(vocabulary), dtype=np.int32), document_frequency)
    for start in range(0, len(docs), BLOCK_POSTINGS):
        block = slice(start, start + BLOCK_POSTINGS)
        frequency = frequencies[block].astype(np.float64)
        norm = k1 * (1 - b + b * lengths_array[docs[block]] / average)
        impacts[block] = idf[owner[block]] * frequency * (k1 + 1) / (frequency + norm)
    np.save(staging / "offsets.npy", offsets)
    np.save(staging / "doc_ids.npy", docs)
    np.save(staging / "impacts.npy", impacts)
    (staging / "terms.txt").write_text("\n".join(vocabulary), encoding="utf-8")
    meta = {
        "format": FORMAT,
        "documents": documents,
        "terms": len(vocabulary),
        "postings": int(len(docs)),
        "k1": k1,
        "b": b,
        "include_title": include_title,
        "average_length": average,
    }
    (staging / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(staging, directory)
    return documents


class BM25Index:
    def __init__(self, directory: Path, name: str = "bm25") -> None:
        self.directory = Path(directory)
        self.name = name
        self.meta = json.loads((self.directory / "meta.json").read_text(encoding="utf-8"))
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{directory} is not a format {FORMAT} index")
        self.documents = int(self.meta["documents"])
        text = (self.directory / "terms.txt").read_text(encoding="utf-8")
        self._terms = {term: number for number, term in enumerate(text.split("\n")) if term}
        self._offsets = np.load(self.directory / "offsets.npy", mmap_mode="r")
        self._doc_ids = np.load(self.directory / "doc_ids.npy", mmap_mode="r")
        self._impacts = np.load(self.directory / "impacts.npy", mmap_mode="r")
        self.store = PassageStore(self.directory)

    def passage(self, document: int) -> Passage:
        return self.store.passage(document)

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.documents, dtype=np.float32)
        numbers = sorted({self._terms[token] for token in tokenize(query) if token in self._terms})
        for number in numbers:
            start, stop = int(self._offsets[number]), int(self._offsets[number + 1])
            scores[self._doc_ids[start:stop]] += self._impacts[start:stop]
        return scores

    def search(self, query: str, k: int) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        scores = self.scores(query)
        candidates = np.flatnonzero(scores)
        values = scores[candidates]
        if len(candidates) > k:
            cutoff = np.partition(values, len(values) - k)[len(values) - k]
            above = candidates[values > cutoff]
            tied = candidates[values == cutoff]
            candidates = np.concatenate([above, tied[: k - len(above)]])
        ranked = candidates[np.lexsort((candidates, -scores[candidates]))]
        return [Hit(self.passage(int(document)), float(scores[document])) for document in ranked]

