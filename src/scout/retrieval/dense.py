from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from scout.retrieval.embedder import Embedder
from scout.retrieval.passages import Passage, encode_passage, iter_passages
from scout.retrieval.search import Hit
from scout.retrieval.store import PassageStore, PassageWriter, write_starts

FORMAT = 1
CHUNK_PASSAGES = 2048
SEARCH_ROWS = 65536
DTYPES = {"float16": np.float16, "float32": np.float32}
VECTORS = "vectors.bin"
PROGRESS = "progress.json"
META = "meta.json"


class Encoder(Protocol):
    dimension: int
    max_length: int

    def encode(self, texts: Sequence[str], kind: str = "passage", batch_size: int = 32) -> np.ndarray: ...


def passage_text(passage: Passage) -> str:
    return f'"{passage.title}"\n{passage.text}'


def _write_json(path: Path, value: dict) -> None:
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def build_dense_index(
    passages: Iterable[Passage],
    directory: Path,
    embedder: Encoder,
    batch_size: int = 64,
    chunk_passages: int = CHUNK_PASSAGES,
    dtype: str = "float16",
    model: str = "",
    progress: Callable[[int], None] | None = None,
) -> int:
    if dtype not in DTYPES or batch_size < 1 or chunk_passages < 1:
        raise ValueError(f"need dtype in {sorted(DTYPES)}, batch_size >= 1 and chunk_passages >= 1")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    staging = directory.with_name(directory.name + ".part")
    settings = {"dimension": embedder.dimension, "dtype": dtype, "model": model, "max_length": embedder.max_length}
    done, passage_bytes, saved_digest = 0, 0, ""
    if (staging / PROGRESS).exists():
        saved = json.loads((staging / PROGRESS).read_text(encoding="utf-8"))
        if {key: saved.get(key) for key in settings} != settings:
            raise ValueError(f"{staging} was started with other settings, delete it to start again")
        done, passage_bytes, saved_digest = saved["passages"], saved["passage_bytes"], saved["digest"]
    else:
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
    itemsize = np.dtype(DTYPES[dtype]).itemsize
    hasher = hashlib.sha256()
    source = iter(passages)
    for _ in range(done):
        passage = next(source, None)
        if passage is None:
            raise ValueError(f"the passages ended before the {done} already embedded in {staging}")
        hasher.update(encode_passage(passage).encode("utf-8"))
    if done and hasher.hexdigest() != saved_digest:
        raise ValueError(f"{staging} was built from different passages, delete it to start again")
    vector_bytes = done * embedder.dimension * itemsize
    if done and (staging / VECTORS).stat().st_size < vector_bytes:
        raise ValueError(f"{staging / VECTORS} is shorter than its progress record")
    writer = PassageWriter(staging, passage_bytes)
    buffer: list[Passage] = []

    def commit() -> None:
        nonlocal done, vector_bytes
        if not buffer:
            return
        vectors = embedder.encode([passage_text(p) for p in buffer], "passage", batch_size)
        if vectors.shape != (len(buffer), embedder.dimension) or not np.isfinite(vectors).all():
            raise ValueError("the embedder returned vectors of the wrong shape or with bad values")
        if (staging / VECTORS).exists():
            os.truncate(staging / VECTORS, vector_bytes)
        with open(staging / VECTORS, "ab") as stream:
            stream.write(np.ascontiguousarray(vectors, dtype=DTYPES[dtype]).tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        for passage in buffer:
            writer.write(passage)
            hasher.update(encode_passage(passage).encode("utf-8"))
        writer.flush()
        done += len(buffer)
        vector_bytes += len(buffer) * embedder.dimension * itemsize
        buffer.clear()
        _write_json(
            staging / PROGRESS,
            {**settings, "passages": done, "passage_bytes": writer.position, "digest": hasher.hexdigest()},
        )
        if progress is not None:
            progress(done)

    try:
        for passage in source:
            buffer.append(passage)
            if len(buffer) == chunk_passages:
                commit()
        commit()
    finally:
        writer.close()
    passages_bytes = writer.position
    if done == 0:
        shutil.rmtree(staging)
        raise ValueError("there are no passages to index")
    if write_starts(staging) != done:
        raise ValueError("the passage store and the vectors disagree about the number of passages")
    description = {"text": "quoted title, newline, text", "prefix": "passage: "}
    meta = {"format": FORMAT, "passages": done, **settings, **description, "passages_bytes": passages_bytes}
    _write_json(staging / META, meta)
    (staging / PROGRESS).unlink()
    os.replace(staging, directory)
    return done


class DenseIndex:
    def __init__(
        self,
        directory: Path,
        embedder: Encoder,
        name: str = "dense",
        device: str | torch.device = "cpu",
        resident: bool = False,
        rows: int = SEARCH_ROWS,
    ) -> None:
        if rows < 1:
            raise ValueError("rows must be positive")
        self.directory = Path(directory)
        self.name = name
        self.embedder = embedder
        self.device = torch.device(device)
        self.rows = rows
        self.meta = json.loads((self.directory / META).read_text(encoding="utf-8"))
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{directory} is not a format {FORMAT} index")
        if embedder.dimension != self.meta["dimension"]:
            raise ValueError(f"embedder has {embedder.dimension} dimensions, index {self.meta['dimension']}")
        self.passages = int(self.meta["passages"])
        self.store = PassageStore(self.directory)
        self.store.verify(self.passages, self.meta.get("passages_bytes"))
        shape = (self.passages, self.meta["dimension"])
        expected = shape[0] * shape[1] * np.dtype(DTYPES[self.meta["dtype"]]).itemsize
        actual = (self.directory / VECTORS).stat().st_size
        if actual != expected:
            raise ValueError(f"{self.directory / VECTORS} is {actual} bytes, expected {expected}, may be incomplete")
        self._vectors = np.memmap(self.directory / VECTORS, dtype=DTYPES[self.meta["dtype"]], mode="r", shape=shape)
        self._resident = torch.from_numpy(np.array(self._vectors)).to(self.device) if resident else None

    def passage(self, document: int) -> Passage:
        return self.store.passage(document)

    def search_vectors(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k < 1:
            raise ValueError("k must be positive")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.meta["dimension"]:
            raise ValueError(f"queries must have shape (n, {self.meta['dimension']})")
        wanted = torch.from_numpy(queries).to(self.device)
        best_scores = best_ids = None
        for start in range(0, self.passages, self.rows):
            stop = min(start + self.rows, self.passages)
            if self._resident is not None:
                block = self._resident[start:stop]
            else:
                block = torch.from_numpy(np.array(self._vectors[start:stop])).to(self.device)
            top = (wanted @ block.float().T).topk(min(k, stop - start), dim=1)
            scores, ids = top.values, top.indices + start
            if best_scores is not None:
                scores, ids = torch.cat([best_scores, scores], dim=1), torch.cat([best_ids, ids], dim=1)
                keep = scores.topk(min(k, scores.shape[1]), dim=1).indices
                scores, ids = scores.gather(1, keep), ids.gather(1, keep)
            best_scores, best_ids = scores, ids
        scores_np, ids_np = best_scores.cpu().numpy(), best_ids.cpu().numpy()
        order = np.stack([np.lexsort((ids_np[row], -scores_np[row])) for row in range(len(scores_np))])
        return np.take_along_axis(scores_np, order, axis=1), np.take_along_axis(ids_np, order, axis=1)

    def search(self, query: str, k: int) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        if not query.strip():
            return []
        scores, ids = self.search_vectors(self.embedder.encode([query], "query"), k)
        return [Hit(self.passage(int(i)), float(s)) for s, i in zip(scores[0], ids[0])]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def already_embedded(directory: Path) -> int:
    record = Path(directory).with_name(Path(directory).name + ".part") / PROGRESS
    return int(json.loads(record.read_text(encoding="utf-8"))["passages"]) if record.exists() else 0


def count_lines(path: Path) -> int:
    count = 0
    with open(path, "rb") as stream:
        while block := stream.read(1 << 24):
            count += block.count(b"\n")
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed passages with an E5 model into a dense index.")
    parser.add_argument("passages", type=Path, help="a passages.jsonl file")
    parser.add_argument("--model", type=Path, required=True, help="folder holding the downloaded model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cpu, mps or cuda (default: fastest available)")
    parser.add_argument("--embed-dtype", choices=("float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16", help="how vectors are stored")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-passages", type=int, default=CHUNK_PASSAGES)
    args = parser.parse_args(argv)
    device = args.device or default_device()
    embed_dtype = args.embed_dtype or ("float32" if device == "cpu" else "float16")
    embedder = Embedder(args.model, device, getattr(torch, embed_dtype), args.max_length)
    total, resumed = count_lines(args.passages), already_embedded(args.out)
    started = time.perf_counter()

    def report(done: int) -> None:
        rate = (done - resumed) / (time.perf_counter() - started)
        left = (total - done) / rate / 60
        print(f"{done}/{total} passages, {rate:.0f} per second, {left:.1f} minutes left", flush=True)

    count = build_dense_index(
        iter_passages(args.passages),
        args.out,
        embedder,
        args.batch_size,
        args.chunk_passages,
        args.dtype,
        f"{args.model.name} {embed_dtype}",
        report,
    )
    print(f"{count} passages -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
