from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from scout.retrieval.passages import Passage, decode_passage, encode_passage

PASSAGES = "passages.jsonl"
STARTS = "starts.npy"
SCAN_BYTES = 64 << 20


class PassageWriter:
    def __init__(self, directory: Path, resume_bytes: int = 0) -> None:
        self.path = Path(directory) / PASSAGES
        if resume_bytes:
            os.truncate(self.path, resume_bytes)
        self.file = open(self.path, "ab" if resume_bytes else "wb")
        self.position = resume_bytes

    def write(self, passage: Passage) -> None:
        data = encode_passage(passage).encode("utf-8")
        self.file.write(data)
        self.position += len(data)

    def flush(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self) -> None:
        self.file.close()


def write_starts(directory: Path) -> int:
    path = Path(directory) / PASSAGES
    starts: list[np.ndarray] = []
    offset = 0
    with open(path, "rb") as stream:
        while block := stream.read(SCAN_BYTES):
            ends = np.flatnonzero(np.frombuffer(block, dtype=np.uint8) == 10) + offset + 1
            starts.append(ends)
            offset += len(block)
    ends_all = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
    if offset and (len(ends_all) == 0 or ends_all[-1] != offset):
        raise ValueError(f"{path} does not end with a newline")
    begins = np.concatenate([[0], ends_all[:-1]]).astype(np.int64) if len(ends_all) else np.zeros(0, dtype=np.int64)
    np.save(Path(directory) / STARTS, begins)
    return len(begins)


def load_array(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r")
    except (ValueError, EOFError) as error:
        raise ValueError(f"{path} is truncated or damaged, the copy may be incomplete") from error


class PassageStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._starts = load_array(self.directory / STARTS)

    def verify(self, count: int, size: int | None = None) -> None:
        path = self.directory / PASSAGES
        actual = path.stat().st_size
        if len(self) != count:
            raise ValueError(f"{self.directory / STARTS} lists {len(self)} passages, expected {count}")
        if size is not None and actual != size:
            raise ValueError(f"{path} is {actual} bytes, expected {size}, the copy may be incomplete")
        if count and int(self._starts[-1]) >= actual:
            raise ValueError(f"{path} ends before its last passage, the copy may be incomplete")
        with open(path, "rb") as stream:
            stream.seek(max(actual - 1, 0))
            if actual and stream.read(1) != b"\n":
                raise ValueError(f"{path} does not end with a newline, the copy may be incomplete")

    def __len__(self) -> int:
        return len(self._starts)

    def passage(self, document: int) -> Passage:
        if not 0 <= document < len(self):
            raise IndexError(document)
        with open(self.directory / PASSAGES, "rb") as stream:
            stream.seek(int(self._starts[document]))
            return decode_passage(stream.readline().decode("utf-8"))
