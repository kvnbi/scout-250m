from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

from scout.distributed import rank

RESERVED = ("kind", "step", "time")


def _clean(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def read_log(path: Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records = []
    for number, line in enumerate(lines, start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines):
                raise ValueError(f"{path} line {number} is not valid JSON") from None
    return records


class RunLog:
    def __init__(self, path: Path, resume_step: int | None = None, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume_step is not None and self.path.exists():
            self._truncate(resume_step)
        self._file = open(self.path, "a", encoding="utf-8")

    def _truncate(self, step: int) -> None:
        text = self.path.read_text(encoding="utf-8")
        kept = [record for record in read_log(self.path) if record["step"] <= step]
        rewritten = "".join(json.dumps(record, allow_nan=False) + "\n" for record in kept)
        if rewritten == text:
            return
        partial = self.path.with_name(self.path.name + ".tmp")
        partial.write_text(rewritten, encoding="utf-8")
        os.replace(partial, self.path)

    def write(self, kind: str, step: int, values: dict[str, object] | None = None) -> None:
        values = values or {}
        if set(values) & set(RESERVED):
            raise ValueError(f"{', '.join(RESERVED)} are reserved keys")
        record = {"kind": kind, "step": step, "time": self.clock(), **_clean(values)}
        self._file.write(json.dumps(record, allow_nan=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def step_logger(log: RunLog, every: int = 1) -> Callable[[int, dict], None]:
    if every < 1:
        raise ValueError("every must be positive")

    def callback(step: int, metrics: dict) -> None:
        if rank() == 0 and step % every == 0:
            log.write("train", step, metrics)

    return callback


def evaluation_logger(log: RunLog) -> Callable[[int, dict], None]:
    return lambda step, results: log.write("eval", step, results)
