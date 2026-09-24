from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq


def iter_parquet(path: Path, columns: Sequence[str], batch_size: int = 1024) -> Iterator[dict[str, object]]:
    if "text" not in columns:
        raise ValueError("columns must include text")
    parquet = pq.ParquetFile(path)
    missing = [column for column in columns if column not in parquet.schema_arrow.names]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        for row in batch.to_pylist():
            if not isinstance(row["text"], str):
                raise ValueError(f"{path} has a row without text: {row.get('id')!r}")
            yield row
