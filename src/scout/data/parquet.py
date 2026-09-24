from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pyarrow.parquet as pq


def iter_parquet(
    path: Path, columns: Sequence[str], batch_size: int = 1024, text_column: str | None = "text"
) -> Iterator[dict[str, object]]:
    if text_column is not None and text_column not in columns:
        raise ValueError(f"columns must include {text_column}")
    parquet = pq.ParquetFile(path)
    missing = [column for column in columns if column not in parquet.schema_arrow.names]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(columns)):
        for row in batch.to_pylist():
            if text_column is not None and not isinstance(row[text_column], str):
                raise ValueError(f"{path} has a row without {text_column}: {row.get('id')!r}")
            yield row
