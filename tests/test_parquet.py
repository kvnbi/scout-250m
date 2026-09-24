import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scout.data.parquet import iter_parquet


def write(path, columns, row_group_size=2):
    pq.write_table(pa.table(columns), path, row_group_size=row_group_size)
    return path


def test_yields_only_the_requested_columns_across_row_groups(tmp_path):
    path = write(tmp_path / "f.parquet", {"text": ["a", "b", "c"], "id": ["1", "2", "3"], "extra": [1, 2, 3]})
    assert list(iter_parquet(path, ("text", "id"), batch_size=2)) == [
        {"text": "a", "id": "1"},
        {"text": "b", "id": "2"},
        {"text": "c", "id": "3"},
    ]


def test_requires_the_text_column_to_be_requested(tmp_path):
    path = write(tmp_path / "f.parquet", {"text": ["a"], "id": ["1"]})
    with pytest.raises(ValueError):
        list(iter_parquet(path, ("id",)))


def test_rejects_missing_columns(tmp_path):
    path = write(tmp_path / "f.parquet", {"text": ["a"]})
    with pytest.raises(ValueError):
        list(iter_parquet(path, ("text", "id")))


def test_text_column_can_be_renamed_or_unchecked(tmp_path):
    path = write(tmp_path / "f.parquet", {"body": ["a", None], "id": ["1", "2"]})
    rows = iter_parquet(path, ("body", "id"), text_column="body")
    assert next(rows)["body"] == "a"
    with pytest.raises(ValueError):
        next(rows)
    assert [row["id"] for row in iter_parquet(path, ("body", "id"), text_column=None)] == ["1", "2"]
    with pytest.raises(ValueError):
        list(iter_parquet(path, ("id",), text_column="body"))


def test_rejects_rows_without_text(tmp_path):
    path = write(tmp_path / "f.parquet", {"text": ["a", None], "id": ["1", "2"]})
    rows = iter_parquet(path, ("text", "id"))
    assert next(rows)["id"] == "1"
    with pytest.raises(ValueError):
        next(rows)
