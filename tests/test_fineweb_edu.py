import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scout.data.fineweb_edu import REPO, iter_records, main

REVISION = "d" * 40


def parquet_bytes(rows, row_group_size=2):
    table = pa.table(
        {
            "text": [r["text"] for r in rows],
            "id": [r["id"] for r in rows],
            "dump": ["CC-MAIN-2024-10"] * len(rows),
            "url": [r["url"] for r in rows],
            "language": ["en"] * len(rows),
            "int_score": [r["score"] for r in rows],
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer, row_group_size=row_group_size)
    return buffer.getvalue()


ROWS = [
    {"text": "Photosynthesis turns light into chemical energy.", "id": "a", "url": "https://a.example", "score": 4},
    {"text": "Unicode stays raw: éè \u2014 中文.", "id": "b", "url": None, "score": 3},
    {"text": "Third row.", "id": "c", "url": "https://c.example", "score": 5},
]


def test_reads_records_across_row_groups(tmp_path):
    path = tmp_path / "f.parquet"
    path.write_bytes(parquet_bytes(ROWS))
    expected = [{"text": r["text"], "url": r["url"] or "", "id": r["id"], "score": r["score"]} for r in ROWS]
    assert list(iter_records(path, batch_size=2)) == expected


def test_rejects_files_without_the_expected_columns(tmp_path):
    path = tmp_path / "f.parquet"
    pq.write_table(pa.table({"text": ["x"]}), path)
    with pytest.raises(ValueError):
        list(iter_records(path))


def test_rejects_rows_without_text(tmp_path):
    path = tmp_path / "f.parquet"
    rows = [dict(ROWS[0]), dict(ROWS[1], text=None)]
    path.write_bytes(parquet_bytes(rows))
    records = iter_records(path)
    assert next(records)["id"] == "a"
    with pytest.raises(ValueError):
        next(records)


def hub(serve, bodies):
    server = serve({f"api/datasets/{REPO}": json.dumps({"sha": REVISION}).encode()})
    entries = []
    for name, body in bodies.items():
        path = f"sample/100BT/{name}"
        entries.append({"type": "file", "path": path, "size": len(body), "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)}})
        server.files[f"datasets/{REPO}/resolve/{REVISION}/{path}"] = body
    server.files[f"api/datasets/{REPO}/tree/{REVISION}/sample/100BT"] = json.dumps(entries).encode()
    return server


def test_command_line_downloads_verified_files_at_a_pinned_revision(serve, tmp_path):
    bodies = {f"{i:03d}_00000.parquet": parquet_bytes([dict(ROWS[0], id=str(i))]) for i in range(4)}
    server = hub(serve, bodies)
    out = tmp_path / "fineweb"
    assert main(["--out", str(out), "--files", "2", "--seed", "3", "--base-url", server.url]) == 0
    selection = json.loads((out / "selection.json").read_text())
    assert selection["revision"] == REVISION
    assert len(selection["files"]) == 2
    for file in selection["files"]:
        local = out / file["path"]
        assert hashlib.sha256(local.read_bytes()).hexdigest() == file["sha256"]
        assert len(list(iter_records(local))) == 1
    downloads = [path for path, _ in server.requests if "/resolve/" in path]
    assert len(downloads) == 2 and all(f"/resolve/{REVISION}/" in path for path in downloads)


def test_corrupt_download_is_rejected(serve, tmp_path):
    body = parquet_bytes(ROWS)
    server = hub(serve, {"000_00000.parquet": body})
    server.files[f"datasets/{REPO}/resolve/{REVISION}/sample/100BT/000_00000.parquet"] = body[:-1] + b"\x00"
    with pytest.raises(OSError):
        main(["--out", str(tmp_path), "--base-url", server.url, "--retries", "1", "--retry-wait", "0"])
    assert not (tmp_path / "sample/100BT/000_00000.parquet").exists()


def test_explicit_revision_skips_the_lookup_and_dry_run_downloads_nothing(serve, tmp_path):
    server = hub(serve, {"000_00000.parquet": parquet_bytes(ROWS)})
    assert main(["--out", str(tmp_path), "--revision", REVISION, "--dry-run", "--base-url", server.url]) == 0
    assert [path for path, _ in server.requests] == [f"/api/datasets/{REPO}/tree/{REVISION}/sample/100BT"]
    assert json.loads((tmp_path / "selection.json").read_text())["files"][0]["path"] == "sample/100BT/000_00000.parquet"
