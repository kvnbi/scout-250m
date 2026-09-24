import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq

from scout.data.finepdfs import DEFAULT_FOLDER, REPO, iter_records, main

REVISION = "e" * 40

ROWS = [
    {"text": "Page one.\n\nPage two of a report.", "id": "a", "url": "https://a.example/report.pdf", "tokens": 12, "extractor": "docling", "refetched": False},
    {"text": "A scanned manual, recovered by OCR.", "id": "b", "url": None, "tokens": 9000, "extractor": "rolmOCR", "refetched": True},
]


def parquet_bytes(rows):
    table = pa.table(
        {
            "text": [r["text"] for r in rows],
            "id": [r["id"] for r in rows],
            "dump": ["CC-MAIN-2024-10"] * len(rows),
            "url": [r["url"] for r in rows],
            "token_count": pa.array([r["tokens"] for r in rows], pa.int64()),
            "extractor": pa.array([r["extractor"] for r in rows]).dictionary_encode(),
            "is_truncated": [r["refetched"] for r in rows],
            "page_ends": [[5, len(r["text"])] for r in rows],
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_targets_english_finepdfs_edu():
    assert REPO == "HuggingFaceFW/finepdfs-edu"
    assert DEFAULT_FOLDER == "data/eng_Latn/train"


def test_reads_records_with_extraction_details(tmp_path):
    path = tmp_path / "f.parquet"
    path.write_bytes(parquet_bytes(ROWS))
    assert list(iter_records(path)) == [
        {
            "text": r["text"],
            "url": r["url"] or "",
            "id": r["id"],
            "tokens": r["tokens"],
            "extractor": r["extractor"],
            "refetched": r["refetched"],
        }
        for r in ROWS
    ]


def test_command_line_downloads_a_verified_file(serve, tmp_path):
    body = parquet_bytes(ROWS)
    path = f"{DEFAULT_FOLDER}/000_00000.parquet"
    listing = [{"type": "file", "path": path, "size": len(body), "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)}}]
    server = serve(
        {
            f"api/datasets/{REPO}": json.dumps({"sha": REVISION}).encode(),
            f"api/datasets/{REPO}/tree/{REVISION}/{DEFAULT_FOLDER}": json.dumps(listing).encode(),
            f"datasets/{REPO}/resolve/{REVISION}/{path}": body,
        }
    )
    assert main(["--out", str(tmp_path), "--base-url", server.url]) == 0
    assert json.loads((tmp_path / "selection.json").read_text())["repo"] == REPO
    assert [r["id"] for r in iter_records(tmp_path / path)] == ["a", "b"]
