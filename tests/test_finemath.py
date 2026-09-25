import hashlib
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq

from scout.data.finemath import DEFAULT_FOLDER, REPO, iter_records, main

REVISION = "9" * 40
WARC = "crawl-data/CC-MAIN-2019-04/segments/1547583656530.8/warc/CC-MAIN-20190115225438-20190116011438-00022.warc.gz"

ROWS = [
    {"text": "Solve $x^2 = 4$.\n\nSo $x = \\pm 2$.", "url": "https://maths.example/q/1", "offset": 1043, "score": 4},
    {"text": "Proof by induction, step by step.", "url": None, "offset": 2_000_000_000, "score": 5},
]


def parquet_bytes(rows):
    table = pa.table(
        {
            "url": [r["url"] for r in rows],
            "fetch_time": pa.array([1547600000] * len(rows), pa.int64()),
            "content_mime_type": ["text/html"] * len(rows),
            "warc_filename": [WARC] * len(rows),
            "warc_record_offset": pa.array([r["offset"] for r in rows], pa.int32()),
            "warc_record_length": pa.array([5000] * len(rows), pa.int32()),
            "text": [r["text"] for r in rows],
            "token_count": pa.array([12] * len(rows), pa.int32()),
            "char_count": pa.array([len(r["text"]) for r in rows], pa.int32()),
            "metadata": ['{"extraction_info": {}}'] * len(rows),
            "score": pa.array([r["score"] + 0.25 for r in rows], pa.float64()),
            "int_score": pa.array([r["score"] for r in rows], pa.int64()),
            "crawl": ["CC-MAIN-2019-04"] * len(rows),
            "snapshot_type": ["latest", "largest"],
            "language": ["en"] * len(rows),
            "language_score": pa.array([0.93] * len(rows), pa.float64()),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


RECORDS = [
    {"text": r["text"], "url": r["url"] or "", "id": f"{WARC}:{r['offset']}", "score": r["score"]} for r in ROWS
]


def test_targets_finemath_4plus():
    assert REPO == "HuggingFaceTB/finemath"
    assert DEFAULT_FOLDER == "finemath-4plus"


def test_reads_records_identified_by_their_warc_record(tmp_path):
    path = tmp_path / "f.parquet"
    path.write_bytes(parquet_bytes(ROWS))
    assert list(iter_records(path)) == RECORDS


def test_command_line_downloads_a_verified_file(serve, tmp_path):
    body = parquet_bytes(ROWS)
    path = f"{DEFAULT_FOLDER}/train-00000-of-00064.parquet"
    listing = [{"type": "file", "path": path, "size": len(body), "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)}}]
    server = serve(
        {
            f"api/datasets/{REPO}": json.dumps({"sha": REVISION}).encode(),
            f"api/datasets/{REPO}/tree/{REVISION}/{DEFAULT_FOLDER}": json.dumps(listing).encode(),
            f"datasets/{REPO}/resolve/{REVISION}/{path}": body,
        }
    )
    assert main(["--out", str(tmp_path), "--base-url", server.url]) == 0
    assert list(iter_records(tmp_path / path)) == RECORDS
