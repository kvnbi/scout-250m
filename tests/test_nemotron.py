import gzip
import hashlib
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from compression import zstd

from scout.data.nemotron import (
    DEFAULT_PARTITIONS,
    MANIFEST_PATH,
    download,
    download_with_retries,
    iter_records,
    main,
    parse_manifest,
    parse_partitions,
    parse_path,
    select_files,
)

ACTUAL = "contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high/kind=actual/kind2=actual/CC-MAIN-2013-20-part-00003.jsonl.zstd"
SYNTHETIC = (
    "contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high/kind=synthetic/kind2=diverse_qa_pairs/"
    "CC-MAIN-2024-10-part-00042.jsonl.zstd"
)


def test_parses_an_actual_path():
    file = parse_path(ACTUAL + "\n")
    assert (file.quality, file.kind, file.kind2, file.crawl, file.part) == ("high", "actual", "actual", "CC-MAIN-2013-20", 3)
    assert not file.synthetic
    assert file.partition == ("high", "actual")
    assert str(file.relative_path) == "quality=high/kind=actual/kind2=actual/CC-MAIN-2013-20-part-00003.jsonl.zstd"


def test_parses_a_synthetic_path():
    file = parse_path(SYNTHETIC)
    assert file.synthetic
    assert file.partition == ("high", "diverse_qa_pairs")
    assert file.part == 42


@pytest.mark.parametrize(
    "path",
    [
        ACTUAL.replace("quality=high", "quality=best"),
        ACTUAL.replace("kind2=actual", "kind2=distill"),
        SYNTHETIC.replace("kind2=diverse_qa_pairs", "kind2=actual"),
        ACTUAL.replace(".jsonl.zstd", ".jsonl"),
        "extra/" + ACTUAL,
        "",
    ],
)
def test_rejects_malformed_paths(path):
    with pytest.raises(ValueError):
        parse_path(path)


def test_manifest_skips_blank_lines():
    assert [f.path for f in parse_manifest([ACTUAL, "", "  \n", SYNTHETIC])] == [ACTUAL, SYNTHETIC]


def manifest(count=6):
    paths = []
    for quality, kind2 in (("high", "actual"), ("high", "distill"), ("medium", "actual")):
        kind = "actual" if kind2 == "actual" else "synthetic"
        for part in range(count):
            paths.append(
                f"contrib/Nemotron/Nemotron-CC/data-jsonl/quality={quality}/kind={kind}/kind2={kind2}/"
                f"CC-MAIN-2020-05-part-{part:05d}.jsonl.zstd"
            )
    return parse_manifest(paths)


def test_selects_whole_partitions_in_a_stable_order():
    chosen = select_files(manifest(), [("high", "distill"), ("high", "actual")])
    assert [f.partition for f in chosen] == [("high", "actual")] * 6 + [("high", "distill")] * 6
    assert chosen == sorted(chosen, key=lambda f: (f.partition, f.path))


def test_keeps_real_and_synthetic_apart():
    chosen = select_files(manifest(), [("high", "actual")])
    assert all(not f.synthetic for f in chosen)
    chosen = select_files(manifest(), [("high", "distill")])
    assert all(f.synthetic for f in chosen)


def test_sampling_is_seeded_and_per_partition():
    first = select_files(manifest(20), [("high", "actual"), ("medium", "actual")], per_partition=3, seed=1)
    again = select_files(manifest(20), [("high", "actual"), ("medium", "actual")], per_partition=3, seed=1)
    other = select_files(manifest(20), [("high", "actual"), ("medium", "actual")], per_partition=3, seed=2)
    assert first == again
    assert first != other
    assert [f.partition for f in first] == [("high", "actual")] * 3 + [("medium", "actual")] * 3


def test_sampling_more_than_available_takes_everything():
    assert len(select_files(manifest(4), [("high", "actual")], per_partition=10)) == 4


def test_selection_rejects_missing_partitions_and_bad_counts():
    with pytest.raises(ValueError):
        select_files(manifest(), [("low", "wrap_medium")])
    with pytest.raises(ValueError):
        select_files(manifest(), [("high", "actual")], per_partition=0)


def test_default_partitions_are_the_high_quality_bucket():
    assert all(quality == "high" for quality, _ in DEFAULT_PARTITIONS)
    assert {kind2 for _, kind2 in DEFAULT_PARTITIONS} == {
        "actual",
        "distill",
        "diverse_qa_pairs",
        "extract_knowledge",
        "knowledge_list",
        "wrap_medium",
    }


def test_parses_partition_lists():
    assert parse_partitions("high/actual, medium-high/actual") == [("high", "actual"), ("medium-high", "actual")]
    for bad in ("high", "best/actual", "high/"):
        with pytest.raises(ValueError):
            parse_partitions(bad)


def encode(records, frames=1):
    lines = [json.dumps(record, ensure_ascii=False) + "\n" for record in records]
    size = -(-len(lines) // frames)
    return b"".join(zstd.compress("".join(lines[i : i + size]).encode()) for i in range(0, len(lines), size))


RECORDS = [
    {"text": "First document.\nWith two lines.", "language": "eng", "warc_record_id": "a", "url": "https://a.example"},
    {"text": "Zweites Dokument mit Umlauten: äöü, and an em dash \u2014 kept raw.", "language": "eng", "warc_record_id": "b", "url": "https://b.example"},
    {"text": "Third.", "language": "eng", "warc_record_id": "c", "url": "https://c.example"},
]


@pytest.mark.parametrize("frames", [1, 3])
def test_streams_records_from_zstd(frames):
    records = list(iter_records(io.BytesIO(encode(RECORDS, frames))))
    assert records == [{"text": r["text"], "url": r["url"], "id": r["warc_record_id"]} for r in RECORDS]


def test_unicode_line_separators_stay_inside_records():
    text = "one\u2028two\u2029three\x85four"
    blob = encode([{"text": text, "language": "eng", "warc_record_id": "a", "url": None}])
    assert list(iter_records(io.BytesIO(blob))) == [{"text": text, "url": "", "id": "a"}]


def test_skips_blank_lines_and_rejects_records_without_text():
    blob = zstd.compress(b'{"text": "ok"}\n\n{"url": "x"}\n')
    stream = iter_records(io.BytesIO(blob))
    assert next(stream)["text"] == "ok"
    with pytest.raises(ValueError):
        next(stream)


class Server:
    def __init__(self, files, honour_ranges=True, truncate_to=None, cut_first=0):
        self.files = files
        self.requests = []
        self.cuts_left = cut_first
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Range")))
                body = outer.files.get(self.path.lstrip("/"))
                if body is None:
                    self.send_error(404)
                    return
                header = self.headers.get("Range")
                if header and honour_ranges:
                    start = int(header.removeprefix("bytes=").rstrip("-"))
                    if start >= len(body):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(body)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    part = body[start:]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    part = body
                    self.send_response(200)
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                limit = truncate_to
                if outer.cuts_left and len(part) > 1:
                    outer.cuts_left -= 1
                    limit = len(part) // 2
                self.wfile.write(part[:limit] if limit is not None else part)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    servers = []

    def start(files, **kwargs):
        servers.append(Server(files, **kwargs))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


BODY = bytes(range(256)) * 400


def test_downloads_a_file(serve, tmp_path):
    server = serve({"f.bin": BODY})
    target = download(server.url + "f.bin", tmp_path / "sub" / "f.bin")
    assert target.read_bytes() == BODY
    assert not (tmp_path / "sub" / "f.bin.part").exists()


def test_resumes_a_partial_download(serve, tmp_path):
    server = serve({"f.bin": BODY})
    (tmp_path / "f.bin.part").write_bytes(BODY[:1000])
    assert download(server.url + "f.bin", tmp_path / "f.bin").read_bytes() == BODY
    assert server.requests == [("/f.bin", "bytes=1000-")]


def test_restarts_when_the_server_ignores_ranges(serve, tmp_path):
    server = serve({"f.bin": BODY}, honour_ranges=False)
    (tmp_path / "f.bin.part").write_bytes(b"stale bytes that must be discarded")
    assert download(server.url + "f.bin", tmp_path / "f.bin").read_bytes() == BODY


def test_truncated_transfer_keeps_the_partial_for_resuming(serve, tmp_path):
    server = serve({"f.bin": BODY}, truncate_to=5000)
    with pytest.raises(OSError):
        download(server.url + "f.bin", tmp_path / "f.bin")
    assert not (tmp_path / "f.bin").exists()
    assert (tmp_path / "f.bin.part").read_bytes() == BODY[:5000]
    server.close()
    healthy = serve({"f.bin": BODY})
    assert download(healthy.url + "f.bin", tmp_path / "f.bin").read_bytes() == BODY


def test_connection_cut_mid_copy_is_a_resumable_error(serve, tmp_path, monkeypatch):
    import http.client

    import scout.data.nemotron as nemotron

    server = serve({"f.bin": BODY})

    def cut(source, target, length):
        target.write(source.read(3000))
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(nemotron.shutil, "copyfileobj", cut)
    with pytest.raises(OSError):
        download(server.url + "f.bin", tmp_path / "f.bin")
    monkeypatch.undo()
    assert (tmp_path / "f.bin.part").read_bytes() == BODY[:3000]
    assert download(server.url + "f.bin", tmp_path / "f.bin").read_bytes() == BODY


def test_complete_partial_is_finished_without_downloading(serve, tmp_path):
    server = serve({"f.bin": BODY})
    (tmp_path / "f.bin.part").write_bytes(BODY)
    assert download(server.url + "f.bin", tmp_path / "f.bin").read_bytes() == BODY
    assert not (tmp_path / "f.bin.part").exists()


def test_oversized_partial_is_discarded(serve, tmp_path):
    server = serve({"f.bin": BODY})
    (tmp_path / "f.bin.part").write_bytes(BODY + b"extra")
    with pytest.raises(OSError):
        download(server.url + "f.bin", tmp_path / "f.bin")
    assert not (tmp_path / "f.bin.part").exists()
    assert download(server.url + "f.bin", tmp_path / "f.bin").read_bytes() == BODY


def test_retries_resume_after_transient_cuts(serve, tmp_path):
    server = serve({"f.bin": BODY}, cut_first=2)
    assert download_with_retries(server.url + "f.bin", tmp_path / "f.bin", retries=3, wait=0.0).read_bytes() == BODY
    assert [header is not None for _, header in server.requests] == [False, True, True]


def test_retries_give_up_after_the_limit(serve, tmp_path):
    server = serve({"f.bin": BODY}, cut_first=5)
    with pytest.raises(OSError):
        download_with_retries(server.url + "f.bin", tmp_path / "f.bin", retries=2, wait=0.0)
    assert len(server.requests) == 3


def test_missing_files_are_not_retried(serve, tmp_path):
    server = serve({})
    with pytest.raises(OSError):
        download_with_retries(server.url + "gone.bin", tmp_path / "gone.bin", retries=3, wait=0.0)
    assert len(server.requests) == 1


def test_existing_file_is_not_downloaded_again(serve, tmp_path):
    server = serve({"f.bin": BODY})
    (tmp_path / "f.bin").write_bytes(b"done")
    assert download(server.url + "f.bin", tmp_path / "f.bin").read_bytes() == b"done"
    assert server.requests == []


def test_command_line_downloads_a_selection(serve, tmp_path):
    files = {}
    paths = []
    for kind2 in ("actual", "distill"):
        kind = "actual" if kind2 == "actual" else "synthetic"
        for part in range(3):
            path = f"contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high/kind={kind}/kind2={kind2}/CC-MAIN-2020-05-part-{part:05d}.jsonl.zstd"
            paths.append(path)
            files[path] = encode([{"text": f"{kind2} {part}", "language": "eng", "warc_record_id": path, "url": ""}])
    files[MANIFEST_PATH] = gzip.compress(("\n".join(paths) + "\n").encode())
    server = serve(files)
    out = tmp_path / "nemotron"
    args = ["--out", str(out), "--partitions", "high/actual,high/distill", "--files-per-partition", "2", "--seed", "4"]
    assert main(args + ["--base-url", server.url]) == 0
    selection = json.loads((out / "selection.json").read_text())
    assert len(selection["files"]) == 4
    assert selection["manifest_sha256"] == hashlib.sha256(files[MANIFEST_PATH]).hexdigest()
    for path in selection["files"]:
        local = out / path.removeprefix("contrib/Nemotron/Nemotron-CC/data-jsonl/")
        with open(local, "rb") as stream:
            assert [r["id"] for r in iter_records(stream)] == [path]
    first_run = [p for p, _ in server.requests]
    assert main(args + ["--base-url", server.url]) == 0
    assert [p for p, _ in server.requests] == first_run


def test_dry_run_selects_without_downloading(serve, tmp_path):
    files = {MANIFEST_PATH: gzip.compress((ACTUAL + "\n").encode())}
    server = serve(files)
    assert main(["--out", str(tmp_path), "--partitions", "high/actual", "--dry-run", "--base-url", server.url]) == 0
    assert json.loads((tmp_path / "selection.json").read_text())["files"] == [ACTUAL]
    assert [p for p, _ in server.requests] == ["/" + MANIFEST_PATH]
