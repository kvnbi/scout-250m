import gzip
import hashlib
import io
import json

import pytest
from compression import zstd

from scout.data.nemotron import (
    DEFAULT_PARTITIONS,
    MANIFEST_PATH,
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
