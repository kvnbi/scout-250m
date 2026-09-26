import gzip
import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from compression import zstd

from scout.tokenizer.sample import (
    CHUNK_CHARACTERS,
    MANIFEST_FILE,
    SAMPLE_FILE,
    SOURCES,
    Source,
    build_sample,
    chunk_text,
    iter_sample,
    main,
    selected_files,
)

TRICKY = "Line one\nline two\r\nNEL\x85LS\u2028tab\tend \u2014 raw \U0001f600 \u00e9\u4e2d."


def read_json_lines(path):
    with open(path, encoding="utf-8", newline="\n") as lines:
        for line in lines:
            yield json.loads(line)


def hub_folder(root, name, files):
    folder = root / name
    entries = []
    for path, records in files.items():
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        entries.append({"path": path, "size": 1, "sha256": "0" * 64})
    selection = {"repo": f"org/{name}", "revision": "a" * 40, "folder": "", "files": entries}
    (folder / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    return folder


def records(prefix, count, size):
    return [{"id": f"{prefix}{i}", "text": f"{prefix}{i:05d} ".ljust(size, "x")} for i in range(count)]


FAKE = (Source("big", 0.75, "big", read_json_lines), Source("small", 0.25, "small", read_json_lines))


@pytest.fixture
def fake_data(tmp_path):
    data = tmp_path / "data"
    hub_folder(data, "big", {"b/0.jsonl": records("b", 10000, 50), "b/1.jsonl": records("c", 10000, 50)})
    hub_folder(data, "small", {"s.jsonl": records("s", 20000, 25)})
    return data


def test_default_mix_covers_every_source_and_adds_up():
    shares = {source.name: source.share for source in SOURCES}
    assert shares == {
        "nemotron": 0.20,
        "nemotron_synthetic": 0.10,
        "fineweb_edu": 0.20,
        "finepdfs": 0.15,
        "wikipedia": 0.10,
        "pes2o": 0.07,
        "stack_v2": 0.07,
        "stackexchange": 0.06,
        "finemath": 0.05,
    }
    assert math.isclose(sum(shares.values()), 1.0)
    assert {source.folder for source in SOURCES} == {
        "nemotron",
        "fineweb_edu",
        "finepdfs",
        "wikipedia",
        "pes2o",
        "stack_v2",
        "stackexchange",
        "finemath",
    }
    assert [source.name for source in SOURCES if source.synthetic] == ["nemotron_synthetic"]


def test_samples_each_source_in_its_share(fake_data, tmp_path):
    manifest = build_sample(fake_data, tmp_path / "out", 200_000, sources=FAKE)
    for name, target in (("big", 150_000), ("small", 50_000)):
        report = manifest["sources"][name]
        assert report["target_bytes"] == target
        assert abs(report["sampled_bytes"] - target) < 0.1 * target
    assert manifest["sources"]["big"]["available_bytes"] == 1_000_000
    assert manifest["sources"]["small"]["keep_probability"] == pytest.approx(0.1)
    kept = list(iter_sample(tmp_path / "out" / SAMPLE_FILE))
    assert {record["id"][0] for record in kept if record["source"] == "big"} == {"b", "c"}


def test_manifest_matches_the_sample(fake_data, tmp_path):
    manifest = build_sample(fake_data, tmp_path / "out", 200_000, seed=3, sources=FAKE)
    kept = list(iter_sample(tmp_path / "out" / SAMPLE_FILE))
    for name, report in manifest["sources"].items():
        texts = [record["text"] for record in kept if record["source"] == name]
        assert report["chunks"] == len(texts)
        assert report["sampled_bytes"] == sum(len(text.encode()) for text in texts)
        assert report["origin"] == {"repo": f"org/{name}", "revision": "a" * 40}
    assert manifest["sources"]["big"]["files"] == ["b/0.jsonl", "b/1.jsonl"]
    assert (manifest["seed"], manifest["total_bytes"], manifest["chunk_characters"]) == (3, 200_000, CHUNK_CHARACTERS)
    assert json.loads((tmp_path / "out" / MANIFEST_FILE).read_text()) == manifest
    assert not (tmp_path / "out" / (SAMPLE_FILE + ".part")).exists()


def test_sampling_is_seeded_and_reproducible(fake_data, tmp_path):
    outputs = []
    for run, seed in enumerate((1, 1, 2)):
        out = tmp_path / f"run{run}"
        manifest = build_sample(fake_data, out, 200_000, seed=seed, sources=FAKE)
        outputs.append(((out / SAMPLE_FILE).read_bytes(), manifest["sources"]))
    assert outputs[0] == outputs[1]
    assert outputs[0][0] != outputs[2][0]


def test_text_survives_exactly(tmp_path):
    data = tmp_path / "data"
    long = (TRICKY + "\n") * 1000 + "no line break " * 2000
    texts = [TRICKY, "", "plain", "\x00 control and \ufeff marks", long]
    hub_folder(data, "all", {"a.jsonl": [{"id": str(i), "text": t} for i, t in enumerate(texts)]})
    total = sum(len(text.encode()) for text in texts)
    assert total > sum(len(text) for text in texts)
    manifest = build_sample(data, tmp_path / "out", total, sources=(Source("all", 1.0, "all", read_json_lines),))
    rebuilt = {}
    for record in iter_sample(tmp_path / "out" / SAMPLE_FILE):
        assert record["part"] == len(rebuilt.setdefault(record["id"], []))
        rebuilt[record["id"]].append(record["text"])
    assert {key: "".join(parts) for key, parts in rebuilt.items()} == {str(i): t for i, t in enumerate(texts) if t}
    assert len(rebuilt["4"]) > 1
    assert manifest["sources"]["all"]["sampled_bytes"] == manifest["sources"]["all"]["available_bytes"] == total


@pytest.mark.parametrize("text", ["", "short", "a\nb\n" * 5000, "x" * 20000, "line\r\n" * 3000 + "tail"])
def test_chunks_rebuild_the_text_and_cut_after_line_breaks(text):
    pieces = list(chunk_text(text, 100))
    assert "".join(pieces) == text
    assert all(0 < len(piece) <= 100 for piece in pieces)
    assert all(piece.endswith("\n") for piece in pieces[:-1] if "\n" in piece)


def test_chunks_fill_up_to_the_last_line_break():
    assert {len(piece) for piece in list(chunk_text("ab\n" * 1000, 100))[:-1]} == {99}
    assert list(chunk_text("x" * CHUNK_CHARACTERS)) == ["x" * CHUNK_CHARACTERS]
    assert [len(piece) for piece in chunk_text("x" * (CHUNK_CHARACTERS + 1))] == [CHUNK_CHARACTERS, 1]
    assert [len(piece) for piece in chunk_text("\n" + "x" * 150, 100)] == [100, 51]


def test_one_huge_document_is_sampled_in_parts(tmp_path):
    data = tmp_path / "data"
    huge = "".join(f"line {i} ".ljust(99, "h") + "\n" for i in range(100_000))
    hub_folder(data, "one", {"a.jsonl": [{"id": "huge", "text": huge}]})
    manifest = build_sample(data, tmp_path / "out", 1_000_000, sources=(Source("one", 1.0, "one", read_json_lines),))
    pieces = list(chunk_text(huge))
    kept = list(iter_sample(tmp_path / "out" / SAMPLE_FILE))
    assert 700_000 < manifest["sources"]["one"]["sampled_bytes"] < 1_300_000
    assert all(record["text"] == pieces[record["part"]] for record in kept)
    assert len({record["part"] for record in kept}) == len(kept) == manifest["sources"]["one"]["chunks"]


def test_rejects_sources_without_enough_text(fake_data, tmp_path):
    with pytest.raises(ValueError, match="small has 500,000 of 1,000,000 bytes"):
        build_sample(fake_data, tmp_path / "out", 4_000_000, sources=FAKE)
    assert not (tmp_path / "out").exists()


def test_rejects_missing_selections_and_downloads(fake_data, tmp_path):
    (fake_data / "small" / "s.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="not downloaded yet"):
        build_sample(fake_data, tmp_path / "out", 1000, sources=FAKE)
    (fake_data / "small" / "selection.json").unlink()
    with pytest.raises(FileNotFoundError, match="selection.json is missing"):
        build_sample(fake_data, tmp_path / "out", 1000, sources=FAKE)


@pytest.mark.parametrize(
    "sources, total",
    [
        ((Source("big", 0.5, "big", read_json_lines), Source("small", 0.25, "small", read_json_lines)), 1000),
        ((Source("big", 0.5, "big", read_json_lines), Source("big", 0.5, "small", read_json_lines)), 1000),
        (FAKE, 0),
    ],
)
def test_rejects_bad_mixes(fake_data, tmp_path, sources, total):
    with pytest.raises(ValueError):
        build_sample(fake_data, tmp_path / "out", total, sources=sources)


NEMOTRON_PREFIX = "contrib/Nemotron/Nemotron-CC/data-jsonl/"
NEMOTRON = NEMOTRON_PREFIX + "quality=high/kind={kind}/kind2={kind2}/CC-MAIN-2020-05-part-00000.jsonl.zstd"


def nemotron_folder(data, count):
    folder = data / "nemotron"
    paths = []
    for kind, kind2 in (("actual", "actual"), ("synthetic", "distill"), ("synthetic", "wrap_medium")):
        path = NEMOTRON.format(kind=kind, kind2=kind2)
        paths.append(path)
        lines = [{"text": f"{kind2} {i} ".ljust(60, "n"), "warc_record_id": f"{kind2}-{i}"} for i in range(count)]
        local = folder / path.removeprefix(NEMOTRON_PREFIX)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(zstd.compress("".join(json.dumps(line) + "\n" for line in lines).encode()))
    selection = {"partitions": "high/actual", "seed": 0, "manifest_sha256": "f" * 64, "files": paths}
    (folder / "selection.json").write_text(json.dumps(selection), encoding="utf-8")


def write_hub_file(data, name, path, body):
    local = data / name / path
    local.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(body, pa.Table):
        pq.write_table(body, local)
    else:
        local.write_bytes(gzip.compress("".join(json.dumps(line) + "\n" for line in body).encode()))
    entry = {"path": path, "size": 1, "sha256": "0" * 64}
    selection = {"repo": f"org/{name}", "revision": "b" * 40, "folder": "", "files": [entry]}
    (data / name / "selection.json").write_text(json.dumps(selection), encoding="utf-8")


def real_formats(data, count):
    nemotron_folder(data, count)
    texts = [f"row {i} ".ljust(60, "t") for i in range(count)]
    ids = [str(i) for i in range(count)]
    fineweb = pa.table({"text": texts, "id": ids, "url": ids, "int_score": [3] * count})
    write_hub_file(data, "fineweb_edu", "sample/100BT/000_00000.parquet", fineweb)
    finepdfs = {"text": texts, "id": ids, "url": ids, "token_count": [9] * count}
    finepdfs |= {"extractor": ["docling"] * count, "is_truncated": [False] * count}
    write_hub_file(data, "finepdfs", "data/eng_Latn/train/000_00000.parquet", pa.table(finepdfs))
    lead = [{"type": "section", "name": "Abstract", "has_parts": [{"type": "paragraph", "value": t}]} for t in texts]
    articles = {"name": ids, "identifier": list(range(count)), "url": ids}
    articles |= {"sections": [json.dumps([section]) for section in lead], "references": [[]] * count}
    write_hub_file(data, "wikipedia", "enwiki/data/enwiki_namespace_0_0.parquet", pa.table(articles))
    metadata = {"license": "CC0", "site": "physics.stackexchange.com"}
    common = [{"id": i, "text": t, "metadata": dict(metadata, url=i)} for i, t in zip(ids, texts)]
    write_hub_file(data, "pes2o", "peS2o-0000.json.gz", common)
    russian = [{"id": "r", "text": "\u041a\u0430\u043a?" * 20, "metadata": {"site": "ru.stackoverflow.com"}}]
    write_hub_file(data, "stackexchange", "stackexchange-dolma-0000.json.gz", russian + common)
    write_hub_file(data, "stack_v2", "stack-edu-0000.json.gz", [dict(line, int_score=3) for line in common])
    maths = {"text": texts, "url": ids, "warc_filename": ids, "warc_record_offset": list(range(count))}
    maths["int_score"] = [4] * count
    write_hub_file(data, "finemath", "finemath-4plus/train-00000-of-00064.parquet", pa.table(maths))


def test_selected_files_follow_each_fetcher_layout(tmp_path):
    real_formats(tmp_path, 3)
    selection, paths = selected_files(tmp_path / "nemotron", synthetic=True)
    assert selection["manifest_sha256"] == "f" * 64
    assert [path.parent.name for path in paths] == ["kind2=distill", "kind2=wrap_medium"]
    _, paths = selected_files(tmp_path / "nemotron")
    assert [path.parent.name for path in paths] == ["kind2=actual"]
    _, paths = selected_files(tmp_path / "wikipedia")
    assert paths == [tmp_path / "wikipedia" / "enwiki/data/enwiki_namespace_0_0.parquet"]


def test_rejects_a_nemotron_selection_without_synthetic_files(tmp_path):
    real_formats(tmp_path, 3)
    listing = tmp_path / "nemotron" / "selection.json"
    selection = json.loads(listing.read_text())
    selection["files"] = selection["files"][:1]
    listing.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="no files selected for nemotron_synthetic"):
        build_sample(tmp_path, tmp_path / "out", 1000)


def test_command_line_samples_every_real_source_format(tmp_path):
    data = tmp_path / "data"
    real_formats(data, 300)
    assert main(["--data", str(data), "--out", str(tmp_path / "out"), "--bytes", "30000", "--seed", "5"]) == 0
    manifest = json.loads((tmp_path / "out" / MANIFEST_FILE).read_text())
    assert list(manifest["sources"]) == [source.name for source in SOURCES]
    kept = list(iter_sample(tmp_path / "out" / SAMPLE_FILE))
    by_source = {}
    for record in kept:
        by_source.setdefault(record["source"], []).append(record)
    assert set(by_source) == {source.name for source in SOURCES}
    assert all(record["id"].startswith("actual-") for record in by_source["nemotron"])
    assert all(not record["id"].startswith("actual-") for record in by_source["nemotron_synthetic"])
    assert all(record["text"].startswith("# ") for record in by_source["wikipedia"])
    assert all(record["id"].startswith("physics.stackexchange.com/") for record in by_source["stackexchange"])
