import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from bert_files import write_model_directory

from scout.retrieval import dense as dense_module
from scout.retrieval.dense import (
    PROGRESS,
    DenseIndex,
    already_embedded,
    build_dense_index,
    count_lines,
    default_device,
    main,
    passage_text,
)
from scout.retrieval.embedder import Embedder
from scout.retrieval.passages import Passage, write_passages
from scout.retrieval.search import SearchBackend, check_backend


class Fake:
    max_length = 256

    def __init__(self, dimension=16, fail_after=None):
        self.dimension = dimension
        self.calls = []
        self.fail_after = fail_after

    def encode(self, texts, kind="passage", batch_size=32):
        self.calls.append((kind, list(texts), batch_size))
        if self.fail_after is not None and len([c for c in self.calls if c[0] == "passage"]) > self.fail_after:
            raise RuntimeError("embedder crashed")
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            vector = np.random.default_rng(seed).standard_normal(self.dimension).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows) if rows else np.zeros((0, self.dimension), dtype=np.float32)


def corpus(count=40):
    return [Passage(f"{i}:0", f"Title {i}", "", f"body text number {i}", "https://example.org/p", "2018-12-20") for i in range(count)]


def unit_rows(rng, count, dimension):
    rows = rng.standard_normal((count, dimension)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def brute_force(index, queries, k):
    matrix = np.asarray(index._vectors, dtype=np.float64)
    scores = np.asarray(queries, dtype=np.float64) @ matrix.T
    order = [np.lexsort((np.arange(scores.shape[1]), -row))[:k] for row in scores]
    return np.stack([row[o] for row, o in zip(scores, order)]), np.stack(order)


def test_building_stores_vectors_passages_and_a_description(tmp_path):
    items = corpus()
    assert build_dense_index(items, tmp_path / "idx", Fake(), dtype="float32", model="fake@1") == 40
    meta = json.loads((tmp_path / "idx" / "meta.json").read_text())
    assert meta == {
        "format": 1, "passages": 40, "dimension": 16, "dtype": "float32", "model": "fake@1", "max_length": 256,
        "text": "quoted title, newline, text", "prefix": "passage: ",
        "passages_bytes": (tmp_path / "idx" / "passages.jsonl").stat().st_size,
    }
    assert (tmp_path / "idx" / "vectors.bin").stat().st_size == 40 * 16 * 4
    assert sorted(f.name for f in (tmp_path / "idx").iterdir()) == ["meta.json", "passages.jsonl", "starts.npy", "vectors.bin"]
    index = DenseIndex(tmp_path / "idx", Fake())
    assert [index.passage(i) for i in range(40)] == items


def test_passages_are_embedded_as_quoted_title_newline_text_and_queries_as_queries(tmp_path):
    items = corpus(5)
    assert passage_text(items[2]) == '"Title 2"\nbody text number 2'
    embedder = Fake()
    build_dense_index(items, tmp_path / "idx", embedder, chunk_passages=2, batch_size=3)
    assert [call[0] for call in embedder.calls] == ["passage"] * 3
    assert [text for call in embedder.calls for text in call[1]] == [passage_text(p) for p in items]
    assert {call[2] for call in embedder.calls} == {3}
    index = DenseIndex(tmp_path / "idx", embedder)
    index.search("who is title 3", 2)
    assert embedder.calls[-1][:2] == ("query", ["who is title 3"])


@pytest.mark.parametrize("k", [1, 5, 50, 400])
@pytest.mark.parametrize("rows", [1, 7, 64, 100000])
def test_search_equals_a_brute_force_reference_whatever_the_chunking(tmp_path, k, rows):
    rng = np.random.default_rng(0)
    items = corpus(300)
    build_dense_index(items, tmp_path / "idx", Fake(32), dtype="float32", chunk_passages=64)
    index = DenseIndex(tmp_path / "idx", Fake(32), rows=rows)
    queries = unit_rows(rng, 6, 32)
    scores, ids = index.search_vectors(queries, k)
    expected_scores, expected_ids = brute_force(index, queries, min(k, 300))
    assert ids.shape == (6, min(k, 300))
    np.testing.assert_allclose(scores, expected_scores, atol=1e-5)
    exact = np.asarray(index._vectors, dtype=np.float64) @ queries.astype(np.float64).T
    for row in range(6):
        assert len(set(ids[row])) == len(ids[row])
        for column in np.flatnonzero(ids[row] != expected_ids[row]):
            assert abs(exact[ids[row][column], row] - exact[expected_ids[row][column], row]) < 1e-6


def test_a_resident_matrix_and_the_mapped_one_agree(tmp_path):
    build_dense_index(corpus(120), tmp_path / "idx", Fake(32), dtype="float16", chunk_passages=50)
    queries = unit_rows(np.random.default_rng(1), 4, 32)
    mapped = DenseIndex(tmp_path / "idx", Fake(32), rows=40).search_vectors(queries, 10)
    resident = DenseIndex(tmp_path / "idx", Fake(32), resident=True, rows=40).search_vectors(queries, 10)
    assert np.array_equal(mapped[1], resident[1]) and np.array_equal(mapped[0], resident[0])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Mac GPU")
def test_the_gpu_gives_the_same_ranking(tmp_path):
    build_dense_index(corpus(200), tmp_path / "idx", Fake(32), dtype="float16")
    queries = unit_rows(np.random.default_rng(2), 5, 32)
    cpu = DenseIndex(tmp_path / "idx", Fake(32)).search_vectors(queries, 10)
    gpu = DenseIndex(tmp_path / "idx", Fake(32), device="mps", resident=True).search_vectors(queries, 10)
    np.testing.assert_allclose(gpu[0], cpu[0], atol=1e-5)
    assert np.array_equal(gpu[1], cpu[1])


def test_half_precision_storage_halves_the_file_and_barely_changes_scores(tmp_path):
    items = corpus(200)
    build_dense_index(items, tmp_path / "full", Fake(64), dtype="float32")
    build_dense_index(items, tmp_path / "half", Fake(64), dtype="float16")
    assert (tmp_path / "half" / "vectors.bin").stat().st_size * 2 == (tmp_path / "full" / "vectors.bin").stat().st_size
    queries = unit_rows(np.random.default_rng(3), 5, 64)
    full = DenseIndex(tmp_path / "full", Fake(64)).search_vectors(queries, 10)
    half = DenseIndex(tmp_path / "half", Fake(64)).search_vectors(queries, 10)
    np.testing.assert_allclose(half[0], full[0], atol=2e-3)


def test_the_hits_carry_the_matching_passage_and_its_score(tmp_path):
    items = corpus(30)
    embedder = Fake()
    build_dense_index(items, tmp_path / "idx", embedder, dtype="float32")
    index = DenseIndex(tmp_path / "idx", embedder, name="e5-2018")
    assert index.name == "e5-2018" and isinstance(index, SearchBackend)
    text = passage_text(items[11])
    vector = embedder.encode([text])[0]
    scores, ids = index.search_vectors(vector[None], 3)
    assert ids[0][0] == 11 and scores[0][0] == pytest.approx(1.0, abs=1e-5)
    hits = index.search(text, 3)
    assert hits[0].passage == items[11] and len(hits) == 3


def test_blank_queries_find_nothing_and_k_must_be_positive_and_large_k_is_capped(tmp_path):
    build_dense_index(corpus(8), tmp_path / "idx", Fake())
    index = DenseIndex(tmp_path / "idx", Fake())
    assert index.search("", 3) == [] and index.search("   \n", 3) == []
    for k in (0, -1):
        with pytest.raises(ValueError):
            index.search("x", k)
        with pytest.raises(ValueError):
            index.search_vectors(np.ones((1, 16), dtype=np.float32), k)
    assert len(index.search("x", 100)) == 8
    check_backend(index, ["a query", "another"], k=5)


def test_identical_passages_tie_and_come_out_in_document_order(tmp_path):
    items = [Passage(f"{i}:0", "Same", "", "same text", "u", "d") for i in range(12)]
    build_dense_index(items, tmp_path / "idx", Fake(), dtype="float32")
    index = DenseIndex(tmp_path / "idx", Fake())
    for rows in (1, 5, 100):
        index.rows = rows
        assert [h.passage.id for h in index.search("anything", 12)] == [p.id for p in items]


def test_query_shapes_are_checked(tmp_path):
    build_dense_index(corpus(5), tmp_path / "idx", Fake())
    index = DenseIndex(tmp_path / "idx", Fake())
    for shape in ((16,), (1, 15), (1, 17), (2, 3, 16)):
        with pytest.raises(ValueError, match="shape"):
            index.search_vectors(np.ones(shape, dtype=np.float32), 1)


def test_the_embedder_must_match_the_index_and_the_format_must_be_known(tmp_path):
    build_dense_index(corpus(5), tmp_path / "idx", Fake(16))
    with pytest.raises(ValueError, match="dimensions"):
        DenseIndex(tmp_path / "idx", Fake(8))
    with pytest.raises(ValueError, match="rows"):
        DenseIndex(tmp_path / "idx", Fake(16), rows=0)
    meta = json.loads((tmp_path / "idx" / "meta.json").read_text())
    (tmp_path / "idx" / "meta.json").write_text(json.dumps({**meta, "format": 9}))
    with pytest.raises(ValueError, match="format 1"):
        DenseIndex(tmp_path / "idx", Fake(16))


def test_progress_is_reported_after_each_committed_chunk(tmp_path):
    seen = []
    build_dense_index(corpus(10), tmp_path / "idx", Fake(), chunk_passages=4, progress=seen.append)
    assert seen == [4, 8, 10]


def test_a_corpus_that_fills_whole_chunks_is_not_committed_again_at_the_end(tmp_path):
    seen, embedder = [], Fake()
    build_dense_index(corpus(8), tmp_path / "idx", embedder, chunk_passages=4, progress=seen.append)
    assert seen == [4, 8] and len(embedder.calls) == 2


def test_the_vector_and_passage_counts_are_cross_checked(tmp_path, monkeypatch):
    import scout.retrieval.dense as dense

    monkeypatch.setattr(dense, "write_starts", lambda directory: 6)
    with pytest.raises(ValueError, match="disagree"):
        build_dense_index(corpus(5), tmp_path / "idx", Fake())
    assert not (tmp_path / "idx").exists()


def test_many_identical_passages_rank_by_document_order(tmp_path):
    items = [Passage(f"{i}:0", "Same", "", "same text", "u", "d") for i in range(150)]
    build_dense_index(items, tmp_path / "idx", Fake(), dtype="float32")
    for rows in (1, 16, 64, 1000):
        index = DenseIndex(tmp_path / "idx", Fake(), rows=rows)
        assert [h.passage.id for h in index.search("anything", 150)] == [p.id for p in items]


def snapshot(directory):
    return {f.name: f.read_bytes() for f in directory.iterdir()}


def test_an_interrupted_build_resumes_and_matches_an_uninterrupted_one(tmp_path):
    items = corpus(23)
    build_dense_index(items, tmp_path / "straight", Fake(), chunk_passages=5)
    crashing = Fake(fail_after=2)
    with pytest.raises(RuntimeError, match="crashed"):
        build_dense_index(items, tmp_path / "idx", crashing, chunk_passages=5)
    assert not (tmp_path / "idx").exists() and (tmp_path / "idx.part" / PROGRESS).exists()
    assert json.loads((tmp_path / "idx.part" / PROGRESS).read_text())["passages"] == 10
    resumed = Fake()
    build_dense_index(items, tmp_path / "idx", resumed, chunk_passages=5)
    assert [t for c in resumed.calls for t in c[1]] == [passage_text(p) for p in items[10:]]
    assert snapshot(tmp_path / "idx") == snapshot(tmp_path / "straight")
    assert not (tmp_path / "idx.part").exists()


def test_bytes_written_after_the_last_record_are_discarded_on_resume(tmp_path):
    items = corpus(14)
    build_dense_index(items, tmp_path / "straight", Fake(), chunk_passages=4)
    with pytest.raises(RuntimeError):
        build_dense_index(items, tmp_path / "idx", Fake(fail_after=2), chunk_passages=4)
    with open(tmp_path / "idx.part" / "vectors.bin", "ab") as stream:
        stream.write(b"\x01" * 37)
    with open(tmp_path / "idx.part" / "passages.jsonl", "ab") as stream:
        stream.write(b'{"torn": "li')
    build_dense_index(items, tmp_path / "idx", Fake(), chunk_passages=4)
    assert snapshot(tmp_path / "idx") == snapshot(tmp_path / "straight")


def crashed_build(tmp_path, items, **kwargs):
    with pytest.raises(RuntimeError):
        build_dense_index(items, tmp_path / "idx", Fake(fail_after=1), chunk_passages=4, **kwargs)


def test_resuming_with_different_passages_or_settings_is_refused(tmp_path):
    items = corpus(14)
    crashed_build(tmp_path, items, model="a")
    changed = [Passage(p.id, p.title, p.section, p.text + "!", p.url, p.date) if i == 2 else p for i, p in enumerate(items)]
    with pytest.raises(ValueError, match="different passages"):
        build_dense_index(changed, tmp_path / "idx", Fake(), chunk_passages=4, model="a")
    with pytest.raises(ValueError, match="other settings"):
        build_dense_index(items, tmp_path / "idx", Fake(), chunk_passages=4, model="b")
    with pytest.raises(ValueError, match="other settings"):
        build_dense_index(items, tmp_path / "idx", Fake(), chunk_passages=4, model="a", dtype="float32")
    with pytest.raises(ValueError, match="other settings"):
        build_dense_index(items, tmp_path / "idx", Fake(8), chunk_passages=4, model="a")
    with pytest.raises(ValueError, match="ended"):
        build_dense_index(items[:3], tmp_path / "idx", Fake(), chunk_passages=4, model="a")
    assert (tmp_path / "idx.part" / PROGRESS).exists()
    build_dense_index(items, tmp_path / "idx", Fake(), chunk_passages=4, model="a")
    assert DenseIndex(tmp_path / "idx", Fake()).passages == 14


def test_a_vector_file_shorter_than_its_record_is_refused(tmp_path):
    items = corpus(14)
    crashed_build(tmp_path, items)
    with open(tmp_path / "idx.part" / "vectors.bin", "r+b") as stream:
        stream.truncate(10)
    with pytest.raises(ValueError, match="shorter"):
        build_dense_index(items, tmp_path / "idx", Fake(), chunk_passages=4)


def test_a_leftover_folder_without_a_record_is_cleared(tmp_path):
    (tmp_path / "idx.part").mkdir()
    (tmp_path / "idx.part" / "junk").write_text("x")
    build_dense_index(corpus(5), tmp_path / "idx", Fake())
    assert sorted(f.name for f in (tmp_path / "idx").iterdir()) == ["meta.json", "passages.jsonl", "starts.npy", "vectors.bin"]


def test_refusals_leave_nothing_behind(tmp_path):
    build_dense_index(corpus(3), tmp_path / "idx", Fake())
    with pytest.raises(FileExistsError):
        build_dense_index(corpus(3), tmp_path / "idx", Fake())
    with pytest.raises(ValueError, match="no passages"):
        build_dense_index([], tmp_path / "empty", Fake())
    assert not (tmp_path / "empty").exists() and not (tmp_path / "empty.part").exists()
    for kwargs in (dict(dtype="float64"), dict(batch_size=0), dict(chunk_passages=0)):
        with pytest.raises(ValueError):
            build_dense_index(corpus(3), tmp_path / "bad", Fake(), **kwargs)
    assert not (tmp_path / "bad").exists() and not (tmp_path / "bad.part").exists()


@pytest.mark.parametrize("bad", ["wrong shape", "nan", "inf"])
def test_an_embedder_returning_bad_vectors_is_caught_before_anything_is_stored(tmp_path, bad):
    class Bad(Fake):
        def encode(self, texts, kind="passage", batch_size=32):
            vectors = super().encode(texts, kind, batch_size)
            if bad == "wrong shape":
                return vectors[:, :8]
            vectors[0, 0] = np.nan if bad == "nan" else np.inf
            return vectors

    with pytest.raises(ValueError, match="wrong shape or with bad values"):
        build_dense_index(corpus(4), tmp_path / "idx", Bad())
    assert not (tmp_path / "idx").exists()
    assert not (tmp_path / "idx.part" / "vectors.bin").exists() or (tmp_path / "idx.part" / "vectors.bin").stat().st_size == 0


def test_a_source_that_fails_midway_keeps_the_committed_chunks_for_a_later_resume(tmp_path):
    items = corpus(12)

    def source():
        yield from items[:9]
        raise OSError("disk went away")

    with pytest.raises(OSError):
        build_dense_index(source(), tmp_path / "idx", Fake(), chunk_passages=4)
    assert json.loads((tmp_path / "idx.part" / PROGRESS).read_text())["passages"] == 8
    build_dense_index(items, tmp_path / "idx", Fake(), chunk_passages=4)
    assert [DenseIndex(tmp_path / "idx", Fake()).passage(i) for i in range(12)] == items


def test_the_real_embedder_class_works_as_the_encoder(tmp_path):
    write_model_directory(tmp_path / "model")
    embedder = Embedder(tmp_path / "model", max_length=12)
    build_dense_index(corpus(10), tmp_path / "idx", embedder, model="tiny")
    index = DenseIndex(tmp_path / "idx", embedder)
    assert json.loads((tmp_path / "idx" / "meta.json").read_text())["max_length"] == 12
    check_backend(index, ["the cat sat", "dog ran far away"], k=4)


def test_lines_are_counted_across_block_boundaries(tmp_path, monkeypatch):
    write_passages(tmp_path / "p.jsonl", corpus(37))
    assert count_lines(tmp_path / "p.jsonl") == 37
    (tmp_path / "empty.jsonl").write_bytes(b"")
    assert count_lines(tmp_path / "empty.jsonl") == 0


def test_the_default_device_is_one_torch_can_use():
    assert default_device() in {"cpu", "mps", "cuda"}
    assert torch.zeros(1, device=default_device()).device.type == default_device()


def test_command_line_builds_an_index_from_a_passages_file(tmp_path, capsys):
    write_model_directory(tmp_path / "model")
    write_passages(tmp_path / "p.jsonl", corpus(10))
    out = tmp_path / "idx"
    arguments = [str(tmp_path / "p.jsonl"), "--model", str(tmp_path / "model"), "--out", str(out)]
    assert main(arguments + ["--device", "cpu", "--max-length", "12", "--chunk-passages", "4", "--batch-size", "3"]) == 0
    text = capsys.readouterr().out
    assert "4/10 passages" in text and "10/10 passages" in text and "10 passages ->" in text
    meta = json.loads((out / "meta.json").read_text())
    assert (meta["passages"], meta["dtype"], meta["max_length"], meta["model"]) == (10, "float16", 12, "model float32")
    embedder = Embedder(tmp_path / "model", max_length=12)
    check_backend(DenseIndex(out, embedder), ["the cat sat", "hello world"], k=3)


@pytest.mark.parametrize(
    "cuda, mps, expected", [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_the_default_device_prefers_cuda_then_the_mac_gpu_then_the_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert default_device() == expected


class Recorder(Embedder):
    made = []

    def __init__(self, directory, device="cpu", dtype=torch.float32, max_length=512):
        super().__init__(directory, "cpu", dtype, min(max_length, 12))
        Recorder.made.append((str(device), dtype, max_length))
        self.batches = []

    def encode(self, texts, kind="passage", batch_size=32):
        Recorder.batches_seen.append(batch_size)
        return super().encode(texts, kind, batch_size)

    batches_seen = []


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    Recorder.made, Recorder.batches_seen = [], []
    monkeypatch.setattr(dense_module, "Embedder", Recorder)
    write_model_directory(tmp_path / "model")
    write_passages(tmp_path / "p.jsonl", corpus(3))
    return [str(tmp_path / "p.jsonl"), "--model", str(tmp_path / "model")]


def test_command_line_defaults(recorder, tmp_path, monkeypatch):
    monkeypatch.setattr(dense_module, "default_device", lambda: "cpu")
    main(recorder + ["--out", str(tmp_path / "cpu")])
    monkeypatch.setattr(dense_module, "default_device", lambda: "mps")
    main(recorder + ["--out", str(tmp_path / "gpu")])
    assert Recorder.made == [("cpu", torch.float32, 256), ("mps", torch.float16, 256)]
    assert Recorder.batches_seen == [64, 64]
    assert json.loads((tmp_path / "gpu" / "meta.json").read_text())["dtype"] == "float16"
    assert json.loads((tmp_path / "cpu" / "meta.json").read_text())["model"] == "model float32"
    assert json.loads((tmp_path / "gpu" / "meta.json").read_text())["model"] == "model float16"


def test_command_line_choices_reach_the_embedder(recorder, tmp_path):
    main(recorder + ["--out", str(tmp_path / "a"), "--device", "cuda", "--embed-dtype", "float32", "--max-length", "9", "--batch-size", "5"])
    assert Recorder.made == [("cuda", torch.float32, 9)] and Recorder.batches_seen == [5]


def test_progress_lines_show_the_rate_and_the_time_left(recorder, tmp_path, capsys, monkeypatch):
    ticks = iter([100.0, 108.0, 116.0])
    monkeypatch.setattr(dense_module, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))
    main(recorder + ["--out", str(tmp_path / "a"), "--chunk-passages", "2", "--device", "cpu"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "2/3 passages, 0 per second, 0.1 minutes left"
    assert lines[1] == "3/3 passages, 0 per second, 0.0 minutes left"


def test_a_resumed_run_reports_the_rate_of_the_new_work_only(recorder, tmp_path, capsys, monkeypatch):
    out = str(tmp_path / "idx")
    original = Recorder.encode
    state = {"calls": 0}

    def flaky(self, texts, kind="passage", batch_size=32):
        state["calls"] += 1
        if state["calls"] == 2:
            raise RuntimeError("laptop went to sleep")
        return original(self, texts, kind, batch_size)

    monkeypatch.setattr(Recorder, "encode", flaky)
    with pytest.raises(RuntimeError):
        main(recorder + ["--out", out, "--chunk-passages", "2", "--device", "cpu"])
    assert already_embedded(Path(out)) == 2
    monkeypatch.setattr(Recorder, "encode", original)
    capsys.readouterr()
    ticks = iter([100.0, 100.5])
    monkeypatch.setattr(dense_module, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))
    main(recorder + ["--out", out, "--chunk-passages", "2", "--device", "cpu"])
    assert capsys.readouterr().out.splitlines()[0] == "3/3 passages, 2 per second, 0.0 minutes left"


def test_nothing_is_counted_as_embedded_without_a_progress_record(tmp_path):
    assert already_embedded(tmp_path / "idx") == 0
    (tmp_path / "idx.part").mkdir()
    assert already_embedded(tmp_path / "idx") == 0
    (tmp_path / "idx.part" / PROGRESS).write_text(json.dumps({"passages": 123}))
    assert already_embedded(tmp_path / "idx") == 123


def test_command_line_options_choose_the_precisions(tmp_path):
    write_model_directory(tmp_path / "model")
    write_passages(tmp_path / "p.jsonl", corpus(3))
    arguments = [str(tmp_path / "p.jsonl"), "--model", str(tmp_path / "model"), "--device", "cpu", "--max-length", "12"]
    main(arguments + ["--out", str(tmp_path / "a"), "--dtype", "float32", "--embed-dtype", "bfloat16"])
    meta = json.loads((tmp_path / "a" / "meta.json").read_text())
    assert (meta["dtype"], meta["model"]) == ("float32", "model bfloat16")
    assert (tmp_path / "a" / "vectors.bin").stat().st_size == 3 * 32 * 4


@pytest.mark.parametrize("arguments", [["--dtype", "float64"], ["--embed-dtype", "int8"], ["--device"], []])
def test_command_line_refuses_bad_arguments(tmp_path, arguments):
    base = [str(tmp_path / "p.jsonl"), "--model", str(tmp_path), "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        main(base + arguments if arguments else [])


def dense_copy(tmp_path, name, change):
    import shutil

    if not (tmp_path / "good").exists():
        build_dense_index(corpus(12), tmp_path / "good", Fake(), dtype="float32")
    shutil.copytree(tmp_path / "good", tmp_path / name)
    change(tmp_path / name)
    return tmp_path / name


@pytest.mark.parametrize("file, keep", [("vectors.bin", 500), ("vectors.bin", 0), ("starts.npy", 140), ("passages.jsonl", 400), ("passages.jsonl", 0)])
def test_a_truncated_dense_file_is_refused_at_load_and_named(tmp_path, file, keep):
    folder = dense_copy(tmp_path, "bad", lambda f: (f / file).write_bytes((f / file).read_bytes()[:keep]))
    with pytest.raises(ValueError, match=file.replace(".", r"\.")):
        DenseIndex(folder, Fake())


def test_extra_bytes_and_wrong_counts_are_refused_by_a_dense_index(tmp_path):
    def grow(name):
        def change(folder):
            with open(folder / name, "ab") as stream:
                stream.write(b"\0" * 16 if name == "vectors.bin" else b"{}\n")

        return change

    for name in ("vectors.bin", "passages.jsonl"):
        with pytest.raises(ValueError, match="bytes, expected"):
            DenseIndex(dense_copy(tmp_path, f"extra-{name}", grow(name)), Fake())

    def lie(folder):
        meta = json.loads((folder / "meta.json").read_text())
        meta["passages"] = 13
        (folder / "meta.json").write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="lists 12 passages, expected 13"):
        DenseIndex(dense_copy(tmp_path, "lie", lie), Fake())


def test_a_dense_index_built_before_sizes_were_recorded_still_loads(tmp_path):
    def drop_size(folder):
        meta = json.loads((folder / "meta.json").read_text())
        del meta["passages_bytes"]
        (folder / "meta.json").write_text(json.dumps(meta))

    index = DenseIndex(dense_copy(tmp_path, "old", drop_size), Fake())
    assert len(index.search("x", 3)) == 3
    size = (tmp_path / "good" / "passages.jsonl").stat().st_size

    def cut_passages(folder):
        drop_size(folder)
        (folder / "passages.jsonl").write_bytes((folder / "passages.jsonl").read_bytes()[: size - 7])

    with pytest.raises(ValueError, match="passages.jsonl"):
        DenseIndex(dense_copy(tmp_path, "old-cut", cut_passages), Fake())
