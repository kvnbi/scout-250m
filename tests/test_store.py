import numpy as np
import pytest

import scout.retrieval.store as store_module
from scout.retrieval.passages import Passage
from scout.retrieval.store import PASSAGES, PassageStore, PassageWriter, write_starts


def passages(count=5):
    return [
        Passage(f"{i}:0", f"Title {i}", "Sec > Sub", f"line one\nline two {i} \"quoted\" \\ \u00e9\u4e2d", f"https://x.org/{i}", "2018-12-20")
        for i in range(count)
    ]


def write_all(directory, items, resume_bytes=0):
    writer = PassageWriter(directory, resume_bytes)
    for item in items:
        writer.write(item)
    writer.flush()
    position = writer.position
    writer.close()
    return position


def test_passages_round_trip_through_the_writer_and_the_store(tmp_path):
    items = passages()
    position = write_all(tmp_path, items)
    assert position == (tmp_path / PASSAGES).stat().st_size
    assert write_starts(tmp_path) == 5
    store = PassageStore(tmp_path)
    assert len(store) == 5 and [store.passage(i) for i in range(5)] == items
    for bad in (-1, 5, 99):
        with pytest.raises(IndexError):
            store.passage(bad)


def test_starting_afresh_replaces_whatever_file_was_there(tmp_path):
    (tmp_path / PASSAGES).write_bytes(b"stale bytes from an older build\n" * 3)
    items = passages(2)
    write_all(tmp_path, items)
    write_starts(tmp_path)
    store = PassageStore(tmp_path)
    assert len(store) == 2 and [store.passage(i) for i in range(2)] == items


def test_the_starts_are_the_byte_offsets_of_each_line(tmp_path):
    write_all(tmp_path, passages(4))
    write_starts(tmp_path)
    starts = np.load(tmp_path / "starts.npy")
    data = (tmp_path / PASSAGES).read_bytes()
    assert starts.dtype == np.int64 and starts[0] == 0
    assert [data[s - 1 : s] for s in starts[1:]] == [b"\n"] * 3
    assert len(data.splitlines()) == 4


def test_resuming_keeps_the_agreed_bytes_and_drops_everything_after_them(tmp_path):
    items = passages(6)
    first = write_all(tmp_path, items[:3])
    write_all(tmp_path, items[3:5], resume_bytes=first)
    with open(tmp_path / PASSAGES, "ab") as stream:
        stream.write(b'{"torn": "line')
    resumed = write_all(tmp_path, items[3:], resume_bytes=first)
    assert resumed == (tmp_path / PASSAGES).stat().st_size
    write_starts(tmp_path)
    store = PassageStore(tmp_path)
    assert [store.passage(i) for i in range(len(store))] == items


def test_scanning_gives_the_same_starts_whatever_the_block_size(tmp_path, monkeypatch):
    write_all(tmp_path, passages(30))
    write_starts(tmp_path)
    reference = np.load(tmp_path / "starts.npy").copy()
    for size in (1, 7, 64, 1000):
        monkeypatch.setattr(store_module, "SCAN_BYTES", size)
        assert write_starts(tmp_path) == 30
        assert np.array_equal(np.load(tmp_path / "starts.npy"), reference)


def test_an_empty_file_has_no_passages_and_a_file_without_a_final_newline_is_refused(tmp_path):
    (tmp_path / PASSAGES).write_bytes(b"")
    assert write_starts(tmp_path) == 0 and len(PassageStore(tmp_path)) == 0
    (tmp_path / PASSAGES).write_bytes(b'{"a": 1}\n{"b": 2')
    with pytest.raises(ValueError, match="newline"):
        write_starts(tmp_path)


def test_the_starts_are_memory_mapped_and_read_only(tmp_path):
    write_all(tmp_path, passages(2))
    write_starts(tmp_path)
    starts = PassageStore(tmp_path)._starts
    assert isinstance(starts, np.memmap)
    with pytest.raises(ValueError):
        starts[0] = 5


def test_a_damaged_array_is_reported_with_its_name_and_a_missing_one_is_not_hidden(tmp_path):
    write_all(tmp_path, passages(3))
    write_starts(tmp_path)
    data = (tmp_path / "starts.npy").read_bytes()
    (tmp_path / "starts.npy").write_bytes(data[:130])
    with pytest.raises(ValueError, match=r"starts\.npy is truncated or damaged"):
        PassageStore(tmp_path)
    (tmp_path / "starts.npy").write_bytes(b"")
    with pytest.raises(ValueError, match=r"starts\.npy is truncated or damaged"):
        PassageStore(tmp_path)
    (tmp_path / "starts.npy").unlink()
    with pytest.raises(FileNotFoundError):
        PassageStore(tmp_path)


def test_a_store_verifies_its_count_its_size_and_its_ending(tmp_path):
    write_all(tmp_path, passages(3))
    write_starts(tmp_path)
    store = PassageStore(tmp_path)
    size = (tmp_path / PASSAGES).stat().st_size
    store.verify(3)
    store.verify(3, size)
    with pytest.raises(ValueError, match="lists 3 passages, expected 4"):
        store.verify(4)
    for wrong in (size - 1, size + 1):
        with pytest.raises(ValueError, match="bytes, expected"):
            store.verify(3, wrong)
    (tmp_path / PASSAGES).write_bytes((tmp_path / PASSAGES).read_bytes()[:-1])
    with pytest.raises(ValueError, match="does not end with a newline"):
        PassageStore(tmp_path).verify(3)
    starts = np.load(tmp_path / "starts.npy")
    (tmp_path / PASSAGES).write_bytes((tmp_path / PASSAGES).read_bytes()[: int(starts[-1])])
    with pytest.raises(ValueError, match="ends before its last passage"):
        PassageStore(tmp_path).verify(3)


def test_an_empty_store_verifies(tmp_path):
    (tmp_path / PASSAGES).write_bytes(b"")
    write_starts(tmp_path)
    PassageStore(tmp_path).verify(0, 0)
