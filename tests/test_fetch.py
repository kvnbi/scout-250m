import http.client
import json

import pytest

from scout.data.fetch import (
    HubFile,
    download,
    download_with_retries,
    hub_file_url,
    hub_files,
    hub_revision,
    sample_files,
    sha256_of,
)

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
    import scout.data.fetch as fetch

    server = serve({"f.bin": BODY})

    def cut(source, target, length):
        target.write(source.read(3000))
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(fetch.shutil, "copyfileobj", cut)
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


def test_complete_partial_is_confirmed_by_sha256_when_the_size_is_hidden(serve, tmp_path):
    import hashlib

    server = serve({"f.bin": BODY}, range_on_416=False)
    (tmp_path / "f.bin.part").write_bytes(BODY)
    assert download(server.url + "f.bin", tmp_path / "f.bin", sha256=hashlib.sha256(BODY).hexdigest()).read_bytes() == BODY
    assert len(server.requests) == 1


def test_unconfirmable_partial_is_downloaded_afresh(serve, tmp_path):
    import hashlib

    server = serve({"f.bin": BODY}, range_on_416=False)
    (tmp_path / "f.bin.part").write_bytes(b"y" * len(BODY))
    got = download_with_retries(server.url + "f.bin", tmp_path / "f.bin", 2, 0.0, hashlib.sha256(BODY).hexdigest())
    assert got.read_bytes() == BODY
    (tmp_path / "g.bin.part").write_bytes(BODY)
    server.files["g.bin"] = BODY
    assert download_with_retries(server.url + "g.bin", tmp_path / "g.bin", 2, 0.0).read_bytes() == BODY


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


def test_verifies_sha256_and_discards_corrupt_files(serve, tmp_path):
    import hashlib

    good = hashlib.sha256(BODY).hexdigest()
    server = serve({"f.bin": BODY})
    assert download(server.url + "f.bin", tmp_path / "ok.bin", sha256=good).read_bytes() == BODY
    with pytest.raises(OSError):
        download(server.url + "f.bin", tmp_path / "bad.bin", sha256="0" * 64)
    assert not (tmp_path / "bad.bin").exists()
    assert not (tmp_path / "bad.bin.part").exists()


def test_sha256_is_checked_after_a_resume(serve, tmp_path):
    import hashlib

    server = serve({"f.bin": BODY})
    (tmp_path / "f.bin.part").write_bytes(b"x" * 1000)
    with pytest.raises(OSError):
        download(server.url + "f.bin", tmp_path / "f.bin", sha256=hashlib.sha256(BODY).hexdigest())
    assert download(server.url + "f.bin", tmp_path / "f.bin", sha256=hashlib.sha256(BODY).hexdigest()).read_bytes() == BODY


def test_sha256_of_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "f.bin"
    path.write_bytes(BODY * 50)
    assert sha256_of(path) == hashlib.sha256(BODY * 50).hexdigest()


REVISION = "a" * 40


def listing_entry(path, sha="b" * 64, size=10, kind="file"):
    entry = {"type": kind, "path": path, "size": size}
    if kind == "file":
        entry["lfs"] = {"oid": sha, "size": size, "pointerSize": 135}
    return entry


def test_reads_the_pinned_revision(serve):
    server = serve({"api/datasets/org/data": json.dumps({"sha": REVISION}).encode()})
    assert hub_revision("org/data", server.url) == REVISION


def test_rejects_a_missing_revision(serve):
    server = serve({"api/datasets/org/data": json.dumps({"sha": "main"}).encode()})
    with pytest.raises(ValueError):
        hub_revision("org/data", server.url)


def test_lists_files_across_pages(serve):
    first = f"api/datasets/org/data/tree/{REVISION}/sample/10BT"
    second = first + "?cursor=next"
    page_one = [listing_entry("sample/10BT/001.parquet", "1" * 64, 7), listing_entry("sample/10BT/sub", kind="directory")]
    page_two = [listing_entry("sample/10BT/000.parquet", "0" * 64, 5), listing_entry("sample/10BT/README.md")]
    server = serve({})
    server.files[first] = (json.dumps(page_one).encode(), {"Link": f'<{server.url}{second}>; rel="next"'})
    server.files[second] = json.dumps(page_two).encode()
    files = hub_files("org/data", REVISION, "sample/10BT", ".parquet", server.url)
    assert files == [HubFile("sample/10BT/000.parquet", 5, "0" * 64), HubFile("sample/10BT/001.parquet", 7, "1" * 64)]


@pytest.mark.parametrize("folder", ["", "/"])
def test_lists_the_repository_root_without_a_redirect(serve, folder):
    entries = [listing_entry("a-0000.json.gz", "2" * 64, 9), listing_entry("README.md"), listing_entry("v0", kind="directory")]
    server = serve({f"api/datasets/org/data/tree/{REVISION}": json.dumps(entries).encode()})
    assert hub_files("org/data", REVISION, folder, ".json.gz", server.url) == [HubFile("a-0000.json.gz", 9, "2" * 64)]
    assert [path for path, _ in server.requests] == [f"/api/datasets/org/data/tree/{REVISION}"]


def test_folder_slashes_are_ignored(serve):
    server = serve({f"api/datasets/org/data/tree/{REVISION}/f": json.dumps([listing_entry("f/x.parquet")]).encode()})
    assert [f.path for f in hub_files("org/data", REVISION, "/f/", ".parquet", server.url)] == ["f/x.parquet"]


def test_listing_requires_sha256_and_matches(serve):
    path = f"api/datasets/org/data/tree/{REVISION}/f"
    server = serve({path: json.dumps([{"type": "file", "path": "f/x.parquet", "size": 3}]).encode()})
    with pytest.raises(ValueError):
        hub_files("org/data", REVISION, "f", ".parquet", server.url)
    server.files[path] = json.dumps([listing_entry("f/x.txt")]).encode()
    with pytest.raises(ValueError):
        hub_files("org/data", REVISION, "f", ".parquet", server.url)


@pytest.mark.parametrize("path", ["/etc/x.parquet", "f/../../x.parquet"])
def test_listing_rejects_unsafe_paths(serve, path):
    server = serve({f"api/datasets/org/data/tree/{REVISION}/f": json.dumps([listing_entry(path)]).encode()})
    with pytest.raises(ValueError):
        hub_files("org/data", REVISION, "f", ".parquet", server.url)


def test_file_urls_are_pinned_and_quoted():
    url = hub_file_url("org/data", REVISION, "data/a b.parquet", "https://hub.example/")
    assert url == f"https://hub.example/datasets/org/data/resolve/{REVISION}/data/a%20b.parquet"


def test_sampling_is_seeded_sorted_and_order_independent():
    files = [HubFile(f"f/{i:03d}.parquet", 1, "c" * 64) for i in range(30)]
    first = sample_files(files, 5, seed=1)
    assert first == sample_files(list(reversed(files)), 5, seed=1)
    assert len({tuple(sample_files(files, 5, seed=s)) for s in range(20)}) > 1
    assert first == sorted(first, key=lambda f: f.path)
    assert len(set(first)) == 5
    assert {f for s in range(300) for f in sample_files(files, 5, seed=s)} == set(files)
    assert sample_files(files, None, seed=1) == files
    assert sample_files(files, 99, seed=1) == files
    with pytest.raises(ValueError):
        sample_files(files, 0, seed=1)


@pytest.mark.parametrize("total, count", [(95, 10), (44, 4), (30, 7), (64, 63), (5, 1)])
def test_sampled_files_are_spread_evenly_so_sorted_groups_keep_their_share(total, count):
    files = [HubFile(f"f/{i:04d}.json.gz", 1, "c" * 64) for i in range(total)]
    for seed in range(50):
        positions = [files.index(f) for f in sample_files(files, count, seed)]
        gaps = [b - a for a, b in zip(positions, positions[1:] + [positions[0] + total])]
        assert set(gaps) <= {total // count, -(-total // count)}
