import hashlib
import re

import pytest

from scout.retrieval.model_fetch import E5_BASE_V2, SOURCES, ModelSource, fetch_model, main, model_file_url

REVISION = "e" * 40


def source_for(files):
    return ModelSource("org/model", REVISION, {name: hashlib.sha256(data).hexdigest() for name, data in files.items()})


FILES = {"config.json": b'{"a": 1}', "model.safetensors": b"weights" * 1000, "1_Pooling/config.json": b"{}"}


def served(files):
    return {f"org/model/resolve/{REVISION}/{name}": data for name, data in files.items()}


def test_the_url_is_the_model_resolve_path_and_not_a_dataset_path():
    assert model_file_url(E5_BASE_V2, "1_Pooling/config.json") == (
        f"https://huggingface.co/intfloat/e5-base-v2/resolve/{E5_BASE_V2.revision}/1_Pooling/config.json"
    )
    assert model_file_url(ModelSource("a/b", REVISION), "dir with space/x.json", "http://h/") == f"http://h/a/b/resolve/{REVISION}/dir%20with%20space/x.json"


def test_the_pinned_e5_source_is_complete_and_well_formed():
    assert SOURCES == {"e5-base-v2": E5_BASE_V2}
    assert E5_BASE_V2.repo == "intfloat/e5-base-v2" and re.fullmatch(r"[0-9a-f]{40}", E5_BASE_V2.revision)
    assert {"config.json", "model.safetensors", "tokenizer.json", "1_Pooling/config.json", "modules.json"} <= set(E5_BASE_V2.files)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in E5_BASE_V2.files.values())
    assert len(set(E5_BASE_V2.files.values())) == len(E5_BASE_V2.files)


def test_every_file_is_downloaded_into_its_folder_and_checked(serve, tmp_path):
    server = serve(served(FILES))
    assert fetch_model(source_for(FILES), tmp_path / "out", base_url=server.url) == tmp_path / "out"
    for name, data in FILES.items():
        assert (tmp_path / "out" / name).read_bytes() == data


def test_files_already_present_are_not_fetched_again(serve, tmp_path):
    server = serve(served(FILES))
    fetch_model(source_for(FILES), tmp_path / "out", base_url=server.url)
    count = len(server.requests)
    fetch_model(source_for(FILES), tmp_path / "out", base_url=server.url)
    assert len(server.requests) == count


def test_a_file_with_the_wrong_checksum_is_rejected_and_not_kept(serve, tmp_path):
    tampered = dict(FILES, **{"model.safetensors": b"evil" * 1000})
    server = serve(served(tampered))
    with pytest.raises(OSError, match="SHA-256"):
        fetch_model(source_for(FILES), tmp_path / "out", retries=0, base_url=server.url)
    assert not (tmp_path / "out" / "model.safetensors").exists()


@pytest.mark.parametrize("path", ["../escape.json", "/absolute.json", "a/../../b.json"])
def test_paths_that_leave_the_folder_are_refused(tmp_path, path):
    with pytest.raises(ValueError, match="unsafe"):
        fetch_model(ModelSource("a/b", REVISION, {path: "0" * 64}), tmp_path / "out")
    assert not list(tmp_path.glob("**/*.json"))


def test_command_line_downloads_a_pinned_model(serve, tmp_path, monkeypatch, capsys):
    server = serve(served(FILES))
    monkeypatch.setitem(SOURCES, "tiny", source_for(FILES))
    assert main(["--out", str(tmp_path / "out"), "--model", "tiny", "--base-url", server.url]) == 0
    assert (tmp_path / "out" / "1_Pooling" / "config.json").read_bytes() == b"{}"
    assert f"org/model at {REVISION}" in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [["--retries", "-1"], ["--retry-wait", "-1"], ["--model", "nope"], []])
def test_command_line_refuses_bad_arguments(tmp_path, arguments):
    base = [] if "--out" in arguments or not arguments else ["--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        main(arguments + base if arguments else [])
