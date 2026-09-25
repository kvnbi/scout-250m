import gzip
import json

import pytest

from scout.data.common_pile import DEFAULT_FOLDER, SUFFIX, iter_documents

TRICKY = "Line one\nline two\r\nafter NEL\x85and LS\u2028end."


def write(path, lines):
    path.write_bytes(gzip.compress("".join(line + "\n" for line in lines).encode()))
    return path


def test_reads_documents_in_order_and_skips_blank_lines(tmp_path):
    documents = [{"id": "1", "text": "First."}, {"id": "2", "text": "", "metadata": {"a": 1}}]
    path = write(tmp_path / "f.json.gz", [json.dumps(documents[0]), "", "  ", json.dumps(documents[1])])
    assert list(iter_documents(path)) == documents


def test_lines_split_only_at_newlines(tmp_path):
    path = write(tmp_path / "f.json.gz", [json.dumps({"text": TRICKY}, ensure_ascii=False)])
    assert [d["text"] for d in iter_documents(path)] == [TRICKY]


def test_reads_files_made_of_several_gzip_members(tmp_path):
    path = tmp_path / "f.json.gz"
    path.write_bytes(gzip.compress(b'{"text": "a"}\n') + gzip.compress(b'{"text": "b"}\n'))
    assert [d["text"] for d in iter_documents(path)] == ["a", "b"]


@pytest.mark.parametrize("line", ['{"id": "x"}', '{"text": null}', '{"text": 3}', '["text"]', '"text"'])
def test_rejects_documents_without_text(tmp_path, line):
    documents = iter_documents(write(tmp_path / "f.json.gz", ['{"text": "ok"}', line]))
    assert next(documents)["text"] == "ok"
    with pytest.raises(ValueError, match="line 2"):
        next(documents)


def test_rejects_broken_json(tmp_path):
    with pytest.raises(ValueError):
        list(iter_documents(write(tmp_path / "f.json.gz", ['{"text": "cut'])))


def test_files_sit_at_the_repository_root_as_gzip_json_lines():
    assert DEFAULT_FOLDER == ""
    assert SUFFIX == ".json.gz"
