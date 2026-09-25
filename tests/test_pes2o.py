import gzip
import hashlib
import json

from scout.data.common_pile import DEFAULT_FOLDER
from scout.data.pes2o import REPO, iter_records, main

REVISION = "d" * 40

DOCUMENTS = [
    {
        "added": "2020-05-18T01:00:17.886Z",
        "created": "2020-05-15T00:00:00.000",
        "id": "218665656",
        "metadata": {
            "extfieldsofstudy": ["Physics"],
            "oa_license": "CCBY",
            "oa_status": "HYBRID",
            "oa_url": "https://link.example/paper.pdf",
            "pdf_src": "Arxiv",
            "provenance": "peS2o-0000.json.gz:1",
            "s2fieldsofstudy": ["Physics"],
            "year": 2020,
        },
        "source": "pes2o/s2orc",
        "text": "A paper title\n\nThe abstract.\n\nIntroduction\nThe body.",
        "version": "v3-fos-license",
    },
    {"id": "7", "text": "Metadata missing entirely."},
]

RECORDS = [
    {
        "text": "A paper title\n\nThe abstract.\n\nIntroduction\nThe body.",
        "url": "https://link.example/paper.pdf",
        "id": "218665656",
        "license": "CCBY",
    },
    {"text": "Metadata missing entirely.", "url": "", "id": "7", "license": ""},
]


def gzip_lines(documents):
    return gzip.compress("".join(json.dumps(d) + "\n" for d in documents).encode())


def test_targets_the_filtered_common_pile_pes2o():
    assert REPO == "common-pile/peS2o_filtered"


def test_reads_records_with_their_licenses(tmp_path):
    path = tmp_path / "peS2o-0000.json.gz"
    path.write_bytes(gzip_lines(DOCUMENTS))
    assert list(iter_records(path)) == RECORDS


def test_command_line_downloads_a_verified_file_from_the_root(serve, tmp_path):
    body = gzip_lines(DOCUMENTS)
    path = "peS2o-0000.json.gz"
    listing = [
        {"type": "file", "path": ".gitattributes", "size": 2461},
        {"type": "file", "path": "README.md", "size": 3000},
        {"type": "file", "path": path, "size": len(body), "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)}},
    ]
    server = serve(
        {
            f"api/datasets/{REPO}": json.dumps({"sha": REVISION}).encode(),
            f"api/datasets/{REPO}/tree/{REVISION}": json.dumps(listing).encode(),
            f"datasets/{REPO}/resolve/{REVISION}/{path}": body,
        }
    )
    assert main(["--out", str(tmp_path), "--base-url", server.url]) == 0
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert (selection["repo"], selection["folder"]) == (REPO, DEFAULT_FOLDER)
    assert [f["path"] for f in selection["files"]] == [path]
    assert list(iter_records(tmp_path / path)) == RECORDS
