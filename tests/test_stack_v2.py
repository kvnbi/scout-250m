import gzip
import hashlib
import json

from scout.data.common_pile import DEFAULT_FOLDER
from scout.data.stack_v2 import REPO, iter_records, main

REVISION = "b" * 40
COMMIT = "3e863f262be0e391d7803cf6713c7242e5494a10"


def code(blob, path, text, terms="MIT", repo_terms="MIT", generated=False, vendor=False):
    return {
        "added": "2024-11-18T18:08:22.422379+00:00",
        "created": "2021-11-09T14:53:37",
        "id": blob,
        "int_score": 3,
        "metadata": {
            "blob_id": blob,
            "detected_licenses": [terms] if terms else [],
            "gha_license_id": repo_terms,
            "is_generated": generated,
            "is_vendor": vendor,
            "language": "Python",
            "license": terms,
            "license_type": "permissive",
            "path": path,
            "provenance": "stack-edu-0073.json.gz:1",
            "repo_name": "octo/demo",
            "url": f"https://raw.githubusercontent.com/octo/demo/{COMMIT}{path}",
        },
        "score": 3.40625,
        "source": "stackv2",
        "text": text,
    }


DOCUMENTS = [
    code("1" * 40, "/src/app.py", "def add(a, b):\r\n    return a + b\r\n"),
    code("2" * 40, "/src/api_pb2.py", "DESCRIPTOR = _descriptor.FileDescriptor()\n", generated=True),
    code("3" * 40, "/vendor/six.py", "import sys\n", vendor=True),
    code("4" * 40, "/src/util.py", "def one():\n    return 1\n", terms="", repo_terms="BSD-3-Clause"),
    {"id": "5" * 40, "text": "# Notes\n"},
]

RECORDS = [
    {
        "text": "def add(a, b):\r\n    return a + b\r\n",
        "url": f"https://raw.githubusercontent.com/octo/demo/{COMMIT}/src/app.py",
        "id": "1" * 40,
        "repo": "octo/demo",
        "language": "Python",
        "score": 3,
        "license": "MIT",
    },
    {
        "text": "def one():\n    return 1\n",
        "url": f"https://raw.githubusercontent.com/octo/demo/{COMMIT}/src/util.py",
        "id": "4" * 40,
        "repo": "octo/demo",
        "language": "Python",
        "score": 3,
        "license": "BSD-3-Clause",
    },
    {"text": "# Notes\n", "url": "", "id": "5" * 40, "repo": "", "language": "", "score": None, "license": ""},
]


def gzip_lines(documents):
    return gzip.compress("".join(json.dumps(d) + "\n" for d in documents).encode())


def test_targets_the_filtered_common_pile_stack_v2_edu():
    assert REPO == "common-pile/stackv2_edu_filtered"


def test_reads_code_and_skips_generated_and_vendored_files(tmp_path):
    path = tmp_path / "stack-edu-0073.json.gz"
    path.write_bytes(gzip_lines(DOCUMENTS))
    assert list(iter_records(path)) == RECORDS


def test_command_line_downloads_a_verified_file_from_the_root(serve, tmp_path):
    body = gzip_lines(DOCUMENTS)
    path = "stack-edu-0073.json.gz"
    listing = [
        {"type": "file", "path": "README.md", "size": 3818},
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
    assert list(iter_records(tmp_path / path)) == RECORDS
