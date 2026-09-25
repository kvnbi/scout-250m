import gzip
import hashlib
import json

from scout.data.common_pile import DEFAULT_FOLDER
from scout.data.stackexchange import NON_ENGLISH_SITES, REPO, iter_records, main

REVISION = "c" * 40
BY_SA_3 = "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/3.0/"
BY_SA_4 = "Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/4.0/"


def thread(site, number, text, terms):
    return {
        "added": "2025-03-21T12:54:35.624436",
        "created": "2017-03-19T18:24:14",
        "id": number,
        "metadata": {
            "all_licenses": [terms],
            "authors": ["someone", f"https://{site}/users/4762"],
            "include_comments": True,
            "license": terms,
            "provenance": "stackexchange-dolma-0000.json.gz:1",
            "site": site,
            "sort": "votes",
            "url": f"https://{site}/questions/{number}",
        },
        "source": "Stack Exchange",
        "text": text,
    }


DOCUMENTS = [
    thread("physics.stackexchange.com", "219", "Why is the sky blue?\nQuestion body.\n\nAn answer.", BY_SA_3),
    thread("ru.stackoverflow.com", "219", "\u041a\u0430\u043a \u0440\u0430\u0437\u0431\u0438\u0442\u044c xml?", BY_SA_4),
    thread("history.stackexchange.com", "219", "Who built it?\nQuestion body.\n\nAn answer.", BY_SA_4),
]

RECORDS = [
    {
        "text": "Why is the sky blue?\nQuestion body.\n\nAn answer.",
        "url": "https://physics.stackexchange.com/questions/219",
        "id": "physics.stackexchange.com/219",
        "site": "physics.stackexchange.com",
        "license": BY_SA_3,
    },
    {
        "text": "Who built it?\nQuestion body.\n\nAn answer.",
        "url": "https://history.stackexchange.com/questions/219",
        "id": "history.stackexchange.com/219",
        "site": "history.stackexchange.com",
        "license": BY_SA_4,
    },
]


def gzip_lines(documents):
    return gzip.compress("".join(json.dumps(d) + "\n" for d in documents).encode())


def test_targets_the_filtered_common_pile_stack_exchange():
    assert REPO == "common-pile/stackexchange_filtered"


def test_question_numbers_repeat_across_sites_so_ids_carry_the_site(tmp_path):
    path = tmp_path / "stackexchange-dolma-0000.json.gz"
    path.write_bytes(gzip_lines(DOCUMENTS))
    records = list(iter_records(path))
    assert records == RECORDS
    assert len({r["id"] for r in records}) == 2


def test_skips_sites_written_mostly_in_other_languages(tmp_path):
    path = tmp_path / "stackexchange-dolma-0003.json.gz"
    path.write_bytes(gzip_lines(DOCUMENTS))
    assert "ru.stackoverflow.com" not in {r["site"] for r in iter_records(path)}
    assert {"es.stackoverflow.com", "ja.stackoverflow.com", "pt.stackoverflow.com", "ru.stackoverflow.com"} <= NON_ENGLISH_SITES
    assert not {"stackoverflow.com", "esperanto.stackexchange.com", "german.stackexchange.com"} & NON_ENGLISH_SITES


def test_command_line_downloads_a_verified_file_from_the_root(serve, tmp_path):
    body = gzip_lines(DOCUMENTS)
    path = "stackexchange-dolma-0000.json.gz"
    listing = [
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
    assert list(iter_records(tmp_path / path)) == RECORDS
