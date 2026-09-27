from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_PATH = Path(__file__).with_name("tokenizer.json")
FILE_SHA256 = "6854cf835b0ad9d1b2be14f962946de20810730bafbaa72a23ee048fb240d48c"
CORE_SHA256 = "b3023f64ec116be6b6bfda89b24688ee0ea91baf7881e3ee422a63e665fa0e5a"


def core_sha256(path: Path = TOKENIZER_PATH) -> str:
    content = json.loads(path.read_text(encoding="utf-8"))
    content.pop("added_tokens")
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_PATH))
