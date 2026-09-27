from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_PATH = Path(__file__).with_name("tokenizer.json")
FILE_SHA256 = "545c56953f9d7a6188cba908776c390d9e90b3c2e3d23af1da4ddececdd81ee9"
CORE_SHA256 = "b3023f64ec116be6b6bfda89b24688ee0ea91baf7881e3ee422a63e665fa0e5a"


def core_sha256(path: Path = TOKENIZER_PATH) -> str:
    content = json.loads(path.read_text(encoding="utf-8"))
    content.pop("added_tokens")
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_PATH))
