from __future__ import annotations

import argparse
import sys
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from scout.data.fetch import HUB_URL, download_with_retries


@dataclass(frozen=True)
class ModelSource:
    repo: str
    revision: str
    files: Mapping[str, str] = field(default_factory=dict)


E5_BASE_V2 = ModelSource(
    "intfloat/e5-base-v2",
    "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
    {
        "config.json": "01cc39aa39538a8179aa131c8a16adc03b506c92009f091502e4eb0c702f5f78",
        "model.safetensors": "d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693",
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "tokenizer_config.json": "ae83fa6ca0333117ff12606020af925d648667ef70d92ff7f27d781ba0ca4544",
        "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
        "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        "modules.json": "c6e29747481e8b5dd2b58401966aeac910de39092f90cda9a704b1545f902b04",
        "1_Pooling/config.json": "f586ab6c734af7b8d14d898bfa7faa886f23b46f86dd1730514309893af13d75",
    },
)
SOURCES = {"e5-base-v2": E5_BASE_V2}


def model_file_url(source: ModelSource, path: str, base_url: str = HUB_URL) -> str:
    return f"{base_url}{source.repo}/resolve/{source.revision}/{urllib.parse.quote(path)}"


def fetch_model(
    source: ModelSource, directory: Path, retries: int = 5, wait: float = 5.0, base_url: str = HUB_URL
) -> Path:
    directory = Path(directory)
    for path, sha256 in source.files.items():
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe path {path!r}")
        download_with_retries(model_file_url(source, path, base_url), directory / pure, retries, wait, sha256)
    return directory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a pinned retrieval model and check every file.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(SOURCES), default="e5-base-v2")
    parser.add_argument("--base-url", default=HUB_URL)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.retries < 0 or args.retry_wait < 0:
        parser.error("retries and retry wait must not be negative")
    source = SOURCES[args.model]
    fetch_model(source, args.out, args.retries, args.retry_wait, args.base_url)
    print(f"{source.repo} at {source.revision} is in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
