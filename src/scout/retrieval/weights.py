from __future__ import annotations

import json
import os
import struct
import sys
from math import prod
from pathlib import Path

import torch

MAX_HEADER_BYTES = 100 * 1024 * 1024
DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    if sys.byteorder != "little":
        raise OSError("safetensors files are little endian")
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        prefix = stream.read(8)
        if len(prefix) < 8:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (header_size,) = struct.unpack("<Q", prefix)
        if header_size > MAX_HEADER_BYTES or 8 + header_size > size:
            raise ValueError(f"{path} has an impossible header length")
        try:
            header = json.loads(stream.read(header_size).decode("utf-8"))
        except ValueError:
            raise ValueError(f"{path} has an unreadable header") from None
        if not isinstance(header, dict):
            raise ValueError(f"{path} has an unreadable header")
        header.pop("__metadata__", None)
        data_start, data_size = 8 + header_size, size - 8 - header_size
        entries = []
        for name, entry in header.items():
            entries.append((_parse(path, name, entry, data_size), name))
        entries.sort(key=lambda item: item[0][2])
        position = 0
        for (_, _, begin, end, _), name in entries:
            if begin != position:
                raise ValueError(f"{path}: tensor {name} leaves a gap or overlaps the one before it")
            position = end
        if position != data_size:
            raise ValueError(f"{path}: the tensors do not cover the whole data section")
        tensors = {}
        for (dtype, shape, begin, end, count), name in entries:
            if count == 0:
                tensors[name] = torch.empty(shape, dtype=dtype)
                continue
            stream.seek(data_start + begin)
            raw = bytearray(stream.read(end - begin))
            tensors[name] = torch.frombuffer(raw, dtype=dtype, count=count).reshape(shape)
        return tensors


def _parse(path: Path, name: str, entry: object, data_size: int) -> tuple[torch.dtype, list[int], int, int, int]:
    if not isinstance(entry, dict) or entry.get("dtype") not in DTYPES:
        raise ValueError(f"{path}: tensor {name} has an unsupported or missing dtype")
    shape, offsets = entry.get("shape"), entry.get("data_offsets")
    if not isinstance(shape, list) or not all(type(n) is int and n >= 0 for n in shape):
        raise ValueError(f"{path}: tensor {name} has a bad shape")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(n) is int for n in offsets):
        raise ValueError(f"{path}: tensor {name} has bad offsets")
    begin, end = offsets
    dtype = DTYPES[entry["dtype"]]
    count = prod(shape)
    if not 0 <= begin <= end <= data_size or end - begin != count * dtype.itemsize:
        raise ValueError(f"{path}: tensor {name} has offsets that do not fit its shape")
    return dtype, shape, begin, end, count
