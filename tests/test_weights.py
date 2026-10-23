import json
import struct

import pytest
import torch
from bert_files import safetensors_bytes

import scout.retrieval.weights as weights
from scout.retrieval.weights import DTYPES, read_safetensors


def write(tmp_path, data, name="m.safetensors"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def raw_file(header, payload=b"", length=None):
    blob = json.dumps(header).encode() if not isinstance(header, bytes) else header
    return struct.pack("<Q", len(blob) if length is None else length) + blob + payload


def f32(*values):
    return struct.pack(f"<{len(values)}f", *values)


def test_values_shapes_and_dtypes_round_trip(tmp_path):
    tensors = {
        "w": torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7,
        "half": torch.tensor([1.5, -2.25], dtype=torch.float16),
        "brain": torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16),
        "ids": torch.tensor([[7, -8]], dtype=torch.int64),
        "small": torch.tensor([1, -2, 3], dtype=torch.int8),
        "bytes": torch.tensor([0, 255], dtype=torch.uint8),
        "flags": torch.tensor([True, False, True]),
        "wide": torch.tensor([1e300, -1e-300], dtype=torch.float64),
        "int32": torch.tensor([2**31 - 1], dtype=torch.int32),
        "int16": torch.tensor([-32768], dtype=torch.int16),
        "scalar": torch.tensor(3.5),
        "empty": torch.empty(0, 4),
    }
    loaded = read_safetensors(write(tmp_path, safetensors_bytes(tensors)))
    assert set(loaded) == set(tensors)
    for name, tensor in tensors.items():
        assert loaded[name].dtype == tensor.dtype and loaded[name].shape == tensor.shape, name
        assert torch.equal(loaded[name], tensor), name


def test_every_supported_dtype_is_covered_by_the_round_trip_test():
    assert set(DTYPES) == {"F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"}


def test_the_order_of_the_header_does_not_matter_and_metadata_is_ignored(tmp_path):
    tensors = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    data = safetensors_bytes(tensors, metadata={"format": "pt", "note": "x"}, order=["b", "a"])
    loaded = read_safetensors(write(tmp_path, data))
    assert sorted(loaded) == ["a", "b"] and torch.equal(loaded["a"], tensors["a"]) and torch.equal(loaded["b"], tensors["b"])


def test_the_tensors_own_their_memory_after_the_file_is_gone(tmp_path):
    path = write(tmp_path, safetensors_bytes({"a": torch.tensor([1.0, 2.0])}))
    loaded = read_safetensors(path)
    path.unlink()
    loaded["a"] += 1
    assert loaded["a"].tolist() == [2.0, 3.0]


def test_a_file_with_no_tensors_is_an_empty_dictionary(tmp_path):
    assert read_safetensors(write(tmp_path, raw_file({}))) == {}


def test_matches_the_reference_implementation_when_it_is_installed(tmp_path):
    torch_format = pytest.importorskip("safetensors.torch")
    tensors = {"w": torch.randn(4, 5), "b": torch.randn(5, dtype=torch.float16), "i": torch.arange(3)}
    torch_format.save_file(tensors, str(tmp_path / "ref.safetensors"), metadata={"format": "pt"})
    loaded = read_safetensors(tmp_path / "ref.safetensors")
    assert all(torch.equal(loaded[name], tensors[name]) for name in tensors)
    (tmp_path / "ours.safetensors").write_bytes(safetensors_bytes(tensors, metadata={"format": "pt"}))
    assert torch_format.load_file(str(tmp_path / "ours.safetensors")).keys() == tensors.keys()


def entry(dtype="F32", shape=(2,), begin=0, end=8):
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, end]}


BAD_FILES = {
    "shorter than the length prefix": (b"\x01\x02\x03", "too short"),
    "seven bytes": (b"\x01" * 7, "too short"),
    "empty": (b"", "too short"),
    "header longer than the file": (raw_file({"a": entry()}, f32(1, 2), length=10_000), "impossible header length"),
    "header over the size limit": (struct.pack("<Q", 200 * 1024 * 1024) + b"{}", "impossible header length"),
    "header not utf-8": (raw_file(b"\xff\xfe{}"), "unreadable header"),
    "header not json": (raw_file(b"not json"), "unreadable header"),
    "header is a list": (raw_file(b"[]"), "unreadable header"),
    "entry is not an object": (raw_file({"a": 5}, f32(1, 2)), "unsupported or missing dtype"),
    "unsupported dtype": (raw_file({"a": entry("U16", (2,), 0, 4)}, b"1234"), "unsupported or missing dtype"),
    "missing dtype": (raw_file({"a": {"shape": [2], "data_offsets": [0, 8]}}, f32(1, 2)), "unsupported or missing dtype"),
    "negative dimension": (raw_file({"a": entry(shape=(-2,))}, f32(1, 2)), "bad shape"),
    "two negative dimensions": (raw_file({"a": entry(shape=(-2, -3), end=24)}, f32(1, 2, 3, 4, 5, 6)), "bad shape"),
    "fractional dimension": (raw_file({"a": {"dtype": "F32", "shape": [2.0], "data_offsets": [0, 8]}}, f32(1, 2)), "bad shape"),
    "boolean dimension": (raw_file({"a": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}}, f32(1)), "bad shape"),
    "shape is not a list": (raw_file({"a": {"dtype": "F32", "shape": 2, "data_offsets": [0, 8]}}, f32(1, 2)), "bad shape"),
    "offsets wrong length": (raw_file({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0]}}, f32(1, 2)), "bad offsets"),
    "offsets not integers": (raw_file({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8.0]}}, f32(1, 2)), "bad offsets"),
    "offsets boolean": (raw_file({"a": {"dtype": "F32", "shape": [0], "data_offsets": [False, False]}}, b""), "bad offsets"),
    "offsets are a number": (raw_file({"a": {"dtype": "F32", "shape": [2], "data_offsets": 5}}, f32(1, 2)), "bad offsets"),
    "offsets are text": (raw_file({"a": {"dtype": "F32", "shape": [2], "data_offsets": "ab"}}, f32(1, 2)), "bad offsets"),
    "offsets reversed": (raw_file({"a": entry(begin=8, end=0)}, f32(1, 2)), "do not fit"),
    "offsets past the data": (raw_file({"a": entry(begin=0, end=16, shape=(4,))}, f32(1, 2)), "do not fit"),
    "shape claims more than the bytes": (raw_file({"a": entry(shape=(3,), begin=0, end=8)}, f32(1, 2)), "do not fit"),
    "shape claims less than the bytes": (raw_file({"a": entry(shape=(1,), begin=0, end=8)}, f32(1, 2)), "do not fit"),
    "gap between tensors": (raw_file({"a": entry(shape=(1,), end=4), "b": entry(shape=(1,), begin=8, end=12)}, f32(1, 2, 3)), "gap or overlaps"),
    "overlapping tensors": (raw_file({"a": entry(), "b": entry(shape=(1,), begin=4, end=8)}, f32(1, 2)), "gap or overlaps"),
    "first tensor does not start at zero": (raw_file({"a": entry(shape=(1,), begin=4, end=8)}, f32(1, 2)), "gap or overlaps"),
    "unreferenced trailing data": (raw_file({"a": entry()}, f32(1, 2) + b"\x00\x00\x00\x00"), "whole data section"),
}


@pytest.mark.parametrize("name", sorted(BAD_FILES))
def test_malformed_files_are_refused_with_the_right_reason(tmp_path, name):
    data, message = BAD_FILES[name]
    with pytest.raises(ValueError, match=message):
        read_safetensors(write(tmp_path, data))


def test_the_header_size_limit_applies_on_its_own(tmp_path, monkeypatch):
    path = write(tmp_path, safetensors_bytes({"a": torch.tensor([1.0, 2.0])}))
    assert read_safetensors(path)["a"].tolist() == [1.0, 2.0]
    monkeypatch.setattr(weights, "MAX_HEADER_BYTES", 10)
    with pytest.raises(ValueError, match="impossible header length"):
        read_safetensors(path)


def test_tensors_may_sit_in_the_data_section_in_a_different_order_from_the_header(tmp_path):
    header = {"later": entry(shape=(1,), begin=4, end=8), "earlier": entry(shape=(1,), begin=0, end=4)}
    loaded = read_safetensors(write(tmp_path, raw_file(header, f32(1.5, 2.5))))
    assert loaded["earlier"].tolist() == [1.5] and loaded["later"].tolist() == [2.5]


def test_a_valid_hand_written_file_reads(tmp_path):
    data = raw_file({"a": entry(), "__metadata__": {"x": "y"}}, f32(1.5, -2.5))
    assert read_safetensors(write(tmp_path, data))["a"].tolist() == [1.5, -2.5]


def test_a_missing_file_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_safetensors(tmp_path / "absent.safetensors")
