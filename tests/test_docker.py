import os
import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "docker" / "Dockerfile").read_text()
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def pins(name):
    result = {}
    for line in (ROOT / "docker" / name).read_text().splitlines():
        package, version = line.split("==")
        key = canonicalize_name(package)
        assert key not in result
        result[key] = Version(version)
    return result


def declared():
    project = PYPROJECT["project"]
    return [Requirement(text) for text in project["dependencies"] + project["optional-dependencies"]["dev"]]


def test_the_base_image_is_pinned_by_digest_on_the_python_the_project_requires():
    bases = re.findall(r"^FROM (\S+)$", DOCKERFILE, re.M)
    assert len(bases) == 1
    match = re.fullmatch(r"python:(\d+\.\d+\.\d+)-slim-[a-z]+@sha256:[0-9a-f]{64}", bases[0])
    assert match
    assert Version(match[1]) in SpecifierSet(PYPROJECT["project"]["requires-python"])


def test_rclone_is_pinned_and_checked_against_its_checksum_before_it_is_unpacked():
    assert re.search(r"^ARG RCLONE_VERSION=v\d+\.\d+\.\d+$", DOCKERFILE, re.M)
    assert re.search(r"^ARG RCLONE_SHA256=[0-9a-f]{64}$", DOCKERFILE, re.M)
    assert DOCKERFILE.index("sha256sum -c") < DOCKERFILE.index("unzip -j")


def test_every_python_install_uses_the_lock_or_adds_nothing():
    blocks = re.findall(r"pip install(?:[^\n]*\\\n)*[^\n]*", DOCKERFILE)
    assert len(blocks) == 2
    for block in blocks:
        assert "--requirement /tmp/requirements.lock" in block or "--no-deps" in block


def test_the_lock_pins_everything_and_matches_the_top_level_requirements_and_the_cuda_index():
    lock = pins("requirements.lock")
    top = pins("requirements.in")
    assert all(lock[name] == version for name, version in top.items())
    tag = lock["torch"].local
    assert tag and f"https://download.pytorch.org/whl/{tag}" in DOCKERFILE
    assert {"triton", "numpy", "pyarrow", "tokenizers", "pytest"} <= set(lock)
    assert any(name.startswith("nvidia-nccl") for name in lock)


def test_the_lock_satisfies_the_declared_requirements():
    lock = pins("requirements.lock")
    for requirement in declared():
        assert requirement.specifier.contains(lock[canonicalize_name(requirement.name)], prereleases=True)


def test_the_floors_are_the_declared_minimums_and_the_check_script_runs_them():
    floors = pins("floors.txt")
    minimums = {}
    for requirement in declared():
        (spec,) = [s for s in requirement.specifier if s.operator == ">="]
        minimums[canonicalize_name(requirement.name)] = Version(spec.version)
    assert floors == minimums
    script = ROOT / "docker" / "check_floors.sh"
    assert os.access(script, os.X_OK)
    assert "docker/floors.txt" in script.read_text()


def test_the_build_context_leaves_out_data_checkpoints_and_history():
    lines = set((ROOT / ".dockerignore").read_text().split())
    assert {".git", "data", "runs", "*.pt"} <= lines
