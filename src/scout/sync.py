from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from scout.data.fetch import sha256_of

MANIFEST = "latest.json"


class StoreError(OSError):
    pass


class DirectoryStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def upload(self, local: Path, key: str) -> None:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        shutil.copyfile(local, partial)
        os.replace(partial, target)

    def download(self, key: str, local: Path) -> None:
        source = self.root / key
        if not source.exists():
            raise FileNotFoundError(key)
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + ".part")
        shutil.copyfile(source, partial)
        os.replace(partial, local)

    def delete(self, key: str) -> None:
        (self.root / key).unlink(missing_ok=True)


class CommandStore:
    def __init__(
        self,
        upload: Sequence[str],
        download: Sequence[str],
        delete: Sequence[str],
        missing_codes: Sequence[int] = (),
        timeout: float | None = None,
    ) -> None:
        self.commands = {"upload": list(upload), "download": list(download), "delete": list(delete)}
        self.missing_codes = tuple(missing_codes)
        self.timeout = timeout

    def _run(self, name: str, **values: str) -> int:
        argv = [part.format(**values) for part in self.commands[name]]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError as error:
            raise StoreError(f"{argv[0]} is not installed") from error
        except subprocess.TimeoutExpired as error:
            raise StoreError(f"{name} timed out after {self.timeout} seconds") from error
        if result.returncode and not (name != "upload" and result.returncode in self.missing_codes):
            raise StoreError(f"{name} failed with code {result.returncode}: {result.stderr.strip()}")
        return result.returncode

    def upload(self, local: Path, key: str) -> None:
        self._run("upload", local=str(local), key=key)

    def download(self, key: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        local.unlink(missing_ok=True)
        if self._run("download", local=str(local), key=key) or not local.exists():
            raise FileNotFoundError(key)

    def delete(self, key: str) -> None:
        self._run("delete", key=key)


def rclone_store(remote: str, flags: Sequence[str] = (), timeout: float | None = None) -> CommandStore:
    def literal(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    base = literal(remote.rstrip("/"))
    extra = [literal(flag) for flag in flags]
    return CommandStore(
        upload=["rclone", *extra, "copyto", "{local}", f"{base}/{{key}}"],
        download=["rclone", *extra, "copyto", f"{base}/{{key}}", "{local}"],
        delete=["rclone", *extra, "deletefile", f"{base}/{{key}}"],
        missing_codes=(3, 4),
        timeout=timeout,
    )


def _retry(action: Callable[[], object], retries: int, wait: float, sleep: Callable[[float], None]) -> object:
    for attempt in range(retries + 1):
        try:
            return action()
        except FileNotFoundError:
            raise
        except OSError:
            if attempt == retries:
                raise
            sleep(wait * 2**attempt)


def _load_manifest(store, directory: Path, prefix: str, retries: int, wait: float, sleep) -> dict | None:
    path = directory / MANIFEST
    try:
        _retry(lambda: store.download(f"{prefix}/{MANIFEST}", path), retries, wait, sleep)
    except FileNotFoundError:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class CheckpointSync:
    def __init__(
        self,
        store,
        prefix: str = "checkpoints",
        keep: int = 3,
        retries: int = 3,
        wait: float = 1.0,
        scratch: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if keep < 1 or retries < 0 or wait < 0:
            raise ValueError("keep must be positive, retries and wait must not be negative")
        self.store, self.prefix, self.keep = store, prefix, keep
        self.retries, self.wait_seconds, self.sleep = retries, wait, sleep
        self.scratch = Path(scratch) if scratch is not None else None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._retained: list[dict] | None = None

    def _call(self, action: Callable[[], object]) -> object:
        return _retry(action, self.retries, self.wait_seconds, self.sleep)

    def _known(self, staging: Path) -> list[dict]:
        if self._retained is None:
            manifest = _load_manifest(self.store, staging, self.prefix, self.retries, self.wait_seconds, self.sleep)
            self._retained = [] if manifest is None else manifest["retained"]
        return self._retained

    def push(self, step: int, checkpoint: Path, extra: Mapping[str, Path] | None = None) -> None:
        self.wait()
        checkpoint = Path(checkpoint)
        staging = (self.scratch or checkpoint.parent) / ".sync"
        staging.mkdir(parents=True, exist_ok=True)
        staged = staging / f"step_{step:08d}.pt"
        staged.unlink(missing_ok=True)
        try:
            os.link(checkpoint, staged)
        except OSError:
            shutil.copyfile(checkpoint, staged)
        copies = {}
        for name, path in (extra or {}).items():
            copies[name] = staging / name
            shutil.copyfile(path, copies[name])
        self._thread = threading.Thread(target=self._work, args=(step, staged, copies, staging), daemon=True)
        self._thread.start()

    def _work(self, step: int, staged: Path, copies: dict[str, Path], staging: Path) -> None:
        try:
            key = f"{self.prefix}/{staged.name}"
            retained = [entry for entry in self._known(staging) if entry["step"] != step]
            self._call(lambda: self.store.upload(staged, key))
            files = {}
            for name, path in copies.items():
                self._call(lambda: self.store.upload(path, f"{self.prefix}/{name}"))
                files[name] = {"key": f"{self.prefix}/{name}", "size": path.stat().st_size, "sha256": sha256_of(path)}
            retained = retained + [{"step": step, "key": key}]
            dropped, retained = retained[: -self.keep], retained[-self.keep :]
            size, digest = staged.stat().st_size, sha256_of(staged)
            latest = {"step": step, "key": key, "size": size, "sha256": digest, "files": files}
            manifest = {"latest": latest, "retained": retained}
            (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            self._call(lambda: self.store.upload(staging / MANIFEST, f"{self.prefix}/{MANIFEST}"))
            self._retained = retained
            for entry in dropped:
                self._call(lambda: self.store.delete(entry["key"]))
        except BaseException as error:
            self._error = error
        finally:
            for path in [staged, *copies.values(), staging / MANIFEST]:
                path.unlink(missing_ok=True)

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    def close(self) -> None:
        self.wait()


def fetch_latest(
    store,
    directory: Path,
    prefix: str = "checkpoints",
    retries: int = 3,
    wait: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Path, dict] | None:
    directory = Path(directory)
    manifest = _load_manifest(store, directory, prefix, retries, wait, sleep)
    if manifest is None:
        return None
    latest = manifest["latest"]
    downloads = {Path(latest["key"]).name: latest, **latest["files"]}
    for name, entry in downloads.items():
        target = directory / name
        _retry(lambda: store.download(entry["key"], target), retries, wait, sleep)
        if target.stat().st_size != entry["size"] or sha256_of(target) != entry["sha256"]:
            target.unlink()
            raise ValueError(f"{entry['key']} does not match its manifest entry")
    return directory / Path(latest["key"]).name, latest
