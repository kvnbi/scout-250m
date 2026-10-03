import json
import threading

import pytest
import torch

from scout.checkpoint import load_checkpoint
from scout.config import ModelConfig
from scout.data.fetch import sha256_of
from scout.loss import IGNORE_INDEX
from scout.model import Scout
from scout.optim.groups import build_optimizers
from scout.runlog import RunLog, step_logger
from scout.sync import MANIFEST, CheckpointSync, CommandStore, DirectoryStore, StoreError, fetch_latest
from scout.train import Batch, TrainSettings, train

TINY = ModelConfig(
    vocab_size=64,
    reserved_slots=8,
    d_model=64,
    n_layers=2,
    n_query_heads=4,
    n_kv_heads=2,
    head_dim=16,
    ffn_hidden=128,
    context=32,
)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def command_store(root):
    return CommandStore(
        upload=["sh", "-c", f'mkdir -p "$(dirname "{root}/{{key}}")" && cp "{{local}}" "{root}/{{key}}"'],
        download=["cp", f"{root}/{{key}}", "{local}"],
        delete=["rm", "-f", f"{root}/{{key}}"],
        missing_codes=(1,),
    )


@pytest.fixture(params=["directory", "command"])
def store(request, tmp_path):
    root = tmp_path / "bucket"
    root.mkdir()
    return DirectoryStore(root) if request.param == "directory" else command_store(root)


def test_both_stores_round_trip_overwrite_and_delete(store, tmp_path):
    source = write(tmp_path / "in" / "a.bin", b"first")
    store.upload(source, "deep/path/a.bin")
    store.download("deep/path/a.bin", tmp_path / "out" / "a.bin")
    assert (tmp_path / "out" / "a.bin").read_bytes() == b"first"
    store.upload(write(tmp_path / "in" / "b.bin", b"second"), "deep/path/a.bin")
    store.download("deep/path/a.bin", tmp_path / "out" / "again.bin")
    assert (tmp_path / "out" / "again.bin").read_bytes() == b"second"
    store.delete("deep/path/a.bin")
    store.delete("deep/path/a.bin")
    with pytest.raises(FileNotFoundError):
        store.download("deep/path/a.bin", tmp_path / "out" / "gone.bin")


def test_directory_store_never_leaves_a_partial_file(tmp_path):
    store = DirectoryStore(tmp_path / "bucket")
    store.upload(write(tmp_path / "a.bin", b"x" * 100), "k/a.bin")
    assert [p.name for p in (tmp_path / "bucket" / "k").iterdir()] == ["a.bin"]


def test_a_missing_executable_or_a_failing_command_is_an_error_not_a_missing_object(tmp_path):
    local = write(tmp_path / "a.bin", b"x")
    broken = CommandStore(["no-such-binary-xyz", "{local}", "{key}"], ["no-such-binary-xyz"], ["no-such-binary-xyz"])
    with pytest.raises(StoreError, match="not installed"):
        broken.upload(local, "k")
    with pytest.raises(StoreError):
        broken.download("k", tmp_path / "out")
    failing = CommandStore(["false"], ["false"], ["false"])
    with pytest.raises(StoreError, match="code 1"):
        failing.download("k", tmp_path / "out")
    assert not isinstance(StoreError("x"), FileNotFoundError)
    slow = CommandStore(["sleep", "5"], ["sleep", "5"], ["sleep", "5"], timeout=0.2)
    with pytest.raises(StoreError, match="timed out"):
        slow.delete("k")


def make_checkpoint(directory, step, size=1000):
    return write(directory / "latest.pt", bytes([step % 256]) * size)


def push_all(sync, directory, steps, extra=None):
    for step in steps:
        sync.push(step, make_checkpoint(directory, step), extra)
        sync.wait()


def test_a_push_uploads_the_checkpoint_the_extra_files_and_a_manifest_last(store, tmp_path):
    local = tmp_path / "run"
    log = write(local / "run.jsonl", b'{"step": 1}\n')
    sync = CheckpointSync(store, keep=3)
    push_all(sync, local, [100], {"run.jsonl": log})
    store.download("checkpoints/latest.json", tmp_path / "m.json")
    manifest = json.loads((tmp_path / "m.json").read_text())
    expected = make_checkpoint(local, 100)
    assert manifest["latest"] == {
        "step": 100,
        "key": "checkpoints/step_00000100.pt",
        "size": 1000,
        "sha256": sha256_of(expected),
        "files": {"run.jsonl": {"key": "checkpoints/run.jsonl", "size": 12, "sha256": sha256_of(log)}},
    }
    assert manifest["retained"] == [{"step": 100, "key": "checkpoints/step_00000100.pt"}]
    assert not (local / ".sync").exists() or not list((local / ".sync").iterdir())


def test_only_the_newest_checkpoints_are_kept_and_the_older_ones_are_deleted(store, tmp_path):
    sync = CheckpointSync(store, keep=2)
    push_all(sync, tmp_path / "run", [10, 20, 30, 40])
    present = []
    for step in (10, 20, 30, 40):
        try:
            store.download(f"checkpoints/step_{step:08d}.pt", tmp_path / f"d{step}")
            present.append(step)
        except FileNotFoundError:
            pass
    assert present == [30, 40]
    store.download("checkpoints/latest.json", tmp_path / "m.json")
    assert [e["step"] for e in json.loads((tmp_path / "m.json").read_text())["retained"]] == [30, 40]


def test_retention_carries_on_across_a_restart(tmp_path):
    store = DirectoryStore(tmp_path / "bucket")
    push_all(CheckpointSync(store, keep=2), tmp_path / "run", [10, 20])
    push_all(CheckpointSync(store, keep=2), tmp_path / "run", [30])
    keys = sorted(p.name for p in (tmp_path / "bucket" / "checkpoints").iterdir())
    assert keys == ["latest.json", "step_00000020.pt", "step_00000030.pt"]


def test_rolling_back_to_an_earlier_step_never_deletes_the_newest_checkpoint(tmp_path):
    store = DirectoryStore(tmp_path / "bucket")
    push_all(CheckpointSync(store, keep=2), tmp_path / "run", [20, 30])
    push_all(CheckpointSync(store, keep=2), tmp_path / "run", [5])
    path, latest = fetch_latest(store, tmp_path / "restored")
    assert latest["step"] == 5 and path.read_bytes() == make_checkpoint(tmp_path / "run", 5).read_bytes()
    manifest = json.loads((tmp_path / "bucket" / "checkpoints" / "latest.json").read_text())
    assert [e["step"] for e in manifest["retained"]] == [30, 5]
    push_all(CheckpointSync(store, keep=1), tmp_path / "run", [7])
    assert fetch_latest(store, tmp_path / "again")[1]["step"] == 7
    assert sorted(p.name for p in (tmp_path / "bucket" / "checkpoints").iterdir()) == ["latest.json", "step_00000007.pt"]


def test_a_missing_manifest_means_a_fresh_start_at_once_but_a_failing_store_is_retried_then_raised(tmp_path):
    waits = []
    assert fetch_latest(DirectoryStore(tmp_path / "empty"), tmp_path / "out", sleep=waits.append) is None
    assert waits == []

    class Down(DirectoryStore):
        def download(self, key, local):
            raise StoreError("network is down")

    with pytest.raises(StoreError):
        fetch_latest(Down(tmp_path / "bucket"), tmp_path / "out", retries=2, wait=0.5, sleep=waits.append)
    assert waits == [0.5, 1.0]


def test_repushing_a_step_replaces_it_without_duplicating(tmp_path):
    store = DirectoryStore(tmp_path / "bucket")
    sync = CheckpointSync(store, keep=3)
    push_all(sync, tmp_path / "run", [10, 20])
    sync.push(20, write(tmp_path / "run" / "latest.pt", b"newer"))
    sync.wait()
    manifest = json.loads((tmp_path / "bucket" / "checkpoints" / "latest.json").read_text())
    assert [e["step"] for e in manifest["retained"]] == [10, 20]
    assert (tmp_path / "bucket" / "checkpoints" / "step_00000020.pt").read_bytes() == b"newer"


def test_fetch_latest_restores_byte_identical_files_and_none_for_an_empty_bucket(store, tmp_path):
    assert fetch_latest(store, tmp_path / "fresh") is None
    local = tmp_path / "run"
    log = write(local / "run.jsonl", b"log line\n")
    push_all(CheckpointSync(store), local, [5, 7], {"run.jsonl": log})
    path, latest = fetch_latest(store, tmp_path / "new machine")
    assert latest["step"] == 7 and path == tmp_path / "new machine" / "step_00000007.pt"
    assert path.read_bytes() == make_checkpoint(local, 7).read_bytes()
    assert (tmp_path / "new machine" / "run.jsonl").read_bytes() == b"log line\n"


def test_fetch_latest_refuses_a_corrupt_or_truncated_object(tmp_path):
    store = DirectoryStore(tmp_path / "bucket")
    push_all(CheckpointSync(store), tmp_path / "run", [5])
    target = tmp_path / "bucket" / "checkpoints" / "step_00000005.pt"
    original = target.read_bytes()
    target.write_bytes(b"\x00" + original[1:])
    with pytest.raises(ValueError, match="manifest"):
        fetch_latest(store, tmp_path / "a")
    assert not (tmp_path / "a" / "step_00000005.pt").exists()
    target.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="manifest"):
        fetch_latest(store, tmp_path / "b")


class Flaky(DirectoryStore):
    def __init__(self, root, fail=lambda key, count: False):
        super().__init__(root)
        self.fail, self.attempts = fail, {}

    def upload(self, local, key):
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if self.fail(key, self.attempts[key]):
            raise StoreError(f"upload of {key} failed")
        super().upload(local, key)


def test_an_interrupted_upload_never_replaces_the_previous_latest_checkpoint(tmp_path):
    for failing in ("step_00000200.pt", MANIFEST):
        root = tmp_path / failing
        store = Flaky(root, lambda key, count, failing=failing: key.endswith(failing) and count > (1 if failing == MANIFEST else 0))
        sync = CheckpointSync(store, retries=1, wait=0.0, sleep=lambda seconds: None)
        push_all(sync, tmp_path / "run", [100])
        sync.push(200, make_checkpoint(tmp_path / "run", 200))
        with pytest.raises(StoreError):
            sync.wait()
        path, latest = fetch_latest(DirectoryStore(root), tmp_path / f"restored-{failing}")
        assert latest["step"] == 100 and path.read_bytes() == make_checkpoint(tmp_path / "run", 100).read_bytes()


def test_transient_failures_are_retried_with_growing_waits(tmp_path):
    store = Flaky(tmp_path / "bucket", lambda key, count: count <= 2)
    waits = []
    sync = CheckpointSync(store, retries=3, wait=0.5, sleep=waits.append)
    push_all(sync, tmp_path / "run", [1])
    assert fetch_latest(DirectoryStore(tmp_path / "bucket"), tmp_path / "out")[1]["step"] == 1
    assert waits[:2] == [0.5, 1.0]


def test_a_permanent_failure_surfaces_on_the_next_push_and_then_clears(tmp_path):
    store = Flaky(tmp_path / "bucket", lambda key, count: True)
    sync = CheckpointSync(store, retries=0, sleep=lambda seconds: None)
    sync.push(1, make_checkpoint(tmp_path / "run", 1))
    with pytest.raises(StoreError):
        sync.push(2, make_checkpoint(tmp_path / "run", 2))
    store.fail = lambda key, count: False
    sync.push(3, make_checkpoint(tmp_path / "run", 3))
    sync.close()
    assert fetch_latest(DirectoryStore(tmp_path / "bucket"), tmp_path / "out")[1]["step"] == 3


class Blocking(DirectoryStore):
    def __init__(self, root):
        super().__init__(root)
        self.started, self.release = threading.Event(), threading.Event()

    def upload(self, local, key):
        self.started.set()
        assert self.release.wait(10)
        super().upload(local, key)


def test_pushing_does_not_wait_for_the_upload_and_the_snapshot_survives_the_next_save(tmp_path):
    store = Blocking(tmp_path / "bucket")
    sync = CheckpointSync(store)
    checkpoint = make_checkpoint(tmp_path / "run", 1)
    sync.push(1, checkpoint)
    assert store.started.wait(10)
    partial = checkpoint.with_name("replacement")
    partial.write_bytes(b"the next checkpoint")
    partial.replace(checkpoint)
    store.release.set()
    sync.wait()
    assert (tmp_path / "bucket" / "checkpoints" / "step_00000001.pt").read_bytes() == bytes([1]) * 1000
    assert checkpoint.read_bytes() == b"the next checkpoint"
    assert not list((tmp_path / "run" / ".sync").iterdir())


def test_a_second_push_waits_for_the_first_so_one_upload_is_ever_in_flight(tmp_path):
    store = Blocking(tmp_path / "bucket")
    sync = CheckpointSync(store)
    sync.push(1, make_checkpoint(tmp_path / "run", 1))
    assert store.started.wait(10)
    finished = threading.Event()

    def second():
        sync.push(2, make_checkpoint(tmp_path / "run", 2))
        finished.set()

    thread = threading.Thread(target=second)
    thread.start()
    assert not finished.wait(0.3)
    store.release.set()
    thread.join(10)
    assert finished.is_set()
    sync.close()


@pytest.mark.parametrize("kwargs", [dict(keep=0), dict(retries=-1), dict(wait=-1.0)])
def test_rejects_invalid_settings(tmp_path, kwargs):
    with pytest.raises(ValueError):
        CheckpointSync(DirectoryStore(tmp_path), **kwargs)


def batches():
    generator = torch.Generator().manual_seed(11)
    while True:
        ids = torch.randint(0, 50, (2, 12), generator=generator)
        targets = torch.roll(ids, -1, 1)
        targets[:, -1] = IGNORE_INDEX
        yield Batch(ids, targets)


def build():
    torch.manual_seed(0)
    model = Scout(TINY)
    return model, build_optimizers(model, 0.02, 0.01)


SETTINGS = TrainSettings(total_steps=9, warmup_steps=1, decay_steps=3)


def test_the_loop_reports_each_saved_checkpoint_to_the_hook(tmp_path):
    model, optimizers = build()
    seen = []
    train(model, optimizers, batches(), SETTINGS, stop_step=7, checkpoint_path=tmp_path / "ck.pt", checkpoint_every=3,
          on_checkpoint=lambda step, path: seen.append((step, path, path.exists())))
    assert seen == [(3, tmp_path / "ck.pt", True), (6, tmp_path / "ck.pt", True), (7, tmp_path / "ck.pt", True)]


def test_training_survives_losing_the_machine(tmp_path):
    straight, straight_optimizers = build()
    train(straight, straight_optimizers, batches(), SETTINGS)

    bucket = DirectoryStore(tmp_path / "bucket")
    first_machine = tmp_path / "machine one"
    model, optimizers = build()
    sync = CheckpointSync(bucket, keep=2)
    log_path = first_machine / "run.jsonl"
    with RunLog(log_path) as log:
        train(model, optimizers, batches(), SETTINGS, stop_step=7, on_step=step_logger(log),
              checkpoint_path=first_machine / "ck.pt", checkpoint_every=3,
              on_checkpoint=lambda step, path: sync.push(step, path, {"run.jsonl": log_path}))
    sync.close()
    assert [e["step"] for e in json.loads((tmp_path / "bucket" / "checkpoints" / "latest.json").read_text())["retained"]] == [6, 7]

    second_machine = tmp_path / "machine two"
    restored = fetch_latest(bucket, second_machine)
    assert restored is not None
    path, latest = restored
    resumed, resumed_optimizers = build()
    step, _ = load_checkpoint(path, resumed, resumed_optimizers)
    assert step == latest["step"] == 7
    stream = batches()
    for _ in range(step):
        next(stream)
    with RunLog(second_machine / "run.jsonl", resume_step=step) as log:
        train(resumed, resumed_optimizers, stream, SETTINGS, start_step=step, on_step=step_logger(log))
    assert all(torch.equal(a, b) for a, b in zip(straight.parameters(), resumed.parameters()))
    assert [json.loads(line)["step"] for line in (second_machine / "run.jsonl").read_text().splitlines()] == list(range(1, 10))
