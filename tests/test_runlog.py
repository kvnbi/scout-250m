import json

import pytest
import torch

import scout.runlog
from scout.checkpoint import load_checkpoint, save_checkpoint
from scout.config import ModelConfig
from scout.evaluate import evaluate_loss
from scout.loss import IGNORE_INDEX
from scout.model import Scout
from scout.optim.groups import build_optimizers
from scout.runlog import RunLog, evaluation_logger, read_log, step_logger
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


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        self.now += 1.0
        return self.now


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_each_record_is_one_json_line_with_kind_step_time_and_the_values(tmp_path):
    with RunLog(tmp_path / "run.jsonl", clock=Clock()) as log:
        log.write("train", 1, {"loss": 4.5, "lrs": [0.1, 0.2], "tokens": 99})
        log.write("eval", 1, {"correct_probability": 0.25})
    records = [json.loads(line) for line in lines(tmp_path / "run.jsonl")]
    assert records == [
        {"kind": "train", "step": 1, "time": 1001.0, "loss": 4.5, "lrs": [0.1, 0.2], "tokens": 99},
        {"kind": "eval", "step": 1, "time": 1002.0, "correct_probability": 0.25},
    ]
    assert list(records[0])[:3] == ["kind", "step", "time"]


def test_every_record_is_on_disk_the_moment_it_is_written(tmp_path):
    log = RunLog(tmp_path / "run.jsonl")
    log.write("train", 1, {"loss": 1.0})
    assert len(lines(tmp_path / "run.jsonl")) == 1
    log.write("train", 2, {"loss": 0.5})
    assert len(lines(tmp_path / "run.jsonl")) == 2
    log.close()


def test_reopening_appends_and_creates_parent_folders(tmp_path):
    path = tmp_path / "deep" / "run.jsonl"
    for step in (1, 2):
        with RunLog(path) as log:
            log.write("train", step, {"loss": float(step)})
    assert [r["step"] for r in read_log(path)] == [1, 2]


def test_non_finite_numbers_become_null_so_the_file_stays_valid_json(tmp_path):
    with RunLog(tmp_path / "run.jsonl") as log:
        log.write("eval", 3, {"loss": float("nan"), "nested": {"a": float("inf")}, "list": [1.0, float("-inf")]})
    record = json.loads(lines(tmp_path / "run.jsonl")[0])
    assert record["loss"] is None and record["nested"] == {"a": None} and record["list"] == [1.0, None]


def test_unicode_and_unserialisable_values(tmp_path):
    with RunLog(tmp_path / "run.jsonl") as log:
        log.write("note", 0, {"text": "caf\u00e9 \u4e2d\u6587"})
        with pytest.raises(TypeError):
            log.write("train", 1, {"loss": torch.tensor(1.0)})
    assert read_log(tmp_path / "run.jsonl")[0]["text"] == "caf\u00e9 \u4e2d\u6587"
    assert len(read_log(tmp_path / "run.jsonl")) == 1


@pytest.mark.parametrize("key", ["kind", "step", "time"])
def test_reserved_keys_are_refused(tmp_path, key):
    with RunLog(tmp_path / "run.jsonl") as log, pytest.raises(ValueError, match="reserved"):
        log.write("train", 1, {key: 1})


def test_writing_after_close_fails(tmp_path):
    log = RunLog(tmp_path / "run.jsonl")
    log.close()
    with pytest.raises(ValueError):
        log.write("train", 1)


def test_read_log_ignores_a_torn_last_line_but_not_a_corrupt_middle_one(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"kind": "train", "step": 1}\n{"kind": "train", "ste', encoding="utf-8")
    assert read_log(path) == [{"kind": "train", "step": 1}]
    path.write_text('{"kind": "train", "step": 1}\nnot json\n{"kind": "train", "step": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_log(path)
    with pytest.raises(FileNotFoundError):
        read_log(tmp_path / "missing.jsonl")


def write_run(path, steps, resume_step=None):
    with RunLog(path, resume_step=resume_step, clock=Clock()) as log:
        for step in steps:
            log.write("train", step, {"loss": 1.0 / step})


def test_resuming_drops_the_steps_that_will_be_redone(tmp_path):
    path = tmp_path / "run.jsonl"
    with RunLog(path) as log:
        log.write("config", 0, {"seed": 1})
    write_run(path, range(1, 8))
    with RunLog(path, resume_step=4) as log:
        log.write("train", 5, {"loss": 9.0})
    records = read_log(path)
    assert [(r["kind"], r["step"]) for r in records] == [("config", 0), *[("train", s) for s in range(1, 5)], ("train", 5)]
    assert records[-1]["loss"] == 9.0


def test_resuming_also_drops_a_torn_line_and_leaves_a_clean_log_untouched(tmp_path):
    path = tmp_path / "run.jsonl"
    write_run(path, range(1, 4))
    with open(path, "a", encoding="utf-8") as stream:
        stream.write('{"kind": "train", "ste')
    RunLog(path, resume_step=3).close()
    assert [r["step"] for r in read_log(path)] == [1, 2, 3]
    assert path.read_text().endswith("\n")
    before, inode = path.read_bytes(), path.stat().st_ino
    RunLog(path, resume_step=3).close()
    RunLog(path, resume_step=99).close()
    RunLog(path).close()
    assert path.read_bytes() == before and path.stat().st_ino == inode
    assert not list(tmp_path.glob("*.tmp"))


def test_resuming_with_no_log_yet_starts_one(tmp_path):
    RunLog(tmp_path / "new.jsonl", resume_step=5).close()
    assert (tmp_path / "new.jsonl").read_text() == ""


def make_batches():
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


SETTINGS = TrainSettings(total_steps=8, warmup_steps=1, decay_steps=2)


def held_out(model):
    generator = torch.Generator().manual_seed(5)
    ids = torch.randint(0, 50, (2, 12), generator=generator)
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = IGNORE_INDEX
    return evaluate_loss(model, [Batch(ids, targets)])


def test_the_loop_logs_every_step_and_every_evaluation(tmp_path):
    model, optimizers = build()
    with RunLog(tmp_path / "run.jsonl", clock=Clock()) as log:
        train(model, optimizers, make_batches(), SETTINGS, on_step=step_logger(log, every=2),
              evaluate=held_out, evaluate_every=4, on_evaluate=evaluation_logger(log))
    records = read_log(tmp_path / "run.jsonl")
    train_records = [r for r in records if r["kind"] == "train"]
    eval_records = [r for r in records if r["kind"] == "eval"]
    assert [r["step"] for r in train_records] == [2, 4, 6, 8]
    assert [r["step"] for r in eval_records] == [4, 8]
    assert set(train_records[0]) == {"kind", "step", "time", "loss", "grad_norm", "tokens", "lrs"}
    assert set(eval_records[0]) == {"kind", "step", "time", "loss", "tokens"}
    assert [r["time"] for r in records] == sorted(r["time"] for r in records)


def test_an_interrupted_and_resumed_run_leaves_the_same_log_as_a_straight_run(tmp_path):
    def without_time(path):
        return [{k: v for k, v in r.items() if k != "time"} for r in read_log(path)]

    def run(log, model, optimizers, batches, **kwargs):
        train(model, optimizers, batches, SETTINGS, on_step=step_logger(log), evaluate=held_out, evaluate_every=3,
              on_evaluate=evaluation_logger(log), **kwargs)

    model, optimizers = build()
    with RunLog(tmp_path / "straight.jsonl", clock=Clock()) as log:
        run(log, model, optimizers, make_batches())

    crashing, crashing_optimizers = build()
    path = tmp_path / "resumed.jsonl"
    with RunLog(path, clock=Clock()) as log:
        logger = step_logger(log)

        def on_step(step, metrics):
            logger(step, metrics)
            if step == 4:
                save_checkpoint(tmp_path / "latest.pt", crashing, crashing_optimizers, step)

        train(crashing, crashing_optimizers, make_batches(), SETTINGS, stop_step=6, on_step=on_step, evaluate=held_out,
              evaluate_every=3, on_evaluate=evaluation_logger(log))
    assert [r["step"] for r in read_log(path) if r["kind"] == "train"] == [1, 2, 3, 4, 5, 6]

    resumed, resumed_optimizers = build()
    step, _ = load_checkpoint(tmp_path / "latest.pt", resumed, resumed_optimizers)
    assert step == 4
    batches = make_batches()
    for _ in range(step):
        next(batches)
    with RunLog(path, resume_step=step, clock=Clock()) as log:
        run(log, resumed, resumed_optimizers, batches, start_step=step)
    assert without_time(path) == without_time(tmp_path / "straight.jsonl")
    assert [r["step"] for r in read_log(path) if r["kind"] == "eval"] == [3, 6, 8]


def test_only_rank_zero_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(scout.runlog, "rank", lambda group=None: 1)
    with RunLog(tmp_path / "run.jsonl") as log:
        step_logger(log)(1, {"loss": 1.0})
    assert read_log(tmp_path / "run.jsonl") == []


def test_logging_interval_must_be_positive(tmp_path):
    with RunLog(tmp_path / "run.jsonl") as log, pytest.raises(ValueError):
        step_logger(log, every=0)
