import random

import pytest
import torch

from scout.checkpoint import FORMAT, load_checkpoint, random_state, restore_random_state, save_checkpoint
from scout.config import ModelConfig
from scout.loss import IGNORE_INDEX
from scout.model import Scout
from scout.optim.groups import build_optimizers

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


def setup(seed=0, config=TINY):
    torch.manual_seed(seed)
    model = Scout(config)
    return model, build_optimizers(model, matrix_lr=0.02, other_lr=0.01)


def train(model, optimizers, steps, seed=5):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ids = torch.randint(0, 64, (2, 16), generator=generator)
        targets = torch.roll(ids, -1, 1)
        targets[:, -1] = IGNORE_INDEX
        for optimizer in optimizers:
            optimizer.zero_grad()
        model.loss(ids, targets).backward()
        for optimizer in optimizers:
            optimizer.step()


def same_state(first, second):
    return all(torch.equal(value, second[key]) for key, value in first.items()) and first.keys() == second.keys()


def test_round_trip_restores_weights_optimizer_state_step_and_extra(tmp_path):
    model, optimizers = setup()
    train(model, optimizers, 3)
    save_checkpoint(tmp_path / "a" / "step.pt", model, optimizers, 3, {"loader": {"shard": 4, "offset": 99}})
    restored, restored_optimizers = setup(seed=1)
    step, extra = load_checkpoint(tmp_path / "a" / "step.pt", restored, restored_optimizers)
    assert step == 3 and extra == {"loader": {"shard": 4, "offset": 99}}
    assert same_state(model.state_dict(), restored.state_dict())
    for old, new in zip(optimizers, restored_optimizers):
        old_state, new_state = old.state_dict(), new.state_dict()
        assert old_state["param_groups"] == new_state["param_groups"]
        for key, value in old_state["state"].items():
            assert same_state({k: torch.as_tensor(v) for k, v in value.items()}, {k: torch.as_tensor(v) for k, v in new_state["state"][key].items()})


def test_training_continues_identically_after_loading(tmp_path):
    model, optimizers = setup()
    train(model, optimizers, 2, seed=1)
    save_checkpoint(tmp_path / "c.pt", model, optimizers, 2)
    train(model, optimizers, 3, seed=2)
    restored, restored_optimizers = setup(seed=9)
    load_checkpoint(tmp_path / "c.pt", restored, restored_optimizers)
    train(restored, restored_optimizers, 3, seed=2)
    assert same_state(model.state_dict(), restored.state_dict())


def test_tied_embeddings_stay_tied_and_the_file_stores_them_once(tmp_path):
    model, optimizers = setup()
    save_checkpoint(tmp_path / "t.pt", model, optimizers, 0)
    restored, restored_optimizers = setup(seed=3)
    load_checkpoint(tmp_path / "t.pt", restored, restored_optimizers)
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    saved = torch.load(tmp_path / "t.pt", weights_only=True)["model"]
    head, embedding = saved["lm_head.weight"], saved["model.embed_tokens.weight"]
    assert head.untyped_storage().data_ptr() == embedding.untyped_storage().data_ptr()


def test_random_number_streams_continue_where_they_stopped(tmp_path):
    model, optimizers = setup()
    random.seed(11)
    torch.manual_seed(12)
    random.random(), torch.rand(3)
    save_checkpoint(tmp_path / "r.pt", model, optimizers, 1)
    expected = (random.random(), torch.rand(4), torch.randint(0, 100, (5,)), torch.randn(2))
    random.seed(0)
    torch.manual_seed(0)
    load_checkpoint(tmp_path / "r.pt", *setup(seed=4))
    got = (random.random(), torch.rand(4), torch.randint(0, 100, (5,)), torch.randn(2))
    assert got[0] == expected[0]
    assert all(torch.equal(a, b) for a, b in zip(got[1:], expected[1:]))


def test_random_state_round_trips_without_a_gpu():
    state = random_state()
    first = (random.random(), torch.rand(3))
    restore_random_state(state)
    second = (random.random(), torch.rand(3))
    assert first[0] == second[0] and torch.equal(first[1], second[1])
    assert state["cuda"] == [] or torch.cuda.is_available()


def test_a_failed_save_leaves_the_previous_checkpoint_and_no_temporary_file(tmp_path, monkeypatch):
    model, optimizers = setup()
    path = tmp_path / "keep.pt"
    save_checkpoint(path, model, optimizers, 1)
    before = path.read_bytes()

    def broken(checkpoint, target):
        target.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError):
        save_checkpoint(path, model, optimizers, 2)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.pt"]


def test_saving_again_replaces_the_file(tmp_path):
    model, optimizers = setup()
    save_checkpoint(tmp_path / "s.pt", model, optimizers, 1)
    save_checkpoint(tmp_path / "s.pt", model, optimizers, 2)
    assert load_checkpoint(tmp_path / "s.pt", *setup(seed=2))[0] == 2
    assert [p.name for p in tmp_path.iterdir()] == ["s.pt"]


def test_loading_into_a_different_model_fails_loudly(tmp_path):
    model, optimizers = setup()
    save_checkpoint(tmp_path / "m.pt", model, optimizers, 1)
    other = ModelConfig(vocab_size=64, reserved_slots=8, d_model=64, n_layers=3, n_query_heads=4, n_kv_heads=2, head_dim=16, ffn_hidden=128)
    with pytest.raises(RuntimeError):
        load_checkpoint(tmp_path / "m.pt", *setup(config=other))


def test_the_optimizer_count_must_match(tmp_path):
    model, optimizers = setup()
    save_checkpoint(tmp_path / "o.pt", model, optimizers, 1)
    with pytest.raises(ValueError, match="2 optimizers"):
        load_checkpoint(tmp_path / "o.pt", model, optimizers[:1])


def test_the_format_number_changes_only_on_purpose():
    assert FORMAT == 1


def test_an_unknown_format_is_refused(tmp_path):
    model, optimizers = setup()
    save_checkpoint(tmp_path / "f.pt", model, optimizers, 1)
    checkpoint = torch.load(tmp_path / "f.pt", weights_only=True)
    checkpoint["format"] = FORMAT + 1
    torch.save(checkpoint, tmp_path / "f.pt")
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(tmp_path / "f.pt", model, optimizers)


def test_checkpoints_never_unpickle_arbitrary_objects(tmp_path):
    model, optimizers = setup()
    save_checkpoint(tmp_path / "x.pt", model, optimizers, 1, {"bad": object()})
    with pytest.raises(Exception):
        load_checkpoint(tmp_path / "x.pt", model, optimizers)


def test_bfloat16_models_round_trip_exactly(tmp_path):
    model, _ = setup()
    model = model.to(torch.bfloat16)
    optimizers = build_optimizers(model, matrix_lr=0.02, other_lr=0.01)
    save_checkpoint(tmp_path / "b.pt", model, optimizers, 1)
    restored, restored_optimizers = setup(seed=6)
    restored = restored.to(torch.bfloat16)
    restored_optimizers = build_optimizers(restored, matrix_lr=0.02, other_lr=0.01)
    load_checkpoint(tmp_path / "b.pt", restored, restored_optimizers)
    assert same_state(model.state_dict(), restored.state_dict())
