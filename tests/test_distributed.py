from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scout.config import ModelConfig
from scout.distributed import broadcast_parameters, rank, reduce_gradients, world_size
from scout.loss import IGNORE_INDEX
from scout.masking import document_ids
from scout.model import Scout
from scout.optim.groups import build_optimizers
from scout.train import Batch, TrainSettings, train, train_step

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
EOT = TINY.reserved_token_ids.start
SHAPES = [(5,), (1000,), (3, 7), (4096,), (11, 13), (1,)]


def launch(worker, world, tmp_path, *args):
    init = f"file://{tmp_path / 'rendezvous'}"
    mp.spawn(worker, args=(world, init, str(tmp_path), *args), nprocs=world, join=True)


def join_group(index, world, init):
    dist.init_process_group("gloo", init_method=init, rank=index, world_size=world)


def rank_gradients(index):
    generator = torch.Generator().manual_seed(1000 + index)
    return [torch.randn(shape, generator=generator) * 10 ** (index - 1) for shape in SHAPES]


MISSING = (2, 2)


def reduction_worker(index, world, init, folder, bucket_bytes):
    join_group(index, world, init)
    parameters = [torch.nn.Parameter(torch.zeros(shape)) for shape in SHAPES]
    for position, (parameter, grad) in enumerate(zip(parameters, rank_gradients(index))):
        parameter.grad = None if (index, position) == MISSING else grad.clone()
    calls = []
    real = dist.all_gather
    dist.all_gather = lambda *args, **kwargs: (calls.append(1), real(*args, **kwargs))[1]
    reduce_gradients(parameters, bucket_bytes=bucket_bytes)
    dist.all_gather = real
    frozen = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    live = torch.nn.Parameter(torch.zeros(4))
    live.grad = torch.full((4,), float(index + 1))
    reduce_gradients([live, frozen])
    kept_frozen_alone = frozen.grad is None and torch.equal(live.grad, torch.full((4,), world * (world + 1) / 2))
    reduce_gradients([frozen])
    kept_frozen_alone = kept_frozen_alone and frozen.grad is None
    mixed = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))]
    try:
        reduce_gradients(mixed)
        refused = False
    except ValueError:
        refused = True
    torch.save({"grads": [p.grad for p in parameters], "calls": len(calls), "refused": refused, "frozen": kept_frozen_alone}, f"{folder}/reduced{index}.pt")
    dist.destroy_process_group()


def ordered_sum(world):
    per_rank = [rank_gradients(index) for index in range(world)]
    if MISSING[0] < world:
        per_rank[MISSING[0]][MISSING[1]] = torch.zeros(SHAPES[MISSING[1]])
    result = []
    for position in range(len(SHAPES)):
        total = per_rank[0][position]
        for index in range(1, world):
            total = total + per_rank[index][position]
        result.append(total)
    return result


def test_reduction_is_the_rank_ordered_sum_and_buckets_split_where_they_should(tmp_path):
    expected = ordered_sum(4)
    for bucket_bytes, buckets in ((16, 6), (4104, 3), (16384, 3), (64 << 20, 1)):
        folder = tmp_path / str(bucket_bytes)
        folder.mkdir()
        launch(reduction_worker, 4, folder, bucket_bytes)
        for index in range(4):
            got = torch.load(folder / f"reduced{index}.pt", weights_only=True)
            assert all(torch.equal(a, b) for a, b in zip(got["grads"], expected)), (bucket_bytes, index)
            assert got["calls"] == buckets, (bucket_bytes, index)
            assert got["refused"] and got["frozen"]


def test_the_rank_order_is_what_makes_it_exact_not_luck():
    per_rank = [rank_gradients(index) for index in range(4)]
    forward = (per_rank[0][1] + per_rank[1][1]) + per_rank[2][1] + per_rank[3][1]
    backward = (per_rank[3][1] + per_rank[2][1]) + per_rank[1][1] + per_rank[0][1]
    assert not torch.equal(forward, backward)


def rank_batch(index, number):
    generator = torch.Generator().manual_seed(100 * index + number)
    ids = torch.randint(0, 50, (2, 16), generator=generator)
    ids[:, 7] = EOT
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = IGNORE_INDEX
    targets[:, 7] = IGNORE_INDEX
    targets[0, 2 : 2 + 3 * index] = IGNORE_INDEX
    return Batch(ids, targets, document_ids(ids, EOT))


def rank_batches(index):
    number = 0
    while True:
        yield rank_batch(index, number)
        number += 1


def build(seed):
    torch.manual_seed(seed)
    model = Scout(TINY)
    return model, build_optimizers(model, 0.02, 0.01, weight_decay=0.1)


def settings(**kwargs):
    return TrainSettings(**{**dict(total_steps=4, warmup_steps=1, decay_steps=1), **kwargs})


def training_worker(index, world, init, folder, accumulate):
    join_group(index, world, init)
    model, optimizers = build(seed=index)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * index)
    broadcast_parameters(model)
    start = [p.detach().clone() for p in model.parameters()]
    seen = []
    train(model, optimizers, rank_batches(index), settings(accumulate=accumulate), on_step=lambda s, m: seen.append(m), group=dist.group.WORLD)
    torch.save({"start": start, "final": [p.detach().clone() for p in model.parameters()], "metrics": seen}, f"{folder}/trained{index}.pt")
    dist.destroy_process_group()


def checkpoint_worker(index, world, init, folder):
    import scout.train

    join_group(index, world, init)
    saved = []
    real = scout.train.save_checkpoint
    scout.train.save_checkpoint = lambda path, model, optimizers, step, extra=None: (saved.append(step), real(path, model, optimizers, step, extra))
    model, optimizers = build(seed=0)
    train(model, optimizers, rank_batches(index), settings(total_steps=3), checkpoint_path=Path(folder) / "latest.pt", checkpoint_every=2, group=dist.group.WORLD)
    torch.save(saved, f"{folder}/saved{index}.pt")
    dist.destroy_process_group()


def test_only_rank_zero_writes_checkpoints(tmp_path):
    launch(checkpoint_worker, 2, tmp_path)
    assert torch.load(tmp_path / "saved0.pt", weights_only=True) == [2, 3]
    assert torch.load(tmp_path / "saved1.pt", weights_only=True) == []
    assert sorted(path.name for path in tmp_path.glob("*.pt*") if "latest" in path.name) == ["latest.pt"]


def interleaved(world):
    number = 0
    while True:
        for index in range(world):
            yield rank_batch(index, number)
        number += 1


def test_four_ranks_equal_one_process_accumulating_the_same_batches_in_rank_order(tmp_path):
    launch(training_worker, 4, tmp_path, 1)
    results = [torch.load(tmp_path / f"trained{index}.pt", weights_only=True) for index in range(4)]
    reference, optimizers = build(seed=0)
    log = []
    train(reference, optimizers, interleaved(4), settings(accumulate=4), on_step=lambda s, m: log.append(m))
    first_start = [p.detach() for p in build(seed=0)[0].parameters()]
    for result in results:
        assert all(torch.equal(a, b) for a, b in zip(result["start"], first_start))
        assert all(torch.equal(a, b) for a, b in zip(result["final"], results[0]["final"]))
        assert all(torch.equal(a, b) for a, b in zip(result["final"], reference.parameters()))
        assert all(abs(a["loss"] - b["loss"]) < 1e-5 for a, b in zip(result["metrics"], log))
        assert [m["tokens"] for m in result["metrics"]] == [m["tokens"] for m in log]


def test_ranks_with_local_accumulation_stay_identical_to_each_other_and_reproducible(tmp_path):
    runs = []
    for number in range(2):
        folder = tmp_path / str(number)
        folder.mkdir()
        launch(training_worker, 3, folder, 2)
        runs.append([torch.load(folder / f"trained{index}.pt", weights_only=True)["final"] for index in range(3)])
    for finals in runs:
        assert all(all(torch.equal(a, b) for a, b in zip(finals[0], other)) for other in finals[1:])
    assert all(torch.equal(a, b) for a, b in zip(runs[0][0], runs[1][0]))
    assert all(torch.isfinite(p).all() for p in runs[0][0])


def test_a_single_process_is_untouched_by_the_distributed_code():
    assert (world_size(), rank()) == (1, 0)
    model, optimizers = build(seed=0)
    before = [p.detach().clone() for p in model.parameters()]
    broadcast_parameters(model)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    parameters = list(model.parameters())
    for p in parameters:
        p.grad = torch.ones_like(p)
    reduce_gradients(parameters)
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in parameters)
    result = train_step(model, optimizers, [rank_batch(0, 0)], settings())
    assert result["tokens"] == int((rank_batch(0, 0).targets != IGNORE_INDEX).sum())
