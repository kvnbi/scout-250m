import math

import pytest
import torch

from scout.optim.adamw import CautiousAdamW
from scout.optim.normuon import NorMuon
from scout.optim.schedule import set_learning_rates, wsd_factor


def curve(total, warmup, decay, final=0.0):
    return [wsd_factor(step, total, warmup, decay, final) for step in range(total)]


def test_warmup_is_linear_and_reaches_the_peak_on_its_last_step():
    values = curve(100, 10, 20)
    assert values[:10] == pytest.approx([(i + 1) / 10 for i in range(10)])
    assert values[0] > 0 and values[9] == 1.0


def test_stable_phase_holds_the_peak():
    assert set(curve(100, 10, 20)[10:80]) == {1.0}


def test_cooldown_is_one_minus_the_square_root_of_progress():
    values = curve(100, 10, 20)
    for offset in range(20):
        assert values[80 + offset] == pytest.approx(1 - math.sqrt(offset / 20))
    assert values[80] == 1.0
    assert values[85] == pytest.approx(1 - 0.5)
    assert 0 < values[99] < 0.25


def test_cooldown_falls_faster_at_first_than_a_linear_one_would():
    values = curve(1000, 0, 400)
    assert values[600 + 40] < 1 - 40 / 400
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_final_factor_sets_the_floor_of_the_cooldown():
    values = curve(100, 0, 50, final=0.1)
    assert values[49] == 1.0 and values[50] == 1.0
    assert min(values) > 0.1
    assert values[-1] == pytest.approx(0.1 + 0.9 * (1 - math.sqrt(49 / 50)))


def test_no_warmup_and_no_decay_are_allowed():
    assert curve(5, 0, 0) == [1.0] * 5
    assert curve(6, 0, 6)[0] == 1.0
    assert curve(4, 4, 0) == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_phases_may_fill_the_whole_run():
    values = curve(20, 8, 12)
    assert values[7] == 1.0 and values[8] == 1.0
    assert values[8:] == pytest.approx([1 - math.sqrt(i / 12) for i in range(12)])


@pytest.mark.parametrize(
    "arguments",
    [(0, 0, 0, 0), (0, 10, -1, 0), (0, 10, 0, -1), (0, 10, 6, 6), (0, 10, 5, 6), (-1, 10, 1, 1), (10, 10, 1, 1), (0, 10, 1, 1, -0.1), (0, 10, 1, 1, 1.1)],
)
def test_rejects_invalid_arguments(arguments):
    with pytest.raises(ValueError):
        wsd_factor(*arguments)


def make_optimizers():
    matrix = torch.nn.Parameter(torch.zeros(4, 4))
    vector = torch.nn.Parameter(torch.zeros(4))
    return [NorMuon([matrix], lr=0.02), CautiousAdamW([vector], lr=0.003)]


def test_every_optimizer_is_scaled_from_its_own_base_rate():
    optimizers = make_optimizers()
    set_learning_rates(optimizers, 0.5)
    assert [o.param_groups[0]["lr"] for o in optimizers] == pytest.approx([0.01, 0.0015])
    set_learning_rates(optimizers, 1.0)
    assert [o.param_groups[0]["lr"] for o in optimizers] == [0.02, 0.003]


def test_repeated_scaling_does_not_compound():
    optimizers = make_optimizers()
    for factor in (0.5, 0.5, 0.25, 0.25):
        set_learning_rates(optimizers, factor)
    assert [o.param_groups[0]["lr"] for o in optimizers] == pytest.approx([0.005, 0.00075])


def test_covers_every_parameter_group():
    first, second = torch.nn.Parameter(torch.zeros(2, 2)), torch.nn.Parameter(torch.zeros(3, 3))
    optimizer = NorMuon([{"params": [first], "lr": 0.1}, {"params": [second], "lr": 0.4}], lr=0.2)
    set_learning_rates([optimizer], 0.5)
    assert [group["lr"] for group in optimizer.param_groups] == [0.05, 0.2]


def test_a_scheduled_rate_survives_a_state_dict_round_trip():
    optimizers = make_optimizers()
    set_learning_rates(optimizers, 0.25)
    restored = make_optimizers()
    for new, old in zip(restored, optimizers):
        new.load_state_dict(old.state_dict())
    set_learning_rates(restored, 0.5)
    assert [o.param_groups[0]["lr"] for o in restored] == pytest.approx([0.01, 0.0015])


def test_drives_a_training_run_through_all_three_phases():
    optimizers = make_optimizers()
    seen = []
    for step in range(30):
        set_learning_rates(optimizers, wsd_factor(step, 30, 5, 10))
        seen.append(optimizers[0].param_groups[0]["lr"])
    assert seen[0] == pytest.approx(0.02 / 5) and max(seen) == pytest.approx(0.02)
    assert seen[4] == pytest.approx(0.02) and seen[19] == pytest.approx(0.02) and seen[-1] < 0.02 / 3
