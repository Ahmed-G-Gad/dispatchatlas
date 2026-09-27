from __future__ import annotations

import math
import random
from statistics import fmean

import pytest

from dispatchatlas.benchmarks import default_registry
from dispatchatlas.benchmarks.machines import (
    VALLADA_RUIZ_SETUPS,
    ParallelMachines,
    parallel_machine_problem,
)
from dispatchatlas.core import ConfigError, ModelError, check_schedule
from dispatchatlas.solvers.operators import random_order
from dispatchatlas.solvers.scheduling import Decoder

REGISTRY = default_registry(plugins=False)


def test_unrelated_machines_structure_and_draws():
    problem = ParallelMachines(jobs=6, machines=3).generate(4)
    rng = random.Random(4)
    expected = [[rng.randint(1, 99) for _ in range(3)] for _ in range(6)]
    assert [t.id for t in problem.tasks] == [f"j{j}" for j in range(6)]
    assert [r.id for r in problem.resources] == ["m0", "m1", "m2"]
    for task, row in zip(problem.tasks, expected, strict=True):
        assert [(dict(m.demands), m.duration) for m in task.modes] == [
            ({f"m{k}": 1.0}, float(time)) for k, time in enumerate(row)
        ]
        assert task.group is None
        assert task.due is None
        assert task.release == 0.0
    assert problem.setups is None
    assert problem.features == {"modes"}


@pytest.mark.parametrize("largest", VALLADA_RUIZ_SETUPS)
def test_setups_follow_vallada_and_ruiz(largest):
    problem = ParallelMachines(jobs=7, machines=2, max_setup=largest).generate(1)
    assert problem.setups is not None
    times = problem.setups.times
    assert len(times) == 2 * 7 * 6
    assert all(previous is not None and previous != following for _, previous, following in times)
    assert all(1 <= value <= largest for value in times.values())
    assert problem.id == f"parallel-machines-7x2-s{largest}-1"


def test_setups_are_drawn_by_machine_then_job_pair():
    problem = ParallelMachines(jobs=3, machines=2, max_setup=9).generate(8)
    assert problem.setups is not None
    rng = random.Random(8)
    for _ in range(3 * 2):
        rng.randint(1, 99)
    for k in range(2):
        for a in range(3):
            for b in range(3):
                if a != b:
                    assert problem.setups.get(f"m{k}", f"j{a}", f"j{b}") == rng.randint(1, 9)


def test_schedules_with_setups_are_feasible():
    problem = ParallelMachines(jobs=10, machines=3, max_setup=124).generate(2)
    decoder = Decoder(problem)
    rng = random.Random(0)
    for _ in range(5):
        assert check_schedule(problem, decoder.schedule(random_order(problem, rng))) == ()


def test_release_dates_spread_over_the_mean_load():
    problem = ParallelMachines(jobs=30, machines=4, release_span=0.5).generate(3)
    load = sum(fmean(m.duration for m in task.modes) for task in problem.tasks) / 4
    latest = math.floor(0.5 * load + 1e-9)
    releases = [task.release for task in problem.tasks]
    assert all(0 <= release <= latest for release in releases)
    assert len(set(releases)) > 1
    assert "release" in problem.features


@pytest.mark.parametrize(("factor", "spread"), [(0.2, 0.2), (0.6, 1.0), (1.0, 0.6)])
def test_due_dates_follow_the_tardiness_factor_scheme(factor, spread):
    generator = ParallelMachines(
        jobs=40, machines=3, release_span=0.3, tardiness_factor=factor, due_date_range=spread
    )
    problem = generator.generate(5)
    load = sum(fmean(m.duration for m in task.modes) for task in problem.tasks) / 3
    low = max(0, math.floor(load * (1 - factor - spread / 2) + 0.5))
    high = math.floor(load * (1 - factor + spread / 2) + 0.5)
    for task in problem.tasks:
        assert task.due is not None
        assert low <= task.due - task.release <= high


def test_weights_are_between_one_and_ten():
    problem = ParallelMachines(jobs=50, weighted=True).generate(0)
    weights = [task.weight for task in problem.tasks]
    assert set(weights) <= set(range(1, 11))
    assert len(set(weights)) > 3


def test_weighted_tardiness_problem():
    problem = REGISTRY.generate(
        "parallel-machines",
        0,
        tardiness_factor=0.4,
        weighted=True,
        objectives=("total_tardiness",),
    )
    assert problem.objectives == ("total_tardiness",)
    assert problem.id == "parallel-machines-20x5-tf0.4-w-total_tardiness-0"


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"max_setup": -1}, "max_setup"),
        ({"release_span": -0.5}, "release_span"),
        ({"tardiness_factor": 1.5}, "tardiness_factor"),
        ({"due_date_range": 0.3}, "needs tardiness_factor"),
        ({"due_date_range": -1.0, "tardiness_factor": 0.5}, "due_date_range"),
        ({"objectives": ("tardy_tasks",)}, "tardiness_factor"),
    ],
)
def test_invalid_parameters_are_refused(parameters, message):
    with pytest.raises(ConfigError, match=message):
        REGISTRY.create("parallel-machines", **parameters)


def test_builder_places_setups_between_different_jobs_only():
    problem = parallel_machine_problem(
        [[2.0, 3.0], [4.0, 1.0]],
        setups=[[[9.0, 1.0], [2.0, 9.0]], [[0.0, 3.0], [4.0, 0.0]]],
        releases=[0.0, 5.0],
    )
    assert problem.setups is not None
    assert dict(problem.setups.times) == {
        ("m0", "j0", "j1"): 1.0,
        ("m0", "j1", "j0"): 2.0,
        ("m1", "j0", "j1"): 3.0,
        ("m1", "j1", "j0"): 4.0,
    }
    assert problem.task("j1").release == 5.0


@pytest.mark.parametrize(
    "setups",
    [[[[0.0, 1.0], [1.0, 0.0]]], [[[0.0]], [[0.0]]], [[[0.0, 1.0]], [[0.0, 1.0]]]],
)
def test_builder_refuses_mismatched_setup_tables(setups):
    with pytest.raises(ModelError, match="setups"):
        parallel_machine_problem([[2.0, 3.0], [4.0, 1.0]], setups=setups)
