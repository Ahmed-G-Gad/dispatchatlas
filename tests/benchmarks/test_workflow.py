"""Tests of the random task graphs of Topcuoglu, Hariri and Wu (2002)."""

from __future__ import annotations

import itertools
import math
import random
from statistics import fmean

import pytest

from dispatchatlas.benchmarks import default_registry
from dispatchatlas.benchmarks.characterize import features
from dispatchatlas.benchmarks.workflow import RandomTaskGraph, _apportion, layered_graph
from dispatchatlas.core import (
    ConfigError,
    check_schedule,
    problem_digest,
    problem_from_dict,
    problem_to_dict,
)
from dispatchatlas.references import bibliography
from dispatchatlas.solvers.scheduling import Decoder

REGISTRY = default_registry(plugins=False)


# ------------------------------------------------------------ layered graph


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize(("tasks", "shape", "out_degree"), [(1, 1.0, 1), (7, 0.5, 2), (40, 2.0, 3)])
def test_layered_graphs_have_one_parent_in_the_level_above(seed, tasks, shape, out_degree):
    graph = layered_graph(random.Random(seed), tasks, shape, out_degree)
    levels = graph.levels
    assert len(levels) == tasks
    assert list(levels) == sorted(levels)
    assert set(levels) == set(range(graph.height))
    assert all(before < after for before, after in graph.edges)
    assert len(set(graph.edges)) == len(graph.edges)
    assert all(levels[after] == levels[before] + 1 for before, after in graph.edges)
    last = graph.height - 1
    assert all(
        any(before == task for before, _ in graph.edges)
        for task in range(tasks)
        if levels[task] < last
    )
    for task in range(tasks):
        if levels[task] > 0:
            assert any(
                after == task and levels[before] == levels[task] - 1
                for before, after in graph.edges
            )
    assert len(graph.edges) <= 2 * out_degree * tasks


def test_layered_graph_shape_controls_height():
    def mean_height(shape: float) -> float:
        return fmean(layered_graph(random.Random(s), 100, shape, 2).height for s in range(40))

    assert mean_height(0.5) == pytest.approx(20.0, rel=0.2)
    assert mean_height(1.0) == pytest.approx(10.0, rel=0.2)
    assert mean_height(2.0) == pytest.approx(5.0, rel=0.25)


@pytest.mark.parametrize("out_degree", [1, 2, 3])
def test_layered_graph_out_degree_has_the_given_mean_on_wide_levels(out_degree):
    degrees = []
    for seed in range(30):
        graph = layered_graph(random.Random(seed), 200, 2.0, out_degree)
        last = graph.height - 1
        degrees += [
            sum(1 for before, _ in graph.edges if before == task)
            for task in range(200)
            if graph.levels[task] < last
        ]
    assert out_degree <= fmean(degrees) <= out_degree + 0.5


def test_apportion_keeps_one_per_part_and_the_total():
    assert _apportion(10, [1.0, 1.0, 2.0]) == [3, 3, 4]
    assert _apportion(3, [5.0, 0.0, 0.0]) == [1, 1, 1]
    assert _apportion(7, [0.0, 0.0]) == [4, 3]
    for seed in range(20):
        rng = random.Random(seed)
        weights = [rng.random() for _ in range(rng.randint(1, 9))]
        parts = _apportion(50, weights)
        assert sum(parts) == 50
        assert min(parts) >= 1


def test_layered_graph_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="tasks"):
        layered_graph(random.Random(0), 0, 1.0, 1)
    with pytest.raises(ValueError, match="shape"):
        layered_graph(random.Random(0), 5, 0.0, 1)
    with pytest.raises(ValueError, match="out_degree"):
        layered_graph(random.Random(0), 5, 1.0, 0)


# ------------------------------------------------------------------- family


def test_family_metadata_is_complete():
    info = REGISTRY.info("workflow")
    assert info.references == ("topcuoglu2002performance",)
    assert all(reference in bibliography() for reference in info.references)
    assert info.objectives == ("makespan",)
    assert set(info.parameters) == {
        "tasks",
        "shape",
        "out_degree",
        "ccr",
        "heterogeneity",
        "processors",
        "mean_duration",
        "objectives",
    }


def test_generation_is_deterministic_and_round_trips():
    first = REGISTRY.generate("workflow", 4, tasks=30)
    assert first == REGISTRY.generate("workflow", 4, tasks=30)
    assert first != REGISTRY.generate("workflow", 5, tasks=30)
    assert problem_from_dict(problem_to_dict(first)) == first
    assert first.id == "workflow-30x4-4"
    assert first.metadata["seed"] == 4
    assert first.metadata["tasks"] == 30


def test_default_instances_keep_their_identity():
    # A generator change changes its instances: update the digest only on purpose.
    digest = "8dba3ded87a750fc7f2384debc69b861fccfc633f987b42f32197cd8772f3776"
    assert problem_digest(REGISTRY.generate("workflow", 1)) == digest


@pytest.mark.parametrize("processors", [1, 3, 8])
def test_every_task_runs_on_every_processor(processors):
    problem = REGISTRY.generate("workflow", 1, tasks=25, processors=processors)
    assert [resource.id for resource in problem.resources] == [f"p{j}" for j in range(processors)]
    assert all(resource.capacity == 1.0 for resource in problem.resources)
    for task in problem.tasks:
        assert [mode.location for mode in task.modes] == [f"p{j}" for j in range(processors)]


@pytest.mark.parametrize("heterogeneity", [0.0, 0.5, 1.0])
def test_durations_spread_by_the_heterogeneity_factor(heterogeneity):
    problem = REGISTRY.generate("workflow", 2, tasks=60, heterogeneity=heterogeneity)
    ratio = (1 + heterogeneity / 2) / (1 - heterogeneity / 2)
    for task in problem.tasks:
        durations = [mode.duration for mode in task.modes]
        assert max(durations) <= min(durations) * ratio + 1e-5
    mean = fmean(mode.duration for task in problem.tasks for mode in task.modes)
    assert mean == pytest.approx(10.0, rel=0.25)


@pytest.mark.parametrize("ccr", [0.1, 1.0, 10.0])
def test_communication_to_computation_ratio_is_exact(ccr):
    problem = REGISTRY.generate("workflow", 3, tasks=40, ccr=ccr)
    mean_communication = fmean(dependency.data for dependency in problem.dependencies)
    mean_duration = fmean(mode.duration for task in problem.tasks for mode in task.modes)
    assert mean_communication / mean_duration == pytest.approx(ccr, rel=1e-4)
    assert features(problem)["ccr"] == pytest.approx(ccr, rel=1e-4)


def test_transfers_cost_their_data_between_processors_only():
    problem = REGISTRY.generate("workflow", 6, tasks=12, processors=3)
    dependency = problem.dependencies[0]
    for a, b in itertools.product(("p0", "p1", "p2"), repeat=2):
        expected = 0.0 if a == b else dependency.data
        assert problem.transfer_time(dependency, a, b) == pytest.approx(expected)
    before = problem.task_index[dependency.before]
    after = problem.task_index[dependency.after]
    order = sorted(range(problem.size), key=lambda i: (i != before, i != after, i))
    modes = [0] * problem.size
    modes[after] = 1
    decoded = Decoder(problem).decode(order, modes)
    assert decoded.starts[after] >= decoded.ends[before] + dependency.data - 1e-9
    assert check_schedule(problem, decoded.schedule(problem)) == ()


def test_single_processors_and_zero_ratios_need_no_network():
    assert REGISTRY.generate("workflow", 1, processors=1).network is None
    assert REGISTRY.generate("workflow", 1, ccr=0.0).network is None
    assert REGISTRY.generate("workflow", 1).network is not None


@pytest.mark.parametrize("tasks", [20, 100, 400])
def test_graphs_stay_sparse_as_they_grow(tasks):
    problem = REGISTRY.generate("workflow", 9, tasks=tasks, out_degree=3)
    density = len(problem.dependencies) / (tasks * (tasks - 1) / 2)
    assert len(problem.dependencies) <= 2 * 3 * tasks
    assert density <= 12 / tasks


def test_default_graphs_decode_to_feasible_schedules():
    problem = RandomTaskGraph().generate(0)
    decoder = Decoder(problem)
    rng = random.Random(0)
    for _ in range(5):
        order = list(range(problem.size))
        rng.shuffle(order)
        modes = [rng.randrange(len(task.modes)) for task in problem.tasks]
        schedule = decoder.schedule(order, modes)
        assert check_schedule(problem, schedule) == ()
        assert math.isfinite(schedule.makespan)


@pytest.mark.parametrize(
    "parameters",
    [
        {"tasks": 0},
        {"shape": 0.0},
        {"out_degree": 0},
        {"ccr": -1.0},
        {"heterogeneity": 2.5},
        {"processors": 0},
        {"mean_duration": 0.0},
        {"objectives": ("total_tardiness",)},
        {"objectives": ()},
    ],
)
def test_invalid_parameters_are_refused(parameters):
    with pytest.raises(ConfigError):
        REGISTRY.create("workflow", **parameters)
