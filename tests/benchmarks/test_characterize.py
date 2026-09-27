"""Tests of the instance features."""

from __future__ import annotations

import math

import pytest

from dispatchatlas.benchmarks import default_registry
from dispatchatlas.benchmarks.characterize import DEFINITIONS, earliest_finish_times, features
from dispatchatlas.core import Dependency, Link, Mode, Network, Problem, Resource, Task

REGISTRY = default_registry(plugins=False)


def hand_problem() -> Problem:
    """Three tasks whose features are computed by hand in the tests below."""
    return Problem(
        "hand",
        tasks=(
            Task("t0", (Mode(2, {"m1": 1}), Mode(4, {"m2": 1})), group="g"),
            Task("t1", (Mode(3, {"m1": 1}),), release=1, due=10, group="g"),
            Task("t2", (Mode(1, {"pool": 2}),), release=2, deadline=8),
        ),
        resources=(
            Resource("m1", tier="a"),
            Resource("m2", tier="b"),
            Resource("pool", capacity=4),
        ),
        dependencies=(Dependency("t0", "t1", lag=0.5, data=4),),
        network=Network((Link("a", "b", bandwidth=2.0, latency=1.0),)),
    )


def test_earliest_finish_times_follow_the_relaxation():
    # t0 finishes at 2 on m1; t1 waits for the lag on m1 (no transfer); t2 at release 2.
    assert earliest_finish_times(hand_problem()) == (2.0, 5.5, 3.0)


def test_features_of_a_hand_computed_problem():
    values = features(hand_problem())
    expected = {
        "tasks": 3,
        "resources": 3,
        "dependencies": 1,
        "jobs": 2,
        "objectives": 1,
        "dependency_density": 1 / 3,
        "depth": 2,
        "critical_path": 5.5,
        "total_work": 6,
        "parallelism": 6 / 5.5,
        "mean_duration": 7 / 3,
        "duration_cv": math.sqrt(8 / 9) / (7 / 3),
        "mean_modes": 4 / 3,
        "mode_spread": 0.5,
        "pinned_share": 2 / 3,
        "capacity_mean": 2,
        "capacity_max": 4,
        "unary_share": 2 / 3,
        "setup_share": 0,
        "area_bound": 7 / 6,
        "ccr": 3 / (7 / 3),
        "data_share": 1,
        "release_span": 2,
        "release_spread": 2 / (7 / 6),
        "due_share": 1 / 3,
        "due_tightness": 10 / 5.5,
        "deadline_share": 1 / 3,
        "deadline_tightness": 6,
    }
    assert values == pytest.approx(expected)


def test_features_follow_the_definitions_in_order():
    values = features(hand_problem())
    assert list(values) == list(DEFINITIONS)
    assert all(type(value) is float for value in values.values())
    assert all(definition.endswith(".") for definition in DEFINITIONS.values())


def test_a_single_task_has_trivial_structure():
    problem = Problem("one", (Task.on("t", "m", 4),), (Resource("m"),))
    values = features(problem)
    assert values["dependency_density"] == 0.0
    assert values["depth"] == 1.0
    assert values["parallelism"] == 1.0
    assert values["ccr"] == 0.0
    assert values["pinned_share"] == 1.0
    assert values["due_tightness"] == 0.0
    assert values["duration_cv"] == 0.0


def test_zero_durations_give_zero_ratios():
    problem = Problem("zero", (Task.on("t", "m", 0),), (Resource("m"),))
    values = features(problem)
    assert values["parallelism"] == 0.0
    assert values["release_spread"] == 0.0
    assert all(math.isfinite(value) for value in values.values())


@pytest.mark.parametrize("family", REGISTRY.ids())
def test_features_of_every_family_are_finite_and_deterministic(family):
    problem = REGISTRY.generate(family, 1)
    values = features(problem)
    assert values == features(REGISTRY.generate(family, 1))
    assert all(math.isfinite(value) and value >= 0.0 for value in values.values())
    assert values["tasks"] == problem.size
    assert 0.0 <= values["pinned_share"] <= 1.0
    assert 0.0 < values["dependency_density"] < 1.0 or not problem.dependencies


@pytest.mark.parametrize(
    ("family", "applications"),
    [("continuum-offloading", 20), ("continuum-workflow", 4), ("continuum-pipeline", 20)],
)
def test_continuum_features_recover_the_generator_settings(family, applications):
    problem = REGISTRY.generate(family, 2, due_factor=1.5, deadline_factor=2.5)
    values = features(problem)
    assert values["jobs"] == applications
    assert values["due_tightness"] == pytest.approx(1.5)
    assert values["deadline_tightness"] == pytest.approx(2.5)
    assert values["due_share"] == values["deadline_share"] == applications / problem.size
    endpoints = 2 * applications
    assert values["pinned_share"] == pytest.approx(endpoints / problem.size)
    assert values["ccr"] > 0.0
    assert values["data_share"] == 1.0


def test_arrivals_spread_the_releases():
    batch = features(REGISTRY.generate("continuum-workflow", 3))
    stream = features(REGISTRY.generate("continuum-workflow", 3, arrival_rate=0.5))
    assert batch["release_span"] == 0.0
    assert stream["release_span"] > 0.0
    assert stream["release_spread"] > 0.0


def test_pinning_raises_the_pinned_share():
    free = features(REGISTRY.generate("continuum-workflow", 3))
    pinned = features(REGISTRY.generate("continuum-workflow", 3, pinned_fraction=0.5))
    assert pinned["pinned_share"] == pytest.approx(free["pinned_share"] + 20 / 48)
