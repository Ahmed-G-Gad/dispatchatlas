"""The contract every classic benchmark family honours.

Each family must replay identically from a seed, name its problems
``<family>-<label>-<seed>``, record its parameters and seed, cite resolvable
sources, reject invalid parameters, and generate problems that the shared
schedule builder turns into feasible schedules.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import pytest

from dispatchatlas.benchmarks import default_registry
from dispatchatlas.core import ConfigError, check_schedule, problem_digest
from dispatchatlas.core.serialization import problem_from_dict, problem_to_dict
from dispatchatlas.references import bibliography
from dispatchatlas.solvers.operators import random_modes, random_order
from dispatchatlas.solvers.scheduling import Decoder

REGISTRY = default_registry(plugins=False)

SMALL = {
    "flow-shop": {"jobs": 6, "machines": 3},
    "job-shop": {"jobs": 5, "machines": 4},
    "open-shop": {"jobs": 4, "machines": 3},
    "flexible-job-shop": {"jobs": 4, "machines": 4, "min_operations": 2, "max_operations": 4},
    "hybrid-flow-shop": {"jobs": 5, "stages": 3},
    "parallel-machines": {"jobs": 8, "machines": 3, "max_setup": 9},
    "rcpsp": {"activities": 12},
    "multi-mode-rcpsp": {"activities": 10},
}
FAMILIES = sorted(SMALL)

HEADLINES = {
    "flow-shop": "20x5",
    "job-shop": "10x10",
    "open-shop": "5x5",
    "flexible-job-shop": "10x6",
    "hybrid-flow-shop": "20x5",
    "parallel-machines": "20x5",
    "rcpsp": "30x4-nc1.5-rf0.5-rs0.5",
    "multi-mode-rcpsp": "20x2-nc1.5-rf0.5-rs0.5",
}


CORNERS: dict[str, dict[str, Any]] = {
    "flow-shop": {"jobs": 1, "machines": 1, "max_time": 1, "due_date_tightness": 1.3},
    "job-shop": {"jobs": 1, "machines": 1, "max_time": 1, "weighted": True},
    "open-shop": {"jobs": 1, "machines": 1, "max_time": 1},
    "flexible-job-shop": {
        "jobs": 2,
        "machines": 1,
        "min_operations": 1,
        "max_operations": 1,
        "flexibility": 1,
        "max_time": 1,
    },
    "hybrid-flow-shop": {"jobs": 2, "stages": 1, "max_machines": 1, "unrelated": True},
    "parallel-machines": {
        "jobs": 2,
        "machines": 1,
        "max_time": 1,
        "max_setup": 1,
        "release_span": 1.0,
        "tardiness_factor": 1.0,
    },
    "rcpsp": {
        "activities": 2,
        "resources": 1,
        "start_activities": 1,
        "finish_activities": 1,
        "resource_factor": 0.0,
        "resource_strength": 0.0,
    },
    "multi-mode-rcpsp": {
        "activities": 2,
        "resources": 1,
        "start_activities": 1,
        "finish_activities": 1,
        "min_modes": 1,
        "max_modes": 1,
    },
}
"""Smallest sizes and extreme options, which the model must accept for every seed."""


def test_every_family_is_registered():
    assert set(FAMILIES) <= set(REGISTRY.ids())


@pytest.mark.parametrize("family", FAMILIES)
def test_the_same_seed_gives_the_same_problem(family):
    first = REGISTRY.generate(family, 11, **SMALL[family])
    second = REGISTRY.create(family, **SMALL[family]).generate(11)
    assert problem_digest(first) == problem_digest(second)
    other = REGISTRY.generate(family, 12, **SMALL[family])
    assert problem_digest(other) != problem_digest(first)


@pytest.mark.parametrize("family", FAMILIES)
def test_default_problems_are_named_after_family_size_and_seed(family):
    problem = REGISTRY.generate(family, 7)
    assert problem.id == f"{family}-{HEADLINES[family]}-7"
    assert problem.family == family


@pytest.mark.parametrize("family", FAMILIES)
def test_metadata_records_every_parameter_and_the_seed(family):
    generator: Any = REGISTRY.create(family, **SMALL[family])
    problem = generator.generate(5)
    expected: dict[str, object] = {"seed": 5}
    for item in dataclasses.fields(generator):
        value = getattr(generator, item.name)
        if value is not None:
            expected[item.name] = ",".join(value) if isinstance(value, tuple) else value
    assert dict(problem.metadata) == expected


@pytest.mark.parametrize("family", FAMILIES)
def test_family_metadata_is_complete(family):
    info = REGISTRY.info(family)
    assert info.summary.endswith(".")
    assert info.references
    for key in info.references:
        assert key in bibliography(), f"{family} cites unknown source {key!r}"
    assert info.objectives == ("makespan",)
    assert "objectives" in info.parameters


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", range(4))
def test_generated_problems_decode_to_feasible_schedules(family, seed):
    problem = REGISTRY.generate(family, seed, **SMALL[family])
    decoder = Decoder(problem)
    rng = random.Random(seed)
    for _ in range(3):
        modes = random_modes(problem, rng) if rng.random() < 0.5 else None
        schedule = decoder.schedule(random_order(problem, rng), modes)
        assert check_schedule(problem, schedule) == ()


@pytest.mark.parametrize("family", FAMILIES)
def test_corner_parameters_give_valid_problems(family):
    for seed in range(5):
        problem = REGISTRY.generate(family, seed, **CORNERS[family])
        schedule = Decoder(problem).schedule(list(range(problem.size)))
        assert check_schedule(problem, schedule) == ()


@pytest.mark.parametrize("family", FAMILIES)
def test_problems_survive_serialization(family):
    problem = REGISTRY.generate(family, 3, **SMALL[family])
    assert problem_digest(problem_from_dict(problem_to_dict(problem))) == problem_digest(problem)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", [-1, True, 1.5, "1"])
def test_invalid_seeds_are_refused(family, seed):
    generator = REGISTRY.create(family, **SMALL[family])
    with pytest.raises(ValueError, match="seed"):
        generator.generate(seed)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "objectives",
    [("unknown",), (), ("makespan", "makespan"), "total_tardiness", (1,)],
)
def test_invalid_objectives_are_refused(family, objectives):
    with pytest.raises(ConfigError):
        REGISTRY.create(family, objectives=objectives)


@pytest.mark.parametrize("family", FAMILIES)
def test_objectives_can_be_chosen(family):
    problem = REGISTRY.generate(
        family, 1, objectives=["makespan", "total_completion_time"], **SMALL[family]
    )
    assert problem.objectives == ("makespan", "total_completion_time")
    assert problem.id.endswith("-makespan+total_completion_time-1")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("value", [0, -3, 2.0, True, None])
def test_invalid_sizes_are_refused(family, value):
    size = next(iter(SMALL[family]))
    with pytest.raises(ConfigError):
        REGISTRY.create(family, **{size: value})


@pytest.mark.parametrize("family", FAMILIES)
def test_unknown_parameters_are_refused(family):
    with pytest.raises(ConfigError, match="no parameters"):
        REGISTRY.create(family, colour="blue")
