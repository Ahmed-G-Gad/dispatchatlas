"""Job shops of Taillard's generator built from seeds that Taillard did not publish."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import pytest

from dispatchatlas.benchmarks import taillard
from dispatchatlas.benchmarks.registry import FamilyRegistry
from dispatchatlas.benchmarks.taillard import (
    MODULUS,
    TaillardJobShopSeeded,
    check_taillard_job_shop,
    job_shop_instances,
    open_shop_instances,
    register,
    taillard_job_shop,
    taillard_job_shop_seeded,
)
from dispatchatlas.core import ConfigError, check_schedule, problem_digest
from dispatchatlas.references import bibliography
from dispatchatlas.solvers.operators import random_order
from dispatchatlas.solvers.scheduling import Decoder

FAMILY = "taillard-job-shop-seeded"
SIZES = [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20)]


def _published() -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for item in (*job_shop_instances(), *open_shop_instances()):
        assert item.machine_seed is not None
        pairs.add((item.time_seed, item.machine_seed))
    return pairs


PUBLISHED = _published()
"""Every ``(time seed, machine seed)`` pair of Taillard's job-shop and open-shop tables."""


def _derives(time_seed: int, machine_seed: int) -> Callable[[int, str, int], int]:
    """A ``derive_seed`` whose values the generator maps onto the two given seeds."""
    values = {0: time_seed - 1, 1: machine_seed - 1}

    def derive(root: int, namespace: str, index: int = 0) -> int:
        return values[index]

    return derive


@pytest.fixture
def unpublished(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the module believe that Taillard published no seed pair."""
    monkeypatch.setattr(taillard, "_published_pairs", frozenset)


# ------------------------------------------------------------------ the seeds


@pytest.mark.parametrize(
    ("jobs", "machines", "seed", "seeds"),
    [
        (15, 15, 1, (680062677, 837839332)),
        (20, 15, 1, (122839419, 1508582515)),
        (30, 20, 12345, (1564288474, 1160083686)),
    ],
)
def test_the_derived_seeds_are_pinned(
    jobs: int, machines: int, seed: int, seeds: tuple[int, int]
) -> None:
    assert taillard._derived_seeds(jobs, machines, seed) == seeds
    metadata = taillard_job_shop_seeded(jobs, machines, seed).metadata
    assert (metadata["time_seed"], metadata["machine_seed"]) == seeds


def test_a_seed_and_a_size_give_the_same_seeds_every_time() -> None:
    first = taillard._derived_seeds(20, 15, 9)
    assert taillard._derived_seeds(20, 15, 9) == first
    assert taillard._derived_seeds(20, 15, 10) != first
    assert taillard._derived_seeds(15, 20, 9) != first
    assert taillard._derived_seeds(20, 20, 9) != first
    # One seed gives unrelated seeds for different sizes, not one stream cut at two lengths.
    assert len({taillard._derived_seeds(jobs, machines, 4)[0] for jobs, machines in SIZES}) == 5


def test_the_seeds_are_states_of_the_generator() -> None:
    assert taillard._generator_state(0) == 1
    assert taillard._generator_state(MODULUS - 2) == MODULUS - 1
    assert taillard._generator_state(MODULUS - 1) == 1
    for seed in range(2000):
        for seeds in (taillard._derived_seeds(15, 15, seed), taillard._derived_seeds(50, 20, seed)):
            assert all(1 <= value <= MODULUS - 1 for value in seeds)


def test_no_seed_derives_a_published_pair() -> None:
    assert len(PUBLISHED) == 140
    for jobs, machines in SIZES:
        for seed in range(2000):
            assert taillard._derived_seeds(jobs, machines, seed) not in PUBLISHED


def test_a_published_pair_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    for published in sorted(PUBLISHED):
        monkeypatch.setattr(taillard, "derive_seed", _derives(*published))
        with pytest.raises(ConfigError, match="a pair that Taillard published; use another seed"):
            taillard_job_shop_seeded(15, 15, 3)


def test_a_published_seed_alone_is_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = job_shop_instances()[:2]
    assert second.machine_seed is not None
    monkeypatch.setattr(taillard, "derive_seed", _derives(first.time_seed, second.machine_seed))
    problem = taillard_job_shop_seeded(15, 15, 3)
    assert problem.metadata["time_seed"] == first.time_seed
    assert problem.metadata["machine_seed"] == second.machine_seed


# --------------------------------------------------------------- the instances


def _rebuild(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Build a published instance from its own seeds, through the derivation of the family."""
    published = taillard_job_shop(name)
    item = taillard.taillard_instance("job-shop", name)
    assert item.machine_seed is not None
    monkeypatch.setattr(taillard, "derive_seed", _derives(item.time_seed, item.machine_seed))
    problem = taillard_job_shop_seeded(item.jobs, item.machines, 0)
    assert problem.tasks == published.tasks
    assert problem.resources == published.resources
    assert problem.dependencies == published.dependencies
    assert (problem.metadata["time_seed"], problem.metadata["machine_seed"]) == (
        item.time_seed,
        item.machine_seed,
    )
    assert problem.id != published.id


def test_ta01s_seeds_rebuild_ta01(monkeypatch: pytest.MonkeyPatch, unpublished: None) -> None:
    _rebuild(monkeypatch, "ta01")


@pytest.mark.parametrize("name", ["ta16", "ta21", "ta33", "ta42"])
def test_the_seeds_of_every_size_rebuild_the_published_instance(
    name: str, monkeypatch: pytest.MonkeyPatch, unpublished: None
) -> None:
    _rebuild(monkeypatch, name)


def test_the_same_arguments_give_the_same_problem() -> None:
    first = taillard_job_shop_seeded(6, 4, 11)
    assert taillard_job_shop_seeded(6, 4, 11) == first
    assert problem_digest(taillard_job_shop_seeded(6, 4, 11)) == problem_digest(first)
    assert problem_digest(taillard_job_shop_seeded(6, 4, 12)) != problem_digest(first)
    assert problem_digest(taillard_job_shop_seeded(4, 6, 11)) != problem_digest(first)


def test_a_problem_is_a_job_shop_of_the_requested_size() -> None:
    problem = taillard_job_shop_seeded(6, 4, 3)
    assert problem.id == "taillard-job-shop-seeded-6x4-3"
    assert problem.family == FAMILY
    assert problem.size == 24
    assert problem.shapes == {"job-shop"}
    assert problem.objectives == ("makespan",)
    assert dict(problem.metadata) == {
        "jobs": 6,
        "machines": 4,
        "seed": 3,
        "time_seed": problem.metadata["time_seed"],
        "machine_seed": problem.metadata["machine_seed"],
    }
    assert len(problem.resources) == 4
    assert len(problem.groups) == 6
    for indices in problem.groups.values():
        route = [problem.tasks[index] for index in indices]
        assert len(route) == 4
        assert sorted(next(iter(task.modes[0].demands)) for task in route) == [
            "m0",
            "m1",
            "m2",
            "m3",
        ]
        assert all(1 <= task.modes[0].duration <= 99 for task in route)


@pytest.mark.parametrize(("jobs", "machines"), [(1, 1), (1, 5), (7, 1), (15, 15)])
def test_the_smallest_sizes_build(jobs: int, machines: int) -> None:
    problem = taillard_job_shop_seeded(jobs, machines, 2)
    assert problem.size == jobs * machines
    assert problem.shapes == {"job-shop"}


def test_a_problem_decodes_to_a_feasible_schedule() -> None:
    problem = taillard_job_shop_seeded(8, 5, 6)
    schedule = Decoder(problem).schedule(random_order(problem, random.Random(0)))
    assert check_schedule(problem, schedule) == ()


def test_the_published_job_shops_keep_their_routes() -> None:
    for item in job_shop_instances():
        check_taillard_job_shop(taillard_job_shop(item.name))


def test_a_published_job_shop_keeps_its_name_and_metadata() -> None:
    problem = taillard_job_shop("ta42")
    assert (problem.id, problem.family) == ("ta42", None)
    assert dict(problem.metadata) == {
        "suite": "taillard-job-shop",
        "instance": "ta42",
        "problem_class": "job-shop",
        "jobs": 30,
        "machines": 20,
        "time_seed": 2116959593,
        "machine_seed": 1546338557,
        "source": "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop2.txt",
    }


@pytest.mark.parametrize(
    ("jobs", "machines", "seed"),
    [
        (0, 3, 1),
        (-1, 3, 1),
        (True, 3, 1),
        (2.0, 3, 1),
        ("3", 3, 1),
        (3, 0, 1),
        (3, -2, 1),
        (3, True, 1),
        (3, 2.5, 1),
        (3, 3, -1),
        (3, 3, True),
        (3, 3, 1.5),
        (3, 3, "1"),
        (3, 3, None),
    ],
)
def test_invalid_arguments_are_refused(jobs: Any, machines: Any, seed: Any) -> None:
    with pytest.raises(ValueError, match="must be"):
        taillard_job_shop_seeded(jobs, machines, seed)


# ------------------------------------------------------------------ the family


def _registry() -> FamilyRegistry:
    registry = FamilyRegistry()
    register(registry)
    return registry


def test_the_family_is_registered_by_its_register_function() -> None:
    registry = _registry()
    assert registry.ids() == (FAMILY,)
    info = registry.info(FAMILY)
    assert info.problem_class == "job-shop"
    assert info.objectives == ("makespan",)
    assert info.summary.endswith(".")
    assert info.parameters == {"jobs": 15, "machines": 15}
    for key in info.references:
        assert key in bibliography(), key
    assert "taillard1993benchmarks" in info.references
    with pytest.raises(ConfigError, match="already registered"):
        register(registry)


def test_the_family_generates_what_the_function_builds() -> None:
    registry = _registry()
    problem = registry.generate(FAMILY, 7, jobs=6, machines=4)
    assert problem == taillard_job_shop_seeded(6, 4, 7)
    assert registry.create(FAMILY, jobs=6, machines=4).generate(7) == problem
    assert TaillardJobShopSeeded(6, 4).generate(7) == problem
    default = registry.generate(FAMILY, 7)
    assert default.id == "taillard-job-shop-seeded-15x15-7"
    assert default.family == FAMILY
    assert default == taillard_job_shop_seeded(15, 15, 7)


def test_the_family_replays_from_a_seed_and_differs_between_seeds() -> None:
    registry = _registry()
    first = registry.generate(FAMILY, 11, jobs=5, machines=3)
    assert problem_digest(registry.generate(FAMILY, 11, jobs=5, machines=3)) == problem_digest(
        first
    )
    assert problem_digest(registry.generate(FAMILY, 12, jobs=5, machines=3)) != problem_digest(
        first
    )
    assert first.metadata["seed"] == 11
    assert registry.generate(FAMILY, 2**62, jobs=3, machines=3).metadata["seed"] == 2**62


@pytest.mark.parametrize(
    "parameters",
    [{"jobs": 0}, {"machines": -1}, {"jobs": 2.0}, {"machines": True}, {"jobs": None}, {"size": 3}],
)
def test_the_family_refuses_invalid_parameters(parameters: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        _registry().create(FAMILY, **parameters)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "1"])
def test_the_family_refuses_an_invalid_seed(seed: Any) -> None:
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        _registry().generate(FAMILY, seed, jobs=3, machines=3)
