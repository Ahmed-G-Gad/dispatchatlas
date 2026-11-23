"""The job-shop view reads chains as jobs and refuses every feature its builders do not model."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from dispatchatlas.benchmarks import taillard_flow_shop, taillard_job_shop
from dispatchatlas.core import (
    FEATURES,
    Dependency,
    Link,
    Mode,
    ModelError,
    Network,
    Problem,
    Resource,
    SetupTimes,
    Task,
)
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from tests.helpers import chain_problem, small_job_shop


def _two_jobs() -> Problem:
    """Job ``a`` runs on ``m0`` then ``m1``, job ``b`` on ``m1`` then ``m0``."""
    return Problem(
        "two-jobs",
        tasks=(
            Task.on("a0", "m0", 2),
            Task.on("a1", "m1", 3),
            Task.on("b0", "m1", 1),
            Task.on("b1", "m0", 2),
        ),
        resources=(Resource("m0"), Resource("m1")),
        dependencies=(Dependency("a0", "a1"), Dependency("b0", "b1")),
    )


def test_the_small_job_shop_reads_as_three_jobs_on_three_machines() -> None:
    shop = JobShop.from_problem(small_job_shop())
    assert shop.operations == ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    assert shop.job == (0, 0, 0, 1, 1, 1, 2, 2, 2)
    assert shop.machine == (0, 1, 2, 0, 2, 1, 1, 0, 2)
    assert shop.duration == (3.0, 2.0, 2.0, 2.0, 1.0, 4.0, 3.0, 2.0, 3.0)
    assert shop.release == (0.0,) * 9
    assert shop.predecessor == (-1, 0, 1, -1, 3, 4, -1, 6, 7)
    assert shop.successor == (1, 2, -1, 4, 5, -1, 7, 8, -1)
    assert shop.slot_job.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert (shop.size, shop.jobs, shop.machines) == (9, 3, 3)
    assert repr(shop) == "JobShop(problem='small-job-shop', jobs=3, machines=3, operations=9)"
    assert all(type(value) is float for value in shop.duration + shop.release)


def test_ta01_reads_as_its_routes() -> None:
    problem = taillard_job_shop("ta01")
    shop = JobShop.from_problem(problem)
    assert (shop.size, shop.jobs, shop.machines) == (225, 15, 15)
    for job, route in enumerate(shop.operations):
        assert [problem.tasks[task].id for task in route] == [
            f"j{job:02d}-o{step:02d}" for step in range(15)
        ]
        # Every Taillard job visits every machine once.
        assert sorted(shop.machine[task] for task in route) == list(range(15))
    for task, item in enumerate(problem.tasks):
        mode = item.modes[0]
        assert problem.resources[shop.machine[task]].id == next(iter(mode.demands))
        assert shop.duration[task] == mode.duration
    assert shop.slot_job.tolist() == [job for job in range(15) for _ in range(15)]


def test_jobs_follow_their_chains_and_are_numbered_by_their_first_task() -> None:
    problem = Problem(
        "listed-out-of-order",
        tasks=(
            Task.on("b1", "m0", 1),
            Task.on("a0", "m1", 2),
            Task.on("b0", "m1", 3),
            Task.on("c0", "m0", 4),
            Task.on("a1", "m0", 5),
        ),
        resources=(Resource("m0"), Resource("m1")),
        dependencies=(Dependency("b0", "b1"), Dependency("a0", "a1")),
    )
    shop = JobShop.from_problem(problem)
    # The first tasks of the chains are a0, b0, and c0, at task indices 1, 2, and 3.
    assert shop.operations == ((1, 4), (2, 0), (3,))
    assert shop.job == (1, 0, 1, 2, 0)
    assert shop.predecessor == (2, -1, -1, -1, 1)
    assert shop.successor == (-1, 4, 0, -1, -1)
    assert shop.slot_job.tolist() == [0, 0, 1, 1, 2]


def test_release_times_due_dates_and_groups_are_accepted() -> None:
    base = _two_jobs()
    problem = replace(
        base,
        tasks=(
            Task.on("a0", "m0", 2, group="a"),
            Task.on("a1", "m1", 3, release=4.0, group="a"),
            Task.on("b0", "m1", 1, release=1.5, group="b"),
            Task.on("b1", "m0", 2, due=6.0, group="b"),
        ),
    )
    assert problem.features == {"precedence", "release", "due", "groups"} == JOB_SHOP_FEATURES
    shop = JobShop.from_problem(problem)
    assert shop.release == (0.0, 4.0, 1.5, 0.0)
    assert JobShop.from_problem(chain_problem()).operations == ((0, 1), (2,))
    assert set(FEATURES) >= JOB_SHOP_FEATURES


def _refused() -> list[Any]:
    """Problems the view refuses, each with the start of every issue it reports."""
    base = _two_jobs()
    _, a1, b0, b1 = base.tasks
    shape = "it lacks the 'job-shop' shape"
    cases = [
        (
            "lags",
            replace(base, dependencies=(Dependency("a0", "a1", lag=1.0), Dependency("b0", "b1"))),
            ["it has the feature 'lags'"],
        ),
        (
            "deadline",
            replace(base, tasks=(Task.on("a0", "m0", 2, deadline=4.0), a1, b0, b1)),
            ["it has the feature 'deadline'"],
        ),
        (
            "setups",
            replace(base, setups=SetupTimes({("m0", "a0", "b1"): 1.0})),
            ["it has the feature 'setups'"],
        ),
        (
            "network",
            replace(
                base,
                dependencies=(Dependency("a0", "a1", data=2.0), Dependency("b0", "b1")),
                network=Network((Link("m0", "m1", latency=1.0),)),
            ),
            ["it has the feature 'network'"],
        ),
        ("permutation", replace(base, permutation=True), ["it has the feature 'permutation'"]),
        (
            # The machine of capacity 2 is held whole, so only the feature is refused.
            "cumulative",
            replace(
                base,
                tasks=(
                    Task.on("a0", "m0", 2, demand=2.0),
                    a1,
                    b0,
                    Task.on("b1", "m0", 2, demand=2.0),
                ),
                resources=(Resource("m0", capacity=2.0), Resource("m1")),
            ),
            ["it has the feature 'cumulative'"],
        ),
        (
            "half a machine",
            replace(
                base,
                tasks=(Task.on("a0", "m0", 2, demand=1.0), a1, b0, b1),
                resources=(Resource("m0", capacity=2.0), Resource("m1")),
            ),
            [shape, "it has the feature 'cumulative'"],
        ),
        (
            "modes",
            replace(
                base, tasks=(Task("a0", (Mode(2, {"m0": 1.0}), Mode(3, {"m1": 1.0}))), a1, b0, b1)
            ),
            [shape, "it has the feature 'modes'"],
        ),
        (
            "multi-resource",
            replace(base, tasks=(Task("a0", (Mode(2, {"m0": 1.0, "m1": 1.0}),)), a1, b0, b1)),
            [shape, "it has the feature 'multi-resource'"],
        ),
        (
            "fork",
            replace(base, dependencies=(Dependency("a0", "a1"), Dependency("a0", "b0"))),
            [shape],
        ),
        (
            "join",
            replace(base, dependencies=(Dependency("a0", "a1"), Dependency("b0", "a1"))),
            [shape],
        ),
        (
            "every refusal at once",
            replace(
                base,
                tasks=(Task.on("a0", "m0", 2, deadline=4.0), a1, b0, b1),
                dependencies=(Dependency("a0", "a1", lag=1.0), Dependency("a0", "b0")),
                setups=SetupTimes({("m0", "a0", "b1"): 1.0}),
            ),
            [
                shape,
                "it has the feature 'deadline'",
                "it has the feature 'lags'",
                "it has the feature 'setups'",
            ],
        ),
    ]
    return [pytest.param(problem, expected, id=label) for label, problem, expected in cases]


@pytest.mark.parametrize(("problem", "expected"), _refused())
def test_every_feature_the_builders_do_not_model_is_refused(
    problem: Problem, expected: list[str]
) -> None:
    with pytest.raises(ModelError, match="problem 'two-jobs' is not a job shop") as caught:
        JobShop.from_problem(problem)
    issues = caught.value.issues
    assert len(issues) == len(expected), issues
    for issue, start in zip(issues, expected, strict=True):
        assert issue.startswith(start), issue


def test_a_feature_refusal_states_what_the_feature_means() -> None:
    problem = replace(
        _two_jobs(), dependencies=(Dependency("a0", "a1", lag=1.0), Dependency("b0", "b1"))
    )
    with pytest.raises(ModelError) as caught:
        JobShop.from_problem(problem)
    assert caught.value.issues == (
        "it has the feature 'lags', which the view does not model: " + FEATURES["lags"],
    )


def test_a_flow_shop_is_refused_for_its_one_job_order() -> None:
    problem = taillard_flow_shop("ta001")
    assert "job-shop" in problem.shapes
    with pytest.raises(ModelError) as caught:
        JobShop.from_problem(problem)
    assert [issue.split(",")[0] for issue in caught.value.issues] == [
        "it has the feature 'permutation'"
    ]


def test_the_slot_layout_cannot_be_changed() -> None:
    shop = JobShop.from_problem(small_job_shop())
    with pytest.raises(ValueError, match="read-only"):
        shop.slot_job[0] = 2
