"""Static task priorities against hand-computed values."""

from __future__ import annotations

import math

import pytest

from dispatchatlas.core.model import Dependency, Link, Mode, Network, Problem, Resource, Task
from dispatchatlas.solvers.priorities import (
    effective_due_dates,
    longest_tails,
    mean_durations,
    min_durations,
    remaining_group_work,
    upward_ranks,
)
from tests.helpers import chain_problem, random_problem

MACHINES = (Resource("m1"), Resource("m2"))


def test_duration_summaries_over_modes() -> None:
    problem = Problem(
        "durations",
        tasks=(
            Task("a", (Mode(3, {"m1": 1}), Mode(5, {"m2": 1}))),
            Task.on("b", "m1", 2),
            Task("c", (Mode(1, {"m1": 1}), Mode(2, {"m2": 1}), Mode(6, {"m1": 1}))),
        ),
        resources=MACHINES,
    )
    assert min_durations(problem) == [3.0, 2.0, 1.0]
    assert mean_durations(problem) == [4.0, 2.0, 3.0]


def test_effective_due_dates_inherit_the_earliest_due_date_of_the_group() -> None:
    problem = Problem(
        "dues",
        tasks=(
            Task.on("a", "m1", 1, group="j1"),
            Task.on("b", "m1", 1, group="j1", due=8),
            Task.on("c", "m1", 1, group="j1", due=5),
            Task.on("d", "m1", 1),
            Task.on("e", "m1", 1, group="j2"),
            Task.on("f", "m1", 1, due=3),
        ),
        resources=MACHINES,
    )
    assert effective_due_dates(problem) == [5.0, 8.0, 5.0, math.inf, math.inf, 3.0]


def _branching() -> Problem:
    """``a`` (3 or 4) -> ``b`` (2, lag 1) -> ``c`` (4); ``a`` -> ``d`` (1); ``x`` elsewhere."""
    return Problem(
        "branching",
        tasks=(
            Task("a", (Mode(3, {"m1": 1}), Mode(4, {"m2": 1})), group="j"),
            Task.on("b", "m1", 2, group="j"),
            Task.on("c", "m2", 4, group="j"),
            Task.on("d", "m2", 1, group="k"),
            Task.on("x", "m1", 7),
        ),
        resources=MACHINES,
        dependencies=(Dependency("a", "b", lag=1), Dependency("b", "c"), Dependency("a", "d")),
    )


def test_longest_tails_follow_the_critical_path() -> None:
    # c = 4, b = 2 + 4, d = 1, a = 3 + max(1 + 6, 1); x stands alone.
    assert longest_tails(_branching()) == [10.0, 6.0, 4.0, 1.0, 7.0]
    assert longest_tails(chain_problem()) == [5.0, 2.0, 4.0]


def test_remaining_group_work_sums_the_group_from_the_task_on() -> None:
    # a's group work is a, b, and c; d belongs to another group; x to none.
    assert remaining_group_work(_branching()) == [9.0, 6.0, 4.0, 1.0, 7.0]


def test_remaining_group_work_follows_dependencies_within_the_group() -> None:
    problem = Problem(
        "detour",
        tasks=(
            Task.on("p", "m1", 1, group="g"),
            Task.on("q", "m1", 2, group="h"),
            Task.on("r", "m1", 4, group="g"),
            Task.on("s", "m1", 8, group="g"),
        ),
        resources=MACHINES,
        dependencies=(Dependency("p", "q"), Dependency("q", "r"), Dependency("p", "s")),
    )
    # r follows p only through q of group h, so it is not part of p's group work.
    assert remaining_group_work(problem) == [9.0, 2.0, 4.0, 8.0]


def test_upward_ranks_by_hand() -> None:
    problem = Problem(
        "heft",
        tasks=(
            Task("a", (Mode(2, {"e1": 1}), Mode(4, {"c1": 1}))),
            Task.on("b", "e1", 1),
            Task("c", (Mode(3, {"c1": 1}), Mode(5, {"e1": 1}))),
        ),
        resources=(Resource("e1", tier="edge"), Resource("c1", tier="cloud")),
        dependencies=(
            Dependency("a", "b", lag=1, data=4),
            Dependency("a", "c", lag=2),
            Dependency("b", "c", data=2),
        ),
        network=Network((Link("edge", "cloud", bandwidth=2, latency=1),)),
    )
    # Mean durations 3, 1, 4. Communication: a -> b is 1 + mean(0, 1 + 4 / 2),
    # a -> c is the lag 2, and b -> c is mean(1 + 2 / 2, 0).
    c = 4.0
    b = 1.0 + (1.0 + c)
    a = 3.0 + max(1.0 + 1.5 + b, 2.0 + c)
    assert upward_ranks(problem) == pytest.approx([a, b, c])
    assert upward_ranks(chain_problem()) == [5.0, 2.0, 4.0]


@pytest.mark.parametrize("seed", range(10))
def test_path_priorities_dominate_their_successors(seed: int) -> None:
    problem = random_problem(seed, tasks=14, density=0.3, network=seed % 2 == 0)
    tails, ranks = longest_tails(problem), upward_ranks(problem)
    shortest, mean = min_durations(problem), mean_durations(problem)
    lags = {(d.before, d.after): d.lag for d in problem.dependencies}
    for task, followers in enumerate(problem.successors):
        name = problem.tasks[task].id
        assert tails[task] >= shortest[task]
        assert ranks[task] >= mean[task]
        for following in followers:
            lag = lags[(name, problem.tasks[following].id)]
            assert tails[task] >= shortest[task] + lag + tails[following] - 1e-9
            assert ranks[task] >= mean[task] + lag + ranks[following] - 1e-9
    assert max(tails) <= sum(shortest) + sum(lags.values())
