"""Feasibility checking: every violation code, tolerances, instants, and permutations."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any

import pytest

from dispatchatlas.core.model import (
    Dependency,
    Link,
    Mode,
    Network,
    Problem,
    Resource,
    SetupTimes,
    Task,
)
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.validation import (
    DEFAULT_TOLERANCE,
    Violation,
    check_schedule,
    deadline_excess,
    is_feasible,
)
from dispatchatlas.solvers.operators import random_modes, random_order
from dispatchatlas.solvers.scheduling import Decoder
from tests.helpers import chain_problem, random_problem


def _schedule(*rows: tuple[str, int, float, float]) -> Schedule:
    return Schedule.of(Assignment(*row) for row in rows)


def _codes(problem: Problem, schedule: Schedule, **options: float) -> list[str]:
    return [violation.code for violation in check_schedule(problem, schedule, **options)]


def _single(problem: Problem, schedule: Schedule, code: str) -> Violation:
    found = [v for v in check_schedule(problem, schedule) if v.code == code]
    assert len(found) == 1, found
    return found[0]


# ------------------------------------------------------------- assignments


def test_a_feasible_schedule_has_no_violation() -> None:
    schedule = _schedule(("a", 0, 0, 3), ("b", 0, 3, 5), ("c", 0, 0, 4))
    assert check_schedule(chain_problem(), schedule) == ()
    assert is_feasible(chain_problem(), schedule)


def test_assignment_errors_are_reported_alone() -> None:
    schedule = _schedule(("a", 0, 0, 3), ("a", 0, 3, 6), ("x", 0, 0, 1), ("c", 0, 0, 1))
    violations = check_schedule(chain_problem(), schedule)
    assert [(v.code, v.task) for v in violations] == [
        ("unknown-task", "x"),
        ("duplicate-task", "a"),
        ("missing-task", "b"),
    ]
    assert not is_feasible(chain_problem(), schedule)


def test_invalid_modes_stop_the_check_after_the_per_task_rules() -> None:
    schedule = _schedule(("a", 1, 0, 3), ("b", -1, 0, 2), ("c", 0, 0, 1))
    violations = check_schedule(chain_problem(), schedule)
    assert [(v.code, v.task) for v in violations] == [
        ("invalid-mode", "a"),
        ("invalid-mode", "b"),
        ("wrong-duration", "c"),
    ]


def test_wrong_duration_reports_the_difference() -> None:
    violation = _single(
        chain_problem(),
        _schedule(("a", 0, 0, 3), ("b", 0, 3, 5), ("c", 0, 0, 4.5)),
        "wrong-duration",
    )
    assert (violation.task, violation.amount) == ("c", 0.5)
    assert "runs 4.5 instead of 4.0" in violation.message


@pytest.mark.parametrize(
    "row",
    [("c", 0, math.nan, math.nan), ("c", 0, math.inf, math.inf), ("c", 0, 0.0, math.inf)],
)
def test_non_finite_times_are_never_feasible(row: tuple[str, int, float, float]) -> None:
    schedule = _schedule(("a", 0, 0, 3), ("b", 0, 3, 5), row)
    assert "wrong-duration" in _codes(chain_problem(), schedule)
    assert not is_feasible(chain_problem(), schedule)


def test_release_times_allow_the_tolerance() -> None:
    problem = Problem(
        "release", tasks=(Task.on("a", "m", 2, release=5),), resources=(Resource("m"),)
    )
    violation = _single(problem, _schedule(("a", 0, 4, 6)), "before-release")
    assert (violation.task, violation.amount) == ("a", 1.0)
    assert check_schedule(problem, _schedule(("a", 0, 5 - DEFAULT_TOLERANCE / 10, 7))) == ()
    assert _codes(problem, _schedule(("a", 0, 4.9, 6.9)), tolerance=0.2) == []


def test_deadlines() -> None:
    problem = Problem(
        "deadlines",
        tasks=(Task.on("a", "m", 3, deadline=4), Task.on("b", "m", 1, deadline=10)),
        resources=(Resource("m"),),
    )
    late = _schedule(("a", 0, 2, 5), ("b", 0, 9.5, 10.5))
    violations = check_schedule(problem, late)
    assert [(v.code, v.task, v.amount) for v in violations] == [
        ("deadline", "a", 1.0),
        ("deadline", "b", 0.5),
    ]
    assert deadline_excess(problem, late) == 1.5
    on_time = _schedule(("a", 0, 1, 4), ("b", 0, 9, 10 + DEFAULT_TOLERANCE / 10))
    assert check_schedule(problem, on_time) == ()
    assert deadline_excess(problem, _schedule(("a", 0, 0, 3), ("b", 0, 3, 4))) == 0.0


# --------------------------------------------------------------- precedence


def _transfer_problem() -> Problem:
    return Problem(
        "transfer",
        tasks=(Task.on("a", "e1", 2), Task("b", (Mode(1, {"e1": 1}), Mode(1, {"c1": 1})))),
        resources=(Resource("e1", tier="edge"), Resource("c1", tier="cloud")),
        dependencies=(Dependency("a", "b", lag=1, data=4),),
        network=Network((Link("edge", "cloud", bandwidth=2, latency=1),)),
    )


def test_precedence_includes_lags_and_transfer_times() -> None:
    problem = _transfer_problem()
    # On the other tier b waits for the lag and a transfer of 1 + 4 / 2.
    assert check_schedule(problem, _schedule(("a", 0, 0, 2), ("b", 1, 6, 7))) == ()
    violation = _single(problem, _schedule(("a", 0, 0, 2), ("b", 1, 5.5, 6.5)), "precedence")
    assert (violation.task, violation.amount) == ("b", 0.5)
    assert "allows it at 6.0" in violation.message
    # On the same resource the data needs no transfer, only the lag.
    assert check_schedule(problem, _schedule(("a", 0, 0, 2), ("b", 0, 3, 4))) == ()
    assert check_schedule(problem, _schedule(("a", 0, 0, 2), ("b", 0, 3 - 1e-7, 4 - 1e-7))) == ()
    violation = _single(problem, _schedule(("a", 0, 0, 2), ("b", 0, 2.5, 3.5)), "precedence")
    assert violation.amount == pytest.approx(0.5)


# ----------------------------------------------------------------- capacity


def _machines(*tasks: Task, capacity: float = 1.0) -> Problem:
    return Problem("machines", tasks=tasks, resources=(Resource("m", capacity=capacity),))


def test_overlap_on_a_machine_is_a_capacity_violation() -> None:
    problem = _machines(Task.on("a", "m", 3), Task.on("b", "m", 2))
    violation = _single(problem, _schedule(("a", 0, 0, 3), ("b", 0, 2, 4)), "capacity")
    assert (violation.resource, violation.amount) == ("m", 1.0)
    assert "over capacity by 1.0 at time 2" in violation.message


def test_a_rounding_sliver_between_back_to_back_tasks_is_not_a_violation() -> None:
    problem = _machines(Task.on("a", "m", 3), Task.on("b", "m", 2))
    rounding = 2.0**-30  # exact in binary, so the duration stays exactly 2
    sliver = _schedule(("a", 0, 0, 3), ("b", 0, 3 - rounding, 5 - rounding))
    assert check_schedule(problem, sliver) == ()
    assert _codes(problem, sliver, tolerance=0.0) == ["capacity"]
    overlap = _schedule(("a", 0, 0, 3), ("b", 0, 2.6, 4.6))
    assert _codes(problem, overlap) == ["capacity"]
    assert _codes(problem, overlap, tolerance=0.5) == []


def test_cumulative_capacity_counts_units() -> None:
    problem = _machines(*(Task.on(f"t{i}", "m", 2) for i in range(3)), capacity=2)
    two = _schedule(("t0", 0, 0, 2), ("t1", 0, 0, 2), ("t2", 0, 2, 4))
    assert check_schedule(problem, two) == ()
    three = _schedule(("t0", 0, 0, 2), ("t1", 0, 1, 3), ("t2", 0, 1.5, 3.5))
    violation = _single(problem, three, "capacity")
    assert violation.amount == 1.0


def test_float_sums_at_the_capacity_boundary_are_feasible() -> None:
    tasks = (Task.on("a", "m", 1, demand=0.1), Task.on("b", "m", 1, demand=0.2))
    problem = _machines(*tasks, Task.on("c", "m", 1, demand=0.1), capacity=0.3)
    assert 0.1 + 0.2 > 0.3
    assert check_schedule(problem, _schedule(("a", 0, 0, 1), ("b", 0, 0, 1), ("c", 0, 1, 2))) == ()
    over = _schedule(("a", 0, 0, 1), ("b", 0, 0, 1), ("c", 0, 0.5, 1.5))
    assert _codes(problem, over) == ["capacity"]


# ------------------------------------------------------------------ instants


def _instant_problem() -> Problem:
    return Problem(
        "instants",
        tasks=(
            Task.on("long", "m", 10),
            Task.on("z", "m", 0),
            Task.on("y", "m", 0),
            Task.on("wide-long", "w", 10, demand=2),
            Task.on("wide-z", "w", 0, demand=2),
            Task.on("pool-a", "p", 10),
            Task.on("pool-b", "p", 10),
            Task.on("pool-z", "p", 0),
        ),
        resources=(Resource("m"), Resource("w", capacity=2), Resource("p", capacity=2)),
    )


def _instants(z: float, y: float, wide: float, pool: float) -> Schedule:
    return _schedule(
        ("long", 0, 0, 10),
        ("z", 0, z, z),
        ("y", 0, y, y),
        ("wide-long", 0, 0, 10),
        ("wide-z", 0, wide, wide),
        ("pool-a", 0, 0, 10),
        ("pool-b", 0, 0, 10),
        ("pool-z", 0, pool, pool),
    )


def test_exclusive_resources_of_the_instant_problem() -> None:
    assert _instant_problem().exclusive_resources == frozenset({"m", "w"})


def test_zero_duration_tasks_may_not_fall_inside_a_task_on_an_exclusive_resource() -> None:
    problem = _instant_problem()
    violations = check_schedule(problem, _instants(z=4, y=10, wide=5, pool=5))
    assert [(v.code, v.task, v.resource, v.amount) for v in violations] == [
        ("capacity", "z", "m", 6.0),
        ("capacity", "wide-z", "w", 5.0),
    ]
    assert "falls inside a task running on exclusive resource 'm'" in violations[0].message


def test_zero_duration_tasks_at_boundaries_or_on_cumulative_resources_are_feasible() -> None:
    problem = _instant_problem()
    # Instants at the start or end of the long task, and within the tolerance
    # of them, are outside it. On the full pool an instant holds nothing.
    for z, y in ((0, 10), (10, 10), (0, 0), (1e-7, 10 - 1e-7), (10 + 1e-7, 11)):
        assert check_schedule(problem, _instants(z=z, y=y, wide=10, pool=5)) == ()


def test_zero_duration_tasks_do_not_block_each_other() -> None:
    problem = _machines(Task.on("a", "m", 0), Task.on("b", "m", 0), Task.on("c", "m", 2))
    assert check_schedule(problem, _schedule(("a", 0, 2, 2), ("b", 0, 2, 2), ("c", 0, 0, 2))) == ()
    assert _codes(problem, _schedule(("a", 0, 1, 1), ("b", 0, 1, 1), ("c", 0, 0, 2))) == [
        "capacity",
        "capacity",
    ]


# -------------------------------------------------------------------- setups


def test_setup_times() -> None:
    problem = Problem(
        "setups",
        tasks=(Task.on("a", "m", 2), Task.on("b", "m", 2), Task.on("c", "n", 1)),
        resources=(Resource("m"), Resource("n")),
        setups=SetupTimes({("m", None, "a"): 2, ("m", "a", "b"): 3, ("m", "b", "a"): 1}),
    )
    assert check_schedule(problem, _schedule(("a", 0, 2, 4), ("b", 0, 7, 9), ("c", 0, 0, 1))) == ()
    assert check_schedule(problem, _schedule(("b", 0, 0, 2), ("a", 0, 3, 5), ("c", 0, 0, 1))) == ()
    first = _single(problem, _schedule(("a", 0, 1, 3), ("b", 0, 6, 8), ("c", 0, 0, 1)), "setup")
    assert (first.task, first.resource, first.amount) == ("a", "m", 1.0)
    between = _single(problem, _schedule(("a", 0, 2, 4), ("b", 0, 6, 8), ("c", 0, 0, 1)), "setup")
    assert (between.task, between.amount) == ("b", 1.0)
    assert "before its setup completes at 7.0" in between.message


# --------------------------------------------------------------- permutation


def _flow_shop(times: dict[str, tuple[float, ...]]) -> Problem:
    """Jobs visit machines ``m0``, ``m1``, ... in order, as groups of a permutation problem."""
    tasks: list[Task] = []
    dependencies: list[Dependency] = []
    machines = len(next(iter(times.values())))
    for job, durations in times.items():
        for machine, duration in enumerate(durations):
            tasks.append(Task.on(f"{job}{machine}", f"m{machine}", duration, group=job))
            if machine:
                dependencies.append(Dependency(f"{job}{machine - 1}", f"{job}{machine}"))
    return Problem(
        "flow",
        tasks=tuple(tasks),
        resources=tuple(Resource(f"m{m}") for m in range(machines)),
        dependencies=tuple(dependencies),
        permutation=True,
    )


def test_one_group_order_on_every_machine_is_a_permutation_schedule() -> None:
    problem = _flow_shop({"A": (1, 3), "B": (1, 2)})
    same = _schedule(("A0", 0, 0, 1), ("B0", 0, 1, 2), ("A1", 0, 1, 4), ("B1", 0, 4, 6))
    assert check_schedule(problem, same) == ()


def test_contradictory_machine_orders_are_a_permutation_violation() -> None:
    problem = _flow_shop({"A": (1, 3), "B": (1, 2)})
    crossed = _schedule(("A0", 0, 0, 1), ("B0", 0, 1, 2), ("B1", 0, 2, 4), ("A1", 0, 4, 7))
    violations = check_schedule(problem, crossed)
    assert [v.code for v in violations] == ["permutation"]
    assert "'A', 'B'" in violations[0].message


def test_a_cycle_through_three_groups_is_found() -> None:
    problem = _flow_shop({"A": (1, 1), "B": (1, 1), "C": (1, 1)})
    cyclic = _schedule(
        ("A0", 0, 0, 1),
        ("B0", 0, 1, 2),
        ("C0", 0, 2, 3),
        ("B1", 0, 2, 3),
        ("C1", 0, 3, 4),
        ("A1", 0, 4, 5),
    )
    violations = check_schedule(problem, cyclic)
    assert [v.code for v in violations] == ["permutation"]
    assert "'A', 'B', 'C'" in violations[0].message


def test_tied_zero_duration_tasks_may_take_either_order() -> None:
    problem = _flow_shop({"A": (0, 3), "B": (0, 2)})
    for later in (("A1", 0, 0, 3), ("B1", 0, 3, 5)), (("B1", 0, 0, 2), ("A1", 0, 2, 5)):
        tied = _schedule(("A0", 0, 0, 0), ("B0", 0, 0, 0), *later)
        assert check_schedule(problem, tied) == ()


def test_zero_duration_tasks_at_different_instants_fix_the_order() -> None:
    problem = _flow_shop({"A": (0, 3), "B": (0, 2)})
    apart = _schedule(("A0", 0, 0, 0), ("B0", 0, 1, 1), ("B1", 0, 1, 3), ("A1", 0, 3, 6))
    assert _codes(problem, apart) == ["permutation"]
    agreed = _schedule(("A0", 0, 0, 0), ("B0", 0, 1, 1), ("A1", 0, 0, 3), ("B1", 0, 3, 5))
    assert check_schedule(problem, agreed) == ()


def test_an_instant_at_the_start_of_a_task_comes_before_it() -> None:
    problem = _flow_shop({"A": (0, 3), "B": (2, 1)})
    # On m0, A's instant at 2 precedes B's task that starts at 2 ...
    before = _schedule(("B0", 0, 2, 4), ("A0", 0, 2, 2), ("B1", 0, 4, 5), ("A1", 0, 5, 8))
    assert _codes(problem, before) == ["permutation"]
    # ... and an instant at the end of a task follows it.
    after = _schedule(("B0", 0, 0, 2), ("A0", 0, 2, 2), ("B1", 0, 2, 3), ("A1", 0, 3, 6))
    assert check_schedule(problem, after) == ()


def test_tasks_without_a_group_are_groups_of_their_own() -> None:
    problem = Problem(
        "solo",
        tasks=(Task.on("a", "m0", 1), Task.on("b", "m0", 1), Task.on("c", "m1", 1)),
        resources=(Resource("m0"), Resource("m1")),
        permutation=True,
    )
    assert check_schedule(problem, _schedule(("a", 0, 1, 2), ("b", 0, 0, 1), ("c", 0, 0, 1))) == ()


# -------------------------------------------------------------- properties


@pytest.mark.parametrize("options", [{}, {"setups": True}, {"network": True}, {"pools": 2}])
def test_the_checker_agrees_with_the_decoder_and_catches_perturbations(
    options: dict[str, Any],
) -> None:
    for seed in range(6):
        problem = random_problem(seed, tasks=10, **options)
        decoder = Decoder(problem)
        rng = random.Random(seed)
        order = random_order(problem, rng)
        schedule = decoder.schedule(order, random_modes(problem, rng))
        assert check_schedule(problem, schedule) == ()
        # Delaying every task by the same amount keeps a schedule feasible.
        shifted = Schedule.of(
            replace(item, start=item.start + 7.25, end=item.end + 7.25)
            for item in schedule.assignments
        )
        assert check_schedule(problem, shifted) == ()
        # Starting a successor before its predecessor ends breaks precedence.
        by_task = schedule.by_task
        for dependency in problem.dependencies:
            before, after = by_task[dependency.before], by_task[dependency.after]
            early = before.end - 0.5 - after.start
            moved = replace(after, start=after.start + early, end=after.end + early)
            broken = Schedule.of(
                moved if item.task == after.task else item for item in schedule.assignments
            )
            codes = _codes(problem, broken)
            assert "precedence" in codes
            break
