"""The Werner-Winkler insertion follows the paper's rules and evaluates one schedule."""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from typing import Any

import pytest

from dispatchatlas.core import (
    ModelError,
    Problem,
    Schedule,
    SolveResult,
    StopCriteria,
    StopReason,
    UnsupportedProblemError,
    check_schedule,
)
from dispatchatlas.references import bibliography
from dispatchatlas.solvers.shop.insertion import WernerWinklerInsertion, insertion_costs
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from tests.helpers import random_problem, small_job_shop
from tests.solvers.shop.contract import check_contract
from tests.solvers.shop.instances import random_job_shop, ta01
from tests.solvers.shop.oracles import (
    job_shop,
    naive_heads_tails,
    naive_insertion,
    naive_insertion_costs,
)

# Example 1 of Werner and Winkler (1995, pp. 193-198): three jobs on the four machines M1 to M4,
# which are m0 to m3 here, with the processing times
#         M1  M2  M3  M4
#     J1  12  14   9   8       route M1, M4, M2, M3
#     J2  17  10  12  11       route M3, M1, M2, M4
#     J3   6  15   3  13       route M2, M3, M4, M1
# The tasks j0-o0 to j0-o3 are J1's operations in route order, then J2's, then J3's, at the
# indices 0 to 3, 4 to 7 and 8 to 11.
EXAMPLE = job_shop(
    "werner-winkler-example-1",
    [
        [(0, 12), (3, 8), (1, 14), (2, 9)],
        [(2, 12), (0, 17), (1, 10), (3, 11)],
        [(1, 15), (2, 3), (3, 13), (0, 6)],
    ],
)

# The machine orders of the paper's final latin rectangle (p. 198), in task indices. Its columns
# order the jobs on M1 as J1, J2, J3, on M2 as J3, J1, J2, on M3 as J2, J3, J1, and on M4 as
# J1, J3, J2.
EXAMPLE_ORDERS = ((0, 5, 11), (8, 2, 6), (4, 9, 3), (1, 10, 7))

# The small job shop has the tasks j0-o0, j0-o1, j0-o2, j1-o0, ..., j2-o2 at the indices 0 to 8:
#   j0: m0 3, m1 2, m2 2     j1: m0 2, m2 1, m1 4     j2: m1 3, m0 2, m2 3,
# so that j0 has 7 in all, j1 has 7 and j2 has 8. Worked by hand: j2 is inserted first, on
# m0 (task 7), m1 (task 6) and m2 (task 8). The other tasks follow by non-increasing time, then
# job, then operation: j1-o2 (4), j0-o0 (3), then j0-o1, j0-o2, j1-o0 (2), then j1-o1 (1), that is
# the tasks 5, 0, 1, 2, 3, 4. Each step lists the machine orders before it, the task, and the cost
# of each position, which is the head, the time and the tail of the task there. The graph holds
# every job arc from the start, so the heads at the first step are j0: 0, 3, 5; j1: 0, 2, 3;
# j2: 0, 3, 5 and the tails j0: 4, 2, 0; j1: 5, 4, 0; j2: 5, 3, 0.
#   task 5, on m1 beside task 6: head 3, time 4, and before task 6 the tail is 3 + 5 = 8, so
#     3 + 4 + 8 = 15, after it the tail is 0, so 3 + 4 + 0 = 7. The position after is taken.
#   task 0: 0 + 3 + 5 = 8 before task 7, 5 + 3 + 4 = 12 after it. Before is taken.
#   task 1: 13, 9, 11 at the three positions of m1; position 1 is taken.
#   task 2 on m2: 10 and 10, a tie, and the earlier position is taken.
#   task 3 on m0: 12, 10, 12; position 1 is taken. Task 4 on m2: 11, 12, 15; position 0 is taken.
SMALL_STEPS = [
    ([[7], [6], [8]], 5, {0: 15.0, 1: 7.0}),
    ([[7], [6, 5], [8]], 0, {0: 8.0, 1: 12.0}),
    ([[0, 7], [6, 5], [8]], 1, {0: 13.0, 1: 9.0, 2: 11.0}),
    ([[0, 7], [6, 1, 5], [8]], 2, {0: 10.0, 1: 10.0}),
    ([[0, 7], [6, 1, 5], [2, 8]], 3, {0: 12.0, 1: 10.0, 2: 12.0}),
    ([[0, 3, 7], [6, 1, 5], [2, 8]], 4, {0: 11.0, 1: 12.0, 2: 15.0}),
]
# The final orders of the steps above, and the semi-active schedule they give: every task starts
# at its head, and the makespan is 11, the optimum of this job shop.
SMALL_ORDERS = ((0, 3, 7), (6, 1, 5), (4, 2, 8))
SMALL_STARTS = (0.0, 3.0, 6.0, 3.0, 5.0, 6.0, 0.0, 5.0, 8.0)


def _schedule(result: SolveResult) -> Schedule:
    assert result.schedule is not None
    return result.schedule


def _starts(problem: Problem, result: SolveResult) -> tuple[float, ...]:
    """Return the start of every task of ``problem``, by task index."""
    by_task = _schedule(result).by_task
    return tuple(by_task[task.id].start for task in problem.tasks)


def _machine_orders(shop: JobShop, starts: Sequence[float]) -> tuple[tuple[int, ...], ...]:
    """Return the tasks of each machine by start time, which orders them when none takes no time."""
    return tuple(
        tuple(
            sorted(
                (task for task in range(shop.size) if shop.machine[task] == machine),
                key=lambda task: (starts[task], task),
            )
        )
        for machine in range(shop.machines)
    )


def _random_shops() -> list[JobShop]:
    """Small random job shops with and without releases and revisits, and one with zero times."""
    zero = job_shop("zero", [[(0, 0), (1, 3)], [(0, 2), (1, 0)], [(1, 0), (0, 1)]])
    return [
        JobShop.from_problem(zero),
        *(
            JobShop.from_problem(
                random_job_shop(
                    index,
                    jobs=2 + index % 7,
                    machines=2 + index % 4,
                    releases=index % 2 == 0,
                    revisits=index % 3 == 0,
                )
            )
            for index in range(36)
        ),
    ]


def _contract_shops() -> list[JobShop]:
    return [
        JobShop.from_problem(small_job_shop()),
        JobShop.from_problem(EXAMPLE),
        ta01(),
        *_random_shops(),
    ]


# ------------------------------------------------------------- hand-computed


@pytest.mark.parametrize(("orders", "task", "costs"), SMALL_STEPS)
def test_each_step_of_the_small_job_shop_costs_its_positions_as_worked_by_hand(
    orders: list[list[int]], task: int, costs: dict[int, float]
) -> None:
    shop = JobShop.from_problem(small_job_shop())
    assert insertion_costs(shop, orders, task) == costs
    assert naive_insertion_costs(shop, orders, task) == costs


def test_the_small_job_shop_insertion_gives_its_hand_computed_schedule() -> None:
    problem = small_job_shop()
    result = WernerWinklerInsertion().solve(problem, StopCriteria(), seed=0)
    starts = _starts(problem, result)
    assert starts == SMALL_STARTS
    assert _machine_orders(JobShop.from_problem(problem), starts) == SMALL_ORDERS
    assert result.value == 11
    assert result.objectives == {"makespan": 11.0}
    assert check_schedule(problem, _schedule(result)) == ()
    # The number of positions costed by the steps above: 2 + 2 + 3 + 2 + 3 + 3.
    assert result.extras == {"position_costs": 15}
    assert result.evaluations == 1
    assert result.stop_reason is StopReason.COMPLETED


def test_the_example_of_the_paper_costs_its_first_insertions_as_the_paper_does() -> None:
    shop = JobShop.from_problem(EXAMPLE)
    # J2, the job with the greatest total, is inserted first: 17 + 10 + 12 + 11 = 50, against 43
    # for J1 and 37 for J3. The operations follow by non-increasing time: (3,2) of 15, (1,2) of 14,
    # (3,4) of 13, ... in the paper's notation, the tasks 8, 2, 10 here.
    # (3,2): the costs of its two positions, 37 and 76 (p. 197).
    assert insertion_costs(shop, [[5], [6], [4], [7]], 8) == {0: 37.0, 1: 76.0}
    # (1,2), into the partial schedule that kept (3,2) first on M2: 71, 55 and 62 (p. 197).
    assert insertion_costs(shop, [[5], [8, 6], [4], [7]], 2) == {0: 71.0, 1: 55.0, 2: 62.0}
    # (3,4), into the one that put (1,2) in the middle of M2: 42 for the first position, which is
    # the one taken (p. 198).
    costs = insertion_costs(shop, [[5], [8, 2, 6], [4], [7]], 10)
    assert costs[0] == 42
    assert min(costs, key=costs.__getitem__) == 0
    assert costs[1] > 42


def test_the_insertion_reaches_the_final_schedule_of_the_example_of_the_paper() -> None:
    problem = EXAMPLE
    result = WernerWinklerInsertion().solve(problem, StopCriteria(), seed=0)
    starts = _starts(problem, result)
    # The final latin rectangle of the paper (p. 198), whose makespan 55 is optimal.
    assert _machine_orders(JobShop.from_problem(problem), starts) == EXAMPLE_ORDERS
    assert result.value == 55
    assert check_schedule(problem, _schedule(result)) == ()
    assert result.evaluations == 1


def test_one_machine_takes_its_jobs_in_increasing_order_of_time_by_the_earliest_position() -> None:
    # On one machine every position costs the same, the sum of all the times, so each
    # operation goes to the front: the first job, of 5, ends up last, and the others, inserted
    # in the order 4, 3, 2, come out in the order 2, 3, 4. This is the tie rule of positions.
    problem = job_shop("one-machine", [[(0, 3)], [(0, 5)], [(0, 2)], [(0, 4)]])
    result = WernerWinklerInsertion().solve(problem, StopCriteria(), seed=0)
    assert _starts(problem, result) == (2, 9, 0, 5)
    assert result.value == 14
    assert result.extras == {"position_costs": 2 + 3 + 4}


def test_a_release_time_enters_the_heads_and_so_the_costs() -> None:
    # j1-o0 is released at 10. Before j0-o0 it would cost 10 + 2 + 4 = 16, after it 10 + 2 = 12.
    # Without the release both positions would cost 6, and the earlier one would be taken.
    problem = job_shop("release", [[(0, 4)], [(0, 2)]], **{"j1-o0": 10})
    result = WernerWinklerInsertion().solve(problem, StopCriteria(), seed=0)
    assert _starts(problem, result) == (0, 10)
    assert result.value == 12
    shop = JobShop.from_problem(problem)
    assert insertion_costs(shop, [[0]], 1) == {0: 16.0, 1: 12.0}


def test_a_job_that_revisits_a_machine_is_treated_by_the_same_rules() -> None:
    # j0 visits m0, m1, m0 (2 each), j1 visits m1 (3), then m0 (1). j0 comes first, with
    # m0: [a, c] and m1: [b]. Then d (3) goes before b on m1, 0 + 3 + 4 = 7 against 4 + 3 + 1 = 8,
    # and e (1) goes between a and c on m0: 3 + 1 + 2 = 6, against 10 before a and 8 after c.
    problem = job_shop("revisit", [[(0, 2), (1, 2), (0, 2)], [(1, 3), (0, 1)]])
    shop = JobShop.from_problem(problem)
    assert insertion_costs(shop, [[0, 2], [1]], 3) == {0: 7.0, 1: 8.0}
    assert insertion_costs(shop, [[0, 2], [3, 1]], 4) == {0: 10.0, 1: 6.0, 2: 8.0}
    result = WernerWinklerInsertion().solve(problem, StopCriteria(), seed=0)
    assert _starts(problem, result) == (0, 3, 5, 0, 3)
    assert result.value == 7
    assert check_schedule(problem, _schedule(result)) == ()


@pytest.mark.parametrize(
    ("routes", "starts", "makespan"),
    [([[(0, 4)]], (0,), 4), ([[(0, 2), (1, 3)]], (0, 2), 5), ([[(0, 2)], [(1, 3)]], (0, 0), 3)],
    ids=["one-operation", "one-job", "independent-jobs"],
)
def test_a_shop_without_a_choice_is_its_first_job_or_its_jobs_side_by_side(
    routes: list[list[tuple[int, float]]], starts: tuple[float, ...], makespan: float
) -> None:
    problem = job_shop("tiny", routes)
    result = WernerWinklerInsertion().solve(problem, StopCriteria(), seed=0)
    assert _starts(problem, result) == starts
    assert result.value == makespan
    assert result.evaluations == 1


# ------------------------------------------------------------- feasible positions


@pytest.mark.parametrize(
    ("routes", "orders", "task", "costs"),
    [
        # a, b, c of j0 on m0, m1, m0: a is before b before c, so a cannot go after c.
        ([[(0, 2), (1, 2), (0, 2)]], [[2], [1]], 0, {0: 6.0}),
        # ... and c cannot go before a.
        ([[(0, 2), (1, 2), (0, 2)]], [[0], [1]], 2, {1: 6.0}),
    ],
    ids=["a-before-its-job-successor", "c-after-its-job-predecessor"],
)
def test_positions_that_break_the_route_of_a_job_are_not_offered(
    routes: list[list[tuple[int, float]]],
    orders: list[list[int]],
    task: int,
    costs: dict[int, float],
) -> None:
    shop = JobShop.from_problem(job_shop("route", routes))
    assert insertion_costs(shop, orders, task) == costs


@pytest.mark.parametrize(
    ("orders", "positions"),
    [
        # d (m1) before b: no path from a to e, so e may go anywhere on m0.
        ([[0, 2], [3, 1]], {0, 1, 2}),
        # b before d: a, b, d, e is a path, so e cannot go before a.
        ([[0, 2], [1, 3]], {1, 2}),
    ],
)
def test_positions_that_close_a_cycle_through_other_machines_are_not_offered(
    orders: list[list[int]], positions: set[int]
) -> None:
    # j0: a (m0), b (m1), c (m0); j1: d (m1), e (m0): the tasks 0 to 2 and 3, 4.
    shop = JobShop.from_problem(job_shop("cycle", [[(0, 2), (1, 2), (0, 2)], [(1, 3), (0, 1)]]))
    costs = insertion_costs(shop, orders, 4)
    assert set(costs) == positions
    assert costs == naive_insertion_costs(shop, orders, 4)
    assert list(costs) == sorted(costs)


@pytest.mark.parametrize("shop", _random_shops()[:12], ids=repr)
def test_the_costs_of_every_insertion_equal_the_brute_force_costs(shop: JobShop) -> None:
    _, steps = naive_insertion(shop)
    for orders, task, costs in steps:
        assert costs, "a partial schedule always has a position"
        assert insertion_costs(shop, orders, task) == costs


def test_the_orders_are_not_changed() -> None:
    shop = JobShop.from_problem(small_job_shop())
    orders = [[0, 7], [6, 1, 5], [8]]
    before = copy.deepcopy(orders)
    insertion_costs(shop, orders, 2)
    assert orders == before
    assert insertion_costs(shop, ((0, 7), (6, 1, 5), (8,)), 2) == {0: 10.0, 1: 10.0}


# ------------------------------------------------------------- the algorithm against brute force


@pytest.mark.parametrize("shop", _random_shops(), ids=repr)
def test_the_insertion_equals_the_brute_force_insertion(shop: JobShop) -> None:
    orders, steps = naive_insertion(shop)
    problem = shop.problem
    result = WernerWinklerInsertion().solve(problem, StopCriteria(), seed=0)
    paths = naive_heads_tails(shop, orders)
    assert paths is not None
    # The schedule is the semi-active schedule of the orders: every task starts at its head.
    assert _starts(problem, result) == tuple(paths[0])
    assert result.extras == {"position_costs": sum(len(costs) for _, _, costs in steps)}
    assert result.evaluations == 1
    assert check_schedule(problem, _schedule(result)) == ()


# ------------------------------------------------------------- what inputs are refused


@pytest.mark.parametrize(
    ("orders", "task", "message"),
    [
        ([[], []], 0, "expected 3 machine orders, got 2"),
        ([[], [], []], 9, "task 9 is not an operation of problem 'small-job-shop'"),
        ([[], [], []], -1, "task -1 is not an operation of problem 'small-job-shop'"),
        ([[1], [], []], 0, "machine order 0 lists task 1, which is unknown, listed twice, or"),
        ([[3, 3], [], []], 0, "machine order 0 lists task 3, which is unknown, listed twice, or"),
        ([[99], [], []], 0, "machine order 0 lists task 99, which is unknown, listed twice, or"),
        ([[0], [], []], 0, "task 0 is already in the machine orders"),
    ],
)
def test_orders_that_are_not_a_partial_schedule_of_the_job_shop_are_refused(
    orders: Any, task: int, message: str
) -> None:
    shop = JobShop.from_problem(small_job_shop())
    with pytest.raises(ModelError, match=message):
        insertion_costs(shop, orders, task)


def test_orders_that_close_a_cycle_with_the_jobs_are_refused() -> None:
    # j0: a (m0) then b (m1); j1: c (m1) then d (m0); a third job to insert. The orders d, a on
    # m0 and b, c on m1 give the cycle a, b, c, d, a.
    shop = JobShop.from_problem(job_shop("cycle", [[(0, 1), (1, 1)], [(1, 1), (0, 1)], [(2, 1)]]))
    with pytest.raises(ModelError, match="the machine orders and the jobs form a cycle"):
        insertion_costs(shop, [[3, 0], [1, 2], []], 4)


# ------------------------------------------------------------- determinism and the contract


@pytest.mark.parametrize("problem", [small_job_shop(), ta01().problem], ids=["3x3", "ta01"])
def test_the_insertion_is_deterministic_and_evaluates_one_schedule(problem: Problem) -> None:
    solver = WernerWinklerInsertion()
    first = solver.solve(problem, StopCriteria(), seed=0)
    again = solver.solve(problem, StopCriteria(), seed=0)
    other = solver.solve(problem, StopCriteria(max_evaluations=500), seed=123)
    assert first.schedule == again.schedule == other.schedule
    assert first.objectives == again.objectives == other.objectives
    assert first.extras == again.extras == other.extras
    assert first.evaluations == again.evaluations == other.evaluations == 1
    assert first.stop_reason is other.stop_reason is StopReason.COMPLETED
    assert {(point.evaluations, point.value) for point in first.trace} == {(1, first.value)}


@pytest.mark.parametrize("shop", _contract_shops(), ids=repr)
def test_the_insertion_passes_the_contract(shop: JobShop) -> None:
    result = check_contract(WernerWinklerInsertion, shop.problem)
    assert result.evaluations == 1
    assert set(result.extras) == {"position_costs"}


def test_the_insertion_completes_within_a_limit_of_one_evaluation() -> None:
    result = check_contract(WernerWinklerInsertion, small_job_shop(), evaluations=1)
    assert result.stop_reason is StopReason.COMPLETED


def test_a_problem_without_the_job_shop_shape_is_refused() -> None:
    problem = random_problem(1, tasks=8, max_modes=1, pools=0)
    assert not WernerWinklerInsertion.info.accepts(problem)
    with pytest.raises(UnsupportedProblemError):
        WernerWinklerInsertion().solve(problem, StopCriteria(), seed=0)


def test_the_insertion_of_ta01_is_feasible_and_takes_seconds_at_most() -> None:
    problem = ta01().problem
    started = time.perf_counter()
    result = WernerWinklerInsertion().solve(problem, StopCriteria(), seed=0)
    # A generous bound: 225 insertions, each with a pass over the 225 operations.
    assert time.perf_counter() - started < 30
    assert check_schedule(problem, _schedule(result)) == ()
    # ta01's optimum is 1231, so no feasible schedule is shorter.
    assert result.value is not None
    assert result.value >= 1231


# ------------------------------------------------------------- metadata


def test_the_insertion_is_labelled_as_the_basic_form_without_a_beam() -> None:
    info = WernerWinklerInsertion.info
    assert info.id == "werner-winkler-insertion"
    assert "basic" in info.name.lower()
    assert "without beam search" in info.summary
    assert "no beam search" in (WernerWinklerInsertion.__doc__ or "")
    assert info.family == "constructive"
    assert info.shape == "job-shop"
    assert info.supports == JOB_SHOP_FEATURES
    assert not info.stochastic
    assert not info.iterative
    assert info.summary.endswith(".")
    assert info.references == ("werner1995insertion",)
    assert set(info.references) <= set(bibliography())
