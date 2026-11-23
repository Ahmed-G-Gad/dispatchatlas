"""The Giffler-Thompson rules schedule the shortest operation, or the most work left, once."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from dispatchatlas.core import (
    Problem,
    Schedule,
    SolveResult,
    StopCriteria,
    StopReason,
    UnsupportedProblemError,
    check_schedule,
)
from dispatchatlas.references import bibliography
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from dispatchatlas.solvers.shop.rules import GtMostWorkRemaining, GtShortestProcessingTime
from tests.helpers import random_problem, small_job_shop
from tests.solvers.shop.contract import check_contract
from tests.solvers.shop.instances import random_job_shop, ta01
from tests.solvers.shop.oracles import job_shop, naive_giffler_thompson

type Rule = type[GtShortestProcessingTime] | type[GtMostWorkRemaining]

RULES: list[Rule] = [GtShortestProcessingTime, GtMostWorkRemaining]

# The small job shop has the tasks j0-o0, j0-o1, j0-o2, j1-o0, ..., j2-o2 at the indices 0 to 8:
#   j0: m0 3, m1 2, m2 2     j1: m0 2, m2 1, m1 4     j2: m1 3, m0 2, m2 3.
# The start of every task, by task index, and the makespan, worked by hand as follows.
HAND_COMPUTED = [
    # Shortest processing time ranks the tasks by (time, job, operation): j1-o1 (1), then
    # j0-o1, j0-o2, j1-o0, j2-o1 (2), then j0-o0, j2-o0, j2-o2 (3), then j1-o2 (4).
    #   m0 at the start: j0-o0 (3) and j1-o0 (2) can start before C* = 2, and j1-o0 goes, [0, 2).
    #   j1-o1 [2, 3) is the only task on m2 that can start before C* = 3, which j2-o0 also attains.
    #   j2-o0 [0, 3), the only task on m1 that can start before C* = 3.
    #   m0 at C* = 5: j0-o0 (3) and j2-o1 (2) can start before it, and j2-o1 goes, [3, 5).
    #   j1-o2 [3, 7); then j0-o0 and j2-o2 both complete at 8 and j0-o0 ranks first, [5, 8);
    #   then j2-o2 [5, 8), j0-o1 [8, 10), j0-o2 [10, 12).
    pytest.param(
        GtShortestProcessingTime,
        (5, 8, 10, 0, 2, 3, 0, 3, 5),
        12,
        id="shortest-processing-time",
    ),
    # Most work remaining counts the task's own time: j0-o0 7, j0-o1 4, j0-o2 2; j1-o0 7,
    # j1-o1 5, j1-o2 4; j2-o0 8, j2-o1 5, j2-o2 3. The ranks sort by (minus the work, job,
    # operation).
    #   m0 at the start: j0-o0 and j1-o0 can start before C* = 2 and have 7 left each, and
    #   the lower job goes: j0-o0 [0, 3). Then j2-o0 [0, 3).
    #   j0-o1, j1-o0 and j2-o1 all complete at 5; j1-o0 ranks first, so m0 is the machine
    #   and the conflict set {j1-o0, j2-o1} gives j1-o0 [3, 5); then j0-o1 [3, 5).
    #   j1-o1 [5, 6); j2-o1 [5, 7); on m2 at C* = 8, j0-o2 (2 left) and j2-o2 (3 left) can
    #   both start before it, and j2-o2 goes, [7, 10); then j1-o2 [6, 10) and j0-o2 [10, 12).
    pytest.param(
        GtMostWorkRemaining,
        (0, 3, 10, 3, 5, 6, 0, 5, 7),
        12,
        id="most-work-remaining",
    ),
]


def _schedule(result: SolveResult) -> Schedule:
    assert result.schedule is not None
    return result.schedule


def _starts(problem: Problem, result: SolveResult) -> tuple[float, ...]:
    """Return the start of every task of ``problem``, by task index."""
    by_task = _schedule(result).by_task
    return tuple(by_task[task.id].start for task in problem.tasks)


def _ranks(shop: JobShop, value: Callable[[JobShop, int, int], float]) -> list[int]:
    """Rank the tasks by the documented key ``(value, job, operation)``, from its definition."""
    key = {}
    for job, route in enumerate(shop.operations):
        for step, task in enumerate(route):
            key[task] = (value(shop, job, step), job, step)
    ranks = [0] * shop.size
    for position, task in enumerate(sorted(range(shop.size), key=key.__getitem__)):
        ranks[task] = position
    return ranks


def _shortest(shop: JobShop, job: int, step: int) -> float:
    """The processing time of a job's operation."""
    return shop.duration[shop.operations[job][step]]


def _most_work(shop: JobShop, job: int, step: int) -> float:
    """Minus the processing time of a job's operation and of the operations after it."""
    return -sum(shop.duration[task] for task in shop.operations[job][step:])


VALUES: dict[Rule, Callable[[JobShop, int, int], float]] = {
    GtShortestProcessingTime: _shortest,
    GtMostWorkRemaining: _most_work,
}


def _shops() -> list[JobShop]:
    """Taillard's ta01, random job shops with and without releases and revisits, zero times."""
    zero = job_shop("zero", [[(0, 0), (1, 3)], [(0, 2), (1, 0)], [(1, 0), (0, 1)]])
    return [
        ta01(),
        JobShop.from_problem(zero),
        *(
            JobShop.from_problem(
                random_job_shop(
                    index,
                    jobs=2 + index % 5,
                    machines=2 + index % 3,
                    releases=index % 2 == 0,
                    revisits=index % 3 == 0,
                )
            )
            for index in range(24)
        ),
    ]


# ------------------------------------------------------------- hand-computed


@pytest.mark.parametrize(("rule", "starts", "makespan"), HAND_COMPUTED)
def test_each_rule_gives_its_hand_computed_schedule_of_the_small_job_shop(
    rule: Rule, starts: tuple[float, ...], makespan: float
) -> None:
    problem = small_job_shop()
    result = rule().solve(problem, StopCriteria(), seed=0)
    assert _starts(problem, result) == starts
    assert result.value == makespan
    assert result.objectives == {"makespan": makespan}
    assert check_schedule(problem, _schedule(result)) == ()
    assert result.evaluations == 1
    assert result.stop_reason is StopReason.COMPLETED


def test_the_rules_choose_different_operations_when_their_values_differ() -> None:
    # j0 is a long task on m0 and then a short one on m1; j1 is the other way round. At the
    # start j0-o0 (5) and j1-o0 (1) can both start before C* = 1. Shortest processing time takes
    # j1-o0. Most work remaining counts the operation's own time: j0 has 5 + 1 = 6 left and j1 has
    # 1 + 3 = 4, so j0-o0 goes first, although it completes last; leaving the operation's own
    # time out would give 1 against 3, and the other order.
    problem = job_shop("own-time", [[(0, 5), (1, 1)], [(0, 1), (1, 3)]])
    shortest = GtShortestProcessingTime().solve(problem, StopCriteria())
    assert _starts(problem, shortest) == (1, 6, 0, 1)
    most_work = GtMostWorkRemaining().solve(problem, StopCriteria())
    assert _starts(problem, most_work) == (0, 5, 5, 6)


@pytest.mark.parametrize("rule", RULES)
def test_equal_values_go_by_job_index(rule: Rule) -> None:
    # Three jobs of one task, all of 2 on one machine: the order is the job order.
    problem = job_shop("equal", [[(0, 2)], [(0, 2)], [(0, 2)]])
    assert _starts(problem, rule().solve(problem, StopCriteria())) == (0, 2, 4)
    # Two jobs of two tasks, each pair with the same value at every step: j0 first on m0.
    problem = job_shop("equal-two", [[(0, 2), (1, 2)], [(0, 2), (1, 2)]])
    assert _starts(problem, rule().solve(problem, StopCriteria())) == (0, 2, 2, 4)


@pytest.mark.parametrize("rule", RULES)
def test_a_rule_chooses_only_among_the_operations_that_can_start_before_c_star(
    rule: Rule,
) -> None:
    # j0-o0 (2) is released at 5, on m0, and its job has 12 left; j1-o0 (3) is on m0 too. Both
    # rules rank j0-o0 first, yet j1-o0 completes first, at 3, and j0-o0 cannot start before
    # then, so only j1-o0 is in the conflict set. Choosing j0-o0 would start j1-o0 at 7.
    problem = job_shop("release", [[(0, 2), (1, 10)], [(0, 3)]], **{"j0-o0": 5})
    assert _starts(problem, rule().solve(problem, StopCriteria())) == (5, 7, 0)


# ------------------------------------------------------------- the definition


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_each_rule_equals_the_textbook_procedure_on_the_ranks_of_its_definition(
    rule: Rule, shop: JobShop
) -> None:
    problem = shop.problem
    expected = naive_giffler_thompson(shop, _ranks(shop, VALUES[rule]))
    result = rule().solve(problem, StopCriteria(), seed=0)
    assert _starts(problem, result) == expected.starts
    assert result.value == expected.makespan
    assert check_schedule(problem, _schedule(result)) == ()


def test_shortest_processing_time_ranks_conflict_with_the_job_order() -> None:
    # No job sequence ranks a job's operations out of route order, but this rule does:
    # j1-o1 (1) ranks ahead of its job predecessor j1-o0 (2).
    shop = JobShop.from_problem(small_job_shop())
    ranks = _ranks(shop, _shortest)
    assert sorted(ranks) == list(range(shop.size))
    assert ranks[4] < ranks[3]


# ------------------------------------------------------------- determinism and the contract


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("problem", [small_job_shop(), ta01().problem], ids=["3x3", "ta01"])
def test_a_rule_is_deterministic_and_evaluates_one_schedule(rule: Rule, problem: Problem) -> None:
    solver = rule()
    first = solver.solve(problem, StopCriteria(), seed=0)
    again = solver.solve(problem, StopCriteria(), seed=0)
    other = solver.solve(problem, StopCriteria(max_evaluations=500), seed=123)
    assert first.schedule == again.schedule == other.schedule
    assert first.objectives == again.objectives == other.objectives
    assert first.evaluations == again.evaluations == other.evaluations == 1
    assert first.stop_reason is other.stop_reason is StopReason.COMPLETED
    assert {(point.evaluations, point.value) for point in first.trace} == {(1, first.value)}
    assert first.extras == {}


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_each_rule_passes_the_contract(rule: Rule, shop: JobShop) -> None:
    result = check_contract(rule, shop.problem)
    assert result.evaluations == 1


@pytest.mark.parametrize("rule", RULES)
def test_a_rule_completes_within_a_limit_of_one_evaluation(rule: Rule) -> None:
    result = check_contract(rule, small_job_shop(), evaluations=1)
    assert result.stop_reason is StopReason.COMPLETED


@pytest.mark.parametrize("rule", RULES)
def test_a_problem_without_the_job_shop_shape_is_refused(rule: Rule) -> None:
    problem = random_problem(1, tasks=8, max_modes=1, pools=0)
    assert not rule.info.accepts(problem)
    with pytest.raises(UnsupportedProblemError):
        rule().solve(problem, StopCriteria(), seed=0)


# ------------------------------------------------------------- metadata


@pytest.mark.parametrize(
    ("rule", "solver_id", "sources"),
    [
        (GtShortestProcessingTime, "gt-shortest-processing-time", {"smith1956various"}),
        (
            GtMostWorkRemaining,
            "gt-most-work-remaining",
            {"demirkol1998benchmarks", "nguyen2013computational"},
        ),
    ],
)
def test_each_rule_describes_itself(rule: Rule, solver_id: str, sources: set[str]) -> None:
    info = rule.info
    assert info.id == solver_id
    assert info.family == "dispatching-rule"
    assert info.shape == "job-shop"
    assert info.supports == JOB_SHOP_FEATURES
    assert not info.stochastic
    assert not info.iterative
    assert info.summary.endswith(".")
    assert "giffler1960algorithms" in info.references
    assert sources <= set(info.references)
    assert set(info.references) <= set(bibliography())
