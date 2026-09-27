import itertools
import random
from collections.abc import Sequence

import pytest

from dispatchatlas.core import (
    Dependency,
    Mode,
    Problem,
    Resource,
    Schedule,
    SolveResult,
    StopCriteria,
    StopReason,
    Task,
    Violation,
    check_schedule,
)
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.constructive import (
    AUTO,
    BeamSearch,
    Items,
    MaxMin,
    MinMin,
    Neh,
    SqueakyWheel,
    Sufferage,
    neh_order,
    score_sequence,
    task_blame,
)
from dispatchatlas.solvers.evaluation import Evaluator, Score
from dispatchatlas.solvers.search import Exhausted, Search
from dispatchatlas.solvers.tree_search import MonteCarloTreeSearch, RankReward
from tests.helpers import random_problem

REGISTRY = default_registry(plugins=False)

# Processing times of four jobs on three machines. Totals 25, 21, 20, 13.
FLOW_SHOP_TIMES = ((8, 9, 8), (8, 9, 4), (3, 9, 8), (3, 2, 8))


def flow_shop(times: Sequence[Sequence[int]] = FLOW_SHOP_TIMES) -> Problem:
    tasks, dependencies = [], []
    for j, row in enumerate(times):
        for m, duration in enumerate(row):
            tasks.append(Task.on(f"j{j}m{m}", f"m{m}", duration, group=f"j{j}"))
            if m:
                dependencies.append(Dependency(f"j{j}m{m - 1}", f"j{j}m{m}"))
    machines = tuple(Resource(f"m{m}") for m in range(len(times[0])))
    return Problem("flow-shop", tuple(tasks), machines, tuple(dependencies), permutation=True)


def job_sequence(problem: Problem, schedule: Schedule | None) -> list[str | None]:
    assert schedule is not None
    first = sorted(schedule.assignments, key=lambda item: (item.start, item.task))
    return [problem.task(item.task).group for item in first if item.task.endswith("m0")]


def unrelated_three_by_two() -> Problem:
    """Three independent tasks, each runnable on either of two machines.

    Execution times (m0, m1): t0 (2, 4), t1 (5, 6), t2 (3, 9).
    """
    times = ((2, 4), (5, 6), (3, 9))
    tasks = tuple(
        Task(f"t{i}", (Mode(a, {"m0": 1}), Mode(b, {"m1": 1}))) for i, (a, b) in enumerate(times)
    )
    return Problem("mapping", tasks, (Resource("m0"), Resource("m1")))


def placements(result: SolveResult) -> dict[str, tuple[int, float, float]]:
    assert result.schedule is not None
    return {item.task: (item.mode, item.start, item.end) for item in result.schedule.assignments}


def value_of(result: SolveResult) -> float:
    assert result.value is not None
    return result.value


def violations(problem: Problem, result: SolveResult) -> list[Violation]:
    """Return every broken constraint of the result except hard-deadline overruns."""
    assert result.schedule is not None
    return [v for v in check_schedule(problem, result.schedule) if v.code != "deadline"]


def optimum_over_orders(problem: Problem) -> float:
    evaluator = Evaluator(problem)
    return min(
        evaluator.evaluate(order)[0].value for order in itertools.permutations(range(problem.size))
    )


# ------------------------------------------------------------------- NEH


def test_neh_builds_the_hand_computed_flow_shop_sequence():
    # Priority by total work: j0, j1, j2, j3.
    # Insert j1 into [j0]: [j1 j0] = 34, [j0 j1] = 30 -> [j0 j1].
    # Insert j2: [j2 j0 j1] = 34, [j0 j2 j1] = 39, [j0 j1 j2] = 43 -> [j2 j0 j1].
    # Insert j3: 37, 40, 41, 42 at positions 0..3 -> [j3 j2 j0 j1], makespan 37.
    problem = flow_shop()
    result = Neh().solve(problem, StopCriteria())
    assert result.value == 37.0
    assert job_sequence(problem, result.schedule) == ["j3", "j2", "j0", "j1"]
    assert violations(problem, result) == []
    # One complete priority order, 2 + 3 partial probes, 4 complete candidates.
    assert result.evaluations == 10
    assert result.stop_reason is StopReason.COMPLETED


def test_neh_reaches_the_optimum_of_the_hand_example():
    problem = flow_shop()
    sequences = [
        [task for j in jobs for task in range(3 * j, 3 * j + 3)]
        for jobs in itertools.permutations(range(4))
    ]
    evaluator = Evaluator(problem)
    assert min(evaluator.evaluate(order)[0].value for order in sequences) == 37.0


def test_neh_order_is_the_expanded_sequence():
    problem = flow_shop()
    order = neh_order(problem)
    assert sorted(order) == list(range(problem.size))
    assert [problem.tasks[task].group for task in order[::3]] == ["j3", "j2", "j0", "j1"]


def test_neh_returns_the_priority_order_when_the_budget_ends_early():
    problem = flow_shop()
    result = Neh().solve(problem, StopCriteria(max_evaluations=4))
    assert result.evaluations == 4
    assert result.stop_reason is StopReason.EVALUATION_LIMIT
    assert result.value == 51.0  # j0 j1 j2 j3, the priority order
    assert violations(problem, result) == []


@pytest.mark.parametrize("seed", range(4))
def test_neh_inserts_tasks_on_precedence_problems(seed):
    problem = random_problem(seed, tasks=10, max_modes=2, pools=1)
    result = Neh().solve(problem, StopCriteria(), seed=seed)
    assert violations(problem, result) == []


# ----------------------------------------------------------------- items


def test_items_are_jobs_of_a_grouped_problem():
    items = Items(flow_shop(), jobs=True)
    assert items.jobs
    assert items.count == 4
    assert items.members[2] == (6, 7, 8)
    assert items.tasks([1, 0]) == [3, 4, 5, 0, 1, 2]
    assert items.work == (25.0, 21.0, 20.0, 13.0)
    assert not items.fixed


def test_items_fall_back_to_tasks_when_groups_depend_cyclically():
    problem = Problem(
        "cyclic-groups",
        tasks=(
            Task.on("a1", "m", 1, group="a"),
            Task.on("a2", "m", 1, group="a"),
            Task.on("b1", "m", 1, group="b"),
            Task.on("b2", "m", 1, group="b"),
        ),
        resources=(Resource("m"),),
        dependencies=(Dependency("a1", "b1"), Dependency("b2", "a2")),
    )
    items = Items(problem, jobs=True)
    assert not items.jobs
    assert items.count == 4


def chain_with_side_task() -> Problem:
    # a -> b -> c, and d independent.
    return Problem(
        "chain",
        tasks=tuple(Task.on(name, "m", 1) for name in "abcd"),
        resources=(Resource("m"),),
        dependencies=(Dependency("a", "b"), Dependency("b", "c")),
    )


def test_insertion_range_respects_relations_through_absent_items():
    items = Items(chain_with_side_task(), jobs=False)
    # b is absent: a must still precede c, and b must go between them.
    assert list(items.insertion_range([0, 3, 2], 1)) == [1, 2]
    # a reinserted while b is absent must still come before c.
    assert list(items.insertion_range([3, 2], 0)) == [0, 1]
    # d is unrelated to everything.
    assert list(items.insertion_range([0, 1, 2], 3)) == [0, 1, 2, 3]


def test_closed_tasks_leave_out_items_whose_ancestors_are_absent():
    items = Items(chain_with_side_task(), jobs=False)
    assert items.closed_tasks([2, 3]) == [3]
    assert items.closed_tasks([0, 2, 3]) == [0, 3]
    assert items.closed_tasks([0, 1, 2, 3]) == [0, 1, 2, 3]


def test_feasible_orders_follow_priority_as_far_as_precedence_allows():
    items = Items(chain_with_side_task(), jobs=False)
    assert items.feasible([2, 3, 1, 0]) == [3, 0, 1, 2]
    assert items.feasible([2, 0]) == [0, 2]


def test_a_chain_admits_a_single_sequence():
    problem = Problem(
        "line",
        tasks=tuple(Task.on(name, "m", 1) for name in "abc"),
        resources=(Resource("m"),),
        dependencies=(Dependency("a", "b"), Dependency("b", "c")),
    )
    assert Items(problem, jobs=False).fixed
    assert not Items(chain_with_side_task(), jobs=False).fixed


@pytest.mark.parametrize("seed", range(6))
def test_random_sequences_are_precedence_feasible(seed):
    problem = random_problem(seed, tasks=12)
    items = Items(problem, jobs=False)
    order = items.tasks(items.random_sequence(random.Random(seed)))
    position = {task: rank for rank, task in enumerate(order)}
    assert sorted(order) == list(range(problem.size))
    for task, before in enumerate(problem.predecessors):
        assert all(position[b] < position[task] for b in before)


# -------------------------------------------------------- partial scores


def test_partial_sequences_are_probed_without_becoming_the_incumbent():
    problem = chain_with_side_task()
    items = Items(problem, jobs=False)
    search = Search(problem, StopCriteria(max_evaluations=3), 0, Neh.info)
    search.evaluate([0, 1, 2, 3])
    incumbent = search.best
    # c waits for b, which is absent, so only a and d are decoded.
    score, decoded = score_sequence(search, items, [0, 3, 2], complete=False)
    assert score.value == 2.0
    assert decoded.sequence == (0, 3)
    assert search.best is incumbent
    assert search.evaluations == 2
    score_sequence(search, items, [0], complete=False)
    with pytest.raises(Exhausted):
        score_sequence(search, items, [0], complete=False)


def test_complete_sequences_are_evaluated_as_candidates():
    problem = chain_with_side_task()
    items = Items(problem, jobs=False)
    search = Search(problem, StopCriteria(), 0, Neh.info)
    score, _ = score_sequence(search, items, [3, 0, 1, 2], complete=True)
    assert score.value == 4.0
    assert search.best is not None
    assert search.best.score == score


def test_auto_settings_give_the_builder_choice_around_fixed_modes():
    problem = unrelated_three_by_two()
    items = Items(problem, jobs=False)
    search = Search(problem, StopCriteria(), 0, Neh.info)
    # t1 fixed on m1; t0 and t2 then finish earliest on m0.
    _, decoded = score_sequence(search, items, [0, 1, 2], (AUTO, 1, AUTO), complete=True)
    assert decoded.modes == (0, 1, 0)
    _, decoded = score_sequence(search, items, [0, 1, 2], complete=True)
    assert decoded.modes == (0, 1, 0)


# ---------------------------------------------------------- beam search


def test_a_wide_beam_is_exhaustive():
    problem = random_problem(3, tasks=6, max_modes=1, pools=0)
    result = BeamSearch(width=1000, filter_width=6).solve(problem, StopCriteria())
    assert result.value == pytest.approx(optimum_over_orders(problem))


@pytest.mark.parametrize("seed", range(6))
def test_beam_search_is_never_worse_than_its_look_ahead_rule(seed):
    problem = random_problem(seed, tasks=15, max_modes=2, pools=1)
    beam = BeamSearch().solve(problem, StopCriteria())
    heft = REGISTRY.create("heft").solve(problem, StopCriteria())
    assert value_of(beam) <= value_of(heft) + 1e-9


def test_beam_search_stops_at_the_evaluation_limit():
    problem = random_problem(4, tasks=15)
    result = BeamSearch().solve(problem, StopCriteria(max_evaluations=12))
    assert result.evaluations == 12
    assert result.stop_reason is StopReason.EVALUATION_LIMIT


def test_beam_search_extends_jobs_of_permutation_problems():
    problem = flow_shop()
    result = BeamSearch(width=24, filter_width=4).solve(problem, StopCriteria())
    assert result.value == 37.0
    assert violations(problem, result) == []


def test_beam_parameters_are_validated():
    with pytest.raises(ValueError, match="width"):
        BeamSearch(width=0)
    with pytest.raises(ValueError, match="filter_width"):
        BeamSearch(filter_width=0)


# ------------------------------------------------------- batch mapping


def test_min_min_on_the_hand_example():
    # Step 1: completions t0 2, t1 5, t2 3 (all on m0) -> t0 on m0 [0, 2].
    # Step 2: t1 min(7, 6) = 6 on m1, t2 min(5, 9) = 5 on m0 -> t2 on m0 [2, 5].
    # Step 3: t1 min(10, 6) = 6 on m1 [0, 6].
    result = MinMin().solve(unrelated_three_by_two(), StopCriteria())
    assert placements(result) == {"t0": (0, 0, 2), "t1": (1, 0, 6), "t2": (0, 2, 5)}
    assert result.value == 6.0


def test_max_min_on_the_hand_example():
    # Step 1: completions t0 2, t1 5, t2 3 -> the largest, t1 on m0 [0, 5].
    # Step 2: t0 min(7, 4) = 4 on m1, t2 min(8, 9) = 8 on m0 -> t2 on m0 [5, 8].
    # Step 3: t0 min(10, 4) = 4 on m1 [0, 4].
    result = MaxMin().solve(unrelated_three_by_two(), StopCriteria())
    assert placements(result) == {"t0": (1, 0, 4), "t1": (0, 0, 5), "t2": (0, 5, 8)}
    assert result.value == 8.0


def test_sufferage_on_the_hand_example():
    # Pass 1: t0 claims m0 (sufferage 4 - 2 = 2); t1 wants m0 with sufferage
    # 1 < 2 and waits; t2 wants m0 with sufferage 6 > 2 and takes it. t2 on m0 [0, 3].
    # Pass 2: t0 claims m1 (5 vs 4, sufferage 1); t1 wants m1 (8 vs 6, sufferage
    # 2 > 1) and takes it. t1 on m1 [0, 6].
    # Pass 3: t0 on m0 [3, 5].
    result = Sufferage().solve(unrelated_three_by_two(), StopCriteria())
    assert placements(result) == {"t0": (0, 3, 5), "t1": (1, 0, 6), "t2": (0, 0, 3)}
    assert result.value == 6.0


@pytest.mark.parametrize("solver", [MinMin, MaxMin, Sufferage])
@pytest.mark.parametrize("seed", range(4))
def test_mapping_heuristics_respect_precedence_and_pools(solver, seed):
    problem = random_problem(seed, tasks=12, max_modes=3, pools=2, network=True)
    result = solver().solve(problem, StopCriteria())
    assert violations(problem, result) == []
    assert result.evaluations == 1


@pytest.mark.parametrize("solver_id", ["min-min", "max-min", "sufferage"])
def test_mapping_heuristics_refuse_permutation_problems(solver_id):
    assert not REGISTRY.info(solver_id).accepts(flow_shop())


# ------------------------------------------------ squeaky wheel optimization


def test_makespan_blame_spares_tasks_with_slack():
    # a -> b -> c and d, all of length 1 on one machine. Earliest finishes 1, 2, 3, 1;
    # successor chains 2, 1, 0, 0; the critical-path bound is 3.
    problem = chain_with_side_task()
    ends = [1.0, 3.0, 4.0, 2.0]  # order a, d, b, c
    assert task_blame(problem, "makespan")(ends) == [0.0, 1.0, 1.0, 0.0]
    assert task_blame(problem, "total_completion_time")(ends) == [0.0, 1.0, 1.0, 1.0]


def test_due_date_blame_is_the_avoidable_weighted_tardiness():
    problem = Problem(
        "dues",
        tasks=(
            Task.on("x", "m", 2, due=1, weight=2),
            Task.on("y", "m", 1, due=5),
            Task.on("z", "m", 3),
        ),
        resources=(Resource("m"),),
        objectives=("total_tardiness",),
    )
    # x can never finish before 2, one unit late; in [z, y, x] it ends at 6.
    assert task_blame(problem, "total_tardiness")([6.0, 4.0, 3.0]) == [8.0, 0.0, 0.0]


def test_blamed_items_move_forward_in_proportion_to_their_blame():
    items = Items(chain_with_side_task(), jobs=False)
    swo = SqueakyWheel(step=0.5, jitter=0.0)
    rng = random.Random(0)
    # d (item 3) moves 0.5 * 4 * 8 / 8 = 2 positions: key 1 ties with b, which stays first.
    assert swo.prioritise(items, [0, 1, 2, 3], [0.0, 0.0, 0.0, 8.0], rng) == [0, 1, 3, 2]
    # c moves 2 positions but cannot pass b, its predecessor.
    assert swo.prioritise(items, [0, 1, 3, 2], [0.0, 0.0, 8.0, 0.0], rng) == [0, 1, 2, 3]
    # Without blame the sequence stays.
    assert swo.prioritise(items, [3, 0, 1, 2], [0.0] * 4, rng) == [3, 0, 1, 2]


def test_squeaky_wheel_cycles_once_per_iteration():
    problem = random_problem(3, tasks=10)
    result = SqueakyWheel().solve(problem, StopCriteria(max_iterations=7))
    assert result.iterations == 7
    assert result.evaluations == 7


def test_squeaky_wheel_parameters_are_validated():
    with pytest.raises(ValueError, match="step"):
        SqueakyWheel(step=0.0)
    with pytest.raises(ValueError, match="jitter"):
        SqueakyWheel(jitter=-1.0)
    with pytest.raises(ValueError, match="restart_after"):
        SqueakyWheel(restart_after=0)


# --------------------------------------------------- monte carlo tree search


def test_rank_rewards_count_the_earlier_scores_beaten():
    reward = RankReward()
    assert reward(Score(0.0, 10.0)) == 0.5
    assert reward(Score(0.0, 8.0)) == 1.0
    assert reward(Score(0.0, 12.0)) == 0.0
    assert reward(Score(0.0, 10.0)) == pytest.approx((1 + 0.5) / 3)
    assert reward(Score(1.0, 1.0)) == 0.0  # any overrun loses to every schedule without one


def test_tree_search_enumerates_small_problems_and_stops():
    problem = random_problem(2, tasks=4, max_modes=1, pools=0, density=0.0)
    result = MonteCarloTreeSearch().solve(problem, StopCriteria(max_evaluations=10_000), seed=1)
    # 4 + 12 + 24 + 24 prefixes: one playout creates each.
    assert result.evaluations == 64
    assert result.extras["nodes"] == 65
    assert result.stop_reason is StopReason.COMPLETED
    assert result.value == pytest.approx(optimum_over_orders(problem))


def test_tree_search_respects_precedence_between_items():
    problem = chain_with_side_task()  # orders: dabc, adbc, abdc, abcd
    result = MonteCarloTreeSearch().solve(problem, StopCriteria(max_evaluations=100))
    # Prefixes a, d; ab, ad, da; abc, abd, adb, dab; and the four orders.
    assert result.evaluations == 2 + 3 + 4 + 4
    assert result.stop_reason is StopReason.COMPLETED
    assert result.value == 4.0


def test_tree_search_parameters_are_validated():
    with pytest.raises(ValueError, match="exploration"):
        MonteCarloTreeSearch(exploration=-1.0)
