import itertools
import math
import random
from collections.abc import Sequence

import pytest

from dispatchatlas.core import (
    ConfigError,
    Dependency,
    Mode,
    Problem,
    Resource,
    StopCriteria,
    StopReason,
    Task,
)
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.constructive import AUTO
from dispatchatlas.solvers.evaluation import Evaluator, Score
from dispatchatlas.solvers.search import Candidate, Search
from dispatchatlas.solvers.trajectory import IteratedGreedy, SequenceSpace
from dispatchatlas.solvers.trajectory.acceptance import (
    Improvement,
    LateAcceptance,
    Metropolis,
    RecordDeviation,
    Threshold,
    WaterLevel,
)
from dispatchatlas.solvers.trajectory.alns import Alns, _pick, _roulette
from dispatchatlas.solvers.trajectory.guided import Penalties, fast_descent
from dispatchatlas.solvers.trajectory.hyper import HeuristicScores, low_level_heuristics
from dispatchatlas.solvers.trajectory.mode_tabu import (
    ModeAssignmentTabu,
    Neighbour,
    tabu_length_bounds,
)
from dispatchatlas.solvers.trajectory.space import Move, move_scale, worsening
from dispatchatlas.solvers.trajectory.tabu import TabuSearch, attributes
from tests.helpers import random_problem
from tests.solvers.test_constructive import (
    flow_shop,
    unrelated_three_by_two,
    value_of,
    violations,
)

REGISTRY = default_registry(plugins=False)

STOCHASTIC = [
    "hill-climbing",
    "simulated-annealing",
    "threshold-accepting",
    "great-deluge",
    "record-to-record-travel",
    "late-acceptance-hill-climbing",
    "tabu-search",
    "iterated-local-search",
    "variable-neighbourhood-search",
    "iterated-greedy",
    "grasp",
    "alns",
    "guided-local-search",
    "selection-hyper-heuristic",
]


BEATS_RANDOM_SEARCH = [*STOCHASTIC, "squeaky-wheel-optimization", "monte-carlo-tree-search"]
"""Every stochastic sequence search, including the repeated constructions."""


class FixedRandom(random.Random):
    """A generator whose ``random()`` always returns the same value."""

    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = value

    def random(self) -> float:
        return self.value


def score(value: float, violation: float = 0.0) -> Score:
    return Score(violation, value)


def random_flow_shop(seed: int, jobs: int, machines: int) -> Problem:
    rng = random.Random(seed)
    times = tuple(tuple(rng.randint(1, 99) for _ in range(machines)) for _ in range(jobs))
    return flow_shop(times)


def placement_problem() -> Problem:
    """Four tasks, each on a fast expensive node or a slow cheap one.

    Both nodes hold all four tasks at once, so every task that moves to the
    slow node saves cost on its own, whatever the others do.
    """
    nodes = (
        Resource("fast", capacity=4, cost_rate=5.0),
        Resource("slow", capacity=4, cost_rate=1.0),
    )
    tasks = tuple(
        Task(f"t{i}", (Mode(2 + i, {"fast": 1}), Mode(4 + 2 * i, {"slow": 1}))) for i in range(4)
    )
    return Problem(
        "placement",
        tasks,
        nodes,
        dependencies=(Dependency("t0", "t2"),),
        objectives=("cost", "makespan"),
    )


def tiny_multi_mode_problem() -> Problem:
    return Problem(
        "tiny-modes",
        tasks=(
            Task("a", (Mode(3, {"m1": 1}), Mode(5, {"m2": 1}))),
            Task("b", (Mode(2, {"m1": 1}), Mode(2, {"m2": 1}))),
            Task.on("c", "m2", 4),
            Task("d", (Mode(4, {"m1": 1}), Mode(1, {"m2": 1})), release=2),
            Task.on("e", "m1", 2),
        ),
        resources=(Resource("m1"), Resource("m2")),
        dependencies=(Dependency("a", "c"), Dependency("b", "d", lag=1)),
    )


def exhaustive_optimum(problem: Problem) -> Score:
    evaluator = Evaluator(problem)
    counts = [range(len(task.modes)) for task in problem.tasks]
    return min(
        evaluator.evaluate(order, modes)[0]
        for order in itertools.permutations(range(problem.size))
        for modes in itertools.product(*counts)
    )


# ------------------------------------------------------ acceptance rules


def test_worsening_orders_by_overrun_first():
    assert worsening(score(5.0), score(3.0)) == 2.0
    assert worsening(score(1.0, violation=2.0), score(9.0, violation=1.0)) == math.inf
    assert worsening(score(9.0, violation=0.0), score(1.0, violation=1.0)) == -math.inf


def test_move_scale_is_the_mean_worsening():
    assert move_scale([2.0, 4.0, -1.0, 0.0, math.inf]) == 3.0
    assert move_scale([-2.0, -4.0, 0.0]) == 3.0
    assert move_scale([0.0, 0.0]) == 0.0


def test_improvement_accepts_strictly_better_candidates_only():
    rule = Improvement()
    assert rule.accept(score(4.0), score(5.0), FixedRandom(0.0))
    assert not rule.accept(score(5.0), score(5.0), FixedRandom(0.0))


def test_metropolis_accepts_with_probability_exp_minus_delta_over_t():
    rule = Metropolis(temperature=1.0, cooling=0.5, epoch=2)
    draw = FixedRandom(0.5)
    assert rule.accept(score(4.0), score(5.0), draw)
    assert rule.accept(score(5.0), score(5.0), draw)
    assert rule.accept(score(5.5), score(5.0), draw)  # exp(-0.5) = 0.61 > 0.5
    assert not rule.accept(score(6.0), score(5.0), draw)  # exp(-1) = 0.37 < 0.5
    assert not rule.accept(score(0.0, violation=1.0), score(9.0), FixedRandom(0.0))
    rule.update(score(5.0), accepted=True)
    assert rule.temperature == 1.0
    rule.update(score(5.0), accepted=False)
    assert rule.temperature == 0.5


def test_threshold_accepts_moves_less_than_the_threshold_worse():
    rule = Threshold(threshold=2.0, cooling=0.5, epoch=1)
    assert rule.accept(score(6.9), score(5.0), FixedRandom(0.9))
    assert not rule.accept(score(7.0), score(5.0), FixedRandom(0.0))
    rule.update(score(5.0), accepted=False)
    assert rule.threshold == 1.0


def test_water_level_drops_only_on_acceptance():
    rule = WaterLevel(level=10.0, rain=0.5)
    assert rule.accept(score(9.9), score(5.0), FixedRandom(0.0))
    assert not rule.accept(score(10.0), score(5.0), FixedRandom(0.0))
    rule.update(score(9.9), accepted=True)
    assert rule.level == 9.5
    rule.update(score(9.9), accepted=False)
    assert rule.level == 9.5
    # A smaller overrun is accepted above the level, a larger one never.
    assert rule.accept(score(50.0), score(5.0, violation=1.0), FixedRandom(0.0))
    assert not rule.accept(score(1.0, violation=2.0), score(5.0, violation=1.0), FixedRandom(0.0))


def test_record_to_record_travel_measures_from_the_record():
    rule = RecordDeviation(record=score(10.0), deviation=1.0)
    assert rule.accept(score(10.9), score(10.5), FixedRandom(0.0))
    assert not rule.accept(score(11.0), score(10.5), FixedRandom(0.0))
    rule.update(score(8.0), accepted=True)
    assert rule.record == score(8.0)
    rule.update(score(9.0), accepted=True)
    assert rule.record == score(8.0)
    assert not rule.accept(score(9.5), score(9.0), FixedRandom(0.0))


def test_late_acceptance_compares_with_the_cost_l_steps_back():
    rule = LateAcceptance([score(10.0)] * 2)
    # Iteration 0 compares with slot 0, which holds 10: 12 is worse than both 10
    # and the current 8, while 9 is worse than the current but not than 10.
    assert not rule.accept(score(12.0), score(8.0), FixedRandom(0.0))
    assert rule.accept(score(9.0), score(8.0), FixedRandom(0.0))
    rule.update(score(9.0), accepted=True)
    # Iteration 1 compares with slot 1, still 10.
    assert rule.accept(score(10.0), score(9.0), FixedRandom(0.0))
    rule.update(score(10.0), accepted=True)
    # Iteration 2 compares with slot 0, which now holds 9: 9.5 improves on the
    # current 10 and is accepted, 11 improves on neither.
    assert rule.accept(score(9.5), score(10.0), FixedRandom(0.0))
    assert not rule.accept(score(11.0), score(10.0), FixedRandom(0.0))


# ------------------------------------------------------------- the space


def space_for(problem: Problem, seed: int = 0) -> SequenceSpace:
    stop = StopCriteria(max_evaluations=1_000_000)
    return SequenceSpace(Search(problem, stop, seed, REGISTRY.info("hill-climbing")))


def assert_feasible_sequence(space: SequenceSpace, sequence: Sequence[int]) -> None:
    items = space.items
    position = {item: index for index, item in enumerate(sequence)}
    assert sorted(sequence) == list(range(items.count))
    for item, before in enumerate(items.predecessors):
        assert all(position[b] < position[item] for b in before)


@pytest.mark.parametrize("seed", range(5))
def test_moves_keep_sequences_precedence_feasible(seed):
    problem = random_problem(seed, tasks=12, max_modes=3, density=0.3)
    space = space_for(problem, seed)
    solution = space.random_start()
    assert solution.modes == tuple(AUTO if len(task.modes) > 1 else 0 for task in problem.tasks)
    for _ in range(200):
        step = space.neighbour(solution)
        assert step is not None
        candidate, move = step
        assert_feasible_sequence(space, candidate.sequence)
        if move.kind == "mode":
            assert candidate.modes is not None
            assert candidate.modes[move.first] == move.second
            assert candidate.sequence == solution.sequence
        else:
            assert candidate.sequence != solution.sequence
            assert candidate.modes == solution.modes
        solution = candidate


def test_mode_moves_pin_or_release_tasks():
    space = space_for(unrelated_three_by_two())
    # An AUTO task running in mode 0 may be pinned to mode 1 only.
    assert space.settings(0, AUTO, 0) == [1]
    # A pinned task may be released or pinned elsewhere.
    assert space.settings(0, 1, 1) == [AUTO, 0]


def test_permutation_problems_move_whole_jobs():
    space = space_for(flow_shop())
    assert space.items.jobs
    assert space.items.count == 4


def test_a_single_solution_space_has_no_move():
    problem = Problem(
        "line",
        tasks=tuple(Task.on(name, "m", 1) for name in "abc"),
        resources=(Resource("m"),),
        dependencies=(Dependency("a", "b"), Dependency("b", "c")),
    )
    space = space_for(problem)
    assert not space.movable
    assert space.neighbour(space.random_start()) is None


@pytest.mark.parametrize("solver_id", STOCHASTIC)
def test_methods_stop_on_a_single_solution_space(solver_id):
    problem = Problem(
        "line",
        tasks=tuple(Task.on(name, "m", 1) for name in "abc"),
        resources=(Resource("m"),),
        dependencies=(Dependency("a", "b"), Dependency("b", "c")),
    )
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=10_000))
    assert result.stop_reason is StopReason.COMPLETED
    assert result.evaluations < 10
    assert result.value == 3.0


def test_descent_ends_in_an_insertion_local_optimum():
    problem = random_flow_shop(2, jobs=7, machines=4)
    space = space_for(problem, 2)
    optimum = space.descend(space.random_start())
    for item in range(space.items.count):
        rest = [other for other in optimum.sequence if other != item]
        for position in space.items.insertion_range(rest, item):
            neighbour = space.evaluate([*rest[:position], item, *rest[position:]], None)
            assert not neighbour.score < optimum.score


def test_climb_never_worsens_its_start():
    problem = random_problem(3, tasks=15, max_modes=2)
    space = space_for(problem, 3)
    start = space.random_start()
    assert not start.score < space.climb(start, 20).score


def test_tabu_attributes_record_positions_and_modes():
    space = space_for(unrelated_three_by_two())
    solution = space.evaluate((2, 0, 1), (AUTO, 1, AUTO))
    assert attributes(solution, Move("insert", 0, 2)) == ([(0, 2, 2)], [(0, 2, 0)])
    assert attributes(solution, Move("swap", 0, 1)) == (
        [(0, 2, 1), (0, 0, 0)],
        [(0, 2, 0), (0, 0, 1)],
    )
    assert attributes(solution, Move("mode", 1, AUTO)) == ([(1, 1, AUTO)], [(1, 1, 1)])


# ------------------------------------------------------ method behaviour


def test_iterated_greedy_temperature_follows_the_published_constant():
    # The twelve operations take 79 in total: T * (79 / 12) / 10 with T = 0.4.
    assert IteratedGreedy().acceptance_temperature(flow_shop()) == pytest.approx(0.4 * 79 / 120)


def test_iterated_greedy_never_ends_worse_than_neh():
    problem = random_flow_shop(5, jobs=8, machines=4)
    neh = REGISTRY.create("neh").solve(problem, StopCriteria())
    greedy = REGISTRY.create("iterated-greedy").solve(problem, StopCriteria(max_evaluations=500))
    assert value_of(greedy) <= value_of(neh)


def test_grasp_with_zero_alpha_builds_the_min_min_schedule_first():
    problem = unrelated_three_by_two()
    grasp = REGISTRY.create("grasp", alpha=0.0).solve(problem, StopCriteria(max_evaluations=1))
    min_min = REGISTRY.create("min-min").solve(problem, StopCriteria())
    assert grasp.schedule == min_min.schedule


@pytest.mark.parametrize("solver_id", STOCHASTIC)
def test_mode_moves_find_the_cheapest_placement(solver_id):
    problem = placement_problem()
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=600), seed=1)
    assert result.schedule is not None
    assert {item.mode for item in result.schedule.assignments} == {1}
    assert result.value == pytest.approx((4 + 6 + 8 + 10) / 4)


@pytest.mark.parametrize("solver_id", STOCHASTIC)
def test_reaches_the_exhaustive_optimum_of_a_tiny_problem(solver_id):
    problem = tiny_multi_mode_problem()
    best = exhaustive_optimum(problem)
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=2000), seed=4)
    assert result.value == pytest.approx(best.value)
    assert violations(problem, result) == []


@pytest.mark.parametrize("solver_id", BEATS_RANDOM_SEARCH)
def test_beats_random_search_at_equal_evaluations(solver_id):
    problem = random_flow_shop(3, jobs=12, machines=8)
    stop = StopCriteria(max_evaluations=800)
    baseline = REGISTRY.create("random-search").solve(problem, stop, seed=3)
    result = REGISTRY.create(solver_id).solve(problem, stop, seed=3)
    assert result.evaluations <= 800
    assert value_of(result) < value_of(baseline)


# ------------------------------------------------------------------ ALNS


def test_roulette_follows_the_weights():
    assert _roulette([1.0, 3.0], FixedRandom(0.2)) == 0
    assert _roulette([1.0, 3.0], FixedRandom(0.3)) == 1


def test_removal_picks_favour_the_head_of_the_ranking():
    assert _pick([7, 8, 9], FixedRandom(0.5), 6.0) == 7  # 0.5 ** 6 * 3 = 0.05
    assert _pick([7, 8, 9], FixedRandom(0.9), 6.0) == 8  # 0.9 ** 6 * 3 = 1.59
    assert _pick([7, 8, 9], FixedRandom(0.9), 1.0) == 9  # 0.9 * 3 = 2.7


def test_operator_weights_blend_the_segment_rewards():
    weights = [[1.0, 1.0, 1.0], [1.0, 1.0]]
    rewards = [[33.0, 0.0, 0.0], [9.0, 0.0]]
    uses = [[2, 1, 0], [3, 0]]
    Alns(reaction=0.1)._adapt(weights, rewards, uses)
    assert weights[0] == pytest.approx([0.9 + 0.1 * 16.5, 0.9, 1.0])
    assert weights[1] == pytest.approx([0.9 + 0.1 * 3.0, 1.0])
    assert rewards == [[0.0] * 3, [0.0] * 2]
    assert uses == [[0] * 3, [0] * 2]


@pytest.mark.parametrize(
    ("solver_id", "parameters"),
    [
        ("simulated-annealing", {"initial_acceptance": 1.0}),
        ("simulated-annealing", {"cooling": 0.0}),
        ("threshold-accepting", {"epoch": 0}),
        ("great-deluge", {"rain_speed": 0.0}),
        ("record-to-record-travel", {"deviation": -1.0}),
        ("late-acceptance-hill-climbing", {"history": 0}),
        ("hill-climbing", {"restart_after": 0}),
        ("tabu-search", {"tenure": 0}),
        ("iterated-local-search", {"strength": 0}),
        ("variable-neighbourhood-search", {"max_neighbourhood": 0}),
        ("iterated-greedy", {"destruction": 0}),
        ("grasp", {"alpha": 1.5}),
        ("alns", {"min_removed": 3, "max_removed": 2}),
        ("guided-local-search", {"alpha": 0.0}),
        ("selection-hyper-heuristic", {"history": 0}),
        ("selection-hyper-heuristic", {"max_score": 0}),
        ("selection-hyper-heuristic", {"removed": 0}),
        ("mode-assignment-tabu", {"adjust_after": 0}),
        ("mode-assignment-tabu", {"diversification": -1.0}),
    ],
)
def test_invalid_parameters_are_refused(solver_id, parameters):
    with pytest.raises(ConfigError):
        REGISTRY.create(solver_id, **parameters)


def test_tabu_search_counts_one_iteration_per_move():
    problem = random_problem(1, tasks=10, max_modes=1)
    result = TabuSearch(candidates=5).solve(problem, StopCriteria(max_iterations=4))
    assert result.iterations == 4
    assert result.evaluations == 1 + 4 * 5


# --------------------------------------------------- guided local search


def test_penalties_go_to_the_features_of_highest_utility():
    space = space_for(flow_shop())  # jobs of work 25, 21, 20, 13
    penalties = Penalties(space)
    solution = space.start([0, 1, 2, 3])
    assert penalties.size == 4
    assert penalties.penalise(solution) == [0]  # utility 25
    assert penalties.penalise(solution) == [1]  # 21 beats 25 / 2
    assert penalties.penalise(solution) == [2]  # 20
    assert penalties.penalise(solution) == [3]  # 13 beats 12.5, 10.5, 10
    assert penalties.penalise(solution) == [0]  # 12.5 beats 10.5, 10, 6.5
    assert penalties.counts == {(0, 0, 0): 2, (0, 1, 1): 1, (0, 2, 2): 1, (0, 3, 3): 1}


def test_equal_utilities_are_penalised_together():
    problem = Problem(
        "twins",
        tasks=tuple(Task.on(name, "m", 2) for name in "ab"),
        resources=(Resource("m"),),
    )
    space = space_for(problem)
    penalties = Penalties(space)
    assert penalties.penalise(space.start([1, 0])) == [0, 1]
    assert penalties.counts == {(0, 1, 0): 1, (0, 0, 1): 1}


def test_mode_features_follow_the_modes_tasks_run_in():
    space = space_for(unrelated_three_by_two())
    penalties = Penalties(space)
    solution = space.start([0, 1, 2])
    modes = [feature for feature, _ in penalties.features(solution) if feature[0] == 1]
    assert modes == [(1, 0, 0), (1, 1, 1), (1, 2, 0)]
    assert penalties.size == 6


def test_the_augmented_objective_adds_the_weighted_penalties_present():
    space = space_for(flow_shop())
    penalties = Penalties(space)
    solution = space.start([0, 1, 2, 3])
    penalties.penalise(solution)
    penalties.penalise(solution)
    penalties.weight = 2.5
    assert penalties.total(solution) == 2
    assert penalties.augmented(solution) == score(solution.score.value + 5.0)
    assert penalties.total(space.start([1, 0, 2, 3])) == 0


def test_fast_descent_only_makes_improving_moves():
    problem = random_problem(4, tasks=10, max_modes=2)
    space = space_for(problem, 4)
    start = space.random_start()
    result = fast_descent(space, Penalties(space), start, range(space.items.count))
    assert result.score < start.score
    assert fast_descent(space, Penalties(space), result, []) is result


def test_penalties_lead_the_search_past_its_first_local_optimum():
    problem = random_flow_shop(3, jobs=12, machines=8)
    space = space_for(problem, 3)
    first = fast_descent(space, Penalties(space), space.random_start(), range(12))
    result = REGISTRY.create("guided-local-search").solve(
        problem, StopCriteria(max_evaluations=800), seed=3
    )
    assert value_of(result) < first.score.value


def test_guided_local_search_checks_the_budget_inside_the_descent():
    problem = random_problem(2, tasks=12, max_modes=2)
    result = REGISTRY.create("guided-local-search").solve(problem, StopCriteria(max_evaluations=7))
    assert result.evaluations == 7
    assert result.iterations == 0
    assert result.stop_reason is StopReason.EVALUATION_LIMIT


# --------------------------------------------- selection hyper-heuristic


def test_reinforcement_scores_stay_within_their_bounds():
    scores = HeuristicScores(3, maximum=2)
    scores.update(0, improved=False)
    assert scores.values == [0, 0, 0]
    for _ in range(3):
        scores.update(1, improved=True)
    assert scores.values == [0, 2, 0]
    assert scores.choose(random.Random(0)) == 1
    scores.update(1, improved=False)
    scores.update(1, improved=False)
    assert scores.values == [0, 0, 0]
    assert {scores.choose(random.Random(seed)) for seed in range(20)} == {0, 1, 2}


def test_low_level_heuristics_follow_the_problem():
    assert list(low_level_heuristics(space_for(flow_shop()), 2)) == [
        "insert",
        "swap",
        "ruin_recreate",
        "reinsert",
    ]
    assert list(low_level_heuristics(space_for(unrelated_three_by_two()), 2)) == [
        "insert",
        "swap",
        "mode",
        "ruin_recreate",
        "reinsert",
    ]
    line = Problem(
        "line",
        tasks=(Task("a", (Mode(1, {"m": 1}), Mode(2, {"n": 1}))), Task.on("b", "m", 1)),
        resources=(Resource("m"), Resource("n")),
        dependencies=(Dependency("a", "b"),),
    )
    assert list(low_level_heuristics(space_for(line), 2)) == ["mode"]


def test_the_hyper_heuristic_applies_one_heuristic_per_iteration_and_uses_all():
    problem = random_problem(5, tasks=12, max_modes=2)
    result = REGISTRY.create("selection-hyper-heuristic").solve(
        problem, StopCriteria(max_evaluations=400), seed=5
    )
    calls = {key: value for key, value in result.extras.items() if key.startswith("calls_")}
    assert set(calls) == {
        "calls_insert",
        "calls_swap",
        "calls_mode",
        "calls_ruin_recreate",
        "calls_reinsert",
    }
    assert all(isinstance(count, int) and count > 0 for count in calls.values())
    assert sum(int(count) for count in calls.values()) == result.iterations


# -------------------------------------------------- mode-assignment tabu


def test_tabu_list_length_bounds_follow_the_square_root_rule():
    assert tabu_length_bounds(9) == (3, 6, 9)
    assert tabu_length_bounds(2) == (1, 3, 4)
    assert tabu_length_bounds(1) == (1, 2, 3)


def neighbours(*entries: tuple[int, int, float]) -> list[Neighbour]:
    decoded = Evaluator(unrelated_three_by_two()).evaluate([0, 1, 2])[1]
    return [
        Neighbour(task, mode, Candidate([0, 1, 2], None, score(value), decoded))
        for task, mode, value in entries
    ]


def test_non_tabu_improving_moves_come_first():
    moves = neighbours((0, 1, 5.0), (1, 1, 7.0))
    chosen = ModeAssignmentTabu()._choose(moves, score(8.0), score(6.0), {(0, 1)}, {}, [0, 0, 0])
    assert (chosen.task, chosen.mode) == (1, 1)


def test_aspiring_tabu_moves_are_admissible_without_improving_alternatives():
    tabu = ModeAssignmentTabu()
    # (0, 1) is tabu but beats the best solution found: it aspires.
    moves = neighbours((0, 1, 5.0), (1, 1, 9.0))
    chosen = tabu._choose(moves, score(8.0), score(6.0), {(0, 1), (1, 1)}, {}, [0, 0, 0])
    assert (chosen.task, chosen.mode) == (0, 1)
    # Regional aspiration: it beats the record of task 0 in mode 1.
    moves = neighbours((0, 1, 7.0), (1, 1, 9.0))
    record = {(0, 1): score(7.5)}
    chosen = tabu._choose(moves, score(6.0), score(5.0), {(0, 1)}, record, [0, 0, 0])
    assert (chosen.task, chosen.mode) == (0, 1)


def test_fully_tabu_neighbourhoods_take_the_best_move():
    moves = neighbours((0, 1, 9.0), (1, 1, 8.5))
    chosen = ModeAssignmentTabu()._choose(
        moves, score(8.0), score(6.0), {(0, 1), (1, 1)}, {}, [0, 0, 0]
    )
    assert (chosen.task, chosen.mode) == (1, 1)


def test_frequent_movers_are_penalised_in_mildest_ascent():
    moves = neighbours((0, 1, 9.0), (1, 1, 9.5), (2, 1, 11.0))
    tabu = ModeAssignmentTabu(diversification=1.0)
    # The mean worsening is 11/6. Task 0 moved most often and pays all of it, task 1 half.
    chosen = tabu._choose(moves, score(8.0), score(6.0), set(), {}, [4, 2, 0])
    assert (chosen.task, chosen.mode) == (1, 1)
    chosen = ModeAssignmentTabu(diversification=0.0)._choose(
        moves, score(8.0), score(6.0), set(), {}, [4, 2, 0]
    )
    assert (chosen.task, chosen.mode) == (0, 1)


def test_mode_assignment_tabu_starts_from_the_shortest_modes():
    problem = tiny_multi_mode_problem()
    result = REGISTRY.create("mode-assignment-tabu").solve(problem, StopCriteria(max_evaluations=1))
    assert result.schedule is not None
    assert {item.task: item.mode for item in result.schedule.assignments} == {
        "a": 0,
        "b": 0,
        "c": 0,
        "d": 1,
        "e": 0,
    }


def test_each_iteration_evaluates_the_whole_mode_neighbourhood():
    problem = tiny_multi_mode_problem()  # three tasks with two modes
    result = REGISTRY.create("mode-assignment-tabu").solve(problem, StopCriteria(max_iterations=2))
    assert result.iterations == 2
    assert result.evaluations == 1 + 2 * 3


def test_single_mode_problems_have_one_mode_vector():
    problem = random_flow_shop(1, jobs=4, machines=3)
    result = REGISTRY.create("mode-assignment-tabu").solve(
        problem, StopCriteria(max_evaluations=100)
    )
    assert result.evaluations == 1
    assert result.stop_reason is StopReason.COMPLETED


def test_mode_assignment_tabu_finds_the_cheapest_placement():
    problem = placement_problem()
    stop = StopCriteria(max_evaluations=40)
    result = REGISTRY.create("mode-assignment-tabu").solve(problem, stop, seed=1)
    baseline = REGISTRY.create("random-search").solve(problem, stop, seed=1)
    assert result.value == pytest.approx((4 + 6 + 8 + 10) / 4)
    assert value_of(result) < value_of(baseline)
