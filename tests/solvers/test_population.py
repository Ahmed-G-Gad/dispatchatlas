import itertools
import math
import random

import pytest

from dispatchatlas.core import (
    ConfigError,
    Dependency,
    Problem,
    Resource,
    StopCriteria,
    StopReason,
    Task,
)
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.constructive import AUTO, Items
from dispatchatlas.solvers.continuous import Box, minimize
from dispatchatlas.solvers.population import (
    AntColonyOptimization,
    Brkga,
    GeneticAlgorithm,
    MemeticAlgorithm,
    ScatterSearch,
    Trails,
    Umda,
)
from dispatchatlas.solvers.population._construction import construct, roulette
from dispatchatlas.solvers.population.colony import desirability, heuristic_rule
from dispatchatlas.solvers.population.genetic import recombine
from dispatchatlas.solvers.population.scatter import combine, distance
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory import SequenceSpace
from tests.helpers import random_problem
from tests.solvers.test_constructive import flow_shop, value_of, violations
from tests.solvers.test_continuous import extra, sphere, weighted_tardiness
from tests.solvers.test_trajectory import (
    exhaustive_optimum,
    placement_problem,
    random_flow_shop,
    tiny_multi_mode_problem,
)

REGISTRY = default_registry(plugins=False)

SEQUENCE_METHODS = [
    "genetic-algorithm",
    "memetic-algorithm",
    "scatter-search",
    "ant-colony-optimization",
]
POPULATION = [*SEQUENCE_METHODS, "brkga", "umda"]


def space_for(problem: Problem, seed: int = 0) -> SequenceSpace:
    stop = StopCriteria(max_evaluations=1_000_000)
    return SequenceSpace(Search(problem, stop, seed, REGISTRY.info("genetic-algorithm")))


def assert_feasible(items: Items, sequence) -> None:
    assert sorted(sequence) == list(range(items.count))
    position = {item: index for index, item in enumerate(sequence)}
    for item, before in enumerate(items.predecessors):
        assert all(position[other] < position[item] for other in before)


def assert_inherited(child_modes, first_modes, second_modes) -> None:
    assert child_modes is not None
    for mode, a, b in zip(child_modes, first_modes, second_modes, strict=True):
        assert mode in (a, b)


def is_order_crossover(child, first, second) -> bool:
    """Whether ``child`` keeps a slice of ``first`` and fills the rest in ``second``'s order."""
    count = len(first)
    for i, j in itertools.combinations(range(count), 2):
        kept = first[i : j + 1]
        if list(child[i : j + 1]) != list(kept):
            continue
        donors = [second[(j + 1 + k) % count] for k in range(count)]
        fill = [item for item in donors if item not in kept]
        slots = [(j + 1 + k) % count for k in range(len(fill))]
        if [child[slot] for slot in slots] == fill:
            return True
    return False


# -------------------------------------------------------- genetic algorithm


@pytest.mark.parametrize("seed", range(4))
def test_crossover_and_mutation_keep_sequences_feasible(seed):
    problem = random_problem(seed, tasks=12, max_modes=3, density=0.3)
    space = space_for(problem, seed)
    rng = space.search.rng
    solver = GeneticAlgorithm(crossover_rate=1.0, mutation_rate=1.0)
    parents = [space.random_start() for _ in range(6)]
    for _ in range(60):
        first, second = rng.sample(parents, 2)
        sequence, modes = recombine(space.items, first, second, rng)
        assert_feasible(space.items, sequence)
        assert_inherited(modes, first.modes, second.modes)
        child = solver._offspring(space, first, second)
        assert_feasible(space.items, child.sequence)
        assert child.modes is not None
        for task, setting in zip(problem.tasks, child.modes, strict=True):
            assert setting == AUTO or 0 <= setting < len(task.modes)
        parents[rng.randrange(len(parents))] = child


def test_crossover_of_independent_jobs_is_the_order_crossover():
    space = space_for(random_flow_shop(1, jobs=9, machines=3))
    rng = random.Random(5)
    for _ in range(30):
        first, second = space.random_start(), space.random_start()
        sequence, modes = recombine(space.items, first, second, rng)
        assert modes is None
        assert is_order_crossover(sequence, first.sequence, second.sequence)


def test_elites_pass_without_a_new_evaluation():
    problem = random_problem(2, tasks=10, max_modes=2)
    solver = GeneticAlgorithm(population_size=10, elites=3, mutation_rate=1.0)
    result = solver.solve(problem, StopCriteria(max_iterations=5), seed=1)
    # Children identical to a parent are not evaluated either.
    assert 10 + 5 * 5 <= result.evaluations <= 10 + 5 * (10 - 3)


def test_memetic_children_are_insertion_local_optima():
    space = space_for(random_flow_shop(8, jobs=7, machines=4), 8)
    child = MemeticAlgorithm()._improve(space, space.random_start())
    items = space.items
    for item in range(items.count):
        rest = [other for other in child.sequence if other != item]
        for position in items.insertion_range(rest, item):
            neighbour = space.evaluate([*rest[:position], item, *rest[position:]], None)
            assert not neighbour.score < child.score


# -------------------------------------------------------------------- BRKGA


def test_brkga_partitions_the_population():
    assert Brkga().partition(100) == (20, 15)
    assert Brkga().partition(10) == (2, 2)
    assert Brkga().partition(3) == (1, 0)
    assert Brkga(elite_fraction=0.45, mutant_fraction=0.5).partition(4) == (2, 1)


@pytest.mark.parametrize("generations", [1, 4])
def test_brkga_never_evaluates_its_elite_again(generations):
    problem = random_problem(3, tasks=10, max_modes=2)
    solver = Brkga(population_size=12)
    elite, _ = solver.partition(12)
    result = solver.solve(problem, StopCriteria(max_iterations=generations), seed=2)
    assert result.evaluations == 12 + generations * (12 - elite)


def test_brkga_children_take_keys_from_the_elite_parent_at_the_inheritance_rate():
    calls: list[list[float]] = []

    def first_key(x):
        calls.append(list(x))
        return x[0]

    solver = Brkga(population_size=3, inheritance=0.8)
    minimize(solver, first_key, Box.cube(200, 0.0, 1.0), StopCriteria(max_iterations=1))
    initial, children = calls[:3], calls[3:]
    elite = min(initial, key=lambda x: x[0])
    others = [x for x in initial if x is not elite]
    assert len(children) == 2
    for child in children:
        assert any(
            all(key in (a, b) for key, a, b in zip(child, elite, other, strict=True))
            for other in others
        )
        inherited = sum(key == a for key, a in zip(child, elite, strict=True)) / len(child)
        assert inherited == pytest.approx(0.8, abs=0.08)


# --------------------------------------------------------------------- UMDA


def test_umda_evaluates_only_its_new_samples():
    problem = random_problem(4, tasks=8, max_modes=2)
    result = Umda(population_size=20, elites=2).solve(problem, StopCriteria(max_iterations=3))
    assert result.evaluations == 20 + 3 * 18


def test_umda_model_contracts_around_good_regions():
    box = Box.cube(4, -5.0, 5.0)
    spread = [
        extra(minimize(Umda(), sphere, box, stop), "spread")
        for stop in (StopCriteria(max_iterations=2), StopCriteria(max_iterations=20))
    ]
    assert spread[1] < spread[0] / 100


# ---------------------------------------------------------------- ant colony


def test_trails_follow_the_max_min_bounds():
    trails = Trails([10] * 10, evaporation=0.1, best_probability=0.05, decisions=10, choices=5)
    root = 0.05 ** (1 / 10)
    assert trails.upper == pytest.approx(10.0)
    assert trails.lower == pytest.approx(10.0 * (1 - root) / (4 * root))
    assert all(value == trails.upper for row in trails.values for value in row)
    rng = random.Random(1)
    for _ in range(300):
        trails.reinforce((row, rng.randrange(10)) for row in range(10) if rng.random() < 0.5)
        assert all(trails.lower <= v <= trails.upper for row in trails.values for v in row)
    for _ in range(300):
        trails.reinforce((row, 0) for row in range(10))
    assert trails.values[0][0] == pytest.approx(trails.upper)
    assert trails.values[0][1] == pytest.approx(trails.lower)


def test_trails_without_room_for_choice_carry_no_information():
    trails = Trails([2, 2], evaporation=0.2, best_probability=0.05, decisions=1, choices=2)
    assert trails.lower == trails.upper == 5.0
    single = Trails([1], evaporation=0.2, best_probability=0.05, decisions=1, choices=1)
    assert single.lower == single.upper


def test_colony_trails_share_the_upper_bound():
    problem = random_problem(6, tasks=12, max_modes=3)
    solver = AntColonyOptimization(modes="pheromone")
    trails = solver.position_trails(space_for(problem).items)
    modes = solver.mode_trails(problem)
    assert modes is not None
    assert trails.upper == modes.upper == pytest.approx(1 / solver.evaporation)
    assert 0 < trails.lower < trails.upper
    assert AntColonyOptimization().mode_trails(problem) is None


def test_desirability_ranks_items_by_priority():
    problem = Problem(
        "four",
        tasks=tuple(Task.on(name, "m", 1) for name in "abcd"),
        resources=(Resource("m"),),
    )
    items = Items(problem, jobs=False)
    assert desirability(items, [3.0, 1.0, 2.0, 2.0]) == [1.0, 0.25, 0.75, 0.75]
    assert desirability(items, [-math.inf, -1.0, -2.0, -3.0]) == [0.25, 1.0, 0.75, 0.5]


def test_heuristic_rules_follow_the_objective():
    assert heuristic_rule("auto", "total_tardiness") == "earliest-due-date"
    assert heuristic_rule("auto", "total_completion_time") == "shortest-processing-time"
    assert heuristic_rule("auto", "makespan") == "upward-rank"
    assert heuristic_rule("longest-tail", "makespan") == "longest-tail"
    assert heuristic_rule("none", "makespan") is None


def test_construction_draws_eligible_items_by_weight():
    problem = random_problem(5, tasks=10, max_modes=1, density=0.3)
    items = Items(problem, jobs=False)
    rng = random.Random(2)
    for _ in range(50):
        assert_feasible(items, construct(items, lambda item, position: 1.0 + item, rng))
    assert roulette([0.0, 5.0, 0.0], rng) == 1
    assert roulette([0.0, 0.0], rng) in (0, 1)
    draws = [roulette([1.0, 3.0], rng) for _ in range(4000)]
    assert draws.count(1) / len(draws) == pytest.approx(0.75, abs=0.03)


def test_mode_pheromone_finds_the_cheapest_placement():
    problem = placement_problem()
    solver = AntColonyOptimization(modes="pheromone")
    result = solver.solve(problem, StopCriteria(max_evaluations=600), seed=1)
    assert result.schedule is not None
    assert {item.mode for item in result.schedule.assignments} == {1}
    assert result.value == pytest.approx((4 + 6 + 8 + 10) / 4)


# ----------------------------------------------------------- scatter search


def test_distance_sums_position_gaps_and_mode_differences():
    space = space_for(tiny_multi_mode_problem())
    first = space.evaluate([0, 1, 2, 3, 4], (AUTO, AUTO, 0, AUTO, 0))
    second = space.evaluate([1, 0, 4, 3, 2], (AUTO, 1, 0, AUTO, 0))
    # Positions differ by 1, 1, 2, 0, 2; one mode setting differs.
    assert distance(first, second) == 7
    assert distance(second, first) == 7
    assert distance(first, first) == 0


@pytest.mark.parametrize("seed", range(3))
def test_combinations_keep_the_precedence_both_parents_respect(seed):
    problem = random_problem(seed, tasks=14, max_modes=2, density=0.25)
    space = space_for(problem, seed)
    rng = random.Random(seed)
    for _ in range(40):
        first, second = space.random_start(), space.random_start()
        sequence, modes = combine(space.items, first, second, rng)
        assert_feasible(space.items, sequence)
        assert_inherited(modes, first.modes, second.modes)


def test_combination_of_a_solution_with_itself_returns_it():
    space = space_for(random_problem(1, tasks=10, max_modes=2))
    solution = space.random_start()
    sequence, modes = combine(space.items, solution, solution, random.Random(0))
    assert (tuple(sequence), modes) == solution.key


def test_reference_set_takes_the_best_then_the_most_distant():
    space = space_for(flow_shop(((5, 1), (4, 2), (3, 3), (2, 4), (1, 5))))
    candidates = [space.evaluate(list(order), None) for order in itertools.permutations(range(5))]
    solver = ScatterSearch(population_size=10, reference_size=4, quality_size=2)
    reference = solver.reference_set(candidates, [])
    assert len(reference) == 4
    assert [member.score for member in reference] == sorted(member.score for member in reference)
    quality = sorted(candidates, key=lambda member: member.score)[:2]
    keys = {member.key for member in quality}
    assert keys <= {member.key for member in reference}
    spread = {member.key: min(distance(member, best) for best in quality) for member in candidates}
    diverse = [spread[member.key] for member in reference if member.key not in keys]
    rest = [spread[member.key] for member in candidates if member.key not in keys]
    assert max(diverse) == max(rest)


def test_reference_set_keeps_the_given_solutions():
    space = space_for(random_problem(2, tasks=8, max_modes=2))
    candidates = [space.random_start() for _ in range(12)]
    worst = max(candidates, key=lambda member: member.score)
    solver = ScatterSearch(population_size=12, reference_size=5, quality_size=3)
    reference = solver.reference_set(candidates, [worst])
    assert worst in reference
    assert len({member.key for member in reference}) == len(reference) == 5


# ------------------------------------------------------------- all methods


@pytest.mark.parametrize("solver_id", POPULATION)
def test_beats_random_search_at_equal_evaluations(solver_id):
    problem = weighted_tardiness(7, 25)
    stop = StopCriteria(max_evaluations=2000)
    baseline = REGISTRY.create("random-search").solve(problem, stop, seed=3)
    result = REGISTRY.create(solver_id).solve(problem, stop, seed=3)
    assert result.evaluations == 2000
    assert value_of(result) < value_of(baseline)


@pytest.mark.parametrize(
    ("solver_id", "parameters"),
    [
        ("genetic-algorithm", {}),
        ("memetic-algorithm", {}),
        ("scatter-search", {}),
        ("ant-colony-optimization", {"modes": "pheromone"}),
    ],
)
def test_reaches_the_exhaustive_optimum_of_a_tiny_problem(solver_id, parameters):
    problem = tiny_multi_mode_problem()
    best = exhaustive_optimum(problem)
    solver = REGISTRY.create(solver_id, **parameters)
    result = solver.solve(problem, StopCriteria(max_evaluations=2000), seed=4)
    assert result.value == pytest.approx(best.value)
    assert violations(problem, result) == []


@pytest.mark.parametrize(
    "solver_id", ["genetic-algorithm", "memetic-algorithm", "scatter-search", "brkga", "umda"]
)
def test_mode_choices_find_the_cheapest_placement(solver_id):
    problem = placement_problem()
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=600), seed=1)
    assert result.schedule is not None
    assert {item.mode for item in result.schedule.assignments} == {1}
    assert result.value == pytest.approx((4 + 6 + 8 + 10) / 4)


@pytest.mark.parametrize("solver_id", POPULATION)
def test_methods_stop_on_a_single_solution_space(solver_id):
    problem = Problem(
        "line",
        tasks=tuple(Task.on(name, "m", 1) for name in "abc"),
        resources=(Resource("m"),),
        dependencies=(Dependency("a", "b"), Dependency("b", "c")),
    )
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=10_000))
    assert result.stop_reason is StopReason.COMPLETED
    assert result.evaluations == 1
    assert result.value == 3.0


@pytest.mark.parametrize("solver_id", POPULATION)
def test_runs_replay_from_a_seed(solver_id):
    problem = random_problem(9, tasks=12, max_modes=2)
    stop = StopCriteria(max_evaluations=300)
    first = REGISTRY.create(solver_id).solve(problem, stop, seed=11)
    second = REGISTRY.create(solver_id).solve(problem, stop, seed=11)
    assert first.schedule == second.schedule
    assert first.value == second.value
    assert first.iterations == second.iterations


@pytest.mark.parametrize(
    ("solver_id", "parameters"),
    [
        ("genetic-algorithm", {"population_size": 1}),
        ("genetic-algorithm", {"elites": 30}),
        ("genetic-algorithm", {"crossover_rate": -0.1}),
        ("genetic-algorithm", {"tournament_size": 0}),
        ("memetic-algorithm", {"local_search_rate": 2.0}),
        ("brkga", {"elite_fraction": 0.0}),
        ("brkga", {"elite_fraction": 0.6, "mutant_fraction": 0.4}),
        ("brkga", {"population_size": 2}),
        ("brkga", {"inheritance": 0.3}),
        ("brkga", {"modes": "auto"}),
        ("umda", {"selection_fraction": 0.0}),
        ("umda", {"elites": 100}),
        ("umda", {"modes": "auto"}),
        ("scatter-search", {"reference_size": 1}),
        ("scatter-search", {"quality_size": 9}),
        ("scatter-search", {"population_size": 5}),
        ("ant-colony-optimization", {"evaporation": 1.0}),
        ("ant-colony-optimization", {"heuristic": "random"}),
        ("ant-colony-optimization", {"deposit": "everyone"}),
        ("ant-colony-optimization", {"modes": "random"}),
    ],
)
def test_invalid_parameters_are_refused(solver_id, parameters):
    with pytest.raises(ConfigError):
        REGISTRY.create(solver_id, **parameters)
