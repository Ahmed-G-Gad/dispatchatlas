"""Tests of the multi-objective evolutionary algorithms and their selection rules."""

from __future__ import annotations

import math
import random
import statistics
from functools import cache

import pytest

from dispatchatlas.benchmarks.continuum.families import ContinuumOffloading
from dispatchatlas.benchmarks.shop import FlowShop
from dispatchatlas.core import (
    ConfigError,
    Dependency,
    FrontPoint,
    Problem,
    Resource,
    StopCriteria,
    StopReason,
    Task,
    check_schedule,
    evaluate,
)
from dispatchatlas.core.pareto import dominates, fast_non_dominated_sort, hypervolume
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.constructive import Items
from dispatchatlas.solvers.multiobjective import (
    AgeMoea2,
    Ibea,
    MoeaD,
    Nsga2,
    Nsga3,
    Rvea,
    SmsEmoa,
    Spea2,
)
from dispatchatlas.solvers.multiobjective._loop import (
    Evolution,
    Member,
    feasible_first,
    keep,
    tournament,
)
from dispatchatlas.solvers.multiobjective.selection import (
    adapt_vectors,
    additive_epsilon,
    age_scores,
    age_survival,
    crowding_survival,
    das_dennis,
    front_curvature,
    geodesic_distance,
    hyperplane_normalize,
    hypervolume_contributions,
    hypervolume_discard,
    ibea_fitness,
    ibea_survival,
    last_front,
    minkowski_norm,
    newton_curvature,
    niching_survival,
    normalize,
    project_to_front,
    rvea_survival,
    simplex_divisions,
    smallest_angles,
    spea2_fitness,
    spea2_survival,
    spea2_truncation,
    tchebycheff,
    unit_vectors,
    weight_neighbourhoods,
)
from dispatchatlas.solvers.operators import random_modes
from dispatchatlas.solvers.search import Search
from tests.helpers import random_problem

REGISTRY = default_registry(plugins=False)

IDS = ("nsga-ii", "nsga-iii", "moea-d", "spea2", "ibea", "sms-emoa", "rvea", "age-moea-ii")
SEEDS = (1, 2, 3)


@cache
def offloading(deadline_factor: float | None = None) -> Problem:
    """Eight requests offloaded across a continuum, trading response time against energy."""
    family = ContinuumOffloading(
        applications=8,
        deadline_factor=deadline_factor,
        objectives=("mean_response_time", "energy"),
    )
    return family.generate(3)


@cache
def tardy_flow_shop(objectives: tuple[str, ...] = ("makespan", "total_tardiness")) -> Problem:
    """A 10-job, 4-machine permutation flow shop with due dates."""
    return FlowShop(jobs=10, machines=4, due_date_tightness=1.2, objectives=objectives).generate(3)


@cache
def random_archive(problem: Problem, seed: int, budget: int) -> tuple[tuple[float, ...], ...]:
    """The archive of random sampling over the same genomes at a given budget."""
    search = Search(
        problem,
        StopCriteria(max_evaluations=budget),
        seed,
        REGISTRY.info("random-search"),
        archive=True,
    )
    items = Items(problem, jobs=problem.permutation)
    with search:
        while True:
            search.next_iteration()
            order = items.tasks(items.random_sequence(search.rng))
            search.evaluate(order, random_modes(problem, search.rng))
    return tuple(search.vector(candidate) for candidate in search.front())


def member(vector: tuple[float, ...], violation: float = 0.0) -> Member:
    return Member((0,), (0,), vector, violation)


# --------------------------------------------------------- shared helpers


def test_das_dennis_lists_the_simplex_lattice_in_lexicographic_order() -> None:
    assert das_dennis(2, 2) == [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    points = das_dennis(4, 3)
    assert len(points) == 15
    assert points[:6] == [
        (0.0, 0.0, 1.0),
        (0.0, 0.25, 0.75),
        (0.0, 0.5, 0.5),
        (0.0, 0.75, 0.25),
        (0.0, 1.0, 0.0),
        (0.25, 0.0, 0.75),
    ]
    assert points[-1] == (1.0, 0.0, 0.0)
    assert points == sorted(points)
    for divisions, objectives in ((1, 1), (3, 1), (12, 2), (12, 3), (6, 5), (3, 4)):
        lattice = das_dennis(divisions, objectives)
        assert len(lattice) == math.comb(divisions + objectives - 1, objectives - 1)
        assert all(sum(point) == pytest.approx(1.0) for point in lattice)
    assert das_dennis(5, 1) == [(1.0,)]
    with pytest.raises(ValueError, match="at least 1"):
        das_dennis(0, 2)


def test_simplex_divisions_fit_the_population() -> None:
    assert simplex_divisions(2, 40) == 39
    assert simplex_divisions(3, 40) == 7
    assert simplex_divisions(3, 36) == 7
    assert simplex_divisions(3, 35) == 6
    assert simplex_divisions(1, 10) == 1
    assert simplex_divisions(5, 3) == 1


def test_normalize_maps_the_range_to_the_unit_interval() -> None:
    assert normalize([(1.0, 10.0), (3.0, 30.0), (2.0, 20.0)]) == [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5),
    ]
    assert normalize([(1.0, 5.0), (2.0, 5.0)]) == [(0.0, 0.0), (1.0, 0.0)]
    assert normalize([(2.0,)], lower=(1.0,), upper=(5.0,)) == [(0.25,)]


# ---------------------------------------------------------------- NSGA-II


CROWDING_POOL = [(0.0, 4.0), (1.0, 2.0), (2.0, 1.0), (4.0, 0.0), (3.0, 3.0), (4.0, 4.0)]


def test_crowding_survival_cuts_the_last_front_by_crowding_distance() -> None:
    # Front 1 is the first four points; B and C both have crowding 0.5 + 0.75.
    survivors = crowding_survival(CROWDING_POOL, 3)
    assert survivors.indices == (0, 3, 1)
    assert survivors.keys == ((0.0, -math.inf), (0.0, -math.inf), (0.0, -1.25))


def test_crowding_survival_keeps_whole_fronts_in_rank_order() -> None:
    survivors = crowding_survival(CROWDING_POOL, 5)
    assert survivors.indices == (0, 1, 2, 3, 4)
    assert [key[0] for key in survivors.keys] == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert survivors.keys[1] == (0.0, -1.25)
    assert crowding_survival(CROWDING_POOL, 10).indices == (0, 1, 2, 3, 4, 5)


# ---------------------------------------------------------------- NSGA-III


def test_hyperplane_normalization_uses_the_intercepts_of_the_extreme_points() -> None:
    points = [(0.0, 4.0), (1.0, 3.0), (3.0, 1.0), (4.0, 0.0), (2.0, 2.0)]
    normalized, extremes = hyperplane_normalize([(x + 10.0, 2.0 * y) for x, y in points])
    assert extremes == [3, 0]
    expected = [(0.0, 1.0), (0.25, 0.75), (0.75, 0.25), (1.0, 0.0), (0.5, 0.5)]
    for got, want in zip(normalized, expected, strict=True):
        assert got == pytest.approx(want)


def test_hyperplane_normalization_falls_back_to_the_nadir_when_degenerate() -> None:
    # Both extreme points are (0, 0) after translation: no hyperplane.
    normalized, _ = hyperplane_normalize([(1.0, 1.0), (2.0, 5.0), (3.0, 3.0)])
    assert normalized == [(0.0, 0.0), (0.5, 1.0), (1.0, 0.5)]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_niching_gives_every_reference_line_its_nearest_point(seed: int) -> None:
    points = [(0.0, 4.0), (1.0, 3.0), (3.0, 1.0), (4.0, 0.0), (2.0, 2.0)]
    chosen = niching_survival(points, 3, das_dennis(2, 2), random.Random(seed))
    assert sorted(chosen) == [0, 3, 4]


def test_niching_counts_the_points_of_earlier_fronts() -> None:
    # Front 1 fills the two outer niches, so the middle one gets the next point.
    points = [(0.0, 2.0), (2.0, 0.0), (1.0, 3.0), (3.0, 1.0), (2.0, 2.0)]
    chosen = niching_survival(points, 3, das_dennis(2, 2), random.Random(0))
    assert chosen == [0, 1, 4]


# ------------------------------------------------------------------- SPEA2


def test_spea2_fitness_adds_raw_fitness_and_density() -> None:
    fitness = spea2_fitness([(0.0, 2.0), (1.0, 1.0), (2.0, 0.0), (2.0, 2.0)])
    assert fitness == pytest.approx([0.25, 1.0 / (2.0 + math.sqrt(2.0)), 0.25, 3.25])


def test_spea2_truncation_removes_the_most_crowded_point_by_its_nearest_distances() -> None:
    # Points 1 and 2 are nearest each other; point 1's second neighbour is nearer.
    points = [(0.0, 4.0), (1.0, 3.0), (1.5, 2.5), (3.0, 1.0), (4.0, 0.0)]
    assert spea2_truncation(points, 4) == [0, 2, 3, 4]
    assert spea2_truncation(points, 2) == [0, 4]


def test_spea2_survival_fills_with_dominated_points_or_truncates() -> None:
    points = [(0.0, 2.0), (1.0, 1.0), (2.0, 0.0), (2.0, 2.0)]
    assert spea2_survival(points, 4).indices == (0, 1, 2, 3)
    survivors = spea2_survival(points, 2)
    assert survivors.indices == (0, 2)
    assert [key[0] for key in survivors.keys] == pytest.approx([1.0 / 3.0, 1.0 / 3.0])
    assert spea2_survival(points, 4).keys[3][0] == pytest.approx(3.0 + 1.0 / 3.0)


# -------------------------------------------------------------------- IBEA


def test_additive_epsilon_is_the_largest_difference() -> None:
    assert additive_epsilon((1.0, 2.0), (2.0, 1.0)) == 1.0
    assert additive_epsilon((0.0, 0.0), (1.0, 1.0)) == -1.0


def test_ibea_fitness_and_its_update_match_hand_values() -> None:
    # Scaled to their range the points are a = (0, 1), b = (0.5, 0), d = (1, 1).
    points = [(0.0, 1.0), (0.5, 0.5), (1.0, 1.0)]
    fitness = ibea_fitness(points, kappa=0.05)
    assert fitness == pytest.approx(
        [-math.exp(-10.0) - math.exp(-20.0), -2.0 * math.exp(-20.0), -1.0 - math.exp(10.0)]
    )
    survivors = ibea_survival(points, 2, kappa=0.05)
    assert survivors.indices == (0, 1)
    assert [key[0] for key in survivors.keys] == pytest.approx([math.exp(-10.0), math.exp(-20.0)])


# ---------------------------------------------------------------- SMS-EMOA


def test_hypervolume_contributions_match_hand_values() -> None:
    assert hypervolume_contributions([(1, 3), (2, 2), (3, 1)], (4, 4)) == [1.0, 1.0, 1.0]
    assert hypervolume_contributions([(1, 2), (2, 1)], (3, 3)) == [1.0, 1.0]
    cube = [(1.0, 2.0, 3.0), (2.0, 3.0, 1.0), (3.0, 1.0, 2.0)]
    assert hypervolume(cube, (4, 4, 4)) == pytest.approx(13.0)
    assert hypervolume_contributions(cube, (4, 4, 4)) == pytest.approx([3.0, 3.0, 3.0])
    assert hypervolume_contributions([(1.0,), (3.0,), (2.0,)], (5.0,)) == [1.0, 0.0, 0.0]


def test_hypervolume_contributions_of_repeated_dominated_and_outside_points() -> None:
    assert hypervolume_contributions([(1, 1), (1, 1)], (2, 2)) == [0.0, 0.0]
    assert hypervolume_contributions([(0, 0), (1, 1)], (2, 2)) == [3.0, 0.0]
    assert hypervolume_contributions([(1, 1), (3, 0)], (2, 2)) == [1.0, 0.0]
    assert hypervolume_contributions([(3, 3)], (2, 2)) == [0.0]


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_hypervolume_contributions_equal_differences_of_hypervolumes(dimension: int) -> None:
    rng = random.Random(dimension)
    for _ in range(150):
        points: list[tuple[float, ...]] = []
        for _ in range(rng.randint(1, 10)):
            if points and rng.random() < 0.2:
                points.append(rng.choice(points))
            else:
                points.append(tuple(float(rng.randint(0, 5)) for _ in range(dimension)))
        reference = tuple(rng.choice((5.0, 6.0)) for _ in range(dimension))
        total = hypervolume(points, reference)
        expected = [
            total - hypervolume(points[:i] + points[i + 1 :], reference) for i in range(len(points))
        ]
        assert hypervolume_contributions(points, reference) == pytest.approx(expected)


def test_last_front_matches_the_last_front_of_the_full_sort() -> None:
    rng = random.Random(4)
    for _ in range(200):
        dimension = rng.randint(1, 3)
        points = [
            tuple(float(rng.randint(0, 4)) for _ in range(dimension))
            for _ in range(rng.randint(1, 12))
        ]
        assert last_front(points) == sorted(fast_non_dominated_sort(points)[-1])


def test_hypervolume_discard_removes_the_smallest_contributor() -> None:
    # Scaled to the unit box, point 1 adds 0.0125, point 2 0.03375, point 3 0.10625.
    front = [(0.0, 4.0), (1.0, 3.0), (1.2, 2.7), (3.0, 1.0), (4.0, 0.0)]
    assert hypervolume_discard(front, random.Random(0)) == 1
    assert hypervolume_discard([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], random.Random(0)) == 2


def test_hypervolume_discard_keeps_the_best_point_in_each_objective() -> None:
    # Point 2 contributes least against the reference point, but it is best in objective 2.
    front = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.45)]
    contributions = hypervolume_contributions(normalize(front), (1.1, 1.1))
    assert min(contributions) == contributions[2]
    assert hypervolume_discard(front, random.Random(0)) == 1


# ------------------------------------------------------------------ MOEA/D


def test_tchebycheff_matches_hand_values() -> None:
    assert tchebycheff((2.0, 3.0), (0.5, 0.5), (1.0, 1.0)) == 1.0
    assert tchebycheff((2.0, 3.0), (1.0, 0.0), (1.0, 1.0)) == 1.0
    assert tchebycheff((2.0, 3.0), (0.0, 1.0), (1.0, 1.0)) == 2.0
    assert tchebycheff((2.0, 3.0), (0.5, 0.5), (1.0, 1.0), (2.0, 4.0)) == 0.25
    # A zero weight still separates points that tie on the other objective.
    assert tchebycheff((2.0, 5.0), (1.0, 0.0), (1.0, 1.0)) == 1.0
    assert tchebycheff((1.0, 5.0), (1.0, 0.0), (1.0, 1.0)) == pytest.approx(4e-6)


def test_weight_neighbourhoods_list_the_nearest_weights_first() -> None:
    assert weight_neighbourhoods(das_dennis(2, 2), 2) == [[0, 1], [1, 0], [2, 1]]
    hoods = weight_neighbourhoods(das_dennis(4, 2), 3)
    assert hoods[0] == [0, 1, 2]
    assert hoods[2] == [2, 1, 3]
    assert hoods[4] == [4, 3, 2]


# -------------------------------------------------------------------- RVEA


def test_unit_vectors_and_smallest_angles() -> None:
    vectors = unit_vectors(das_dennis(2, 2))
    assert vectors[1] == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))
    assert smallest_angles(vectors) == pytest.approx([math.pi / 4] * 3)
    assert smallest_angles([(1.0,)]) == [1.0]
    assert unit_vectors([(0.0, 0.0)]) == [(0.0, 0.0)]


RVEA_POINTS = [(0.0, 3.0), (0.5, 2.5), (1.0, 1.0), (2.0, 2.0), (3.0, 0.0)]


def test_angle_penalized_distance_moves_from_convergence_to_spread() -> None:
    vectors = unit_vectors(das_dennis(2, 2))
    gammas = smallest_angles(vectors)
    # Early, the point nearest the ideal point wins the first vector; late, the aligned one.
    assert rvea_survival(RVEA_POINTS, vectors, gammas, progress=0.0) == [1, 2, 4]
    assert rvea_survival(RVEA_POINTS, vectors, gammas, progress=1.0) == [0, 2, 4]


def test_angle_penalized_selection_prefers_feasible_members_within_a_group() -> None:
    vectors = unit_vectors(das_dennis(2, 2))
    gammas = smallest_angles(vectors)
    violations = [0.0, 1.0, 0.0, 0.0, 5.0]
    assert rvea_survival(RVEA_POINTS, vectors, gammas, 0.0, 2.0, violations) == [0, 2, 4]


def test_reference_vectors_adapt_to_the_objective_ranges() -> None:
    adapted = adapt_vectors(unit_vectors(das_dennis(2, 2)), (0.0, 0.0), (1.0, 3.0))
    assert adapted[0] == pytest.approx((0.0, 1.0))
    assert adapted[1] == pytest.approx((1.0 / math.sqrt(10.0), 3.0 / math.sqrt(10.0)))
    assert adapted[2] == pytest.approx((1.0, 0.0))


# ------------------------------------------------------------- AGE-MOEA-II


def test_newton_curvature_recovers_the_exponent_of_a_point() -> None:
    assert newton_curvature((1 / 3, 1 / 3, 1 / 3)) == pytest.approx(1.0, abs=1e-2)
    assert newton_curvature((0.5774, 0.5774, 0.5774)) == pytest.approx(2.0, abs=1e-2)
    assert newton_curvature((0.1111, 0.1111, 0.1111)) == pytest.approx(0.5, abs=1e-2)
    assert newton_curvature((0.25, 0.25)) == pytest.approx(0.5, abs=1e-3)
    assert math.isnan(newton_curvature((0.0, 0.0)))
    assert math.isnan(newton_curvature((1.0, 0.0)))


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 3.0])
def test_age_moea_ii_recovers_the_curvature_of_a_known_front(p: float) -> None:
    unit = [(0.0, 1.0), (1.0, 0.0)] + [
        (t, (1.0 - t**p) ** (1.0 / p)) for t in (0.15, 0.3, 0.45, 0.6, 0.8)
    ]
    front = [(5.0 + 10.0 * x, 50.0 + 1000.0 * y) for x, y in unit]
    _, estimate = age_scores(front)
    assert estimate == pytest.approx(p, rel=1e-2)


def test_front_curvature_on_normalized_fronts() -> None:
    linear = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.5, 0.0), (1 / 3,) * 3]
    sphere = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5774, 0.5774, 0.5774)]
    convex = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.1111, 0.1111, 0.1111)]
    assert front_curvature(linear) == pytest.approx(1.0, abs=1e-2)
    assert front_curvature(sphere) == pytest.approx(2.0, abs=1e-2)
    assert front_curvature(convex) == pytest.approx(0.5, abs=1e-2)
    assert front_curvature([(1.0, 0.0), (0.0, 1.0)]) == 1.0


def test_projection_and_geodesic_distance() -> None:
    assert project_to_front((1.0, 1.0), 2.0) == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))
    assert project_to_front((0.0, 0.0), 2.0) == (0.0, 0.0)
    assert minkowski_norm((3.0, 4.0), 2.0) == pytest.approx(5.0)
    assert minkowski_norm((1.0, 1.0), 1.0) == pytest.approx(2.0)
    assert minkowski_norm((0.0, 0.0), 0.5) == 0.0
    assert geodesic_distance((1.0, 0.0), (0.0, 1.0), 1.0) == pytest.approx(math.sqrt(2.0))
    # On the circle the two chords through the projected midpoint lie below the arc.
    arc = geodesic_distance((1.0, 0.0), (0.0, 1.0), 2.0)
    assert arc == pytest.approx(2.0 * math.sqrt(2.0 - math.sqrt(2.0)))
    assert math.sqrt(2.0) < arc < math.pi / 2


CIRCLE = [
    (1.0, 0.0),
    (0.0, 1.0),
    (math.cos(math.radians(20)), math.sin(math.radians(20))),
    (math.cos(math.radians(50)), math.sin(math.radians(50))),
]


def test_age_scores_rank_points_greedily_by_geodesic_spread() -> None:
    def distance(degrees: float) -> float:
        """Distance on the unit circle between points ``degrees`` apart, through the midpoint."""
        return 4.0 * math.sin(math.radians(degrees) / 4.0)

    scores, p = age_scores(CIRCLE)
    assert p == pytest.approx(2.0, rel=1e-3)
    assert scores[:2] == [math.inf, math.inf]
    # The point at 50 degrees is farther from the extremes, so it is ranked first...
    assert scores[3] == pytest.approx(distance(50) + distance(40), rel=1e-3)
    # ...and the point at 20 degrees then lies between the extreme at 0 and it.
    assert scores[2] == pytest.approx(distance(20) + distance(30), rel=1e-3)


def test_age_survival_scores_the_first_front_and_ranks_later_fronts_by_proximity() -> None:
    pool = [*CIRCLE, (1.0, 1.0)]
    survivors = age_survival(pool, 3)
    assert survivors.indices == (0, 1, 3)
    everyone = age_survival(pool, 5)
    assert everyone.indices == (0, 1, 2, 3, 4)
    assert everyone.keys[4][0] == 1.0
    assert everyone.keys[4][1] == pytest.approx(-1.0 / math.sqrt(2.0), rel=1e-3)


# ---------------------------------------------------------- the shared loop


def test_constrained_domination_ranks_infeasible_members_by_overrun() -> None:
    pool = [
        member((1.0, 2.0)),
        member((0.0, 0.0), 3.0),
        member((2.0, 1.0)),
        member((0.0, 1.0), 1.0),
    ]
    kept = feasible_first(
        pool,
        3,
        lambda members, size: keep(members, crowding_survival([m.vector for m in members], size)),
    )
    assert kept == [pool[0], pool[2], pool[3]]
    assert [m.fitness[0] for m in kept] == [0.0, 0.0, 1.0]
    # The infeasible member wins a tournament only when it is drawn twice.
    assert pool[2].fitness < pool[3].fitness
    rng = random.Random(0)
    wins = sum(tournament([pool[2], pool[3]], rng) is pool[3] for _ in range(2000))
    assert 0.2 < wins / 2000 < 0.3


def test_variation_keeps_sequences_feasible_and_differs_from_the_parents() -> None:
    problem = random_problem(3, tasks=10, max_modes=3, pools=1, density=0.3)
    search = Search(problem, StopCriteria(max_evaluations=10_000), 0, Nsga2.info)
    evolution = Evolution(search, crossover_rate=0.9, mutation_rate=0.2, mode_mutation=1.0)
    parents = [evolution.random_member() for _ in range(6)]
    items = evolution.items
    pick = random.Random(0)
    for _ in range(200):
        first, second = pick.sample(parents, 2)
        sequence, modes = evolution.vary(first, second)
        assert sorted(sequence) == list(range(items.count))
        position = {item: rank for rank, item in enumerate(sequence)}
        for item, before in enumerate(items.predecessors):
            assert all(position[other] < position[item] for other in before)
        assert all(
            0 <= mode < len(task.modes) for mode, task in zip(modes, problem.tasks, strict=True)
        )
        assert (tuple(sequence), tuple(modes)) not in (first.genome, second.genome)


def test_progress_follows_the_deterministic_budget() -> None:
    problem = random_problem(1, tasks=6)
    search = Search(problem, StopCriteria(max_evaluations=10, time_limit=100.0), 0, Nsga2.info)
    evolution = Evolution(search, 0.9, 0.2, 1.0)
    assert evolution.progress() == 0.0
    for _ in range(4):
        evolution.random_member()
    assert evolution.progress() == pytest.approx(0.4)


def test_a_problem_with_a_single_genome_is_evaluated_once() -> None:
    problem = Problem(
        "pair",
        tasks=(Task.on("a", "m", 2.0), Task.on("b", "m", 3.0)),
        resources=(Resource("m"),),
        dependencies=(Dependency("a", "b"),),
        objectives=("makespan", "total_completion_time"),
    )
    for solver_id in IDS:
        result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=50), seed=1)
        assert result.evaluations == 1
        assert result.stop_reason is StopReason.COMPLETED
        assert [point.values for point in result.front] == [(5.0, 7.0)]


# ----------------------------------------------------------------- solvers


def assert_valid_front(
    problem: Problem, result_front: tuple[FrontPoint, ...], primary: float | None
) -> None:
    values = [point.values for point in result_front]
    assert values, "the front is empty"
    assert values == sorted(values)
    for a in values:
        assert not any(dominates(b, a) for b in values)
    for point in result_front:
        assert point.schedule is not None
        assert check_schedule(problem, point.schedule) == ()
        assert point.values == pytest.approx(tuple(evaluate(problem, point.schedule).values()))
    assert min(value[0] for value in values) == pytest.approx(primary)


@pytest.mark.parametrize("solver_id", IDS)
def test_front_is_non_dominated_feasible_and_exact(solver_id: str) -> None:
    problem = offloading()
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=400), seed=4)
    assert result.feasible
    assert len(result.front) > 1
    assert_valid_front(problem, result.front, result.value)


@pytest.mark.parametrize("solver_id", IDS)
def test_front_meets_hard_deadlines_that_random_schedules_miss(solver_id: str) -> None:
    problem = offloading(deadline_factor=1.1)
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=400), seed=2)
    assert result.feasible
    assert_valid_front(problem, result.front, result.value)


@pytest.mark.parametrize("solver_id", IDS)
def test_three_objectives(solver_id: str) -> None:
    problem = offloading().with_objectives("mean_response_time", "energy", "cost")
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=300), seed=5)
    assert_valid_front(problem, result.front, result.value)
    assert all(len(point.values) == 3 for point in result.front)


@pytest.mark.parametrize("solver_id", IDS)
def test_replays_the_same_front_from_a_seed(solver_id: str) -> None:
    problem = offloading()
    stop = StopCriteria(max_evaluations=300)
    first = REGISTRY.create(solver_id).solve(problem, stop, seed=9)
    second = REGISTRY.create(solver_id).solve(problem, stop, seed=9)
    assert first.front == second.front
    assert first.schedule == second.schedule


@pytest.mark.parametrize(
    "problem_factory", [offloading, tardy_flow_shop], ids=["offloading", "flow-shop"]
)
@pytest.mark.parametrize("solver_id", IDS)
def test_front_beats_random_search_at_equal_evaluations(solver_id: str, problem_factory) -> None:
    problem = problem_factory()
    budget = 500
    stop = StopCriteria(max_evaluations=budget)
    fronts = [
        [point.values for point in REGISTRY.create(solver_id).solve(problem, stop, seed).front]
        for seed in SEEDS
    ]
    baselines = [random_archive(problem, seed, budget) for seed in SEEDS]
    points = [point for front in (*fronts, *baselines) for point in front]
    low = [min(values) for values in zip(*points, strict=True)]
    high = [max(values) for values in zip(*points, strict=True)]
    reference = [top + 0.1 * (top - bottom) + 1e-9 for bottom, top in zip(low, high, strict=True)]
    found = statistics.median(hypervolume(front, reference) for front in fronts)
    sampled = statistics.median(hypervolume(front, reference) for front in baselines)
    assert found > sampled


@pytest.mark.parametrize("solver_id", IDS)
def test_one_objective_improves_on_random_search(solver_id: str) -> None:
    problem = tardy_flow_shop(("makespan",))
    budget = 400
    stop = StopCriteria(max_evaluations=budget)
    results = [REGISTRY.create(solver_id).solve(problem, stop, seed) for seed in SEEDS]
    for result in results:
        assert len(result.front) == 1
        assert result.front[0].values == (result.value,)
    sampled = statistics.median(
        min(v[0] for v in random_archive(problem, s, budget)) for s in SEEDS
    )
    values = [result.value for result in results]
    assert None not in values
    assert statistics.median(value for value in values if value is not None) < sampled


def test_reference_methods_size_their_population_from_the_lattice() -> None:
    problem = offloading().with_objectives("mean_response_time", "energy", "cost")
    stop = StopCriteria(max_iterations=1)
    assert MoeaD().solve(problem, stop, 1).extras["population"] == 36
    assert Rvea().solve(problem, stop, 1).extras["population"] == 36
    assert Nsga3().solve(problem, stop, 1).extras["population"] == 40
    assert Nsga2(population_size=12).solve(problem, stop, 1).extras["population"] == 12
    assert MoeaD().solve(offloading(), stop, 1).extras["population"] == 40


def test_one_generation_breeds_one_child_per_member() -> None:
    problem = offloading()
    for factory in (Nsga2, Nsga3, MoeaD, Spea2, Ibea, SmsEmoa, Rvea, AgeMoea2):
        result = factory(population_size=10).solve(problem, StopCriteria(max_iterations=3), 1)
        assert result.evaluations == 40
        assert result.iterations == 3


@pytest.mark.parametrize(
    ("solver_id", "parameters"),
    [
        ("nsga-ii", {"population_size": 1}),
        ("nsga-iii", {"crossover_rate": 1.5}),
        ("spea2", {"mutation_rate": -0.1}),
        ("sms-emoa", {"mode_mutation": -1.0}),
        ("ibea", {"kappa": 0.0}),
        ("moea-d", {"neighbourhood_size": 0}),
        ("rvea", {"alpha": -1.0}),
        ("rvea", {"adaptation_frequency": 0.0}),
    ],
)
def test_rejects_invalid_parameters(solver_id: str, parameters: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        REGISTRY.create(solver_id, **parameters)
