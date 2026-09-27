import math
import random
import statistics
import time
from collections.abc import Callable, Iterable, Sequence

import pytest

from dispatchatlas.core import (
    ConfigError,
    Dependency,
    Mode,
    Problem,
    Resource,
    SolverError,
    StopCriteria,
    StopReason,
    Task,
)
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.continuous import (
    Box,
    Clpso,
    CmaEs,
    DifferentialEvolution,
    FunctionRun,
    LShade,
    Minimum,
    Optimizer,
    ParticleSwarmOptimization,
    minimize,
    score_improvement,
)
from dispatchatlas.solvers.continuous.clpso import learning_probabilities, select_exemplars
from dispatchatlas.solvers.continuous.cmaes import Strategy
from dispatchatlas.solvers.continuous.core import initial_population, midpoint_repair
from dispatchatlas.solvers.continuous.encoded import ScheduleRun, heuristic_seed
from dispatchatlas.solvers.continuous.lshade import (
    SuccessMemory,
    lehmer_mean,
    planned_population,
)
from dispatchatlas.solvers.encoding import VectorEncoding
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.population import Umda
from dispatchatlas.solvers.priorities import upward_ranks
from dispatchatlas.solvers.search import Exhausted, Search
from tests.helpers import random_problem
from tests.solvers.test_constructive import value_of, violations
from tests.solvers.test_trajectory import placement_problem

REGISTRY = default_registry(plugins=False)

CONTINUOUS = [
    "differential-evolution",
    "particle-swarm-optimization",
    "clpso",
    "lshade",
    "cma-es",
]
VECTOR_METHODS = [*CONTINUOUS, "brkga", "umda"]


def sphere(x):
    return sum(value * value for value in x)


def rastrigin(x):
    return 10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x)


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def ellipsoid(x):
    """Condition number 10^6: coefficients 1 to 10^6 on a log scale."""
    return sum(10 ** (6 * i / (len(x) - 1)) * v * v for i, v in enumerate(x))


def median_value(
    optimizer: Optimizer,
    function: Callable[[Sequence[float]], float],
    box: Box,
    stop: StopCriteria,
    seeds: Iterable[int] = range(3),
) -> float:
    return statistics.median(minimize(optimizer, function, box, stop, seed=s).value for s in seeds)


def extra(result: Minimum, key: str) -> float:
    value = result.extras[key]
    assert isinstance(value, int | float)
    return float(value)


def weighted_tardiness(seed: int, tasks: int) -> Problem:
    """One machine, weighted tardiness: random search stays far from the optimum."""
    rng = random.Random(seed)
    durations = [rng.randint(1, 20) for _ in range(tasks)]
    horizon = sum(durations)
    items = tuple(
        Task.on(
            f"t{i}",
            "m",
            duration,
            due=rng.randint(horizon // 5, 4 * horizon // 5),
            weight=rng.randint(1, 10),
        )
        for i, duration in enumerate(durations)
    )
    return Problem(f"tardiness-{seed}", items, (Resource("m"),), objectives=("total_tardiness",))


def two_machine_tardiness(seed: int, tasks: int) -> Problem:
    """Two unrelated machines, weighted tardiness: each task has one mode per machine."""
    rng = random.Random(seed)
    durations = [(rng.randint(1, 20), rng.randint(1, 20)) for _ in range(tasks)]
    horizon = sum(min(pair) for pair in durations) // 2
    items = tuple(
        Task(
            f"t{i}",
            (Mode(first, {"m1": 1}), Mode(second, {"m2": 1})),
            due=rng.randint(horizon // 5, 4 * horizon // 5),
            weight=rng.randint(1, 10),
        )
        for i, (first, second) in enumerate(durations)
    )
    machines = (Resource("m1"), Resource("m2"))
    return Problem(f"machines-{seed}", items, machines, objectives=("total_tardiness",))


# ------------------------------------------------------------ analytic tests

SPHERE_TOLERANCE = {
    "de": (DifferentialEvolution(), 1e-20),
    "pso": (ParticleSwarmOptimization(), 1e-10),
    "clpso": (Clpso(), 1e-3),
    "lshade": (LShade(), 1e-15),
    "cma-es": (CmaEs(), 1e-50),
    "umda": (Umda(), 1e-15),
}


@pytest.mark.parametrize("name", sorted(SPHERE_TOLERANCE))
def test_every_core_minimises_the_sphere(name):
    # 5 dimensions, 10 000 evaluations: the tolerances hold with margin over seeds.
    optimizer, tolerance = SPHERE_TOLERANCE[name]
    stop = StopCriteria(max_evaluations=10_000)
    assert median_value(optimizer, sphere, Box.cube(5, -5.12, 5.12), stop) < tolerance


@pytest.mark.parametrize(
    ("optimizer", "function", "bound", "tolerance"),
    [
        (DifferentialEvolution(), rastrigin, 5.12, 1e-6),
        (LShade(), rastrigin, 5.12, 1e-6),
        (Clpso(population_size=10), rastrigin, 5.12, 1e-6),
        (LShade(), rosenbrock, 2.048, 1e-6),
        (CmaEs(), rosenbrock, 2.048, 1e-10),
        (ParticleSwarmOptimization(), rosenbrock, 2.048, 2.0),
        (DifferentialEvolution(), rosenbrock, 2.048, 2.0),
    ],
    ids=[
        "de-rastrigin",
        "lshade-rastrigin",
        "clpso-rastrigin",
        "lshade-rosenbrock",
        "cmaes-rosenbrock",
        "pso-rosenbrock",
        "de-rosenbrock",
    ],
)
def test_cores_reach_documented_tolerances_on_harder_functions(
    optimizer, function, bound, tolerance
):
    # Rastrigin has a local minimum in every unit cell; Rosenbrock a curved valley.
    # The swarm and DE only enter the valley within this budget.
    stop = StopCriteria(max_evaluations=10_000)
    assert median_value(optimizer, function, Box.cube(5, -bound, bound), stop) < tolerance


def test_minimize_honours_every_limit_and_counts_exactly():
    box = Box.cube(3, -1.0, 1.0)
    by_evaluations = minimize(
        DifferentialEvolution(), sphere, box, StopCriteria(max_evaluations=77)
    )
    assert by_evaluations.evaluations == 77
    assert by_evaluations.reason is StopReason.EVALUATION_LIMIT
    by_iterations = minimize(CmaEs(), sphere, box, StopCriteria(max_iterations=5))
    assert by_iterations.iterations == 5
    assert by_iterations.evaluations == 5 * Strategy.default(3).offspring
    assert by_iterations.reason is StopReason.ITERATION_LIMIT
    by_target = minimize(CmaEs(), sphere, box, StopCriteria(max_evaluations=10_000, target=1e-3))
    assert by_target.reason is StopReason.TARGET_REACHED
    assert by_target.value <= 1e-3
    by_time = minimize(LShade(), sphere, box, StopCriteria(time_limit=0.2))
    assert by_time.reason is StopReason.TIME_LIMIT
    assert by_time.elapsed < 1.5


def test_minimize_needs_a_bounded_budget_and_a_valid_box():
    with pytest.raises(SolverError):
        minimize(CmaEs(), sphere, Box.cube(2, 0.0, 1.0), StopCriteria(target=0.0))
    with pytest.raises(ValueError, match="below its upper bound"):
        Box((0.0, 1.0), (1.0, 1.0))
    with pytest.raises(ValueError, match="equal"):
        Box((0.0,), (1.0, 2.0))


def test_minimize_replays_from_a_seed():
    box = Box.cube(4, -2.0, 2.0)
    stop = StopCriteria(max_evaluations=500)
    for optimizer in (
        DifferentialEvolution(),
        ParticleSwarmOptimization(),
        Clpso(),
        LShade(),
        CmaEs(),
    ):
        first = minimize(optimizer, rastrigin, box, stop, seed=5)
        again = minimize(optimizer, rastrigin, box, stop, seed=5)
        other = minimize(optimizer, rastrigin, box, stop, seed=6)
        assert (first.vector, first.value, first.extras) == (
            again.vector,
            again.value,
            again.extras,
        )
        assert first.vector != other.vector


def test_initial_points_are_evaluated_first_and_seed_the_population():
    box = Box.cube(2, -5.0, 5.0)
    result = minimize(
        DifferentialEvolution(), sphere, box, StopCriteria(max_evaluations=1), initial=[(0.0, 0.0)]
    )
    assert result.vector == (0.0, 0.0)
    run = FunctionRun(sphere, box, StopCriteria(max_evaluations=100))
    run.seeds.append(([1.0, 1.0], 2.0))
    members, fitness = initial_population(run, 4, random.Random(0))
    assert members[0] == [1.0, 1.0]
    assert fitness[0] == 2.0
    assert run.evaluations == 3


def test_budget_progress_follows_the_tightest_limit():
    stop = StopCriteria(time_limit=10.0, max_evaluations=100, max_iterations=4)
    assert stop.progress(50, 1, 1.0) == 0.5
    assert stop.progress(10, 3, 1.0) == 0.75
    assert stop.progress(10, 1, 9.0) == 0.25
    assert stop.progress(10, 1, 9.0, deterministic=False) == 0.9
    assert stop.progress(500, 1, 1.0) == 1.0
    assert StopCriteria(time_limit=10.0, target=1.0).progress(10**6, 10**6, 2.0) == 0.2


def test_midpoint_repair_moves_halfway_back_to_the_parent():
    assert midpoint_repair(-3.0, 0.5, 0.0, 1.0) == 0.25
    assert midpoint_repair(4.0, 0.5, 0.0, 1.0) == 0.75
    assert midpoint_repair(0.3, 0.5, 0.0, 1.0) == 0.3


# ------------------------------------------------------------------- CMA-ES


def test_cma_es_strategy_follows_the_default_formulas():
    strategy = Strategy.default(10)
    assert strategy.offspring == 4 + int(3 * math.log(10)) == 10
    assert strategy.parents == 5
    raw = [math.log(5.5) - math.log(i) for i in range(1, 6)]
    assert strategy.weights == pytest.approx([w / sum(raw) for w in raw])
    mu_eff = 1 / sum((w / sum(raw)) ** 2 for w in raw)
    assert strategy.mu_eff == pytest.approx(mu_eff)
    assert strategy.c_sigma == pytest.approx((mu_eff + 2) / (10 + mu_eff + 5))
    assert strategy.c_1 == pytest.approx(2 / (11.3**2 + mu_eff))
    assert strategy.c_mu == pytest.approx(2 * (mu_eff - 2 + 1 / mu_eff) / (12**2 + mu_eff))
    assert strategy.chi_n == pytest.approx(math.sqrt(10) * (1 - 1 / 40 + 1 / 2100))
    # An odd population weighs ln((lambda + 1) / 2) - ln i, not ln(mu + 1/2) - ln i.
    odd = Strategy.default(3)
    assert odd.offspring == 7
    raw = [math.log(4.0) - math.log(i) for i in range(1, 4)]
    assert odd.weights == pytest.approx([w / sum(raw) for w in raw])


def test_cma_es_step_size_shrinks_on_the_sphere_and_grows_on_a_slope():
    box = Box.cube(5, -1.0, 1.0)
    converged = minimize(CmaEs(), sphere, box, StopCriteria(max_evaluations=3000), seed=1)
    assert extra(converged, "sigma") < 1e-10
    # On a linear function without bounds, cumulation lengthens the path and sigma grows.
    free = Box.cube(5, -1.0, 1.0, strict=False)
    walking = minimize(CmaEs(), lambda x: x[0], free, StopCriteria(max_iterations=40), seed=1)
    assert extra(walking, "sigma") > 100 * CmaEs().step_size


def test_cma_es_learns_the_shape_of_an_ellipsoid():
    # C approaches the inverse Hessian, whose axis ratio is sqrt(10^6) = 1000.
    result = minimize(
        CmaEs(), ellipsoid, Box.cube(5, -1.0, 1.0), StopCriteria(max_evaluations=4000)
    )
    assert result.value < 1e-20
    assert 300 < extra(result, "axis_ratio") < 3000
    isotropic = minimize(
        CmaEs(), sphere, Box.cube(5, -1.0, 1.0), StopCriteria(max_evaluations=4000)
    )
    assert extra(isotropic, "axis_ratio") < 5


def test_cma_es_keeps_its_mean_inside_a_strict_box():
    # The optimum sits in a corner, so most samples leave the box and are repaired.
    corner = minimize(CmaEs(), math.fsum, Box.cube(4, 0.0, 1.0), StopCriteria(max_evaluations=2000))
    assert corner.value == pytest.approx(0.0, abs=1e-9)
    assert all(0.0 <= v <= 1.0 for v in corner.vector)


# ------------------------------------------------------------------ L-SHADE


def test_planned_population_falls_linearly_to_the_minimum():
    assert planned_population(180, 4, 0.0) == 180
    assert planned_population(180, 4, 0.5) == 92
    assert planned_population(180, 4, 1.0) == 4
    assert planned_population(180, 4, 2.0) == 4


def test_lehmer_mean_weights_large_values_more():
    assert lehmer_mean([0.2, 0.8], [0.5, 0.5]) == pytest.approx((0.04 + 0.64) / 1.0)
    assert lehmer_mean([0.2, 0.8], [0.9, 0.1]) == pytest.approx(
        (0.9 * 0.04 + 0.1 * 0.64) / (0.9 * 0.2 + 0.1 * 0.8)
    )
    assert lehmer_mean([0.0, 0.0], [0.5, 0.5]) == 0.0


def test_success_memory_updates_one_slot_per_successful_generation():
    memory = SuccessMemory(3)
    memory.update([], [], [])
    assert memory.cursor == 0
    memory.update([0.2, 0.8], [0.6, 0.4], [3.0, 1.0])
    weights = [0.75, 0.25]
    assert memory.crossover[0] == pytest.approx(lehmer_mean([0.2, 0.8], weights))
    assert memory.scale[0] == pytest.approx(lehmer_mean([0.6, 0.4], weights))
    assert memory.cursor == 1
    # A slot whose successes all had CR = 0 turns terminal and stays terminal.
    memory.update([0.0, 0.0], [0.5, 0.7], [1.0, 1.0])
    assert memory.terminal[1]
    memory.cursor = 1
    memory.update([0.9], [0.5], [1.0])
    assert memory.terminal[1]
    rng = random.Random(0)
    memory.cursor = 0
    for _ in range(200):
        rate, scale = memory.sample(rng)
        assert 0.0 <= rate <= 1.0
        assert 0.0 < scale <= 1.0


def test_success_memory_draws_zero_crossover_from_terminal_slots():
    memory = SuccessMemory(1)
    memory.terminal[0] = True
    rng = random.Random(1)
    assert all(memory.sample(rng)[0] == 0.0 for _ in range(50))


@pytest.mark.parametrize(
    "stop",
    [StopCriteria(max_evaluations=6000), StopCriteria(max_iterations=60)],
    ids=["evaluations", "iterations"],
)
def test_lshade_population_reaches_its_minimum_under_a_budget(stop):
    result = minimize(LShade(), sphere, Box.cube(5, -5.0, 5.0), stop, seed=2)
    assert result.extras["population"] == LShade().min_population


def test_lshade_population_reduction_follows_the_clock_under_a_time_limit():
    # Near the end of the run each generation is short, so the last planned size
    # is at most a few members above the minimum, whatever the machine's speed.
    result = minimize(LShade(), sphere, Box.cube(5, -5.0, 5.0), StopCriteria(time_limit=0.5))
    assert extra(result, "population") <= LShade().min_population + 3


class ClockRun(FunctionRun):
    """A run whose progress is a simulated clock that advances with evaluations."""

    def __init__(self, seconds_per_evaluation: float, limit: float) -> None:
        super().__init__(sphere, Box.cube(3, -1.0, 1.0), StopCriteria(max_evaluations=10**6))
        self.seconds = seconds_per_evaluation
        self.limit = limit

    @property
    def progress(self) -> float:
        return min(1.0, self.evaluations * self.seconds / self.limit)

    def next_iteration(self) -> None:
        if self.evaluations * self.seconds >= self.limit:
            raise Exhausted(StopReason.TIME_LIMIT)
        super().next_iteration()


def test_lshade_reduction_is_keyed_to_the_binding_budget():
    run = ClockRun(seconds_per_evaluation=0.001, limit=2.0)
    with pytest.raises(Exhausted):
        LShade().optimize(run, random.Random(3))
    assert run.extras["population"] == LShade().min_population
    # 18 D = 54 members at first, but only about 2000 evaluations in the clock's budget.
    assert run.evaluations < 2100


# ------------------------------------------------------------ CLPSO and PSO


def test_clpso_learning_probabilities_rise_from_five_to_fifty_percent():
    values = learning_probabilities(10)
    assert values[0] == pytest.approx(0.05)
    assert values[-1] == pytest.approx(0.5)
    assert values == sorted(values)
    assert values[5] == pytest.approx(0.05 + 0.45 * (math.exp(50 / 9) - 1) / (math.exp(10) - 1))


def test_clpso_exemplars_always_include_another_particle():
    rng = random.Random(4)
    fitness = [5.0, 1.0, 3.0, 2.0]
    for particle in range(4):
        for _ in range(50):
            exemplars = select_exemplars(particle, fitness, 0.0, 6, rng)
            others = [choice for choice in exemplars if choice != particle]
            assert len(others) == 1
            assert all(0 <= choice < 4 for choice in exemplars)
    # With certainty of learning, particle 0 follows tournament winners, never itself.
    assert all(
        choice != 0 for _ in range(20) for choice in select_exemplars(0, fitness, 1.0, 6, rng)
    )


def test_particle_swarm_inertia_decreases_linearly_over_the_budget():
    swarm = ParticleSwarmOptimization()
    assert swarm.inertia(0.0) == 0.9
    assert swarm.inertia(0.5) == pytest.approx(0.65)
    assert swarm.inertia(1.0) == pytest.approx(0.4)


def test_differential_evolution_crossover_takes_one_or_all_mutant_coordinates():
    rng = random.Random(0)
    box = Box.cube(6, -10.0, 10.0)
    population = [[rng.random() for _ in range(6)] for _ in range(5)]
    for rate, expected in ((0.0, 1), (1.0, 6)):
        solver = DifferentialEvolution(crossover_rate=rate)
        for _ in range(20):
            trial = solver._trial(population, 0, box, rng)
            assert sum(value != population[0][j] for j, value in enumerate(trial)) == expected


# ---------------------------------------------------------- the scheduling bridge


def test_score_improvement_measures_the_leading_difference():
    assert score_improvement(Score(3.0, 10.0), Score(1.0, 50.0)) == 2.0
    assert score_improvement(Score(0.0, 10.0), Score(0.0, 7.0)) == 3.0
    assert score_improvement(Score(2.0, 10.0), Score(2.0, 4.0)) == 6.0
    assert score_improvement(Score(0.0, 5.0), Score(0.0, 5.0)) == 0.0


def test_heuristic_seed_encodes_the_upward_rank_schedule():
    problem = random_problem(3, tasks=10, max_modes=3)
    search = Search(problem, StopCriteria(max_evaluations=10), 0, REGISTRY.info("cma-es"))
    encoding = VectorEncoding(problem)
    vector, score = heuristic_seed(search, encoding)
    assert search.evaluations == 1
    order, modes = encoding.decode(vector)
    ranks = upward_ranks(problem)
    assert order == sorted(range(problem.size), key=lambda task: (-ranks[task], task))
    assert search.evaluator.evaluate(order, modes)[0] == score


@pytest.mark.parametrize("modes", ["keys", "builder"])
@pytest.mark.parametrize("solver_id", VECTOR_METHODS)
def test_seeded_runs_start_from_the_heuristic_schedule(solver_id, modes):
    problem = random_problem(4, tasks=10, max_modes=2)
    heft = REGISTRY.create("heft").solve(problem, StopCriteria())
    first = REGISTRY.create(solver_id, seeded=True, modes=modes).solve(
        problem, StopCriteria(max_evaluations=1), seed=0
    )
    assert first.schedule == heft.schedule
    later = REGISTRY.create(solver_id, seeded=True, modes=modes).solve(
        problem, StopCriteria(max_evaluations=300), seed=0
    )
    assert value_of(later) <= value_of(heft)


def test_schedule_runs_evaluate_through_the_search():
    problem = random_problem(5, tasks=8, max_modes=2)
    search = Search(problem, StopCriteria(max_evaluations=5), 0, REGISTRY.info("lshade"))
    run = ScheduleRun(search, VectorEncoding(problem))
    assert not run.box.strict
    assert run.box.dimension == VectorEncoding(problem).dimension
    run.evaluate([0.5] * run.box.dimension)
    assert search.evaluations == 1
    run.note("population", 7)
    assert search.extras["population"] == 7
    builder = ScheduleRun(search, VectorEncoding(problem, modes="builder"))
    assert builder.box.dimension == problem.size


@pytest.mark.parametrize("solver_id", CONTINUOUS)
def test_a_single_schedule_is_evaluated_once(solver_id):
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


def test_builder_modes_leave_a_chain_a_single_schedule():
    problem = Problem(
        "modal-line",
        tasks=tuple(Task(name, (Mode(1, {"m": 1}), Mode(2, {"m": 1}))) for name in "abc"),
        resources=(Resource("m"),),
        dependencies=(Dependency("a", "b"), Dependency("b", "c")),
    )
    stop = StopCriteria(max_evaluations=50)
    searched = REGISTRY.create("cma-es").solve(problem, stop)
    assert searched.evaluations == 50
    single = REGISTRY.create("cma-es", modes="builder").solve(problem, stop)
    assert single.stop_reason is StopReason.COMPLETED
    assert single.evaluations == 1
    assert single.value == 3.0


@pytest.mark.parametrize("solver_id", CONTINUOUS)
def test_mode_keys_find_the_cheapest_placement(solver_id):
    problem = placement_problem()
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=600), seed=1)
    assert result.schedule is not None
    assert {item.mode for item in result.schedule.assignments} == {1}
    assert violations(problem, result) == []


@pytest.mark.parametrize("solver_id", VECTOR_METHODS)
def test_builder_modes_run_every_task_in_its_earliest_finishing_mode(solver_id):
    # The fast, expensive mode finishes first, so the builder never saves cost.
    problem = placement_problem()
    solver = REGISTRY.create(solver_id, modes="builder")
    result = solver.solve(problem, StopCriteria(max_evaluations=100), seed=1)
    assert result.schedule is not None
    assert {item.mode for item in result.schedule.assignments} == {0}


@pytest.mark.parametrize(
    ("solver_id", "budget"),
    [(solver_id, 5000 if solver_id == "clpso" else 2000) for solver_id in CONTINUOUS],
)
def test_beats_random_search_at_equal_evaluations(solver_id, budget):
    # CLPSO keeps its swarm diverse and converges slowly by design, so it gets
    # a longer budget; random search gets the same one.
    problem = weighted_tardiness(7, 25)
    stop = StopCriteria(max_evaluations=budget)
    baseline = REGISTRY.create("random-search").solve(problem, stop, seed=3)
    result = REGISTRY.create(solver_id).solve(problem, stop, seed=3)
    assert result.evaluations == budget
    assert value_of(result) < value_of(baseline)


@pytest.mark.parametrize("solver_id", VECTOR_METHODS)
def test_builder_modes_beat_random_search_on_a_multi_mode_problem(solver_id):
    # Random search lets the builder choose modes as well; with mode keys the
    # vector methods would also have to learn a machine for every task.
    problem = two_machine_tardiness(3, 30)
    stop = StopCriteria(max_evaluations=2000)
    baseline = REGISTRY.create("random-search").solve(problem, stop, seed=3)
    result = REGISTRY.create(solver_id, modes="builder").solve(problem, stop, seed=3)
    assert result.evaluations == 2000
    assert value_of(result) < value_of(baseline)


@pytest.mark.parametrize(
    ("solver_id", "parameters"),
    [
        ("differential-evolution", {"population_size": 3}),
        ("differential-evolution", {"scale_factor": 0.0}),
        ("differential-evolution", {"crossover_rate": 1.5}),
        ("particle-swarm-optimization", {"inertia_end": 0.95}),
        ("particle-swarm-optimization", {"velocity_limit": 0.0}),
        ("clpso", {"population_size": 2}),
        ("clpso", {"refresh_gap": 0}),
        ("lshade", {"min_population": 3}),
        ("lshade", {"population_size": 3}),
        ("lshade", {"best_fraction": 0.0}),
        ("cma-es", {"population_size": 1}),
        ("cma-es", {"step_size": 0.0}),
        ("cma-es", {"modes": "random"}),
    ],
)
def test_invalid_parameters_are_refused(solver_id, parameters):
    with pytest.raises(ConfigError):
        REGISTRY.create(solver_id, **parameters)


def test_time_limited_schedule_runs_stop_promptly():
    problem = random_problem(6, tasks=30, max_modes=2)
    started = time.perf_counter()
    result = REGISTRY.create("cma-es").solve(problem, StopCriteria(time_limit=0.2), seed=0)
    assert result.stop_reason is StopReason.TIME_LIMIT
    assert time.perf_counter() - started < 1.5
