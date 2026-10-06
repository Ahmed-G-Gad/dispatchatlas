"""Tests of the population metaheuristics on priority vectors."""

from __future__ import annotations

import functools
import math
import random
import statistics
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
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
    check_schedule,
)
from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.references import bibliography
from dispatchatlas.solvers import rules, swarm
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.scheduling import Decoder
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.swarm import (
    METHODS,
    ArtificialBeeColony,
    GreyWolfOptimizer,
    Jaya,
    MarinePredatorsAlgorithm,
    SalpSwarmAlgorithm,
    Swarm,
    SwarmMethod,
    TeachingLearningBasedOptimization,
)
from dispatchatlas.solvers.swarm._engine import (
    UPPER,
    bound,
    iterate,
    levy,
    magnitudes,
    mantegna_sigma,
    method_info,
)
from tests.helpers import random_problem


def _registry() -> SolverRegistry:
    registry = SolverRegistry()
    rules.register(registry)
    swarm.register(registry)
    return registry


REGISTRY = _registry()

IDS = [
    "grey-wolf-optimizer",
    "whale-optimization-algorithm",
    "firefly-algorithm",
    "moth-flame-optimization",
    "sine-cosine-algorithm",
    "arithmetic-optimization-algorithm",
    "harris-hawks-optimization",
    "teaching-learning-based-optimization",
    "salp-swarm-algorithm",
    "marine-predators-algorithm",
    "cuckoo-search",
    "artificial-bee-colony",
    "gravitational-search-algorithm",
    "jaya",
]

# Evaluations per iteration with the default population of 20: a fixed count,
# or the inclusive range of a method whose trials depend on its draws.
PER_ITERATION: dict[str, tuple[int, int]] = {
    **dict.fromkeys(IDS, (20, 20)),
    "harris-hawks-optimization": (20, 40),
    "teaching-learning-based-optimization": (40, 40),
    "marine-predators-algorithm": (40, 40),
    "cuckoo-search": (25, 25),
    "artificial-bee-colony": (40, 41),
}


def single_machine(seed: int, jobs: int) -> Problem:
    """Jobs on one machine, minimising weighted tardiness: every key matters."""
    rng = random.Random(seed)
    durations = [rng.randint(1, 20) for _ in range(jobs)]
    horizon = sum(durations)
    tasks = tuple(
        Task.on(
            f"t{i}",
            "m",
            duration,
            due=float(rng.randint(duration, horizon // 2)),
            weight=float(rng.randint(1, 5)),
        )
        for i, duration in enumerate(durations)
    )
    return Problem(f"single-{seed}", tasks, (Resource("m"),), objectives=("total_tardiness",))


def two_machines(seed: int, jobs: int) -> Problem:
    """Jobs on two unrelated machines, one mode per machine, minimising weighted tardiness."""
    rng = random.Random(seed)
    durations = [(rng.randint(1, 20), rng.randint(1, 20)) for _ in range(jobs)]
    horizon = sum(min(pair) for pair in durations) // 2
    tasks = tuple(
        Task(
            f"t{i}",
            (Mode(first, {"a": 1}), Mode(second, {"b": 1})),
            due=float(rng.randint(horizon // 5, 4 * horizon // 5)),
            weight=float(rng.randint(1, 10)),
        )
        for i, (first, second) in enumerate(durations)
    )
    machines = (Resource("a"), Resource("b"))
    return Problem(f"two-{seed}", tasks, machines, objectives=("total_tardiness",))


def flow_shop(seed: int, jobs: int, machines: int) -> Problem:
    """A permutation flow shop with processing times uniform in 1..99."""
    rng = random.Random(seed)
    tasks, dependencies = [], []
    for j in range(jobs):
        for m in range(machines):
            tasks.append(Task.on(f"j{j}m{m}", f"m{m}", rng.randint(1, 99), group=f"j{j}"))
            if m:
                dependencies.append(Dependency(f"j{j}m{m - 1}", f"j{j}m{m}"))
    return Problem(
        f"flow-{seed}",
        tuple(tasks),
        tuple(Resource(f"m{m}") for m in range(machines)),
        tuple(dependencies),
        permutation=True,
    )


class SphereSwarm(Swarm):
    """Scores vectors by their squared distance to a point off the centre and corners."""

    def __init__(self, search: Search, size: int) -> None:
        super().__init__(search, size)
        self.target = np.linspace(0.2, 0.8, self.dimension)
        self.calls = 0

    def score(self, position: np.ndarray) -> Score:
        self.calls += 1
        return Score(0.0, float(np.sum((position - self.target) ** 2)))


def sphere_swarm(method: SwarmMethod, stop: StopCriteria, seed: int = 0) -> SphereSwarm:
    """Return an initialised sphere swarm inside an open search, ready to iterate."""
    problem = Problem("free", tuple(Task.on(f"t{i}", "m", 1) for i in range(10)), (Resource("m"),))
    return SphereSwarm(Search(problem, stop, seed, method.info), method.population_size)


def value(result: object) -> float:
    number = getattr(result, "value", None)
    assert isinstance(number, float)
    return number


# ------------------------------------------------------------------ registry


def test_the_package_registers_the_fourteen_methods():
    assert [method.info.id for method in METHODS] == IDS
    for solver_id in IDS:
        info = REGISTRY.info(solver_id)
        assert info.family == "swarm"
        assert info.stochastic
        assert info.iterative
        assert info.summary.endswith(".")
        assert info.parameters["population_size"] == 20
        assert info.parameters["rule"] == "random-key"
        assert info.parameters["modes"] == "keys"
        assert "bean1994genetic" in info.references
        assert all(key in bibliography() for key in info.references)


@pytest.mark.parametrize(
    ("solver_id", "parameters"),
    [
        ("grey-wolf-optimizer", {"population_size": 2}),
        ("jaya", {"population_size": 1}),
        ("jaya", {"population_size": 2.5}),
        ("jaya", {"rule": "smallest-value"}),
        ("jaya", {"modes": "random"}),
        ("whale-optimization-algorithm", {"spiral_shape": 0.0}),
        ("firefly-algorithm", {"noise_decay": 0.0}),
        ("firefly-algorithm", {"attraction": -1.0}),
        ("moth-flame-optimization", {"spiral_shape": math.inf}),
        ("sine-cosine-algorithm", {"amplitude": -2.0}),
        ("arithmetic-optimization-algorithm", {"moa_min": 0.9, "moa_max": 0.2}),
        ("arithmetic-optimization-algorithm", {"mu": 0.0}),
        ("marine-predators-algorithm", {"jump_probability": 1.5}),
        ("cuckoo-search", {"replaced_fraction": 0.0}),
        ("artificial-bee-colony", {"limit": 0}),
        ("gravitational-search-algorithm", {"gain": 0.0}),
    ],
)
def test_invalid_parameters_are_refused(solver_id, parameters):
    with pytest.raises(ConfigError):
        REGISTRY.create(solver_id, **parameters)


# -------------------------------------------------------------- accounting


@pytest.mark.parametrize("solver_id", IDS)
def test_reported_evaluations_are_the_schedules_decoded(solver_id, monkeypatch):
    decode = Decoder.decode
    calls = []

    def counting(self, order, modes=None):
        calls.append(1)
        return decode(self, order, modes)

    monkeypatch.setattr(Decoder, "decode", counting)
    problem = random_problem(3, tasks=10, max_modes=2, pools=1, deadlines=True)
    low, high = PER_ITERATION[solver_id]
    for iterations in (1, 3):
        calls.clear()
        result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_iterations=iterations))
        assert result.evaluations == len(calls)
        assert result.iterations == iterations
        assert 20 + iterations * low <= result.evaluations <= 20 + iterations * high
    calls.clear()
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=53), seed=2)
    assert result.evaluations == len(calls) == 53
    assert result.stop_reason is StopReason.EVALUATION_LIMIT


@pytest.mark.parametrize("solver_id", IDS)
def test_a_seed_replays_the_run(solver_id):
    problem = random_problem(4, tasks=12, max_modes=2, pools=1)
    stop = StopCriteria(max_evaluations=150)
    first = REGISTRY.create(solver_id).solve(problem, stop, seed=11)
    second = REGISTRY.create(solver_id).solve(problem, stop, seed=11)
    other = REGISTRY.create(solver_id).solve(problem, stop, seed=12)
    assert first.schedule == second.schedule
    assert first.objectives == second.objectives
    assert [(p.evaluations, p.value) for p in first.trace] == [
        (p.evaluations, p.value) for p in second.trace
    ]
    assert [(p.evaluations, p.value) for p in first.trace] != [
        (p.evaluations, p.value) for p in other.trace
    ]


@pytest.mark.parametrize("modes", ["keys", "builder"])
@pytest.mark.parametrize("rule", ["random-key", "largest-value"])
def test_every_representation_gives_feasible_schedules(rule, modes):
    problem = random_problem(5, tasks=10, max_modes=3, pools=1, network=True)
    solver = REGISTRY.create("jaya", rule=rule, modes=modes)
    result = solver.solve(problem, StopCriteria(max_evaluations=80))
    assert result.schedule is not None
    assert [v for v in check_schedule(problem, result.schedule) if v.code != "deadline"] == []


def test_builder_modes_leave_only_the_task_keys_to_search():
    problem = random_problem(6, tasks=10, max_modes=3, pools=1)
    flexible = sum(len(task.modes) > 1 for task in problem.tasks)
    assert flexible > 0
    search = Search(problem, StopCriteria(max_evaluations=10), 0, Jaya.info)
    assert Swarm(search, 2).dimension == problem.size + flexible
    assert Swarm(search, 2, modes="builder").dimension == problem.size


# ------------------------------------------------------------------ engine


def test_bound_reflects_components_into_the_unit_interval():
    folded = bound([-0.2, 1.3, 2.25, -1.5, 0.4, 1.0, math.nan, math.inf])
    assert folded.tolist() == pytest.approx([0.2, 0.7, 0.25, 0.5, 0.4, UPPER, 0.5, 0.5])
    assert folded[5] < 1.0


def test_magnitudes_rank_deadline_misses_after_every_feasible_member():
    scores = [Score(0.0, 5.0), Score(2.0, 1.0), Score(0.0, 7.0), Score(1.0, 3.0)]
    assert magnitudes(scores) == [5.0, 9.0, 7.0, 8.0]
    assert magnitudes([Score(2.0, 1.0), Score(1.0, 9.0)]) == [2.0, 1.0]


def test_the_elite_memory_keeps_the_best_vectors_in_order():
    swarm_ = sphere_swarm(Jaya(population_size=3), StopCriteria(max_iterations=1))
    target = swarm_.target
    first, _ = swarm_.evaluate(target + 0.05)
    for offset in (0.15, 0.1, 0.05, 0.12):
        swarm_.evaluate(target + offset)
    kept = [float(np.max(np.abs(vector - target))) for vector in swarm_.elite(5)]
    assert kept == pytest.approx([0.05, 0.05, 0.1])
    assert swarm_.elite(1)[0] is first is swarm_.best
    assert swarm_.best_score.value == pytest.approx(10 * 0.05**2)


@dataclass(frozen=True)
class Recorder(SwarmMethod):
    """Records the budget progress at every iteration and moves one member."""

    seen: list[float] | None = None

    info: ClassVar[SolverInfo] = method_info("recorder", "Recorder", "Moves one member.")

    def step(self, swarm: Swarm) -> None:
        assert self.seen is not None
        self.seen.append(swarm.progress)
        swarm.replace(0, swarm.rng.random(swarm.dimension))


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        (StopCriteria(max_iterations=4), [0.0, 0.25, 0.5, 0.75]),
        # The fifth iteration starts with the budget spent and stops at its evaluation.
        (StopCriteria(max_evaluations=24), [0.0, 0.25, 0.5, 0.75, 1.0]),
    ],
)
def test_progress_is_the_budget_fraction_used_after_initialisation(stop, expected):
    seen: list[float] = []
    method = Recorder(population_size=20, seen=seen)
    method.solve(random_problem(1, tasks=6, max_modes=1, pools=0), stop)
    assert seen == pytest.approx(expected)


@dataclass(frozen=True)
class Delayed(SwarmMethod):
    """Records the budget progress and loses ``delay`` seconds in iteration ``at``."""

    seen: list[float] | None = None
    delay: float = 0.0
    at: int = 1

    info: ClassVar[SolverInfo] = method_info("delayed", "Delayed", "Moves one member.")

    def step(self, swarm: Swarm) -> None:
        assert self.seen is not None
        self.seen.append(swarm.progress)
        if swarm.iteration == self.at:
            swarm.search._started -= self.delay
        swarm.replace(0, swarm.rng.random(swarm.dimension))


def _delayed(stop: StopCriteria, *, delay: float, at: int = 1):
    seen: list[float] = []
    method = Delayed(population_size=20, seen=seen, delay=delay, at=at)
    result = method.solve(random_problem(1, tasks=6, max_modes=1, pools=0), stop)
    return seen, result


def test_the_time_limit_plays_no_part_in_a_deterministic_schedule():
    counted = StopCriteria(time_limit=10.0, max_evaluations=24)
    seen, result = _delayed(counted, delay=5.0)
    assert seen == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert result.stop_reason is StopReason.EVALUATION_LIMIT
    assert result.extras == {"schedule_clock": "evaluations", "schedule_progress": 1.0}

    # The time limit ends the third of four iterations: the schedule stops short of its end.
    seen, result = _delayed(StopCriteria(time_limit=10.0, max_iterations=4), delay=20.0, at=3)
    assert seen == pytest.approx([0.0, 0.25, 0.5])
    assert result.stop_reason is StopReason.TIME_LIMIT
    assert result.extras == {"schedule_clock": "iterations", "schedule_progress": 0.75}


def test_a_binding_schedule_follows_the_nearest_limit():
    binding = StopCriteria(time_limit=10.0, max_evaluations=24, schedule="binding")
    seen, result = _delayed(binding, delay=5.0)
    assert seen == pytest.approx([0.0, 0.5, 0.5, 0.75, 1.0], abs=0.01)
    assert result.extras == {"schedule_clock": "evaluations", "schedule_progress": 1.0}

    timed = StopCriteria(time_limit=10.0, max_iterations=4, schedule="binding")
    seen, result = _delayed(timed, delay=20.0, at=3)
    assert seen == pytest.approx([0.0, 0.25, 0.5], abs=0.01)
    assert result.stop_reason is StopReason.TIME_LIMIT
    assert result.extras == {"schedule_clock": "time", "schedule_progress": 1.0}


def test_a_schedule_follows_the_time_when_it_is_the_only_limit():
    for schedule in ("deterministic", "binding"):
        stop = StopCriteria(time_limit=10.0, schedule=schedule)
        seen, result = _delayed(stop, delay=20.0, at=3)
        assert seen == pytest.approx([0.0, 0.0, 0.0], abs=0.01)
        assert result.extras == {"schedule_clock": "time", "schedule_progress": 1.0}


def test_a_run_that_starts_no_iteration_records_no_schedule():
    seen, result = _delayed(StopCriteria(max_evaluations=5), delay=0.0)
    assert seen == []
    assert result.extras == {}


def _slow_start(monkeypatch, seconds: float) -> None:
    """Make every swarm lose ``seconds`` after it measured its first iteration."""
    advance = Swarm.advance

    def slow_start(self: Swarm) -> None:
        advance(self)
        if self.search.iterations == 1:
            self.search._started -= seconds

    monkeypatch.setattr(Swarm, "advance", slow_start)


@pytest.mark.parametrize("method", METHODS, ids=IDS)
def test_a_slower_start_leaves_a_run_under_an_evaluation_limit_unchanged(method, monkeypatch):
    problem = random_problem(4, tasks=10)
    stop = StopCriteria(time_limit=60.0, max_evaluations=300)
    reference = method().solve(problem, stop, seed=5)
    assert reference.extras == {"schedule_clock": "evaluations", "schedule_progress": 1.0}

    _slow_start(monkeypatch, 20.0)
    slowed = method().solve(problem, stop, seed=5)
    assert slowed.schedule == reference.schedule
    assert slowed.objectives == reference.objectives
    assert [(point.evaluations, point.value) for point in slowed.trace] == [
        (point.evaluations, point.value) for point in reference.trace
    ]
    assert slowed.extras == reference.extras


def test_a_registered_method_records_the_clock_of_a_binding_schedule(monkeypatch):
    problem = random_problem(4, tasks=10)
    binding = StopCriteria(time_limit=60.0, max_evaluations=300, schedule="binding")
    counted = GreyWolfOptimizer().solve(problem, binding, seed=5)
    assert counted.extras == {"schedule_clock": "evaluations", "schedule_progress": 1.0}

    _slow_start(monkeypatch, 20.0)
    timed = StopCriteria(time_limit=15.0, max_evaluations=10**6, schedule="binding")
    ended = GreyWolfOptimizer().solve(problem, timed, seed=5)
    assert ended.stop_reason is StopReason.TIME_LIMIT
    assert ended.extras == {"schedule_clock": "time", "schedule_progress": 1.0}


@pytest.mark.parametrize("method", METHODS, ids=IDS)
def test_every_rule_improves_a_shifted_sphere(method):
    solver = method()
    swarm_ = sphere_swarm(solver, StopCriteria(max_iterations=80), seed=3)
    with swarm_.search:
        swarm_.initialize()
        start = swarm_.best_score.value
        iterate(solver, swarm_)
    assert swarm_.best_score.value < 0.75 * start


def test_the_leader_moves_collapse_to_the_leaders_mean_at_the_end():
    method = GreyWolfOptimizer()
    swarm_ = sphere_swarm(method, StopCriteria(max_iterations=5), seed=1)
    with swarm_.search:
        swarm_.initialize()
        expected = np.mean(swarm_.elite(3), axis=0)
        swarm_.progress = 1.0  # a = 0: every point lands on its leader
        method.step(swarm_)
    assert np.allclose(swarm_.positions, expected)


def test_the_chain_moves_each_member_halfway_to_the_one_ahead():
    method = SalpSwarmAlgorithm()
    swarm_ = sphere_swarm(method, StopCriteria(max_iterations=5), seed=2)
    with swarm_.search:
        swarm_.initialize()
        before = swarm_.positions.copy()
        swarm_.progress = 0.5
        method.step(swarm_)
    after = swarm_.positions
    assert np.allclose(after[1:], (before[1:] + after[:-1]) / 2.0)


@pytest.mark.parametrize(
    "method",
    [
        TeachingLearningBasedOptimization(),
        Jaya(),
        MarinePredatorsAlgorithm(),
        ArtificialBeeColony(limit=10**6),
    ],
    ids=["teaching-learning", "jaya", "marine-predators", "bee-colony"],
)
def test_greedy_rules_never_worsen_a_member(method):
    swarm_ = sphere_swarm(method, StopCriteria(max_iterations=30), seed=4)
    with swarm_.search:
        swarm_.initialize()
        for _ in range(10):
            before = list(swarm_.scores)
            swarm_.advance()
            method.step(swarm_)
            assert all(a <= b for a, b in zip(swarm_.scores, before, strict=True))


def test_the_bee_colony_restarts_one_stagnant_member_per_iteration():
    method = ArtificialBeeColony(limit=1)
    swarm_ = sphere_swarm(method, StopCriteria(max_iterations=10), seed=5)
    with swarm_.search:
        swarm_.initialize()
        swarm_.calls = 0
        swarm_.advance()
        method.step(swarm_)
    assert swarm_.calls == 2 * 20 + 1


# ------------------------------------------------------------------ Lévy


def test_mantegna_scale_matches_hand_computed_values():
    # beta = 1.5: (Gamma(2.5) sin(0.75 pi) / (Gamma(1.25) 1.5 2^0.25))^(1/1.5)
    #           = (1.32934 * 0.70711 / (0.90640 * 1.5 * 1.18921))^(2/3) = 0.69657
    assert mantegna_sigma(1.5) == pytest.approx(0.69657, abs=1e-5)
    assert mantegna_sigma(1.0) == pytest.approx(1.0)


def test_levy_steps_are_symmetric_and_heavy_tailed():
    steps = levy(np.random.default_rng(1), 200_000, beta=1.5)
    assert np.mean(steps > 0.0) == pytest.approx(0.5, abs=0.005)
    low, high = np.quantile(steps, [0.1, 0.9])
    assert low == pytest.approx(-high, rel=0.03)
    assert np.mean(np.abs(steps) > 10.0) > 0.005  # a normal law gives 1e-23


def test_levy_steps_of_index_one_follow_the_standard_cauchy_law():
    steps = levy(np.random.default_rng(2), 200_000, beta=1.0)
    assert np.median(np.abs(steps)) == pytest.approx(1.0, rel=0.02)
    assert np.quantile(steps, 0.25) == pytest.approx(-1.0, rel=0.03)
    assert np.quantile(steps, 0.75) == pytest.approx(1.0, rel=0.03)


@pytest.mark.parametrize("beta", [0.0, 2.0, -1.0])
def test_levy_refuses_indices_outside_zero_to_two(beta):
    with pytest.raises(ValueError, match="beta"):
        levy(np.random.default_rng(0), 3, beta=beta)


# ------------------------------------------------------ search effectiveness

SEQUENCING = single_machine(5, 12)
SEEDS = (1, 2, 3, 4, 5)


def median_value(problem: Problem, solver_id: str, budget: int, **parameters: str) -> float:
    stop = StopCriteria(max_evaluations=budget)
    solver = REGISTRY.create(solver_id, **parameters)
    return statistics.median(value(solver.solve(problem, stop, seed=seed)) for seed in SEEDS)


@functools.cache
def random_sequencing() -> float:
    return median_value(SEQUENCING, "random-search", 500)


NOT_ON_SEQUENCING = {
    "salp-swarm-algorithm": "only the head of the chain explores, one vector per iteration",
    "cuckoo-search": "its Lévy steps are a hundredth of the offset from the best vector, "
    "which rarely reorders keys, and a quarter of the vectors restart at random each time",
}


@pytest.mark.parametrize(
    "solver_id",
    [
        pytest.param(
            solver_id,
            marks=pytest.mark.xfail(strict=True, reason=NOT_ON_SEQUENCING[solver_id]),
        )
        if solver_id in NOT_ON_SEQUENCING
        else solver_id
        for solver_id in IDS
    ],
)
def test_beats_random_search_at_equal_evaluations(solver_id):
    assert median_value(SEQUENCING, solver_id, 500) < random_sequencing()


def test_cuckoo_search_beats_random_search_on_a_flow_shop():
    problem = flow_shop(3, 12, 8)
    random_median = median_value(problem, "random-search", 500)
    assert median_value(problem, "cuckoo-search", 500) < random_median


MACHINES = two_machines(3, 20)


@functools.cache
def random_on_machines() -> float:
    return median_value(MACHINES, "random-search", 500)


@pytest.mark.parametrize(
    "solver_id",
    [
        "grey-wolf-optimizer",
        "whale-optimization-algorithm",
        "teaching-learning-based-optimization",
        "marine-predators-algorithm",
        "jaya",
    ],
)
def test_builder_modes_make_the_methods_competitive_on_a_multi_mode_problem(solver_id):
    # Random search lets the builder choose every task's machine; with mode keys
    # the methods must also learn the machines from keys.
    builder = median_value(MACHINES, solver_id, 500, modes="builder")
    assert builder < random_on_machines()
    assert builder < median_value(MACHINES, solver_id, 500, modes="keys")
