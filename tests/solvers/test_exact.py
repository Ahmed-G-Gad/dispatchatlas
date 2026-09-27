"""Correctness evidence for the exact solvers.

``enumeration`` is the reference: it decodes every task order and mode vector
and needs no third-party package. It is itself checked against a plain brute
force over all task permutations. ``cp-sat`` and ``milp`` must then reproduce
its optimum, or its proof of infeasibility, on random problems that exercise
releases, lags, modes, cumulative pools, deadlines, network transfers, setup
times, the permutation constraint, zero processing times, and every objective
each solver models.

Beyond agreement, the tests pin hand-computed optima, the instants that
zero-duration tasks occupy on exclusive resources, the ft06 job shop of Fisher
and Thompson (1963), the orb07 job shop of Applegate and Cook (1991) with its
zero processing time (marked slow), honest statuses under time limits (a run
that cannot prove optimality reports ``FEASIBLE`` with a bound, never
``OPTIMAL``), and the refusal of anything a solver does not model exactly.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import logging
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.core import (
    ConfigError,
    Dependency,
    Link,
    MissingDependencyError,
    Mode,
    Network,
    Objective,
    Problem,
    Resource,
    Schedule,
    SetupTimes,
    SolverError,
    SolveResult,
    Status,
    StopCriteria,
    StopReason,
    Task,
    UnsupportedProblemError,
    check_schedule,
    evaluate,
)
from dispatchatlas.solvers import exact as exact_package
from dispatchatlas.solvers.evaluation import Evaluator
from dispatchatlas.solvers.exact import CpSat, Enumeration, TimeIndexedMilp, register
from dispatchatlas.solvers.exact import milp as milp_module
from dispatchatlas.solvers.exact.common import decimal_scale, serial_horizon
from dispatchatlas.solvers.registry import SolverRegistry
from tests.helpers import chain_problem, random_problem

OBJECTIVE_REGISTRY = importlib.import_module("dispatchatlas.core.objectives")._REGISTRY

HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None
HAS_PULP = importlib.util.find_spec("pulp") is not None

requires_ortools = pytest.mark.skipif(not HAS_ORTOOLS, reason="needs the ortools package")
requires_pulp = pytest.mark.skipif(not HAS_PULP, reason="needs the pulp package")

COMPLETE = StopCriteria()
GENEROUS = StopCriteria(time_limit=60.0)

OBJECTIVES = (
    "makespan",
    "total_completion_time",
    "total_flow_time",
    "mean_response_time",
    "total_tardiness",
    "max_lateness",
    "tardy_tasks",
    "energy",
    "cost",
)
MILP_OBJECTIVES = ("makespan", "total_completion_time", "total_tardiness")

FT06 = (
    (2, 1, 0, 3, 1, 6, 3, 7, 5, 3, 4, 6),
    (1, 8, 2, 5, 4, 10, 5, 10, 0, 10, 3, 4),
    (2, 5, 3, 4, 5, 8, 0, 9, 1, 1, 4, 7),
    (1, 5, 0, 5, 2, 5, 3, 3, 4, 8, 5, 9),
    (2, 9, 1, 3, 4, 5, 5, 4, 0, 3, 3, 1),
    (1, 3, 3, 3, 5, 9, 0, 10, 4, 4, 2, 1),
)
"""Fisher and Thompson (1963) 6x6 job shop ft06, from the OR-Library file jobshop1.

Each row is one job: (machine, duration) pairs in visiting order. The optimal
makespan is 55.
"""

ORB07 = (
    (0, 32, 1, 14, 2, 15, 3, 37, 4, 18, 5, 43, 6, 19, 7, 27, 8, 28, 9, 31),
    (0, 8, 3, 12, 4, 49, 8, 24, 9, 52, 6, 19, 7, 23, 5, 19, 2, 17, 1, 32),
    (0, 25, 7, 19, 3, 27, 2, 45, 6, 21, 4, 15, 1, 13, 5, 16, 8, 43, 9, 19),
    (0, 24, 1, 18, 3, 41, 8, 29, 5, 14, 2, 17, 4, 23, 7, 15, 6, 18, 9, 23),
    (0, 27, 6, 29, 1, 39, 3, 21, 5, 15, 2, 15, 9, 25, 7, 26, 8, 44, 4, 20),
    (4, 17, 0, 15, 6, 51, 8, 17, 2, 46, 3, 16, 9, 33, 7, 25, 5, 30, 1, 25),
    (4, 15, 3, 31, 0, 25, 9, 12, 2, 13, 1, 51, 8, 19, 7, 21, 6, 12, 5, 26),
    (4, 8, 6, 29, 9, 25, 3, 15, 8, 17, 2, 22, 7, 32, 5, 20, 0, 11, 1, 28),
    (2, 41, 6, 10, 8, 32, 7, 5, 4, 21, 9, 59, 3, 26, 1, 10, 5, 16, 0, 29),
    (2, 20, 9, 7, 5, 44, 8, 22, 6, 33, 3, 25, 7, 29, 4, 12, 1, 14, 0, 0),
)
"""Applegate and Cook (1991) 10x10 job shop orb07, from the OR-Library file jobshop1.

The last operation of the last job takes no time. The optimal makespan, 397,
is the value the JSPLIB collection lists (github.com/tamy0612/JSPLIB,
``instances.json``).
"""


# ------------------------------------------------------------------ builders


def job_shop(name: str, rows) -> Problem:
    """Build a job shop from rows of (machine, duration) pairs."""
    tasks, dependencies = [], []
    machines = set()
    for j, row in enumerate(rows):
        for o in range(len(row) // 2):
            machine, duration = row[2 * o], row[2 * o + 1]
            machines.add(machine)
            tasks.append(Task.on(f"j{j}o{o}", f"m{machine}", duration, group=f"j{j}"))
            if o:
                dependencies.append(Dependency(f"j{j}o{o - 1}", f"j{j}o{o}"))
    return Problem(
        name,
        tasks=tuple(tasks),
        resources=tuple(Resource(f"m{m}") for m in sorted(machines)),
        dependencies=tuple(dependencies),
    )


def random_job_shop(seed: int, jobs: int, machines: int) -> Problem:
    """A random job shop with durations 1-99 and random machine routes."""
    rng = random.Random(seed)
    rows = []
    for _ in range(jobs):
        route = list(range(machines))
        rng.shuffle(route)
        rows.append(tuple(v for m in route for v in (m, rng.randint(1, 99))))
    return job_shop(f"job-shop-{seed}", rows)


def flow_shop(seed: int, jobs: int, machines: int, *, parallel: int = 1) -> Problem:
    """A random permutation flow shop, optionally with parallel machines per stage."""
    rng = random.Random(seed)
    tasks, dependencies = [], []
    for j in range(jobs):
        release = float(rng.randint(0, 4)) if rng.random() < 0.4 else 0.0
        due = float(rng.randint(10, 40)) if rng.random() < 0.6 else None
        for stage in range(machines):
            modes = tuple(Mode(rng.randint(1, 9), {f"s{stage}k{k}": 1.0}) for k in range(parallel))
            tasks.append(
                Task(
                    f"j{j}s{stage}",
                    modes,
                    release=release if stage == 0 else 0.0,
                    due=due if stage == machines - 1 else None,
                    group=f"j{j}",
                )
            )
            if stage:
                lag = float(rng.randint(0, 1))
                dependencies.append(Dependency(f"j{j}s{stage - 1}", f"j{j}s{stage}", lag=lag))
    resources = tuple(
        Resource(f"s{stage}k{k}") for stage in range(machines) for k in range(parallel)
    )
    return Problem(
        f"flow-shop-{seed}",
        tasks=tuple(tasks),
        resources=resources,
        dependencies=tuple(dependencies),
        permutation=True,
    )


def powered(problem: Problem, seed: int) -> Problem:
    """Give resources idle and busy power and a cost rate, as exact decimals.

    Only machines draw busy power and cost, because a task holding one unit of
    a three-unit pool uses a third of it, which no decimal states exactly.
    """
    rng = random.Random(seed)
    resources = []
    for resource in problem.resources:
        idle = rng.choice((0.0, 0.5, 1.0))
        extra = rng.choice((0.5, 1.0, 2.0, 3.5)) if resource.unary else 0.0
        rate = rng.choice((0.0, 0.1, 0.25, 1.0)) if resource.unary else 0.0
        resources.append(
            replace(resource, idle_power=idle, busy_power=idle + extra, cost_rate=rate)
        )
    return replace(problem, resources=tuple(resources))


def core_problem(seed: int, extra: int = 0) -> Problem:
    """A random problem with the features every exact solver models.

    It has five or six tasks, plus ``extra``, on two or three machines and
    possibly a pool, with releases, lags, and, for some seeds, two modes per
    task and hard deadlines.
    """
    return random_problem(
        seed,
        tasks=5 + seed % 2 + extra,
        machines=2 + seed % 2,
        pools=seed % 2,
        max_modes=2 if seed % 3 == 0 else 1,
        deadlines=seed % 4 == 0,
    )


def permutation_flow_shop(name: str, times: list[list[int]]) -> Problem:
    """A permutation flow shop: job ``j`` spends ``times[j][s]`` on machine ``s``."""
    rows = [
        tuple(v for stage, duration in enumerate(job) for v in (stage, duration)) for job in times
    ]
    return replace(job_shop(name, rows), permutation=True)


def zero_time_job_shop(seed: int) -> Problem:
    """A job shop of two or three jobs on two or three machines, durations 0-9.

    About a third of the operations take no time.
    """
    rng = random.Random(seed)
    jobs, machines = 2 + seed % 2, 2 + seed // 2 % 2
    rows = []
    for _ in range(jobs):
        route = rng.sample(range(machines), machines)
        durations = [0 if rng.random() < 0.35 else rng.randint(1, 9) for _ in route]
        rows.append(tuple(v for pair in zip(route, durations, strict=True) for v in pair))
    return job_shop(f"zero-time-job-shop-{seed}", rows)


def zeroed(problem: Problem, seed: int) -> Problem:
    """Give about a third of the modes of ``problem`` zero duration."""
    rng = random.Random(seed)
    tasks = []
    for task in problem.tasks:
        modes = tuple(
            replace(mode, duration=0.0) if rng.random() < 0.35 else mode for mode in task.modes
        )
        tasks.append(replace(task, modes=modes))
    return replace(problem, id=f"{problem.id}-zeroed", tasks=tuple(tasks))


def has_zero_duration(problem: Problem) -> bool:
    """Return whether some mode of ``problem`` takes no time."""
    return any(mode.duration == 0.0 for task in problem.tasks for mode in task.modes)


# ------------------------------------------------------------------ oracles


def schedule_of(result: SolveResult) -> Schedule:
    """Return the result's schedule, which the test expects to exist."""
    assert result.schedule is not None
    return result.schedule


def value_of(result: SolveResult) -> float:
    """Return the result's primary objective value, which the test expects to exist."""
    assert result.value is not None
    return result.value


def bound_of(result: SolveResult) -> float:
    """Return the result's bound, which the test expects to exist."""
    assert result.bound is not None
    return result.bound


def gap_of(result: SolveResult) -> float:
    """Return the result's gap, which the test expects to exist."""
    assert result.gap is not None
    return result.gap


def optimum(problem: Problem, *, max_tasks: int = 8):
    """Solve ``problem`` to completion with the enumeration."""
    result = Enumeration(max_tasks=max_tasks).solve(problem, COMPLETE)
    assert result.status in (Status.OPTIMAL, Status.INFEASIBLE)
    assert result.extras["complete"] is True
    return result


def brute_force(problem: Problem) -> float | None:
    """Best primary objective over every task permutation and mode vector, or None.

    A plain loop without pruning, merging, or precedence-aware generation. It
    scores deadline-feasible decoded schedules only, so ``None`` means that no
    decoded schedule meets the deadlines.
    """
    evaluator = Evaluator(problem)
    choices = [range(len(task.modes)) for task in problem.tasks]
    best: float | None = None
    for order in itertools.permutations(range(problem.size)):
        for modes in itertools.product(*choices):
            score, _ = evaluator.evaluate(order, modes)
            if score.feasible and (best is None or score.value < best - 1e-9):
                best = score.value
    return best


def permutation_makespan(times: list[list[int]]) -> int:
    """Optimal makespan of a permutation flow shop by the classic recurrence."""
    best = math.inf
    for order in itertools.permutations(range(len(times))):
        finish = [0] * len(times[0])
        for job in order:
            for stage, duration in enumerate(times[job]):
                ready = max(finish[stage], finish[stage - 1] if stage else 0)
                finish[stage] = ready + duration
        best = min(best, finish[-1])
    return int(best)


def assert_matches(problem: Problem, result, reference) -> None:
    """Check that ``result`` proves the same optimum, or infeasibility, as ``reference``."""
    assert result.status == reference.status, (problem.id, result.status, reference.status)
    if reference.status is Status.INFEASIBLE:
        assert result.schedule is None
        return
    assert result.value == pytest.approx(reference.value, rel=1e-9, abs=1e-6), problem.id
    assert result.bound == pytest.approx(result.value)
    assert result.gap == pytest.approx(0.0, abs=1e-9)
    assert check_schedule(problem, schedule_of(result)) == ()
    assert result.objectives == pytest.approx(evaluate(problem, result.schedule))


# ----------------------------------------------------------- hand instances


def lag_problem() -> Problem:
    """One machine; ``b`` waits 4 after ``a``; ``c`` fills the gap. Optimum 9."""
    return Problem(
        "lags",
        tasks=(Task.on("a", "m", 2), Task.on("b", "m", 3), Task.on("c", "m", 1)),
        resources=(Resource("m"),),
        dependencies=(Dependency("a", "b", lag=4),),
    )


def window_problem() -> Problem:
    """``b`` must run in [1, 3], which pushes ``a`` behind it. Optimum 6."""
    return Problem(
        "window",
        tasks=(Task.on("a", "m", 3), Task.on("b", "m", 2, release=1, deadline=3)),
        resources=(Resource("m"),),
    )


def mode_problem() -> Problem:
    """``a`` runs 4 on the busy machine or 6 on a free one. Optimum 6."""
    return Problem(
        "modes",
        tasks=(Task("a", (Mode(4, {"m1": 1}), Mode(6, {"m2": 1}))), Task.on("b", "m1", 5)),
        resources=(Resource("m1"), Resource("m2")),
    )


def pool_problem() -> Problem:
    """Two 2-unit tasks cannot share a 3-unit pool; the 1-unit task runs beside both.

    Optimum 6.
    """
    return Problem(
        "pool",
        tasks=(
            Task.on("a", "pool", 3, demand=2),
            Task.on("b", "pool", 3, demand=2),
            Task.on("c", "pool", 6, demand=1),
        ),
        resources=(Resource("pool", capacity=3),),
    )


def network_problem() -> Problem:
    """Moving ``b`` to the cloud costs a 3-unit transfer but saves 4. Optimum 6."""
    return Problem(
        "network",
        tasks=(
            Task.on("a", "m0", 2),
            Task("b", (Mode(5, {"m0": 1}), Mode(1, {"m1": 1}))),
        ),
        resources=(Resource("m0", tier="edge"), Resource("m1", tier="cloud")),
        dependencies=(Dependency("a", "b", data=2.0),),
        network=Network((Link("edge", "cloud", bandwidth=1.0, latency=1.0),)),
    )


def tardiness_problem() -> Problem:
    """Shortest-due-first leaves ``a`` 2 late; the other order leaves ``b`` 4 late."""
    return Problem(
        "tardiness",
        tasks=(Task.on("a", "m", 4, due=4), Task.on("b", "m", 2, due=2)),
        resources=(Resource("m"),),
        objectives=("total_tardiness",),
    )


def infeasible_problem() -> Problem:
    """Two 3-unit tasks on one machine cannot both finish by 4."""
    return Problem(
        "infeasible",
        tasks=(Task.on("a", "m", 3, deadline=4), Task.on("b", "m", 3, deadline=4)),
        resources=(Resource("m"),),
    )


def setup_problem() -> Problem:
    """Starting with ``a`` gives makespan 8; starting with ``b`` gives 9."""
    return Problem(
        "setups",
        tasks=(Task.on("a", "m", 2), Task.on("b", "m", 2)),
        resources=(Resource("m"),),
        setups=SetupTimes({("m", None, "a"): 1, ("m", None, "b"): 5, ("m", "a", "b"): 3}),
    )


def instant_problem(at: float, *, shared: bool = False) -> Problem:
    """Task ``a`` runs 10 on the resource on which task ``z`` takes no time at ``at``.

    On a machine, ``a`` may not run across the instant ``z`` occupies. On a
    two-unit pool, where each task takes one unit, ``z`` holds nothing.
    """
    return Problem(
        f"instant-{at:g}-{'pool' if shared else 'machine'}",
        tasks=(Task.on("a", "r", 10), Task.on("z", "r", 0, release=at, deadline=at)),
        resources=(Resource("r", capacity=2.0 if shared else 1.0),),
    )


def johnson_problem() -> Problem:
    """Two-machine flow shop; Johnson's rule gives the optimal makespan 12."""
    times = {"j1": (3, 6), "j2": (5, 2), "j3": (1, 2)}
    tasks = []
    dependencies = []
    for job, (first, second) in times.items():
        tasks.append(Task.on(f"{job}a", "m1", first, group=job))
        tasks.append(Task.on(f"{job}b", "m2", second, group=job))
        dependencies.append(Dependency(f"{job}a", f"{job}b"))
    return Problem(
        "johnson",
        tasks=tuple(tasks),
        resources=(Resource("m1"), Resource("m2")),
        dependencies=tuple(dependencies),
        permutation=True,
    )


def decimal_problem() -> Problem:
    """Quarter-unit times: ``a`` then ``b`` on one machine, ``c`` waits 0.25 after ``a``.

    Optimum 2.25, stated exactly at time scale 100.
    """
    return Problem(
        "decimals",
        tasks=(Task.on("a", "m1", 1.25), Task.on("b", "m1", 0.5), Task.on("c", "m2", 0.75)),
        resources=(Resource("m1"), Resource("m2")),
        dependencies=(Dependency("a", "c", lag=0.25),),
    )


HAND_INSTANCES = [
    (lag_problem, 9.0),
    (window_problem, 6.0),
    (mode_problem, 6.0),
    (pool_problem, 6.0),
    (tardiness_problem, 2.0),
    (decimal_problem, 2.25),
]


# -------------------------------------------------------------- enumeration


@pytest.mark.parametrize(("build", "expected"), HAND_INSTANCES)
def test_enumeration_proves_hand_computed_optima(build, expected):
    result = Enumeration().solve(build(), COMPLETE)
    assert result.status is Status.OPTIMAL
    assert result.value == pytest.approx(expected)
    assert result.bound == pytest.approx(expected)
    assert result.stop_reason is StopReason.COMPLETED


def test_enumeration_models_network_transfers():
    result = Enumeration().solve(network_problem(), COMPLETE)
    assert result.value == pytest.approx(6.0)
    assert schedule_of(result).by_task["b"].mode == 1


def test_enumeration_solves_a_flow_shop_by_group_orders():
    result = Enumeration().solve(johnson_problem(), COMPLETE)
    assert result.status is Status.OPTIMAL
    assert result.value == pytest.approx(12.0)
    assert check_schedule(johnson_problem(), schedule_of(result)) == ()


@pytest.mark.parametrize("seed", range(10))
def test_enumeration_matches_plain_brute_force(seed):
    problem = core_problem(seed).with_objectives(OBJECTIVES[seed % 7])
    result = optimum(problem)
    expected = brute_force(problem)
    if expected is None:
        assert result.status is Status.INFEASIBLE
    else:
        assert result.value == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(6))
def test_enumeration_matches_the_flow_shop_recurrence(seed):
    rng = random.Random(seed)
    times = [[rng.randint(1, 9) for _ in range(3)] for _ in range(2 + seed % 2)]
    problem = permutation_flow_shop(f"pfsp-{seed}", times)
    # Group orders, not task orders, are enumerated, so nine tasks are cheap.
    result = Enumeration(max_tasks=9).solve(problem, COMPLETE)
    assert result.value == pytest.approx(permutation_makespan(times))


def test_enumeration_proves_infeasibility():
    result = Enumeration().solve(infeasible_problem(), COMPLETE)
    assert result.status is Status.INFEASIBLE
    assert result.schedule is None
    assert result.objectives == {}
    assert result.stop_reason is StopReason.COMPLETED


def test_enumeration_never_claims_optimality_when_cut_short():
    problem = core_problem(3)
    short = Enumeration().solve(problem, StopCriteria(max_evaluations=2))
    assert short.status is Status.FEASIBLE
    assert short.bound is None
    assert short.stop_reason is StopReason.EVALUATION_LIMIT
    assert short.extras["complete"] is False
    target = Enumeration().solve(problem, StopCriteria(target=1e9))
    assert target.status is Status.FEASIBLE
    assert target.stop_reason is StopReason.TARGET_REACHED


def test_enumeration_counts_one_evaluation_per_complete_schedule():
    result = Enumeration().solve(chain_problem(), COMPLETE)
    assert result.evaluations == result.iterations >= 1
    assert int(result.extras["states"]) >= result.evaluations


def test_enumeration_refuses_what_it_cannot_prove():
    with pytest.raises(UnsupportedProblemError, match="at most 8 tasks"):
        Enumeration().solve(random_problem(1, tasks=9), COMPLETE)
    with pytest.raises(UnsupportedProblemError, match="setups"):
        Enumeration().solve(setup_problem(), COMPLETE)
    # Decoding whole groups would place j1a before its predecessor j3b.
    johnson = johnson_problem()
    crossed = replace(johnson, dependencies=(*johnson.dependencies, Dependency("j3b", "j1a")))
    with pytest.raises(UnsupportedProblemError, match="joins two groups"):
        Enumeration().solve(crossed, COMPLETE)
    chain = Problem(
        "chain-9",
        tasks=tuple(Task.on(f"t{i}", "m", 1) for i in range(9)),
        resources=(Resource("m"),),
        dependencies=tuple(Dependency(f"t{i}", f"t{i + 1}") for i in range(8)),
    )
    assert Enumeration(max_tasks=9).solve(chain, COMPLETE).value == pytest.approx(9.0)


def test_enumeration_parameters_are_validated():
    registry = SolverRegistry()
    register(registry)
    with pytest.raises(ConfigError):
        registry.create("enumeration", max_tasks=0)


# ------------------------------------------------------------------ cp-sat


@pytest.mark.exact
@requires_ortools
@pytest.mark.parametrize("seed", range(36))
def test_cpsat_matches_the_enumeration(seed):
    problem = powered(core_problem(seed, 1), seed).with_objectives(OBJECTIVES[seed % 9])
    assert_matches(problem, CpSat().solve(problem, GENEROUS), optimum(problem))


@pytest.mark.exact
@requires_ortools
@pytest.mark.parametrize("seed", range(12))
def test_cpsat_matches_the_enumeration_with_network_transfers(seed):
    problem = random_problem(
        seed, tasks=5, machines=3, pools=seed % 2, max_modes=3, network=True
    ).with_objectives(OBJECTIVES[(seed + 4) % 9])
    assert_matches(problem, CpSat().solve(problem, GENEROUS), optimum(problem))


@pytest.mark.exact
@requires_ortools
@pytest.mark.parametrize("seed", range(8))
def test_cpsat_matches_brute_force_with_setup_times(seed):
    problem = random_problem(
        seed, tasks=5, machines=2, pools=seed % 2, max_modes=1 + seed % 2, setups=True
    ).with_objectives(OBJECTIVES[seed % 7])
    result = CpSat().solve(problem, GENEROUS)
    expected = brute_force(problem)
    if expected is None:
        assert result.status is Status.INFEASIBLE
        return
    assert result.status is Status.OPTIMAL
    assert result.value == pytest.approx(expected)
    assert check_schedule(problem, schedule_of(result)) == ()


@pytest.mark.exact
@requires_ortools
@pytest.mark.parametrize("seed", range(8))
def test_cpsat_matches_the_enumeration_on_permutation_flow_shops(seed):
    problem = flow_shop(seed, 3 + seed % 2, 2, parallel=1 + seed % 2)
    problem = problem.with_objectives(OBJECTIVES[seed % 7])
    assert_matches(problem, CpSat().solve(problem, GENEROUS), optimum(problem))


@pytest.mark.exact
@requires_ortools
@pytest.mark.parametrize(("build", "expected"), [*HAND_INSTANCES, (network_problem, 6.0)])
def test_cpsat_proves_hand_computed_optima(build, expected):
    result = CpSat().solve(build(), GENEROUS)
    assert result.status is Status.OPTIMAL
    assert result.value == pytest.approx(expected)


@pytest.mark.exact
@requires_ortools
def test_cpsat_models_setups_and_permutations_exactly():
    setups = CpSat().solve(setup_problem(), GENEROUS)
    assert setups.value == pytest.approx(8.0)
    assert [a.task for a in schedule_of(setups).assignments] == ["a", "b"]
    johnson = CpSat().solve(johnson_problem(), GENEROUS)
    assert johnson.value == pytest.approx(12.0)


@pytest.mark.exact
@requires_ortools
def test_cpsat_solves_ft06_to_its_known_optimum():
    problem = job_shop("ft06", FT06)
    result = CpSat().solve(problem, StopCriteria(time_limit=120.0))
    assert result.status is Status.OPTIMAL
    assert result.value == pytest.approx(55.0)
    assert result.bound == pytest.approx(55.0)
    assert check_schedule(problem, schedule_of(result)) == ()


@pytest.mark.slow
@pytest.mark.exact
@requires_ortools
def test_cpsat_proves_the_orb07_optimum_despite_its_zero_processing_time():
    problem = job_shop("orb07", ORB07)
    instant = problem.task("j9o9")
    assert instant.modes[0].duration == 0.0
    # One worker proves it in seconds; the limit only guards a stalled run.
    result = CpSat().solve(problem, StopCriteria(time_limit=600.0))
    assert result.status is Status.OPTIMAL
    assert result.value == pytest.approx(397.0)
    assert result.bound == pytest.approx(397.0)
    assert check_schedule(problem, schedule_of(result)) == ()


@pytest.mark.exact
@requires_ortools
def test_cpsat_reports_feasible_with_a_bound_when_the_time_limit_stops_it():
    # A 20x10 job shop that CP-SAT cannot close in seconds with one worker.
    problem = random_job_shop(3, 20, 10)
    result = CpSat().solve(problem, StopCriteria(time_limit=0.5))
    assert result.status is Status.FEASIBLE
    assert result.stop_reason is StopReason.TIME_LIMIT
    assert bound_of(result) < value_of(result)
    assert gap_of(result) > 0.0
    assert result.elapsed < 2.0
    assert check_schedule(problem, schedule_of(result)) == ()
    assert result.trace[-1].value == pytest.approx(result.value)


@pytest.mark.exact
@requires_ortools
def test_cpsat_proves_infeasibility():
    result = CpSat().solve(infeasible_problem(), GENEROUS)
    assert result.status is Status.INFEASIBLE
    assert result.schedule is None
    assert result.stop_reason is StopReason.COMPLETED


@pytest.mark.exact
@requires_ortools
def test_cpsat_counts_solutions_as_evaluations():
    problem = random_job_shop(1, 8, 5)
    result = CpSat().solve(problem, StopCriteria(max_evaluations=1))
    assert result.evaluations == result.iterations == 1
    assert result.stop_reason in (StopReason.EVALUATION_LIMIT, StopReason.COMPLETED)
    if result.stop_reason is StopReason.EVALUATION_LIMIT:
        assert result.status is Status.FEASIBLE
    with pytest.raises(SolverError, match="limit"):
        CpSat().solve(problem, StopCriteria(target=10.0))


@pytest.mark.exact
@requires_ortools
def test_cpsat_does_not_claim_optimality_at_a_positive_gap():
    problem = random_job_shop(3, 20, 10)
    result = CpSat(relative_gap=0.5).solve(problem, StopCriteria(time_limit=5.0))
    assert result.status is Status.FEASIBLE
    assert gap_of(result) <= 0.5
    assert result.extras["relative_gap"] == 0.5


@pytest.mark.exact
@requires_ortools
def test_cpsat_records_backend_facts():
    result = CpSat(threads=2).solve(decimal_problem(), GENEROUS)
    extras = result.extras
    assert extras["raw_status"] == "OPTIMAL"
    assert extras["threads"] == 2
    assert extras["time_scale"] == 100
    assert extras["backend_version"] != "unknown"


@pytest.mark.exact
@requires_ortools
def test_cpsat_log_goes_to_the_library_logger(caplog):
    with caplog.at_level(logging.INFO, logger="dispatchatlas"):
        CpSat(log=True).solve(chain_problem(), GENEROUS)
    assert any("CP-SAT" in record.getMessage() for record in caplog.records)


@pytest.mark.exact
@requires_ortools
def test_cpsat_refuses_what_it_cannot_state_exactly(monkeypatch):
    thirds = Problem("thirds", tasks=(Task.on("a", "m", 1 / 3),), resources=(Resource("m"),))
    with pytest.raises(UnsupportedProblemError, match="decimals"):
        CpSat().solve(thirds, GENEROUS)
    custom = Objective("makespan", "min", "time", "A redefinition.", lambda p, s: 0.0)
    monkeypatch.setitem(OBJECTIVE_REGISTRY, "makespan", custom)
    with pytest.raises(UnsupportedProblemError, match="custom definition"):
        CpSat().solve(chain_problem(), GENEROUS)


def test_cpsat_parameters_are_validated():
    registry = SolverRegistry()
    register(registry)
    for parameters in ({"threads": 0}, {"relative_gap": -0.1}, {"relative_gap": 2}, {"log": 1}):
        with pytest.raises(ConfigError):
            registry.create("cp-sat", **parameters)


def test_cpsat_reports_a_missing_ortools(monkeypatch):
    monkeypatch.delitem(sys.modules, "dispatchatlas.solvers.exact._cpsat", raising=False)
    monkeypatch.delattr(exact_package, "_cpsat", raising=False)
    for name in ("ortools", "ortools.sat", "ortools.sat.python", "ortools.sat.python.cp_model"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(MissingDependencyError, match="ortools"):
        CpSat().solve(chain_problem(), GENEROUS)


# -------------------------------------------------------------------- milp


@pytest.mark.exact
@requires_pulp
@pytest.mark.parametrize("seed", range(36))
def test_milp_matches_the_enumeration(seed):
    problem = core_problem(seed, 1).with_objectives(MILP_OBJECTIVES[seed % 3])
    assert_matches(problem, TimeIndexedMilp().solve(problem, GENEROUS), optimum(problem))


@pytest.mark.exact
@requires_pulp
@pytest.mark.parametrize(("build", "expected"), HAND_INSTANCES)
def test_milp_proves_hand_computed_optima(build, expected):
    result = TimeIndexedMilp().solve(build(), GENEROUS)
    assert result.status is Status.OPTIMAL
    assert result.value == pytest.approx(expected)


@pytest.mark.exact
@requires_pulp
def test_milp_proves_infeasibility():
    # Each task fits its window alone, so only the solver's search proves it.
    result = TimeIndexedMilp().solve(infeasible_problem(), GENEROUS)
    assert result.status is Status.INFEASIBLE
    assert result.schedule is None
    # A task whose deadline precedes its release plus duration has no start period.
    late = Problem(
        "late", tasks=(Task.on("a", "m", 3, release=2, deadline=4),), resources=(Resource("m"),)
    )
    result = TimeIndexedMilp().solve(late, GENEROUS)
    assert result.status is Status.INFEASIBLE
    assert result.extras["raw_status"] == "NO_START_PERIOD"


@pytest.mark.exact
@requires_pulp
def test_milp_never_claims_optimality_under_a_short_time_limit():
    # Single machine with releases: CBC needs far longer than a second to close it.
    rng = random.Random(6)
    tasks = tuple(
        Task.on(
            f"t{i}", "m", rng.randint(1, 9), release=rng.randint(0, 105), weight=rng.randint(1, 3)
        )
        for i in range(35)
    )
    problem = Problem(
        "single-machine", tasks, (Resource("m"),), objectives=("total_completion_time",)
    )
    result = TimeIndexedMilp().solve(problem, StopCriteria(time_limit=1.0))
    assert result.status in (Status.FEASIBLE, Status.UNKNOWN)
    assert result.stop_reason is StopReason.TIME_LIMIT
    bound = bound_of(result)
    if result.status is Status.FEASIBLE:
        assert bound <= value_of(result)
        assert check_schedule(problem, schedule_of(result)) == ()


def scripted_cbc(monkeypatch, status: int, sol_status: int, log: str) -> None:
    """Make PuLP's CBC command solve for real but report a scripted verdict and log."""
    import pulp  # noqa: PLC0415 - optional dependency

    class Scripted(pulp.COIN_CMD):  # type: ignore[misc]  # PuLP ships no type hints
        def actualSolve(self, lp, **kwargs):  # noqa: N802 - PuLP's hook name
            super().actualSolve(lp, **kwargs)
            Path(self.optionsDict["logPath"]).write_text(log, encoding="utf-8")
            lp.assignStatus(status, sol_status)
            return status

    monkeypatch.setattr(pulp, "COIN_CMD", Scripted)


@pytest.mark.exact
@requires_pulp
def test_milp_reads_a_time_limited_cbc_incumbent_as_feasible(monkeypatch):
    import pulp  # noqa: PLC0415 - optional dependency

    # PuLP maps CBC's "Stopped on time - objective value" to LpStatusOptimal.
    log = (
        "Result - Stopped on time limit\n\nObjective value:                6.00000000\n"
        "Lower bound:                    3.500\nGap:                            0.42\n"
        "Enumerated nodes:               12\n"
        "Total time (CPU seconds):       1.00   (Wallclock seconds):       1.02\n"
    )
    scripted_cbc(monkeypatch, pulp.LpStatusOptimal, pulp.LpSolutionIntegerFeasible, log)
    result = TimeIndexedMilp().solve(mode_problem(), StopCriteria(time_limit=5.0))
    assert result.status is Status.FEASIBLE
    assert result.stop_reason is StopReason.TIME_LIMIT
    assert result.value == pytest.approx(6.0)
    # CBC prints six significant digits, so the bound is loosened below 3.5.
    assert 3.499 < bound_of(result) < 3.5
    assert result.extras["raw_status"] == "Stopped on time limit"
    assert result.extras["nodes"] == 12


@pytest.mark.exact
@requires_pulp
@pytest.mark.parametrize(("wall", "expected"), [(0.15, Status.INFEASIBLE), (99.0, Status.UNKNOWN)])
def test_milp_trusts_cbc_infeasibility_only_before_its_time_limit(monkeypatch, wall, expected):
    import pulp  # noqa: PLC0415 - optional dependency

    # CBC reports an interrupted preprocessing in the words of a genuine proof.
    log = (
        "Cgl0000I Cut generators found to be infeasible! (or unbounded)\n"
        "Pre-processing says infeasible or unbounded\n"
        f"Total time (CPU seconds):       0.11   (Wallclock seconds):       {wall}\n"
    )
    scripted_cbc(monkeypatch, pulp.LpStatusInfeasible, pulp.LpSolutionNoSolutionFound, log)
    result = TimeIndexedMilp().solve(mode_problem(), StopCriteria(time_limit=5.0))
    assert result.status is expected
    assert result.schedule is None


@pytest.mark.exact
@requires_pulp
def test_milp_does_not_claim_optimality_at_a_positive_gap():
    problem = core_problem(6).with_objectives("total_completion_time")
    result = TimeIndexedMilp(relative_gap=0.9).solve(problem, GENEROUS)
    assert result.status in (Status.OPTIMAL, Status.FEASIBLE)
    if result.status is Status.OPTIMAL:
        assert result.bound == pytest.approx(result.value)


@pytest.mark.exact
@requires_pulp
def test_milp_refuses_what_it_does_not_model():
    for problem in (network_problem(), setup_problem(), johnson_problem()):
        with pytest.raises(UnsupportedProblemError, match="features"):
            TimeIndexedMilp().solve(problem, GENEROUS)
    with pytest.raises(UnsupportedProblemError, match="cannot optimise objective 'energy'"):
        TimeIndexedMilp().solve(chain_problem().with_objectives("energy"), GENEROUS)
    thirds = Problem("thirds", tasks=(Task.on("a", "m", 1 / 3),), resources=(Resource("m"),))
    with pytest.raises(UnsupportedProblemError, match="decimals"):
        TimeIndexedMilp().solve(thirds, GENEROUS)
    with pytest.raises(UnsupportedProblemError, match="max_variables"):
        TimeIndexedMilp(max_variables=5).solve(chain_problem(), GENEROUS)


@pytest.mark.exact
@requires_pulp
def test_milp_records_backend_facts():
    result = TimeIndexedMilp().solve(decimal_problem(), GENEROUS)
    extras = result.extras
    assert extras["backend"] == "cbc"
    assert str(extras["raw_status"]).startswith("Optimal")
    assert extras["threads"] == 1
    assert extras["time_scale"] == 100
    assert extras["period"] == pytest.approx(0.25)
    assert result.evaluations == 1


@pytest.mark.exact
@requires_pulp
@pytest.mark.parametrize("backend", ["highs", "scip", "gurobi"])
def test_milp_backends_match_the_enumeration(backend):
    module = {"highs": "highspy", "scip": "pyscipopt", "gurobi": "gurobipy"}[backend]
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"needs the {module} package")
    for problem in (core_problem(0), core_problem(4), infeasible_problem(), mode_problem()):
        result = TimeIndexedMilp(backend=backend).solve(problem, GENEROUS)
        assert_matches(problem, result, optimum(problem))
        assert result.extras["backend"] == backend


@pytest.mark.exact
@requires_pulp
@pytest.mark.parametrize("backend", ["highs", "scip", "gurobi"])
def test_milp_reports_a_missing_backend(monkeypatch, backend):
    monkeypatch.setattr(milp_module, "find_spec", lambda name: None)
    with pytest.raises(MissingDependencyError):
        TimeIndexedMilp(backend=backend).solve(chain_problem(), GENEROUS)


def test_milp_reports_a_missing_pulp(monkeypatch):
    monkeypatch.setitem(sys.modules, "pulp", None)
    with pytest.raises(MissingDependencyError, match="pulp"):
        TimeIndexedMilp().solve(chain_problem(), GENEROUS)


def test_milp_parameters_are_validated():
    registry = SolverRegistry()
    register(registry)
    for parameters in (
        {"backend": "cplex"},
        {"threads": 0},
        {"backend": "scip", "threads": 2},
        {"relative_gap": 1.5},
        {"max_variables": 0},
    ):
        with pytest.raises(ConfigError):
            registry.create("milp", **parameters)


# ---------------------------------------------------------- zero durations

EXACT_SOLVERS = (
    pytest.param(Enumeration(max_tasks=9), id="enumeration"),
    pytest.param(CpSat(), id="cp-sat", marks=(pytest.mark.exact, requires_ortools)),
    pytest.param(TimeIndexedMilp(), id="milp", marks=(pytest.mark.exact, requires_pulp)),
)
MODEL_SOLVERS = EXACT_SOLVERS[1:]


@pytest.mark.parametrize("solver", EXACT_SOLVERS)
@pytest.mark.parametrize(
    ("at", "shared", "expected"),
    [(0.0, False, 10.0), (5.0, False, 15.0), (10.0, False, 10.0), (5.0, True, 10.0)],
)
def test_zero_duration_tasks_are_instants_on_exclusive_resources_only(solver, at, shared, expected):
    # On the machine, a may touch the instant at either end but not run across it.
    problem = instant_problem(at, shared=shared)
    result = solver.solve(problem, GENEROUS)
    assert result.status is Status.OPTIMAL
    assert result.value == pytest.approx(expected)
    assert check_schedule(problem, schedule_of(result)) == ()


@pytest.mark.parametrize("solver", MODEL_SOLVERS)
@pytest.mark.parametrize("seed", range(8))
def test_zero_processing_times_in_job_shops_match_the_enumeration(solver, seed):
    problem = zero_time_job_shop(seed).with_objectives(MILP_OBJECTIVES[seed % 2])
    assert has_zero_duration(problem)
    assert_matches(problem, solver.solve(problem, GENEROUS), optimum(problem, max_tasks=9))


@pytest.mark.parametrize("solver", MODEL_SOLVERS)
@pytest.mark.parametrize("seed", range(8))
def test_zero_durations_on_machines_and_pools_match_the_enumeration(solver, seed):
    base = random_problem(
        seed, tasks=6, machines=2, pools=1, max_modes=1 + seed % 2, deadlines=seed % 3 == 0
    )
    problem = zeroed(base, seed).with_objectives(MILP_OBJECTIVES[seed % 3])
    assert has_zero_duration(problem)
    assert_matches(problem, solver.solve(problem, GENEROUS), optimum(problem))


@pytest.mark.parametrize("solver", EXACT_SOLVERS[:2])
@pytest.mark.parametrize("seed", range(6))
def test_zero_processing_times_in_flow_shops_follow_the_recurrence(solver, seed):
    rng = random.Random(seed)
    jobs, machines = 3 + seed % 2, 3 - seed % 2
    times = [
        [0 if rng.random() < 0.3 else rng.randint(1, 9) for _ in range(machines)]
        for _ in range(jobs)
    ]
    problem = permutation_flow_shop(f"zero-time-flow-shop-{seed}", times)
    assert has_zero_duration(problem)
    result = solver.solve(problem, GENEROUS)
    assert result.status is Status.OPTIMAL
    assert result.value == pytest.approx(permutation_makespan(times))
    assert check_schedule(problem, schedule_of(result)) == ()


# ---------------------------------------------------------------- scaling


def test_decimal_scale_is_the_smallest_exact_power_of_ten():
    assert decimal_scale([3.0, 7.0], "times") == 1
    assert decimal_scale([0.5, 2.0], "times") == 10
    assert decimal_scale([0.25, 1.1, 0.84], "times") == 100
    assert decimal_scale([0.1 + 0.2], "times") == 10
    assert decimal_scale([1e-6], "times") == 10**6
    with pytest.raises(UnsupportedProblemError, match="decimals"):
        decimal_scale([1 / 3], "times")
    with pytest.raises(UnsupportedProblemError, match="too large"):
        decimal_scale([1e15 + 0.5], "times")
    with pytest.raises(UnsupportedProblemError, match="too large"):
        decimal_scale([2.0**60], "times")
    # A float that is not within 1e-9 of any six-decimal value is refused.
    with pytest.raises(UnsupportedProblemError, match="decimals"):
        decimal_scale([0.1234567], "times")


def test_serial_horizon_bounds_every_task_run_one_after_another():
    assert serial_horizon(chain_problem(), 1) == 9
    assert serial_horizon(lag_problem(), 1) == 2 + 3 + 1 + 4
    assert serial_horizon(decimal_problem(), 100) == 125 + 50 + 75 + 25
    # Durations 2 + 2, plus the largest setup into each task: 1 into a, 5 into b.
    assert serial_horizon(setup_problem(), 1) == 2 + 2 + 1 + 5
