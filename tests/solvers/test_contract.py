"""The contract every registered solver must honour.

Each test runs over the whole default registry, so a new solver is checked the
moment it registers: it must return a feasible schedule on small problems,
honour every stop criterion, report counts and objective values exactly, and
replay identically from the same seed.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from dispatchatlas.core import (
    ModelError,
    Problem,
    StopCriteria,
    StopReason,
    UnsupportedProblemError,
    check_schedule,
    evaluate,
)
from dispatchatlas.core.model import FEATURES
from dispatchatlas.references import bibliography
from dispatchatlas.solvers import default_registry
from tests.helpers import chain_problem, random_problem, small_job_shop

EXTRA_MODULES = {"exact": ("ortools", "pulp"), "gurobi": ("gurobipy",)}

REGISTRY = default_registry(plugins=False)


def _available(solver_id: str) -> bool:
    for extra in REGISTRY.info(solver_id).requires:
        for module in EXTRA_MODULES.get(extra, (extra,)):
            if importlib.util.find_spec(module) is None:
                return False
    return True


SOLVERS = [solver_id for solver_id in REGISTRY.ids() if _available(solver_id)]
ITERATIVE = [s for s in SOLVERS if REGISTRY.info(s).iterative]


def _problems() -> list[Problem]:
    return [
        chain_problem(),
        random_problem(1, tasks=8, max_modes=1, pools=0),
        random_problem(2, tasks=8, max_modes=2, pools=1),
        random_problem(3, tasks=8, max_modes=2, pools=0, network=True),
        random_problem(4, tasks=8, max_modes=1, pools=0, setups=True),
        small_job_shop(),
    ]


def _accepted(solver_id: str) -> list[Problem]:
    info = REGISTRY.info(solver_id)
    # The enumeration decodes every order, so it refuses problems above its max_tasks.
    largest = info.parameters.get("max_tasks", math.inf)
    return [problem for problem in _problems() if info.accepts(problem) and problem.size <= largest]


@pytest.mark.parametrize("solver_id", SOLVERS)
def test_returns_feasible_schedules_with_exact_objectives(solver_id: str) -> None:
    problems = _accepted(solver_id)
    assert problems, f"{solver_id} accepts none of the contract problems"
    for problem in problems:
        solver = REGISTRY.create(solver_id)
        result = solver.solve(problem, StopCriteria(max_evaluations=60, time_limit=20.0), seed=7)
        assert result.solver == solver_id
        assert result.problem == problem.id
        assert result.schedule is not None
        violations = [v for v in check_schedule(problem, result.schedule) if v.code != "deadline"]
        assert violations == [], f"{solver_id} on {problem.id}: {violations[:3]}"
        assert result.objectives == pytest.approx(evaluate(problem, result.schedule))
        assert result.objective == problem.objectives[0]
        assert result.evaluations >= 0
        assert result.elapsed >= 0.0
        if result.trace:
            assert result.trace[-1].value == pytest.approx(result.value)


@pytest.mark.parametrize("solver_id", ITERATIVE)
def test_honours_the_evaluation_limit(solver_id: str) -> None:
    problem = _accepted(solver_id)[0]
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=37), seed=3)
    assert 1 <= result.evaluations <= 37
    assert result.stop_reason in (StopReason.EVALUATION_LIMIT, StopReason.COMPLETED)


@pytest.mark.parametrize("solver_id", ITERATIVE)
def test_honours_the_iteration_limit(solver_id: str) -> None:
    problem = _accepted(solver_id)[0]
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(max_iterations=3), seed=3)
    assert result.iterations <= 3


@pytest.mark.parametrize("solver_id", ITERATIVE)
def test_honours_the_time_limit(solver_id: str) -> None:
    info = REGISTRY.info(solver_id)
    problem = random_problem(5, tasks=40, max_modes=2, pools=0)
    if not info.accepts(problem):
        problem = _accepted(solver_id)[0]
    result = REGISTRY.create(solver_id).solve(problem, StopCriteria(time_limit=0.3), seed=5)
    assert result.elapsed <= 1.5, f"{solver_id} ran {result.elapsed:.2f}s against 0.3s"


@pytest.mark.parametrize("solver_id", ITERATIVE)
def test_stops_when_the_target_is_reached(solver_id: str) -> None:
    problem = chain_problem()
    if not REGISTRY.info(solver_id).accepts(problem):
        pytest.skip("solver does not accept the chain problem")
    result = REGISTRY.create(solver_id).solve(
        problem, StopCriteria(max_evaluations=500, target=1e9), seed=1
    )
    assert result.stop_reason in (StopReason.TARGET_REACHED, StopReason.COMPLETED)


@pytest.mark.parametrize("solver_id", SOLVERS)
def test_replays_identically_from_a_seed(solver_id: str) -> None:
    problem = _accepted(solver_id)[0]
    stop = StopCriteria(max_evaluations=40)
    first = REGISTRY.create(solver_id).solve(problem, stop, seed=11)
    second = REGISTRY.create(solver_id).solve(problem, stop, seed=11)
    assert first.objectives == second.objectives
    assert first.schedule == second.schedule
    assert first.evaluations == second.evaluations


@pytest.mark.parametrize("solver_id", SOLVERS)
def test_metadata_is_complete(solver_id: str) -> None:
    info = REGISTRY.info(solver_id)
    assert info.summary.endswith(".")
    assert info.references, f"{solver_id} cites no source"
    known = bibliography()
    for source in info.references:
        cited = source in known or source.startswith(("doi:", "https://"))
        assert cited, f"{solver_id} cites unknown source {source!r}"
    assert set(info.supports) <= set(FEATURES)


@pytest.mark.parametrize("solver_id", SOLVERS)
def test_refuses_features_it_does_not_model(solver_id: str) -> None:
    info = REGISTRY.info(solver_id)
    candidates = [
        random_problem(6, tasks=6, max_modes=1, pools=1),
        random_problem(7, tasks=6, max_modes=3, pools=0),
        random_problem(8, tasks=6, max_modes=1, pools=0, setups=True),
        random_problem(9, tasks=6, max_modes=1, pools=0, network=True),
        random_problem(10, tasks=6, max_modes=1, pools=0, deadlines=True),
    ]
    for problem in candidates:
        if info.accepts(problem):
            continue
        with pytest.raises((UnsupportedProblemError, ModelError)):
            REGISTRY.create(solver_id).solve(problem, StopCriteria(max_evaluations=10), seed=0)
