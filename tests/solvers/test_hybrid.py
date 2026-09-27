"""The restricted neighbourhood search: exact windows inside a heuristic search."""

import importlib.util
import sys

import pytest

from dispatchatlas.core import (
    ConfigError,
    MissingDependencyError,
    Problem,
    StopCriteria,
    StopReason,
)
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers import exact as exact_package
from dispatchatlas.solvers.evaluation import Evaluator
from dispatchatlas.solvers.hybrid import RestrictedNeighbourhoodSearch
from dispatchatlas.solvers.priorities import upward_ranks
from tests.helpers import chain_problem, random_problem
from tests.solvers.test_constructive import flow_shop, value_of, violations

REGISTRY = default_registry(plugins=False)

requires_ortools = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None, reason="needs the ortools package"
)


def first_schedule_value(problem: Problem) -> float:
    ranks = upward_ranks(problem)
    order = sorted(range(problem.size), key=lambda task: (-ranks[task], task))
    return Evaluator(problem).evaluate(order)[0].value


@pytest.mark.exact
@requires_ortools
@pytest.mark.parametrize("seed", range(3))
def test_a_window_over_every_task_proves_the_optimum(seed):
    problem = random_problem(seed, tasks=6, max_modes=2, pools=1)
    optimum = REGISTRY.create("cp-sat").solve(problem, StopCriteria(time_limit=60.0))
    result = RestrictedNeighbourhoodSearch(window=6).solve(
        problem, StopCriteria(max_evaluations=50), seed=seed
    )
    assert result.stop_reason is StopReason.COMPLETED
    assert result.evaluations == 2
    assert result.extras["exact_calls"] == 1
    assert result.value == pytest.approx(value_of(optimum))
    assert violations(problem, result) == []


@pytest.mark.exact
@requires_ortools
@pytest.mark.parametrize("seed", range(3))
def test_every_exact_call_costs_one_evaluation_and_never_worsens(seed):
    problem = random_problem(seed, tasks=14, max_modes=2, pools=1, network=seed == 2)
    result = RestrictedNeighbourhoodSearch(window=3).solve(
        problem, StopCriteria(max_evaluations=9), seed=seed
    )
    assert result.evaluations == 9
    assert result.iterations == result.extras["exact_calls"] == 8
    assert value_of(result) <= first_schedule_value(problem)
    assert violations(problem, result) == []


@pytest.mark.exact
@requires_ortools
def test_permutation_problems_free_whole_jobs():
    problem = flow_shop()
    result = RestrictedNeighbourhoodSearch(window=2).solve(
        problem, StopCriteria(max_evaluations=12), seed=0
    )
    assert violations(problem, result) == []
    assert result.value == 37.0


@pytest.mark.exact
@requires_ortools
def test_restricted_neighbourhood_search_beats_random_search_at_equal_evaluations():
    problem = random_problem(11, tasks=30, max_modes=3, pools=1)
    stop = StopCriteria(max_evaluations=20)
    baseline = REGISTRY.create("random-search").solve(problem, stop, seed=0)
    result = REGISTRY.create("restricted-neighbourhood-search").solve(problem, stop, seed=0)
    assert value_of(result) < value_of(baseline)


@pytest.mark.parametrize("parameters", [{"window": 0}, {"call_limit": 0.0}])
def test_restricted_neighbourhood_parameters_are_validated(parameters):
    with pytest.raises(ConfigError):
        REGISTRY.create("restricted-neighbourhood-search", **parameters)


def test_restricted_neighbourhood_search_reports_a_missing_ortools(monkeypatch):
    monkeypatch.delitem(sys.modules, "dispatchatlas.solvers.exact._cpsat", raising=False)
    monkeypatch.delattr(exact_package, "_cpsat", raising=False)
    for name in ("ortools", "ortools.sat", "ortools.sat.python", "ortools.sat.python.cp_model"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(MissingDependencyError, match="ortools"):
        RestrictedNeighbourhoodSearch().solve(chain_problem(), StopCriteria(max_evaluations=5))
