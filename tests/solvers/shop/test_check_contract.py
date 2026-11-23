"""The contract helper for job-shop solvers passes sound solvers and fails unsound ones."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import pytest

from dispatchatlas.core import (
    Assignment,
    Problem,
    Schedule,
    SolveResult,
    SolverInfo,
    StopCriteria,
    StopReason,
)
from dispatchatlas.solvers.rules import InputOrder, RandomSearch
from tests.helpers import chain_problem, small_job_shop
from tests.solvers.shop.contract import check_contract


@dataclass(frozen=True)
class ShortRun:
    """Stops after five evaluations, whatever limit it is given."""

    info: ClassVar[SolverInfo] = RandomSearch.info

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return RandomSearch().solve(problem, StopCriteria(max_evaluations=5), seed)


@dataclass(frozen=True)
class Misnamed:
    """Reports another solver's id in its result."""

    info: ClassVar[SolverInfo] = RandomSearch.info

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return replace(RandomSearch().solve(problem, stop, seed), solver="another-solver")


@dataclass(frozen=True)
class Overlapping:
    """Reports a schedule that starts every task at time zero."""

    info: ClassVar[SolverInfo] = RandomSearch.info

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        result = RandomSearch().solve(problem, stop, seed)
        assert result.schedule is not None
        flat = Schedule.of(
            Assignment(item.task, item.mode, 0.0, item.duration)
            for item in result.schedule.assignments
        )
        return replace(result, schedule=flat)


@dataclass(frozen=True)
class Narrow:
    """Models no feature, so it accepts no problem with a precedence."""

    info: ClassVar[SolverInfo] = replace(RandomSearch.info, id="narrow", supports=frozenset())

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        raise AssertionError("the helper must not solve a problem the solver does not accept")


def test_an_iterative_solver_passes_when_it_stops_on_the_evaluation_limit() -> None:
    problem = small_job_shop()
    result = check_contract(RandomSearch, problem, evaluations=25, seed=3)
    assert (result.evaluations, result.stop_reason) == (25, StopReason.EVALUATION_LIMIT)
    assert result.problem == problem.id
    again = check_contract(RandomSearch, problem, evaluations=25, seed=3)
    assert again.schedule == result.schedule


def test_a_solver_that_is_not_iterative_passes_when_it_completes() -> None:
    result = check_contract(InputOrder, small_job_shop())
    assert result.stop_reason is StopReason.COMPLETED
    assert result.evaluations == 1


def test_parameters_reach_the_solver() -> None:
    with_modes = check_contract(RandomSearch, chain_problem(), random_modes=True)
    assert with_modes.evaluations == 40
    with pytest.raises(TypeError):
        check_contract(RandomSearch, chain_problem(), no_such_parameter=1)


def test_a_run_that_makes_fewer_evaluations_than_the_limit_fails() -> None:
    with pytest.raises(AssertionError, match="5 == 40"):
        check_contract(ShortRun, small_job_shop())
    check_contract(ShortRun, small_job_shop(), evaluations=5)


def test_a_result_under_another_solver_id_fails() -> None:
    with pytest.raises(AssertionError, match="another-solver"):
        check_contract(Misnamed, small_job_shop())


def test_an_infeasible_schedule_fails() -> None:
    with pytest.raises(AssertionError, match=r"Violation.*precedence"):
        check_contract(Overlapping, small_job_shop())


def test_a_problem_the_solver_does_not_accept_fails_before_it_runs() -> None:
    with pytest.raises(AssertionError, match="narrow does not accept small-job-shop"):
        check_contract(Narrow, small_job_shop())
