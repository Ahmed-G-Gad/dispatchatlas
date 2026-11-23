"""The contract check for job-shop solvers, for use before their package registers them.

``tests/solvers/test_contract.py`` runs every registered solver through the
contract. A job-shop solver joins the registry with the others of its package,
so its own tests call :func:`check_contract`, which runs one solver class under a
small evaluation limit and checks what the contract demands of the result.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from dispatchatlas.core import (
    Problem,
    Solver,
    SolveResult,
    StopCriteria,
    StopReason,
    check_schedule,
    evaluate,
)


def check_contract(
    solver_cls: Callable[..., Solver],
    problem: Problem,
    *,
    evaluations: int = 40,
    seed: int = 7,
    **parameters: Any,
) -> SolveResult:
    """Solve ``problem`` under an evaluation limit and check the result against the contract.

    The solver must accept the problem and return a schedule without any
    violation but a missed deadline, whose objective values equal those of
    :func:`~dispatchatlas.core.objectives.evaluate`. An iterative solver must
    stop on the evaluation limit having made exactly that many evaluations. A
    solver that is not iterative must complete within the limit.

    Args:
        solver_cls: The solver class, or any factory of a solver.
        problem: The problem to solve.
        evaluations: The evaluation limit.
        seed: The seed of the run.
        **parameters: Parameters of the solver.

    Returns:
        The result, for further checks.
    """
    solver = solver_cls(**parameters)
    info = solver.info
    assert info.accepts(problem), f"{info.id} does not accept {problem.id}"
    result = solver.solve(problem, StopCriteria(max_evaluations=evaluations), seed=seed)
    assert result.solver == info.id
    assert result.problem == problem.id
    assert result.objective == problem.objectives[0]
    assert result.schedule is not None, f"{info.id} returned no schedule for {problem.id}"
    violations = [v for v in check_schedule(problem, result.schedule) if v.code != "deadline"]
    assert violations == [], f"{info.id} on {problem.id}: {violations[:3]}"
    assert result.objectives == pytest.approx(evaluate(problem, result.schedule))
    if result.trace:
        assert result.trace[-1].value == pytest.approx(result.value)
    if info.iterative:
        assert result.stop_reason is StopReason.EVALUATION_LIMIT, result.stop_reason
        assert result.evaluations == evaluations
    else:
        assert result.stop_reason is StopReason.COMPLETED, result.stop_reason
        assert 1 <= result.evaluations <= evaluations
    return result
