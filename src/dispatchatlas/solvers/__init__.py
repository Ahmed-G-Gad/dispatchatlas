"""Solvers, the solver registry, and the shared search machinery.

Look solvers up by id through :func:`default_registry`, or call :func:`solve`
directly::

    from dispatchatlas.solvers import solve
    result = solve(problem, "simulated-annealing", time_limit=5.0, seed=1)

Building blocks for new solvers: :class:`~dispatchatlas.solvers.scheduling.Decoder`
(orders to schedules), :class:`~dispatchatlas.solvers.search.Search` (budget,
incumbent, trace), :mod:`~dispatchatlas.solvers.operators`, and
:mod:`~dispatchatlas.solvers.encoding`.
"""

from __future__ import annotations

from typing import Any

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, StopCriteria
from dispatchatlas.solvers.evaluation import Evaluator, Score
from dispatchatlas.solvers.registry import (
    PLUGIN_GROUP,
    SolverRegistry,
    default_registry,
    load_plugins,
)
from dispatchatlas.solvers.scheduling import Builder, Decoded, Decoder, precedence_feasible
from dispatchatlas.solvers.search import Candidate, Exhausted, Search

__all__ = [
    "PLUGIN_GROUP",
    "Builder",
    "Candidate",
    "Decoded",
    "Decoder",
    "Evaluator",
    "Exhausted",
    "Score",
    "Search",
    "SolverRegistry",
    "default_registry",
    "load_plugins",
    "precedence_feasible",
    "solve",
]


def solve(
    problem: Problem,
    solver: str,
    *,
    time_limit: float | None = None,
    max_evaluations: int | None = None,
    max_iterations: int | None = None,
    target: float | None = None,
    patience: int | None = None,
    seed: int = 0,
    **parameters: Any,
) -> SolveResult:
    """Solve a problem with a registered solver.

    Args:
        problem: The problem.
        solver: Registered solver id, for example ``"heft"``.
        time_limit: Wall-clock limit in seconds.
        max_evaluations: Limit on evaluated schedules.
        max_iterations: Limit on main-loop iterations.
        target: Stop once the primary objective reaches this value.
        patience: Stop after this many iterations without improvement.
        seed: Seed of the run.
        **parameters: Solver parameters.

    Returns:
        The solver's result.
    """
    stop = StopCriteria(time_limit, max_evaluations, max_iterations, target, patience)
    return default_registry().create(solver, **parameters).solve(problem, stop, seed)
