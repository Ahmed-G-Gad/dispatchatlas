"""Exact solvers: they prove optimality or infeasibility, or say that they did not.

Three solvers share one contract:

- ``cp-sat`` (:class:`CpSat`) states the problem as an interval model and
  solves it with OR-Tools CP-SAT. It models every problem feature.
- ``milp`` (:class:`TimeIndexedMilp`) states the problem as a time-indexed
  0-1 program and solves it with CBC, HiGHS, Gurobi, or SCIP through PuLP.
- ``enumeration`` (:class:`Enumeration`) decodes every task order and mode
  vector with the shared schedule builder. It needs no third-party package and
  is the reference the other two are tested against on small problems.

What the contract guarantees:

- ``Status.OPTIMAL`` only when the backend proved optimality, or the
  enumeration completed. A run stopped by a limit reports ``FEASIBLE`` with the
  backend's proven lower bound in ``SolveResult.bound``, so ``SolveResult.gap``
  measures how far from proven the schedule is.
- ``Status.INFEASIBLE`` only when infeasibility was proven.
- The published schedule is the solver's own, checked with
  :func:`~dispatchatlas.core.validation.check_schedule`; the model-based solvers
  also check that its objective value equals the value the backend proved.
- A feature or objective a solver does not model exactly raises
  :class:`~dispatchatlas.core.errors.UnsupportedProblemError`; nothing is
  approximated or replaced by a neighbouring problem. Time values are scaled
  to integers exactly or refused.
- The time limit covers model building. Threads default to one and are
  recorded, with the backend's version and raw status, in
  ``SolveResult.extras``.
"""

from __future__ import annotations

from dispatchatlas.solvers.exact.cpsat import CpSat
from dispatchatlas.solvers.exact.enumeration import Enumeration
from dispatchatlas.solvers.exact.milp import TimeIndexedMilp
from dispatchatlas.solvers.registry import SolverRegistry

__all__ = ["CpSat", "Enumeration", "TimeIndexedMilp", "register"]


def register(registry: SolverRegistry) -> None:
    """Add the exact solvers to ``registry``."""
    for factory in (CpSat, TimeIndexedMilp, Enumeration):
        registry.register(factory)
