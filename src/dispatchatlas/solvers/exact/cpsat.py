"""The ``cp-sat`` solver: an exact interval model solved with OR-Tools CP-SAT.

The model itself lives in :mod:`dispatchatlas.solvers.exact._cpsat`, which is
imported only when a solve starts, so registering this solver needs no
third-party package.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.errors import MissingDependencyError
from dispatchatlas.core.model import FEATURES, Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.exact.common import (
    Clock,
    primary_objective,
    require_bounded,
    require_features,
)

__all__ = ["OBJECTIVES", "CpSat"]

OBJECTIVES = frozenset(
    {
        "makespan",
        "total_completion_time",
        "total_flow_time",
        "mean_response_time",
        "total_tardiness",
        "max_lateness",
        "tardy_tasks",
        "energy",
        "cost",
    }
)
"""Objectives the ``cp-sat`` solver optimises exactly. Others are refused."""


@dataclass(frozen=True)
class CpSat:
    """Prove optimal schedules with the OR-Tools CP-SAT solver.

    Each task gets one optional interval per mode, exactly one present, and
    the resources, dependencies, setups, and permutation constraint become
    CP-SAT constraints (see
    :class:`~dispatchatlas.solvers.exact._cpsat.IntervalModel`). The solver
    models every problem feature and optimises the problem's primary objective
    when it is one of :data:`OBJECTIVES`; any other objective is refused.

    Time is stated exactly: every duration, release, lag, deadline, due date,
    transfer time, and setup time is multiplied by the smallest power of ten,
    up to ``10**6``, that makes all of them integers. Weights, demands, and
    energy or cost coefficients are scaled the same way. A problem that needs
    more decimals is refused rather than rounded.

    Continuum placement is modelled as it is posed: a task's modes are its
    candidate nodes, and a node of capacity above one is a cumulative
    resource. A dependency that carries data waits for the transfer time of
    the pair of chosen nodes, zero when both tasks run on one node. The model
    states each distinct transfer time above the smallest as a conditional
    precedence enforced when both mode literals hold, so co-location needs no
    extra variable. The network has no contention: links carry any number of
    transfers at once, as the schedule checker assumes. A mode without a
    location exchanges its data at no delay; the ``unlocated_transfers`` entry of the result's
    ``extras`` counts the dependencies this affects, ``transfer_pairs`` the
    conditional precedences, and ``model_variables`` and ``model_constraints``
    the size of the model. ``mean_response_time`` is the mean over the
    groups, the workflows, of the completion of the group less its earliest
    release.

    Statuses are CP-SAT's own proofs. ``OPTIMAL`` means CP-SAT proved the
    schedule optimal, ``FEASIBLE`` that it stopped earlier, with ``bound``
    holding its proven lower bound, ``INFEASIBLE`` that it proved no schedule
    exists, and ``UNKNOWN`` that it stopped before finding a schedule. The
    published schedule is CP-SAT's own solution, checked with
    :func:`~dispatchatlas.core.validation.check_schedule`, and its objective
    value must equal the model's.

    The ``bounds`` of the result record how the proven bound rose. Where the
    installed OR-Tools offers a bound callback, they hold every improvement
    CP-SAT reports, between solutions too; otherwise they hold the bound CP-SAT
    has with each solution. Either way the last bound is the ``bound`` of the
    result, and the entry ``bound_callback`` of its ``extras`` says which kind
    the run recorded. A result without a bound, such as a proof of
    infeasibility, has no bounds over time.

    Stop criteria: the time limit covers model building and search. Each
    improving solution CP-SAT reports counts as one evaluation and one
    iteration, so evaluation and iteration limits stop the search after that
    many solutions; the search between two solutions is not bounded, so pair
    them with a time limit on hard problems. A target stops the search at the
    first solution that reaches it. Patience has no effect, because every
    reported solution improves on the previous one. Without a time limit the
    search ends only on a proof or another criterion.

    With one worker (the default) and no time limit, runs are deterministic.
    With several workers, a few more solutions may arrive, and be counted,
    while the search stops. The run seed is not used.

    Attributes:
        threads: CP-SAT workers. More than one runs CP-SAT's parallel
            portfolio, which is faster but not deterministic.
        relative_gap: Relative gap between solution and bound at which CP-SAT
            may stop. The default 0 asks for a proof. A run that stops at a
            positive gap reports ``FEASIBLE`` unless its bound meets the value.
        log: Send CP-SAT's search log to the ``dispatchatlas`` logger at
            level INFO.
    """

    threads: int = 1
    relative_gap: float = 0.0
    log: bool = False

    info: ClassVar[SolverInfo] = SolverInfo(
        id="cp-sat",
        name="CP-SAT interval model",
        family="exact",
        summary=(
            "States each task as optional intervals, one per mode with exactly one present, "
            "under no-overlap, cumulative, precedence, and circuit constraints, and proves "
            "optimality with the OR-Tools CP-SAT solver, which combines clause learning "
            "with linear relaxations."
        ),
        references=("perron2026ortools", "perron2023cpsatlp"),
        stochastic=False,
        iterative=True,
        supports=frozenset(FEATURES),
        requires=("exact",),
    )

    def __post_init__(self) -> None:
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError(f"threads must be a positive integer, got {self.threads!r}")
        gap = self.relative_gap
        if isinstance(gap, bool) or not isinstance(gap, (int, float)) or math.isnan(gap):
            raise ValueError(f"relative_gap must be a number, got {gap!r}")
        if not 0.0 <= gap <= 1.0:
            raise ValueError(f"relative_gap must lie between 0 and 1, got {gap!r}")
        object.__setattr__(self, "relative_gap", float(gap))
        if not isinstance(self.log, bool):
            raise ValueError(f"log must be true or false, got {self.log!r}")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Build and solve the CP-SAT model of ``problem``.

        Args:
            problem: The problem.
            stop: Stop criteria; at least one of a time, evaluation, or
                iteration limit.
            seed: Ignored; the search is deterministic with one worker.

        Returns:
            The result with CP-SAT's schedule, status, bound, and bounds over time.

        Raises:
            UnsupportedProblemError: If the objective is not modelled, or a
                value cannot be scaled to an integer exactly.
            SolverError: If the stop criteria are unbounded.
            MissingDependencyError: If OR-Tools is not installed.
        """
        del seed
        require_features(self.info, problem)
        require_bounded(self.info, stop)
        objective = primary_objective(problem, OBJECTIVES, self.info.id)
        _load_model()
        from dispatchatlas.solvers.exact import _cpsat  # noqa: PLC0415 - loads OR-Tools

        clock = Clock(stop.time_limit)
        return _cpsat.solve(
            problem,
            stop,
            self.info,
            objective,
            clock,
            threads=self.threads,
            relative_gap=self.relative_gap,
            log=self.log,
        )

    def warm_up(self) -> None:
        """Import OR-Tools and the model, which the first call in a process pays for.

        Raises:
            MissingDependencyError: If OR-Tools is not installed.
        """
        _load_model()


def _load_model() -> None:
    """Import the module of the model, which loads OR-Tools, or explain what is missing."""
    try:
        importlib.import_module("dispatchatlas.solvers.exact._cpsat")
    except ModuleNotFoundError as error:
        if (error.name or "").partition(".")[0] != "ortools":
            raise
        raise MissingDependencyError("ortools", "exact", "The cp-sat solver") from error
