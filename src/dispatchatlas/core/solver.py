"""The solver contract: stop criteria, results, metadata, and the protocol.

A solver is any object with an :attr:`info` attribute and a
:meth:`~Solver.solve` method. The library's own solvers and third-party plugins
implement the same contract, so campaigns and analyses treat them alike.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dispatchatlas.core._frozen import FrozenMap
from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import FEATURES, Scalar

if TYPE_CHECKING:
    from dispatchatlas.core.model import Problem
    from dispatchatlas.core.schedule import Schedule

__all__ = [
    "SOLVER_FAMILIES",
    "FrontPoint",
    "SolveResult",
    "Solver",
    "SolverInfo",
    "Status",
    "StopCriteria",
    "StopReason",
    "TracePoint",
]

_SLUG = re.compile(r"^[a-z0-9]+(?:[-.][a-z0-9]+)*$")

SOLVER_FAMILIES: Mapping[str, str] = FrozenMap(
    {
        "dispatching-rule": "Builds one schedule from a fixed priority rule.",
        "list-scheduling": "Ranks tasks, then maps each to the resource that finishes it first.",
        "constructive": "Builds a schedule step by step with look-ahead or insertion.",
        "local-search": "Improves one solution through neighbourhood moves.",
        "population": "Evolves a population of solutions through variation and selection.",
        "swarm": "Moves a population of real-valued priority vectors by update rules.",
        "multi-objective": "Approximates the Pareto front of several objectives.",
        "exact": "Proves optimality or infeasibility with a mathematical model.",
        "matheuristic": "Improves a solution by solving restricted subproblems exactly.",
        "learning": "Learns a scheduling policy from experience gathered while solving.",
        "hyper-heuristic": "Searches over heuristics rather than over schedules.",
        "random-sampling": "Draws random solutions and keeps the best, as a baseline.",
    }
)
"""Standard solver families. Plugins may use other kebab-case family names."""


class StopReason(StrEnum):
    """Why a solver stopped."""

    COMPLETED = "completed"
    """The algorithm finished on its own, for example a constructive heuristic."""
    TIME_LIMIT = "time-limit"
    EVALUATION_LIMIT = "evaluation-limit"
    ITERATION_LIMIT = "iteration-limit"
    TARGET_REACHED = "target-reached"
    STAGNATION = "stagnation"
    """No improvement for ``patience`` iterations."""


class Status(StrEnum):
    """What is known about the returned schedule."""

    OPTIMAL = "optimal"
    """Feasible and proven optimal for the primary objective."""
    FEASIBLE = "feasible"
    """Feasible, optimality not proven."""
    VIOLATED = "violated"
    """A schedule was built but it misses a hard deadline."""
    INFEASIBLE = "infeasible"
    """Proven that no feasible schedule exists."""
    UNKNOWN = "unknown"
    """No schedule found and infeasibility not proven."""


@dataclass(frozen=True, slots=True)
class StopCriteria:
    """When an iterative solver must stop. The first criterion met ends the run.

    Attributes:
        time_limit: Wall-clock seconds.
        max_evaluations: Schedules built and scored.
        max_iterations: Main-loop iterations, as each solver defines them.
        target: Stop once the primary objective reaches this value or better.
        patience: Stop after this many iterations without improvement.
    """

    time_limit: float | None = None
    max_evaluations: int | None = None
    max_iterations: int | None = None
    target: float | None = None
    patience: int | None = None

    def __post_init__(self) -> None:
        if self.time_limit is not None:
            value = self.time_limit
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ModelError("time_limit must be a positive number of seconds")
            if math.isinf(value):
                raise ModelError("time_limit must be finite")
            object.__setattr__(self, "time_limit", float(value))
        for name in ("max_evaluations", "max_iterations", "patience"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ModelError(f"{name} must be a positive integer")
        if self.target is not None:
            value = self.target
            if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
                raise ModelError("target must be a number")
            object.__setattr__(self, "target", float(self.target))

    @property
    def bounded(self) -> bool:
        """Whether a time, evaluation, or iteration limit guarantees termination."""
        return (
            self.time_limit is not None
            or self.max_evaluations is not None
            or self.max_iterations is not None
        )

    def progress(
        self, evaluations: int, iterations: int, elapsed: float, *, deterministic: bool = True
    ) -> float:
        """Return the fraction of the binding budget a run has used, in [0, 1].

        The fraction is the largest of evaluations over the evaluation limit,
        iterations over the iteration limit, and elapsed seconds over the time
        limit, among the limits that are set, so it reaches 1 when the first of
        them ends the run. The target and the patience end runs at points no
        one can predict and play no part. Without a limit the fraction is 0.

        Args:
            evaluations: Evaluations so far.
            iterations: Iterations started so far.
            elapsed: Seconds since the run started.
            deterministic: Count the time limit only when neither an
                evaluation nor an iteration limit is set, the default. A run
                with one of those then follows the same schedule on every
                machine and replays exactly, though its schedule stops short
                of 1 when the time limit ends it first. ``False`` counts every
                set limit.
        """
        fractions: list[float] = []
        if self.max_evaluations is not None:
            fractions.append(evaluations / self.max_evaluations)
        if self.max_iterations is not None:
            fractions.append(iterations / self.max_iterations)
        if self.time_limit is not None and not (deterministic and fractions):
            fractions.append(elapsed / self.time_limit)
        return min(1.0, max(fractions, default=0.0))

    def to_dict(self) -> dict[str, Any]:
        """Return the criteria that are set, as a JSON-ready dictionary."""
        return {
            name: getattr(self, name)
            for name in ("time_limit", "max_evaluations", "max_iterations", "target", "patience")
            if getattr(self, name) is not None
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> StopCriteria:
        """Build criteria from a dictionary such as :meth:`to_dict` returns."""
        unknown = set(values) - {
            "time_limit",
            "max_evaluations",
            "max_iterations",
            "target",
            "patience",
        }
        if unknown:
            raise ModelError(f"unknown stop criteria: {sorted(unknown)}")
        return cls(**dict(values))


@dataclass(frozen=True, slots=True)
class TracePoint:
    """The best solution found so far, sampled when it improves.

    Attributes:
        elapsed: Seconds since the run started.
        evaluations: Schedules evaluated so far.
        iteration: Main-loop iteration.
        value: Primary objective of the best schedule so far.
        feasible: Whether that schedule meets every hard deadline.
    """

    elapsed: float
    evaluations: int
    iteration: int
    value: float
    feasible: bool = True


@dataclass(frozen=True, slots=True)
class FrontPoint:
    """One non-dominated solution of a multi-objective run.

    Attributes:
        values: Objective values in the problem's objective order.
        schedule: The schedule that achieves them.
    """

    values: tuple[float, ...]
    schedule: Schedule | None = None


@dataclass(frozen=True)
class SolveResult:
    """The outcome of one solver run.

    Attributes:
        solver: Id of the solver.
        problem: Id of the problem.
        objective: Name of the primary objective.
        status: What is known about the schedule.
        schedule: The best schedule found, or ``None``.
        objectives: Values of every problem objective for ``schedule``.
        stop_reason: Why the run ended.
        elapsed: Wall-clock seconds of the run.
        evaluations: Schedules evaluated.
        iterations: Main-loop iterations.
        bound: Proven bound on the primary objective, for exact solvers.
        trace: Improvements of the best solution over time.
        front: Non-dominated solutions, for multi-objective solvers.
        extras: Solver-specific facts, such as a backend's raw status.
    """

    solver: str
    problem: str
    objective: str
    status: Status
    schedule: Schedule | None
    objectives: Mapping[str, float]
    stop_reason: StopReason
    elapsed: float
    evaluations: int = 0
    iterations: int = 0
    bound: float | None = None
    trace: tuple[TracePoint, ...] = ()
    front: tuple[FrontPoint, ...] = ()
    extras: Mapping[str, Scalar] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        """Whether the schedule meets every constraint."""
        return self.status in (Status.OPTIMAL, Status.FEASIBLE)

    @property
    def value(self) -> float | None:
        """Primary objective value, or ``None`` without a schedule."""
        return self.objectives.get(self.objective)

    @property
    def gap(self) -> float | None:
        """Relative distance between the value and the bound, when both exist.

        Computed as ``|value - bound| / max(|value|, 1e-9)``, the convention of
        MILP solvers.
        """
        if self.bound is None or self.value is None:
            return None
        return abs(self.value - self.bound) / max(abs(self.value), 1e-9)


@dataclass(frozen=True)
class SolverInfo:
    """Metadata describing a solver.

    Attributes:
        id: Registry key in kebab case, for example ``"simulated-annealing"``.
        name: Human-readable name.
        family: A key of :data:`SOLVER_FAMILIES`, or another kebab-case family.
        summary: One sentence describing what the solver does, in mechanism
            terms.
        references: Bibliography keys (see :mod:`dispatchatlas.references`),
            DOIs, or URLs of the sources the solver implements.
        stochastic: Whether results depend on the seed.
        iterative: Whether the solver runs until a stop criterion fires.
            Iterative solvers require bounded :class:`StopCriteria`.
        multi_objective: Whether the solver optimises every problem objective
            and returns a Pareto front.
        supports: Problem features the solver models. See
            :data:`~dispatchatlas.core.model.FEATURES`.
        requires: Optional-dependency extras the solver needs, for example
            ``("exact",)``.
        parameters: Default values of the solver's parameters.
    """

    id: str
    name: str
    family: str
    summary: str
    references: tuple[str, ...] = ()
    stochastic: bool = True
    iterative: bool = True
    multi_objective: bool = False
    supports: frozenset[str] = frozenset(FEATURES)
    requires: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not _SLUG.match(self.id):
            raise ModelError(f"solver id {self.id!r} must be lowercase kebab case")
        if not _SLUG.match(self.family):
            raise ModelError(f"solver family {self.family!r} must be lowercase kebab case")
        if not self.name or not self.summary:
            raise ModelError(f"solver {self.id!r} needs a name and a summary")
        unknown = set(self.supports) - set(FEATURES)
        if unknown:
            raise ModelError(f"solver {self.id!r} declares unknown features {sorted(unknown)}")
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "supports", frozenset(self.supports))
        object.__setattr__(self, "requires", tuple(self.requires))
        object.__setattr__(self, "parameters", FrozenMap(self.parameters))

    def unsupported(self, problem: Problem) -> frozenset[str]:
        """Return the features of ``problem`` this solver does not model."""
        return problem.features - self.supports

    def accepts(self, problem: Problem) -> bool:
        """Return whether this solver models every feature of ``problem``."""
        return not self.unsupported(problem)


@runtime_checkable
class Solver(Protocol):
    """The interface every solver implements."""

    @property
    def info(self) -> SolverInfo:
        """Metadata of the solver."""
        ...

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Solve ``problem`` within ``stop`` using randomness derived from ``seed``."""
        ...
