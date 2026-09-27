"""Which solvers run on which problems, and what to keep in mind when they do.

A solver declares the problem features it models in
:attr:`~dispatchatlas.core.solver.SolverInfo.supports`. A campaign never pairs
a solver with a problem that has a feature outside that set: the cell is
excluded and the plan records the exclusion with the missing features, so a
comparison that is absent is always explained.

Caveats mark pairings that run but need care when results are compared:

- a single-objective solver on a problem with several objectives optimises
  the primary objective only; the others describe its schedule;
- on problems with more than three objectives most solutions become mutually
  non-dominated, so a solver whose selection relies on Pareto dominance loses
  selection pressure (Deb and Jain 2014).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from dispatchatlas.core.errors import SerializationError
from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolverInfo

__all__ = ["MANY_OBJECTIVES", "Caveat", "Compatibility", "Exclusion", "assess"]

MANY_OBJECTIVES = 4
"""Objective count from which dominance-based selection is flagged (Deb and Jain 2014)."""

_SINGLE_OBJECTIVE = (
    "optimises only the primary objective of problems with several objectives; the other "
    "values describe its schedule"
)
_MANY_OBJECTIVES = (
    f"meets problems with {MANY_OBJECTIVES} or more objectives, where most solutions are "
    "mutually non-dominated and Pareto-dominance selection loses its pressure"
)


@dataclass(frozen=True)
class Exclusion:
    """A problem and solver that a campaign does not pair.

    Attributes:
        problem: Problem id.
        solver: Solver name.
        features: Features of the problem the solver does not model, sorted.
    """

    problem: str
    solver: str
    features: tuple[str, ...]

    @property
    def reason(self) -> str:
        """Why the cell is excluded, as a sentence fragment."""
        return f"{self.solver} does not model {', '.join(self.features)} (problem {self.problem})"

    def to_dict(self) -> dict[str, Any]:
        """Return the exclusion as a JSON-ready dictionary."""
        return {
            "problem": self.problem,
            "solver": self.solver,
            "features": list(self.features),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, document: object) -> Exclusion:
        """Build an exclusion from :meth:`to_dict` output.

        Raises:
            SerializationError: If the document is malformed.
        """
        if not isinstance(document, dict):
            raise SerializationError("an exclusion must be a JSON object")
        problem, solver, features = (document.get(key) for key in ("problem", "solver", "features"))
        if (
            not isinstance(problem, str)
            or not isinstance(solver, str)
            or not isinstance(features, list)
            or not all(isinstance(item, str) for item in features)
        ):
            raise SerializationError("an exclusion needs a problem, a solver, and features")
        return cls(problem, solver, tuple(features))


@dataclass(frozen=True)
class Caveat:
    """A pairing that runs but needs care in the analysis.

    Attributes:
        solver: Solver name.
        problems: Ids of the problems concerned.
        message: What to keep in mind.
        reference: Bibliography key of the source, if any.
    """

    solver: str
    problems: tuple[str, ...]
    message: str
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the caveat as a JSON-ready dictionary."""
        return {
            "solver": self.solver,
            "problems": list(self.problems),
            "message": self.message,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, document: object) -> Caveat:
        """Build a caveat from :meth:`to_dict` output.

        Raises:
            SerializationError: If the document is malformed.
        """
        if not isinstance(document, dict):
            raise SerializationError("a caveat must be a JSON object")
        solver, problems, message = (document.get(key) for key in ("solver", "problems", "message"))
        reference = document.get("reference")
        if (
            not isinstance(solver, str)
            or not isinstance(message, str)
            or not isinstance(problems, list)
            or not all(isinstance(item, str) for item in problems)
            or not (reference is None or isinstance(reference, str))
        ):
            raise SerializationError("a caveat needs a solver, problems, and a message")
        return cls(solver, tuple(problems), message, reference)


@dataclass(frozen=True)
class Compatibility:
    """The exclusions and caveats of a set of problems and solvers.

    Attributes:
        exclusions: Pairs that do not run, in problem order, then solver order.
        caveats: Pairings that run but need care, in solver order.
    """

    exclusions: tuple[Exclusion, ...] = ()
    caveats: tuple[Caveat, ...] = ()

    @cached_property
    def _pairs(self) -> frozenset[tuple[str, str]]:
        return frozenset((item.problem, item.solver) for item in self.exclusions)

    def excludes(self, problem: str, solver: str) -> bool:
        """Return whether the pair of a problem id and a solver name is excluded."""
        return (problem, solver) in self._pairs

    def to_dict(self) -> dict[str, Any]:
        """Return the assessment as a JSON-ready dictionary."""
        return {
            "exclusions": [item.to_dict() for item in self.exclusions],
            "caveats": [item.to_dict() for item in self.caveats],
        }


def assess(problems: Sequence[Problem], solvers: Sequence[tuple[str, SolverInfo]]) -> Compatibility:
    """Decide which problem and solver pairs run, and which need caveats.

    Args:
        problems: The problems, with the objectives they will be solved for.
        solvers: Solver names with the metadata of the registered solver.

    Returns:
        The exclusions and caveats.
    """
    exclusions: list[Exclusion] = []
    grouped: dict[tuple[str, str, str | None], list[str]] = {}
    for problem in problems:
        count = len(problem.objectives)
        for name, info in solvers:
            missing = info.unsupported(problem)
            if missing:
                exclusions.append(Exclusion(problem.id, name, tuple(sorted(missing))))
                continue
            if not info.multi_objective and count > 1:
                grouped.setdefault((name, _SINGLE_OBJECTIVE, None), []).append(problem.id)
            if info.multi_objective and count >= MANY_OBJECTIVES:
                key = (name, _MANY_OBJECTIVES, "deb2014evolutionary")
                grouped.setdefault(key, []).append(problem.id)
    order = {name: index for index, (name, _) in enumerate(solvers)}
    caveats = tuple(
        Caveat(solver, tuple(ids), message, reference)
        for (solver, message, reference), ids in sorted(
            grouped.items(), key=lambda item: order[item[0][0]]
        )
    )
    return Compatibility(tuple(exclusions), caveats)
