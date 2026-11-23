"""Costs: seconds per evaluation measured on run records, and what they project.

A :class:`CostModel` holds one number for each group of runs: the mean, over
the runs of the group, of a run's seconds divided by its evaluations, and the
number of runs behind the mean (:class:`Cost`). The caller says what a group
is with a function from a solver name and a problem id to a hashable key, for
example the solver's family, representation, and builder together with the
problem's size class. The model fits no curve and extrapolates to no other
group: it projects a run only for a group that has runs, and refuses any other.

:func:`first_fitting` takes candidate configurations in the caller's order of
preference and returns the first whose projected cost, times a factor, fits a
limit. :func:`largest_budget` is the case of budget levels, tried from the
largest down. Both take the projection as a function, so the same rule serves
a plan priced from a model (:func:`~dispatchatlas.campaigns.estimation.estimate`
projects a whole campaign) or from any other cost.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from dispatchatlas.campaigns.records import RunRecord
from dispatchatlas.core.errors import ConfigError

__all__ = ["Cost", "CostModel", "Grouping", "first_fitting", "largest_budget"]

type Grouping = Callable[[str, str], Hashable]
"""Maps the name of a solver and the id of a problem to the group of runs they belong to."""


@dataclass(frozen=True)
class Cost:
    """Seconds per evaluation of one group of runs.

    Attributes:
        seconds_per_evaluation: The mean, over the runs of the group, of a
            run's seconds divided by its evaluations.
        runs: Runs the mean is taken over.
    """

    seconds_per_evaluation: float
    runs: int


@dataclass(frozen=True)
class CostModel:
    """The measured cost of each group of runs, and the function that groups runs.

    Build one from run records with :meth:`fit`.

    Attributes:
        groups: The cost of each group that has runs, by group key, in the
            order the records first named them.
        key: Maps the name of a solver and the id of a problem to a group key.
            It groups the records the model was fitted on and the runs it
            projects.
    """

    groups: Mapping[Hashable, Cost] = field(hash=False)
    key: Grouping = field(compare=False, hash=False)

    @classmethod
    def fit(
        cls,
        records: Iterable[RunRecord],
        key: Grouping,
        *,
        seconds: Callable[[RunRecord], float] | None = None,
    ) -> CostModel:
        """Measure the seconds per evaluation of each group of run records.

        Args:
            records: Records of completed runs. Each must have evaluated at
                least one schedule.
            key: Maps the solver name and the problem id of a record to its
                group.
            seconds: The seconds a record is charged. ``None`` charges the
                wall seconds of the attempt that completed the run
                (:attr:`~dispatchatlas.campaigns.records.Provenance.wall_seconds`).
                A function can charge less, for example the wall seconds
                without the time a solver recorded in its extras for work the
                cost should not include.

        Returns:
            The model.

        Raises:
            ConfigError: If there is no record, a record evaluated no
                schedule, or the seconds charged to a record are not a finite
                number of at least zero.
        """
        measured: dict[Hashable, list[float]] = {}
        for record in records:
            where = f"run {record.run_id} ({record.solver.name} on {record.problem.id})"
            evaluations = record.outcome.evaluations
            if evaluations < 1:
                raise ConfigError(
                    f"{where} evaluated no schedule, so it has no cost per evaluation"
                )
            spent = record.provenance.wall_seconds if seconds is None else seconds(record)
            if not (math.isfinite(spent) and spent >= 0.0):
                raise ConfigError(
                    f"{where} is charged {spent!r} seconds, which is not a finite number of at "
                    "least zero"
                )
            group = key(record.solver.name, record.problem.id)
            measured.setdefault(group, []).append(spent / evaluations)
        if not measured:
            raise ConfigError("a cost model needs at least one run record")
        groups = {
            group: Cost(math.fsum(values) / len(values), len(values))
            for group, values in measured.items()
        }
        return cls(MappingProxyType(groups), key)

    def cost(self, solver: str, problem: str) -> Cost:
        """Return the measured cost of the group a solver and a problem belong to.

        Args:
            solver: Name of the solver.
            problem: Id of the problem.

        Raises:
            ConfigError: If no run of the group was measured.
        """
        group = self.key(solver, problem)
        try:
            return self.groups[group]
        except KeyError:
            raise ConfigError(
                f"no run of group {group!r} was measured, so the cost of solver {solver!r} on "
                f"problem {problem!r} is unknown"
            ) from None

    def seconds(self, solver: str, problem: str, evaluations: int) -> float:
        """Project the seconds a run takes: its evaluations at the measured cost of its group.

        Args:
            solver: Name of the solver.
            problem: Id of the problem.
            evaluations: Evaluations of the run.

        Raises:
            ConfigError: If no run of the group was measured, or
                ``evaluations`` is not an integer of at least zero.
        """
        if isinstance(evaluations, bool) or not isinstance(evaluations, int) or evaluations < 0:
            raise ConfigError(f"evaluations must be an integer of at least 0, got {evaluations!r}")
        return evaluations * self.cost(solver, problem).seconds_per_evaluation


def first_fitting[T](
    candidates: Iterable[T],
    project: Callable[[T], float],
    limit: float,
    factor: float = 1.0,
) -> T | None:
    """Return the first candidate whose projected cost, times a factor, is at most a limit.

    The candidates are tried in the order given, so the order states which
    the caller prefers. Nothing is sorted, and a candidate is projected only
    when every one before it exceeded the limit.

    Args:
        candidates: Candidate configurations, the preferred first.
        project: The cost of a candidate, in the unit of ``limit``.
        limit: The most the cost may amount to, a finite number above zero.
        factor: The multiple of the projected cost that is compared with the
            limit, for example the allowance for runs that are repeated. A
            finite number above zero.

    Returns:
        The first candidate with ``project(candidate) * factor <= limit``, or
        ``None`` when none fits.

    Raises:
        ConfigError: If ``limit`` or ``factor`` is not a finite number above
            zero, or a projection is not a finite number of at least zero.
    """
    for name, value in (("limit", limit), ("factor", factor)):
        if not (math.isfinite(value) and value > 0.0):
            raise ConfigError(f"{name} must be a finite number above 0, got {value!r}")
    for candidate in candidates:
        cost = project(candidate)
        if not (math.isfinite(cost) and cost >= 0.0):
            raise ConfigError(
                f"the projected cost of {candidate!r} is {cost!r}, which is not a finite number "
                "of at least zero"
            )
        if cost * factor <= limit:
            return candidate
    return None


def largest_budget[N: (int, float)](
    levels: Iterable[N],
    project: Callable[[N], float],
    limit: float,
    factor: float = 1.0,
) -> N | None:
    """Return the largest budget level whose projected cost, times a factor, is at most a limit.

    It is :func:`first_fitting` over the levels from the largest down.

    Args:
        levels: Budget levels, in any order.
        project: The cost of running at a level, in the unit of ``limit``.
        limit: The most the cost may amount to, a finite number above zero.
        factor: The multiple of the projected cost that is compared with the
            limit. A finite number above zero.

    Returns:
        The largest level with ``project(level) * factor <= limit``, or
        ``None`` when none fits.

    Raises:
        ConfigError: As :func:`first_fitting`.
    """
    return first_fitting(sorted(levels, reverse=True), project, limit, factor)
