"""Relative percentage deviation from reference values.

Scheduling benchmarks usually publish a best-known or optimal value per
instance, and results are compared on the scale-free relative percentage
deviation (RPD) from it (Vallada, Ruiz and Framinan 2015):

- ``RPD = 100 (value - reference) / reference`` for a minimised objective and
  ``100 (reference - value) / reference`` for a maximised one, so a positive
  RPD is always worse than the reference and a negative one improves on it;
- the average RPD (ARPD) of a solver is the mean over all of its runs on all
  problems together.

Only succeeded runs have a value to deviate from. Each row reports how many
of the planned runs that is, so an ARPD over few surviving runs is visible as
such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dispatchatlas.analysis._scope import resolve_scope
from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.dataset import Dataset
from dispatchatlas.analysis.stats.ranks import Sense, check_sense
from dispatchatlas.core.errors import ConfigError

__all__ = ["DeviationRow", "arpd", "relative_deviation"]


@dataclass(frozen=True, slots=True)
class DeviationRow:
    """The average relative percentage deviation of one solver.

    Attributes:
        solver: Solver name.
        problems: Problems with at least one succeeded run.
        runs: Succeeded runs, whose deviations are averaged.
        planned: Planned runs.
        arpd: Mean deviation over the succeeded runs, in percent, or
            ``None`` without one.
        rpd_min: Smallest deviation.
        rpd_max: Largest deviation.
    """

    solver: str
    problems: int
    runs: int
    planned: int
    arpd: float | None
    rpd_min: float | None
    rpd_max: float | None


def relative_deviation(value: float, reference: float, sense: Sense = "min") -> float:
    """Return the relative percentage deviation of a value from a reference.

    Args:
        value: The objective value.
        reference: The reference value, such as the best known, positive.
        sense: Whether the objective is minimised or maximised.

    Returns:
        The deviation in percent, positive when worse than the reference.

    Raises:
        ConfigError: If the reference is not positive or a value is not finite.
    """
    orientation = check_sense(sense)
    if not (math.isfinite(value) and math.isfinite(reference)):
        raise ConfigError("a relative deviation needs finite values")
    if reference <= 0.0:
        raise ConfigError(f"a relative deviation needs a positive reference, got {reference!r}")
    difference = value - reference if orientation == "min" else reference - value
    return 100.0 * difference / reference


def arpd(dataset: Dataset, config: AnalysisConfig) -> tuple[DeviationRow, ...]:
    """Return the average relative percentage deviation of every solver.

    Args:
        dataset: The campaign's runs.
        config: The analysis settings; ``references`` must give a value for
            every analysed problem.

    Returns:
        One row per analysed solver, in campaign order.

    Raises:
        ConfigError: If the configuration has no references or lacks one
            for an analysed problem, or the scope cannot be resolved.
    """
    if config.references is None:
        raise ConfigError("the relative percentage deviation needs reference values")
    scope = resolve_scope(dataset, config)
    missing = [
        problem.id for problem in scope.dataset.problems if problem.id not in config.references
    ]
    if missing:
        raise ConfigError("no reference value for problems: " + ", ".join(missing))
    rows = []
    for entry in scope.dataset.solvers:
        deviations: list[float] = []
        solved: set[str] = set()
        planned = 0
        for run in scope.dataset.runs:
            if run.solver != entry.name:
                continue
            planned += 1
            value = run.value(scope.objective)
            if not run.succeeded or value is None:
                continue
            deviations.append(
                relative_deviation(value, config.references[run.problem], scope.sense)
            )
            solved.add(run.problem)
        rows.append(
            DeviationRow(
                solver=entry.name,
                problems=len(solved),
                runs=len(deviations),
                planned=planned,
                arpd=math.fsum(deviations) / len(deviations) if deviations else None,
                rpd_min=min(deviations) if deviations else None,
                rpd_max=max(deviations) if deviations else None,
            )
        )
    return tuple(rows)
