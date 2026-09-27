"""Multi-objective quality indicators: hypervolume, IGD, and IGD+.

On every problem with at least two objectives, :func:`indicator_report` scores each
run's approximation set, the front a multi-objective solver records or the
single objective vector of any other solver's best schedule, then aggregates
the runs of each solver:

1. Objective vectors are put in minimisation form.
2. The **reference set** of the problem is the non-dominated union of every
   run's set, over all analysed solvers. Its per-objective minimum is the
   *ideal* point and its maximum the *nadir* point.
3. With ``normalize`` (the default), every objective is rescaled so that the
   ideal is 0 and the nadir 1; an objective on which the reference set does
   not vary is only shifted.
4. The **reference point** is shared by every solver on the problem. It is
   the configured point, if any, or it is derived from the ideal and nadir:
   the ``"margin"`` rule places it ``m`` spans beyond the nadir
   (``r = nadir + m (nadir - ideal)``, default ``m = 0.1``), the
   ``"ishibuchi"`` rule at ``r = ideal + (1 + 1/h)(nadir - ideal)``
   (Ishibuchi et al. 2018). On an objective without spread, both rules move
   ``|nadir| m``, or ``m`` when the nadir is 0, beyond it (``m = 1/h`` for
   the second rule).
5. Each run gets the **hypervolume** of its set, the volume it dominates up
   to the reference point (Zitzler and Thiele 1999), computed exactly by
   :func:`dispatchatlas.core.pareto.hypervolume`; the **IGD**, the mean
   Euclidean distance from each reference-set point to the nearest point of
   the run's set (Coello Coello and Reyes Sierra 2004); and the **IGD+**,
   the same mean with the distance counting only the objectives in which the
   run's point is worse (Ishibuchi et al. 2015). Hypervolume is to be
   maximised, IGD and IGD+ minimised.
6. A run that did not succeed has an empty set: hypervolume 0 and infinite
   IGD and IGD+. The configured statistic (mean or median) of the planned
   runs gives each solver's value on the problem, so failures count.

A reference point derived from the data moves when solvers are added or
removed, so hypervolumes are comparable only between results that share
their reference points; :func:`ensure_shared_reference_points` checks it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from dispatchatlas.analysis._scope import aggregate, narrow
from dispatchatlas.analysis.config import AnalysisConfig, ReferenceRule
from dispatchatlas.analysis.dataset import Dataset
from dispatchatlas.analysis.results import Limitation
from dispatchatlas.core.errors import ConfigError, ModelError
from dispatchatlas.core.objectives import Objective, get_objective
from dispatchatlas.core.pareto import hypervolume, non_dominated

__all__ = [
    "IndicatorReport",
    "IndicatorRow",
    "ReferencePoint",
    "ensure_shared_reference_points",
    "igd",
    "igd_plus",
    "indicator_report",
    "reference_point",
]

_SPAN_EPSILON = 1e-12
"""Objective spans at most this wide count as no spread."""

type Point = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    """The frame in which one problem's indicators are computed.

    Attributes:
        problem: Problem id.
        objectives: The objectives, in the problem's order.
        ideal: Per-objective minimum of the reference set, in minimisation form.
        nadir: Per-objective maximum of the reference set, in minimisation form.
        point: The hypervolume reference point, in objective units and
            minimisation form.
        rule: ``"fixed"`` for a configured point, else the deriving rule.
        parameter: The rule's margin or ``h``; ``None`` for a fixed point.
        normalized: Whether the indicators use normalised objectives.
    """

    problem: str
    objectives: tuple[str, ...]
    ideal: Point
    nadir: Point
    point: Point
    rule: Literal["fixed", "margin", "ishibuchi"]
    parameter: float | None
    normalized: bool


@dataclass(frozen=True, slots=True)
class IndicatorRow:
    """One solver's indicators on one problem, aggregated over its planned runs.

    Attributes:
        solver: Solver name.
        problem: Problem id.
        planned: Planned runs.
        succeeded: Runs with a feasible result.
        hypervolume: Hypervolume, higher is better.
        igd: Inverted generational distance, lower is better.
        igd_plus: IGD+, lower is better.
    """

    solver: str
    problem: str
    planned: int
    succeeded: int
    hypervolume: float
    igd: float
    igd_plus: float


@dataclass(frozen=True)
class IndicatorReport:
    """The indicators of every solver on every multi-objective problem.

    Attributes:
        protocol: The analysed protocol.
        statistic: How runs were aggregated.
        rows: One row per planned solver and problem, by problem then solver.
        reference_points: The frame of each problem.
        limitations: Problems that could not be scored, and why.
    """

    protocol: str
    statistic: str
    rows: tuple[IndicatorRow, ...]
    reference_points: tuple[ReferencePoint, ...]
    limitations: tuple[Limitation, ...]


def reference_point(
    points: Sequence[Sequence[float]], rule: ReferenceRule = "margin", parameter: float = 0.1
) -> Point:
    """Derive a hypervolume reference point from a set of minimisation vectors.

    Args:
        points: The reference front, one vector per point.
        rule: ``"margin"`` or ``"ishibuchi"``, see the module documentation.
        parameter: The margin (positive) or ``h`` (a positive integer).

    Returns:
        The reference point.

    Raises:
        ConfigError: If ``points`` is empty or ragged, or the parameter does
            not suit the rule.
    """
    vectors = _vectors(points)
    if not vectors:
        raise ConfigError("a reference point needs at least one point")
    if not (math.isfinite(parameter) and parameter > 0.0):
        raise ConfigError(f"the rule parameter must be positive, got {parameter!r}")
    if rule == "ishibuchi":
        if parameter != int(parameter):
            raise ConfigError(f"the ishibuchi rule needs an integer h >= 1, got {parameter!r}")
        factor, margin = 1.0 + 1.0 / parameter, 1.0 / parameter
    elif rule == "margin":
        factor, margin = 1.0 + parameter, parameter
    else:
        raise ConfigError(f"unknown reference-point rule {rule!r}")
    result = []
    for axis in range(len(vectors[0])):
        low = min(vector[axis] for vector in vectors)
        high = max(vector[axis] for vector in vectors)
        span = high - low
        if span > _SPAN_EPSILON:
            result.append(low + factor * span)
        else:
            result.append(high + (abs(high) * margin or margin))
    return tuple(result)


def igd(front: Sequence[Sequence[float]], reference_set: Sequence[Sequence[float]]) -> float:
    """Return the inverted generational distance of a front to a reference set.

    The mean over reference points of the Euclidean distance to the nearest
    front point; infinite for an empty front.

    Raises:
        ConfigError: If the reference set is empty or the vectors are ragged.
    """
    return _mean_distance(front, reference_set, plus=False)


def igd_plus(front: Sequence[Sequence[float]], reference_set: Sequence[Sequence[float]]) -> float:
    """Return the IGD+ of a front to a reference set, in minimisation form.

    Like :func:`igd`, but the distance from a reference point ``z`` to a front
    point ``a`` is ``sqrt(sum_k max(a_k - z_k, 0)^2)``: only the objectives
    in which ``a`` is worse count, which makes IGD+ weakly Pareto compliant.

    Raises:
        ConfigError: If the reference set is empty or the vectors are ragged.
    """
    return _mean_distance(front, reference_set, plus=True)


def indicator_report(dataset: Dataset, config: AnalysisConfig | None = None) -> IndicatorReport:
    """Return the indicators of every solver on every problem with several objectives.

    Args:
        dataset: The campaign's runs.
        config: The analysis settings: protocol, selection, aggregation
            statistic, reference points or rule, and normalisation.

    Returns:
        The rows and the reference point of each problem. Problems with one
        objective are skipped.

    Raises:
        ConfigError: If the scope cannot be resolved, an objective is not
            registered, or a configured reference point does not fit its
            problem.
    """
    settings = config if config is not None else AnalysisConfig()
    narrowed, protocol = narrow(dataset, settings)
    rows: list[IndicatorRow] = []
    frames: list[ReferencePoint] = []
    limitations: list[Limitation] = []
    for problem in narrowed.problems:
        names = tuple(problem.objectives)
        if len(names) < 2:
            continue
        orient = [_objective(name) for name in names]
        sets: dict[str, list[list[Point]]] = {}
        for entry in narrowed.solvers:
            cell = narrowed.cell(entry.name, problem.id, protocol)
            if cell:
                sets[entry.name] = [_minimised(run.points(names), orient) for run in cell]
        pooled = [point for runs in sets.values() for points in runs for point in points]
        if not pooled:
            limitations.append(
                Limitation(
                    "indicators",
                    f"No run succeeded on problem {problem.id!r}, so it has no reference front "
                    "and no indicator values.",
                )
            )
            continue
        reference = _front(pooled)
        frame, transform = _frame(problem.id, names, reference, settings)
        frames.append(frame)
        working_reference = [transform(point) for point in reference]
        working_point = transform(frame.point)
        for solver, approximations in sets.items():
            volumes: list[float] = []
            distances: list[float] = []
            plus: list[float] = []
            for points in approximations:
                shifted = [transform(point) for point in points]
                volumes.append(hypervolume(shifted, working_point) if shifted else 0.0)
                distances.append(igd(shifted, working_reference))
                plus.append(igd_plus(shifted, working_reference))
            rows.append(
                IndicatorRow(
                    solver=solver,
                    problem=problem.id,
                    planned=len(approximations),
                    succeeded=sum(1 for points in approximations if points),
                    hypervolume=aggregate(volumes, settings.statistic),
                    igd=aggregate(distances, settings.statistic),
                    igd_plus=aggregate(plus, settings.statistic),
                )
            )
    return IndicatorReport(
        protocol=protocol,
        statistic=settings.statistic,
        rows=tuple(rows),
        reference_points=tuple(frames),
        limitations=tuple(limitations),
    )


def ensure_shared_reference_points(reports: Sequence[IndicatorReport]) -> None:
    """Check that reports scoring the same problem used the same frame.

    Hypervolumes measured against different reference points, or IGD values
    computed on differently normalised objectives, are on different scales.

    Raises:
        ConfigError: If two reports score a problem against different
            reference points, rules, or normalisations.
    """
    seen: dict[str, ReferencePoint] = {}
    for report in reports:
        for frame in report.reference_points:
            first = seen.setdefault(frame.problem, frame)
            if (first.objectives, first.point, first.rule, first.parameter, first.normalized) != (
                frame.objectives,
                frame.point,
                frame.rule,
                frame.parameter,
                frame.normalized,
            ):
                raise ConfigError(
                    f"problem {frame.problem!r} is scored against different reference points; "
                    "fix them with reference_points to compare the results"
                )


def _frame(
    problem: str, names: tuple[str, ...], reference: Sequence[Point], config: AnalysisConfig
) -> tuple[ReferencePoint, _Transform]:
    """Return the problem's reference point and the map into the working objective space."""
    ideal = tuple(min(point[axis] for point in reference) for axis in range(len(names)))
    nadir = tuple(max(point[axis] for point in reference) for axis in range(len(names)))
    if config.normalize:
        scale = tuple(
            high - low if high - low > _SPAN_EPSILON else 1.0
            for low, high in zip(ideal, nadir, strict=True)
        )
        transform = _Transform(ideal, scale)
    else:
        transform = _Transform(tuple(0.0 for _ in names), tuple(1.0 for _ in names))
    fixed = None if config.reference_points is None else config.reference_points.get(problem)
    if fixed is not None:
        if len(fixed) != len(names):
            raise ConfigError(
                f"the reference point of {problem!r} has {len(fixed)} values for "
                f"{len(names)} objectives"
            )
        point: Point = tuple(fixed)
        rule: Literal["fixed", "margin", "ishibuchi"] = "fixed"
        parameter: float | None = None
    else:
        working = reference_point(
            [transform(point) for point in (ideal, nadir)],
            config.reference_rule,
            config.reference_parameter,
        )
        point = transform.inverse(working)
        rule, parameter = config.reference_rule, config.reference_parameter
    frame = ReferencePoint(problem, names, ideal, nadir, point, rule, parameter, config.normalize)
    return frame, transform


class _Transform:
    """The affine map ``(x - shift) / scale`` into the working objective space."""

    def __init__(self, shift: Point, scale: Point) -> None:
        self.shift = shift
        self.scale = scale

    def __call__(self, point: Sequence[float]) -> Point:
        return tuple(
            (value - low) / width
            for value, low, width in zip(point, self.shift, self.scale, strict=True)
        )

    def inverse(self, point: Sequence[float]) -> Point:
        return tuple(
            low + value * width
            for value, low, width in zip(point, self.shift, self.scale, strict=True)
        )


def _objective(name: str) -> Objective:
    try:
        return get_objective(name)
    except ModelError as error:
        raise ConfigError(f"{error}; the indicators need every objective's sense") from error


def _minimised(points: Sequence[Sequence[float]], orient: Sequence[Objective]) -> list[Point]:
    """Return the non-dominated points of a run in minimisation form."""
    return _front(
        [
            tuple(
                objective.minimized(value) for objective, value in zip(orient, point, strict=True)
            )
            for point in points
        ]
    )


def _front(points: Sequence[Point]) -> list[Point]:
    """Return the distinct non-dominated points, sorted."""
    distinct = sorted(set(points))
    return [distinct[index] for index in non_dominated(distinct)]


def _vectors(points: Sequence[Sequence[float]]) -> list[Point]:
    vectors = [tuple(float(value) for value in point) for point in points]
    if vectors and len({len(vector) for vector in vectors}) != 1:
        raise ConfigError("all points need the same number of objectives")
    for vector in vectors:
        if not all(math.isfinite(value) for value in vector):
            raise ConfigError("indicator points must be finite")
    return vectors


def _mean_distance(
    front: Sequence[Sequence[float]], reference_set: Sequence[Sequence[float]], *, plus: bool
) -> float:
    references = _vectors(reference_set)
    if not references:
        raise ConfigError("an IGD needs a non-empty reference set")
    points = _vectors(front)
    if not points:
        return math.inf
    if len(points[0]) != len(references[0]):
        raise ConfigError("the front and the reference set need the same number of objectives")
    total = 0.0
    for target in references:
        nearest = math.inf
        for point in points:
            if plus:
                squares = sum(max(a - z, 0.0) ** 2 for a, z in zip(point, target, strict=True))
            else:
                squares = sum((a - z) ** 2 for a, z in zip(point, target, strict=True))
            nearest = min(nearest, squares)
        total += math.sqrt(nearest)
    return total / len(references)
