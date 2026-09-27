"""Pareto fronts of bi-objective problems, as attainment surfaces.

A run attains an objective vector when one of its solutions is at least as
good in both objectives. With several replicates, a solver's front is a
summary attainment surface (Fonseca and Fleming 1996): the boundary of the
vectors attained in at least ``k`` of its planned runs. The median surface,
``k`` the smallest integer of at least half the runs, shows what a typical
run achieves; the best surface, ``k = 1``, is the non-dominated set of all
the solver's solutions. A failed run attains nothing, so failures pull the
median surface back without being dropped.

A multi-objective solver contributes its recorded front; any other solver
contributes the objective vector of its best schedule. Problems with exactly
two objectives get a figure each.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dispatchatlas.analysis.figures.axes import numeric_axis
from dispatchatlas.analysis.figures.chart import Series, label_for, placeholder, xy_chart
from dispatchatlas.analysis.figures.drawing import Figure, Point
from dispatchatlas.analysis.figures.frames import (
    Aggregation,
    Observation,
    check_aggregation,
    objective_sense,
    observe,
)
from dispatchatlas.analysis.figures.theme import DEFAULT_THEME, Theme
from dispatchatlas.analysis.stats.ranks import Sense

__all__ = ["ParetoFronts", "Surface", "attainment_surface", "pareto_fronts", "plot_pareto_fronts"]


@dataclass(frozen=True)
class Surface:
    """One solver's attainment surface on a problem.

    Attributes:
        solver: Solver name.
        points: The corners of the surface, in objective units, ordered by
            the first objective from best to worst.
        level: The surface bounds the vectors attained in at least this many runs.
        runs: Planned runs of the solver on the problem.
    """

    solver: str
    points: tuple[Point, ...]
    level: int
    runs: int


@dataclass(frozen=True)
class ParetoFronts:
    """The attainment surfaces of every solver on one bi-objective problem.

    Attributes:
        problem: Problem id.
        objectives: The two objectives, in the problem's order.
        senses: Their senses.
        surfaces: One surface per solver with planned runs, in campaign order.
        aggregation: ``"median"`` or ``"best"``.
    """

    problem: str
    objectives: tuple[str, str]
    senses: tuple[Sense, Sense]
    surfaces: tuple[Surface, ...]
    aggregation: Aggregation


def attainment_surface(runs: Sequence[Sequence[Point]], level: int) -> tuple[Point, ...]:
    """Return the corners of the ``level``-th summary attainment surface of point sets.

    Args:
        runs: The solutions of each run, as minimisation vectors; an empty
            set is a run that attained nothing.
        level: How many runs must attain a vector, from 1.

    Returns:
        The corners, ascending in the first coordinate and strictly
        descending in the second; empty when fewer than ``level`` runs
        attain anything.
    """
    ordered = [sorted(points) for points in runs]
    if level < 1 or level > len(ordered):
        return ()
    firsts = sorted({point[0] for points in ordered for point in points})
    cursor = [0] * len(ordered)
    reach = [math.inf] * len(ordered)
    corners: list[Point] = []
    for first in firsts:
        for index, points in enumerate(ordered):
            while cursor[index] < len(points) and points[cursor[index]][0] <= first:
                reach[index] = min(reach[index], points[cursor[index]][1])
                cursor[index] += 1
        second = sorted(reach)[level - 1]
        if math.isfinite(second) and (not corners or second < corners[-1][1]):
            corners.append((first, second))
    return tuple(corners)


def pareto_fronts(source: object, *, aggregation: str = "median") -> tuple[ParetoFronts, ...]:
    """Return the attainment surfaces of every solver on every bi-objective problem.

    Args:
        source: Runs of one protocol, anything
            :func:`~dispatchatlas.analysis.figures.frames.observe` accepts.
        aggregation: ``"median"`` for the surface attained in at least half
            of the planned runs, ``"best"`` for the one attained in any run.

    Returns:
        One entry per problem with exactly two objectives, in plan order.

    Raises:
        ConfigError: If the aggregation is unknown or the runs span several
            protocols.
    """
    chosen = check_aggregation(aggregation)
    observations = observe(source).comparable()
    fronts = []
    for problem in observations.problems:
        names = observations.objectives.get(problem, ())
        if len(names) != 2:
            continue
        senses = (objective_sense(names[0]), objective_sense(names[1]))
        signs = tuple(1.0 if sense == "min" else -1.0 for sense in senses)
        surfaces = []
        for solver in observations.solvers:
            runs = observations.cell(solver, problem)
            if not runs:
                continue
            sets = [_vectors(run, names, signs) for run in runs]
            level = 1 if chosen == "best" else math.ceil(len(runs) / 2)
            corners = attainment_surface(sets, level)
            natural = tuple((signs[0] * a, signs[1] * b) for a, b in corners)
            surfaces.append(Surface(solver, natural, level, len(runs)))
        fronts.append(ParetoFronts(problem, (names[0], names[1]), senses, tuple(surfaces), chosen))
    return tuple(fronts)


def _vectors(run: Observation, names: tuple[str, ...], signs: tuple[float, ...]) -> list[Point]:
    """Return a succeeded run's solutions as minimisation vectors."""
    if not run.succeeded or run.record is None:
        return []
    outcome = run.record.outcome
    vectors: list[tuple[float, ...]]
    if outcome.front:
        vectors = [point for point in outcome.front if len(point) == 2]
    else:
        vectors = [(outcome.objectives[names[0]], outcome.objectives[names[1]])]
    return [(signs[0] * vector[0], signs[1] * vector[1]) for vector in vectors]


def plot_pareto_fronts(
    fronts: ParetoFronts,
    *,
    labels: Mapping[str, str] | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Draw each solver's attainment surface on one problem as a staircase.

    Args:
        fronts: The surfaces of one problem.
        labels: Display names of solvers; a solver without one shows its name.
        theme: The appearance.

    Returns:
        The figure, named ``pareto-front-<problem>`` with characters that
        cannot name a file replaced by ``_``, or a placeholder when no surface
        has a point.
    """
    active = theme or DEFAULT_THEME
    name = "pareto-front-" + re.sub(r"[^A-Za-z0-9._-]", "_", fronts.problem)
    title = f"Pareto fronts on {fronts.problem}"
    corners = [point for surface in fronts.surfaces for point in surface.points]
    if not corners:
        return placeholder(name, title, "No solver attained any objective vector.", active)
    axes = []
    for position, (objective, sense) in enumerate(
        zip(fronts.objectives, fronts.senses, strict=True)
    ):
        suffix = " (maximised)" if sense == "max" else ""
        axes.append(numeric_axis(objective + suffix, [point[position] for point in corners]))
    x, y = axes
    worst_x = x.high if fronts.senses[0] == "min" else x.low
    worst_y = y.high if fronts.senses[1] == "min" else y.low
    series = []
    for index, surface in enumerate(fronts.surfaces):
        style = active.series(index)
        label = label_for(surface.solver, labels)
        if not surface.points:
            series.append(Series(f"{label} (nothing attained)", (), style, kind="points"))
            continue
        outline: list[Point] = [(surface.points[0][0], worst_y)]
        for (first, second), following in zip(
            surface.points, [*surface.points[1:], None], strict=True
        ):
            outline.append((first, second))
            if following is not None:
                outline.append((following[0], second))
        outline.append((worst_x, surface.points[-1][1]))
        series.append(Series(label, tuple(outline), style, kind="line", markers=False))
        series.append(Series(label, surface.points, style, kind="points", legend=False))
    which = (
        "attained, that is weakly dominated, in at least half of its planned runs"
        if fronts.aggregation == "median"
        else "attained in at least one of its runs"
    )
    notes = [
        f"Each staircase bounds the objective vectors a solver {which} (Fonseca and Fleming "
        "1996); markers are its corners, and failed runs attain nothing.",
    ]
    description = (
        f"Attainment surfaces of {len(fronts.surfaces)} solvers on problem {fronts.problem}, "
        f"objectives {fronts.objectives[0]} and {fronts.objectives[1]}. Corners: "
        + "; ".join(
            f"{label_for(surface.solver, labels)} {len(surface.points)}"
            for surface in fronts.surfaces
        )
        + ". "
        + notes[0]
    )
    return xy_chart(name, title, description, x=x, y=y, series=series, notes=notes, theme=active)
