"""Convergence: how the best value found improves with the effort spent.

Values of different problems are made comparable by a gap in ``[0, 1]``:
0 at the best value any run found on the problem, 1 at the worst first
feasible value of a run (see
:class:`~dispatchatlas.analysis.figures.frames.QualityRange`). A run's gap
at a budget is that of the best feasible value it had found by then; before
its first feasible schedule, and throughout a run that did not succeed, the
gap is 1. A solver's curve is the median gap over all its planned runs, on
every problem, with the interquartile range as a band, sampled on a grid of
budgets spaced evenly on a logarithmic scale.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass

from dispatchatlas.analysis.figures.axes import log_axis, numeric_axis
from dispatchatlas.analysis.figures.chart import Series, label_for, placeholder, xy_chart
from dispatchatlas.analysis.figures.drawing import Figure
from dispatchatlas.analysis.figures.frames import (
    Budget,
    Observation,
    Observations,
    QualityRange,
    check_budget,
    observe,
    quality_ranges,
)
from dispatchatlas.analysis.figures.theme import DEFAULT_THEME, Theme
from dispatchatlas.core.errors import ConfigError

__all__ = ["Convergence", "ConvergenceCurve", "convergence", "plot_convergence"]

GRID_POINTS = 100
"""Budgets at which convergence curves are sampled by default."""


@dataclass(frozen=True)
class ConvergenceCurve:
    """The gap of one solver's runs, by budget.

    Attributes:
        solver: Solver name.
        budgets: The sampled budgets, ascending.
        median: Median gap at each budget.
        lower: First quartile of the gap.
        upper: Third quartile of the gap.
        runs: Planned runs summarised.
    """

    solver: str
    budgets: tuple[float, ...]
    median: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    runs: int


@dataclass(frozen=True)
class Convergence:
    """Convergence curves of every solver.

    Attributes:
        curves: One curve per solver with planned runs, in campaign order.
        budget: What measures the effort.
        problems: The problems the runs solved.
    """

    curves: tuple[ConvergenceCurve, ...]
    budget: Budget
    problems: tuple[str, ...]


def convergence(
    source: object, *, budget: str = "evaluations", points: int = GRID_POINTS
) -> Convergence:
    """Return each solver's median convergence curve with its interquartile range.

    Quartiles follow the inclusive method, as in the box plots.

    Args:
        source: Runs of one protocol, anything :func:`observe` accepts.
        budget: ``"evaluations"`` or ``"time"``.
        points: Budgets on the sampling grid, at least 2.

    Returns:
        The curves. They are empty when no run recorded a positive budget.

    Raises:
        ConfigError: If the budget is unknown, ``points`` is below 2, or the
            runs span several protocols.
    """
    measure = check_budget(budget)
    if isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise ConfigError(f"a convergence grid needs at least 2 points, got {points!r}")
    observations = observe(source).comparable()
    ranges = quality_ranges(observations, measure)
    steps = {
        run.run_id: _step_function(observations, run, ranges, measure) for run in observations.runs
    }
    grid = _grid(observations, measure, points)
    curves = []
    for solver in observations.solvers:
        runs = [run for run in observations.runs if run.solver == solver]
        if not runs or not grid:
            continue
        medians, lowers, uppers = [], [], []
        for spent in grid:
            gaps = [_gap_at(steps[run.run_id], spent) for run in runs]
            lower, middle, upper = _quartiles(gaps)
            medians.append(middle)
            lowers.append(lower)
            uppers.append(upper)
        curves.append(
            ConvergenceCurve(
                solver, tuple(grid), tuple(medians), tuple(lowers), tuple(uppers), len(runs)
            )
        )
    return Convergence(tuple(curves), measure, observations.problems)


def _step_function(
    observations: Observations,
    run: Observation,
    ranges: Mapping[str, QualityRange],
    budget: Budget,
) -> tuple[list[float], list[float]]:
    """Return a run's gap as a step function: the budgets where it drops, and the gaps."""
    span = ranges.get(run.problem)
    if span is None:
        return [], []
    progress = observations.progress(run, budget)
    return [spent for spent, _ in progress], [span.gap(value) for _, value in progress]


def _gap_at(step: tuple[list[float], list[float]], spent: float) -> float:
    budgets, gaps = step
    index = bisect_right(budgets, spent) - 1
    return 1.0 if index < 0 else gaps[index]


def _grid(observations: Observations, budget: Budget, points: int) -> list[float]:
    """Return budgets spaced evenly on a logarithmic scale over the effort the runs recorded."""
    first = math.inf
    for run in observations.runs:
        for spent, _ in observations.progress(run, budget):
            if spent > 0.0:
                first = min(first, spent)
                break
    last = max((run.spent(budget) for run in observations.runs), default=0.0)
    if not math.isfinite(first):
        return []
    if last <= first:
        return [first]
    ratio = math.log(last / first)
    return [first * math.exp(ratio * index / (points - 1)) for index in range(points)]


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    if len(values) == 1:
        return values[0], values[0], values[0]
    lower, middle, upper = statistics.quantiles(values, n=4, method="inclusive")
    return lower, middle, upper


def plot_convergence(
    curves: Convergence,
    *,
    labels: Mapping[str, str] | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Draw median gaps as step functions of the budget, with interquartile bands.

    Args:
        curves: The convergence curves.
        labels: Display names of solvers; a solver without one shows its name.
        theme: The appearance.

    Returns:
        The figure, named ``convergence-evaluations`` or ``convergence-time``,
        or a placeholder when no run recorded any progress.
    """
    active = theme or DEFAULT_THEME
    over = "evaluations" if curves.budget == "evaluations" else "time"
    name, title = f"convergence-{over}", f"Convergence over {over}"
    if not curves.curves:
        return placeholder(name, title, "No run recorded a feasible solution.", active)
    budgets = [spent for curve in curves.curves for spent in curve.budgets]
    unit = "Evaluations" if curves.budget == "evaluations" else "Elapsed seconds"
    x = log_axis(unit, budgets) if max(budgets) > min(budgets) else numeric_axis(unit, budgets)
    y = numeric_axis("Gap to the best value found", [], low=0.0, high=1.0)
    series = [
        Series(
            label_for(curve.solver, labels),
            tuple(zip(curve.budgets, curve.median, strict=True)),
            active.series(index),
            kind="step",
            band=tuple(zip(curve.budgets, curve.lower, curve.upper, strict=True)),
            until=x.high,
        )
        for index, curve in enumerate(curves.curves)
    ]
    notes = [
        "Median over each solver's planned runs on all problems, with the interquartile range "
        "shaded. The gap is 0 at the best value any run found on a problem and 1 at the worst "
        "first value of a run; a run counts as 1 until its first feasible schedule, and a failed "
        "run throughout.",
    ]
    description = (
        f"Convergence of {len(curves.curves)} solvers over {len(curves.problems)} problems. "
        "Median gap at the largest budget: "
        + "; ".join(
            f"{label_for(curve.solver, labels)} {curve.median[-1]:.3f}" for curve in curves.curves
        )
        + ". "
        + notes[0]
    )
    return xy_chart(name, title, description, x=x, y=y, series=series, notes=notes, theme=active)
