"""The distribution of each solver's results, as box plots.

Values of different problems are made comparable by their relative
percentage deviation from the best value any run found on the problem,
computed by :func:`~dispatchatlas.analysis.deviation.relative_deviation`, so
0 is the best value found and larger is worse under either sense. Each
solver's box pools the deviations of its succeeded runs on every problem;
the row also states how many of its planned runs succeeded, since a failed
run has no value to place. A problem whose best value is not positive has
no relative deviation and is left out, and reported.

The box spans the inclusive quartiles (type 7 of Hyndman and Fan 1996), the
inner line marks the median, and the whiskers reach the smallest and
largest values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from dispatchatlas.analysis.deviation import relative_deviation
from dispatchatlas.analysis.figures.axes import numeric_axis
from dispatchatlas.analysis.figures.chart import (
    BoxStats,
    Row,
    box_stats,
    label_for,
    placeholder,
    row_chart,
)
from dispatchatlas.analysis.figures.drawing import Figure
from dispatchatlas.analysis.figures.frames import observe
from dispatchatlas.analysis.figures.theme import DEFAULT_THEME, Theme

__all__ = ["Box", "Distribution", "deviation_distribution", "plot_distribution"]


@dataclass(frozen=True)
class Box:
    """The deviations of one solver.

    Attributes:
        solver: Solver name.
        stats: Box-plot statistics of the deviations, in percent, or ``None``
            when no run succeeded.
        succeeded: Succeeded runs on the problems of the distribution.
        planned: Planned runs on those problems.
    """

    solver: str
    stats: BoxStats | None
    succeeded: int
    planned: int


@dataclass(frozen=True)
class Distribution:
    """Box plots of every solver's relative deviations.

    Attributes:
        boxes: One box per solver, in campaign order.
        problems: The problems whose runs the boxes pool.
        excluded: Problems left out because their best value is not positive.
        objective: Name of the objective measured, empty when unknown.
    """

    boxes: tuple[Box, ...]
    problems: tuple[str, ...]
    excluded: tuple[str, ...]
    objective: str = ""


def deviation_distribution(source: object) -> Distribution:
    """Return each solver's relative percentage deviations from the best value found.

    Args:
        source: Runs of one protocol, anything
            :func:`~dispatchatlas.analysis.figures.frames.observe` accepts.

    Returns:
        The box of every solver.

    Raises:
        ConfigError: If the runs span several protocols.
    """
    observations = observe(source).comparable()
    deviations: dict[str, list[float]] = {solver: [] for solver in observations.solvers}
    planned = dict.fromkeys(observations.solvers, 0)
    kept: list[str] = []
    excluded: list[str] = []
    for problem in observations.problems:
        sense = observations.sense(problem)
        runs = [
            run for solver in observations.solvers for run in observations.cell(solver, problem)
        ]
        scores = [observations.score(run) for run in runs if observations.value(run) is not None]
        best = min(scores, default=math.inf)
        reference = best if sense == "min" else -best
        if math.isfinite(best) and reference <= 0.0:
            excluded.append(problem)
            continue
        kept.append(problem)
        for run in runs:
            planned[run.solver] += 1
            value = observations.value(run)
            if value is not None:
                deviations[run.solver].append(relative_deviation(value, reference, sense))
    boxes = tuple(
        Box(
            solver,
            box_stats(values) if values else None,
            len(values),
            planned[solver],
        )
        for solver, values in deviations.items()
    )
    return Distribution(boxes, tuple(kept), tuple(excluded), observations.objective_label())


def plot_distribution(
    boxes: Distribution,
    *,
    labels: Mapping[str, str] | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Draw one horizontal box plot per solver.

    Args:
        boxes: The distribution.
        labels: Display names of solvers; a solver without one shows its name.
        theme: The appearance.

    Returns:
        The figure, or a placeholder when no run succeeded.
    """
    active = theme or DEFAULT_THEME
    name, title = "distribution", "Relative deviation from the best value found"
    values = [
        value
        for box in boxes.boxes
        if box.stats is not None
        for value in (box.stats.minimum, box.stats.maximum)
    ]
    if not values:
        return placeholder(name, title, "No run succeeded, so there is no value to place.", active)
    measured = f" of {boxes.objective}" if boxes.objective else ""
    x = numeric_axis(
        f"Relative deviation{measured} from the best value found (%)", [*values, 0.0], low=0.0
    )
    rows = [
        Row(
            label_for(box.solver, labels),
            box.stats,
            active.series(index),
            f"{box.succeeded} of {box.planned} runs",
        )
        for index, box in enumerate(boxes.boxes)
    ]
    notes = [
        "Boxes span the quartiles and whiskers the extremes of the relative percentage deviation "
        "of each succeeded run from the best value any run found on its problem; the count "
        "states the succeeded runs among the planned ones.",
    ]
    if boxes.excluded:
        notes.append(
            f"{len(boxes.excluded)} problem(s) left out because their best value is not positive: "
            + ", ".join(boxes.excluded)
            + "."
        )
    description = (
        f"Distribution of the relative deviation of {len(boxes.boxes)} solvers over "
        f"{len(boxes.problems)} problems. Median deviations: "
        + "; ".join(
            f"{label_for(box.solver, labels)} "
            + (f"{box.stats.median:.3g}%" if box.stats is not None else "none")
            for box in boxes.boxes
        )
        + ". "
        + " ".join(notes)
    )
    return row_chart(
        name, title, description, x=x, rows=rows, row_height=26.0, notes=notes, theme=active
    )
