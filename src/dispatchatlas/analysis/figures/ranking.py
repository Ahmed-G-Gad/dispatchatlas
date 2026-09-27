"""The critical-difference diagram of Demšar (2006).

Solvers sit on an axis of average ranks, rank 1 best. When the omnibus test
rejects, the Nemenyi critical difference ``CD`` is drawn as a bar of that
length, and every maximal group of solvers, consecutive in rank order, whose
average ranks lie within ``CD`` of each other is joined by a thick line: the
members of a group are not significantly different. A pair of solvers two
groups apart differs significantly. Groups of one are not drawn.

A diagram can be built from values computed elsewhere, such as the
:class:`~dispatchatlas.analysis.results.CriticalDifference` of a summary
(:func:`rank_diagram`), or from runs (:func:`average_ranks`), which follows
the procedure of :func:`~dispatchatlas.analysis.summary.summarize` with the
default :class:`~dispatchatlas.analysis.config.AnalysisConfig`: each cell
scores the mean (or configured statistic) of its planned runs, a failure
counting as the worst value; the complete blocks are ranked; the omnibus
test gates the Nemenyi test, whose critical values come from the statistics
package.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.figures.chart import PAD, draw_notes, draw_title, label_for, placeholder
from dispatchatlas.analysis.figures.drawing import (
    Anchor,
    Figure,
    Line,
    Shape,
    Stroke,
    Text,
    text_width,
    truncate,
)
from dispatchatlas.analysis.figures.frames import observe
from dispatchatlas.analysis.figures.theme import DEFAULT_THEME, Theme
from dispatchatlas.analysis.stats.nonparametric import (
    friedman_test,
    iman_davenport,
    nemenyi_critical_difference,
)
from dispatchatlas.analysis.stats.ranks import rank_block
from dispatchatlas.core.errors import ConfigError

__all__ = ["RankDiagram", "average_ranks", "cliques", "plot_critical_difference", "rank_diagram"]

_ROW = 18.0
_CLIQUE_ROW = 7.0
_OMNIBUS_NAMES = {"friedman": "Friedman", "iman-davenport": "Iman-Davenport"}


@dataclass(frozen=True)
class RankDiagram:
    """What a critical-difference diagram shows.

    Attributes:
        ranks: Each solver's average rank, best first.
        critical_difference: The Nemenyi critical difference, or ``None``
            when no post-hoc test applies.
        alpha: Significance level of the test.
        blocks: Problems the ranks average over.
        cliques: Groups of at least two solvers, in rank order, whose ranks
            lie within the critical difference of each other.
        note: Why no critical difference is drawn, when none is.
        objective: Name of the objective ranked, empty when unknown.
    """

    ranks: tuple[tuple[str, float], ...]
    critical_difference: float | None
    alpha: float
    blocks: int
    cliques: tuple[tuple[str, ...], ...]
    note: str = ""
    objective: str = ""


def cliques(
    ranks: Sequence[tuple[str, float]], critical_difference: float
) -> tuple[tuple[str, ...], ...]:
    """Return the maximal groups of rank-ordered solvers within the critical difference.

    A group is a run of consecutive solvers in rank order whose first and
    last ranks differ by at most ``critical_difference``; a group contained
    in another is not returned, and neither are groups of one.
    """
    ordered = sorted(ranks, key=lambda item: (item[1], item[0]))
    spans: list[tuple[int, int]] = []
    for start, (_, low) in enumerate(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][1] - low <= critical_difference + 1e-12:
            end += 1
        if end > start and (not spans or end > spans[-1][1]):
            spans.append((start, end))
    return tuple(tuple(name for name, _ in ordered[start : end + 1]) for start, end in spans)


def rank_diagram(
    ranks: Mapping[str, float] | Sequence[tuple[str, float]],
    critical_difference: float | None = None,
    *,
    alpha: float = 0.05,
    blocks: int = 0,
    note: str = "",
    objective: str = "",
) -> RankDiagram:
    """Return a diagram of average ranks and a critical difference computed elsewhere.

    Args:
        ranks: Each solver's average rank.
        critical_difference: The critical difference, or ``None`` to draw
            the ranks only.
        alpha: Significance level the critical difference belongs to.
        blocks: Problems the ranks average over, shown in the notes.
        note: Why no critical difference is given, shown in the notes.
        objective: Name of the objective ranked, shown in the notes.

    Raises:
        ConfigError: If a rank or the critical difference is not a positive
            finite number, or a solver is ranked twice.
    """
    items = list(ranks.items()) if isinstance(ranks, Mapping) else list(ranks)
    names = [name for name, _ in items]
    if len(set(names)) != len(names):
        raise ConfigError("a solver is ranked twice")
    for name, rank in items:
        if not (math.isfinite(rank) and rank >= 1.0):
            raise ConfigError(f"the average rank of {name!r} must be at least 1, got {rank!r}")
    if critical_difference is not None and not (
        math.isfinite(critical_difference) and critical_difference > 0.0
    ):
        raise ConfigError(f"the critical difference must be positive, got {critical_difference}")
    ordered = tuple(sorted(((name, float(rank)) for name, rank in items), key=_rank_order))
    groups = () if critical_difference is None else cliques(ordered, critical_difference)
    return RankDiagram(ordered, critical_difference, alpha, blocks, groups, note, objective)


def _rank_order(item: tuple[str, float]) -> tuple[float, str]:
    return item[1], item[0]


def average_ranks(source: object, config: AnalysisConfig | None = None) -> RankDiagram:
    """Return average ranks and the gated Nemenyi critical difference, from runs.

    Args:
        source: Runs of one protocol, anything
            :func:`~dispatchatlas.analysis.figures.frames.observe` accepts.
        config: The analysis settings, the defaults when ``None``: the runs
            of its protocol, solvers, and problems are ranked on its
            objective and sense, with its ``statistic``, ``omnibus`` test,
            and ``alpha``.

    Returns:
        The diagram, with ``critical_difference`` ``None`` and a note when
        there are fewer than three solvers or two complete blocks, when the
        omnibus test does not reject, or when no critical value is tabulated.

    Raises:
        ConfigError: If the runs span several protocols, or the settings
            name something the runs lack.
    """
    settings = config if config is not None else AnalysisConfig()
    observations = observe(source).configured(settings).comparable()
    solvers = observations.solvers
    blocks = [
        problem
        for problem in observations.problems
        if solvers and all(observations.cell(solver, problem) for solver in solvers)
    ]
    rows: list[dict[str, float]] = []
    for problem in blocks:
        rows.append(
            {
                solver: _aggregate(
                    [observations.score(run) for run in observations.cell(solver, problem)],
                    settings.statistic,
                )
                for solver in solvers
            }
        )
    objective = observations.objective_label()
    if not rows:
        return rank_diagram({}, note="No problem is planned for every solver.")
    totals = dict.fromkeys(solvers, 0.0)
    for row in rows:
        for solver, rank in rank_block(row, "min").items():
            totals[solver] += rank
    ranks = {solver: total / len(rows) for solver, total in totals.items()}
    if len(solvers) < 3 or len(rows) < 2:
        note = (
            "The omnibus test needs at least three solvers and two problems that every solver "
            "is planned on, so no critical difference is drawn."
        )
        return rank_diagram(
            ranks, alpha=settings.alpha, blocks=len(rows), note=note, objective=objective
        )
    friedman = friedman_test(rows, "min")
    p_value = (
        friedman.p_value if settings.omnibus == "friedman" else iman_davenport(friedman).p_value
    )
    if not p_value < settings.alpha:
        note = (
            f"The {_OMNIBUS_NAMES[settings.omnibus]} test does not reject at alpha = "
            f"{settings.alpha:g} (p = {p_value:.3g}), so no critical difference is drawn."
        )
        return rank_diagram(
            ranks, alpha=settings.alpha, blocks=len(rows), note=note, objective=objective
        )
    try:
        value = nemenyi_critical_difference(len(solvers), len(rows), settings.alpha)
    except ConfigError as error:
        note = f"No Nemenyi test: {error}."
        return rank_diagram(
            ranks, alpha=settings.alpha, blocks=len(rows), note=note, objective=objective
        )
    return rank_diagram(ranks, value, alpha=settings.alpha, blocks=len(rows), objective=objective)


def _aggregate(scores: list[float], statistic: str) -> float:
    """Return the mean or median of minimisation scores, infinite when failures decide it."""
    if statistic == "median":
        return float(statistics.median(scores))
    if any(math.isinf(score) for score in scores):
        return math.inf
    return math.fsum(scores) / len(scores)


def plot_critical_difference(
    diagram: RankDiagram,
    *,
    labels: Mapping[str, str] | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Draw a critical-difference diagram.

    The better half of the solvers is labelled on the left, the other half
    on the right, each label joined to the solver's rank by a line; groups
    that the Nemenyi test does not separate are joined by thick bars under
    the axis.

    Args:
        diagram: Ranks, critical difference, and groups.
        labels: Display names of solvers; a solver without one shows its name.
        theme: The appearance.

    Returns:
        The figure, or a placeholder with fewer than two ranked solvers.
    """
    active = theme or DEFAULT_THEME
    name, title = "critical-difference", "Critical-difference diagram"
    if len(diagram.ranks) < 2:
        reason = diagram.note or "A critical-difference diagram needs at least two ranked solvers."
        return placeholder(name, title, reason, active)
    shapes: list[Shape] = []
    top = draw_title(shapes, title, active)
    count = len(diagram.ranks)
    half = math.ceil(count / 2)
    left_group = list(diagram.ranks[:half])
    right_group = list(reversed(diagram.ranks[half:]))
    size = active.legend_size
    texts = {solver: f"{label_for(solver, labels)} ({rank:.2f})" for solver, rank in diagram.ranks}
    budget = active.width * 0.28
    left_width = min(budget, max(text_width(texts[s], size) for s, _ in left_group))
    right_width = min(
        budget, max((text_width(texts[s], size) for s, _ in right_group), default=0.0)
    )
    axis_left = PAD + left_width + 24.0
    axis_right = active.width - PAD - right_width - 24.0
    high = float(count)

    def x_of(rank: float) -> float:
        return axis_left + (min(max(rank, 1.0), high) - 1.0) / (high - 1.0) * (
            axis_right - axis_left
        )

    axis_y = top + active.tick_size + 10.0
    if diagram.critical_difference is not None:
        axis_y += active.tick_size + 16.0
        bar_y = top + active.tick_size + 4.0
        end = x_of(1.0 + diagram.critical_difference)
        heavy = Stroke(active.foreground, 2.0)
        shapes.append(Line((axis_left, bar_y), (end, bar_y), heavy))
        shapes.append(Line((axis_left, bar_y - 4.0), (axis_left, bar_y + 4.0), heavy))
        shapes.append(Line((end, bar_y - 4.0), (end, bar_y + 4.0), heavy))
        label = f"CD = {diagram.critical_difference:.3f}"
        shapes.append(
            Text(
                (axis_left + end) / 2.0,
                bar_y - 6.0,
                label,
                active.tick_size,
                active.foreground,
                "middle",
            )
        )
    axis = Stroke(active.axis, 1.0)
    shapes.append(Line((axis_left, axis_y), (axis_right, axis_y), axis, "axis"))
    spacing = (axis_right - axis_left) / (high - 1.0)
    every = max(1, math.ceil(20.0 / spacing))
    for tick in range(1, count + 1):
        position = x_of(float(tick))
        shapes.append(Line((position, axis_y - 5.0), (position, axis_y), axis, "axis"))
        if (tick - 1) % every == 0 or tick == count:
            shapes.append(
                Text(
                    position, axis_y - 8.0, str(tick), active.tick_size, active.foreground, "middle"
                )
            )
    ranks = dict(diagram.ranks)
    for index, members in enumerate(diagram.cliques):
        y = axis_y + 8.0 + index * _CLIQUE_ROW
        start, end = x_of(ranks[members[0]]) - 3.0, x_of(ranks[members[-1]]) + 3.0
        shapes.append(Line((start, y), (end, y), Stroke(active.foreground, 3.0)))
    first_row = axis_y + 8.0 + len(diagram.cliques) * _CLIQUE_ROW + 12.0
    line = Stroke(active.muted, 1.0)
    for side, group in (("left", left_group), ("right", right_group)):
        for index, (solver, rank) in enumerate(group):
            y = first_row + index * _ROW
            position = x_of(rank)
            edge = axis_left - 16.0 if side == "left" else axis_right + 16.0
            shapes.append(Line((position, axis_y), (position, y), line))
            shapes.append(Line((position, y), (edge, y), line))
            width = left_width if side == "left" else right_width
            text = truncate(texts[solver], size, width)
            anchor: Anchor = "end" if side == "left" else "start"
            label_x = edge - 4.0 if side == "left" else edge + 4.0
            shapes.append(Text(label_x, y + size * 0.35, text, size, active.foreground, anchor))
    bottom = first_row + (max(len(left_group), len(right_group)) - 1) * _ROW + size
    ranked = f"Average ranks of {diagram.objective}" if diagram.objective else "Average ranks"
    notes = [
        f"{ranked} over {diagram.blocks} problems every solver is planned on; rank 1 is best."
        if diagram.blocks
        else f"{ranked}; rank 1 is best.",
    ]
    if diagram.critical_difference is not None:
        notes.append(
            "Solvers joined by a bar are not significantly different by the Nemenyi test at "
            f"alpha = {diagram.alpha:g} (Dem\u0161ar 2006)."
        )
    if diagram.note:
        notes.append(diagram.note)
    bottom = draw_notes(shapes, notes, bottom + 6.0, active)
    description = (
        "Average ranks: "
        + "; ".join(f"{label_for(solver, labels)} {rank:.2f}" for solver, rank in diagram.ranks)
        + ". "
        + " ".join(notes)
    )
    return Figure(name, title, description, active.width, bottom + PAD, tuple(shapes), active)
