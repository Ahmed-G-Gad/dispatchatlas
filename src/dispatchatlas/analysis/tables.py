"""Tables of analysis results, rendered as Markdown, CSV, or LaTeX.

A :class:`Table` holds typed columns and raw values; the renderers format
them. Markdown and LaTeX show six significant digits (three for p-values),
an em dash for a missing value, and intervals as ``[low, high]``. CSV keeps
full precision, writes an interval as two columns ``<key>_low`` and
``<key>_high``, and names columns by their keys, one row per record.

Every renderer refuses a table that reports a p-value without an effect size
and a confidence interval beside it, as Carrasco et al. (2020) recommend:
:func:`check_p_values` enforces it from the column kinds.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from dispatchatlas.analysis.config import TimeoutScore
from dispatchatlas.analysis.dataset import Dataset
from dispatchatlas.analysis.deviation import DeviationRow
from dispatchatlas.analysis.indicators import IndicatorReport
from dispatchatlas.analysis.results import BaselineComparison, PairwiseComparison, Summary
from dispatchatlas.core.errors import ConfigError

__all__ = [
    "Cell",
    "Column",
    "ColumnKind",
    "Table",
    "baselines_table",
    "cells_table",
    "check_p_values",
    "deviation_table",
    "indicators_table",
    "omnibus_table",
    "pairwise_table",
    "runs_table",
    "solvers_table",
    "to_csv",
    "to_latex",
    "to_markdown",
]

type ColumnKind = Literal["text", "count", "number", "p-value", "effect", "interval"]
"""What a column holds, which decides its formatting and the p-value check."""

type Cell = str | int | float | bool | tuple[float | None, float | None] | None
"""A table value; intervals are ``(low, high)`` pairs."""

_CORRECTION_NAMES = {
    "none": "Unadjusted",
    "bonferroni": "Bonferroni",
    "holm": "Holm",
    "hochberg": "Hochberg",
    "shaffer": "Shaffer",
    "finner": "Finner",
}
_MISSING = "—"


@dataclass(frozen=True, slots=True)
class Column:
    """A table column.

    Attributes:
        key: Machine name, used as the CSV header.
        header: Human-readable header, used in Markdown and LaTeX.
        kind: What the column holds.
    """

    key: str
    header: str
    kind: ColumnKind = "text"


@dataclass(frozen=True)
class Table:
    """A titled table of typed columns.

    Attributes:
        name: File stem, for example ``"pairwise"``.
        title: One-line caption.
        columns: The columns.
        rows: One value per column in every row.
        notes: Sentences printed under the table in Markdown and as comments
            in LaTeX.
    """

    name: str
    title: str
    columns: tuple[Column, ...]
    rows: tuple[tuple[Cell, ...], ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ConfigError(f"table {self.name!r} has a row of the wrong width")


def check_p_values(table: Table) -> None:
    """Refuse a table that reports a p-value without an effect size and an interval.

    Raises:
        ConfigError: If the table has a p-value column but no effect-size
            column or no interval column.
    """
    kinds = {column.kind for column in table.columns}
    if "p-value" in kinds and not {"effect", "interval"} <= kinds:
        raise ConfigError(
            f"table {table.name!r} reports p-values without an effect size and a confidence "
            "interval beside them (Carrasco et al. 2020)"
        )


def to_markdown(table: Table) -> str:
    """Render a table as GitHub-flavoured Markdown, with its title and notes."""
    check_p_values(table)
    lines = [f"**{_escape_markdown(table.title)}**", ""]
    lines.append("| " + " | ".join(_escape_markdown(c.header) for c in table.columns) + " |")
    lines.append(
        "| " + " | ".join("---" if c.kind == "text" else "---:" for c in table.columns) + " |"
    )
    for row in table.rows:
        cells = [
            _escape_markdown(_display(value, c.kind))
            for value, c in zip(row, table.columns, strict=True)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if not table.rows:
        lines.extend(["", "_No rows._"])
    for note in table.notes:
        lines.extend(["", _escape_markdown(note)])
    return "\n".join(lines) + "\n"


def to_csv(table: Table) -> str:
    """Render a table as CSV at full precision, headed by the column keys."""
    check_p_values(table)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    header: list[str] = []
    for column in table.columns:
        if column.kind == "interval":
            header.extend([f"{column.key}_low", f"{column.key}_high"])
        else:
            header.append(column.key)
    writer.writerow(header)
    for row in table.rows:
        values: list[str] = []
        for value, column in zip(row, table.columns, strict=True):
            if column.kind == "interval":
                low, high = value if isinstance(value, tuple) else (None, None)
                values.extend([_exact(low), _exact(high)])
            else:
                values.append(_exact(value))
        writer.writerow(values)
    return buffer.getvalue()


def to_latex(table: Table) -> str:
    r"""Render a table as a LaTeX ``tabular`` with booktabs rules.

    The title and notes become comments above and below, so the fragment can
    be placed in any float. It needs ``\usepackage{booktabs}``.
    """
    check_p_values(table)
    spec = "".join("l" if column.kind == "text" else "r" for column in table.columns)
    lines = [f"% {_comment(table.title)}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    lines.append(" & ".join(_escape_latex(column.header) for column in table.columns) + r" \\")
    lines.append("\\midrule")
    for row in table.rows:
        cells = [_latex_value(value, c.kind) for value, c in zip(row, table.columns, strict=True)]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    lines.extend(f"% {_comment(note)}" for note in table.notes)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ builders


def runs_table(dataset: Dataset, timeout_score: TimeoutScore = "worst") -> Table:
    """Return one row per run of the dataset: its facts and every objective value.

    A failed run names the kind of its last failed attempt. The column
    ``scored_by`` says where the score of a run comes from: its ``result``,
    the ``worst`` value, or, with ``timeout_score`` ``"incumbent"``, the
    ``incumbent`` a run killed at its hard limit held when its time limit
    passed. The objective columns of such a run hold the values of that
    incumbent, and its ``evaluations`` and ``elapsed`` columns say when its
    search found it, which is within the time limit.

    Args:
        dataset: The runs.
        timeout_score: How the analysis the table belongs to scores a run
            that was killed at its hard limit.
    """
    objectives = dataset.objectives
    columns = (
        Column("run_id", "Run"),
        Column("protocol", "Protocol"),
        Column("problem", "Problem"),
        Column("solver", "Solver"),
        Column("replicate", "Replicate", "count"),
        Column("seed", "Seed", "count"),
        Column("status", "Status"),
        Column("failure", "Failure"),
        Column("scored_by", "Scored by"),
        Column("stop_reason", "Stop reason"),
        Column("evaluations", "Evaluations", "count"),
        Column("iterations", "Iterations", "count"),
        Column("elapsed", "Elapsed (s)", "number"),
        Column("cpu_seconds", "CPU (s)", "number"),
        *(Column(f"objective.{name}", name, "number") for name in objectives),
    )
    rows = []
    for run in dataset.runs:
        record = run.record
        outcome = None if record is None else record.outcome
        held = run.incumbent if timeout_score == "incumbent" else None
        source = "result" if run.succeeded else ("worst" if held is None else "incumbent")
        timed = outcome if held is None else held
        failed = run.status == "failed" and bool(run.failures)
        rows.append(
            (
                run.run_id,
                run.protocol,
                run.problem,
                run.solver,
                run.replicate,
                run.seed,
                run.status,
                run.failures[-1].error.kind.value if failed else None,
                source,
                None if outcome is None else outcome.stop_reason.value,
                None if timed is None else timed.evaluations,
                None if outcome is None else outcome.iterations,
                None if timed is None else timed.elapsed,
                None if record is None else record.provenance.cpu_seconds,
                *(
                    run.value(name) if held is None else held.objectives.get(name)
                    for name in objectives
                ),
            )
        )
    return Table("runs", f"Runs of campaign {dataset.campaign}", columns, tuple(rows))


def cells_table(summary: Summary) -> Table:
    """Return the per-problem results of every solver."""
    level = _percent(summary.config.confidence)
    columns = (
        Column("problem", "Problem"),
        Column("solver", "Solver"),
        Column("planned", "Runs", "count"),
        Column("succeeded", "Succeeded", "count"),
        Column("score", f"Score ({summary.config.statistic})", "number"),
        Column("mean", "Mean", "number"),
        Column("mean_ci", f"Mean {level} CI", "interval"),
        Column("median", "Median", "number"),
        Column("std", "Std dev", "number"),
        Column("best", "Best", "number"),
    )
    rows = tuple(
        (
            cell.problem,
            cell.solver,
            cell.planned,
            cell.succeeded,
            cell.score,
            cell.mean,
            (cell.mean_low, cell.mean_high),
            cell.median,
            cell.std,
            cell.best,
        )
        for cell in summary.cells
    )
    kept = (
        ", or, for a run killed at its hard limit, as the feasible incumbent it held when "
        "its time limit passed"
        if summary.config.timeout_score == "incumbent"
        else ""
    )
    notes = (
        f"The score is the {summary.config.statistic} of every planned run, a run without a "
        f"feasible result counting as the worst value{kept}; the other statistics describe "
        "the succeeded runs.",
    )
    return Table("cells", _title("Results per problem", summary), columns, rows, notes)


def solvers_table(summary: Summary) -> Table:
    """Return each solver's reliability and average rank, best rank first."""
    columns = (
        Column("solver", "Solver"),
        Column("planned", "Runs", "count"),
        Column("succeeded", "Succeeded", "count"),
        Column("infeasible", "Infeasible", "count"),
        Column("failed", "Failed", "count"),
        Column("missing", "Missing", "count"),
        Column("success_rate", "Success rate", "number"),
        Column("problems", "Problems", "count"),
        Column("average_rank", "Average rank", "number"),
    )
    order = {name: index for index, name in enumerate(summary.solvers)}
    ranked = sorted(
        summary.solver_summaries,
        key=lambda row: (
            row.average_rank is None,
            row.average_rank or 0.0,
            order[row.solver],
        ),
    )
    rows = tuple(
        (
            row.solver,
            row.planned,
            row.succeeded,
            row.infeasible,
            row.failed,
            row.missing,
            row.success_rate,
            row.problems,
            row.average_rank,
        )
        for row in ranked
    )
    notes = (
        f"Average ranks are taken over the {len(summary.blocks)} problems every solver is "
        "planned on, rank 1 best.",
    )
    return Table("solvers", _title("Solvers", summary), columns, rows, notes)


def pairwise_table(summary: Summary) -> Table:
    """Return every pairwise comparison with its effect sizes and intervals."""
    return _comparison_table(
        "pairwise",
        _title("Pairwise comparisons", summary),
        summary.pairwise,
        summary.config.confidence,
        summary.config.correction,
        summary.objective,
    )


def baselines_table(comparison: BaselineComparison) -> Table:
    """Return the comparisons of a focal solver with its baselines."""
    title = (
        f"{comparison.focal} against baselines ({comparison.objective}, protocol "
        f"{comparison.protocol})"
    )
    confidence = comparison.rows[0].confidence if comparison.rows else 0.95
    table = _comparison_table(
        "baselines",
        title,
        comparison.rows,
        confidence,
        comparison.correction,
        comparison.objective,
    )
    verdict = "is" if comparison.better_than_all else "is not"
    note = (
        f"{comparison.focal} {verdict} significantly better than every baseline at alpha = "
        f"{comparison.alpha:g} after the correction over {len(comparison.rows)} comparisons."
    )
    return Table(table.name, table.title, table.columns, table.rows, (*table.notes, note))


def omnibus_table(summary: Summary) -> Table:
    """Return the omnibus tests with Kendall's W, and the post-hoc test as notes."""
    level = _percent(summary.config.confidence)
    columns = (
        Column("test", "Test"),
        Column("statistic", "Statistic", "number"),
        Column("df", "df"),
        Column("p_value", "p", "p-value"),
        Column("kendalls_w", "Kendall's W", "effect"),
        Column("kendalls_w_ci", f"W {level} CI", "interval"),
        Column("decides", "Decides", "text"),
    )
    omnibus = summary.omnibus
    notes: list[str] = []
    rows: tuple[tuple[Cell, ...], ...] = ()
    if omnibus is None:
        notes.append("No omnibus test was run; see the limitations.")
    else:
        interval = (omnibus.kendalls_w_low, omnibus.kendalls_w_high)
        groups, blocks = omnibus.groups, omnibus.blocks
        rows = (
            (
                "Friedman",
                omnibus.friedman_statistic,
                str(groups - 1),
                omnibus.friedman_p_value,
                omnibus.kendalls_w,
                interval,
                "yes" if omnibus.test == "friedman" else "no",
            ),
            (
                "Iman-Davenport",
                omnibus.iman_davenport_statistic,
                f"{groups - 1}, {(groups - 1) * (blocks - 1)}",
                omnibus.iman_davenport_p_value,
                omnibus.kendalls_w,
                interval,
                "yes" if omnibus.test == "iman-davenport" else "no",
            ),
        )
        notes.append(f"{groups} solvers ranked on {blocks} problems.")
    critical = summary.critical_difference
    if critical is not None:
        different = "; ".join(f"{a} and {b}" for a, b in critical.significant_pairs) or "none"
        groups_text = "; ".join(", ".join(clique) for clique in critical.cliques)
        notes.append(
            f"Nemenyi critical difference {critical.value:.4g} at alpha = {critical.alpha:g}. "
            f"Average ranks that differ by more: {different}. "
            f"Solvers within the critical difference of each other: {groups_text}."
        )
    return Table("omnibus", _title("Omnibus tests", summary), columns, rows, tuple(notes))


def deviation_table(
    rows: Sequence[DeviationRow], objective: str, source: str | None = None
) -> Table:
    """Return the average relative percentage deviation of every solver."""
    columns = (
        Column("solver", "Solver"),
        Column("problems", "Problems", "count"),
        Column("runs", "Runs", "count"),
        Column("planned", "Planned", "count"),
        Column("arpd", "ARPD (%)", "number"),
        Column("rpd_min", "RPD min (%)", "number"),
        Column("rpd_max", "RPD max (%)", "number"),
    )
    body = tuple(
        (row.solver, row.problems, row.runs, row.planned, row.arpd, row.rpd_min, row.rpd_max)
        for row in rows
    )
    notes = ["Deviations are averaged over the succeeded runs; positive is worse."]
    if source:
        notes.append(f"Reference values: {source}.")
    return Table(
        "deviation", f"Relative percentage deviation ({objective})", columns, body, tuple(notes)
    )


def indicators_table(report: IndicatorReport) -> Table:
    """Return every solver's hypervolume, IGD, and IGD+ on every multi-objective problem."""
    columns = (
        Column("problem", "Problem"),
        Column("solver", "Solver"),
        Column("planned", "Runs", "count"),
        Column("succeeded", "Succeeded", "count"),
        Column("hypervolume", "Hypervolume", "number"),
        Column("igd", "IGD", "number"),
        Column("igd_plus", "IGD+", "number"),
    )
    rows = tuple(
        (
            row.problem,
            row.solver,
            row.planned,
            row.succeeded,
            row.hypervolume,
            row.igd,
            row.igd_plus,
        )
        for row in report.rows
    )
    notes = [
        f"The {report.statistic} over the planned runs; a run without a feasible result has "
        "hypervolume 0 and infinite IGD and IGD+."
    ]
    for frame in report.reference_points:
        space = "normalised objectives" if frame.normalized else "objective units"
        point = ", ".join(f"{value:.6g}" for value in frame.point)
        rule = "fixed" if frame.rule == "fixed" else f"{frame.rule} {frame.parameter:g}"
        notes.append(f"{frame.problem}: reference point ({point}), {rule}, {space}.")
    return Table(
        "indicators",
        f"Multi-objective indicators (protocol {report.protocol})",
        columns,
        rows,
        tuple(notes),
    )


# ------------------------------------------------------------------ helpers


def _comparison_table(
    name: str,
    title: str,
    comparisons: Sequence[PairwiseComparison],
    confidence: float,
    correction: str,
    objective: str,
) -> Table:
    level = _percent(confidence)
    columns = (
        Column("left", "Solver"),
        Column("right", "Against"),
        Column("pairs", "Problems", "count"),
        Column("wins", "Wins", "count"),
        Column("losses", "Losses", "count"),
        Column("ties", "Ties", "count"),
        Column("win_share", "Win share", "effect"),
        Column("win_share_ci", f"Win share {level} CI", "interval"),
        Column("median_advantage", "Median advantage", "number"),
        Column("mean_advantage", "Mean advantage", "number"),
        Column("advantage_ci", f"Mean advantage {level} CI", "interval"),
        Column("cohens_dz", "Cohen's d_z", "effect"),
        Column("statistic", "W", "number"),
        Column("p_value", "p", "p-value"),
        Column("adjusted_p_value", f"{_CORRECTION_NAMES.get(correction, correction)} p", "p-value"),
        Column("method", "Null distribution"),
    )
    rows = tuple(
        (
            row.left,
            row.right,
            row.pairs,
            row.wins,
            row.losses,
            row.ties,
            row.win_share,
            (row.win_share_low, row.win_share_high),
            row.median_advantage,
            row.mean_advantage,
            (row.advantage_low, row.advantage_high),
            row.cohens_dz,
            row.statistic,
            row.p_value,
            row.adjusted_p_value,
            row.method,
        )
        for row in comparisons
    )
    notes = (
        "Wilcoxon signed-rank tests over the shared problems. An advantage is how much better "
        f"the solver's score is than the other's, in units of {objective}; a win share above "
        "0.5 and a positive advantage or d_z favour the solver. A failure makes an advantage "
        "infinite, which leaves the mean advantage and d_z undefined.",
    )
    return Table(name, title, columns, rows, notes)


def _title(what: str, summary: Summary) -> str:
    sense = "minimised" if summary.sense == "min" else "maximised"
    return f"{what} ({summary.objective}, {sense}; protocol {summary.protocol})"


def _percent(level: float) -> str:
    return f"{level * 100:g}%"


def _display(value: Cell, kind: ColumnKind) -> str:
    if isinstance(value, tuple):
        low, high = value
        if low is None or high is None:
            return _MISSING
        return f"[{_number(low)}, {_number(high)}]"
    if value is None:
        return _MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if kind == "count" and isinstance(value, int):
        return str(value)
    if kind == "p-value":
        return f"{value:.3g}"
    return _number(value)


def _number(value: float) -> str:
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:.6g}"


def _exact(value: Cell) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _escape_markdown(text: str) -> str:
    return text.replace("|", "\\|")


_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape_latex(text: str) -> str:
    return "".join(_LATEX_SPECIAL.get(character, character) for character in text)


def _latex_value(value: Cell, kind: ColumnKind) -> str:
    if isinstance(value, float) and math.isinf(value):
        return r"$\infty$" if value > 0 else r"$-\infty$"
    text = _display(value, kind)
    if text == _MISSING:
        return "--"
    return _escape_latex(text.replace("∞", "inf"))


def _comment(text: str) -> str:
    return " ".join(text.split())
