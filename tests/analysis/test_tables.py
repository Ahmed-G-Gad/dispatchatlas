"""Tables: the p-value rule, and the Markdown, CSV, and LaTeX renderings."""

from __future__ import annotations

import csv
import io
import math

import pytest

from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.results import Summary
from dispatchatlas.analysis.summary import summarize
from dispatchatlas.analysis.tables import (
    Column,
    Table,
    cells_table,
    check_p_values,
    omnibus_table,
    pairwise_table,
    runs_table,
    solvers_table,
    to_csv,
    to_latex,
    to_markdown,
)
from dispatchatlas.core.errors import ConfigError
from tests.analysis.analysis_data import Entry, make_dataset

CELLS: dict[str, dict[str, list[Entry]]] = {
    "a": {f"p{i}": [10.0 + i, 10.5 + i] for i in range(6)},
    "b_2": {f"p{i}": [11.0 + 1.1 * i, "failed"] for i in range(6)},
    "c": {f"p{i}": [13.0 + 1.3 * i, 13.0] for i in range(6)},
}


def _summary() -> Summary:
    return summarize(make_dataset(CELLS), AnalysisConfig(resamples=200))


def test_a_p_value_needs_an_effect_size_and_an_interval_beside_it():
    bare = Table("bare", "Bare", (Column("p", "p", "p-value"),), ((0.01,),))
    with pytest.raises(ConfigError, match="Carrasco"):
        check_p_values(bare)
    for render in (to_markdown, to_csv, to_latex):
        with pytest.raises(ConfigError):
            render(bare)
    effect_only = Table(
        "half", "Half", (Column("p", "p", "p-value"), Column("e", "Effect", "effect")), ()
    )
    with pytest.raises(ConfigError):
        check_p_values(effect_only)
    summary = _summary()
    for table in (pairwise_table(summary), omnibus_table(summary)):
        check_p_values(table)


def test_markdown_formats_values_and_keeps_notes():
    text = to_markdown(pairwise_table(_summary()))
    lines = text.splitlines()
    assert lines[0].startswith("**Pairwise comparisons (makespan, minimised; protocol p)**")
    assert lines[2].startswith("| Solver | Against | Problems |")
    assert "| --- | --- | ---: |" in lines[3]
    assert "Holm p" in lines[2]
    assert "—" in text  # an undefined mean advantage
    assert text.endswith("\n")
    assert "Wilcoxon signed-rank tests over the shared problems" in text


def test_csv_is_tidy_and_exact():
    table = cells_table(_summary())
    rows = list(csv.DictReader(io.StringIO(to_csv(table))))
    assert len(rows) == 18
    assert {"mean_ci_low", "mean_ci_high", "score"} <= set(rows[0])
    failed = next(row for row in rows if row["solver"] == "b_2")
    assert (failed["score"], failed["succeeded"], failed["std"]) == ("inf", "1", "")
    first = next(row for row in rows if row["solver"] == "a")
    assert float(first["mean"]) == 10.25
    assert repr(float(first["mean"])) == first["mean"]


def test_latex_escapes_text_and_marks_missing_values():
    text = to_latex(pairwise_table(_summary()))
    assert text.startswith("% Pairwise comparisons")
    assert "\\begin{tabular}{llrrrrrrrrrrrrrl}" in text
    assert "b\\_2" in text
    assert "--" in text
    assert text.splitlines()[-1].startswith(
        "% Wilcoxon signed-rank tests over the shared problems."
    )
    infinite = Table("inf", "Inf", (Column("x", "x", "number"),), ((math.inf,), (None,)))
    assert "$\\infty$" in to_latex(infinite)
    assert "∞" in to_markdown(infinite)


def test_solvers_are_listed_best_rank_first_and_runs_one_per_row():
    summary = _summary()
    solvers = list(csv.DictReader(io.StringIO(to_csv(solvers_table(summary)))))
    # b_2 fails one run of every cell, so its mean scores are infinite and it ranks last.
    assert [row["solver"] for row in solvers] == ["a", "c", "b_2"]
    assert (solvers[2]["success_rate"], solvers[2]["average_rank"]) == ("0.5", "3.0")
    dataset = make_dataset(CELLS)
    runs = list(csv.DictReader(io.StringIO(to_csv(runs_table(dataset)))))
    assert len(runs) == len(dataset.runs) == 36
    assert runs[0]["objective.makespan"] == "10.0"
    assert {row["status"] for row in runs} == {"succeeded", "failed"}


def test_a_row_of_the_wrong_width_is_refused():
    with pytest.raises(ConfigError, match="wrong width"):
        Table("t", "T", (Column("a", "A"),), (("x", "y"),))
