"""Runs killed at their hard limit: scored as the worst value or by their incumbent."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.analysis import AnalysisConfig, load_dataset, summarize
from dispatchatlas.analysis._scope import run_scores
from dispatchatlas.analysis.dataset import Dataset, Run
from dispatchatlas.analysis.tables import cells_table, runs_table, to_csv
from dispatchatlas.campaigns import FamilySource, Protocol, SolverSpec, run
from dispatchatlas.campaigns.records import (
    Category,
    ErrorKind,
    FailureRecord,
    Incumbent,
    ProblemRef,
    Provenance,
    RunError,
)
from dispatchatlas.core import ConfigError, StopCriteria
from tests.analysis.analysis_data import make_dataset
from tests.campaigns.solvers import registry, tiny_campaign

INCUMBENT = "incumbent"
WORST = "worst"


def _failure(
    dataset: Dataset,
    item: Run,
    kind: ErrorKind,
    value: float | None = None,
    *,
    feasible: bool = True,
    attempt: int = 1,
) -> FailureRecord:
    held = None
    if value is not None:
        held = Incumbent("makespan", value, {"makespan": value}, feasible, 12, 3, 0.4)
    solver = next(entry for entry in dataset.solvers if entry.name == item.solver)
    return FailureRecord(
        run_id=item.run_id,
        campaign="synthetic",
        config_hash="0" * 64,
        problem=ProblemRef(item.problem, None, "0" * 64),
        solver=solver,
        protocol=dataset.protocols[0],
        replicate=item.replicate,
        seed=item.seed,
        category=Category.RECOVERABLE if kind is ErrorKind.STARTUP else Category.TERMINAL,
        error=RunError(kind, "ended"),
        provenance=Provenance(attempt, 1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:06Z", 6.0),
        solve_started=kind is not ErrorKind.STARTUP,
        last_incumbent=held,
    )


def _killed_dataset() -> Dataset:
    """Solver ``a`` completes; ``b`` fails in a different way on every problem."""
    dataset = make_dataset(
        {
            "a": {f"p{index}": [20.0, 20.0] for index in range(1, 6)},
            "b": {f"p{index}": [10.0, "failed"] for index in range(1, 6)},
        }
    )
    kinds = {
        "p1": (ErrorKind.TIMEOUT, 12.0, True),  # killed, holding a feasible incumbent
        "p2": (ErrorKind.TIMEOUT, 14.0, True),
        "p3": (ErrorKind.TIMEOUT, None, True),  # killed, nothing reported
        "p4": (ErrorKind.TIMEOUT, 9.0, False),  # killed, holding an infeasible incumbent
        "p5": (ErrorKind.STARTUP, None, True),  # never began its solve
    }
    runs = []
    for item in dataset.runs:
        if item.status == "failed":
            kind, value, feasible = kinds[item.problem]
            failure = _failure(dataset, item, kind, value, feasible=feasible)
            item = replace(item, failures=(failure,))  # noqa: PLW2901
        runs.append(item)
    return replace(dataset, runs=tuple(runs))


def test_the_setting_is_optional_and_checked() -> None:
    assert AnalysisConfig().timeout_score == WORST
    assert AnalysisConfig().to_dict()["timeout_score"] == WORST
    assert AnalysisConfig(timeout_score="incumbent").timeout_score == INCUMBENT
    with pytest.raises(ConfigError, match="timeout_score must be one of"):
        AnalysisConfig(timeout_score="best")  # type: ignore[arg-type]


def test_a_killed_run_is_known_by_its_last_failed_attempt() -> None:
    dataset = _killed_dataset()
    failed = {item.problem: item for item in dataset.runs if item.status == "failed"}
    assert [failed[f"p{index}"].killed for index in range(1, 6)] == [True, True, True, True, False]
    held = [failed[f"p{index}"].incumbent for index in range(1, 6)]
    assert [None if item is None else item.value for item in held] == [12.0, 14.0, None, None, None]
    first = failed["p1"]
    retried = _failure(dataset, first, ErrorKind.EXCEPTION, attempt=2)
    assert not replace(first, failures=(*first.failures, retried)).killed
    completed = next(item for item in dataset.runs if item.succeeded)
    assert not replace(completed, failures=first.failures).killed
    assert replace(completed, failures=first.failures).incumbent is None


def test_run_scores_use_the_incumbent_only_when_asked() -> None:
    dataset = _killed_dataset()
    cell = dataset.cell("b", "p1", "p")
    assert run_scores(cell, "makespan", "min") == [10.0, math.inf]
    assert run_scores(cell, "makespan", "min", "worst") == [10.0, math.inf]
    assert run_scores(cell, "makespan", "min", "incumbent") == [10.0, 12.0]
    assert run_scores(cell[1:], "total_tardiness", "min", "incumbent") == [math.inf]
    for problem in ("p3", "p4", "p5"):
        scores = run_scores(dataset.cell("b", problem, "p"), "makespan", "min", "incumbent")
        assert scores == [10.0, math.inf]
    assert run_scores(dataset.cell("b", "p3", "p"), "makespan", "max", "incumbent")[1] == -math.inf


def test_the_summary_scores_killed_runs_as_the_worst_value_by_default() -> None:
    summary = summarize(_killed_dataset())
    scores = {cell.problem: cell.score for cell in summary.cells if cell.solver == "b"}
    assert set(scores.values()) == {math.inf}
    assert {cell.succeeded for cell in summary.cells if cell.solver == "b"} == {1}
    limitations = {item.topic: item.message for item in summary.limitations}
    assert limitations["failures"] == (
        "5 runs failed and 0 completed without a feasible schedule; each counts as the worst value."
    )
    assert limitations["timeouts"] == (
        "4 runs were killed at their hard limit and never completed, 2 of them holding a "
        "feasible incumbent; with timeout_score 'worst' each counts as the worst value."
    )
    (row,) = [item for item in summary.pairwise if {item.left, item.right} == {"a", "b"}]
    assert max(row.wins, row.losses) == 5


def test_the_summary_scores_killed_runs_by_their_incumbent_when_asked() -> None:
    summary = summarize(_killed_dataset(), AnalysisConfig(timeout_score="incumbent"))
    scores = {cell.problem: cell.score for cell in summary.cells if cell.solver == "b"}
    assert scores == {"p1": 11.0, "p2": 12.0, "p3": math.inf, "p4": math.inf, "p5": math.inf}
    cells = {cell.problem: cell for cell in summary.cells if cell.solver == "b"}
    assert (cells["p1"].succeeded, cells["p1"].mean, cells["p1"].best) == (1, 10.0, 10.0)
    limitations = {item.topic: item.message for item in summary.limitations}
    assert limitations["failures"] == (
        "5 runs failed and 0 completed without a feasible schedule; each counts as the worst "
        "value, except 2 killed runs scored by their incumbent."
    )
    assert limitations["timeouts"] == (
        "4 runs were killed at their hard limit and never completed, 2 of them holding a "
        "feasible incumbent; with timeout_score 'incumbent' those 2 are scored by that "
        "incumbent, a value their solver reported and no schedule confirms, and the others "
        "count as the worst value."
    )
    (row,) = [item for item in summary.pairwise if {item.left, item.right} == {"a", "b"}]
    assert sorted((row.wins, row.losses)) == [2, 3]
    assert "as the feasible incumbent it held" in cells_table(summary).notes[0]
    assert "incumbent" not in cells_table(summarize(_killed_dataset())).notes[0]


def test_a_campaign_without_killed_runs_reports_no_timeouts() -> None:
    dataset = make_dataset({"a": {"p1": [5.0, "failed"]}, "b": {"p1": [6.0, 7.0]}})
    for config in (AnalysisConfig(), AnalysisConfig(timeout_score="incumbent")):
        summary = summarize(dataset, config)
        topics = [item.topic for item in summary.limitations]
        assert "timeouts" not in topics
        assert "failures" in topics


def test_the_runs_table_marks_how_every_run_is_scored() -> None:
    dataset = _killed_dataset()

    def rows(timeout_score: str) -> dict[str, dict[str, str]]:
        table = runs_table(dataset, timeout_score)  # type: ignore[arg-type]
        listed = csv.DictReader(io.StringIO(to_csv(table)))
        return {row["problem"]: row for row in listed if row["solver"] == "b" and row["failure"]}

    default = rows("worst")
    assert {row["scored_by"] for row in default.values()} == {"worst"}
    assert {row["objective.makespan"] for row in default.values()} == {""}
    assert [default[f"p{index}"]["failure"] for index in range(1, 6)] == [
        "timeout",
        "timeout",
        "timeout",
        "timeout",
        "startup",
    ]
    asked = rows("incumbent")
    assert [asked[f"p{index}"]["scored_by"] for index in range(1, 6)] == [
        "incumbent",
        "incumbent",
        "worst",
        "worst",
        "worst",
    ]
    assert [asked[f"p{index}"]["objective.makespan"] for index in range(1, 6)] == [
        "12.0",
        "14.0",
        "",
        "",
        "",
    ]
    assert {row["status"] for row in asked.values()} == {"failed"}
    completed = [
        row
        for row in csv.DictReader(io.StringIO(to_csv(runs_table(dataset))))
        if row["status"] == "succeeded"
    ]
    assert {(row["scored_by"], row["failure"]) for row in completed} == {("result", "")}


def test_a_killed_run_of_a_real_campaign_is_scored_either_way(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(
            SolverSpec("test-stalled", "prompt", {"seconds": 0.0}),
            SolverSpec("test-stalled", "stalled", {"seconds": 30.0}),
        ),
        protocols=(Protocol("clock", StopCriteria(time_limit=0.2, max_evaluations=10)),),
        instances=(FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),),
        replicates=1,
        backend="process",
        grace=0.3,
    )
    summary = run(campaign, tmp_path, registry=registry())
    assert (summary.completed, summary.failed) == (1, 1)
    dataset = load_dataset(summary.directory)
    prompt, stalled = dataset.runs
    assert (prompt.status, stalled.status, stalled.killed) == ("succeeded", "failed", True)
    value = prompt.value("makespan")
    assert value is not None
    assert stalled.incumbent is not None
    assert stalled.incumbent.objectives == prompt.objectives

    default = {cell.solver: cell.score for cell in summarize(dataset).cells}
    assert default == {"prompt": value, "stalled": math.inf}
    asked = summarize(dataset, AnalysisConfig(timeout_score="incumbent"))
    assert {cell.solver: cell.score for cell in asked.cells} == {"prompt": value, "stalled": value}
    for report in (summarize(dataset), asked):
        (message,) = [item.message for item in report.limitations if item.topic == "timeouts"]
        assert message.startswith(
            "1 runs were killed at their hard limit and never completed, 1 of them holding"
        )
