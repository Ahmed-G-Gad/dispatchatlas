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
    late: float | None = None,
) -> FailureRecord:
    """Return a failed attempt holding ``value`` at its time limit and ``late`` after it."""
    held = None
    if value is not None:
        held = Incumbent("makespan", value, {"makespan": value}, feasible, 12, 3, 0.4)
    last = (
        held if late is None else Incumbent("makespan", late, {"makespan": late}, True, 40, 9, 2.5)
    )
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
        last_incumbent=last,
        limit_incumbent=held,
    )


PROBLEMS = tuple(f"p{index}" for index in range(1, 7))


def _killed_dataset() -> Dataset:
    """Solver ``a`` completes; ``b`` fails in a different way on every problem."""
    dataset = make_dataset(
        {
            "a": {problem: [20.0, 20.0] for problem in PROBLEMS},
            "b": {problem: [10.0, "failed"] for problem in PROBLEMS},
        }
    )
    kinds: dict[str, tuple[ErrorKind, float | None, bool, float | None]] = {
        "p1": (ErrorKind.TIMEOUT, 12.0, True, None),  # killed, holding a feasible incumbent
        "p2": (ErrorKind.TIMEOUT, 14.0, True, 8.0),  # and a better one after its time limit
        "p3": (ErrorKind.TIMEOUT, None, True, 7.0),  # killed, holding one after the limit only
        "p4": (ErrorKind.TIMEOUT, 9.0, False, None),  # killed, holding an infeasible incumbent
        "p5": (ErrorKind.STARTUP, None, True, None),  # never began its solve
        "p6": (ErrorKind.TIMEOUT, None, True, None),  # killed, nothing reported
    }
    runs = []
    for item in dataset.runs:
        if item.status == "failed":
            kind, value, feasible, late = kinds[item.problem]
            failure = _failure(dataset, item, kind, value, feasible=feasible, late=late)
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
    assert [failed[problem].killed for problem in PROBLEMS] == [True, True, True, True, False, True]
    assert [failed[problem].unstarted for problem in PROBLEMS] == [False] * 4 + [True, False]
    held = [failed[problem].incumbent for problem in PROBLEMS]
    assert [None if item is None else item.value for item in held] == [12.0, 14.0] + [None] * 4
    assert [failed[problem].late_incumbent for problem in PROBLEMS] == [False, True, True] + (
        [False] * 3
    )
    first = failed["p1"]
    retried = _failure(dataset, first, ErrorKind.EXCEPTION, attempt=2)
    assert not replace(first, failures=(*first.failures, retried)).killed
    completed = next(item for item in dataset.runs if item.succeeded)
    assert not replace(completed, failures=first.failures).killed
    assert replace(completed, failures=first.failures).incumbent is None
    assert not replace(completed, failures=failed["p2"].failures).late_incumbent
    assert not replace(completed, failures=failed["p5"].failures).unstarted


def test_a_record_without_an_incumbent_at_the_limit_is_scored_as_the_worst_value() -> None:
    dataset = _killed_dataset()
    (item,) = [run for run in dataset.cell("b", "p1", "p") if run.status == "failed"]
    document = item.failures[-1].to_dict()
    del document["limit_incumbent"]
    earlier = replace(item, failures=(FailureRecord.from_dict(document),))
    assert earlier.killed
    assert earlier.incumbent is None
    assert run_scores([earlier], "makespan", "min", "incumbent") == [math.inf]


def test_run_scores_use_the_incumbent_only_when_asked() -> None:
    dataset = _killed_dataset()
    cell = dataset.cell("b", "p1", "p")
    assert run_scores(cell, "makespan", "min") == [10.0, math.inf]
    assert run_scores(cell, "makespan", "min", "worst") == [10.0, math.inf]
    assert run_scores(cell, "makespan", "min", "incumbent") == [10.0, 12.0]
    assert run_scores(cell[1:], "total_tardiness", "min", "incumbent") == [math.inf]
    for problem in ("p3", "p4", "p5", "p6"):
        scores = run_scores(dataset.cell("b", problem, "p"), "makespan", "min", "incumbent")
        assert scores == [10.0, math.inf]
    assert run_scores(dataset.cell("b", "p6", "p"), "makespan", "max", "incumbent")[1] == -math.inf


def test_an_incumbent_found_after_the_time_limit_is_never_a_score() -> None:
    dataset = _killed_dataset()
    # On p2 the run held 14 at its time limit and found 8 afterwards; on p3 it found 7 afterwards.
    assert run_scores(dataset.cell("b", "p2", "p"), "makespan", "min", "incumbent") == [10.0, 14.0]
    assert run_scores(dataset.cell("b", "p3", "p"), "makespan", "min", "incumbent") == [
        10.0,
        math.inf,
    ]


def test_the_summary_scores_killed_runs_as_the_worst_value_by_default() -> None:
    summary = summarize(_killed_dataset())
    scores = {cell.problem: cell.score for cell in summary.cells if cell.solver == "b"}
    assert set(scores.values()) == {math.inf}
    assert {cell.succeeded for cell in summary.cells if cell.solver == "b"} == {1}
    limitations = {item.topic: item.message for item in summary.limitations}
    assert limitations["failures"] == (
        "6 runs failed and 0 completed without a feasible schedule; each counts as the worst value."
    )
    assert limitations["startup"] == (
        "1 of the failed runs ended at the start-up limit: their worker did not begin the "
        "solve in time, which says nothing about their solver. They count as the worst value "
        "until another attempt completes; running the campaign again with retry_failed "
        "(--retry-failed on the command line) gives them one."
    )
    assert limitations["timeouts"] == (
        "5 runs were killed at their hard limit and never completed, 2 of them holding a "
        "feasible incumbent when their time limit passed; with timeout_score 'worst' each "
        "counts as the worst value. 2 of the killed runs reported an incumbent after their "
        "time limit, which no score uses."
    )
    (row,) = [item for item in summary.pairwise if {item.left, item.right} == {"a", "b"}]
    assert max(row.wins, row.losses) == 6


def test_the_summary_scores_killed_runs_by_their_incumbent_when_asked() -> None:
    summary = summarize(_killed_dataset(), AnalysisConfig(timeout_score="incumbent"))
    scores = {cell.problem: cell.score for cell in summary.cells if cell.solver == "b"}
    assert scores == {"p1": 11.0, "p2": 12.0} | dict.fromkeys(PROBLEMS[2:], math.inf)
    cells = {cell.problem: cell for cell in summary.cells if cell.solver == "b"}
    assert (cells["p1"].succeeded, cells["p1"].mean, cells["p1"].best) == (1, 10.0, 10.0)
    limitations = {item.topic: item.message for item in summary.limitations}
    assert limitations["failures"] == (
        "6 runs failed and 0 completed without a feasible schedule; each counts as the worst "
        "value, except 2 killed runs scored by their incumbent."
    )
    assert limitations["startup"].startswith("1 of the failed runs ended at the start-up limit")
    assert limitations["timeouts"] == (
        "5 runs were killed at their hard limit and never completed, 2 of them holding a "
        "feasible incumbent when their time limit passed; with timeout_score 'incumbent' a "
        "run holding one is scored by it, a value its solver reported and no schedule "
        "confirms, and the others count as the worst value. 2 of the killed runs reported an "
        "incumbent after their time limit, which no score uses."
    )
    (row,) = [item for item in summary.pairwise if {item.left, item.right} == {"a", "b"}]
    assert sorted((row.wins, row.losses)) == [2, 4]
    assert "incumbent it held when its time limit passed" in cells_table(summary).notes[0]
    assert "incumbent" not in cells_table(summarize(_killed_dataset())).notes[0]


def test_a_campaign_without_killed_runs_reports_no_timeouts() -> None:
    dataset = make_dataset({"a": {"p1": [5.0, "failed"]}, "b": {"p1": [6.0, 7.0]}})
    for config in (AnalysisConfig(), AnalysisConfig(timeout_score="incumbent")):
        summary = summarize(dataset, config)
        topics = [item.topic for item in summary.limitations]
        assert "timeouts" not in topics
        assert "startup" not in topics
        assert "failures" in topics


def test_the_runs_table_marks_how_every_run_is_scored() -> None:
    dataset = _killed_dataset()

    def rows(timeout_score: str) -> dict[str, dict[str, str]]:
        table = runs_table(dataset, timeout_score)  # type: ignore[arg-type]
        listed = csv.DictReader(io.StringIO(to_csv(table)))
        return {row["problem"]: row for row in listed if row["solver"] == "b" and row["failure"]}

    default = rows("worst")
    assert {row["scored_by"] for row in default.values()} == {"worst"}
    for column in ("objective.makespan", "evaluations", "elapsed"):
        assert {row[column] for row in default.values()} == {""}
    assert [default[problem]["failure"] for problem in PROBLEMS] == ["timeout"] * 4 + [
        "startup",
        "timeout",
    ]
    asked = rows("incumbent")
    assert [asked[problem]["scored_by"] for problem in PROBLEMS] == ["incumbent"] * 2 + (
        ["worst"] * 4
    )
    # The incumbent held at the time limit, and when its search found it: on p2 not the
    # better one found after the limit.
    for column, held in (
        ("objective.makespan", ["12.0", "14.0"]),
        ("evaluations", ["12", "12"]),
        ("elapsed", ["0.4", "0.4"]),
    ):
        assert [asked[problem][column] for problem in PROBLEMS] == held + [""] * 4
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
        assert message.endswith(
            "0 of the killed runs reported an incumbent after their time limit, "
            "which no score uses."
        )


def test_a_run_that_searched_past_its_time_limit_is_scored_by_what_it_held_at_the_limit(
    tmp_path: Path,
) -> None:
    # Both solvers evaluate the same two orders, the worse one first. One evaluates both at
    # once and returns, within the time limit of 0.2 s; the other finds the better one only
    # at 0.35 s, ignores its limit, and is killed.
    campaign = tiny_campaign(
        solvers=(
            SolverSpec("test-overrun", "on-time", {"after": 0.0, "seconds": 0.0}),
            SolverSpec("test-overrun", "overrun", {"after": 0.35, "seconds": 30.0}),
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
    on_time, overrun = dataset.runs
    assert (on_time.status, overrun.status, overrun.killed) == ("succeeded", "failed", True)
    best = on_time.value("makespan")
    last, held = overrun.failures[-1].last_incumbent, overrun.incumbent
    assert best is not None
    assert last is not None
    assert held is not None
    assert last.value == best < held.value
    assert held.elapsed <= 0.2 < last.elapsed
    assert overrun.late_incumbent

    asked = summarize(dataset, AnalysisConfig(timeout_score="incumbent"))
    scores = {cell.solver: cell.score for cell in asked.cells}
    assert scores == {"on-time": best, "overrun": held.value}
    listed = csv.DictReader(io.StringIO(to_csv(runs_table(dataset, "incumbent"))))
    (row,) = [item for item in listed if item["solver"] == "overrun"]
    assert (row["scored_by"], float(row["objective.makespan"])) == ("incumbent", held.value)
    assert (int(row["evaluations"]), float(row["elapsed"])) == (held.evaluations, held.elapsed)
    for report in (summarize(dataset), asked):
        (message,) = [item.message for item in report.limitations if item.topic == "timeouts"]
        assert message.endswith(
            "1 of the killed runs reported an incumbent after their time limit, which no "
            "score uses."
        )
