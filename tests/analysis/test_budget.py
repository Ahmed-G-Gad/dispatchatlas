"""Realised time against the budget: the runs table and the budget, timing, and failures topics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

from dispatchatlas.analysis import AnalysisConfig, load_dataset, summarize
from dispatchatlas.analysis.dataset import Dataset
from dispatchatlas.analysis.tables import runs_table
from dispatchatlas.campaigns import Category, Protocol, SolverSpec, run
from dispatchatlas.campaigns.records import ErrorKind, FailureRecord, RunError
from dispatchatlas.core import ConfigError, StopCriteria
from tests.analysis.analysis_data import Entry, make_dataset
from tests.campaigns.solvers import registry, tiny_campaign

LIMIT = 0.1


@pytest.fixture(scope="module")
def late_campaign(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A worker runs a solver that ends 0.3 s after a 0.1 s limit, inside the grace of the kill."""
    campaign = tiny_campaign(
        "late",
        solvers=(SolverSpec("test-tardy", parameters={"seconds": 0.3}), SolverSpec("input-order")),
        protocols=(Protocol("clock", StopCriteria(time_limit=LIMIT)),),
        replicates=1,
        backend="process",
        workers=1,
    )
    summary = run(campaign, tmp_path_factory.mktemp("late"), registry=registry())
    assert (summary.completed, summary.failed) == (4, 0)
    return load_dataset(summary.directory)


def _rows(dataset: Dataset) -> list[dict[str, object]]:
    table = runs_table(dataset)
    names = [column.key for column in table.columns]
    return [dict(zip(names, row, strict=True)) for row in table.rows]


def test_a_run_that_ends_late_inside_the_grace_shows_its_overrun(late_campaign: Dataset) -> None:
    rows = _rows(late_campaign)
    assert {row["status"] for row in rows} == {"succeeded"}
    late = [row for row in rows if row["solver"] == "test-tardy"]
    on_time = [row for row in rows if row["solver"] == "input-order"]
    assert len(late) == len(on_time) == 2
    for row in late:
        record = late_campaign.runs[rows.index(row)].record
        assert record is not None
        # The kill was armed, and the run ended before it: an ordinary success.
        assert record.provenance.hard_limit is not None
        assert record.provenance.wall_seconds < record.provenance.hard_limit
        assert row["time_limit"] == LIMIT
        assert isinstance(row["wall_seconds"], float)
        assert row["wall_seconds"] >= 0.3
        assert row["overrun_seconds"] == row["wall_seconds"] - LIMIT
        assert isinstance(row["cpu_share"], float)
        assert 0.0 <= row["cpu_share"] < 0.9
        assert row["concurrent"] == 1
        assert row["attempts"] == 1
        assert isinstance(row["elapsed"], float)
        assert row["elapsed"] < row["wall_seconds"]
    for row in on_time:
        assert isinstance(row["overrun_seconds"], float)
        assert row["overrun_seconds"] < 0.0


def test_the_budget_limitation_names_the_solver_that_ran_late(late_campaign: Dataset) -> None:
    summary = summarize(late_campaign)
    (budget,) = [item for item in summary.limitations if item.topic == "budget"]
    assert "Under protocol clock, whose time limit is 0.1 s" in budget.message
    assert "overrun_tolerance 0.05" in budget.message
    assert "test-tardy in 2 of 2 (up to " in budget.message
    assert "input-order" not in budget.message
    (timing,) = [item for item in summary.limitations if item.topic == "timing"]
    # The two runs of the rule end within microseconds and are left out.
    assert "Over the 2 completed runs that lasted at least half" in timing.message
    assert "between 1 and 1 runs executed at once" in timing.message
    # A tolerance wide enough for the late runs leaves nothing to name.
    lenient = summarize(late_campaign, AnalysisConfig(overrun_tolerance=100.0))
    assert "budget" not in {item.topic for item in lenient.limitations}


def _timed(
    dataset: Dataset,
    walls: Mapping[str, Sequence[float]],
    *,
    cpu: Mapping[str, float] | None = None,
    concurrent: Mapping[str, int] | None = None,
    stop: StopCriteria = StopCriteria(time_limit=1.0),  # noqa: B008 - immutable
) -> Dataset:
    """Give the protocol a time limit of 1 s and each solver's runs the listed wall times."""
    protocol = Protocol("p", stop)
    seen: dict[str, int] = {}
    runs = []
    for item in dataset.runs:
        record = item.record
        if record is not None:
            index = seen.get(item.solver, 0)
            seen[item.solver] = index + 1
            wall = walls[item.solver][index]
            provenance = replace(
                record.provenance,
                wall_seconds=wall,
                cpu_seconds=None if cpu is None else cpu[item.solver] * wall,
                concurrent=None if concurrent is None else concurrent[item.solver],
            )
            record = replace(record, protocol=protocol, provenance=provenance)
        runs.append(replace(item, record=record))
    plan = replace(dataset.plan, protocols=(protocol,))
    return replace(dataset, plan=plan, runs=tuple(runs), protocols=(protocol,))


CELLS = {"a": {"p1": [5.0, 5.0], "p2": [5.0, 5.0]}, "b": {"p1": [6.0, 6.0], "p2": [6.0, 6.0]}}


def test_the_budget_limitation_counts_runs_beyond_the_tolerance_and_states_the_largest_ratio() -> (
    None
):
    walls = {"a": [1.0, 1.04, 1.05, 0.2], "b": [1.06, 4.02, 1.0, 2.5]}
    dataset = _timed(make_dataset(CELLS), walls)
    summary = summarize(dataset)
    (budget,) = [item for item in summary.limitations if item.topic == "budget"]
    assert budget.message == (
        "Under protocol p, whose time limit is 1 s, completed runs took more wall time than "
        "the limit with overrun_tolerance 0.05 allows: b in 3 of 4 (up to 4.02 times the limit). "
        "These runs searched for longer than the protocol states, and their results are "
        "compared as they are."
    )
    # With no share allowed, the clock margin of the engine still applies: the runs of
    # 1.04 s and 1.05 s are within 0.05 s of the limit and are not named.
    strict = summarize(dataset, AnalysisConfig(overrun_tolerance=0.0))
    (budget,) = [item for item in strict.limitations if item.topic == "budget"]
    assert "a in" not in budget.message
    assert "b in 3 of 4 (up to 4.02 times the limit)" in budget.message
    wide = summarize(dataset, AnalysisConfig(overrun_tolerance=2.0))
    (budget,) = [item for item in wide.limitations if item.topic == "budget"]
    assert "b in 1 of 4 (up to 4.02 times the limit)" in budget.message
    rows = _rows(dataset)
    assert [row["overrun_seconds"] for row in rows if row["solver"] == "b"] == pytest.approx(
        [0.06, 3.02, 0.0, 1.5]
    )
    assert {row["time_limit"] for row in rows} == {1.0}


def test_a_campaign_inside_its_limits_reports_no_budget_limitation() -> None:
    walls = {"a": [0.99, 1.0, 1.01, 1.04], "b": [1.0, 1.0, 1.0, 1.05]}
    summary = summarize(_timed(make_dataset(CELLS), walls))
    assert "budget" not in {item.topic for item in summary.limitations}
    # A protocol without a time limit has no budget to compare with, and no timing to report.
    untimed = summarize(make_dataset(CELLS))
    assert {"budget", "timing"}.isdisjoint(item.topic for item in untimed.limitations)
    assert {row["time_limit"] for row in _rows(make_dataset(CELLS))} == {None}
    assert {row["overrun_seconds"] for row in _rows(make_dataset(CELLS))} == {None}


def test_the_timing_limitation_reports_the_processor_share_and_the_concurrency() -> None:
    walls = {"a": [1.0, 1.0, 1.0, 1.0], "b": [1.0, 1.0, 1.0, 1.0]}
    dataset = _timed(
        make_dataset(CELLS), walls, cpu={"a": 0.98, "b": 0.5}, concurrent={"a": 2, "b": 8}
    )
    summary = summarize(dataset)
    (timing,) = [item for item in summary.limitations if item.topic == "timing"]
    assert timing.message == (
        "Protocol p gives every run the same wall-clock time, so a run searches as far as the "
        "processor time it received in that time allows. Over the 8 completed runs that lasted "
        "at least half of the time limit, the processor share of a run, the processor seconds "
        "of its own process over its wall seconds, lies between 0.50 and 0.98 with median 0.74, "
        "and its median per solver between 0.50 (b) and 0.98 (a); between 2 and 8 runs executed "
        "at once, with median 5."
    )
    rows = _rows(dataset)
    assert {row["cpu_share"] for row in rows if row["solver"] == "b"} == {0.5}
    assert {row["concurrent"] for row in rows if row["solver"] == "b"} == {8}
    # Records that state neither time nor concurrency leave nothing to report.
    silent = summarize(_timed(make_dataset(CELLS), walls))
    assert "timing" not in {item.topic for item in silent.limitations}
    # A run that ended long before the limit is left out: the clock did not bound it.
    early = {"a": [1.0, 0.5, 0.49, 0.01], "b": [1.0, 1.0, 1.0, 1.0]}
    partly = summarize(_timed(make_dataset(CELLS), early, cpu={"a": 0.98, "b": 0.5}))
    (timing,) = [item for item in partly.limitations if item.topic == "timing"]
    assert "Over the 6 completed runs that lasted at least half of the time limit" in timing.message
    none = {"a": [0.01] * 4, "b": [0.02] * 4}
    quick = summarize(_timed(make_dataset(CELLS), none, cpu={"a": 0.98, "b": 0.5}))
    assert "timing" not in {item.topic for item in quick.limitations}


@pytest.mark.parametrize("bound", [{"max_evaluations": 50}, {"max_iterations": 50}])
def test_the_timing_limitation_of_a_protocol_with_another_bound_does_not_call_the_time_equal(
    bound: dict[str, int],
) -> None:
    walls = {"a": [1.0, 1.0, 1.0, 0.1], "b": [1.0, 1.0, 1.0, 1.0]}
    stop = StopCriteria.from_dict({"time_limit": 1.0, **bound})
    dataset = _timed(make_dataset(CELLS), walls, cpu={"a": 0.98, "b": 0.5}, stop=stop)
    (timing,) = [item for item in summarize(dataset).limitations if item.topic == "timing"]
    assert "same wall-clock time" not in timing.message
    assert timing.message.startswith(
        "Protocol p bounds a run by a time limit of 1 s beside its other limits, so a run that "
        "the time limit stops searched as far as the processor time it received allows. Over "
        "the 7 completed runs that lasted at least half of the time limit, the processor share"
    )
    # Patience and a target bound no run in advance: the time limit stays the sole bound.
    sole = StopCriteria(time_limit=1.0, patience=5, target=0.0)
    dataset = _timed(make_dataset(CELLS), walls, cpu={"a": 0.98, "b": 0.5}, stop=sole)
    (timing,) = [item for item in summarize(dataset).limitations if item.topic == "timing"]
    assert timing.message.startswith("Protocol p gives every run the same wall-clock time, so")


def _clocked(dataset: Dataset, stated: Sequence[bool | None]) -> Dataset:
    """Give the completed runs, in turn, the listed statements about their solver's clock."""
    marks = iter(stated)
    runs = []
    for item in dataset.runs:
        record = item.record
        if record is not None:
            provenance = replace(record.provenance, shared_clock=next(marks))
            record = replace(record, provenance=provenance)
        runs.append(replace(item, record=record))
    return replace(dataset, runs=tuple(runs))


def test_the_clock_limitation_reports_records_that_do_not_all_state_a_shared_clock() -> None:
    walls = {"a": [1.0] * 4, "b": [1.0] * 4}
    timed = _timed(make_dataset(CELLS), walls)
    mixed = summarize(_clocked(timed, [True, True, True, None, None, False, True, True]))
    (clock,) = [item for item in mixed.limitations if item.topic == "clock"]
    assert clock.message.startswith(
        "Under protocol p, 5 of the 8 completed runs state that the clock of their solver "
        "started with the engine's, at the start of the attempt, and 3 do not state where it "
        "started."
    )
    # Records that agree, all stating it or none, leave nothing to report.
    for stated in ([True] * 8, [None] * 8):
        agreed = summarize(_clocked(timed, stated))
        assert "clock" not in {item.topic for item in agreed.limitations}
    # Without a time limit no clock bounds a run.
    untimed = summarize(_clocked(make_dataset(CELLS), [True] * 4 + [None] * 4))
    assert "clock" not in {item.topic for item in untimed.limitations}


def test_a_campaign_states_the_shared_clock_of_every_completed_run(late_campaign: Dataset) -> None:
    records = [run.record for run in late_campaign.runs if run.record is not None]
    assert len(records) == 4
    assert all(record.provenance.shared_clock is True for record in records)
    assert "clock" not in {item.topic for item in summarize(late_campaign).limitations}


def _failed(
    dataset: Dataset, kinds: Mapping[str, Sequence[ErrorKind]], first: ErrorKind | None = None
) -> Dataset:
    """Give every failed run of a solver the next of its listed kinds as its last failure.

    With ``first``, every such run holds an earlier failed attempt of that kind.
    """
    seen: dict[str, int] = {}
    runs = []
    for item in dataset.runs:
        if item.status == "failed":
            index = seen.get(item.solver, 0)
            seen[item.solver] = index + 1
            done = next(run.record for run in dataset.runs if run.record is not None)
            failure = FailureRecord(
                run_id=item.run_id,
                campaign="synthetic",
                config_hash="0" * 64,
                problem=done.problem,
                solver=next(entry for entry in dataset.solvers if entry.name == item.solver),
                protocol=dataset.protocols[0],
                replicate=item.replicate,
                seed=item.seed,
                category=Category.TERMINAL,
                error=RunError(kinds[item.solver][index], "ended"),
                provenance=replace(done.provenance, attempt=2, wall_seconds=3.0),
            )
            earlier = () if first is None else (_earlier(failure, first),)
            item = replace(item, failures=(*earlier, failure))  # noqa: PLW2901
        runs.append(item)
    return replace(dataset, runs=tuple(runs))


def _earlier(failure: FailureRecord, kind: ErrorKind) -> FailureRecord:
    """Return the first attempt of a run whose second attempt is ``failure``."""
    provenance = replace(failure.provenance, attempt=1, wall_seconds=0.5)
    return replace(
        failure,
        category=Category.RECOVERABLE,
        error=RunError(kind, "ended"),
        provenance=provenance,
    )


def test_the_failures_limitation_names_the_error_kinds_per_solver() -> None:
    cells: Mapping[str, Mapping[str, Sequence[Entry]]] = {
        "a": {"p1": ["failed", "failed"], "p2": ["failed", 5.0]},
        "b": {"p1": [6.0, "failed"], "p2": [6.0, 6.0]},
        "c": {"p1": [7.0, 7.0], "p2": [7.0, 7.0]},
    }
    kinds = {
        "a": [ErrorKind.TIMEOUT, ErrorKind.EXCEPTION, ErrorKind.TIMEOUT],
        "b": [ErrorKind.WORKER_EXIT],
    }
    dataset = _failed(make_dataset(cells), kinds)
    summary = summarize(dataset)
    (failures,) = [item for item in summary.limitations if item.topic == "failures"]
    assert failures.message == (
        "4 runs failed and 0 completed without a feasible schedule; each counts as the worst "
        "value. The failed runs by solver and by the kind of error of their last attempt: "
        "a 1 exception, 2 timeout; b 1 worker-exit. What each kind means: exception: the "
        "solver, its warm-up, or the checks of its result raised an exception; timeout: "
        "the engine killed the worker at the hard limit of the run; worker-exit: the worker "
        "process ended during the run, as after a crash in native code or when memory runs "
        "out, and the engine does not know the cause."
    )
    # A worker that ends on its own is not laid at the engine's door.
    assert "the engine ended" not in failures.message
    rows = [row for row in _rows(dataset) if row["status"] == "failed"]
    assert [row["failure"] for row in rows if row["solver"] == "b"] == ["worker-exit"]
    assert {row["attempts"] for row in rows} == {2}
    assert {row["wall_seconds"] for row in rows} == {3.0}
    # A failed run without a failure record has no kind to name.
    plain = summarize(make_dataset(cells))
    (failures,) = [item for item in plain.limitations if item.topic == "failures"]
    assert failures.message.endswith("each counts as the worst value.")
    assert {row["attempts"] for row in _rows(make_dataset(cells)) if row["status"] == "failed"} == {
        None
    }


def test_the_failures_limitation_words_only_the_kinds_that_occur() -> None:
    cells: Mapping[str, Mapping[str, Sequence[Entry]]] = {
        "a": {"p1": ["failed", 5.0], "p2": ["failed", 5.0]},
        "b": {"p1": [6.0, "failed"], "p2": [6.0, 6.0]},
    }

    def message(kinds: Mapping[str, Sequence[ErrorKind]]) -> str:
        summary = summarize(_failed(make_dataset(cells), kinds))
        (failures,) = [item for item in summary.limitations if item.topic == "failures"]
        return failures.message

    only = message({"a": [ErrorKind.EXCEPTION] * 2, "b": [ErrorKind.EXCEPTION]})
    assert only.endswith(
        "a 2 exception; b 1 exception. What each kind means: exception: the solver, its "
        "warm-up, or the checks of its result raised an exception."
    )
    others = message({"a": [ErrorKind.STARTUP, ErrorKind.TRANSFER], "b": [ErrorKind.TRANSFER]})
    assert others.endswith(
        "a 1 startup, 1 transfer; b 1 transfer. What each kind means: startup: the engine "
        "killed a worker that had not begun the solve at the start-up limit; transfer: the run "
        "could not be sent to a worker or rebuilt in it, as when its solver cannot be pickled."
    )
    assert "exception" not in others
    assert "worker-exit" not in others


def test_the_failures_limitation_counts_a_run_by_the_kind_of_its_last_attempt() -> None:
    cells: Mapping[str, Mapping[str, Sequence[Entry]]] = {
        "a": {"p1": ["failed", "failed"], "p2": [5.0, 5.0]},
        "b": {"p1": [6.0, 6.0], "p2": [6.0, 6.0]},
    }
    kinds = {"a": [ErrorKind.TIMEOUT, ErrorKind.WORKER_EXIT]}
    dataset = _failed(make_dataset(cells), kinds, first=ErrorKind.EXCEPTION)
    assert all(len(run.failures) == 2 for run in dataset.runs if run.status == "failed")
    (failures,) = [item for item in summarize(dataset).limitations if item.topic == "failures"]
    # Each run failed first with an exception: only the kind that settled it is counted.
    assert "attempt: a 1 timeout, 1 worker-exit. What each kind means: timeout:" in failures.message
    assert "exception" not in failures.message
    rows = [row for row in _rows(dataset) if row["status"] == "failed"]
    assert [row["failure"] for row in rows] == ["timeout", "worker-exit"]
    assert {row["attempts"] for row in rows} == {2}
    assert {row["wall_seconds"] for row in rows} == {3.0}


def test_the_row_of_a_failed_run_reads_one_attempt_and_claims_no_overrun() -> None:
    cells: Mapping[str, Mapping[str, Sequence[Entry]]] = {
        "a": {"p1": ["failed", 5.0], "p2": [5.0, 5.0]},
        "b": {"p1": [6.0, 6.0], "p2": [6.0, 6.0]},
    }
    walls = {"a": [1.0, 1.0, 1.0], "b": [1.0, 1.0, 1.0, 1.0]}
    timed = _timed(
        make_dataset(cells), walls, cpu={"a": 0.5, "b": 0.5}, concurrent={"a": 2, "b": 2}
    )
    dataset = _failed(timed, {"a": [ErrorKind.EXCEPTION]}, first=ErrorKind.WORKER_EXIT)
    (row,) = [row for row in _rows(dataset) if row["status"] == "failed"]
    # Every measured cell describes the last failed attempt: 3 s of wall time, 0.5 s of processor.
    assert (row["wall_seconds"], row["cpu_seconds"], row["attempts"]) == (3.0, 0.5, 2)
    assert row["concurrent"] == 2
    assert row["time_limit"] == 1.0
    # The attempt did not end for its budget: no overrun and no processor share are claimed.
    assert row["overrun_seconds"] is None
    assert row["cpu_share"] is None
    (failed,) = [run for run in dataset.runs if run.status == "failed"]
    assert failed.cpu_share is None
    done = [row for row in _rows(dataset) if row["status"] != "failed"]
    assert {row["overrun_seconds"] for row in done} == {0.0}
    assert {row["cpu_share"] for row in done} == {0.5}


def test_the_overrun_tolerance_is_a_share_of_the_limit_that_is_not_negative() -> None:
    assert AnalysisConfig().overrun_tolerance == 0.05
    assert AnalysisConfig().to_dict()["overrun_tolerance"] == 0.05
    assert AnalysisConfig(overrun_tolerance=0).overrun_tolerance == 0.0
    for invalid in (-0.1, float("inf"), float("nan"), "0.1", True, None):
        with pytest.raises(ConfigError, match="overrun_tolerance must be"):
            AnalysisConfig(overrun_tolerance=invalid)  # type: ignore[arg-type]


def test_the_run_of_a_dataset_loaded_from_disk_keeps_its_notes(late_campaign: Dataset) -> None:
    noted = {run.solver: run.record.notes for run in late_campaign.runs if run.record is not None}
    assert any(note.startswith("the run took") for note in noted["test-tardy"])
    assert noted["input-order"] == ()
