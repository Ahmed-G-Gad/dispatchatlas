"""The engine's notes on a run, the clock origin of an attempt, and what replays."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.analysis import load_dataset
from dispatchatlas.analysis.tables import runs_table
from dispatchatlas.campaigns import (
    Campaign,
    Protocol,
    SolverSpec,
    Verdict,
    plan,
    read_plan,
    read_runs,
    replay,
    run,
)
from dispatchatlas.campaigns.execution import (
    CLOCK_MARGIN_SECONDS,
    OVERRUN_MARGIN,
    Completion,
    Task,
    attempt,
    observe,
)
from dispatchatlas.campaigns.records import ProblemRef, Provenance, RunRecord
from dispatchatlas.campaigns.store import run_path
from dispatchatlas.core import SerializationError, Solver, StopCriteria, StopReason, problem_digest
from dispatchatlas.core.solver import TracePoint
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.rules import RandomSearch
from tests.campaigns.solvers import Scripted, SlowSetup, Tardy, registry, tiny_campaign
from tests.helpers import chain_problem, random_problem

REGISTRY = registry()
EVALUATIONS = StopCriteria(max_evaluations=20)


def _attempt(solver: Solver, stop: StopCriteria = EVALUATIONS) -> Completion:
    problem = chain_problem()
    task = Task("run-0", 1, problem, problem_digest(problem), solver, stop, seed=3)
    result = attempt(task)
    assert isinstance(result, Completion)
    return result


def _noted(notes: tuple[str, ...], start: str) -> bool:
    return any(note.startswith(start) for note in notes)


def test_a_run_inside_its_limits_earns_no_note() -> None:
    assert _attempt(RandomSearch()).notes == ()
    assert _attempt(RandomSearch(), StopCriteria(time_limit=0.05)).notes == ()
    assert _attempt(Scripted(times=(0.0, 0.0), evaluations=(1, 1))).notes == ()


def test_no_built_in_solver_earns_a_note_under_an_evaluation_limit() -> None:
    solvers = default_registry(plugins=False)
    completed = 0
    for info in solvers.infos():
        problem = chain_problem()
        solver = solvers.create(info.id)
        stop = StopCriteria(max_evaluations=30)
        result = attempt(Task("run-0", 1, problem, problem_digest(problem), solver, stop, seed=2))
        if isinstance(result, Completion):
            completed += 1
            assert result.notes == (), info.id
    assert completed >= 60


def test_a_run_beyond_its_time_limit_is_noted_with_its_wall_time() -> None:
    late = _attempt(Tardy(seconds=0.3), StopCriteria(time_limit=0.1))
    assert late.measurement.wall_seconds >= 0.3
    (overrun,) = [note for note in late.notes if note.startswith("the run took")]
    assert "under a time limit of 0.1 s" in overrun
    assert "times the limit" in overrun
    assert late.outcome.stop_reason is StopReason.COMPLETED
    # The same work under a limit it stays inside earns no such note.
    assert not _noted(_attempt(Tardy(seconds=0.3), StopCriteria(time_limit=5.0)).notes, "the run")


def test_an_elapsed_time_the_engine_contradicts_is_noted() -> None:
    # The solver read its clock before 0.3 s of work: it reports less than the engine saw.
    under = _attempt(Tardy(seconds=0.3)).notes
    assert _noted(under, "the solver reports")
    assert "the engine measured" in under[0]
    (over,) = _attempt(Scripted(times=(0.0,), evaluations=(1,), elapsed=30.0)).notes
    assert over.startswith("the solver reports 30 s elapsed, the engine measured")


@pytest.mark.parametrize(
    ("script", "start"),
    [
        ({"times": (-1.0, 0.0), "evaluations": (1, 1)}, "the trace holds a negative time"),
        ({"times": (0.0, -2.0), "evaluations": (1, 1)}, "the times of the trace decrease"),
        ({"times": (0.0, 5.0), "evaluations": (1, 1)}, "the trace holds a time (5.0 s) beyond"),
        ({"times": (0.0, 0.0), "evaluations": (2, 1)}, "the evaluation counts of the trace"),
        (
            {"times": (0.0, 0.0), "evaluations": (1, 1), "last": 123.0},
            "the last value of the trace (123.0) differs from the value of the outcome",
        ),
        ({}, "the solver is iterative and returned a schedule without a trace"),
    ],
)
def test_a_trace_that_cannot_be_drawn_is_noted(script: dict[str, Any], start: str) -> None:
    completion = _attempt(Scripted(**script))
    assert _noted(completion.notes, start)
    # The run is recorded all the same: a note qualifies it, it does not fail it.
    assert completion.outcome.feasible


def test_a_constructive_solver_needs_no_trace() -> None:
    outcome = replace(_attempt(RandomSearch()).outcome, trace=())
    assert observe(outcome, EVALUATIONS, 0.0, iterative=False) == ()
    assert observe(outcome, EVALUATIONS, 0.0, iterative=True) == (
        "the solver is iterative and returned a schedule without a trace",
    )


def test_the_margins_of_the_time_notes() -> None:
    point = TracePoint(0.0, 1, 0, 0.0)
    outcome = replace(_attempt(RandomSearch()).outcome, trace=(), elapsed=1.0)
    outcome = replace(outcome, trace=(replace(point, value=outcome.value or 0.0),))
    stop = StopCriteria(time_limit=1.0)
    assert observe(outcome, stop, 1.0 + OVERRUN_MARGIN, iterative=True) == ()
    assert _noted(observe(outcome, stop, 1.2, iterative=True), "the run took 1.2 s")
    quick = replace(outcome, elapsed=0.0)
    assert observe(quick, EVALUATIONS, CLOCK_MARGIN_SECONDS, iterative=True) == ()
    assert _noted(observe(quick, EVALUATIONS, 0.06, iterative=True), "the solver reports 0 s")
    # A long run may differ by the same share of its wall time.
    assert observe(replace(outcome, elapsed=96.0), EVALUATIONS, 100.0, iterative=True) == ()
    assert observe(replace(outcome, elapsed=94.0), EVALUATIONS, 100.0, iterative=True) != ()


def test_notes_are_logged_and_stored_in_the_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-tardy", parameters={"seconds": 0.3}), SolverSpec("input-order")),
        protocols=(Protocol("clock", StopCriteria(time_limit=0.1)),),
        replicates=1,
    )
    with caplog.at_level(logging.WARNING, logger="dispatchatlas"):
        summary = run(campaign, tmp_path, registry=REGISTRY)
    records = {record.solver.name: record for record in read_runs(summary.directory)[:2]}
    tardy = records["test-tardy"]
    assert len(tardy.notes) == 2
    assert _noted(tardy.notes, "the run took")
    assert _noted(tardy.notes, "the solver reports")
    assert records["input-order"].notes == ()
    assert tardy.verify() == ()
    assert RunRecord.from_dict(tardy.to_dict()) == tardy
    logged = [entry.getMessage() for entry in caplog.records]
    for note in tardy.notes:
        assert any(note in message for message in logged)


def test_records_without_notes_still_load() -> None:
    completion = _attempt(RandomSearch())
    document = _document(completion, notes=("the run took 2 s",))
    assert RunRecord.from_dict(document).notes == ("the run took 2 s",)
    del document["notes"]
    assert RunRecord.from_dict(document).notes == ()
    for malformed in ("a note", [1], [None]):
        document["notes"] = malformed
        with pytest.raises(SerializationError, match="note"):
            RunRecord.from_dict(document)


def test_a_directory_whose_records_lack_notes_resumes_analyses_and_replays(tmp_path: Path) -> None:
    campaign = tiny_campaign(replicates=1)
    summary = run(campaign, tmp_path, registry=REGISTRY)
    for item in read_plan(summary.directory).runs:
        path = run_path(summary.directory, item.solver, item.problem, item.replicate, item.protocol)
        document = json.loads(path.read_text())
        del document["notes"]
        path.write_text(json.dumps(document))
    again = run(campaign, tmp_path, registry=REGISTRY)
    assert (again.attempted, again.completed) == (0, summary.completed)
    records = read_runs(summary.directory)
    assert len(records) == summary.completed
    assert all(record.notes == () for record in records)
    assert replay(campaign, tmp_path, registry=REGISTRY).ok
    table = runs_table(load_dataset(summary.directory))
    assert len(table.rows) == summary.completed
    wall = [column.key for column in table.columns].index("wall_seconds")
    assert all(isinstance(row[wall], float) for row in table.rows)


def _document(completion: Completion, **changes: Any) -> dict[str, Any]:
    campaign = tiny_campaign(solvers=(SolverSpec("random-search"),), replicates=1)
    return _stored(campaign, completion, **changes).to_dict()


def _stored(campaign: Campaign, completion: Completion, **changes: Any) -> RunRecord:
    planned = plan(campaign, registry=REGISTRY)
    item = planned.runs[0]
    problem = planned.problems[0]
    measured = completion.measurement
    values: dict[str, Any] = {
        "run_id": item.run_id,
        "campaign": planned.campaign,
        "config_hash": planned.config_hash,
        "problem": ProblemRef(problem.id, problem.family, problem.digest),
        "solver": planned.solvers[0],
        "protocol": planned.protocols[0],
        "replicate": item.replicate,
        "seed": item.seed,
        "outcome": completion.outcome,
        "result_hash": completion.result_hash,
        "schedule": completion.schedule,
        "provenance": Provenance(
            1, 1, measured.started, measured.finished, measured.wall_seconds, measured.cpu_seconds
        ),
        "notes": completion.notes,
    }
    values.update(changes)
    return RunRecord(**values)


# ------------------------------------------------------------- clock origin


def test_the_set_up_of_a_solver_counts_in_a_campaign_and_not_outside_one() -> None:
    problem = random_problem(5)
    solver = SlowSetup(setup=0.5)
    stop = StopCriteria(time_limit=0.75)
    # Called on its own, the solver starts its clock after its set-up.
    began = time.perf_counter()
    alone = solver.solve(problem, stop, 3)
    alone_wall = time.perf_counter() - began
    assert alone.stop_reason is StopReason.TIME_LIMIT
    assert 0.75 <= alone.elapsed < 1.2
    assert alone_wall >= 1.25
    # In an attempt the engine's origin holds: the set-up is inside the
    # elapsed time and the limit, so the search stops 0.5 s earlier.
    task = Task("run-0", 1, problem, problem_digest(problem), solver, stop, seed=3)
    inside = attempt(task)
    assert isinstance(inside, Completion)
    assert inside.outcome.stop_reason is StopReason.TIME_LIMIT
    assert 0.75 <= inside.outcome.elapsed < 1.2
    assert inside.measurement.wall_seconds < 1.2
    assert inside.outcome.trace[0].elapsed >= 0.5
    assert inside.outcome.evaluations < alone.evaluations
    assert not _noted(inside.notes, "the solver reports")


def test_a_mapping_heuristic_past_a_short_limit_is_flagged() -> None:
    problem = random_problem(7, tasks=120, machines=4)
    solver = default_registry(plugins=False).create("min-min")
    stop = StopCriteria(time_limit=1e-4)
    result = attempt(Task("run-0", 1, problem, problem_digest(problem), solver, stop, seed=1))
    assert isinstance(result, Completion)
    # The heuristic checks no limit: it completes, late, and only the note says so.
    assert result.outcome.stop_reason is StopReason.COMPLETED
    assert result.outcome.evaluations == 1
    assert result.measurement.wall_seconds > 1e-4 * (1.0 + OVERRUN_MARGIN)
    assert _noted(result.notes, "the run took")
    assert "under a time limit of 0.0001 s" in result.notes[0]


# -------------------------------------------------------------- replayable


def test_a_run_under_a_time_only_protocol_is_not_replayable_whatever_stopped_it() -> None:
    completion = _attempt(RandomSearch(), StopCriteria(time_limit=5.0, patience=3))
    assert completion.outcome.stop_reason is StopReason.STAGNATION
    assert completion.outcome.replayable
    campaign = tiny_campaign(
        solvers=(SolverSpec("random-search"),),
        protocols=(Protocol("clock", StopCriteria(time_limit=5.0, patience=3)),),
        replicates=1,
    )
    stagnated = _stored(campaign, completion)
    assert not stagnated.replayable
    # An evaluation or iteration limit beside the time limit fixes the schedule of the search.
    for bound in ({"max_evaluations": 50}, {"max_iterations": 50}):
        stop = StopCriteria(time_limit=5.0, patience=3, **bound)
        assert replace(stagnated, protocol=Protocol("both", stop)).replayable
    assert replace(stagnated, protocol=Protocol("evals", EVALUATIONS)).replayable
    # A solver that is not iterative follows no schedule: its completed run reproduces.
    constructive = replace(stagnated.solver, iterative=False)
    assert replace(stagnated, solver=constructive).replayable


def test_replay_skips_the_runs_of_a_time_only_protocol(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("input-order"), SolverSpec("random-search")),
        protocols=(Protocol("clock", StopCriteria(time_limit=5.0, patience=3)),),
        replicates=1,
    )
    summary = run(campaign, tmp_path, registry=REGISTRY)
    reasons = {record.outcome.stop_reason for record in read_runs(summary.directory)}
    assert reasons == {StopReason.COMPLETED, StopReason.STAGNATION}
    report = replay(campaign, tmp_path, registry=REGISTRY)
    verdicts = {(entry.solver, entry.verdict) for entry in report.entries}
    assert verdicts == {("input-order", Verdict.MATCH), ("random-search", Verdict.SKIPPED)}
    skipped = [entry for entry in report.entries if entry.verdict is Verdict.SKIPPED]
    assert all("only bound of the run's protocol is its time limit" in e.detail for e in skipped)
    assert report.ok
