"""Records: the outcome of a run, its result hash, and the settlement rule."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dispatchatlas.campaigns import Protocol
from dispatchatlas.campaigns.execution import Failure, Task, attempt, classify, describe
from dispatchatlas.campaigns.records import (
    Category,
    ErrorKind,
    FailureRecord,
    Outcome,
    ProblemRef,
    Provenance,
    RunError,
    RunRecord,
    SolverEntry,
    schedule_digest,
    settled,
)
from dispatchatlas.core import (
    ConfigError,
    MissingDependencyError,
    ModelError,
    Schedule,
    SerializationError,
    SolverError,
    StopCriteria,
    StopReason,
    UnsupportedProblemError,
    problem_digest,
)
from dispatchatlas.solvers import default_registry
from tests.helpers import chain_problem

SOLVERS = default_registry(plugins=False)
STOP = StopCriteria(max_evaluations=20)


def _completed(solver_id: str = "random-search", seed: int = 3) -> tuple[Outcome, Schedule | None]:
    problem = chain_problem()
    result = SOLVERS.create(solver_id).solve(problem, STOP, seed)
    outcome, schedule, notes = describe(problem, result)
    assert notes == ()
    return outcome, schedule


def _record(outcome: Outcome, schedule: Schedule | None) -> RunRecord:
    problem = chain_problem()
    return RunRecord(
        run_id="run-0123456789abcdef",
        campaign="records",
        config_hash="0" * 64,
        problem=ProblemRef(problem.id, problem.family, problem_digest(problem)),
        solver=SolverEntry(
            "random-search", "random-search", None, {}, "random-sampling", True, True, False
        ),
        protocol=Protocol("evals", STOP),
        replicate=0,
        seed=3,
        outcome=outcome,
        result_hash=outcome.result_hash(),
        schedule=schedule,
        provenance=Provenance(1, 1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", 1.0, 0.9, 1024),
    )


def test_outcome_checks_the_schedule_and_recomputes_objectives() -> None:
    outcome, schedule = _completed()
    assert outcome.feasible
    assert outcome.violations == ()
    assert outcome.objectives == {"makespan": 5.0}
    assert outcome.value == 5.0
    assert schedule is not None
    assert outcome.schedule_digest == schedule_digest(schedule)
    assert outcome.evaluations == 20


def test_result_hash_ignores_wall_times_and_extras_only() -> None:
    outcome, _ = _completed()
    timed = replace(
        outcome,
        elapsed=outcome.elapsed + 5.0,
        trace=tuple(replace(point, elapsed=point.elapsed + 1.0) for point in outcome.trace),
        extras={"backend_status": "done"},
    )
    assert timed.result_hash() == outcome.result_hash()
    for changed in (
        replace(outcome, value=6.0),
        replace(outcome, evaluations=19),
        replace(outcome, schedule_digest="1" * 64),
        replace(outcome, trace=outcome.trace[:-1]),
        replace(outcome, stop_reason=StopReason.TIME_LIMIT),
    ):
        assert changed.result_hash() != outcome.result_hash()


def test_a_second_run_from_the_same_seed_has_the_same_result_hash() -> None:
    first, _ = _completed(seed=8)
    second, _ = _completed(seed=8)
    assert first.result_hash() == second.result_hash()
    assert first.hashed() == second.hashed()


def test_run_records_round_trip_and_verify() -> None:
    outcome, schedule = _completed()
    record = _record(outcome, schedule)
    document = record.to_dict()
    assert document["schema"] == "dispatchatlas.run/1"
    assert RunRecord.from_dict(document) == record
    assert record.verify() == ()
    without_schedule = _record(outcome, None)
    assert without_schedule.result_hash == record.result_hash
    assert RunRecord.from_dict(without_schedule.to_dict()).verify() == ()

    edited = dict(document, outcome={**document["outcome"], "value": 4.0})
    assert RunRecord.from_dict(edited).verify() == ("the outcome does not match the result hash",)
    rows = [list(row) for row in document["schedule"]]
    rows[0][2] += 1.0
    moved = RunRecord.from_dict(dict(document, schedule=rows))
    assert moved.verify() == ("the schedule does not match the outcome's schedule digest",)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema": "dispatchatlas.run/2"}, "expected a 'dispatchatlas.run/1' document"),
        ({"seed": "3"}, "must be an integer"),
        ({"schedule": [["a", 0, 0.0]]}, "schedule row"),
        ({"provenance": {"attempt": 1}}, "lacks the field"),
    ],
)
def test_malformed_run_records_are_rejected(change: dict[str, object], message: str) -> None:
    outcome, schedule = _completed()
    document = {**_record(outcome, schedule).to_dict(), **change}
    with pytest.raises(SerializationError, match=message):
        RunRecord.from_dict(document)


def test_failure_records_round_trip() -> None:
    outcome, schedule = _completed()
    base = _record(outcome, schedule)
    failure = FailureRecord(
        run_id=base.run_id,
        campaign=base.campaign,
        config_hash=base.config_hash,
        problem=base.problem,
        solver=base.solver,
        protocol=base.protocol,
        replicate=0,
        seed=3,
        category=Category.RECOVERABLE,
        error=RunError(ErrorKind.EXCEPTION, "boom", "RuntimeError", "Traceback ..."),
        provenance=Provenance(2, 1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", 1.0),
    )
    document = failure.to_dict()
    assert document["schema"] == "dispatchatlas.failure/1"
    assert FailureRecord.from_dict(document) == failure


@pytest.mark.parametrize(
    "error",
    [
        ConfigError("bad"),
        ModelError("bad"),
        SerializationError("bad"),
        SolverError("bad"),
        UnsupportedProblemError("bad"),
        MissingDependencyError("ortools", "exact", "cp-sat"),
    ],
)
def test_typed_errors_are_terminal(error: Exception) -> None:
    assert classify(error) is Category.TERMINAL


@pytest.mark.parametrize(
    "error", [RuntimeError("x"), OSError("x"), MemoryError(), ValueError("x"), KeyError("x")]
)
def test_other_errors_are_recoverable(error: Exception) -> None:
    assert classify(error) is Category.RECOVERABLE


def _failure(attempt_number: int, category: Category) -> FailureRecord:
    outcome, schedule = _completed()
    base = _record(outcome, schedule)
    return FailureRecord(
        base.run_id,
        base.campaign,
        base.config_hash,
        base.problem,
        base.solver,
        base.protocol,
        0,
        3,
        category,
        RunError(ErrorKind.EXCEPTION, "boom"),
        Provenance(attempt_number, 1, "", "", 0.0),
    )


def test_settlement_rule() -> None:
    recoverable, terminal = Category.RECOVERABLE, Category.TERMINAL
    assert not settled([], 1)
    assert settled([_failure(1, recoverable)], 1)
    assert not settled([_failure(1, recoverable)], 2)
    assert settled([_failure(1, recoverable), _failure(2, recoverable)], 2)
    assert settled([_failure(1, terminal)], 5)
    assert settled([_failure(2, terminal), _failure(1, recoverable)], 5)
    assert not settled([_failure(2, recoverable), _failure(1, terminal)], 5)


def test_attempt_turns_solver_errors_into_failures() -> None:
    problem = chain_problem()
    task = Task(
        run_id="run-1",
        attempt=3,
        problem=problem,
        digest=problem_digest(problem),
        solver=SOLVERS.create("random-search"),
        stop=StopCriteria(),  # unbounded: an iterative solver refuses it
        seed=1,
    )
    failure = attempt(task)
    assert isinstance(failure, Failure)
    assert failure.run_id == "run-1"
    assert failure.attempt == 3
    assert failure.category is Category.TERMINAL
    assert failure.error.type == "dispatchatlas.core.errors.SolverError"
    assert "needs a time, evaluation, or iteration limit" in failure.error.message
    assert failure.error.traceback is not None
    assert "SolverError" in failure.error.traceback
    assert failure.measurement.worker == "main"
