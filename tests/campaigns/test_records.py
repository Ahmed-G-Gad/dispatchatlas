"""Records: the outcome of a run, its result hash, and the settlement rule."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.campaigns import Protocol
from dispatchatlas.campaigns.execution import (
    Completion,
    Failure,
    Task,
    attempt,
    classify,
    describe,
)
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
    BoundPoint,
    ConfigError,
    MissingDependencyError,
    ModelError,
    Schedule,
    SerializationError,
    SolverError,
    Status,
    StopCriteria,
    StopReason,
    TracePoint,
    UnsupportedProblemError,
    digest,
    problem_digest,
)
from dispatchatlas.solvers import default_registry
from tests.helpers import chain_problem, small_job_shop

SOLVERS = default_registry(plugins=False)
STOP = StopCriteria(max_evaluations=20)
EARLIER = Path(__file__).parent / "data" / "earlier" / "tiny" / "runs"
"""Run records written by the library before outcomes could hold bounds over time."""

OUTCOME_DIGEST = "30e2ba95c7ddded2588023072b4c0c3a2a065db0b9bce57c63b121e2e693ccc6"
"""SHA-256 of the canonical form of ``_unbounded().to_dict()``, computed before outcomes had bounds.

Changing it changes the documents of stored records.
"""

OUTCOME_HASH = "87f36aef2bd5964f97ec3d76d071954c939212450293a9bcec4079b62ea48e0d"
"""``_unbounded().result_hash()``, computed as above. Changing it breaks stored result hashes."""

BOUNDS = (BoundPoint(0.01, 3.0), BoundPoint(0.02, 4.0))


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


def _unbounded() -> Outcome:
    """An outcome of a time-limited exact run, written out literally."""
    return Outcome(
        status=Status.FEASIBLE,
        stop_reason=StopReason.TIME_LIMIT,
        objective="makespan",
        value=57.0,
        objectives={"makespan": 57.0},
        bound=55.0,
        gap=2 / 57,
        feasible=True,
        violations=(),
        evaluations=4,
        iterations=4,
        elapsed=1.5,
        trace=(
            TracePoint(0.1, 1, 1, 70.0),
            TracePoint(0.4, 3, 3, 62.0),
            TracePoint(1.2, 4, 4, 57.0),
        ),
        front=(),
        schedule_digest="ab" * 32,
        extras={"backend": "ortools-cp-sat"},
    )


def test_an_outcome_without_bounds_keeps_its_document_and_its_hash() -> None:
    outcome = _unbounded()
    document = outcome.to_dict()
    assert outcome.bounds == ()
    assert "bounds" not in document
    assert "bounds" not in outcome.hashed()
    assert digest(document) == OUTCOME_DIGEST
    assert outcome.result_hash() == OUTCOME_HASH
    assert Outcome.from_dict(document) == outcome


def test_records_written_before_bounds_still_read_and_verify() -> None:
    paths = sorted(EARLIER.rglob("*.json"))
    assert len(paths) == 6
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        record = RunRecord.from_dict(document)
        assert record.outcome.bounds == ()
        assert record.outcome.to_dict() == document["outcome"]
        assert record.outcome.result_hash() == document["result_hash"]
        assert record.verify() == ()


def test_bounds_over_time_enter_the_document_and_the_hash_without_their_times() -> None:
    outcome, _ = _completed()
    bounded = replace(outcome, bounds=BOUNDS)
    document = bounded.to_dict()
    assert document["bounds"] == [[0.01, 3.0], [0.02, 4.0]]
    assert {key: value for key, value in document.items() if key != "bounds"} == outcome.to_dict()
    assert bounded.hashed()["bounds"] == [[3.0], [4.0]]
    assert Outcome.from_dict(json.loads(json.dumps(document))) == bounded
    later = replace(bounded, bounds=tuple(replace(point, elapsed=1.0) for point in BOUNDS))
    assert later.result_hash() == bounded.result_hash()
    for changed in (
        replace(bounded, bounds=BOUNDS[:1]),
        replace(bounded, bounds=(BOUNDS[0], BoundPoint(0.02, 4.5))),
        outcome,
    ):
        assert changed.result_hash() != bounded.result_hash()


def test_a_record_with_bounds_round_trips_and_an_edited_bound_is_found() -> None:
    outcome, schedule = _completed()
    record = _record(replace(outcome, bounds=BOUNDS), schedule)
    document = json.loads(json.dumps(record.to_dict()))
    assert document["outcome"]["bounds"] == [[0.01, 3.0], [0.02, 4.0]]
    assert RunRecord.from_dict(document) == record
    assert record.verify() == ()
    document["outcome"]["bounds"][1][1] = 4.5
    assert RunRecord.from_dict(document).verify() == ("the outcome does not match the result hash",)


def test_an_outcome_document_without_bounds_or_with_an_empty_list_has_none() -> None:
    outcome, _ = _completed()
    document = outcome.to_dict()
    assert "bounds" not in document
    assert Outcome.from_dict(document).bounds == ()
    assert Outcome.from_dict({**document, "bounds": []}).bounds == ()


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        (5, "the bounds of an outcome must be a list"),
        ([0.1], "a bound row must be a list"),
        ([[0.1]], "a bound row must list elapsed, value"),
        ([[0.1, 3.0, 1]], "a bound row must list elapsed, value"),
        ([[None, 3.0]], "a bound time must be a number"),
        ([[0.1, "3"]], "a bound value must be a number"),
        ([[0.1, math.nan]], "a bound value must be finite"),
    ],
)
def test_malformed_bounds_of_an_outcome_are_rejected(bounds: object, message: str) -> None:
    outcome, _ = _completed()
    with pytest.raises(SerializationError, match=message):
        Outcome.from_dict({**outcome.to_dict(), "bounds": bounds})


def test_describe_carries_the_bounds_of_a_result_into_its_outcome() -> None:
    problem = chain_problem()
    result = SOLVERS.create("random-search").solve(problem, STOP, 3)
    outcome, _, notes = describe(problem, replace(result, bounds=BOUNDS))
    assert notes == ()
    assert outcome.bounds == BOUNDS
    assert describe(problem, result)[0].bounds == ()


@pytest.mark.parametrize(
    "point",
    [
        BoundPoint(math.nan, 3.0),
        BoundPoint(math.inf, 3.0),
        BoundPoint(0.1, math.nan),
        BoundPoint(0.1, -math.inf),
    ],
)
def test_a_bound_that_is_not_finite_cannot_be_recorded(point: BoundPoint) -> None:
    problem = chain_problem()
    result = SOLVERS.create("random-search").solve(problem, STOP, 3)
    with pytest.raises(SolverError, match=r"bound 1 holds a (time|value) of"):
        describe(problem, replace(result, bounds=(BOUNDS[0], point)))


@pytest.mark.exact
def test_the_bounds_of_a_cp_sat_attempt_reach_its_outcome_and_replay_to_one_hash() -> None:
    pytest.importorskip("ortools")
    problem = small_job_shop()

    def run() -> Completion:
        task = Task(
            run_id="run-1",
            attempt=1,
            problem=problem,
            digest=problem_digest(problem),
            solver=SOLVERS.create("cp-sat"),
            stop=StopCriteria(time_limit=60.0),
            seed=1,
        )
        completion = attempt(task)
        assert isinstance(completion, Completion)
        return completion

    first, second = run(), run()
    outcome = first.outcome
    assert outcome.bounds
    assert outcome.bounds[-1].value == outcome.bound == 11.0
    assert [point.value for point in outcome.bounds] == sorted({p.value for p in outcome.bounds})
    assert outcome.bounds[-1].elapsed <= outcome.elapsed
    assert Outcome.from_dict(json.loads(json.dumps(outcome.to_dict()))) == outcome
    # The two attempts differ in the times of their bounds, which the result hash leaves out.
    assert first.result_hash == second.result_hash


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
