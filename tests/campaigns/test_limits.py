"""Start-up and hard limits: when they count from, and what a killed run keeps."""

from __future__ import annotations

from typing import Any

import pytest

from dispatchatlas.campaigns import Execution, Protocol, loads_campaign
from dispatchatlas.campaigns.config import STARTUP_SECONDS
from dispatchatlas.campaigns.execution import Measurement, engine_failure
from dispatchatlas.campaigns.records import (
    Category,
    ErrorKind,
    FailureRecord,
    Incumbent,
    ProblemRef,
    Provenance,
    RunError,
    SessionRecord,
    SessionStatus,
    SolverEntry,
)
from dispatchatlas.core import ConfigError, SerializationError, StopCriteria

INCUMBENT = Incumbent(
    objective="makespan",
    value=12.0,
    objectives={"makespan": 12.0, "total_tardiness": 3.5},
    feasible=True,
    evaluations=7,
    iteration=2,
    elapsed=0.25,
)


def _failure(**changes: Any) -> FailureRecord:
    fields: dict[str, Any] = {
        "run_id": "run-0123456789abcdef",
        "campaign": "limits",
        "config_hash": "0" * 64,
        "problem": ProblemRef("p", "flow-shop", "1" * 64),
        "solver": SolverEntry("sleeper", "sleeper", None, {}, "random-sampling", True, True, False),
        "protocol": Protocol("clock", StopCriteria(time_limit=1.0)),
        "replicate": 0,
        "seed": 3,
        "category": Category.TERMINAL,
        "error": RunError(ErrorKind.TIMEOUT, "killed"),
        "provenance": Provenance(
            1, 1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:06Z", 6.0, waited_seconds=1.5
        ),
        "solve_started": True,
        "last_incumbent": INCUMBENT,
    }
    fields.update(changes)
    return FailureRecord(**fields)


def test_a_failure_record_round_trips_with_its_incumbent() -> None:
    failure = _failure()
    document = failure.to_dict()
    assert document["schema"] == "dispatchatlas.failure/1"
    assert document["solve_started"] is True
    assert document["last_incumbent"] == {
        "objective": "makespan",
        "value": 12.0,
        "objectives": {"makespan": 12.0, "total_tardiness": 3.5},
        "feasible": True,
        "evaluations": 7,
        "iteration": 2,
        "elapsed": 0.25,
    }
    assert document["provenance"]["waited_seconds"] == 1.5
    assert FailureRecord.from_dict(document) == failure
    plain = _failure(solve_started=False, last_incumbent=None)
    assert FailureRecord.from_dict(plain.to_dict()) == plain


def test_failure_records_without_the_new_fields_still_load() -> None:
    document = _failure().to_dict()
    del document["solve_started"]
    del document["last_incumbent"]
    for key in ("waited_seconds", "cpu", "concurrent"):
        del document["provenance"][key]
    loaded = FailureRecord.from_dict(document)
    assert loaded.solve_started is None
    assert loaded.last_incumbent is None
    assert loaded.provenance.waited_seconds is None
    assert loaded.error.kind is ErrorKind.TIMEOUT
    assert loaded.provenance.wall_seconds == 6.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"value": float("inf")}, "must be finite"),
        ({"objectives": {"makespan": "12"}}, "must be a number"),
        ({"objectives": [12.0]}, "must be an object"),
        ({"feasible": 1}, "must be true or false"),
        ({"evaluations": 1.5}, "must be an integer"),
    ],
)
def test_a_malformed_incumbent_is_rejected(change: dict[str, object], message: str) -> None:
    document = _failure().to_dict()
    document["last_incumbent"] = {**document["last_incumbent"], **change}
    with pytest.raises(SerializationError, match=message):
        FailureRecord.from_dict(document)
    partial = INCUMBENT.to_dict()
    del partial["elapsed"]
    with pytest.raises(SerializationError, match="lacks the field 'elapsed'"):
        Incumbent.from_dict(partial)


def test_only_start_up_failures_and_dead_workers_are_recoverable() -> None:
    measurement = Measurement("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", 1.0)
    categories = {
        kind: engine_failure("run", 1, kind, "ended", measurement).category
        for kind in (
            ErrorKind.STARTUP,
            ErrorKind.WORKER_EXIT,
            ErrorKind.TIMEOUT,
            ErrorKind.TRANSFER,
        )
    }
    assert categories == {
        ErrorKind.STARTUP: Category.RECOVERABLE,
        ErrorKind.WORKER_EXIT: Category.RECOVERABLE,
        ErrorKind.TIMEOUT: Category.TERMINAL,
        ErrorKind.TRANSFER: Category.TERMINAL,
    }
    killed = engine_failure(
        "run",
        1,
        ErrorKind.TIMEOUT,
        "ended",
        measurement,
        solve_started=True,
        last_incumbent=INCUMBENT,
    )
    assert (killed.solve_started, killed.last_incumbent) == (True, INCUMBENT)


def test_the_start_up_limit_is_optional_and_positive() -> None:
    assert Execution().startup_limit == STARTUP_SECONDS == 120.0
    assert Execution(startup_limit=30).startup_limit == 30.0
    assert Execution().to_dict()["startup_limit"] == 120.0
    invalid: tuple[Any, ...] = (0, -1.0, float("inf"), "60", True)
    for value in invalid:
        with pytest.raises(ConfigError, match="startup_limit"):
            Execution(startup_limit=value)


def test_a_campaign_file_sets_the_start_up_limit() -> None:
    text = """
        schema_version = 1
        id = "small"
        seed = 0
        instances = [{ family = "job-shop", seeds = [1] }]
        solvers = [{ id = "input-order" }]
        protocols = [{ name = "p", stop = { max_iterations = 5 } }]
        """
    assert loads_campaign(text).execution.startup_limit == 120.0
    campaign = loads_campaign(text + "[execution]\nstartup_limit = 45\n")
    assert campaign.execution.startup_limit == 45.0
    with pytest.raises(ConfigError, match="startup_limit"):
        loads_campaign(text + "[execution]\nstartup_limit = 0\n")


def test_session_records_without_a_start_up_limit_still_load() -> None:
    record = SessionRecord(
        session=1,
        campaign="limits",
        config_hash="0" * 64,
        started="2026-01-01T00:00:00Z",
        finished=None,
        status=SessionStatus.RUNNING,
        execution=Execution(grace=2.0),
        workers=1,
        environment={},
    )
    document = record.to_dict()
    assert document["execution"]["startup_limit"] == 120.0
    del document["execution"]["startup_limit"]
    assert SessionRecord.from_dict(document) == record
