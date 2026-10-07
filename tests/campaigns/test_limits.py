"""Start-up and hard limits: when they count from, and what a killed run keeps."""

from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.campaigns import (
    Execution,
    FamilySource,
    Protocol,
    SolverSpec,
    loads_campaign,
    read_failures,
    read_runs,
    run,
)
from dispatchatlas.campaigns import timeout as limits
from dispatchatlas.campaigns.backends import (
    PROGRESS_INTERVAL,
    ProcessBackend,
    _Reporter,
    _Slot,
    backend_for,
)
from dispatchatlas.campaigns.config import STARTUP_SECONDS
from dispatchatlas.campaigns.execution import Measurement, Task, engine_failure
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
from dispatchatlas.core import ConfigError, SerializationError, SolverInfo, StopCriteria
from dispatchatlas.solvers.rules import InputOrder
from dispatchatlas.solvers.search import Improvement, Search, listening
from tests.campaigns.solvers import registry, tiny_campaign
from tests.helpers import random_problem

REGISTRY = registry()
ONE_PROBLEM = (FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),)
CLOCK = (Protocol("clock", StopCriteria(time_limit=0.2, max_evaluations=10)),)
ROOT = Path(__file__).resolve().parents[2]

INCUMBENT = Incumbent(
    objective="makespan",
    value=12.0,
    objectives={"makespan": 12.0, "total_tardiness": 3.5},
    feasible=True,
    evaluations=7,
    iteration=2,
    elapsed=1.25,
)
AT_LIMIT_CHANGES: dict[str, Any] = {
    "value": 14.0,
    "objectives": {"makespan": 14.0, "total_tardiness": 5.0},
    "evaluations": 3,
    "iteration": 1,
    "elapsed": 0.75,
}
AT_LIMIT = replace(INCUMBENT, **AT_LIMIT_CHANGES)
LARGE = (FamilySource("flow-shop", (1,), {"jobs": 400, "machines": 10}),)
DELAY = "DISPATCHATLAS_TEST_START_DELAY"


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
        "limit_incumbent": AT_LIMIT,
    }
    fields.update(changes)
    return FailureRecord(**fields)


def test_a_failure_record_round_trips_with_its_incumbents() -> None:
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
        "elapsed": 1.25,
    }
    assert document["limit_incumbent"] == {**document["last_incumbent"], **AT_LIMIT_CHANGES}
    assert document["provenance"]["waited_seconds"] == 1.5
    assert FailureRecord.from_dict(document) == failure
    plain = _failure(solve_started=False, last_incumbent=None, limit_incumbent=None)
    assert FailureRecord.from_dict(plain.to_dict()) == plain
    late = _failure(limit_incumbent=None)
    assert FailureRecord.from_dict(late.to_dict()) == late


def test_failure_records_without_the_new_fields_still_load() -> None:
    document = _failure().to_dict()
    del document["solve_started"]
    del document["last_incumbent"]
    del document["limit_incumbent"]
    for key in ("waited_seconds", "cpu", "concurrent", "shared_clock"):
        del document["provenance"][key]
    loaded = FailureRecord.from_dict(document)
    assert loaded.solve_started is None
    assert loaded.last_incumbent is None
    assert loaded.limit_incumbent is None
    assert loaded.provenance.waited_seconds is None
    assert loaded.provenance.shared_clock is None
    earlier = _failure().to_dict()
    del earlier["limit_incumbent"]
    assert FailureRecord.from_dict(earlier) == _failure(limit_incumbent=None)
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
    document = _failure().to_dict()
    document["limit_incumbent"] = {**document["limit_incumbent"], **change}
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


# ------------------------------------------------------------- the backend


def _alive(pid: int) -> bool:
    """Return whether a process with this pid is running."""
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel32.WaitForSingleObject.restype = ctypes.c_ulong
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x00100000, 0, pid)  # SYNCHRONIZE
        running = False
        if handle:
            running = kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
            running = True
        except ProcessLookupError:
            running = False
        except PermissionError:
            running = True
        status = Path(f"/proc/{pid}/stat")
        if running and status.exists():
            # A killed process stays listed as a zombie until its parent reaps it.
            running = status.read_text(encoding="ascii").rpartition(")")[2].split()[0] != "Z"
    return running


def _gone(pid: int, within: float = 15.0) -> bool:
    """Return whether the process ends within ``within`` seconds."""
    deadline = time.monotonic() + within
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _pids(marker: Path, within: float = 90.0) -> tuple[int, int]:
    """Return the worker's and its child's pid once the scripted solver has written them."""
    deadline = time.monotonic() + within
    while not marker.exists():
        assert time.monotonic() < deadline, "the scripted solver never started"
        time.sleep(0.05)
    worker, child = marker.read_text(encoding="ascii").split()
    return int(worker), int(child)


def _end(pid: int) -> None:
    """End a process at once, without any chance for it to clean up."""
    os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


def test_a_slow_start_does_not_count_against_the_hard_limit(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-slow-arrival", parameters={"delay": 1.5}),),
        protocols=CLOCK,
        instances=ONE_PROBLEM,
        backend="process",
        grace=0.3,
    )
    summary = run(campaign, tmp_path, registry=REGISTRY)
    assert (summary.completed, summary.failed, summary.errors) == (1, 0, 0)
    (record,) = read_runs(summary.directory)
    assert record.provenance.hard_limit == pytest.approx(0.5)
    assert record.provenance.waited_seconds is not None
    assert record.provenance.waited_seconds >= 1.5
    assert record.provenance.wall_seconds < 0.5


def test_a_killed_run_keeps_its_last_incumbent_and_its_limit_counts_from_the_solve(
    tmp_path: Path,
) -> None:
    campaign = tiny_campaign(
        solvers=(
            SolverSpec("test-stalled", "prompt", {"seconds": 0.0}),
            SolverSpec("test-stalled", "stalled", {"seconds": 30.0, "delay": 1.0}),
        ),
        protocols=CLOCK,
        instances=ONE_PROBLEM,
        backend="process",
        grace=0.3,
        retries=2,
    )
    started = time.perf_counter()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    assert time.perf_counter() - started < 25.0
    assert (summary.completed, summary.failed, summary.errors) == (1, 1, 1)
    (record,) = read_runs(summary.directory)
    (failure,) = read_failures(summary.directory)
    assert failure.solver.name == "stalled"
    assert (failure.category, failure.error.kind) == (Category.TERMINAL, ErrorKind.TIMEOUT)
    assert "hard limit of 0.5 s" in failure.error.message
    assert "counted from the start of its solve" in failure.error.message
    assert failure.solve_started is True
    assert failure.provenance.waited_seconds is not None
    assert failure.provenance.waited_seconds >= 1.0
    assert 0.5 <= failure.provenance.wall_seconds < 1.5
    incumbent = failure.last_incumbent
    assert incumbent is not None
    assert failure.limit_incumbent == incumbent
    assert incumbent.objectives == record.outcome.objectives
    assert incumbent.objective == record.outcome.objective
    assert incumbent.value == record.outcome.value
    assert (incumbent.feasible, incumbent.evaluations, incumbent.iteration) == (True, 1, 0)
    assert 0.0 <= incumbent.elapsed < 0.5
    assert FailureRecord.from_dict(failure.to_dict()) == failure


def test_a_solver_outside_a_search_reports_no_incumbent(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-sleeper", parameters={"seconds": 30.0}),),
        protocols=CLOCK,
        instances=ONE_PROBLEM,
        backend="process",
        grace=0.3,
    )
    summary = run(campaign, tmp_path, registry=REGISTRY)
    (failure,) = read_failures(summary.directory)
    assert failure.error.kind is ErrorKind.TIMEOUT
    assert (failure.solve_started, failure.last_incumbent) == (True, None)
    assert failure.limit_incumbent is None


def test_a_killed_run_keeps_apart_the_incumbent_it_held_at_its_time_limit(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-overrun", parameters={"after": 0.35, "seconds": 30.0}),),
        protocols=CLOCK,
        instances=ONE_PROBLEM,
        replicates=1,
        backend="process",
        grace=0.3,
    )
    summary = run(campaign, tmp_path, registry=REGISTRY)
    (failure,) = read_failures(summary.directory)
    assert failure.error.kind is ErrorKind.TIMEOUT
    limit = failure.protocol.stop.time_limit
    last, held = failure.last_incumbent, failure.limit_incumbent
    assert limit == 0.2
    assert last is not None
    assert held is not None
    assert (held.evaluations, last.evaluations) == (3, 4)
    assert held.elapsed <= limit < last.elapsed
    assert last.value < held.value  # the search went on improving after its time limit


def test_an_incumbent_leaves_the_worker_before_a_call_that_keeps_the_interpreter_lock(
    tmp_path: Path,
) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-holds-lock", parameters={"seconds": 20.0}),),
        protocols=CLOCK,
        instances=ONE_PROBLEM,
        replicates=1,
        backend="process",
        grace=0.3,
    )
    started = time.perf_counter()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    assert time.perf_counter() - started < 15.0
    (failure,) = read_failures(summary.directory)
    assert failure.error.kind is ErrorKind.TIMEOUT
    incumbent = failure.last_incumbent
    assert incumbent is not None
    assert (incumbent.feasible, incumbent.evaluations) == (True, 1)
    assert failure.limit_incumbent == incumbent


def _slow_start(monkeypatch: pytest.MonkeyPatch, directory: Path, seconds: float) -> None:
    """Make every interpreter this process starts sleep ``seconds`` seconds as it starts."""
    directory.mkdir()
    script = f"import os, time\ntime.sleep(float(os.environ.get({DELAY!r}, '0')))\n"
    (directory / "sitecustomize.py").write_text(script, encoding="ascii")
    inherited = os.environ.get("PYTHONPATH")
    path = [str(directory)] if not inherited else [str(directory), inherited]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path))
    monkeypatch.setenv(DELAY, repr(seconds))


def test_the_start_of_a_worker_counts_as_waiting_and_against_the_start_up_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The problem makes a message larger than the buffer of a pipe, which a
    # worker takes in only once its interpreter has started.
    _slow_start(monkeypatch, tmp_path / "site", 1.5)
    campaign = tiny_campaign(
        solvers=(SolverSpec("input-order"),),
        protocols=CLOCK,
        instances=LARGE,
        replicates=1,
        backend="process",
        grace=0.3,
    )
    summary = run(campaign, tmp_path / "waited", registry=REGISTRY)
    assert (summary.completed, summary.failed) == (1, 0)
    (record,) = read_runs(summary.directory)
    assert record.provenance.waited_seconds is not None
    assert record.provenance.waited_seconds >= 1.5

    monkeypatch.setenv(DELAY, "20.0")
    late = replace(campaign, id="late", execution=replace(campaign.execution, startup_limit=0.5))
    started = time.perf_counter()
    summary = run(late, tmp_path / "late", registry=REGISTRY)
    assert time.perf_counter() - started < 15.0
    assert (summary.completed, summary.failed) == (0, 1)
    (failure,) = read_failures(summary.directory)
    assert (failure.category, failure.error.kind) == (Category.RECOVERABLE, ErrorKind.STARTUP)
    assert failure.solve_started is False
    assert failure.provenance.waited_seconds is None
    assert 0.5 <= failure.provenance.wall_seconds < 5.0


def test_a_kill_at_the_hard_limit_does_not_wait_for_another_worker_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first run is killed while the second, which began 0.4 s later, still
    # has time. The fresh worker that replaces the killed one takes 2 s to start.
    _slow_start(monkeypatch, tmp_path / "site", 2.0)
    campaign = tiny_campaign(
        solvers=(
            SolverSpec("test-stalled", "first", {"seconds": 30.0}),
            SolverSpec("test-stalled", "second", {"seconds": 30.0, "delay": 0.4}),
            SolverSpec("input-order"),
        ),
        protocols=CLOCK,
        instances=LARGE,
        replicates=1,
        backend="process",
        workers=2,
        grace=0.3,
    )
    summary = run(campaign, tmp_path / "store", registry=REGISTRY)
    assert (summary.completed, summary.failed) == (1, 2)
    failures = {failure.solver.name: failure for failure in read_failures(summary.directory)}
    assert set(failures) == {"first", "second"}
    for failure in failures.values():
        assert failure.error.kind is ErrorKind.TIMEOUT
        assert failure.provenance.hard_limit == pytest.approx(0.5)
        assert 0.5 <= failure.provenance.wall_seconds < 1.5
        assert failure.provenance.waited_seconds is not None
        assert failure.provenance.waited_seconds >= 2.0
    (record,) = read_runs(summary.directory)
    assert record.provenance.waited_seconds is not None
    assert record.provenance.waited_seconds >= 2.0


def test_a_worker_that_does_not_begin_in_time_is_a_start_up_failure(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-slow-arrival", parameters={"delay": 60.0}),),
        protocols=CLOCK,
        instances=ONE_PROBLEM,
        backend="process",
        grace=0.3,
        retries=1,
        startup_limit=1.0,
    )
    started = time.perf_counter()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    assert time.perf_counter() - started < 40.0
    assert (summary.completed, summary.failed, summary.errors) == (0, 1, 2)
    failures = read_failures(summary.directory)
    assert [(f.provenance.attempt, f.category, f.error.kind) for f in failures] == [
        (1, Category.RECOVERABLE, ErrorKind.STARTUP),
        (2, Category.RECOVERABLE, ErrorKind.STARTUP),
    ]
    for failure in failures:
        assert "start-up limit of 1 s" in failure.error.message
        assert failure.solve_started is False
        assert failure.last_incumbent is None
        assert failure.provenance.waited_seconds is None
        assert failure.provenance.wall_seconds >= 1.0


def test_a_kill_reaches_the_processes_a_solver_started(tmp_path: Path) -> None:
    marker = tmp_path / "pids"
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-nested", parameters={"marker": str(marker)}),),
        protocols=(Protocol("clock", StopCriteria(time_limit=0.5)),),
        instances=ONE_PROBLEM,
        replicates=1,
        backend="process",
        grace=1.5,
    )
    summary = run(campaign, tmp_path / "store", registry=REGISTRY)
    (failure,) = read_failures(summary.directory)
    assert failure.error.kind is ErrorKind.TIMEOUT
    worker, child = _pids(marker, within=0.0)
    assert _gone(worker)
    assert _gone(child)


@pytest.mark.skipif(sys.platform != "win32", reason="job objects are a Windows facility")
def test_without_a_job_object_a_kill_still_ends_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(limits, "_new_job", lambda pid: (None, "refused for the test"))
    marker = tmp_path / "pids"
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-nested", parameters={"marker": str(marker)}),),
        protocols=(Protocol("clock", StopCriteria(time_limit=0.5)),),
        instances=ONE_PROBLEM,
        replicates=1,
        backend="process",
        grace=1.5,
    )
    with caplog.at_level("WARNING", logger="dispatchatlas"):
        summary = run(campaign, tmp_path / "store", registry=REGISTRY)
    (failure,) = read_failures(summary.directory)
    assert failure.error.kind is ErrorKind.TIMEOUT
    assert "is in no job object (refused for the test)" in caplog.text
    worker, child = _pids(marker, within=0.0)
    assert _gone(worker)
    assert _alive(child)  # the fallback reaches the worker alone
    _end(child)
    assert _gone(child)


def test_workers_end_when_the_supervising_process_is_killed(tmp_path: Path) -> None:
    marker = tmp_path / "pids"
    command = [sys.executable, "-m", "tests.campaigns.supervised", str(tmp_path / "s"), str(marker)]
    launched = subprocess.Popen(command, cwd=ROOT)
    try:
        worker, child = _pids(marker)
        supervisor = int(Path(f"{marker}.supervisor").read_text(encoding="ascii"))
        assert _alive(supervisor)
        assert _alive(worker)
        assert _alive(child)
        _end(supervisor)
        # Reaped here: a killed child stays listed until its parent has waited for it.
        launched.wait(15)
        assert _gone(supervisor)
        assert _gone(worker)
        assert _gone(child)
    finally:
        launched.kill()
        launched.wait(30)


def test_a_worker_without_its_supervisor_ends_only_a_process_group_it_leads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[tuple[int, int]] = []
    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    if sys.platform == "win32":
        limits._end_self()
        assert exits == [1]
        return
    monkeypatch.setattr(os, "killpg", lambda group, sent: killed.append((group, sent)))
    monkeypatch.setattr(os, "getpgrp", lambda: os.getpid() + 1)  # the group of another process
    limits._end_self()
    assert (killed, exits) == ([], [1])
    monkeypatch.setattr(os, "getpgrp", os.getpid)
    limits._end_self()
    assert (killed, exits) == ([(os.getpid(), signal.SIGKILL)], [1, 1])


@pytest.mark.parametrize("limit", [0.0, -1.0, float("inf"), float("nan")])
def test_the_backend_refuses_a_start_up_limit_that_is_not_positive_and_finite(
    limit: float,
) -> None:
    with pytest.raises(ConfigError, match="positive, finite start-up limit"):
        ProcessBackend(1, startup_limit=limit)
    with pytest.raises(ConfigError, match="positive, finite start-up limit"):
        backend_for("process", 1, startup_limit=limit)


def _task(hard_limit: float | None = 2.0) -> Task:
    return Task(
        run_id="run-1",
        attempt=1,
        problem=random_problem(1, tasks=6, machines=2),
        digest="0" * 64,
        solver=InputOrder(),
        stop=StopCriteria(time_limit=1.0),
        seed=1,
        hard_limit=hard_limit,
    )


def _waiting_slot(task: Task, since: float) -> _Slot:
    """Return a slot whose task was handed over ``since`` seconds ago and has not begun."""
    slot = _Slot(1, multiprocessing.get_context("spawn"), {}, None, 60.0)
    slot.task = task
    slot._clock = time.perf_counter() - since
    slot.deadline = slot._clock + 60.0
    return slot


def test_the_hard_limit_and_the_wait_count_from_the_instant_the_worker_states() -> None:
    task = _task()
    slot = _waiting_slot(task, 3.0)
    handed = slot._clock
    slot._begin((task.run_id, task.attempt, handed + 1.0, "2026-01-01T00:00:01.000000Z"))
    assert slot._waited == pytest.approx(1.0)
    assert slot.deadline == pytest.approx(handed + 1.0 + 2.0)
    assert slot._started == "2026-01-01T00:00:01.000000Z"
    assert slot.expired()  # the solve began 2 s ago, although its report was read just now
    measured = slot._measurement(task)
    assert measured.waited_seconds == pytest.approx(1.0)
    assert measured.hard_limit == 2.0
    assert measured.wall_seconds == pytest.approx(2.0, abs=0.2)


@pytest.mark.parametrize("offset", [-1.0, 3600.0])
def test_an_instant_outside_the_wait_is_replaced_by_the_reading_time(offset: float) -> None:
    task = _task()
    slot = _waiting_slot(task, 3.0)
    handed = slot._clock
    slot._begin((task.run_id, task.attempt, handed + offset, "2026-01-01T00:00:01.000000Z"))
    assert slot._waited == pytest.approx(3.0, abs=0.2)
    assert slot.deadline == pytest.approx(handed + 3.0 + 2.0, abs=0.2)
    assert slot._started != "2026-01-01T00:00:01.000000Z"
    assert not slot.expired()


def test_a_report_of_another_attempt_does_not_start_the_hard_limit() -> None:
    task = _task()
    slot = _waiting_slot(task, 3.0)
    deadline = slot.deadline
    for report in (
        (task.run_id, task.attempt + 1, slot._clock + 1.0, "2026-01-01T00:00:01.000000Z"),
        (task.run_id, task.attempt),
        None,
    ):
        slot._begin(report)
        assert (slot._solving, slot._waited, slot.deadline) == (False, None, deadline)


def test_the_supervisor_keeps_the_last_incumbent_and_the_one_within_the_time_limit() -> None:
    task = _task()
    slot = _waiting_slot(task, 0.0)
    early = replace(INCUMBENT, elapsed=0.5).to_dict()
    late = replace(INCUMBENT, value=11.0, elapsed=1.5).to_dict()
    slot._keep((task.run_id, task.attempt, [(early, True)]))
    assert slot._incumbent == slot._limit_incumbent == Incumbent.from_dict(early)
    slot._keep((task.run_id, task.attempt, [(late, False)]))
    assert slot._incumbent == Incumbent.from_dict(late)
    assert slot._limit_incumbent == Incumbent.from_dict(early)
    # A report marked as within the limit whose search ran longer is not kept as such,
    # and a malformed one or one of another attempt changes nothing.
    slot._keep((task.run_id, task.attempt, [(late, True), ({"value": 1.0}, True), "x"]))
    slot._keep((task.run_id, task.attempt + 1, [(early, True)]))
    slot._keep((task.run_id, task.attempt, early))
    assert slot._incumbent == Incumbent.from_dict(late)
    assert slot._limit_incumbent == Incumbent.from_dict(early)


# ------------------------------------------------------------ the reporter


class _Sink:
    """A connection that keeps what it is sent, and which thread sent it."""

    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.senders: list[int] = []

    def send(self, message: Any) -> None:
        self.sent.append(message)
        self.senders.append(threading.get_ident())


TOY = SolverInfo(id="toy", name="Toy", family="random-sampling", summary="Evaluates one order.")


def _improving(count: int, reporter: _Reporter | None = None) -> float:
    """Time a search whose every evaluation is a new incumbent."""
    problem = random_problem(1, tasks=6, machines=2)
    order = list(range(problem.size))
    clock = time.perf_counter()
    stop = StopCriteria(max_evaluations=count + 1)
    if reporter is None:
        search = Search(problem, stop, 1, TOY)
    else:
        with listening(reporter.note):
            search = Search(problem, stop, 1, TOY)
    for _ in range(count):
        search.best = None  # so the next candidate improves on it
        search.evaluate(order)
    return time.perf_counter() - clock


def _improvements(count: int) -> list[Improvement]:
    """Return ``count`` successive incumbents of one search."""
    heard: list[Improvement] = []
    with listening(heard.append):
        search = Search(
            random_problem(1, tasks=6, machines=2), StopCriteria(max_evaluations=count + 1), 1, TOY
        )
    for _ in range(count):
        search.best = None
        search.evaluate(list(range(search.problem.size)))
    return heard


def _reports(sink: _Sink) -> list[tuple[int, bool]]:
    """Return the evaluation count and the within-limit mark of every incumbent sent."""
    sent: list[tuple[int, bool]] = []
    for kind, (run_id, attempt_number, documents) in sink.sent:
        assert (kind, run_id, attempt_number) == ("progress", "run-1", 3)
        sent.extend(
            (Incumbent.from_dict(document).evaluations, within) for document, within in documents
        )
    return sent


def test_the_reporter_forwards_the_latest_incumbent_at_a_bounded_rate() -> None:
    improvements = _improvements(40)
    sink = _Sink()
    interval = 0.2
    noted = 0
    begun = time.perf_counter()
    with _Reporter(sink, "run-1", 3, begun, None, interval) as reporter:  # type: ignore[arg-type]
        while time.perf_counter() - begun < 3 * interval:
            reporter.note(improvements[noted % len(improvements)])
            noted += 1
        busy = len(sink.sent)
        time.sleep(2 * interval)
    assert noted > 100
    assert 3 <= busy <= 5
    assert all(len(message[1][2]) == 1 for message in sink.sent)
    last = improvements[(noted - 1) % len(improvements)]
    assert _reports(sink)[0] == (1, True)
    assert _reports(sink)[-1] == (last.point.evaluations, True)
    count = len(sink.sent)
    reporter.note(improvements[0])
    time.sleep(interval)
    assert len(sink.sent) == count  # nothing leaves a closed reporter


def test_the_reporter_sends_a_due_incumbent_from_the_thread_of_the_search() -> None:
    first, second, third = _improvements(3)
    sink = _Sink()
    interval = 0.3
    begun = time.perf_counter()
    with _Reporter(sink, "run-1", 3, begun, None, interval) as reporter:  # type: ignore[arg-type]
        reporter.note(first)
        assert _reports(sink) == [(1, True)]  # sent before note returned
        reporter.note(second)
        assert _reports(sink) == [(1, True)]  # too soon: left to the forwarding thread
        time.sleep(3 * interval)
        assert _reports(sink) == [(1, True), (2, True)]
        reporter.note(third)
        assert _reports(sink) == [(1, True), (2, True), (3, True)]
    here = threading.get_ident()
    assert [sender == here for sender in sink.senders] == [True, False, True]


def test_the_reporter_keeps_apart_the_incumbent_found_within_the_time_limit() -> None:
    first, second, third, fourth = _improvements(4)
    sink = _Sink()
    limit, interval = 0.2, 0.5
    begun = time.perf_counter()
    with _Reporter(sink, "run-1", 3, begun, limit, interval) as reporter:  # type: ignore[arg-type]
        reporter.note(first)  # sent at once, within the limit
        reporter.note(second)  # within the limit, and waits for the interval
        time.sleep(limit - (time.perf_counter() - begun) + 0.05)
        reporter.note(third)  # past the limit, in the same report as the second
        time.sleep(2 * interval)
        reporter.note(fourth)
    assert _reports(sink) == [(1, True), (2, True), (3, False), (4, False)]
    assert [len(message[1][2]) for message in sink.sent] == [1, 2, 1]


class _Overtaken(threading.Event):
    """An event whose first timed wait reports a timeout although the event was set meanwhile.

    ``threading.Event.wait`` may do so when its timeout and the setting fall together: the
    waiting thread has timed out, and another thread sets the event before the waiting one
    runs again. The first wait here stays in that state until ``resume`` is set.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.resume = threading.Event()

    def wait(self, timeout: float | None = None) -> bool:
        if self.entered.is_set():
            return super().wait(timeout)
        self.entered.set()
        self.resume.wait(60.0)
        return False


def test_closing_the_reporter_ends_its_thread_when_a_due_report_and_the_closing_fall_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _improvements(2)
    sink = _Sink()
    closed = _Overtaken()
    reporter = _Reporter(sink, "run-1", 3, time.perf_counter(), None, 60.0)  # type: ignore[arg-type]
    reporter._closed = closed
    woken = threading.Event()
    set_news = reporter._news.set

    def wake() -> None:
        set_news()
        if closed.is_set():
            woken.set()

    monkeypatch.setattr(reporter._news, "set", wake)
    reporter.__enter__()
    reporter.note(first)  # sent at once: the next report is due a minute later
    reporter.note(second)  # kept: the thread waits for the due time
    assert closed.entered.wait(60.0)
    # The search ends, and its thread closes the reporter, while the thread of the
    # reporter has timed out and not yet run again.
    closing = threading.Thread(target=reporter.__exit__, args=(None, None, None), daemon=True)
    closing.start()
    assert woken.wait(60.0)  # the closing has set both events and waits for the thread
    closed.resume.set()
    closing.join(10.0)
    # A reporter that lost the wake-up of its closing would keep the worker from ever
    # sending the result of the run, which would end at its hard limit.
    assert not closing.is_alive()
    assert _reports(sink) == [(1, True)]


def test_reporting_does_not_slow_a_search_whose_every_evaluation_improves() -> None:
    count = 1500
    _improving(200)
    plain: list[float] = []
    reported: list[float] = []
    sent = 0
    for _ in range(5):
        plain.append(_improving(count))
        sink = _Sink()
        begun = time.perf_counter()
        with _Reporter(sink, "run-1", 1, begun, 0.01) as reporter:  # type: ignore[arg-type]
            reported.append(_improving(count, reporter))
        sent = max(sent, len(sink.sent))
        assert len(sink.sent) <= reported[-1] / PROGRESS_INTERVAL + 2
    assert sent >= 1
    assert min(reported) <= 1.25 * min(plain)
