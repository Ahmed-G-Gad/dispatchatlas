"""Start-up and hard limits: when they count from, and what a killed run keeps."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
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
from dispatchatlas.campaigns.backends import PROGRESS_INTERVAL, ProcessBackend, _Reporter
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
from dispatchatlas.core import ConfigError, SerializationError, SolverInfo, StopCriteria
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
    assert 0.5 <= failure.provenance.wall_seconds < 20.0
    incumbent = failure.last_incumbent
    assert incumbent is not None
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
        assert _gone(supervisor)
        assert _gone(worker)
        assert _gone(child)
    finally:
        launched.kill()
        launched.wait(30)


def test_the_backend_refuses_a_start_up_limit_that_is_not_positive() -> None:
    with pytest.raises(ConfigError, match="positive start-up limit"):
        ProcessBackend(1, startup_limit=0.0)


# ------------------------------------------------------------ the reporter


class _Sink:
    """A connection that keeps what it is sent."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    def send(self, message: Any) -> None:
        self.sent.append(message)


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


def test_the_reporter_forwards_the_latest_incumbent_at_a_bounded_rate() -> None:
    heard: list[Improvement] = []
    with listening(heard.append):
        search = Search(
            random_problem(1, tasks=6, machines=2), StopCriteria(max_evaluations=3), 1, TOY
        )
    for _ in range(2):
        search.best = None
        search.evaluate(list(range(search.problem.size)))
    first, second = heard
    sink = _Sink()
    interval = 0.2
    with _Reporter(sink, "run-1", 3, interval) as reporter:  # type: ignore[arg-type]
        started = time.perf_counter()
        reporter.note(first)
        while time.perf_counter() - started < 3 * interval:
            reporter.note(second)
        busy = len(sink.sent)
        time.sleep(2 * interval)
    assert 1 <= busy <= 5
    kinds = {kind for kind, _ in sink.sent}
    assert kinds == {"progress"}
    run_id, attempt_number, document = sink.sent[-1][1]
    assert (run_id, attempt_number) == ("run-1", 3)
    assert Incumbent.from_dict(document).evaluations == second.point.evaluations == 2
    count = len(sink.sent)
    reporter.note(first)
    time.sleep(interval)
    assert len(sink.sent) == count  # nothing leaves a closed reporter


def test_reporting_does_not_slow_a_search_whose_every_evaluation_improves() -> None:
    count = 1500
    _improving(200)
    plain: list[float] = []
    reported: list[float] = []
    sent = 0
    for _ in range(5):
        plain.append(_improving(count))
        sink = _Sink()
        with _Reporter(sink, "run-1", 1) as reporter:  # type: ignore[arg-type]
            reported.append(_improving(count, reporter))
        sent = max(sent, len(sink.sent))
        assert len(sink.sent) <= reported[-1] / PROGRESS_INTERVAL + 2
    assert sent >= 1
    assert min(reported) <= 1.25 * min(plain)
