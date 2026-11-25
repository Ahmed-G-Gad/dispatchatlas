from __future__ import annotations

import contextlib
import json
import os
import select
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.core import ConfigError
from dispatchatlas.processes import affinity, calls
from dispatchatlas.processes.affinity import current_mask, mask_of_cpus, pinning_refusal_text
from dispatchatlas.processes.calls import (
    Call,
    CallError,
    CallFacts,
    CallReport,
    CallsStoppedError,
    interleaved,
    run_calls,
)
from dispatchatlas.processes.rows import RowWriter, TornRowsError, read_rows
from dispatchatlas.processes.snapshots import descendants, is_running, snapshot
from tests.processes.helpers import (
    ROOT,
    gone,
    needs_pinning,
    posix_only,
    slot_cpus,
    wait_for,
)

PIN = pinning_refusal_text() is None
CPUS = slot_cpus(2)
WORKERS = len(CPUS)
TARGETS = "tests.processes.child_targets"


def _key(row: Mapping[str, Any]) -> Any:
    return row["key"]


def _echo(index: int, **extra: Any) -> Call:
    return Call(
        key=["e", index], target=f"{TARGETS}:echo", arguments={"key": ["e", index], **extra}
    )


def _run(work: list[Call], rows: Path, **options: Any) -> Any:
    settings: dict[str, Any] = {
        "key_of": _key,
        "cpus": CPUS,
        "pin": PIN,
        "cwd": ROOT,
        "max_failures": 0,
    }
    settings.update(options)
    return run_calls(work, rows, **settings)


def _errors(rows: Path) -> list[dict[str, Any]]:
    path = rows.with_name(f"{rows.stem}.errors.jsonl")
    return list(read_rows(path)) if path.exists() else []


def test_every_call_writes_one_row_with_its_facts(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    report = _run([_echo(index) for index in range(6)], rows, stamp={"revision": "abc"})
    written = list(read_rows(rows))
    assert (report.rows_written, report.done_before, report.failures) == (6, 0, 0)
    assert sorted(row["key"][1] for row in written) == list(range(6))
    for row in written:
        run = row["run"]
        assert row["revision"] == "abc"
        assert run["workers"] == WORKERS
        assert 1 <= run["workers_live_min"] <= run["workers_live_start"] <= WORKERS
        assert run["ballast"] is False
        assert run["attempt"] == 1
        if PIN:
            assert run["cpu"] in CPUS
            assert run["affinity_mask"] == 1 << run["cpu"]
            assert run["affinity_inherited"] == 1 << run["cpu"]  # created on its processor
    assert _errors(rows) == []
    stored = json.loads(rows.with_name("rows.report.json").read_text(encoding="utf-8"))
    assert stored["rows_written"] == 6


def test_a_second_invocation_runs_only_what_has_no_row(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    work = [_echo(index) for index in range(6)]
    _run(work[:3], rows)
    report = _run(work, rows)
    assert (report.done_before, report.rows_written) == (3, 3)
    keys = [tuple(row["key"]) for row in read_rows(rows)]
    assert len(keys) == len(set(keys)) == 6


def test_a_torn_rows_file_is_refused_before_anything_starts(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b'{"key":["e",0]}\n{"key":')
    with pytest.raises(TornRowsError):
        _run([_echo(1)], rows)
    assert not rows.with_name("rows.children.log").exists()


def test_a_failure_goes_to_the_sidecar_and_is_retried_on_resume(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    message = r"Secret failed reading C:\data\secret\in.csv"
    bad = Call(key=["f", 0], target=f"{TARGETS}:fail", arguments={"message": message})
    work = [_echo(0), bad, _echo(1)]
    report = _run(work, rows, clean_tokens=["secret"])
    assert (report.rows_written, report.failures) == (2, 1)
    (record,) = _errors(rows)
    error = CallError.from_dict(record)
    assert (error.type, error.attempt, error.returncode, error.last_line) == (
        "ValueError",
        1,
        1,
        True,
    )
    assert error.message == "<token> failed reading <path>"
    assert all(row["key"][0] == "e" for row in read_rows(rows))
    again = _run(work, rows)
    assert (again.done_before, again.failures) == (2, 1)
    assert [record["attempt"] for record in _errors(rows)] == [1, 2]
    skipped = _run(work, rows, skip_failed=True)
    assert (skipped.skipped_failed, skipped.failures) == (1, 0)


def test_a_child_that_dies_is_recorded_as_such(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    _run([Call(key=["d", 0], target=f"{TARGETS}:die", arguments={"code": 7})], rows)
    (record,) = _errors(rows)
    assert (record["type"], record["returncode"], record["last_line"]) == ("ProcessDied", 7, False)


@pytest.mark.slow
def test_max_failures_stops_the_run_at_once_and_ends_every_child(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    marker = tmp_path / "sleeper.pid"
    work = [
        Call(
            key=["s", 0],
            target=f"{TARGETS}:sleep",
            arguments={"seconds": 120, "marker": str(marker)},
        ),
        Call(key=["f", 0], target=f"{TARGETS}:fail", arguments={"message": "one"}),
        Call(key=["f", 1], target=f"{TARGETS}:fail", arguments={"message": "two"}),
        *[_echo(index) for index in range(20)],
    ]
    with pytest.raises(CallsStoppedError) as stopped:
        _run(work, rows, max_failures=1)
    report = stopped.value.report
    assert report.stopped == "max-failures"
    assert report.failures == 1
    assert report.rows_written < 20
    if marker.exists():
        assert gone(int(marker.read_text(encoding="ascii")))
    assert json.loads(rows.with_name("rows.report.json").read_text())["stopped"] == "max-failures"


def test_a_call_past_its_time_limit_is_ended(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    marker = tmp_path / "sleeper.pid"
    work = [
        Call(
            key=["s", 0],
            target=f"{TARGETS}:sleep",
            arguments={"seconds": 120, "marker": str(marker)},
            timeout_seconds=10.0,
        )
    ]
    _run(work, rows)
    (record,) = _errors(rows)
    assert record["type"] == "Timeout"
    if marker.exists():
        assert gone(int(marker.read_text(encoding="ascii")))


@pytest.mark.slow
def test_ballast_keeps_every_slot_busy_until_the_last_real_row(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    work = [_echo(index) for index in range(3)]
    work.append(
        Call(key=["e", 9], target=f"{TARGETS}:sleep", arguments={"seconds": 4, "key": ["e", 9]})
    )
    report = _run(work, rows, ballast=True)
    written = list(read_rows(rows))
    assert len(written) == 4
    assert all(row["run"]["workers_live_min"] == WORKERS for row in written)
    assert report.ballast_runs > 0
    ballast = list(read_rows(rows.with_name("rows.ballast.jsonl")))
    assert ballast
    assert {record["outcome"] for record in ballast} <= {"row", "stopped", "error"}


def test_finish_shapes_the_row_and_a_wrong_key_is_refused(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"

    def flatten(row: dict[str, Any], facts: CallFacts) -> dict[str, Any]:
        return {**row, "workers_live_min": facts.workers_live_min, "pid": facts.pid}

    _run([_echo(0)], rows, finish=flatten)
    (row,) = list(read_rows(rows))
    assert "run" not in row
    assert row["workers_live_min"] >= 1
    mislabelled = Call(key=["e", 5], target=f"{TARGETS}:echo", arguments={"key": ["e", 6]})
    report = _run([mislabelled], rows)
    assert report.rows_written == 0
    (record,) = _errors(rows)
    assert record["type"] == "KeyMismatch"
    clash = _run([_echo(7, revision="mine")], rows, stamp={"revision": "x"}, allow_changed_run=True)
    assert clash.failures == 1
    assert _errors(rows)[-1]["type"] == "RowRefused"


def test_interleaving_is_seeded_and_keeps_classes_in_order() -> None:
    work = [Call(key=index, target=f"{TARGETS}:echo") for index in range(20)]
    first = interleaved(work, seed=1)
    assert first == interleaved(work, seed=1)
    assert first != interleaved(work, seed=2)
    assert sorted(int(call.key) for call in first) == list(range(20))  # type: ignore[arg-type]
    classed = interleaved(work, seed=1, classes=lambda call: 0 if call.key in range(15, 20) else 1)
    assert {call.key for call in classed[:5]} == set(range(15, 20))
    assert [call.key for call in work] == list(range(20))


def test_invalid_runs_are_refused_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = tmp_path / "rows.jsonl"
    with pytest.raises(ConfigError, match="not distinct"):
        _run([_echo(0), _echo(0)], rows)
    with pytest.raises(ConfigError, match="workers"):
        _run([_echo(0)], rows, workers=len(CPUS) + 1)
    with pytest.raises(ConfigError, match="not distinct"):
        _run([_echo(0)], rows, cpus=[CPUS[0], CPUS[0]])
    if PIN:
        with pytest.raises(ConfigError, match="outside"):
            _run([_echo(0)], rows, cpus=[(os.cpu_count() or 1) + 8])
    monkeypatch.setattr(calls, "pinning_refusal_text", lambda: "no affinity call")
    with pytest.raises(ConfigError, match="no affinity call"):
        _run([_echo(0)], rows, pin=True)
    assert list(tmp_path.iterdir()) == []
    assert affinity.pinning_refusal_text() == pinning_refusal_text()


def test_children_get_the_thread_limits(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    call = Call(key=["t", 0], target=f"{TARGETS}:records_env", arguments={"key": ["t", 0]})
    _run([call], rows, workers=1)
    (row,) = list(read_rows(rows))
    assert set(row["threads"].values()) == {"1"}
    assert len(row["threads"]) == 6
    assert row["unbuffered"] == "1"
    if PIN:
        assert row["mask"] == 1 << row["run"]["cpu"]


def test_an_error_record_survives_a_round_trip(tmp_path: Path) -> None:
    record = CallError(
        key=["a", 1],
        attempt=2,
        type="Timeout",
        message="m",
        returncode=None,
        last_line=False,
        facts=None,
        cpu=3,
        started_utc="s",
        ended_utc="e",
    )
    path = tmp_path / "errors.jsonl"
    with RowWriter(path, fsync=False) as writer:
        writer.write(record.to_dict())
    (read,) = list(read_rows(path))
    assert CallError.from_dict(read) == record


# ------------------------------------------------------------ hardening


def _within(seconds: float, action: Callable[[], Any]) -> Any:
    """Run ``action`` in a thread; fail, rather than hang, if it has not ended in time."""
    outcome: dict[str, Any] = {}

    def body() -> None:
        try:
            outcome["value"] = action()
        except BaseException as error:  # handed back to the test below
            outcome["error"] = error

    worker = threading.Thread(target=body, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), f"the run had not returned after {seconds} s"
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _sleep_call(key: list[Any], seconds: float, **extra: Any) -> Call:
    return Call(
        key=key, target=f"{TARGETS}:sleep", arguments={"seconds": seconds, "key": key, **extra}
    )


def _unpinned(slots: int) -> dict[str, Any]:
    return {"cpus": list(range(slots)), "pin": False}


def test_a_resume_on_other_processors_is_refused_without_pinning(tmp_path: Path) -> None:
    # An unpinned run still names its processors, so a changed set is a changed run.
    rows = tmp_path / "rows.jsonl"
    _run([_echo(0)], rows, cpus=[0, 1], pin=False)
    before = rows.read_bytes()
    with pytest.raises(ConfigError, match=r"cpus: \[0, 1\] then, \[2, 3\] now"):
        _run([_echo(1)], rows, cpus=[2, 3], pin=False)
    with pytest.raises(ConfigError, match=r"cpus: \[0, 1\] then, \[1, 0\] now"):
        _run([_echo(1)], rows, cpus=[1, 0], pin=False)
    assert rows.read_bytes() == before
    assert _run([_echo(1)], rows, cpus=[0, 1], pin=False).rows_written == 1


@posix_only
def test_a_run_ends_where_the_platform_has_no_waitid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(os, "waitid", raising=False)
    rows = tmp_path / "rows.jsonl"
    report = _within(60, lambda: _run([_echo(0), _echo(1)], rows, **_unpinned(2)))
    assert report.rows_written == 2


def test_waiting_without_waitid_or_kqueue_still_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "waitid", raising=False)
    monkeypatch.delattr(select, "kqueue", raising=False)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.2)"], stdin=subprocess.DEVNULL
    )
    _within(60, lambda: calls._wait_unreaped(process))
    assert process.poll() is not None


def test_a_failure_while_attending_a_child_is_recorded_not_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def lost(*arguments: Any) -> None:
        raise RuntimeError("lost the child")

    monkeypatch.setattr(calls, "_await_child", lost)
    rows = tmp_path / "rows.jsonl"
    report = _within(60, lambda: _run([_sleep_call(["s", 0], 30.0)], rows))
    assert (report.rows_written, report.failures) == (0, 1)
    (record,) = _errors(rows)
    assert record["type"] == "AttendFailed"
    assert "lost the child" in record["message"]


def test_output_after_the_reply_cannot_replace_it(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    work = [
        Call(key=["x", 0], target=f"{TARGETS}:prints_at_exit", arguments={"key": ["x", 0]}),
        Call(key=["x", 1], target=f"{TARGETS}:helper_after_reply", arguments={"key": ["x", 1]}),
    ]
    report = _run(work, rows)
    assert (report.rows_written, report.failures) == (2, 0)
    assert _errors(rows) == []
    log = rows.with_name("rows.children.log").read_text(encoding="utf-8", errors="replace")
    assert "late" in log
    assert "printed at exit" in log


def test_a_second_runner_on_the_same_rows_is_refused(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    marker = tmp_path / "first.pid"
    first: dict[str, Any] = {}

    def run_first() -> None:
        first["report"] = _run([_sleep_call(["s", 0], 3.0, marker=str(marker))], rows)

    worker = threading.Thread(target=run_first)
    worker.start()
    try:
        wait_for(marker)
        with pytest.raises(ConfigError, match="in use by another runner") as refused:
            _run([_echo(5)], rows)
        assert f"pid {os.getpid()}" in str(refused.value)
    finally:
        worker.join(120)
    assert first["report"].rows_written == 1
    assert [row["key"] for row in read_rows(rows)] == [["s", 0]]
    assert _run([_echo(5)], rows).rows_written == 1  # the lock ended with the first run


def test_a_rows_file_with_a_repeated_key_is_refused(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    with RowWriter(rows, fsync=False) as writer:
        for key in (1, 2, 1):
            writer.write({"key": key})
    with pytest.raises(ConfigError, match=r"1 key\(s\): 1;"):
        _run([_echo(3)], rows)
    assert not rows.with_name("rows.children.log").exists()


def test_the_error_type_loses_the_callers_tokens(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    call = Call(key=["r", 0], target=f"{TARGETS}:raises_secret")
    _run([call], rows, clean_tokens=["secret"])
    (record,) = _errors(rows)
    assert "secret" not in record["type"].lower()
    assert "<token>" in record["type"]
    assert "secret" not in record["message"].lower()


def test_a_resume_that_differs_from_the_first_run_is_refused(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    _run([_echo(0)], rows, stamp={"rev": "a"})
    with pytest.raises(ConfigError, match="stamp rev: 'a' then, 'b' now"):
        _run([_echo(0), _echo(1)], rows, stamp={"rev": "b"})
    assert len(list(read_rows(rows))) == 1
    mixed = _run([_echo(0), _echo(1)], rows, stamp={"rev": "b"}, allow_changed_run=True)
    assert mixed.rows_written == 1
    runs = list(read_rows(rows.with_name("rows.runs.jsonl")))
    assert [run["stamp"] for run in runs] == [{"rev": "a"}, {"rev": "b"}]

    other = tmp_path / "other.jsonl"
    _run([_echo(0)], other, **_unpinned(2))
    with pytest.raises(ConfigError, match="workers: 2 then, 1 now"):
        _run([_echo(0), _echo(1)], other, workers=1, **_unpinned(2))

    older = tmp_path / "older.jsonl"  # written before the runs sidecar existed
    with RowWriter(older, fsync=False) as writer:
        writer.write({"key": ["e", 0], "rev": "a"})
    with pytest.raises(ConfigError, match="stamp rev: 'a' in the rows, 'b' now"):
        _run([_echo(1)], older, stamp={"rev": "b"})


@needs_pinning
def test_the_runner_is_pinned_to_the_whole_set(tmp_path: Path) -> None:
    whole = slot_cpus(2)
    if len(whole) < 2:
        pytest.skip("needs two processors")
    before = current_mask()
    seen: list[int | None] = []

    def finish(row: dict[str, Any], facts: CallFacts) -> dict[str, Any]:
        seen.append(current_mask())
        return row

    _run([_echo(0)], tmp_path / "rows.jsonl", cpus=whole, workers=1, finish=finish)
    assert seen == [mask_of_cpus(whole)]
    assert current_mask() == before


@pytest.mark.slow
def test_ballast_rows_never_reach_the_rows_file_and_keep_their_order(tmp_path: Path) -> None:
    if WORKERS < 2:
        pytest.skip("needs two slots")
    rows = tmp_path / "rows.jsonl"
    echoes = [_echo(index) for index in range(3)]
    report = _run([_sleep_call(["e", 9], 4.0), *echoes], rows, ballast=echoes)
    keys = sorted(tuple(row["key"]) for row in read_rows(rows))
    assert keys == [("e", 0), ("e", 1), ("e", 2), ("e", 9)]
    assert report.ballast_runs > 0
    ballast = list(read_rows(rows.with_name("rows.ballast.jsonl")))
    assert "row" in {record["outcome"] for record in ballast}
    assert ballast[0]["key"] == ["e", 0]  # the given order, not a shuffle


def test_a_call_past_its_limit_ends_with_its_grandchild(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    marker = tmp_path / "pids"
    call = Call(
        key=["g", 0],
        target=f"{TARGETS}:spawn_grandchild",
        arguments={"marker": str(marker), "seconds": 60},
        timeout_seconds=5.0,
    )
    began = time.monotonic()
    _within(60, lambda: _run([call], rows))
    assert time.monotonic() - began < 30
    child, grandchild = map(int, marker.read_text(encoding="ascii").split())
    assert gone(child)
    assert gone(grandchild)
    assert _errors(rows)[0]["type"] == "Timeout"


def test_max_failures_ends_every_running_child_and_grandchild(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    markers = [tmp_path / "a", tmp_path / "b"]
    work = [
        Call(
            key=["g", index],
            target=f"{TARGETS}:spawn_grandchild",
            arguments={"marker": str(marker), "seconds": 60},
        )
        for index, marker in enumerate(markers)
    ]
    work.append(
        Call(key=["f", 0], target=f"{TARGETS}:fail_after", arguments={"seconds": 6, "message": "x"})
    )
    began = time.monotonic()

    def run() -> None:
        with pytest.raises(CallsStoppedError):
            _run(work, rows, max_failures=1, **_unpinned(3))

    _within(90, run)
    assert time.monotonic() - began < 40
    for marker in markers:
        for pid in map(int, marker.read_text(encoding="ascii").split()):
            assert gone(pid)


def test_a_failing_finish_is_recorded_without_paths_or_tokens(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"

    def finish(row: dict[str, Any], facts: CallFacts) -> dict[str, Any]:
        raise ValueError(f"bad {tmp_path} secret")

    _run([_echo(0)], rows, finish=finish, clean_tokens=["secret"])
    (record,) = _errors(rows)
    assert record["type"] == "RowRefused"
    assert str(tmp_path) not in record["message"]
    assert tmp_path.name not in record["message"]
    assert "secret" not in record["message"]
    assert "<token>" in record["message"]


def test_the_live_counts_follow_the_slots(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    _run([_sleep_call(["s", 0], 2.0), _echo(0)], rows, **_unpinned(2))
    (slow,) = [row for row in read_rows(rows) if row["key"] == ["s", 0]]
    assert (slow["run"]["workers_live_start"], slow["run"]["workers_live_min"]) == (2, 1)


@needs_pinning
def test_a_start_that_fails_after_the_child_exists_leaves_no_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = affinity.confined

    @contextlib.contextmanager
    def restore_fails(cpus: Any) -> Iterator[None]:
        with real(cpus):
            yield
        if isinstance(cpus, int):  # a slot's confinement, not the runner's
            raise RuntimeError("cannot return to the runner's processors")

    monkeypatch.setattr(calls, "confined", restore_fails)
    before = {info.pid for info in descendants(os.getpid(), snapshot())}
    rows = tmp_path / "rows.jsonl"
    report: CallReport = _within(60, lambda: _run([_sleep_call(["s", 0], 30.0)], rows))
    assert report.failures == 1
    assert _errors(rows)[0]["type"] == "StartFailed"
    left = [
        info.pid
        for info in descendants(os.getpid(), snapshot())
        if info.pid not in before and is_running(info.pid)
    ]
    assert all(gone(pid, within=5) for pid in left), left


def test_ballast_that_keeps_failing_stops(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    broken = Call(key=["b", 0], target=f"{TARGETS}:fail", arguments={"message": "no"})
    report = _run([_sleep_call(["s", 0], 3.0)], rows, ballast=[broken], **_unpinned(2))
    assert report.rows_written == 1
    assert 1 <= report.ballast_runs <= 3 + report.workers
    assert report.failures == 0


def test_an_explicit_ballast_seed_orders_the_repeats(tmp_path: Path) -> None:
    work = [_sleep_call(["s", 0], 3.0), *[_echo(index) for index in range(6)]]
    seeded = interleaved(work, seed=11)
    rows = tmp_path / "rows.jsonl"
    _run(work, rows, ballast=True, ballast_seed=11, **_unpinned(2))
    ballast = list(read_rows(rows.with_name("rows.ballast.jsonl")))
    assert ballast
    assert ballast[0]["key"] == seeded[0].key
