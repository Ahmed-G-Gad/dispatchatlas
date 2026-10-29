from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.core import ConfigError
from dispatchatlas.processes import affinity, calls
from dispatchatlas.processes.affinity import pinning_refusal_text
from dispatchatlas.processes.calls import (
    Call,
    CallError,
    CallFacts,
    CallsStoppedError,
    interleaved,
    run_calls,
)
from dispatchatlas.processes.rows import RowWriter, TornRowsError, read_rows
from tests.processes.helpers import ROOT, gone, slot_cpus

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
    clash = _run([_echo(7, revision="mine")], rows, stamp={"revision": "x"})
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
