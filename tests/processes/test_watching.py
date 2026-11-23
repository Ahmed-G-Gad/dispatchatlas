from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from dispatchatlas.processes import watching
from dispatchatlas.processes.snapshots import ProcessInfo, descendants, end_process, snapshot
from dispatchatlas.processes.watching import format_status, watch
from tests.processes.helpers import console_script, needs_proc, orphan, sleeper, slot_cpus


def _info(pid: int, parent: int, name: str, executable: str | None = None) -> ProcessInfo:
    return ProcessInfo(pid, parent, name, executable, 0b1, None)


def test_the_watch_follows_the_runner_and_its_rows_until_it_ends(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b'{"a":1}\n')
    runner = sleeper(6.0)
    first = threading.Event()

    def append() -> None:
        # After the watch's first status, however slow the process scan of a loaded machine.
        first.wait(30)
        with rows.open("ab") as stream:
            stream.write(b'{"a":2}\n{"a":3}\n')

    writer = threading.Thread(target=append)
    writer.start()
    statuses = []
    for status in watch(runner.pid, rows=rows, cpus=slot_cpus(2), interval=0.5, until=120):
        statuses.append(status)
        first.set()
    writer.join()
    runner.wait(30)
    assert statuses[0].runner_alive
    assert statuses[0].rows_lines == 1
    assert not statuses[-1].runner_alive
    assert statuses[-1].rows_lines == 3
    assert statuses[-1].rows_bytes == rows.stat().st_size
    assert format_status(statuses[-1]).startswith("RUNNER-DEAD ")
    assert not format_status(statuses[0]).startswith("RUNNER-DEAD")
    assert all(
        status.available_memory_bytes is None or status.available_memory_bytes > 0
        for status in statuses
    )


def test_a_repaired_rows_file_is_counted_again(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b"a\nb\nc\n")
    runner = sleeper(60.0)
    try:
        polls = watch(runner.pid, rows=rows, cpus=slot_cpus(1), interval=0.1, until=2)
        assert next(polls).rows_lines == 3
        rows.write_bytes(b"a\n")
        assert next(polls).rows_lines == 1
    finally:
        runner.kill()
        runner.wait(30)


def test_launchers_above_the_watcher_and_the_runner_are_not_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = os.getpid()
    shell, trampoline, own_launcher, runner_launcher, runner, child, stranger = (
        900_000,
        900_001,
        900_002,
        900_010,
        900_011,
        900_012,
        900_020,
    )
    launcher, base = r"C:\venv\Scripts\python.exe", r"C:\Python313\python.exe"
    listed = (
        _info(shell, 1, "pwsh.exe"),
        # A console script's launcher, as uv run pytest starts; it is no interpreter.
        _info(trampoline, shell, "pytest.exe", r"C:\venv\Scripts\pytest.exe"),
        _info(own_launcher, trampoline, "python.exe", launcher),
        _info(own, own_launcher, "python.exe", base),
        _info(runner_launcher, shell, "python.exe", launcher),
        _info(runner, runner_launcher, "python.exe", base),
        _info(child, runner, "python.exe", base),
        _info(stranger, shell, "python.exe", base),
    )
    monkeypatch.setattr(watching, "snapshot", lambda: listed)
    monkeypatch.setattr(watching, "is_running", lambda pid: True)
    (status,) = watch(runner, rows=tmp_path / "rows.jsonl", cpus=[0], interval=0.0, until=1)
    assert status.foreign_pids == (stranger,)
    assert (status.live_children, status.live_interpreters) == (1, 1)
    assert "interpreters=1 " in format_status(status)


def test_console_scripts_are_interpreters_by_their_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = os.getpid()
    shell, study, runner, child, helper, stranger, other_user = (
        900_000,
        900_010,
        900_011,
        900_012,
        900_013,
        900_020,
        900_021,
    )
    python = "/usr/bin/python3.13"
    listed = (
        _info(shell, 1, "bash", "/usr/bin/bash"),
        _info(own, shell, "dispatchatlas", python),  # the watcher, run as a console script
        _info(study, shell, "mystudy", python),  # a console script that started the runner
        _info(runner, study, "python3", python),
        _info(child, runner, "solver", f"{python} (deleted)"),  # its interpreter replaced since
        _info(helper, runner, "sh", "/usr/bin/dash"),
        _info(stranger, shell, "pytest", python),  # a stray test run
        _info(other_user, 1, "unattended-upgr"),  # another user's: no executable to read
    )
    monkeypatch.setattr(watching, "snapshot", lambda: listed)
    monkeypatch.setattr(watching, "is_running", lambda pid: True)
    (status,) = watch(runner, rows=tmp_path / "rows.jsonl", cpus=[0], interval=0.0, until=1)
    assert status.foreign_pids == (stranger,)
    assert (status.live_children, status.live_interpreters) == (2, 1)


@pytest.mark.parametrize(
    "through_script", [False, pytest.param(True, marks=needs_proc)], ids=["python", "script"]
)
def test_the_watch_counts_children_and_sees_a_foreign_interpreter(
    tmp_path: Path, through_script: bool
) -> None:
    if not snapshot():
        pytest.skip("no process snapshot on this platform")
    # Started by its own path, a console script is named after itself on Linux.
    command = (
        [str(console_script(tmp_path))]
        if through_script
        else [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    runner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "time.sleep(30)",
        ],
        stdin=subprocess.DEVNULL,
    )
    stranger = orphan(command)
    stray = [stranger, *(info.pid for info in descendants(stranger))]
    try:
        found = False
        for status in watch(
            runner.pid, rows=tmp_path / "rows.jsonl", cpus=slot_cpus(2), interval=0.5, until=60
        ):
            if status.live_children > 0 and stranger in status.foreign_pids:
                found = True
                break
        assert found
    finally:
        for pid in [*stray, *(info.pid for info in descendants(runner.pid))]:
            end_process(pid)
        runner.kill()
        runner.wait(30)
