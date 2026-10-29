from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from dispatchatlas.processes import watching
from dispatchatlas.processes.snapshots import ProcessInfo, descendants, end_process, snapshot
from dispatchatlas.processes.watching import format_status, watch
from tests.processes.helpers import sleeper, slot_cpus


def test_the_watch_follows_the_runner_and_its_rows_until_it_ends(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b'{"a":1}\n')
    runner = sleeper(6.0)

    def append() -> None:
        time.sleep(1.0)
        with rows.open("ab") as stream:
            stream.write(b'{"a":2}\n{"a":3}\n')

    writer = threading.Thread(target=append)
    writer.start()
    statuses = list(watch(runner.pid, rows=rows, cpus=slot_cpus(2), interval=0.5, until=120))
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
    shell, own_launcher, runner_launcher, runner, child, stranger = (
        900_000,
        900_001,
        900_010,
        900_011,
        900_012,
        900_020,
    )

    def info(pid: int, parent: int, name: str) -> ProcessInfo:
        return ProcessInfo(pid, parent, name, None, 0b1, None)

    listed = (
        info(shell, 1, "pwsh.exe"),
        info(own_launcher, shell, "python.exe"),
        info(own, own_launcher, "python.exe"),
        info(runner_launcher, shell, "python.exe"),
        info(runner, runner_launcher, "python.exe"),
        info(child, runner, "python.exe"),
        info(stranger, shell, "python.exe"),
    )
    monkeypatch.setattr(watching, "snapshot", lambda: listed)
    monkeypatch.setattr(watching, "is_running", lambda pid: True)
    (status,) = watch(runner, rows=tmp_path / "rows.jsonl", cpus=[0], interval=0.0, until=1)
    assert status.foreign_pids == (stranger,)
    assert (status.live_children, status.live_interpreters) == (1, 1)
    assert "interpreters=1 " in format_status(status)


def test_the_watch_counts_children_and_sees_a_foreign_interpreter(tmp_path: Path) -> None:
    if not snapshot():
        pytest.skip("no process snapshot on this platform")
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
    orphaned = subprocess.run(
        [
            sys.executable,
            "-c",
            "import subprocess, sys; p = subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(60)'], stdin=subprocess.DEVNULL, "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print(p.pid)",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    stranger = int(orphaned.stdout.split()[-1])
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
