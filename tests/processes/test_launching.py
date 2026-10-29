from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.processes.launching import launch_detached
from dispatchatlas.processes.snapshots import end_process, is_running
from tests.processes.helpers import ROOT, gone, wait_for, windows_only


def _log_holds(log: Path, text: str, within: float = 60.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if log.exists() and text in log.read_text(encoding="utf-8", errors="replace"):
            return True
        time.sleep(0.1)
    return False


def test_a_detached_command_logs_its_output_and_reads_no_input(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "run.log"
    script = "import sys; print('hi', sys.stdin.read() == '')"
    launched = launch_detached(
        [sys.executable, "-c", script], log=log, cwd=tmp_path, environment={"EXTRA": "1"}
    )
    assert launched.pid > 0
    assert isinstance(launched.breakaway, bool)
    assert _log_holds(log, "hi True")
    assert log.read_text(encoding="utf-8").startswith("--- launched ")
    assert gone(launched.pid, within=60)


def test_a_detached_command_outlives_the_process_that_launched_it(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    marker = tmp_path / "pid"
    child = f"import os, time; open({str(marker)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dispatchatlas.processes.launching",
            "--log",
            str(log),
            "--cwd",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            child,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert completed.stdout.startswith("started pid ")
    launcher_pid = int(completed.stdout.split()[2].rstrip(","))
    interpreter = int(wait_for(marker))
    try:
        assert is_running(interpreter)
    finally:
        end_process(launcher_pid)
        end_process(interpreter)
    assert gone(interpreter)


def test_a_detached_command_reads_no_input_although_its_launcher_has_some(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    reader = "import sys; data = sys.stdin.read(); print('eof', repr(data), flush=True)"
    script = (
        "import sys\n"
        "from dispatchatlas.processes.launching import launch_detached\n"
        f"launch_detached([sys.executable, '-c', {reader!r}], log={str(log)!r}, cwd='.')\n"
        "sys.stdin.read()\n"
    )
    launcher = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, cwd=ROOT)
    try:
        # The launcher's input stays open; a command that inherited it would wait.
        assert _log_holds(log, "eof ''", within=60)
    finally:
        if launcher.stdin is not None:
            launcher.stdin.close()
        launcher.wait(60)


def test_a_missing_command_is_raised_not_taken_for_a_job_refusal(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    with pytest.raises(FileNotFoundError):
        launch_detached(["no-such-command-dispatchatlas-xyz"], log=log, cwd=tmp_path)
    assert "forbids breakaway" not in log.read_text(encoding="utf-8")


@windows_only
def test_breakaway_is_asked_first_and_dropped_only_when_the_job_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flags: list[int] = []
    refusals: list[OSError] = []

    class Started:
        pid = 4321

    def popen(command: list[str], **options: Any) -> Started:
        flags.append(int(options["creationflags"]))
        assert options["stdin"] is subprocess.DEVNULL
        if refusals:
            raise refusals.pop(0)
        return Started()

    monkeypatch.setattr(subprocess, "Popen", popen)
    breakaway = int(getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000))

    log = tmp_path / "a.log"
    assert launch_detached(["x"], log=log, cwd=tmp_path).breakaway is True
    assert flags == [flags[0]]
    assert flags[0] & breakaway

    denied = OSError(0, "Access is denied", None, 5)  # winerror 5
    refusals.append(denied)
    flags.clear()
    assert launch_detached(["x"], log=log, cwd=tmp_path).breakaway is False
    assert flags[0] & breakaway
    assert not flags[1] & breakaway
    assert "forbids breakaway" in log.read_text(encoding="utf-8")
