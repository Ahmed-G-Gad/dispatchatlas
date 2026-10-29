from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from dispatchatlas.processes.launching import launch_detached
from dispatchatlas.processes.snapshots import end_process, is_running
from tests.processes.helpers import ROOT, gone, wait_for


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
