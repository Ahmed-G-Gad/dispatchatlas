"""Shared helpers of the tests of the processes package."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from dispatchatlas.campaigns.processors import allowed_cpus
from dispatchatlas.processes.affinity import pinning_refusal_text
from dispatchatlas.processes.snapshots import is_running

ROOT = Path(__file__).resolve().parents[2]

needs_pinning = pytest.mark.skipif(
    pinning_refusal_text() is not None, reason="the platform cannot pin a process"
)
windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="job objects are a Windows facility"
)
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="sessions are a POSIX facility")
needs_proc = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(), reason="the /proc snapshot is a Linux facility"
)


def slot_cpus(count: int = 2) -> list[int]:
    """Return up to ``count`` processors of this process's set, from its end."""
    allowed = allowed_cpus()
    return list(allowed[-min(count, len(allowed)) :])


def gone(pid: int, within: float = 15.0) -> bool:
    """Return whether process ``pid`` ends within ``within`` seconds."""
    deadline = time.monotonic() + within
    while is_running(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def wait_for(path: Path, within: float = 60.0) -> str:
    """Return the text of ``path`` once it exists and is not empty."""
    deadline = time.monotonic() + within
    while not (path.exists() and path.read_text(encoding="ascii").strip()):
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)
    return path.read_text(encoding="ascii")


def sleeper(seconds: float = 60.0) -> subprocess.Popen[bytes]:
    """Start a Python process that sleeps."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def console_script(directory: Path, name: str = "straytest") -> Path:
    """Write a script whose ``#!`` line names this interpreter, as a console script's does.

    Started by its own path (POSIX), the script writes its process id to the
    file of its name with ``.pid`` added, then sleeps for a minute.
    """
    if any(character.isspace() for character in sys.executable):
        pytest.skip("a #! line cannot name an interpreter whose path holds a space")
    script = directory / name
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "open(sys.argv[0] + '.pid', 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def orphan(command: Sequence[str]) -> int:
    """Start ``command`` from a process that ends at once, and return its process id.

    The command's process is then no descendant of this one, as a run
    started from another shell would not be.
    """
    started = subprocess.run(
        [
            sys.executable,
            "-c",
            "import subprocess, sys; print(subprocess.Popen(sys.argv[1:], "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
            "stderr=subprocess.DEVNULL).pid)",
            *command,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return int(started.stdout.split()[-1])
