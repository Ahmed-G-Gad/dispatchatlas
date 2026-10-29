"""Functions that the tests of the processes package call in child processes."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dispatchatlas.processes.affinity import current_mask
from dispatchatlas.processes.child import THREAD_VARIABLES


def echo(**arguments: Any) -> dict[str, Any]:
    """Return the arguments as the row."""
    return dict(arguments)


def fail(message: str, **arguments: Any) -> dict[str, Any]:
    """Raise with a message that holds a file path and a caller's word."""
    raise ValueError(message)


def die(code: int = 7, **arguments: Any) -> dict[str, Any]:
    """End the process at once, without a reply."""
    os._exit(code)


def sleep(seconds: float, marker: str | None = None, **arguments: Any) -> dict[str, Any]:
    """Sleep, then return the arguments; write the process id to ``marker`` first."""
    if marker is not None:
        Path(marker).write_text(str(os.getpid()), encoding="ascii")
    time.sleep(seconds)
    return {"seconds": seconds, **arguments}


def prints_then_row(**arguments: Any) -> dict[str, Any]:
    """Print junk to standard output before returning the row."""
    print("junk that is not the reply")  # noqa: T201 - the test needs a print
    print('{"row": {"forged": true}}')  # noqa: T201 - the test needs a print
    sys.stdout.flush()
    return dict(arguments)


def not_a_row(**arguments: Any) -> Any:
    """Return something that is not a mapping."""
    return [1, 2, 3]


def spawn_grandchild(marker: str, seconds: float = 60.0, **arguments: Any) -> dict[str, Any]:
    """Start a sleeping grandchild, write both process ids to ``marker``, and sleep."""
    grandchild = subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    Path(marker).write_text(f"{os.getpid()} {grandchild.pid}", encoding="ascii")
    time.sleep(seconds)
    return dict(arguments)


def records_env(**arguments: Any) -> dict[str, Any]:
    """Return the thread variables, PYTHONUNBUFFERED, and the affinity mask."""
    return {
        **arguments,
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "unbuffered": os.environ.get("PYTHONUNBUFFERED"),
        "mask": current_mask(),
    }
