"""Starting a command that must outlive the shell or session that starts it.

:func:`launch_detached` starts a command with no console window and no
standard input, its output appended to a log file, in a process group of its
own, and does not wait for it. On Windows it first asks to leave the job
object of the calling process (``CREATE_BREAKAWAY_FROM_JOB``), which an agent
session or a CI step may hold with kill-on-close; where the job forbids
that, it starts the command inside the job and says so, since the command
then ends with the job. On POSIX systems the command starts a new session.

From the command line::

    python -m dispatchatlas.processes.launching --log run.log --cwd . [--env K=V ...] -- COMMAND ...

prints ``started pid N, breakaway yes|no`` and returns at once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["Launched", "launch_detached", "main"]

_ACCESS_DENIED = 5
"""The Windows error of a breakaway that the caller's job does not permit."""


@dataclass(frozen=True)
class Launched:
    """A command :func:`launch_detached` started.

    Attributes:
        pid: Id of the started process.
        breakaway: Whether it left the job object of the caller (Windows);
            ``False`` on Windows means it ends with that job. Always ``True``
            on POSIX systems, where no job holds it.
        log: The log file its output is appended to.
    """

    pid: int
    breakaway: bool
    log: Path


def launch_detached(
    command: Sequence[str],
    *,
    log: str | Path,
    cwd: str | Path,
    environment: Mapping[str, str] | None = None,
) -> Launched:
    """Start ``command`` detached from the caller and return without waiting.

    The command gets the caller's environment with ``PYTHONUNBUFFERED=1``
    and ``PYTHONIOENCODING=utf-8`` added, then ``environment``; standard
    input from the null device; standard output and error appended to
    ``log``, which is created with its folder when absent.

    Raises:
        ValueError: If the command is empty.
        OSError: If the command cannot be started.
    """
    if not command:
        raise ValueError("the command to launch is empty")
    target = Path(log)
    target.parent.mkdir(parents=True, exist_ok=True)
    variables = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    variables.update(environment or {})
    with target.open("ab") as stream:
        stream.write(f"--- launched {_utc_now()}: {' '.join(command)}\n".encode())
        stream.flush()
        options: dict[str, Any] = {
            "cwd": str(cwd),
            "env": variables,
            "stdin": subprocess.DEVNULL,
            "stdout": stream,
            "stderr": subprocess.STDOUT,
            "close_fds": True,
        }
        if sys.platform == "win32":
            detached = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            try:
                process = subprocess.Popen(
                    list(command),
                    creationflags=detached | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                    **options,
                )
                breakaway = True
            except OSError as error:
                if getattr(error, "winerror", None) != _ACCESS_DENIED:
                    raise  # a missing command, say: not the job's doing
                stream.write(
                    b"--- warning: the caller's job forbids breakaway; the command ends with it\n"
                )
                stream.flush()
                process = subprocess.Popen(list(command), creationflags=detached, **options)
                breakaway = False
        else:
            process = subprocess.Popen(list(command), start_new_session=True, **options)
            breakaway = True
    return Launched(pid=process.pid, breakaway=breakaway, log=target)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry; see the module docstring."""
    parser = argparse.ArgumentParser(
        prog="python -m dispatchatlas.processes.launching",
        description="Start a command detached from this shell, its output appended to a log.",
    )
    parser.add_argument("--log", required=True, help="file the command's output is appended to")
    parser.add_argument("--cwd", default=".", help="working directory of the command")
    parser.add_argument(
        "--env", action="append", default=[], metavar="K=V", help="add an environment variable"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND ...")
    arguments = parser.parse_args(argv)
    command = list(arguments.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("give the command after --")
    environment = dict(_pair(item, parser) for item in arguments.env)
    launched = launch_detached(
        command, log=arguments.log, cwd=arguments.cwd, environment=environment
    )
    sys.stdout.write(
        f"started pid {launched.pid}, breakaway {'yes' if launched.breakaway else 'no'}\n"
    )
    return 0


def _pair(item: str, parser: argparse.ArgumentParser) -> tuple[str, str]:
    name, equals, value = item.partition("=")
    if not equals or not name:
        parser.error(f"--env takes NAME=VALUE, not {item!r}")
    return name, value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
