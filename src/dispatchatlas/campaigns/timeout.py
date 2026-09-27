"""Hard time limits: stopping a run that ignores its own.

Stop criteria are cooperative: a solver checks them itself, and one that
does not, or that is stuck inside a native extension, keeps running. The
process backend therefore gives every run whose protocol sets ``time_limit``
a hard limit of ``timeout_factor * time_limit + grace`` seconds, counted from
the moment the run is handed to its worker. When the limit passes, the
supervising process kills the worker and records a terminal timeout failure:
the same instance, seed, and budget would overrun again. A fresh worker takes
the next run. Killing a process is the one interruption that reaches code
inside a native extension.

On POSIX systems each worker leads its own process group, and the kill
reaches the whole group, so processes a solver started die with it. On
Windows only the worker itself is killed.
"""

from __future__ import annotations

import contextlib
import os
import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from dispatchatlas.campaigns.config import Execution
    from dispatchatlas.core.solver import StopCriteria

__all__ = ["detach", "hard_limit", "kill"]


def hard_limit(stop: StopCriteria, execution: Execution) -> float | None:
    """Return the hard limit of a run in seconds.

    Args:
        stop: Stop criteria of the run's protocol.
        execution: The campaign's execution settings.

    Returns:
        ``timeout_factor * time_limit + grace``, or ``None`` when the
        protocol sets no time limit, hard limits are disabled, or the backend
        is sequential and cannot enforce them.
    """
    if (
        execution.backend != "process"
        or execution.timeout_factor is None
        or stop.time_limit is None
    ):
        return None
    return execution.timeout_factor * stop.time_limit + execution.grace


def detach() -> None:
    """Make the calling process the leader of a new session and process group.

    Called first thing in every worker, so that :func:`kill` reaches the
    processes a solver starts. Does nothing where sessions do not exist.
    """
    if hasattr(os, "setsid"):
        with contextlib.suppress(OSError):
            os.setsid()


def kill(process: BaseProcess, *, wait: float = 5.0) -> None:
    """Kill a worker process, with its process group on POSIX, and reap it.

    Args:
        process: The worker.
        wait: Seconds to wait for the process to be reaped.
    """
    if process.pid is not None and hasattr(os, "killpg"):
        # Before the worker has called detach(), no group has its pid, and
        # the call fails harmlessly; the direct kill below covers that case.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    if process.is_alive():
        process.kill()
    process.join(wait)
