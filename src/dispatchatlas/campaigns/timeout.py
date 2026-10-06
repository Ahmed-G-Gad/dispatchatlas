"""Hard time limits: stopping a run that ignores its own.

Stop criteria are cooperative: a solver checks them itself, and one that
does not, or that is stuck inside a native extension, keeps running. The
process backend therefore gives every run whose protocol sets ``time_limit``
a hard limit of ``timeout_factor * time_limit + grace`` seconds, counted from
the instant its worker states as the start of the solve. When the limit
passes, the supervising process kills the worker and records a terminal
timeout failure: the same instance, seed, and budget would overrun again. The
failure holds the last incumbent the run's search had reported, and, apart
from it, the last one the search had found within the time limit. A fresh
worker takes the next run. Killing a process is the one interruption that
reaches code inside a native extension.

Until its solve begins, a run is under the start-up limit instead:
``startup_limit`` seconds from its hand-over, in which the worker starts its
interpreter and loads plugins, if it started for this run, and rebuilds the
solver and the problem. The supervising process waits for none of that: a
fresh worker gets its run when it reports that it is ready. A worker that
does not begin in time is killed too, and the attempt is a recoverable
start-up failure, not a timeout.

A kill reaches the processes a solver started. On POSIX systems each worker
leads its own process group (:func:`detach`), and the kill reaches the whole
group. On Windows the supervising process puts each worker in a job object
of its own (:func:`contain`), which every process the worker starts joins,
and the kill ends the job. Where a job object cannot be created or the
worker cannot join it, a warning says so and the kill reaches the worker
alone.

A worker does not outlive its supervising process either
(:func:`end_with_parent`): it watches that process and ends itself, with the
process group it leads, when the process is gone. On Windows the job object
also ends with the supervising process, which holds its only handle.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import sys
import threading
from typing import TYPE_CHECKING, Any

from dispatchatlas.core.logs import get_logger

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from dispatchatlas.campaigns.config import Execution
    from dispatchatlas.core.solver import StopCriteria

__all__ = ["contain", "detach", "end_with_parent", "hard_limit", "kill", "release"]

_LOG = get_logger(__name__)

_KILL_ON_JOB_CLOSE = 0x2000
"""``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``: closing the last handle ends every process."""

_EXTENDED_LIMIT_INFORMATION = 9
"""``JobObjectExtendedLimitInformation``, the class of the limits a job is given."""

_SET_QUOTA_AND_TERMINATE = 0x0100 | 0x0001
"""The access rights a process handle needs for the process to join a job."""

_TERMINATED = 0x10000
"""Exit code of the processes of a terminated job, the one ``multiprocessing`` kills with."""


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


def end_with_parent() -> None:
    """End the calling worker process when its supervising process is gone.

    A worker whose supervisor was killed would otherwise compute until its
    run returns and it meets the closed pipe, beside the workers of a resumed
    session. A daemon thread waits for the process that started this one
    through ``multiprocessing`` and then ends this process at once. On POSIX
    systems it ends the process group with it, so that processes a solver
    started end too, but only a group this process leads: where
    :func:`detach` could not start one, the group is the supervising
    process's, and holds whatever launched the campaign. The thread needs
    the interpreter to act: a worker inside a native extension that keeps
    the interpreter lock ends when the extension returns. Does nothing in a
    process that ``multiprocessing`` did not start.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        _end_self()

    threading.Thread(target=watch, name="dispatchatlas-supervisor-watch", daemon=True).start()


def _end_self() -> None:
    """End the calling process at once, with its process group if it leads one."""
    if sys.platform != "win32" and os.getpgrp() == os.getpid():
        with contextlib.suppress(OSError):
            os.killpg(os.getpgrp(), signal.SIGKILL)
    os._exit(1)


def contain(process: BaseProcess) -> int | None:
    """Put a started worker process in a Windows job object of its own.

    The job ends every process in it when it is terminated or when its last
    handle closes, and every process the worker starts joins it. The calling
    process holds the only handle, so the worker and what it started also
    end when the calling process dies.

    Args:
        process: A started worker.

    Returns:
        The handle of the job, for :func:`kill` and :func:`release`. ``None``
        on other systems, and where the job cannot be created or the worker
        cannot join it, for example on a Windows version without nested jobs
        when the calling process is in a job itself. A warning reports that
        case, and :func:`kill` then reaches the worker alone.
    """
    if sys.platform == "win32" and process.pid is not None:
        job, error = _new_job(process.pid)
        if job is None:
            _LOG.warning(
                "worker process %d is in no job object (%s); a kill reaches the worker but "
                "not the processes a solver starts",
                process.pid,
                error,
            )
        return job
    return None


def kill(process: BaseProcess, job: int | None = None, *, wait: float = 5.0) -> None:
    """Kill a worker process and the processes it started, and reap it.

    The processes it started are reached through the worker's process group
    on POSIX systems and through its job object on Windows.

    Args:
        process: The worker.
        job: The worker's job object, as :func:`contain` returned it.
        wait: Seconds to wait for the process to be reaped.
    """
    if job is not None:
        _end_job(job, terminate=True)
    if process.pid is not None and hasattr(os, "killpg"):
        # Before the worker has called detach(), no group has its pid, and
        # the call fails harmlessly; the direct kill below covers that case.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    if process.is_alive():
        process.kill()
    process.join(wait)


def release(job: int | None) -> None:
    """Close the handle of a worker's job object, if it has one.

    Processes still in the job end with it, so nothing a solver started
    outlives the worker's slot on Windows.
    """
    if job is not None:
        _end_job(job, terminate=False)


def _kernel32() -> Any:
    """Return ``kernel32`` with the signatures of the job functions declared, on Windows."""
    library: Any = None
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        library = ctypes.WinDLL("kernel32", use_last_error=True)
        library.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        library.CreateJobObjectW.restype = wintypes.HANDLE
        library.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        library.SetInformationJobObject.restype = wintypes.BOOL
        library.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        library.OpenProcess.restype = wintypes.HANDLE
        library.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        library.AssignProcessToJobObject.restype = wintypes.BOOL
        library.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        library.TerminateJobObject.restype = wintypes.BOOL
        library.CloseHandle.argtypes = [wintypes.HANDLE]
        library.CloseHandle.restype = wintypes.BOOL
    return library


def _new_job(pid: int) -> tuple[int | None, str | None]:
    """Create a kill-on-close job holding process ``pid``; return its handle, or the error."""
    job: int | None = None
    error: str | None = "job objects are a Windows facility"
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        class BasicLimits(ctypes.Structure):
            _fields_ = (
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            )

        class ExtendedLimits(ctypes.Structure):
            _fields_ = (
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            )

        def failed(step: str) -> str:
            return f"{step}: {ctypes.FormatError(ctypes.get_last_error()).strip()}"

        kernel32 = _kernel32()
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        created = kernel32.CreateJobObjectW(None, None)
        error = None
        if not created:
            error = failed("the job object could not be created")
        elif not kernel32.SetInformationJobObject(
            created, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = failed("the job object could not be configured")
        else:
            handle = kernel32.OpenProcess(_SET_QUOTA_AND_TERMINATE, False, pid)
            if not handle:
                error = failed("the worker process could not be opened")
            else:
                if not kernel32.AssignProcessToJobObject(created, handle):
                    error = failed("the worker process could not join the job object")
                kernel32.CloseHandle(handle)
        if error is None:
            job = int(created)
        elif created:
            kernel32.CloseHandle(created)
    return job, error


def _end_job(job: int, *, terminate: bool) -> None:
    """Terminate the processes of a job object, or close its handle."""
    if sys.platform == "win32":
        kernel32 = _kernel32()
        if terminate:
            kernel32.TerminateJobObject(job, _TERMINATED)
        else:
            kernel32.CloseHandle(job)
