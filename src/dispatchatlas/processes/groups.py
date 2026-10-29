"""Kill-on-close process groups: what a kill reaches, and what ends with the parent.

A :class:`ProcessGroup` holds the processes of one worker slot so that one
call ends all of them, and so that they end when the process that created
the group dies.

- On Windows the group is a *named* job object with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Each member starts suspended
  (:meth:`ProcessGroup.popen_options`), and the parent assigns it to the
  job before resuming it (:meth:`ProcessGroup.assign`), so a launcher
  executable cannot create the process that computes outside the job. Job
  membership is inherited, so everything a member starts is in the job. The
  child also joins the job by name (:func:`join_group`), which proves the
  membership from inside and covers a member the parent could not assign.
  The parent holds the only lasting handle: when it dies for any reason,
  the kernel ends every process in the job.
- On POSIX systems each child is started in a session of its own
  (:meth:`ProcessGroup.popen_options`), so a kill of its process group
  reaches what it started, and the child ends its group when its parent is
  gone (:func:`end_with_parent_process`).

This module uses the standard library only; errors are returned as text, or
raised as :class:`RuntimeError` on the child side.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from types import TracebackType
from typing import Any, Literal, Self

__all__ = [
    "ProcessGroup",
    "end_with_parent_process",
    "group_name",
    "join_group",
]

_KILL_ON_JOB_CLOSE = 0x2000
"""``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``: closing the last handle ends every process."""

_EXTENDED_LIMIT_INFORMATION = 9
"""``JobObjectExtendedLimitInformation``, the class of the limits a job is given."""

_SET_QUOTA_AND_TERMINATE = 0x0100 | 0x0001
"""The access rights a process handle needs for the process to join a job."""

_ASSIGN_AND_QUERY = 0x0001 | 0x0004
"""``JOB_OBJECT_ASSIGN_PROCESS | JOB_OBJECT_QUERY``: what a child needs to join a job by name."""

_SUSPEND_RESUME = 0x0800
"""``PROCESS_SUSPEND_RESUME``: what a handle needs to resume a process started suspended."""

_CREATE_SUSPENDED = 0x00000004
"""``CREATE_SUSPENDED``: the process starts with its first thread suspended."""

_ALREADY_EXISTS = 183
"""``ERROR_ALREADY_EXISTS``: ``CreateJobObjectW`` opened a job of that name instead."""

TERMINATED = 0x10000
"""Exit code of the processes of a terminated job, the one ``multiprocessing`` kills with."""

_PARENT_POLL_SECONDS = 1.0


def group_name(owner_pid: int, slot: int) -> str:
    """Return the name of the group of slot ``slot`` of process ``owner_pid``.

    On Windows the name is in the session's local namespace, so only
    processes of the same session can open it.
    """
    stem = f"dispatchatlas-{owner_pid}-{slot}"
    return f"Local\\{stem}" if sys.platform == "win32" else stem


class ProcessGroup:
    """One kill-on-close group of processes.

    Create it with :meth:`create`, start each member with
    :meth:`popen_options`, then :meth:`assign` it. :meth:`terminate` ends
    every member; :meth:`close` releases the group, and on Windows ends every
    process still in it.

    Attributes:
        name: The group's name, which a child passes to :func:`join_group`.
        platform: ``"job"`` (a Windows job object), ``"session"`` (a POSIX
            session per member), or ``"none"`` (the job could not be
            created; see ``error``).
        handle: The job's handle on Windows, else ``None``.
        error: Why the group could not be created, else ``None``.
    """

    def __init__(
        self,
        name: str,
        platform: Literal["job", "session", "none"],
        handle: int | None = None,
        error: str | None = None,
    ) -> None:
        self.name = name
        self.platform = platform
        self.handle = handle
        self.error = error
        self._members: list[int] = []
        self._lock = threading.Lock()

    @classmethod
    def create(cls, name: str) -> ProcessGroup:
        """Create a group; on Windows a named kill-on-close job object.

        A Windows job of the same name that already exists is refused, so a
        group never holds another owner's processes.
        """
        group = cls(name, "session")
        if sys.platform == "win32":
            handle, error = _create_job(name)
            group = cls(name, "none", error=error) if handle is None else cls(name, "job", handle)
        return group

    def popen_options(self) -> dict[str, Any]:
        """Return the keyword arguments of :class:`subprocess.Popen` for a member.

        Windows: a new console process group, so a Ctrl-C in the parent's
        console does not reach the member, and a *suspended* start: the
        process runs no instruction until :meth:`assign` has put it in the
        job and resumed it, so everything it starts inherits the job.
        POSIX: a new session.
        """
        options: dict[str, Any] = {"start_new_session": True}
        if sys.platform == "win32":
            options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | _CREATE_SUSPENDED}
        return options

    def assign(self, pid: int) -> str | None:
        """Put a started process in the group; return the error, if any.

        On Windows the process is resumed afterwards, whether or not it
        could join, since :meth:`popen_options` starts it suspended. On POSIX
        systems the process must have been started with :meth:`popen_options`;
        its id is recorded as a process group id.
        """
        error: str | None = None
        if self.platform == "job" and self.handle is not None:
            error = _assign(self.handle, pid, resume=True)
        elif self.platform == "none":
            error = self.error or "the group could not be created"
            _assign(None, pid, resume=True)
        if error is None:
            with self._lock:
                self._members.append(pid)
        return error

    def assign_process(self, process: subprocess.Popen[Any]) -> str | None:
        """Put a process started with :meth:`popen_options` in the group; return the error.

        Like :meth:`assign`, but on Windows it uses the handle that
        :class:`subprocess.Popen` already holds (with full access) instead of
        opening the process again by its id, so a process that cannot be
        opened is still assigned and, above all, resumed.
        """
        handle = getattr(process, "_handle", None) if sys.platform == "win32" else None
        if handle is None:
            return self.assign(process.pid)
        error: str | None
        if self.platform == "job" and self.handle is not None:
            error = _assign_handle(self.handle, int(handle), process.pid, resume=True)
        else:
            error = self.error or "the group could not be created"
            _assign_handle(None, int(handle), process.pid, resume=True)
        if error is None:
            with self._lock:
                self._members.append(process.pid)
        return error

    def forget(self, pid: int) -> None:
        """Stop tracking a member that has ended (POSIX: its process group)."""
        with self._lock, contextlib.suppress(ValueError):
            self._members.remove(pid)

    def terminate(self, code: int = TERMINATED) -> None:
        """End every process of the group; the group stays usable.

        Windows: ``TerminateJobObject``. POSIX: ``SIGKILL`` to the process
        group of every member still tracked.
        """
        if sys.platform == "win32":
            if self.handle is not None:
                _kernel32().TerminateJobObject(self.handle, code)
        else:
            with self._lock:
                members = list(self._members)
            for pid in members:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pid, signal.SIGKILL)

    def close(self) -> None:
        """Release the group. On Windows every process still in it ends."""
        if sys.platform == "win32" and self.handle is not None:
            _kernel32().CloseHandle(self.handle)
        self.handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"ProcessGroup(name={self.name!r}, platform={self.platform!r})"


def join_group(name: str) -> str:
    """Join the group ``name`` from inside a member process.

    Returns:
        ``"job"`` when the process joined the Windows job, ``"job
        (already)"`` when it was in it (inherited from a launcher the parent
        had assigned), and ``"session"`` on POSIX systems, where a member
        leads its own session from its start and nothing is left to do.

    Raises:
        RuntimeError: If the Windows job cannot be opened or joined.
    """
    joined = "session"
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        kernel32 = _kernel32()
        job = kernel32.OpenJobObjectW(_ASSIGN_AND_QUERY, False, name)
        if not job:
            raise RuntimeError(f"the job object could not be opened: {_last_error()}")
        try:
            inside = wintypes.BOOL(False)
            current = kernel32.GetCurrentProcess()
            if not kernel32.IsProcessInJob(current, job, ctypes.byref(inside)):
                raise RuntimeError(f"the job membership could not be read: {_last_error()}")
            joined = "job (already)" if inside.value else "job"
            if not inside.value and not kernel32.AssignProcessToJobObject(job, current):
                raise RuntimeError(f"the process could not join the job object: {_last_error()}")
        finally:
            kernel32.CloseHandle(job)
    return joined


def end_with_parent_process(runner: int | None = None) -> None:
    """End the calling process and its process group when its parent is gone (POSIX).

    Args:
        runner: The id of the process that started this one, as that process
            recorded it. When the parent seen now is another process and
            ``runner`` no longer exists, the runner died before this call
            (during the interpreter's start), and the process ends at once.
            A parent seen now that differs while ``runner`` lives is a
            launcher in between, which is watched instead.

    A daemon thread then polls the parent's id once a second; when it
    changes, the parent has died, and the thread kills the process group this
    process leads, or this process alone. On Windows it does nothing: the job
    object of the group ends with the parent's handle.
    """
    if sys.platform != "win32":
        parent = os.getppid()
        if runner is not None and parent != runner and not _exists(runner):
            _end_own_group()

        def watch() -> None:
            while os.getppid() == parent:
                time.sleep(_PARENT_POLL_SECONDS)
            _end_own_group()

        threading.Thread(target=watch, name="dispatchatlas-parent-watch", daemon=True).start()


def _exists(pid: int) -> bool:
    """Return whether a process ``pid`` exists (POSIX); one owned by another user counts."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _end_own_group() -> None:
    """Kill the process group this process leads, or end this process alone (POSIX)."""
    if sys.platform != "win32":
        if os.getpgrp() == os.getpid():
            with contextlib.suppress(OSError):
                os.killpg(os.getpgrp(), signal.SIGKILL)
    os._exit(1)


# ----------------------------------------------------------------- Windows


def _kernel32() -> Any:
    """Return ``kernel32`` with the signatures of the job functions declared, on Windows."""
    library: Any = None
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        library = ctypes.WinDLL("kernel32", use_last_error=True)
        library.GetCurrentProcess.restype = wintypes.HANDLE
        library.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        library.CreateJobObjectW.restype = wintypes.HANDLE
        library.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        library.OpenJobObjectW.restype = wintypes.HANDLE
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
        library.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        library.IsProcessInJob.restype = wintypes.BOOL
        library.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        library.TerminateJobObject.restype = wintypes.BOOL
        library.CloseHandle.argtypes = [wintypes.HANDLE]
        library.CloseHandle.restype = wintypes.BOOL
    return library


def _last_error() -> str:
    """Return the text of the calling thread's last Windows error."""
    text = "not a Windows system"
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        text = ctypes.FormatError(ctypes.get_last_error()).strip()
    return text


def _create_job(name: str | None) -> tuple[int | None, str | None]:
    """Create a kill-on-close job object; return its handle, or the error."""
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

        kernel32 = _kernel32()
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        created = kernel32.CreateJobObjectW(None, name)
        error = None
        if not created:
            error = f"the job object could not be created: {_last_error()}"
        elif name is not None and ctypes.get_last_error() == _ALREADY_EXISTS:
            error = f"a job object named {name} already exists"
        elif not kernel32.SetInformationJobObject(
            created, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = f"the job object could not be configured: {_last_error()}"
        if error is None:
            job = int(created)
        elif created:
            kernel32.CloseHandle(created)
    return job, error


def _assign(job: int | None, pid: int, *, resume: bool = False) -> str | None:
    """Put process ``pid`` in a job object, then resume it if asked; return the error, if any.

    Resuming a process none of whose threads is suspended changes nothing.
    """
    error: str | None = "job objects are a Windows facility"
    if sys.platform == "win32":
        kernel32 = _kernel32()
        access = _SET_QUOTA_AND_TERMINATE | (_SUSPEND_RESUME if resume else 0)
        handle = kernel32.OpenProcess(access, False, pid)
        if not handle:
            error = f"process {pid} could not be opened: {_last_error()}"
        else:
            error = _assign_handle(job, handle, pid, resume=resume)
            kernel32.CloseHandle(handle)
    return error


def _assign_handle(job: int | None, handle: int, pid: int, *, resume: bool) -> str | None:
    """Put the process of an open ``handle`` in a job object and resume it if asked (Windows)."""
    error: str | None = "job objects are a Windows facility"
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        kernel32 = _kernel32()
        error = None
        if job is not None and not kernel32.AssignProcessToJobObject(job, handle):
            error = f"process {pid} could not join the job object: {_last_error()}"
        if resume:
            ntdll = ctypes.WinDLL("ntdll")
            ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
            ntdll.NtResumeProcess.restype = ctypes.c_long
            status = ntdll.NtResumeProcess(handle)
            if status != 0 and error is None:
                error = f"process {pid} could not be resumed: status {status & 0xFFFFFFFF:#x}"
    return error


def _new_job(pid: int) -> tuple[int | None, str | None]:
    """Create an unnamed kill-on-close job holding process ``pid``; return it, or the error."""
    job, error = _create_job(None)
    if job is None:
        return None, error
    error = _assign(job, pid)
    if error is not None:
        _kernel32().CloseHandle(job)
        return None, error
    return job, None


def _end_job(job: int, *, terminate: bool) -> None:
    """Terminate the processes of a job object, or close its handle."""
    if sys.platform == "win32":
        kernel32 = _kernel32()
        if terminate:
            kernel32.TerminateJobObject(job, TERMINATED)
        else:
            kernel32.CloseHandle(job)
