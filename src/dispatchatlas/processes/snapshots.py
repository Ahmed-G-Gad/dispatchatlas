"""A snapshot of the machine's processes, without extra dependencies.

:func:`snapshot` lists every process with its parent, executable, affinity
mask, and start time: on Windows through the Toolhelp snapshot and one
limited query per process, on Linux from ``/proc``. Elsewhere it returns an
empty tuple. A field the caller may not read (a process of another user, a
protected process) is ``None``.

:func:`descendants` follows parent ids from a process, :func:`interpreters_on`
finds Python interpreters that may run on given processors, and
:func:`end_process` ends one process on request. Nothing here runs a shell
command or a management query.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dispatchatlas.processes.affinity import mask_of_cpus

__all__ = [
    "ProcessInfo",
    "descendants",
    "end_process",
    "interpreters_on",
    "is_running",
    "snapshot",
]

_QUERY_LIMITED_INFORMATION = 0x1000
_TERMINATE = 0x0001
_SNAPSHOT_PROCESSES = 0x2
_INVALID_HANDLE = -1
_ACCESS_DENIED = 5
_STILL_ACTIVE = 259
_FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=UTC)
_PYTHON = re.compile(r"^python[\w.]*?(\.exe)?$", re.IGNORECASE)
_SYSCTL_PROC_PID = (1, 14, 1)  # CTL_KERN, KERN_PROC, KERN_PROC_PID: kern.proc.pid on macOS
_ZOMBIE = 5  # SZOMB: on macOS, a process that has ended and awaits its parent's wait
_KINFO_PROC_BUFFER = 1024  # room for one struct kinfo_proc, 648 bytes on 64-bit macOS


@dataclass(frozen=True)
class ProcessInfo:
    """One process of a :func:`snapshot`.

    Attributes:
        pid: Process id.
        parent_pid: Id of the process that created it, ``None`` when unknown.
            The parent may have ended and its id been reused since.
        name: Executable file name (Windows) or command name (Linux). On
            Linux, a script run through its ``#!`` line, such as a console
            script, gives the script's name, not the interpreter's.
        executable: Full path of the executable, ``None`` when unreadable.
        affinity_mask: Processors the process may run on, ``None`` when
            unreadable.
        started_utc: Creation time, ISO 8601 in UTC, ``None`` when unreadable.
    """

    pid: int
    parent_pid: int | None
    name: str
    executable: str | None
    affinity_mask: int | None
    started_utc: str | None


def snapshot() -> tuple[ProcessInfo, ...]:
    """Return every process of the machine, ordered by id; empty where unsupported."""
    if sys.platform == "win32":
        found = _windows_snapshot()
    elif Path("/proc/self/stat").exists():
        found = _proc_snapshot()
    else:
        found = []
    return tuple(sorted(found, key=lambda info: info.pid))


def descendants(
    pid: int, processes: Sequence[ProcessInfo] | None = None
) -> tuple[ProcessInfo, ...]:
    """Return the processes created by ``pid`` and by them, at any depth.

    A process that names a parent but started before it holds a reused
    parent id, and is left out with what it created.

    Args:
        pid: The root process; it is not part of the result.
        processes: A snapshot to search; a fresh one by default.
    """
    listed = snapshot() if processes is None else tuple(processes)
    by_pid = {info.pid: info for info in listed}
    children: dict[int, list[ProcessInfo]] = {}
    for info in listed:
        if info.parent_pid is not None and info.parent_pid != info.pid:
            children.setdefault(info.parent_pid, []).append(info)
    found: list[ProcessInfo] = []
    pending = [pid]
    seen = {pid}
    while pending:
        parent = pending.pop()
        parent_start = by_pid[parent].started_utc if parent in by_pid else None
        for child in children.get(parent, ()):
            if child.pid in seen:
                continue
            if parent_start and child.started_utc and child.started_utc < parent_start:
                continue  # the parent id was reused by a later process
            seen.add(child.pid)
            found.append(child)
            pending.append(child.pid)
    return tuple(sorted(found, key=lambda info: info.pid))


def interpreters_on(
    cpus: Iterable[int],
    processes: Sequence[ProcessInfo] | None = None,
    *,
    exclude: Iterable[int] = (),
) -> tuple[ProcessInfo, ...]:
    """Return the Python interpreters whose affinity mask meets ``cpus``.

    An interpreter is a process named ``python``, ``python3``,
    ``python3.13``, ``pythonw``, with or without ``.exe``; on Linux, one
    running a console script such as ``pytest`` through its ``#!`` line is
    named after the script and is not found. A process whose mask cannot be
    read is left out.

    Args:
        cpus: Processor indices.
        processes: A snapshot to search; a fresh one by default.
        exclude: Process ids to leave out, such as a runner and its children.
    """
    wanted = mask_of_cpus(cpus)
    left_out = set(exclude)
    listed = snapshot() if processes is None else tuple(processes)
    return tuple(
        info
        for info in listed
        if info.pid not in left_out
        and _PYTHON.match(info.name)
        and info.affinity_mask is not None
        and info.affinity_mask & wanted
    )


def is_running(pid: int) -> bool:
    """Return whether process ``pid`` exists and has not ended.

    A process that has ended but whose parent has not reaped it yet (a POSIX
    zombie, or a Windows process whose handle is still open) is not
    running. A zombie is told from ``/proc`` on Linux and from ``sysctl`` on
    macOS; elsewhere it counts as running. A process the caller may not
    query counts as running.
    """
    running = True
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        kernel32 = _kernel32()
        handle = kernel32.OpenProcess(_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            running = ctypes.get_last_error() == _ACCESS_DENIED
        else:
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                running = code.value == _STILL_ACTIVE
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            running = False
        except PermissionError:
            running = True
        else:
            running = not _is_zombie(pid)
    return running


def end_process(pid: int) -> str | None:
    """End process ``pid`` at once; return the error text, or ``None``.

    Windows: ``TerminateProcess`` with exit code 1. POSIX: ``SIGKILL``.
    """
    error: str | None = None
    if sys.platform == "win32":
        kernel32 = _kernel32()
        handle = kernel32.OpenProcess(_TERMINATE, False, pid)
        if not handle:
            error = f"process {pid} could not be opened: {_last_error()}"
        else:
            if not kernel32.TerminateProcess(handle, 1):
                error = f"process {pid} could not be ended: {_last_error()}"
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as failure:
            error = f"process {pid} could not be ended: {failure.strerror or failure}"
    return error


def _is_zombie(pid: int) -> bool:
    """Return whether process ``pid`` has ended and awaits its parent's wait (POSIX).

    ``kill(pid, 0)`` succeeds on such a process, as POSIX asks; its state
    tells it apart: ``Z`` in ``/proc/<pid>/stat`` on Linux, ``SZOMB`` from
    ``sysctl`` on macOS. ``False`` where neither can be read.
    """
    zombie = False
    if sys.platform == "darwin":
        zombie = _darwin_zombie(pid)
    else:
        with contextlib.suppress(OSError):
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            zombie = stat.rpartition(")")[2].split()[:1] == ["Z"]
    return zombie


# ------------------------------------------------------------------- Linux


def _proc_snapshot() -> list[ProcessInfo]:
    """Read every process from ``/proc``."""
    boot = _boot_time()
    ticks = 100
    if sys.platform != "win32":
        with contextlib.suppress(AttributeError, ValueError, OSError):
            ticks = os.sysconf("SC_CLK_TCK")
    found: list[ProcessInfo] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # The command name is in parentheses and may hold spaces and parentheses.
        head, _, tail = stat.rpartition(")")
        name = head.partition("(")[2]
        fields = tail.split()
        parent = int(fields[1]) if len(fields) > 1 and fields[1].isdigit() else None
        started: str | None = None
        if boot is not None and len(fields) > 19 and fields[19].isdigit():
            moment = datetime.fromtimestamp(boot, UTC) + timedelta(seconds=int(fields[19]) / ticks)
            started = moment.isoformat(timespec="microseconds")
        executable: str | None = None
        with contextlib.suppress(OSError):
            executable = str((entry / "exe").readlink())
        found.append(
            ProcessInfo(
                pid=int(entry.name),
                parent_pid=parent,
                name=name,
                executable=executable,
                affinity_mask=_proc_mask(entry / "status"),
                started_utc=started,
            )
        )
    return found


def _boot_time() -> float | None:
    with contextlib.suppress(OSError):
        for line in Path("/proc/stat").read_text(encoding="ascii").splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    return None


def _proc_mask(status: Path) -> int | None:
    with contextlib.suppress(OSError, ValueError):
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Cpus_allowed:"):
                return int(line.split(":", 1)[1].strip().replace(",", ""), 16)
    return None


# ------------------------------------------------------------------- macOS


def _darwin_zombie(pid: int) -> bool:  # pragma: no cover - macOS only
    """Return whether process ``pid`` is a zombie, from its ``struct kinfo_proc``.

    ``sysctl`` with ``kern.proc.pid.<pid>`` returns the structure of a
    zombie too. Its first member, ``struct extern_proc`` of
    ``<sys/proc.h>``, holds the state ``p_stat`` and the process id
    ``p_pid``; a record that names another process is not trusted.
    """
    zombie = False
    if sys.platform == "darwin":
        import ctypes  # noqa: PLC0415 - macOS only

        class Head(ctypes.Structure):
            _fields_ = (
                ("p_un", ctypes.c_byte * 16),  # two pointers, or a struct timeval
                ("p_vmspace", ctypes.c_void_p),
                ("p_sigacts", ctypes.c_void_p),
                ("p_flag", ctypes.c_int),
                ("p_stat", ctypes.c_ubyte),  # a char
                ("p_pid", ctypes.c_int),
            )

        with contextlib.suppress(AttributeError, OSError):
            library = ctypes.CDLL(None)
            library.sysctl.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            library.sysctl.restype = ctypes.c_int
            name = (ctypes.c_int * 4)(*_SYSCTL_PROC_PID, pid)
            record = ctypes.create_string_buffer(_KINFO_PROC_BUFFER)
            size = ctypes.c_size_t(ctypes.sizeof(record))
            read = library.sysctl(name, len(name), record, ctypes.byref(size), None, 0) == 0
            if read and size.value >= ctypes.sizeof(Head):
                head = Head.from_buffer_copy(record.raw)
                zombie = head.p_pid == pid and head.p_stat == _ZOMBIE
    return zombie


# ----------------------------------------------------------------- Windows


def _kernel32() -> Any:
    """Return ``kernel32`` with the signatures used here declared, on Windows."""
    library: Any = None
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        library = ctypes.WinDLL("kernel32", use_last_error=True)
        library.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        library.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        library.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        library.OpenProcess.restype = wintypes.HANDLE
        library.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        library.TerminateProcess.restype = wintypes.BOOL
        library.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        library.GetExitCodeProcess.restype = wintypes.BOOL
        library.CloseHandle.argtypes = [wintypes.HANDLE]
        library.CloseHandle.restype = wintypes.BOOL
        library.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        library.QueryFullProcessImageNameW.restype = wintypes.BOOL
        library.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        library.GetProcessAffinityMask.restype = wintypes.BOOL
        library.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            *[ctypes.POINTER(wintypes.FILETIME)] * 4,
        ]
        library.GetProcessTimes.restype = wintypes.BOOL
    return library


def _last_error() -> str:
    text = "not a Windows system"
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        text = ctypes.FormatError(ctypes.get_last_error()).strip()
    return text


def _windows_snapshot() -> list[ProcessInfo]:  # pragma: no cover - Windows only
    """Read every process from a Toolhelp snapshot."""
    kernel32 = _kernel32()
    found: list[ProcessInfo] = []
    for pid, parent, name in _toolhelp_rows():
        executable, mask, started = _windows_details(kernel32, pid)
        found.append(
            ProcessInfo(
                pid=pid,
                parent_pid=parent if pid else None,
                name=name,
                executable=executable,
                affinity_mask=mask,
                started_utc=started,
            )
        )
    return found


def _toolhelp_rows() -> list[tuple[int, int, str]]:  # pragma: no cover - Windows only
    """Return the id, parent id, and executable name of every process, from Toolhelp."""
    rows: list[tuple[int, int, str]] = []
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        class Entry(ctypes.Structure):
            _fields_ = (
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * 260),
            )

        kernel32 = _kernel32()
        kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
        kernel32.Process32NextW.restype = wintypes.BOOL
        listing = kernel32.CreateToolhelp32Snapshot(_SNAPSHOT_PROCESSES, 0)
        if listing and listing != ctypes.c_void_p(_INVALID_HANDLE).value:
            try:
                entry = Entry()
                entry.dwSize = ctypes.sizeof(Entry)
                more = kernel32.Process32FirstW(listing, ctypes.byref(entry))
                while more:
                    rows.append(
                        (int(entry.th32ProcessID), int(entry.th32ParentProcessID), entry.szExeFile)
                    )
                    more = kernel32.Process32NextW(listing, ctypes.byref(entry))
            finally:
                kernel32.CloseHandle(listing)
    return rows


def _windows_details(
    kernel32: Any, pid: int
) -> tuple[str | None, int | None, str | None]:  # pragma: no cover - Windows only
    """Return the executable, affinity mask, and start time of a process, where readable."""
    executable: str | None = None
    mask: int | None = None
    started: str | None = None
    if sys.platform == "win32" and pid != 0:
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        handle = kernel32.OpenProcess(_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            try:
                buffer = ctypes.create_unicode_buffer(32768)
                size = wintypes.DWORD(len(buffer))
                if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                    executable = buffer.value
                process, system = ctypes.c_size_t(), ctypes.c_size_t()
                if kernel32.GetProcessAffinityMask(
                    handle, ctypes.byref(process), ctypes.byref(system)
                ):
                    mask = int(process.value)
                times = [wintypes.FILETIME() for _ in range(4)]
                if kernel32.GetProcessTimes(handle, *(ctypes.byref(item) for item in times)):
                    ticks = times[0].dwHighDateTime << 32 | times[0].dwLowDateTime
                    if ticks:
                        moment = _FILETIME_EPOCH + timedelta(microseconds=ticks // 10)
                        started = moment.isoformat(timespec="microseconds")
            finally:
                kernel32.CloseHandle(handle)
    return executable, mask, started
