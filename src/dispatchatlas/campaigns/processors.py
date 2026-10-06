"""The processors a process may run on, and placing a process on one of them.

An operating system can confine a process to some of the machine's logical
processors: a scheduler affinity set on Linux (``taskset``, the cpuset of a
container) or a process affinity mask on Windows. A new process starts with
the set of the thread that creates it. :func:`allowed_cpus` reads the set of
the calling process, so a worker count or a warning follows what the process
may use and not what the machine has. :func:`confined` narrows the caller to
one processor while it creates a process, which therefore runs there from its
first instruction, and :func:`pin_process` confines a process that is already
running.

- Where ``os.sched_getaffinity`` exists (Linux and some other POSIX systems),
  the set is read and written with the scheduler calls. They act on one
  thread: a thread started later inherits the set, a thread that is already
  running keeps its own.
- On Windows the set is read with ``GetProcessAffinityMask`` and written with
  ``SetProcessAffinityMask``, which act on every thread of a process. One
  mask describes at most 64 processors, so on a machine with several
  processor groups the set cannot be read and pinning is refused.
- Elsewhere (macOS) the system offers no such call: the set cannot be read
  and pinning is refused.

A set that cannot be read counts as every processor of the machine;
:func:`affinity_cpus` tells a reading from that count. A limit on processor
time, such as the CPU quota of a container, is not an affinity and is not
seen.

Processors are named by the index the operating system gives them, from 0.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterable, Iterator

from dispatchatlas.core.errors import ConfigError

__all__ = [
    "affinity_cpus",
    "allowed_cpus",
    "available_cpus",
    "confined",
    "describe_cpus",
    "pin_process",
    "pinning_refusal",
]

_PROCESS_SET_INFORMATION = 0x0200
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def affinity_cpus() -> tuple[int, ...] | None:
    """Return the processors the operating system confines this process to, sorted.

    Returns:
        The processor indices, or ``None`` when the set cannot be read: the
        platform has no affinity call, the call fails, or one Windows mask
        cannot describe the process.
    """
    indices: tuple[int, ...] | None = None
    if sys.platform == "win32":
        mask = _windows_mask()
        if mask:
            indices = tuple(index for index in range(mask.bit_length()) if mask >> index & 1)
    else:
        read = getattr(os, "sched_getaffinity", None)
        if read is not None:
            with contextlib.suppress(OSError):
                indices = tuple(sorted(read(0))) or None
    return indices


def allowed_cpus() -> tuple[int, ...]:
    """Return the indices of the processors this process may run on, sorted.

    When the set cannot be read (see :func:`affinity_cpus`), every processor
    of the machine is returned. The result is never empty.
    """
    read = affinity_cpus()
    if read is not None:
        return read
    return tuple(range(max(1, os.cpu_count() or 1)))


def available_cpus() -> int:
    """Return how many processors this process may run on, at least one.

    This is the size of :func:`allowed_cpus`, so affinity masks, the cpuset
    of a container, and ``taskset`` limits are respected.
    """
    return len(allowed_cpus())


def describe_cpus(indices: Iterable[int]) -> str:
    """Return processor indices as text, with runs collapsed: ``"0-3, 8"``."""
    ordered = sorted(set(indices))
    parts: list[str] = []
    start = 0
    for end in range(1, len(ordered) + 1):
        if end < len(ordered) and ordered[end] == ordered[end - 1] + 1:
            continue
        first, last = ordered[start], ordered[end - 1]
        parts.append(str(first) if first == last else f"{first}-{last}")
        start = end
    return ", ".join(parts) if parts else "none"


def pinning_refusal() -> str | None:
    """Return why this platform cannot pin a process, or ``None`` when it can."""
    if sys.platform == "win32":
        if _windows_mask() is None:
            return (
                "one affinity mask does not describe this process, as on a machine whose "
                "processors form several processor groups"
            )
    elif getattr(os, "sched_setaffinity", None) is None:
        return "this platform has no call that confines a process to a processor"
    return None


@contextlib.contextmanager
def confined(cpu: int) -> Iterator[None]:
    """Confine the caller to one processor while it creates a process.

    A process created inside the context starts on ``cpu``, and so does every
    thread it starts, whenever it starts it. On Windows every thread of the
    calling process is confined for the duration; elsewhere only the calling
    thread is. The caller has its earlier processors back on exit.

    Args:
        cpu: Index of the processor, one of :func:`allowed_cpus`.

    Raises:
        ConfigError: If the platform cannot pin a process (see
            :func:`pinning_refusal`), the calling process may not use
            ``cpu``, or the operating system refuses the request.
    """
    what = f"cannot create a process on processor {cpu}"
    allowed = _checked(cpu, what)
    error = _set_cpus(0, (cpu,))
    if error is not None:
        raise ConfigError(f"{what}: {error}")
    try:
        yield
    finally:
        error = _set_cpus(0, allowed)
        if error is not None:
            raise ConfigError(
                f"cannot return to processor(s) {describe_cpus(allowed)} after creating a "
                f"process on processor {cpu}: {error}"
            )


def pin_process(pid: int, cpu: int) -> None:
    """Confine a running process to one processor.

    On Windows every thread of the process moves. Elsewhere the main thread
    moves, with every thread started from then on; a thread that is already
    running stays where it is, so use :func:`confined` to place a process
    from its start.

    Args:
        pid: Id of the process, which the caller must be allowed to change:
            itself or one of its children.
        cpu: Index of the processor, one of :func:`allowed_cpus`: a process
            is never moved outside the set of the calling process.

    Raises:
        ConfigError: If the platform cannot pin a process (see
            :func:`pinning_refusal`), the calling process may not use
            ``cpu``, or the operating system refuses the request.
    """
    what = f"cannot pin process {pid} to processor {cpu}"
    _checked(cpu, what)
    error = _set_cpus(pid, (cpu,))
    if error is not None:
        raise ConfigError(f"{what}: {error}")


def _checked(cpu: int, what: str) -> tuple[int, ...]:
    """Return the allowed processors, or raise when a process cannot be put on ``cpu``."""
    refusal = pinning_refusal()
    if refusal is not None:
        raise ConfigError(f"{what}: {refusal}")
    allowed = allowed_cpus()
    if cpu not in allowed:
        raise ConfigError(
            f"{what}: the calling process may not use it; it may use {describe_cpus(allowed)}"
        )
    return allowed


def _set_cpus(pid: int, cpus: Iterable[int]) -> str | None:
    """Give a process its processors and return the error, if any.

    ``pid`` 0 is the caller: the whole calling process on Windows, the
    calling thread elsewhere.
    """
    error: str | None = None
    if sys.platform == "win32":
        mask = 0
        for index in cpus:
            mask |= 1 << index
        error = _windows_set(pid or os.getpid(), mask)
    else:
        write = getattr(os, "sched_setaffinity", None)
        if write is None:
            error = "this platform has no call that confines a process to a processor"
        else:
            try:
                write(pid, set(cpus))
            except (OSError, ValueError) as failure:
                error = str(failure)
    return error


def _windows_mask() -> int | None:
    """Return the affinity mask of this process, or ``None`` when no single mask describes it."""
    mask: int | None = None
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetActiveProcessorGroupCount.restype = ctypes.c_ushort
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.GetProcessAffinityMask.restype = ctypes.c_int
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if kernel32.GetActiveProcessorGroupCount() == 1 and kernel32.GetProcessAffinityMask(
            kernel32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)
        ):
            mask = int(process.value) or None
    return mask


def _windows_set(pid: int, mask: int) -> str | None:
    """Set the affinity mask of a process; return the error, if any."""
    error: str | None = "not a Windows system"
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        access = _PROCESS_SET_INFORMATION | _PROCESS_QUERY_LIMITED_INFORMATION
        handle = kernel32.OpenProcess(access, 0, pid)
        if handle:
            try:
                done = kernel32.SetProcessAffinityMask(handle, mask)
                error = None if done else ctypes.FormatError(ctypes.get_last_error()).strip()
            finally:
                kernel32.CloseHandle(handle)
        else:
            error = ctypes.FormatError(ctypes.get_last_error()).strip()
    return error
