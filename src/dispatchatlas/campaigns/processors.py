"""The processors a process may run on, and pinning a process to one of them.

An operating system can confine a process to some of the machine's logical
processors: a scheduler affinity set on Linux (``taskset``, the cpuset of a
container) or a process affinity mask on Windows. A child process inherits
the set of its parent. :func:`allowed_cpus` reads the set of the calling
process, so a worker count or a warning follows what the process may use and
not what the machine has, and :func:`pin_process` confines another process
to one processor.

- Where ``os.sched_getaffinity`` exists (Linux and some other POSIX systems),
  the set is read and written with the scheduler calls.
- On Windows the set is read with ``GetProcessAffinityMask`` and written with
  ``SetProcessAffinityMask``. One mask describes at most 64 processors, so on
  a machine with several processor groups the set is reported as every
  processor of the machine and pinning is refused.
- Elsewhere (macOS) the system offers no such call: the set is every
  processor of the machine and pinning is refused.

Processors are named by the index the operating system gives them, from 0.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Iterable

from dispatchatlas.core.errors import ConfigError

__all__ = [
    "allowed_cpus",
    "available_cpus",
    "describe_cpus",
    "pin_process",
    "pinning_refusal",
]

_PROCESS_SET_INFORMATION = 0x0200
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def allowed_cpus() -> tuple[int, ...]:
    """Return the indices of the processors this process may run on, sorted.

    When the platform has no affinity call, the call fails, or one mask
    cannot describe the machine, every processor of the machine is returned.
    The result is never empty.
    """
    if sys.platform == "win32":
        mask = _windows_mask()
        if mask:
            return tuple(index for index in range(mask.bit_length()) if mask >> index & 1)
    else:
        read = getattr(os, "sched_getaffinity", None)
        if read is not None:
            try:
                indices = tuple(sorted(read(0)))
            except OSError:
                indices = ()
            if indices:
                return indices
    return tuple(range(max(1, os.cpu_count() or 1)))


def available_cpus() -> int:
    """Return how many processors this process may run on, at least one.

    This is the size of :func:`allowed_cpus`, so affinity masks, containers,
    and ``taskset`` limits are respected.
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
                "the processors of this machine form several processor groups, which one "
                "affinity mask cannot describe"
            )
    elif getattr(os, "sched_setaffinity", None) is None:
        return "this platform has no call that confines a process to a processor"
    return None


def pin_process(pid: int, cpu: int) -> None:
    """Confine a process, with every thread it starts from now on, to one processor.

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
    refusal = pinning_refusal()
    if refusal is not None:
        raise ConfigError(f"cannot pin process {pid} to processor {cpu}: {refusal}")
    allowed = allowed_cpus()
    if cpu not in allowed:
        raise ConfigError(
            f"cannot pin process {pid} to processor {cpu}: the calling process may not use "
            f"it; it may use {describe_cpus(allowed)}"
        )
    error: str | None
    if sys.platform == "win32":
        error = _windows_pin(pid, cpu)
    else:
        try:
            os.sched_setaffinity(pid, {cpu})
            error = None
        except (OSError, ValueError) as failure:
            error = str(failure)
    if error is not None:
        raise ConfigError(f"cannot pin process {pid} to processor {cpu}: {error}")


def _windows_mask() -> int | None:
    """Return the affinity mask of this process, or ``None`` when no single mask describes it."""
    mask: int | None = None
    if sys.platform == "win32":
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


def _windows_pin(pid: int, cpu: int) -> str | None:
    """Set the affinity mask of a process to one processor; return the error, if any."""
    error: str | None = "not a Windows system"
    if sys.platform == "win32":
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
                done = kernel32.SetProcessAffinityMask(handle, 1 << cpu)
                error = None if done else ctypes.FormatError(ctypes.get_last_error()).strip()
            finally:
                kernel32.CloseHandle(handle)
        else:
            error = ctypes.FormatError(ctypes.get_last_error()).strip()
    return error
