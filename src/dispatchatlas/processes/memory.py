"""The memory of the calling process and of the machine.

- :func:`peak_memory_bytes`: the peak resident set of the calling process
  (Windows ``PeakWorkingSetSize``, Linux ``VmHWM``, macOS ``ru_maxrss``).
- :func:`available_memory_bytes`: the physical memory available to new
  work (Windows ``ullAvailPhys``, Linux ``MemAvailable``).
- :func:`total_memory_bytes`: the physical memory of the machine.

Each returns ``None`` where the platform does not tell. The module uses the
standard library only.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["available_memory_bytes", "peak_memory_bytes", "total_memory_bytes"]

_STATUS = Path("/proc/self/status")
_MEMINFO = Path("/proc/meminfo")


def peak_memory_bytes() -> int | None:
    """Return the peak resident set of the calling process in bytes, or ``None``."""
    peak: int | None
    if sys.platform == "win32":
        peak = _windows_peak()
    elif sys.platform == "darwin":
        import resource  # noqa: PLC0415 - POSIX only

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # bytes on macOS
    else:
        peak = _proc_field(_STATUS, "VmHWM:")
    return peak


def available_memory_bytes() -> int | None:
    """Return the physical memory available to new work in bytes, or ``None``."""
    available: int | None
    if sys.platform == "win32":
        status = _windows_status()
        available = None if status is None else int(status.available_physical)
    else:
        available = _proc_field(_MEMINFO, "MemAvailable:")
    return available


def total_memory_bytes() -> int | None:
    """Return the physical memory of the machine in bytes, or ``None``."""
    total: int | None = None
    if sys.platform == "win32":
        status = _windows_status()
        total = None if status is None else int(status.total_physical)
    else:
        with contextlib.suppress(AttributeError, OSError, ValueError):
            total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    return total


def _proc_field(path: Path, field: str) -> int | None:
    """Return a ``kB`` field of a ``/proc`` file in bytes, or ``None``."""
    try:
        text = path.read_text(encoding="ascii")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(field):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def _windows_status() -> Any:  # pragma: no cover - Windows only
    """Return the ``MEMORYSTATUSEX`` of the machine, or ``None``."""
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        class _Status(ctypes.Structure):
            _fields_ = (
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            )

        status = _Status()
        status.length = ctypes.sizeof(_Status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status
    return None


def _windows_peak() -> int | None:  # pragma: no cover - Windows only
    """Return ``PeakWorkingSetSize`` of the calling process, or ``None``."""
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        class _Counters(ctypes.Structure):
            _fields_ = (
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return int(counters.PeakWorkingSetSize)
    return None
