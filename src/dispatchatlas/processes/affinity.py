"""Processor sets as text and masks, and a process pinning itself with a read-back.

A process pinned from outside is not always the process that computes: where
the ``python`` executable of an environment is a launcher that starts the real
interpreter as its child, a mask set on the launcher's handle reaches the
launcher alone. :func:`pin_self` therefore pins the *calling* process and reads
its mask back, and refuses when the operating system reports anything other
than what was asked.

- On Windows the mask is set with ``SetProcessAffinityMask`` on the current
  process, which moves every thread, and read with ``GetProcessAffinityMask``.
  One mask describes at most 64 processors, so on a machine with several
  processor groups nothing is pinned.
- Where ``os.sched_setaffinity`` exists (Linux), the set of the calling thread
  is written, and every thread started later inherits it; a thread that is
  already running keeps its own, so pin first thing.
- Elsewhere (macOS) the system has no such call and pinning is refused.

This module uses the standard library only, so a child process can import it
in a few milliseconds. Its errors are :class:`RuntimeError` and
:class:`ValueError`; :mod:`dispatchatlas.campaigns.processors` wraps the same
primitives with the library's own errors.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

__all__ = [
    "PinReport",
    "confined",
    "cpus_of_mask",
    "current_cpus",
    "current_mask",
    "describe_cpus",
    "mask_of_cpus",
    "parse_cpus",
    "pin_self",
    "pinning_refusal_text",
]

_PROCESS_SET_INFORMATION = 0x0200
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_RANGE = re.compile(r"^(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class PinReport:
    """What :func:`pin_self` set and read back.

    Attributes:
        wanted: The mask asked for.
        inherited: The mask the process had before the call, ``None`` when it
            could not be read. It shows whether a parent that created the
            process on a processor reached the process that computes.
        read: The mask read back after the call, equal to ``wanted``.
        cpus: The processors of ``read``, sorted.
    """

    wanted: int
    inherited: int | None
    read: int
    cpus: tuple[int, ...]


def parse_cpus(text: str) -> tuple[int, ...]:
    """Return the processors a text names, sorted and distinct.

    Three forms are accepted: a hexadecimal mask (``"0xF0F"``), ranges and
    indices separated by commas (``"4-7,12"``), and a single index (``"3"``).

    Raises:
        ValueError: If the text is none of these forms, names a processor
            twice, has a range whose end precedes its start, or names no
            processor.
    """
    stripped = text.strip()
    if stripped.lower().startswith("0x"):
        try:
            mask = int(stripped, 16)
        except ValueError:
            raise ValueError(f"{text!r} is not a hexadecimal processor mask") from None
        cpus = cpus_of_mask(mask)
    else:
        found: list[int] = []
        for part in stripped.split(","):
            match = _RANGE.match(part.strip())
            if match is None:
                raise ValueError(
                    f"{text!r} names processors neither as a mask (0x...) nor as indices and "
                    "ranges such as 4-7,12"
                )
            first = int(match.group(1))
            last = first if match.group(2) is None else int(match.group(2))
            if last < first:
                raise ValueError(
                    f"{text!r} has the range {part.strip()} whose end precedes its start"
                )
            found.extend(range(first, last + 1))
        if len(set(found)) != len(found):
            raise ValueError(f"{text!r} names a processor more than once")
        cpus = tuple(sorted(found))
    if not cpus:
        raise ValueError(f"{text!r} names no processor")
    return cpus


def cpus_of_mask(mask: int) -> tuple[int, ...]:
    """Return the processors whose bits are set in ``mask``, sorted."""
    if mask < 0:
        raise ValueError(f"a processor mask is not negative: {mask}")
    return tuple(index for index in range(mask.bit_length()) if mask >> index & 1)


def mask_of_cpus(cpus: Iterable[int]) -> int:
    """Return the mask with the bits of ``cpus`` set."""
    mask = 0
    for index in cpus:
        if index < 0:
            raise ValueError(f"a processor index is not negative: {index}")
        mask |= 1 << index
    return mask


def current_mask() -> int | None:
    """Return the affinity mask of the calling process (Linux: thread), or ``None``.

    ``None`` means the platform has no affinity call, the call failed, or one
    Windows mask does not describe the process.
    """
    mask: int | None = None
    if sys.platform == "win32":
        mask = _windows_mask()
    elif hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(OSError):
            mask = mask_of_cpus(os.sched_getaffinity(0)) or None
    return mask


def current_cpus() -> tuple[int, ...] | None:
    """Return the processors of :func:`current_mask`, or ``None`` when it cannot be read."""
    mask = current_mask()
    return None if mask is None else cpus_of_mask(mask)


def pinning_refusal_text() -> str | None:
    """Return why this platform cannot pin a process, or ``None`` when it can."""
    if sys.platform == "win32":
        if _windows_mask() is None:
            return (
                "one affinity mask does not describe this process, as on a machine whose "
                "processors form several processor groups"
            )
    elif not hasattr(os, "sched_setaffinity"):
        return "this platform has no call that confines a process to a processor"
    return None


def pin_self(cpus: int | Iterable[int]) -> PinReport:
    """Confine the calling process to processors, then read its mask back.

    Args:
        cpus: A processor index, or several. Each must be in the set the
            process has now: a process never widens its own set.

    Returns:
        The masks wanted, inherited, and read back.

    Raises:
        RuntimeError: If the platform cannot pin (see
            :func:`pinning_refusal_text`), a processor is outside the current
            set, the operating system refuses, or the mask read back differs
            from the one asked for.
    """
    wanted_cpus = (cpus,) if isinstance(cpus, int) else tuple(cpus)
    wanted = mask_of_cpus(wanted_cpus)
    refusal = pinning_refusal_text()
    if refusal is not None:
        raise RuntimeError(f"cannot pin this process: {refusal}")
    if not wanted:
        raise RuntimeError("cannot pin this process to no processor")
    inherited = _read_mask()
    if inherited is not None and wanted & ~inherited:
        raise RuntimeError(
            f"cannot pin this process to processor(s) {describe_cpus(cpus_of_mask(wanted))}: "
            f"it may use {describe_cpus(cpus_of_mask(inherited))}"
        )
    error = _set_cpus(0, wanted_cpus)
    if error is not None:
        raise RuntimeError(f"cannot pin this process: {error}")
    read = _read_mask()
    if read != wanted:
        raise RuntimeError(
            f"the affinity mask read back ({_hex(read)}) differs from the one set ({_hex(wanted)})"
        )
    return PinReport(wanted=wanted, inherited=inherited, read=read, cpus=cpus_of_mask(read))


@contextlib.contextmanager
def confined(cpus: int | Iterable[int]) -> Iterator[None]:
    """Confine the caller to processors for the duration of the context.

    A process created inside the context starts on ``cpus``. On Windows every
    thread of the calling process is confined for the duration; elsewhere
    only the calling thread is. The earlier set is restored on exit.

    Raises:
        RuntimeError: If the set cannot be read or written, or a processor
            is outside it.
    """
    wanted = (cpus,) if isinstance(cpus, int) else tuple(cpus)
    what = f"cannot confine this process to processor(s) {describe_cpus(wanted)}"
    before = current_cpus()
    if before is None:
        raise RuntimeError(f"{what}: {pinning_refusal_text()}")
    if not wanted or not set(wanted) <= set(before):
        raise RuntimeError(f"{what}: this process may use {describe_cpus(before)}")
    error = _set_cpus(0, wanted)
    if error is not None:
        raise RuntimeError(f"{what}: {error}")
    try:
        yield
    finally:
        error = _set_cpus(0, before)
        if error is not None:
            raise RuntimeError(f"cannot return to processor(s) {describe_cpus(before)}: {error}")


def describe_cpus(cpus: Iterable[int]) -> str:
    """Return processor indices with runs collapsed: ``"0-3, 8"``."""
    ordered = sorted(set(cpus))
    parts: list[str] = []
    start = 0
    for end in range(1, len(ordered) + 1):
        if end < len(ordered) and ordered[end] == ordered[end - 1] + 1:
            continue
        first, last = ordered[start], ordered[end - 1]
        parts.append(str(first) if first == last else f"{first}-{last}")
        start = end
    return ", ".join(parts) if parts else "none"


def _hex(mask: int | None) -> str:
    return "unreadable" if mask is None else f"{mask:#x}"


def _read_mask() -> int | None:
    """Return the current mask; a seam for tests of the read-back refusal."""
    return current_mask()


def _set_cpus(pid: int, cpus: Iterable[int]) -> str | None:
    """Give a process its processors and return the error, if any.

    ``pid`` 0 is the caller: the whole calling process on Windows, the
    calling thread elsewhere.
    """
    error: str | None = None
    if sys.platform == "win32":
        error = _windows_set(pid or os.getpid(), mask_of_cpus(cpus))
    elif hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(pid, set(cpus))
        except (OSError, ValueError) as failure:
            error = str(failure)
    else:
        error = "this platform has no call that confines a process to a processor"
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
    """Set the affinity mask of a process; return the error, if any.

    The calling process is set through its pseudo-handle, so no access right
    is needed; another process is opened by id.
    """
    error: str | None = "not a Windows system"
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        if pid == os.getpid():
            done = kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask)
            return None if done else ctypes.FormatError(ctypes.get_last_error()).strip()
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
