"""The environment a campaign session runs in.

:func:`capture_environment` records, once per session, the facts that can
change a measurement or explain a difference between sessions: the Python
version and implementation, the operating system and machine, the processor
model and count, the memory size, the versions of DispatchAtlas, NumPy, and
the optional solver backends that are installed, and the git commit of the
DispatchAtlas source and of the working directory when either is a git
checkout. It records no host or user name.
"""

from __future__ import annotations

import contextlib
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dispatchatlas.campaigns.config import available_cpus

__all__ = ["PACKAGES", "capture_environment", "git_state"]

PACKAGES = ("dispatchatlas", "numpy", "ortools", "pulp", "highspy", "pyscipopt", "gurobipy")
"""Distributions whose versions a session records when they are installed."""

_GIT_TIMEOUT = 10.0
"""Seconds a git query may take before it counts as unanswered."""


def capture_environment(directory: str | Path | None = None) -> dict[str, Any]:
    """Describe the machine and software a session runs on.

    Args:
        directory: The working directory whose git commit to record.
            Defaults to the current directory.

    Returns:
        A JSON-ready dictionary with the keys ``python``, ``platform``,
        ``cpu``, ``memory_bytes``, ``packages``, and ``git``. Facts that
        cannot be determined are ``None``; packages that are not installed
        and directories that are not git checkouts are left out.
    """
    source = Path(__file__).resolve().parent.parent
    git: dict[str, Any] = {}
    if _git(["ls-files", "--error-unmatch", "__init__.py"], source) is not None:
        git["dispatchatlas"] = git_state(source)
    working = git_state(Path.cwd() if directory is None else Path(directory))
    if working is not None:
        git["working_directory"] = working
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "description": platform.platform(),
        },
        "cpu": {"model": _cpu_model(), "logical": os.cpu_count(), "available": available_cpus()},
        "memory_bytes": _memory_bytes(),
        "packages": _package_versions(),
        "git": git,
    }


def git_state(directory: Path) -> dict[str, Any] | None:
    """Return the commit of the git checkout holding ``directory``, and whether it is modified.

    Returns:
        ``{"commit": ..., "dirty": ...}``, where ``dirty`` says whether
        tracked files differ from the commit (``None`` if unknown), or
        ``None`` when ``directory`` is not in a git checkout or git is not
        available.
    """
    commit = _git(["rev-parse", "HEAD"], directory)
    if not commit:
        return None
    status = _git(["status", "--porcelain", "--untracked-files=no"], directory)
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _git(arguments: list[str], directory: Path) -> str | None:
    """Run a read-only git query and return its output, or ``None`` if it fails."""
    command = ["git", "-c", "core.fsmonitor=false", *arguments]
    try:
        completed = subprocess.run(
            command,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGES:
        with contextlib.suppress(PackageNotFoundError):
            versions[name] = version(name)
    return versions


def _cpu_model() -> str | None:
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition(":")
                if key.strip() in ("model name", "Model", "cpu model", "Hardware"):
                    return value.strip() or None
    if sys.platform == "darwin":
        model = _sysctl("machdep.cpu.brand_string")
        if model:
            return model
    return platform.processor() or None


def _memory_bytes() -> int | None:
    if sys.platform == "darwin":
        size = _sysctl("hw.memsize")
        return int(size) if size and size.isdigit() else None
    if sys.platform == "win32":
        return _windows_memory()
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _windows_memory() -> int | None:  # pragma: no cover - Windows only
    """Return the physical memory size from ``GlobalMemoryStatusEx``."""
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only

        class _Status(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.length = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    return None
