"""The environment a campaign session runs in.

:func:`capture_environment` records, once per session, the facts that can
change a measurement or explain a difference between sessions: the Python
version and implementation, the operating system and machine, the processor
model with its sockets and cores, the processors the process may use, the
thread limits its environment sets for numerical libraries, the memory size,
the versions of DispatchAtlas, NumPy, and the optional solver backends that
are installed, where the DispatchAtlas distribution was installed from and at
which revision, a digest of the DispatchAtlas source files, every installed
plugin distribution with its version and revision, and the git state of the
DispatchAtlas source and of the working directory when either is a git
checkout. It records no host or user name.

:func:`sample_load` measures how busy the processors of the machine are over
a short interval. A session takes one sample before its first run and one
after its last, so its record shows a machine that other work shared.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import Distribution, PackageNotFoundError, distribution, entry_points
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from dispatchatlas.campaigns.config import THREAD_VARIABLES
from dispatchatlas.campaigns.processors import affinity_cpus, available_cpus

__all__ = [
    "LIBRARY",
    "LOAD_SAMPLE_SECONDS",
    "PACKAGES",
    "PLUGIN_GROUPS",
    "UNTRACKED_BYTES",
    "UNTRACKED_HASHED",
    "capture_environment",
    "distribution_state",
    "git_state",
    "plugin_states",
    "sample_load",
    "source_digest",
]

LIBRARY = "dispatchatlas"
"""Name of the distribution and of the package whose source a session identifies."""

PACKAGES = (LIBRARY, "numpy", "ortools", "pulp", "highspy", "pyscipopt", "gurobipy")
"""Distributions whose versions a session records when they are installed."""

PLUGIN_GROUPS = ("dispatchatlas.solvers", "dispatchatlas.benchmarks")
"""Entry-point groups whose distributions a session records as plugins."""

LOAD_SAMPLE_SECONDS = 0.1
"""Seconds over which :func:`sample_load` measures the processors by default."""

UNTRACKED_HASHED = 1000
"""Most untracked files whose content enters the digest of a git state."""

UNTRACKED_BYTES = 1 << 24
"""Most bytes of untracked files that are read for the digest of a git state."""

_GIT_TIMEOUT = 10.0
"""Seconds a git query may take before it counts as unanswered."""

_CHUNK = 1 << 20
"""Bytes read at a time from a file that is hashed."""

_STAT_FIELDS = 8
"""Leading fields of a ``cpu`` line of ``/proc/stat`` that partition the processor time: user,
nice, system, idle, iowait, irq, softirq, and steal. The guest fields after them repeat time
that user and nice already count."""


def capture_environment(
    directory: str | Path | None = None, *, exclude: Sequence[str | Path] = ()
) -> dict[str, Any]:
    """Describe the machine and software a session runs on.

    Args:
        directory: The working directory whose git state to record.
            Defaults to the current directory.
        exclude: Directories left out of every git state: that of the
            working directory, of the DispatchAtlas source, and of the
            checkout of every editable plugin. The directory a campaign
            writes its results to is one, since its files change with
            every session.

    Returns:
        A JSON-ready dictionary with the keys ``python``, ``platform``,
        ``cpu``, ``threads``, ``memory_bytes``, ``packages``, ``library``,
        ``plugins``, and ``git``.

        ``cpu`` holds the processor ``model``, its ``sockets`` and physical
        ``cores``, the ``logical`` processors of the machine, and the
        processors the process may use: their indices as ``allowed`` and
        their number as ``available``. When the operating system does not
        tell which processors the process is confined to, ``allowed`` is
        ``None`` and ``available`` counts every processor of the machine.

        ``threads`` holds the value of every variable of
        :data:`~dispatchatlas.campaigns.config.THREAD_VARIABLES` in the
        environment of the process, ``None`` when it is not set; worker
        processes start with the campaign's ``threads_per_worker`` in each
        of them instead.

        ``library`` describes the DispatchAtlas distribution as
        :func:`distribution_state` does, with three differences. Its
        ``revision`` is the commit of the source when the source files are
        tracked in a git checkout. ``source_digest`` is
        :func:`source_digest` of the package. And when the package that
        runs is not the one the installed distribution provides, for
        example a copy that comes first on the module search path, the
        installer's record does not describe it: ``origin`` is then
        ``"path"``, and ``version``, ``editable``, and, outside a git
        checkout, ``revision`` are ``None``, while ``packages`` still
        states the version of the installed distribution. ``plugins`` maps
        the name of every distribution that provides an entry point of
        :data:`PLUGIN_GROUPS` to its :func:`distribution_state`.

        ``git`` holds the :func:`git_state` of the DispatchAtlas source
        under ``dispatchatlas`` and of the working directory under
        ``working_directory``. Facts that cannot be determined are ``None``;
        packages that are not installed and directories that are not git
        checkouts are left out.
    """
    source = Path(__file__).resolve().parent.parent
    library = _library_state(source)
    git: dict[str, Any] = {}
    if _git(["ls-files", "--error-unmatch", "__init__.py"], source) is not None:
        checkout = git_state(source, exclude=exclude)
        if checkout is not None:
            git[LIBRARY] = checkout
            library["revision"] = checkout["commit"]
    library["source_digest"] = source_digest(source)
    working = git_state(Path.cwd() if directory is None else Path(directory), exclude=exclude)
    if working is not None:
        git["working_directory"] = working
    confined = affinity_cpus()
    sockets, cores = _cpu_topology()
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
        "cpu": {
            "model": _cpu_model(),
            "sockets": sockets,
            "cores": cores,
            "logical": os.cpu_count(),
            "available": available_cpus(),
            "allowed": None if confined is None else list(confined),
        },
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "memory_bytes": _memory_bytes(),
        "packages": _package_versions(),
        "library": library,
        "plugins": plugin_states(exclude=exclude),
        "git": git,
    }


# ------------------------------------------------------------ distributions


def distribution_state(name: str, *, exclude: Sequence[str | Path] = ()) -> dict[str, Any] | None:
    """Return the version of an installed distribution and where it was installed from.

    The installer's record of a direct installation (``direct_url.json`` in
    the distribution's metadata) tells the origin.

    Args:
        name: Name of the distribution.
        exclude: Directories left out of the git state of an editable
            distribution's checkout.

    Returns:
        ``None`` when no such distribution is installed. Otherwise a
        dictionary with ``version``; ``origin``, which is ``"vcs"`` for an
        installation from a version-control URL, ``"archive"`` for one from
        an archive file or URL, ``"directory"`` for one from a local
        directory, and ``"index"`` when the installer recorded no direct
        origin, as for a package index; ``editable``, whether the installed
        distribution runs the files of that local directory; and
        ``revision``: the commit a ``"vcs"`` origin names, the hash of an
        ``"archive"``, and the commit of an editable directory that is a git
        checkout, whose :func:`git_state` is then under ``git``. The
        revision of a copy built from a local directory is ``None``: the
        record does not say which state of the directory was built.
    """
    try:
        found = distribution(name)
    except PackageNotFoundError:
        return None
    state, folder = _described(found)
    if folder is not None:
        checkout = git_state(folder, exclude=exclude)
        if checkout is not None:
            state["revision"] = checkout["commit"]
            state["git"] = checkout
    return state


def plugin_states(*, exclude: Sequence[str | Path] = ()) -> dict[str, dict[str, Any]]:
    """Return the :func:`distribution_state` of every installed plugin distribution, by name.

    A plugin distribution is one that provides an entry point of
    :data:`PLUGIN_GROUPS`.

    Args:
        exclude: Directories left out of the git state of every editable
            plugin's checkout.
    """
    names: set[str] = set()
    for group in PLUGIN_GROUPS:
        for entry in entry_points(group=group):
            provider = entry.dist
            if provider is not None and provider.name != LIBRARY:
                names.add(provider.name)
    states: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        state = distribution_state(name, exclude=exclude)
        if state is not None:
            states[name] = state
    return states


def source_digest(package: str | Path | None = None) -> str | None:
    """Return a digest of the files of a package directory.

    The SHA-256 digest covers the relative path and the content of every
    file under the directory, in path order, except compiled bytecode. Line
    endings are read as line feeds, so a checkout and an installed copy of
    the same files have one digest on every platform.

    Args:
        package: The package directory. Defaults to the DispatchAtlas
            package this module belongs to.

    Returns:
        The hexadecimal digest, or ``None`` when a file cannot be read.
    """
    root = Path(__file__).resolve().parent.parent if package is None else Path(package)
    hasher = hashlib.sha256()
    try:
        files = sorted(
            (path.relative_to(root).as_posix(), path)
            for path in root.rglob("*")
            if "__pycache__" not in path.parts and path.suffix != ".pyc" and path.is_file()
        )
        for name, path in files:
            content = path.read_bytes().replace(b"\r\n", b"\n")
            hasher.update(name.encode("utf-8") + b"\0")
            hasher.update(hashlib.sha256(content).digest())
    except OSError:
        return None
    return hasher.hexdigest()


def _library_state(package: Path) -> dict[str, Any]:
    """Return the state of the DispatchAtlas distribution, as far as it describes ``package``.

    ``package`` is the directory of the package that runs. The installer's
    record describes it only when the installed distribution provides that
    directory; for any other directory the origin is ``"path"`` and the
    other facts are unknown.
    """
    unknown = {"version": None, "origin": None, "editable": None, "revision": None}
    try:
        found = distribution(LIBRARY)
    except PackageNotFoundError:
        return unknown
    state, folder = _described(found)
    if not _provides(found, folder, package):
        return {**unknown, "origin": "path"}
    return state


def _provides(found: Distribution, folder: Path | None, package: Path) -> bool:
    """Return whether a distribution provides the package directory ``package``.

    An editable distribution provides the packages under the directory it
    was installed from, ``folder``; any other provides the package
    directory among its installed files.
    """
    try:
        if folder is not None:
            return package.resolve().is_relative_to(folder.resolve())
        return Path(str(found.locate_file(package.name))).samefile(package)
    except (OSError, ValueError):
        return False


def _described(found: Distribution) -> tuple[dict[str, Any], Path | None]:
    """Return the state of a distribution without its git state, and its editable directory."""
    state: dict[str, Any] = {
        "version": found.version,
        "origin": "index",
        "editable": False,
        "revision": None,
    }
    record = _direct_url(found)
    vcs, archive, folder = (record.get(key) for key in ("vcs_info", "archive_info", "dir_info"))
    if isinstance(vcs, Mapping):
        commit = vcs.get("commit_id")
        state.update(origin="vcs", revision=commit if isinstance(commit, str) else None)
    elif isinstance(archive, Mapping):
        state.update(origin="archive", revision=_archive_hash(archive))
    elif isinstance(folder, Mapping):
        state.update(origin="directory", editable=folder.get("editable") is True)
        if state["editable"]:
            return state, _local_path(record.get("url"))
    return state, None


def _direct_url(found: Distribution) -> Mapping[str, Any]:
    """Return the installer's record of a direct installation, empty when there is none."""
    try:
        text = found.read_text("direct_url.json")
        record = json.loads(text) if text else {}
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, Mapping) else {}


def _archive_hash(archive: Mapping[str, Any]) -> str | None:
    """Return the hash an archive record states, as ``<algorithm>=<digest>``."""
    hashes = archive.get("hashes")
    if isinstance(hashes, Mapping) and hashes:
        algorithm = "sha256" if "sha256" in hashes else sorted(str(key) for key in hashes)[0]
        return f"{algorithm}={hashes[algorithm]}"
    stated = archive.get("hash")
    return stated if isinstance(stated, str) else None


def _local_path(url: object) -> Path | None:
    """Return the local path a ``file`` URL names, or ``None`` for any other value."""
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return Path(url2pathname(parsed.path))


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGES:
        with contextlib.suppress(PackageNotFoundError):
            versions[name] = distribution(name).version
    return versions


# ---------------------------------------------------------------------- git


def git_state(directory: Path, *, exclude: Sequence[str | Path] = ()) -> dict[str, Any] | None:
    """Return the commit of the git checkout holding ``directory``, and how it is modified.

    Args:
        directory: A directory in the checkout.
        exclude: Directories whose files are left out of the comparison
            with the commit. One outside the checkout changes nothing.

    Returns:
        ``None`` when ``directory`` is not in a git checkout or git is not
        available. Otherwise a dictionary with ``commit``; ``dirty``,
        whether the checkout differs from the commit in a tracked file or
        holds an untracked file that git does not ignore; ``untracked``,
        the number of such untracked files; and ``digest``, a SHA-256
        digest of the uncommitted changes: the status of every changed
        path, the difference of the tracked files from the commit, and the
        content of the untracked files. Two checkouts of one commit with
        different uncommitted changes therefore have different digests. The
        digest of a checkout that is not dirty is ``None``, and ``dirty``,
        ``untracked``, and ``digest`` are ``None`` when git does not answer.

    The untracked files are read in path order, at most
    :data:`UNTRACKED_BYTES` bytes in all: a file that would pass that total
    enters the digest by its path, its size, and its modification time
    instead of its content. More than :data:`UNTRACKED_HASHED` untracked
    files enter by their paths only. The queries leave the checkout as it
    is: git writes neither its index nor any other file for them.
    """
    answer = _git(["rev-parse", "--show-toplevel", "HEAD"], directory)
    lines = [] if answer is None else answer.decode("utf-8", "replace").splitlines()
    if len(lines) != 2 or not lines[1]:  # the top level and the commit
        return None
    top, commit = Path(lines[0]), lines[1]
    unknown = {"commit": commit, "dirty": None, "untracked": None, "digest": None}
    paths = ["--", ":(top)", *_excluded(top, exclude)]
    status = _git(["status", "--porcelain", "-z", "--untracked-files=all", *paths], top)
    if status is None:
        return unknown
    entries = [entry for entry in status.split(b"\0") if entry]
    untracked = [entry[3:] for entry in entries if entry.startswith(b"?? ")]
    if not entries:
        return {"commit": commit, "dirty": False, "untracked": 0, "digest": None}
    hasher = hashlib.sha256(status)
    if len(untracked) < len(entries):
        difference = _git(["diff", "HEAD", "--binary", "--no-ext-diff", *paths], top)
        if difference is None:
            return unknown
        hasher.update(difference)
    if len(untracked) <= UNTRACKED_HASHED:
        budget = UNTRACKED_BYTES
        for name in sorted(untracked):
            entry, budget = _untracked_entry(top / os.fsdecode(name), budget)
            hasher.update(name + b"\0" + entry)
    return {
        "commit": commit,
        "dirty": True,
        "untracked": len(untracked),
        "digest": hasher.hexdigest(),
    }


def _excluded(top: Path, exclude: Sequence[str | Path]) -> list[str]:
    """Return the pathspecs that leave the directories of ``exclude`` inside ``top`` out."""
    specs: list[str] = []
    for item in exclude:
        with contextlib.suppress(ValueError, OSError):
            relative = Path(item).resolve().relative_to(top.resolve()).as_posix()
            if relative != ".":
                specs.append(f":(top,exclude){relative}")
    return specs


def _untracked_entry(path: Path, budget: int) -> tuple[bytes, int]:
    """Return what an untracked file adds to a digest, and the bytes still to be read.

    A file of at most ``budget`` bytes adds the digest of its content and
    takes its size from the budget. A larger one adds its size and its
    modification time, and one that cannot be examined the digest of nothing.
    """
    try:
        found = path.stat()
    except OSError:
        return hashlib.sha256().digest(), budget
    if found.st_size > budget:
        return f"size {found.st_size} modified {found.st_mtime_ns}".encode("ascii"), budget
    return _file_digest(path), budget - found.st_size


def _file_digest(path: Path) -> bytes:
    """Return the SHA-256 digest of a file, or of nothing when it cannot be read."""
    hasher = hashlib.sha256()
    with contextlib.suppress(OSError), path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            hasher.update(chunk)
    return hasher.digest()


def _git(arguments: list[str], directory: Path) -> bytes | None:
    """Run a git query and return its output, or ``None`` if it fails.

    The query runs with git's optional locks off, so a command that
    refreshes the index, such as ``status``, does not write it back.
    """
    command = ["git", "-c", "core.fsmonitor=false", "-c", "core.quotepath=false", *arguments]
    try:
        completed = subprocess.run(
            command,
            cwd=directory,
            capture_output=True,
            timeout=_GIT_TIMEOUT,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


# --------------------------------------------------------------------- load


def sample_load(seconds: float = LOAD_SAMPLE_SECONDS) -> dict[str, Any]:
    """Measure how busy the processors of the machine are over a short interval.

    The processor times of the whole machine are read twice, ``seconds``
    apart, so the sample counts the work of every process, this one
    included.

    Args:
        seconds: Length of the interval.

    Returns:
        A JSON-ready dictionary with ``at``, the UTC time the sample began
        in ISO 8601; ``seconds``; ``busy``, the share of the processor time
        of the machine that was not idle over the interval, between 0 and
        1; and ``load_average``, the one-minute load average of the system.
        ``busy`` is ``None`` where the platform does not give the processor
        times, and no interval then passes; ``load_average`` is ``None``
        where the platform has none.
    """
    at = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    busy: float | None = None
    first = _processor_times()
    if first is not None:
        time.sleep(max(0.0, seconds))
        second = _processor_times()
        if second is not None and second[1] > first[1]:
            idle = (second[0] - first[0]) / (second[1] - first[1])
            busy = min(1.0, max(0.0, 1.0 - idle))
    average: float | None = None
    read = getattr(os, "getloadavg", None)
    if read is not None:
        with contextlib.suppress(OSError):
            average = float(read()[0])
    return {"at": at, "seconds": float(seconds), "busy": busy, "load_average": average}


def _processor_times() -> tuple[float, float] | None:
    """Return the idle and the total processor time of the machine, in one unit."""
    if sys.platform == "win32":
        times = _windows_processor_times()
    else:
        times = _stat_processor_times()
    return times


def _stat_processor_times() -> tuple[float, float] | None:
    """Return the idle and the total processor time from the first line of ``/proc/stat``."""
    try:
        line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    return _stat_times(line)


def _stat_times(line: str) -> tuple[float, float] | None:
    """Return the idle and the total processor time a ``cpu`` line of ``/proc/stat`` states.

    The idle time includes the time spent waiting for input and output.
    """
    fields = line.split()
    try:
        values = [float(value) for value in fields[1:]]
    except ValueError:
        return None
    if not fields or fields[0] != "cpu" or len(values) < 4:  # user, nice, system, idle
        return None
    waiting = values[4] if len(values) > 4 else 0.0
    return values[3] + waiting, sum(values[:_STAT_FIELDS])


def _windows_processor_times() -> tuple[float, float] | None:  # pragma: no cover - Windows only
    """Return the idle and the total processor time from ``GetSystemTimes``."""
    times: tuple[float, float] | None = None
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        def ticks(value: wintypes.FILETIME) -> float:
            return float((value.dwHighDateTime << 32) | value.dwLowDateTime)

        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            # The kernel time of the system includes its idle time.
            times = (ticks(idle), ticks(kernel) + ticks(user))
    return times


# ---------------------------------------------------------------- processor


def _cpu_model() -> str | None:
    if sys.platform.startswith("linux"):
        for key, value in _cpuinfo():
            if key in ("model name", "Model", "cpu model", "Hardware") and value:
                return value
    if sys.platform == "darwin":
        model = _sysctl("machdep.cpu.brand_string")
        if model:
            return model
    if sys.platform == "win32":
        model = _windows_cpu_model()
        if model:
            return model
    return platform.processor() or None


def _cpu_topology() -> tuple[int | None, int | None]:
    """Return the processor sockets and the physical cores of the machine, where known."""
    if sys.platform == "win32":
        counts = _windows_topology()
    elif sys.platform == "darwin":
        counts = (_count(_sysctl("hw.packages")), _count(_sysctl("hw.physicalcpu")))
    else:
        sockets: set[str] = set()
        cores: set[tuple[str, str]] = set()
        socket = ""
        for key, value in _cpuinfo():
            if key == "physical id":
                socket = value
                sockets.add(value)
            elif key == "core id":
                cores.add((socket, value))
        counts = (len(sockets) or None, len(cores) or None)
    return counts


def _cpuinfo() -> list[tuple[str, str]]:
    """Return the keys and values of ``/proc/cpuinfo``, empty when it cannot be read."""
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [
        (key.strip(), value.strip()) for key, _, value in (line.partition(":") for line in lines)
    ]


def _count(text: str | None) -> int | None:
    return int(text) if text and text.isdigit() and int(text) > 0 else None


def _windows_cpu_model() -> str | None:  # pragma: no cover - Windows only
    """Return the processor name the registry holds for the first processor."""
    model: str | None = None
    if sys.platform == "win32":
        import winreg  # noqa: PLC0415 - Windows only

        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with contextlib.suppress(OSError), winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
            value, _ = winreg.QueryValueEx(handle, "ProcessorNameString")
            model = str(value).strip() or None
    return model


def _windows_topology() -> tuple[int | None, int | None]:  # pragma: no cover - Windows only
    """Return the sockets and cores that ``GetLogicalProcessorInformationEx`` lists."""
    counts: list[int | None] = [None, None]
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        query = ctypes.windll.kernel32.GetLogicalProcessorInformationEx
        for index, relation in enumerate((3, 0)):  # processor packages, then processor cores
            size = wintypes.DWORD(0)
            query(relation, None, ctypes.byref(size))
            buffer = ctypes.create_string_buffer(max(1, size.value))
            if size.value == 0 or not query(relation, buffer, ctypes.byref(size)):
                continue
            # Each entry states its relationship in four bytes, then its size in four.
            data, offset, found = buffer.raw[: size.value], 0, 0
            while offset + 8 <= len(data):
                length = int.from_bytes(data[offset + 4 : offset + 8], "little")
                if length <= 0:
                    break
                found += 1
                offset += length
            counts[index] = found or None
    return counts[0], counts[1]


# ------------------------------------------------------------------- memory


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
