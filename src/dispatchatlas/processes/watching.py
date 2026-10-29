"""Watching a long run from outside: one status line per interval, never a restart.

:func:`watch` polls a runner process and reports whether it is alive, how
many processes it has started that are still running, how many *other*
Python interpreters may run on the run's processors (a second run, a stray
test, an orphan of an earlier run), how many lines its rows file holds, and
how much memory the machine has available. It starts, restarts, and ends
nothing: what to do about a status is the operator's decision.

From the command line::

    python -m dispatchatlas.processes.watching --pid P --rows FILE --cpus MASK
        [--interval 60] [--status FILE] [--keep]

prints :func:`format_status` once per poll and appends it to ``--status``,
and stops after the first poll that finds the runner gone unless ``--keep``.
Where :func:`~dispatchatlas.processes.snapshots.snapshot` is unsupported
(macOS), the process counts are zero; whether the runner is alive is still
probed directly (:func:`~dispatchatlas.processes.snapshots.is_running`).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dispatchatlas.processes.affinity import parse_cpus
from dispatchatlas.processes.memory import available_memory_bytes
from dispatchatlas.processes.rows import count_lines
from dispatchatlas.processes.snapshots import descendants, interpreters_on, is_running, snapshot

__all__ = ["Status", "format_status", "main", "watch"]


@dataclass(frozen=True)
class Status:
    """One poll of :func:`watch`.

    Attributes:
        time_utc: When the poll was taken, ISO 8601 in UTC.
        runner_alive: Whether the runner process was running.
        live_children: Processes the runner started, at any depth, running.
        foreign_interpreters: Python interpreters on the watched processors
            that are neither the runner, its descendants, nor the watcher.
        foreign_pids: Their process ids.
        rows_lines: Complete lines of the rows file.
        rows_bytes: Size of the rows file.
        available_memory_bytes: Memory available on the machine, or ``None``.
    """

    time_utc: str
    runner_alive: bool
    live_children: int
    foreign_interpreters: int
    foreign_pids: tuple[int, ...]
    rows_lines: int
    rows_bytes: int
    available_memory_bytes: int | None


def watch(
    pid: int,
    *,
    rows: str | Path,
    cpus: Iterable[int],
    interval: float = 60.0,
    until: int | None = None,
    keep: bool = False,
) -> Iterator[Status]:
    """Yield a :class:`Status` of a runner every ``interval`` seconds.

    Args:
        pid: The runner's process id.
        rows: Its rows file, counted incrementally.
        cpus: The processors the run owns.
        interval: Seconds between polls.
        until: Stop after this many polls; ``None`` for no limit.
        keep: Keep polling after the runner is gone; by default the
            iterator stops after the first status that finds it gone.
    """
    watched = tuple(cpus)
    own = os.getpid()
    lines, offset = 0, 0
    polls = 0
    while True:
        listed = snapshot()
        alive = is_running(pid)
        tree = descendants(pid, listed)
        children = tuple(info for info in tree if is_running(info.pid)) if alive else ()
        excluded = {own, pid, *(info.pid for info in tree)}
        excluded |= {info.pid for info in descendants(own, listed)}
        foreign = tuple(
            info
            for info in interpreters_on(watched, listed, exclude=excluded)
            if is_running(info.pid)
        )
        added, size = count_lines(rows, offset=offset)
        if size < offset:  # the file was truncated (repaired): count again
            added, size = count_lines(rows)
            lines = 0
        lines += added
        offset = size
        yield Status(
            time_utc=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            runner_alive=alive,
            live_children=len(children),
            foreign_interpreters=len(foreign),
            foreign_pids=tuple(info.pid for info in foreign),
            rows_lines=lines,
            rows_bytes=size,
            available_memory_bytes=available_memory_bytes(),
        )
        polls += 1
        if (not alive and not keep) or (until is not None and polls >= until):
            return
        time.sleep(interval)


def format_status(status: Status) -> str:
    """Return a status as one line, fields in a fixed order; ``RUNNER-DEAD`` first when gone."""
    memory = (
        "unknown"
        if status.available_memory_bytes is None
        else f"{status.available_memory_bytes / 2**30:.1f}GiB"
    )
    foreign = ",".join(str(pid) for pid in status.foreign_pids) or "-"
    head = "RUNNER-DEAD " if not status.runner_alive else ""
    return (
        f"{head}{status.time_utc} runner={'alive' if status.runner_alive else 'dead'} "
        f"children={status.live_children} foreign={status.foreign_interpreters} "
        f"foreign_pids={foreign} rows={status.rows_lines} bytes={status.rows_bytes} "
        f"available={memory}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line entry; see the module docstring."""
    parser = argparse.ArgumentParser(
        prog="python -m dispatchatlas.processes.watching",
        description="Report a runner's state once per interval; never restart anything.",
    )
    parser.add_argument("--pid", type=int, required=True, help="the runner's process id")
    parser.add_argument("--rows", required=True, help="the runner's rows file")
    parser.add_argument("--cpus", required=True, help="processors: a mask 0x..., or 4-7,12")
    parser.add_argument("--interval", type=float, default=60.0, help="seconds between polls")
    parser.add_argument("--status", help="file each status line is appended to")
    parser.add_argument("--keep", action="store_true", help="keep polling after the runner ends")
    arguments = parser.parse_args(argv)
    try:
        cpus = parse_cpus(arguments.cpus)
    except ValueError as error:
        parser.error(str(error))
    return run_watch(
        arguments.pid,
        rows=arguments.rows,
        cpus=cpus,
        interval=arguments.interval,
        status=arguments.status,
        keep=arguments.keep,
    )


def run_watch(
    pid: int,
    *,
    rows: str | Path,
    cpus: Sequence[int],
    interval: float,
    status: str | Path | None,
    keep: bool,
) -> int:
    """Print and append status lines until the watch ends; 1 when the runner is gone."""
    last: Status | None = None
    for last in watch(pid, rows=rows, cpus=cpus, interval=interval, keep=keep):
        line = format_status(last)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        if status is not None:
            with Path(status).open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")
    return 0 if last is None or last.runner_alive else 1


if __name__ == "__main__":
    raise SystemExit(main())
