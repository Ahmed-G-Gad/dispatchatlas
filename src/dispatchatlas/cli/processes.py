"""``dispatchatlas processes``: launch, watch, list, and end processes; check rows files."""

from __future__ import annotations

import argparse

from dispatchatlas.cli._common import (
    PROG,
    CommandError,
    Commands,
    UsageError,
    add_command,
    add_group,
    emit_json,
    emit_table,
    existing_file,
    positive_number,
)
from dispatchatlas.processes.affinity import parse_cpus
from dispatchatlas.processes.launching import launch_detached
from dispatchatlas.processes.rows import TornRowsError, count_lines, read_rows, repair_rows
from dispatchatlas.processes.snapshots import descendants, end_process, interpreters_on, snapshot
from dispatchatlas.processes.watching import run_watch

__all__ = ["register"]


def register(commands: Commands) -> None:
    """Add the ``processes`` command group."""
    group = add_group(
        commands,
        "processes",
        "Launch, watch, list, and end processes; check and repair rows files.",
        "Tools for long runs of calls in pinned child processes: start a command detached "
        "from this shell, watch a runner, list processes with their affinity masks, end "
        "processes on request, and check or repair a rows file.",
    )
    launch = add_command(
        group,
        "launch",
        _launch,
        "Start a command detached from this shell, its output appended to a log.",
        epilog="examples:\n"
        f"  {PROG} processes launch --log run.log --cwd . -- python driver.py --workers 8",
    )
    launch.add_argument("--log", required=True, help="file the command's output is appended to")
    launch.add_argument("--cwd", default=".", help="working directory of the command")
    launch.add_argument(
        "--env", action="append", default=[], metavar="K=V", help="add an environment variable"
    )
    launch.add_argument("argv", nargs=argparse.REMAINDER, metavar="-- COMMAND", help="the command")

    watch = add_command(
        group,
        "watch",
        _watch,
        "Print a status line about a runner once per interval; never restart anything.",
        epilog="examples:\n"
        f"  {PROG} processes watch --pid 4242 --rows rows.jsonl --cpus 0-7 --status status.log",
    )
    watch.add_argument("--pid", type=int, required=True, help="the runner's process id")
    watch.add_argument("--rows", required=True, help="the runner's rows file")
    watch.add_argument("--cpus", required=True, help="the run's processors: 0x... or 4-7,12")
    watch.add_argument(
        "--interval", type=positive_number, default=60.0, help="seconds between polls"
    )
    watch.add_argument("--status", help="file each status line is appended to")
    watch.add_argument("--keep", action="store_true", help="keep polling after the runner ends")

    listing = add_command(
        group,
        "list",
        _list,
        "List processes with parent, affinity mask, and start time.",
    )
    listing.add_argument("--children-of", type=int, metavar="PID", help="only this process's tree")
    listing.add_argument(
        "--interpreters-on", metavar="CPUS", help="only Python interpreters on these processors"
    )

    end = add_command(group, "end", _end, "End processes at once, on request.")
    end.add_argument("pids", nargs="+", type=int, metavar="PID", help="process id")

    check = add_command(
        group,
        "rows-check",
        _rows_check,
        "Count the rows of a rows file, or report its torn line.",
    )
    check.add_argument("file", help="the rows file")

    repair = add_command(
        group,
        "rows-repair",
        _rows_repair,
        "Move the bytes after the last line feed of a rows file aside.",
    )
    repair.add_argument("file", help="the rows file")


def _cpus(text: str) -> tuple[int, ...]:
    try:
        return parse_cpus(text)
    except ValueError as error:
        raise UsageError(str(error)) from None


def _launch(arguments: argparse.Namespace) -> int:
    command = list(arguments.argv)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise UsageError("give the command after --")
    environment: dict[str, str] = {}
    for item in arguments.env:
        name, equals, value = item.partition("=")
        if not equals or not name:
            raise UsageError(f"--env takes NAME=VALUE, not {item!r}")
        environment[name] = value
    launched = launch_detached(
        command, log=arguments.log, cwd=arguments.cwd, environment=environment
    )
    if arguments.json:
        emit_json({"pid": launched.pid, "breakaway": launched.breakaway, "log": str(launched.log)})
    else:
        print(f"started pid {launched.pid}, breakaway {'yes' if launched.breakaway else 'no'}")
    return 0


def _watch(arguments: argparse.Namespace) -> int:
    return run_watch(
        arguments.pid,
        rows=arguments.rows,
        cpus=_cpus(arguments.cpus),
        interval=arguments.interval,
        status=arguments.status,
        keep=arguments.keep,
    )


def _list(arguments: argparse.Namespace) -> int:
    listed = snapshot()
    if arguments.children_of is not None:
        listed = descendants(arguments.children_of, listed)
    if arguments.interpreters_on is not None:
        listed = interpreters_on(_cpus(arguments.interpreters_on), listed)
    if arguments.json:
        emit_json(
            [
                {
                    "pid": info.pid,
                    "parent_pid": info.parent_pid,
                    "name": info.name,
                    "executable": info.executable,
                    "affinity_mask": info.affinity_mask,
                    "started_utc": info.started_utc,
                }
                for info in listed
            ]
        )
        return 0
    emit_table(
        ("pid", "parent", "name", "mask", "started"),
        [
            (
                info.pid,
                "-" if info.parent_pid is None else info.parent_pid,
                info.name,
                "-" if info.affinity_mask is None else f"{info.affinity_mask:#x}",
                info.started_utc or "-",
            )
            for info in listed
        ],
    )
    return 0


def _end(arguments: argparse.Namespace) -> int:
    failed = 0
    for pid in arguments.pids:
        error = end_process(pid)
        if error is None:
            print(f"ended {pid}")
        else:
            failed += 1
            print(f"not ended: {error}")
    return 1 if failed else 0


def _rows_check(arguments: argparse.Namespace) -> int:
    path = existing_file(arguments.file)
    count = 0
    try:
        for _ in read_rows(path):
            count += 1
    except TornRowsError as error:
        if arguments.json:
            emit_json(
                {
                    "rows": count,
                    "torn_line": error.line_number,
                    "byte_offset": error.byte_offset,
                    "reason": error.reason,
                }
            )
            return 1
        raise CommandError(
            f"{path}: line {error.line_number} at byte {error.byte_offset} {error.reason}; "
            f"{count} complete row(s) before it",
            hint=f"'{PROG} processes rows-repair {path}' moves a torn last line aside",
        ) from None
    if arguments.json:
        emit_json({"rows": count, "bytes": count_lines(path)[1]})
    else:
        print(f"{count} row(s)")
    return 0


def _rows_repair(arguments: argparse.Namespace) -> int:
    report = repair_rows(existing_file(arguments.file))
    if arguments.json:
        emit_json(
            {
                "aside": None if report.aside is None else str(report.aside),
                "removed_bytes": report.removed_bytes,
                "kept_lines": report.kept_lines,
            }
        )
    elif report.aside is None:
        print(f"nothing to repair: {report.kept_lines} complete line(s)")
    else:
        print(
            f"moved {report.removed_bytes} byte(s) to {report.aside}; "
            f"{report.kept_lines} complete line(s) kept"
        )
    return 0
