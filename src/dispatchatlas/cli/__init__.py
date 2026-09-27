"""The ``dispatchatlas`` command line.

Every command is a thin layer over the public library API: it parses its
arguments, calls the library, and prints the result. The commands::

    dispatchatlas [--version] [--debug] [--quiet] COMMAND ...
        version                              package, Python, and optional backends
        solvers list [--family F]            registered solvers
        solvers show ID                      metadata, parameters, features, citations
        benchmarks list | show ID            benchmark families
        benchmarks generate ID [--seed N] [--set KEY=VALUE ...] [--output FILE]
        suites list | show ID                published instance suites
        suites load ID INSTANCE [--directory D] [--output FILE]
        catalogs list | show NAME            named instance sets
        catalogs build NAME --output DIR [--seed N] [--overwrite]
        solve PROBLEM --solver ID [--set KEY=VALUE ...] [limits] [--seed N] [--output FILE]
        check PROBLEM [SCHEDULE]             validate a problem, and a schedule against it
        campaign validate | estimate | run | replay
        analyze CAMPAIGN_DIR --output DIR [analysis options]
        references [KEY ...] [--format text|bibtex]

Results go to standard output, as text or, with ``--json``, as JSON. Log
messages and errors go to standard error. The exit code is 0 on success, 1
when the requested operation fails, 2 on a usage error, and 130 when the
command is interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import traceback
from collections.abc import Sequence

from dispatchatlas import __version__
from dispatchatlas.cli import (
    analyze,
    benchmarks,
    campaign,
    catalogs,
    check,
    references,
    solve,
    solvers,
    suites,
    version,
)
from dispatchatlas.cli._common import (
    PROG,
    Handler,
    HelpFormatter,
    UsageError,
    global_options,
    logging_to_stderr,
    report_error,
)

__all__ = ["build_parser", "main"]

_COMMANDS = (
    version,
    solvers,
    benchmarks,
    suites,
    catalogs,
    solve,
    check,
    campaign,
    analyze,
    references,
)
"""The command modules, in the order ``--help`` lists them."""

_INTERRUPTED = 130
"""Exit code of an interrupted command: 128 plus the number of SIGINT."""


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the whole command line."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Reproducible benchmarking of scheduling algorithms.",
        epilog=f"Run '{PROG} COMMAND --help' for the options of a command.",
        parents=[global_options(suppress=False)],
        formatter_class=HelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    commands = parser.add_subparsers(title="commands", metavar="COMMAND", required=True)
    for module in _COMMANDS:
        module.register(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit code.

    Args:
        argv: The arguments after the program name. Defaults to
            ``sys.argv[1:]``.

    Returns:
        0 on success, 1 when the requested operation fails, 2 on a usage
        error, and 130 when the command is interrupted.
    """
    parser = build_parser()
    try:
        arguments = parser.parse_args(None if argv is None else list(argv))
    except SystemExit as stop:
        return _exit_code(stop.code)
    debug = bool(arguments.debug)
    level = logging.DEBUG if debug else logging.WARNING if arguments.quiet else logging.INFO
    with logging_to_stderr(level, detailed=debug):
        return _run(arguments, debug=debug)


def _run(arguments: argparse.Namespace, *, debug: bool) -> int:
    """Run the command's handler and turn every failure into a message and an exit code."""
    handler: Handler = arguments.handler
    try:
        return handler(arguments)
    except UsageError as error:
        return _usage_error(arguments.command_parser, str(error))
    except KeyboardInterrupt:
        print(f"{PROG}: interrupted", file=sys.stderr)
        if arguments.interrupted:
            print(f"{PROG}: hint: {arguments.interrupted}", file=sys.stderr)
        return _INTERRUPTED
    except BrokenPipeError:
        if debug:
            traceback.print_exc()
        _silence_stdout()
        return 1
    except Exception as error:  # every failure ends as a message and exit code 1
        if debug:
            traceback.print_exc()
        report_error(error, str(arguments.command))
        return 1


def _usage_error(parser: argparse.ArgumentParser, message: str) -> int:
    """Print a command's usage and ``message``, as the parser does, and return its exit code."""
    try:
        parser.error(message)
    except SystemExit as stop:
        return _exit_code(stop.code)


def _exit_code(code: str | int | None) -> int:
    """Return the exit code of a ``SystemExit`` raised by the parser."""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _silence_stdout() -> None:
    """Point standard output at the null device after its reader went away.

    Without this, the interpreter reports the broken pipe again when it
    flushes standard output at exit.
    """
    with contextlib.suppress(OSError, ValueError, AttributeError):
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, sys.stdout.fileno())
