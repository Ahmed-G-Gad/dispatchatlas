"""Machinery shared by the command modules: parsers, input files, output, and errors.

Every command writes its result to standard output, as text or, with
``--json``, as one JSON document, and writes log messages and errors to
standard error. Errors are one line with a recovery hint; ``--debug`` adds
the traceback.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import sys
import textwrap
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from dispatchatlas.benchmarks.formats import EXTENSIONS, FORMATS, load_instance
from dispatchatlas.core.errors import (
    ConfigError,
    DispatchAtlasError,
    MissingDependencyError,
    ModelError,
    SerializationError,
    SolverError,
    UnsupportedProblemError,
)
from dispatchatlas.core.logs import get_logger
from dispatchatlas.core.model import Problem
from dispatchatlas.core.serialization import (
    dumps_problem,
    load_problem,
    loads_problem,
    save_problem,
)
from dispatchatlas.core.validation import Violation

__all__ = [
    "PROG",
    "CommandError",
    "Commands",
    "Handler",
    "UsageError",
    "add_command",
    "add_group",
    "add_problem_arguments",
    "assignment",
    "emit_fields",
    "emit_json",
    "emit_table",
    "existing_file",
    "finite_number",
    "global_options",
    "jsonable",
    "logging_to_stderr",
    "non_negative_integer",
    "parameter_lines",
    "parameters",
    "parse_value",
    "positive_integer",
    "positive_number",
    "read_problem",
    "report_error",
    "seconds",
    "text",
    "violation_to_dict",
    "workers",
    "write_problem",
]

PROG = "dispatchatlas"
"""Name of the command, used in messages."""

type Handler = Callable[[argparse.Namespace], int]
"""Runs one command from its parsed arguments and returns the exit code."""

type Commands = argparse._SubParsersAction[argparse.ArgumentParser]
"""The object that command modules add their parsers to."""


class CommandError(DispatchAtlasError):
    """A command cannot do what was asked.

    Attributes:
        hint: How to recover, shown under the error.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class UsageError(Exception):
    """Arguments that parse one by one but do not fit together."""


class HelpFormatter(argparse.HelpFormatter):
    """Wraps each line of a description or epilog, but keeps indented lines, the examples."""

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        lines: list[str] = []
        for line in text.splitlines():
            if line.startswith(" ") or not line.strip():
                lines.append(indent + line)
            else:
                lines.extend(
                    textwrap.wrap(line, width, initial_indent=indent, subsequent_indent=indent)
                )
        return "\n".join(lines)


# ------------------------------------------------------------------ parsers


def global_options(*, suppress: bool) -> argparse.ArgumentParser:
    """Return a parent parser with the options every command accepts.

    The top-level parser takes ``--debug`` and ``--quiet`` with real
    defaults. Each command takes them again with suppressed defaults, so an
    option given after the command name counts and one given before it is not
    reset, and takes ``--json`` too.

    Args:
        suppress: Leave the options unset unless given, and add ``--json``,
            for command parsers.
    """
    parent = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    group = parent.add_argument_group("common options")
    if suppress:
        group.add_argument(
            "--json", action="store_true", help="print the result as JSON instead of text"
        )
    unset: Any = argparse.SUPPRESS if suppress else False
    group.add_argument(
        "--debug",
        action="store_true",
        default=unset,
        help="show tracebacks of errors and debug messages",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=unset,
        help="log only warnings and errors",
    )
    return parent


def add_group(commands: Commands, name: str, summary: str, description: str) -> Commands:
    """Add a command group, such as ``solvers``, and return its subcommand registry.

    Args:
        commands: The registry to add the group to.
        name: Name of the group.
        summary: One line shown in the parent's command list.
        description: Text shown by the group's ``--help``.
    """
    parser = commands.add_parser(
        name,
        help=summary,
        description=description,
        formatter_class=HelpFormatter,
        allow_abbrev=False,
    )
    return parser.add_subparsers(title="commands", metavar="COMMAND", required=True)


def add_command(
    commands: Commands,
    name: str,
    handler: Handler,
    summary: str,
    *,
    description: str | None = None,
    epilog: str | None = None,
    interrupted: str | None = None,
) -> argparse.ArgumentParser:
    """Add one command and return its parser, for the command's own arguments.

    The command gets the common options (``--json``, ``--debug``,
    ``--quiet``), and its handler, name, and parser are stored as defaults.

    Args:
        commands: The registry to add the command to.
        name: Name of the command.
        handler: Runs the command.
        summary: One line shown in the parent's command list.
        description: Text shown by ``--help``. Defaults to ``summary``.
        epilog: Text shown after the options, such as examples.
        interrupted: Hint shown when the user interrupts the command.
    """
    parser = commands.add_parser(
        name,
        help=summary,
        description=description if description is not None else summary,
        epilog=epilog,
        parents=[global_options(suppress=True)],
        formatter_class=HelpFormatter,
        allow_abbrev=False,
    )
    parser.set_defaults(
        handler=handler, command=parser.prog, command_parser=parser, interrupted=interrupted
    )
    return parser


def add_problem_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``PROBLEM`` argument and the ``--format`` and ``--instance`` options.

    :func:`read_problem` reads the problem they name.
    """
    parser.add_argument(
        "problem",
        metavar="PROBLEM",
        help="problem document (.json), instance file in a published format, or - for a "
        "problem document on standard input",
    )
    parser.add_argument(
        "--format",
        dest="format_name",
        choices=sorted(FORMATS),
        help="read PROBLEM in this instance format instead of the one its extension implies",
    )
    parser.add_argument(
        "--instance",
        metavar="NAME",
        help="instance to read from a multi-instance OR-Library file",
    )


# ------------------------------------------------------------ argument types


def parse_value(raw: str) -> Any:
    """Return a command-line value as JSON when it parses as JSON, else as the string.

    JSON arrays become tuples, as in campaign files, so the values can be
    passed to solvers and generators alike. ``NaN`` and ``Infinity`` stay
    strings.

    Example:
        ``10`` gives ``10``, ``[1, 2]`` gives ``(1, 2)``, ``true`` gives
        ``True``, ``random-key`` gives ``"random-key"``, and ``'"7"'`` gives
        ``"7"``.
    """
    try:
        value = json.loads(raw, parse_constant=_no_constant)
    except ValueError:
        return raw
    return _tuples(value)


def assignment(argument: str) -> tuple[str, Any]:
    """Parse a ``KEY=VALUE`` argument, reading the value with :func:`parse_value`.

    Raises:
        argparse.ArgumentTypeError: If the argument has no ``=`` or no key.
    """
    key, separator, raw = argument.partition("=")
    key = key.strip()
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {argument!r}")
    return key, parse_value(raw)


def parameters(pairs: Iterable[tuple[str, Any]] | None) -> dict[str, Any]:
    """Collect ``--set`` assignments into keyword arguments.

    Raises:
        UsageError: If a key is set twice.
    """
    values: dict[str, Any] = {}
    for key, value in pairs or ():
        if key in values:
            raise UsageError(f"--set gives {key!r} twice")
        values[key] = value
    return values


def non_negative_integer(argument: str) -> int:
    """Parse an integer of at least 0, such as a seed."""
    value = _integer(argument)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be an integer of at least 0, got {argument!r}")
    return value


def positive_integer(argument: str) -> int:
    """Parse an integer of at least 1, such as an evaluation limit."""
    value = _integer(argument)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be an integer of at least 1, got {argument!r}")
    return value


def finite_number(argument: str) -> float:
    """Parse a finite number."""
    try:
        value = float(argument)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number, got {argument!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be finite, got {argument!r}")
    return value


def positive_number(argument: str) -> float:
    """Parse a finite number above 0, such as a time limit in seconds."""
    value = finite_number(argument)
    if value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be above 0, got {argument!r}")
    return value


def workers(argument: str) -> int | str:
    """Parse a worker count: an integer of at least 1, or ``auto``."""
    if argument == "auto":
        return argument
    try:
        return positive_integer(argument)
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError(
            f"must be an integer of at least 1 or 'auto', got {argument!r}"
        ) from None


def _integer(argument: str) -> int:
    try:
        return int(argument)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {argument!r}") from None


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a finite number")


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    return value


# ------------------------------------------------------------------- inputs


def existing_file(argument: str | Path) -> Path:
    """Return the path of a file that exists.

    Raises:
        CommandError: If nothing exists at the path, or it is a directory.
    """
    path = Path(argument)
    if not path.exists():
        raise CommandError(f"{path} does not exist", hint="check the path")
    if not path.is_file():
        raise CommandError(f"{path} is not a file", hint="pass a file, not a directory")
    return path


def read_problem(
    source: str, format_name: str | None = None, instance: str | None = None
) -> Problem:
    """Read a problem from a problem document or an instance file.

    Args:
        source: A problem document (``.json``), an instance file in a
            published format, or ``-`` for a problem document on standard
            input.
        format_name: Instance format that overrides the file extension, a
            key of :data:`~dispatchatlas.benchmarks.formats.FORMATS`.
        instance: Instance to take from a multi-instance OR-Library file.

    Raises:
        UsageError: If ``format_name`` or ``instance`` is given for a problem
            document.
        CommandError: If the file is missing or its format cannot be told.
    """
    if source == "-" or (format_name is None and Path(source).suffix.lower() == ".json"):
        if instance is not None:
            raise UsageError(
                "--instance picks an instance of an OR-Library file, not of a document"
            )
        if source == "-":
            if format_name is not None:
                raise UsageError("standard input must hold a problem document; drop --format")
            return loads_problem(sys.stdin.read())
        return load_problem(existing_file(source))
    path = existing_file(source)
    if format_name is None and path.suffix.lower() not in EXTENSIONS:
        raise CommandError(
            f"cannot tell the format of {path.name} from its extension",
            hint=f"pass --format, one of {', '.join(sorted(FORMATS))}; problem documents "
            "end in .json",
        )
    options = {} if instance is None else {"instance": instance}
    return load_instance(path, format=format_name, **options)


# ------------------------------------------------------------------ outputs


def jsonable(value: Any) -> Any:
    """Return ``value`` as JSON types.

    Mappings become objects, sequences and sets become arrays (sets sorted),
    enumerations their values, paths strings, and floats that are not finite
    the strings ``"inf"``, ``"-inf"``, or ``"nan"``, which JSON cannot hold
    as numbers.
    """
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def emit_json(value: Any) -> None:
    """Print ``value`` as an indented JSON document."""
    print(json.dumps(jsonable(value), indent=2, ensure_ascii=True, allow_nan=False))


def violation_to_dict(violation: Violation) -> dict[str, Any]:
    """Return a violation that :func:`~dispatchatlas.core.validation.check_schedule` found."""
    return {
        "code": violation.code,
        "message": violation.message,
        "task": violation.task,
        "resource": violation.resource,
        "amount": violation.amount,
    }


def text(value: object) -> str:
    """Render a value for text output.

    ``None`` is a dash, booleans are ``yes`` and ``no``, integral floats
    lose their decimal point, other floats keep ten significant digits, and
    sequences are joined with commas.
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return format(value, ".10g")
    if isinstance(value, (list, tuple)):
        return ", ".join(text(item) for item in value)
    return str(value)


def parameter_lines(values: Mapping[str, Any]) -> str:
    """Render parameters as ``name = value`` lines, each value in the JSON ``--set`` reads."""
    return "\n".join(f"{name} = {json.dumps(jsonable(value))}" for name, value in values.items())


def seconds(value: float) -> str:
    """Render a duration in seconds, minutes, or hours."""
    if value < 60.0:
        return f"{value:.3g} s"
    if value < 3600.0:
        return f"{value / 60.0:.3g} min"
    return f"{value / 3600.0:.3g} h"


def emit_fields(rows: Iterable[tuple[str, object]]) -> None:
    """Print labelled values in two aligned columns; multi-line values stay aligned."""
    items = [(label, text(value)) for label, value in rows]
    width = max((len(label) for label, _ in items), default=0)
    for label, value in items:
        lines = value.splitlines() or [""]
        print(f"{label:<{width}}  {lines[0]}".rstrip())
        for line in lines[1:]:
            print(f"{'':<{width}}  {line}".rstrip())


def emit_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Print rows as columns aligned under the headers."""
    cells = [[text(value) for value in row] for row in rows]
    widths = [
        max([len(header), *(len(row[index]) for row in cells)])
        for index, header in enumerate(headers)
    ]
    for row in [list(headers), *cells]:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        )


def write_problem(problem: Problem, output: str | None, *, as_json: bool) -> None:
    """Print a problem document, or write it to ``output`` and report where it went."""
    if output is None:
        sys.stdout.write(dumps_problem(problem))
        return
    path = save_problem(problem, output)
    summary = problem.summary()
    if as_json:
        emit_json({"path": path, "problem": summary})
        return
    print(
        f"wrote problem {problem.id} ({summary['tasks']} tasks, {summary['resources']} "
        f"resources) to {path}"
    )


# --------------------------------------------------------------------- logs


class _LogFormatter(logging.Formatter):
    """Formats a record as ``dispatchatlas: <level>: <message>``."""

    def __init__(self, *, detailed: bool) -> None:
        super().__init__()
        self._detailed = detailed

    def format(self, record: logging.LogRecord) -> str:
        where = f" [{record.name}]" if self._detailed else ""
        line = f"{PROG}: {record.levelname.lower()}{where}: {record.getMessage()}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


@contextlib.contextmanager
def logging_to_stderr(level: int, *, detailed: bool = False) -> Iterator[None]:
    """Show the library's log records on standard error while the block runs.

    The handler is removed and the logger's level restored afterwards, so
    calling the command line from Python leaves logging as it was.

    Args:
        level: Lowest level shown.
        detailed: Name the logger of every record.
    """
    logger = get_logger(PROG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_LogFormatter(detailed=detailed))
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


# ------------------------------------------------------------------- errors


def report_error(error: BaseException, command: str) -> None:
    """Print an error and a recovery hint to standard error.

    Args:
        error: The error.
        command: The command that failed, such as ``dispatchatlas solve``,
            for hints that point to its help.
    """
    lines = _message(error).splitlines() or [type(error).__name__]
    print(f"{PROG}: error: {lines[0]}", file=sys.stderr)
    for line in lines[1:]:
        print(line, file=sys.stderr)
    advice = _hint(error, command)
    if advice:
        print(f"{PROG}: hint: {advice}", file=sys.stderr)


def _message(error: BaseException) -> str:
    if isinstance(error, OSError) and error.filename is not None and error.strerror:
        return f"{error.filename}: {error.strerror}"
    return str(error)


def _hint(error: BaseException, command: str) -> str | None:
    if isinstance(error, CommandError):
        return error.hint
    if isinstance(error, MissingDependencyError):
        return None
    if isinstance(error, UnsupportedProblemError):
        return f"'{PROG} solvers show ID' lists the problem features a solver models"
    if isinstance(error, SolverError):
        return f"check the solver's parameters and limits; '{PROG} solvers show ID' describes them"
    if isinstance(error, ModelError):
        return "correct the problem; every defect found is listed above"
    if isinstance(error, SerializationError):
        return "check that the file is a DispatchAtlas document of the kind the command expects"
    if isinstance(error, ConfigError):
        return f"see '{command} --help'"
    if isinstance(error, FileNotFoundError):
        return "check the path"
    if isinstance(error, PermissionError):
        return "check the permissions of the path"
    if isinstance(error, OSError):
        return "check the path and the free disk space"
    return "this is unexpected: rerun with --debug to see the traceback, and report it"
