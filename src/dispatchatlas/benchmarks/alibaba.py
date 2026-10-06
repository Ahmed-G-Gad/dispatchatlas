"""Reading the batch jobs of the Alibaba 2018 cluster trace onto a continuum platform.

The Alibaba cluster trace ``cluster-trace-v2018`` (Alibaba Group 2018) records
eight days of a production cluster of about 4000 machines that runs online
services and batch jobs side by side (Guo et al. 2019). Its table
``batch_task.csv`` has one row per task of a batch job, and the task names
encode the dependencies inside each job; Tian, Zheng and Wang (2019)
characterise the resulting graphs. :func:`iter_alibaba_batch` reads the table
as a stream and yields one problem per job on a
:class:`~dispatchatlas.benchmarks.continuum.platform.Platform`:

- Every task of the job becomes a task with one mode per eligible node. The
  task's work, chosen by ``work``, makes it run ``work / speed`` seconds on
  one core of the chosen node.
- Every dependency decoded from the task names becomes a dependency. The
  table records no data volumes, so every dependency carries ``edge_data``,
  0 by default. With a positive value, the network prices the transfer
  whenever the two tasks run on different nodes.
- The tasks of a job form one group, named after the job, and are released
  at time 0. The job name is the problem's id.

**Rows.** The table is UTF-8 text without a header and without quoted
fields. A byte-order mark at its start and Windows line endings are accepted.
Each row has nine comma-separated fields, listed in :data:`COLUMNS`:
``task_name``, ``instance_num``, ``job_name``, ``task_type``, ``status``,
``start_time``, ``end_time``, ``plan_cpu`` (100 for one core), and
``plan_mem`` (memory normalised to the range 0 to 100). ``instance_num``,
``start_time``, and ``end_time`` are integers, written as at most 18 ASCII
digits after an optional minus sign; the times count seconds from the start
of the trace. ``plan_cpu`` and ``plan_mem`` are finite decimal numbers,
written in ASCII digits with an optional minus sign, fraction, and exponent.
A numeric field may be empty, which stands for a missing value: the trace
leaves ``instance_num``, ``plan_cpu``, and ``plan_mem`` empty in some rows.
Blank lines are ignored. Any other row that does not have nine fields, whose
task or job name is empty or has surrounding spaces, or whose numeric field
has another form, such as ``nan``, ``inf``, ``1_000``, a plus sign, spaces
around the number, or digits outside ASCII, raises
:class:`~dispatchatlas.core.errors.SerializationError` naming the row.

A task name and a job name consist of printable ASCII characters other than
the two quote characters, as they do in the trace. A name with a quote, a
control character, or a character outside ASCII raises
:class:`~dispatchatlas.core.errors.SerializationError` naming the row. The
reader does not interpret quoted fields, and a stray character in front of a
task name would hide its dependency code: a byte-order mark inside the table,
as where two files were joined, or one at its start that was decoded with
another encoding than UTF-8. A file that is passed open must therefore be
opened as UTF-8.

**Jobs.** The rows of one job are consecutive in the trace, and the reader
relies on it. It holds the rows of one job at a time and closes the job at
the first row that names another, so its memory does not grow with the
number of jobs. In that memory it cannot check the assumption completely. It
keeps the names of the last 1024 jobs and raises
:class:`~dispatchatlas.core.errors.SerializationError` when one of them
returns, because the job would be read in pieces. A name that returns after
more jobs than that is not noticed: each run of its rows is read as a job of
its own, and the problems of the runs share an id. In a table whose jobs are
consecutive, as they are in the trace, every job is read whole and the ids of
the problems are distinct.

**Task names.** :func:`parse_alibaba_task_name` reads a name in one of four
ways:

- A *dependency code* is one or more letters, the task's number in its job,
  and then the numbers of the tasks it depends on, each after an underscore.
  ``M1`` is task 1 and depends on nothing, ``R4_2`` is task 4 and depends on
  task 2, and ``M5_3_4`` is task 5 and depends on tasks 3 and 4. The letters
  say nothing about dependencies. A task may start once every task it depends
  on has finished.
- A *cut-off code* is a code followed by an underscore, such as ``M5_3_``,
  or a name of 50 characters or more that starts like a code, with letters
  and a digit. The trace keeps 50 characters of a name. A cut removes whole
  dependencies or the last digits of one, and a complete code of 50
  characters cannot be told from a cut one, so the dependencies of neither
  can be recovered.
- An *unknown code* is any other name that starts like a code, such as
  ``M2_1_Stg3``, whose suffix the documentation of the trace does not define.
- Any other name, such as ``task_`` followed by random characters or
  ``MergeTask``, belongs to an independent task, which depends on nothing.

**Skipped jobs.** The reader never completes a job with a value the trace
does not record. A job is skipped whole, and ``on_skip`` is called with its
name and a reason, when one of the following holds. The reason is the first
that applies, in the order of :data:`SKIP_REASONS`:

- ``"status"``: the status of a task is not one of ``statuses``.
- ``"time"``: the start or end time of a task is missing or reversed. The
  trace writes 0 where it records no time: both times of most waiting tasks
  are 0, and so is the end time of many running tasks. A 0 cannot be told
  from the first second of the trace, so a time that is empty or not positive
  is missing. An end time before the start time is reversed. Equal times are
  accepted, as a recorded span of zero.
- ``"demand"``: with ``work="core_seconds"``, the ``instance_num`` or
  ``plan_cpu`` of a task is empty or not positive.
- ``"truncated-name"``: a task name is a cut-off code.
- ``"unknown-name"``: a task name is an unknown code.
- ``"duplicate-task"``: two tasks have the same name or the same number.
- ``"missing-dependency"``: a task depends on a number that no task of the
  job has.
- ``"cycle"``: the dependencies form a cycle, as when a task depends on
  itself.
- ``"dependency-time"``: with ``work="after_dependencies"``, a task ends
  before a task it depends on ends, so the times of the two contradict the
  dependency.
- ``"too-few-tasks"`` and ``"too-many-tasks"``: the job has fewer tasks than
  ``min_tasks`` or more than ``max_tasks``.

**Work.** ``work`` chooses one of three ways, listed in :data:`WORK_RULES`,
to derive the work of a task from the times the trace records:

- ``"duration"``, the default, takes the recorded span of the task,
  ``end_time - start_time`` seconds.
- ``"core_seconds"`` multiplies the span by ``instance_num`` and by
  ``plan_cpu / 100``: the core-seconds the instances of the task request when
  each of them runs for the whole span.
- ``"after_dependencies"`` takes the part of the span that lies after the
  tasks the task depends on: ``end_time - max(start_time, latest)`` seconds,
  where ``latest`` is the latest ``end_time`` among the tasks it depends on.
  For a task that depends on nothing it is the span.

The span of a task is not its run time. The reader takes the times of each
task as they are recorded, and the trace gives the two tasks of a dependency
one start time in about four dependencies of five, although no instance of a
task starts before every instance of the tasks it depends on has completed
(Alibaba Group 2018). The span of a dependent task therefore contains the
time it waited, and ``"duration"`` and ``"core_seconds"`` overstate its work.
In about four of five jobs with dependencies of the trace, the spans along
the longest path of dependencies add up to more than the time from the job's
earliest start to its latest end, and at the median to twice that time.

``"after_dependencies"`` leaves out the wait that the recorded times show.
What remains is the longest the task can have run, an upper bound and not a
measurement, because the table does not record when a task began to run.
Under this rule the works along any path of dependencies add up to at most
the time from the job's earliest start to its latest end. The times are whole
seconds, so a task that ends in the second in which the last task it depends
on ends has a work of zero: about a fifth of the tasks of the trace under
this rule, against about a twentieth whose span is zero.

**Metadata.** Each task records its ``instance_num``, ``plan_cpu``,
``plan_mem``, ``task_type``, ``status``, ``start_time``, and ``end_time``
under those names, without the fields the trace leaves empty. Each problem
records its ``source``, the job name as ``job``, the earliest start time of
its tasks as ``arrival``, and the work rule as ``work``, so jobs can be
filtered further or laid out on the time axis of the trace.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, deque
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from dispatchatlas.benchmarks._common import (
    READ_WITHOUT_DUE_DATES,
    check_int,
    check_number,
    check_objectives,
    check_source_path,
    decimal_number,
    eligible_nodes,
)
from dispatchatlas.benchmarks.continuum.platform import Platform
from dispatchatlas.core.errors import ConfigError, ModelError, SerializationError
from dispatchatlas.core.model import Dependency, Problem, Resource, Scalar, Task

__all__ = [
    "COLUMNS",
    "SKIP_REASONS",
    "STATUSES",
    "WORK_RULES",
    "iter_alibaba_batch",
    "parse_alibaba_task_name",
]

COLUMNS: tuple[str, ...] = (
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
)
"""The fields of a row of ``batch_task.csv``, in order."""

STATUSES: tuple[str, ...] = ("Terminated", "Failed", "Running", "Waiting")
"""The task statuses of the trace."""

WORK_RULES: tuple[str, ...] = ("duration", "core_seconds", "after_dependencies")
"""The ways :func:`iter_alibaba_batch` derives the work of a task."""

SKIP_REASONS: tuple[str, ...] = (
    "status",
    "time",
    "demand",
    "truncated-name",
    "unknown-name",
    "duplicate-task",
    "missing-dependency",
    "cycle",
    "dependency-time",
    "too-few-tasks",
    "too-many-tasks",
)
"""The reasons :func:`iter_alibaba_batch` skips a job, first to apply first."""

_SOURCE = "alibaba-cluster-trace-v2018"
"""The ``source`` metadata entry of every problem."""

_BYTE_ORDER_MARK = chr(0xFEFF)
"""The character a UTF-8 byte-order mark decodes to."""

_NAME_LIMIT = 50
"""Characters the trace keeps of a task name."""

_REMEMBERED_NAMES = 4096
"""Task names whose reading is kept; the jobs of the trace repeat a small set of names."""

_REMEMBERED_JOBS = 1024
"""Finished jobs whose names are kept to notice a job whose rows are not consecutive."""

_INTEGER_DIGITS = 18
"""Most digits of an integer field; every such integer fits in 64 bits."""

_SHOWN = 40
"""Most characters of a malformed field that an error message repeats."""

_CODE = re.compile(r"[A-Za-z]+([0-9]+)((?:_[0-9]+)*)(_?)")
"""A dependency code: letters, the task number, the dependencies, and a cut's underscore."""

_CODE_START = re.compile(r"[A-Za-z]+[0-9]")
"""The start of a name that is meant as a dependency code."""

_CODED, _INDEPENDENT, _CUT, _UNKNOWN = range(4)
"""The ways a task name is read."""

_SPAN, _CORES, _AFTER = range(len(WORK_RULES))
"""Indices of :data:`WORK_RULES`."""

(
    _STATUS,
    _TIME,
    _DEMAND,
    _TRUNCATED_NAME,
    _UNKNOWN_NAME,
    _DUPLICATE,
    _MISSING,
    _CYCLE,
    _DEPENDENCY_TIME,
    _TOO_FEW,
    _TOO_MANY,
    _READABLE,
) = range(len(SKIP_REASONS) + 1)
"""Indices of :data:`SKIP_REASONS`, and the index that stands for no reason."""


class _Row(NamedTuple):
    """One row of the table, numbers converted and empty numbers as ``None``."""

    name: str
    instances: int | None
    kind: str
    status: str
    start: int | None
    end: int | None
    cpu: float | None
    memory: float | None


class _Job(NamedTuple):
    """The consecutive rows of one job and what the reader made of them.

    Attributes:
        name: The job name.
        rows: The rows, in the order of the source.
        reason: Why the job is skipped, or ``None`` when it is read.
        edges: For a job that is read, its dependencies as pairs of row
            positions, predecessor first, sorted.
        work: For a job that is read, the work of each row's task.
        arrival: For a job that is read, the earliest start time of its tasks.
    """

    name: str
    rows: list[_Row]
    reason: str | None
    edges: tuple[tuple[int, int], ...]
    work: tuple[float, ...]
    arrival: int


def parse_alibaba_task_name(name: str) -> tuple[int, tuple[int, ...]] | None:
    """Decode the dependency code of a task name of the Alibaba 2018 trace.

    Args:
        name: The ``task_name`` field of a row of ``batch_task.csv``.

    Returns:
        The task's number in its job and the numbers of the tasks it depends
        on, in the order of the name and each once, or ``None`` for the name
        of an independent task. ``"J4_2_3"`` gives ``(4, (2, 3))`` and
        ``"M1"`` gives ``(1, ())``.

    Raises:
        ConfigError: If ``name`` is not a string.
        SerializationError: If the name is a cut-off code or an unknown code,
            as the module documentation defines them.
    """
    value: object = name
    if not isinstance(value, str):
        raise ConfigError(f"a task name must be a string, got {type(value).__name__}")
    kind, number, needed = _classify(value)
    if kind == _CODED:
        return number, needed
    if kind == _INDEPENDENT:
        return None
    if kind == _CUT:
        raise SerializationError(
            f"task name {_shown(value)} is cut off: it ends in '_' or fills the "
            f"{_NAME_LIMIT} characters the trace keeps, so its dependencies cannot be recovered"
        )
    raise SerializationError(
        f"task name {_shown(value)} starts like a dependency code and is not one"
    )


def iter_alibaba_batch(
    source: str | os.PathLike[str] | Iterable[str],
    platform: Platform,
    *,
    tiers: Sequence[str] | None = None,
    objectives: str | Sequence[str] = ("makespan",),
    statuses: Sequence[str] = ("Terminated",),
    min_tasks: int = 1,
    max_tasks: int | None = None,
    work: str = "duration",
    edge_data: float = 0.0,
    on_skip: Callable[[str, str], None] | None = None,
) -> Generator[Problem, None, None]:
    """Read the batch jobs of the Alibaba 2018 trace as problems on ``platform``.

    The arguments are checked when the function is called. The source is read
    while the returned generator is consumed: the reader takes rows until one
    names another job, then yields the finished job or reports it to
    ``on_skip`` before it reads further.

    Args:
        source: The path of ``batch_task.csv``, or any iterable of its text
            lines, such as a file opened as UTF-8. A string is always a path.
            A path is opened when the first problem is asked for, and the
            file stays open until the generator is exhausted, raises, or is
            closed. An iterable is never closed by the reader.
        platform: Nodes and network the jobs run on.
        tiers: Tiers whose nodes may run tasks, as a sequence of names other
            than a string, each tier once and each with a node. Defaults to
            every tier of the platform except ``"device"``.
        objectives: One objective name or several, primary first. The
            problems have no due dates, so an objective that needs them is
            refused.
        statuses: Statuses, of :data:`STATUSES`, that the tasks of a job may
            have, as a sequence of names other than a string.
        min_tasks: Smallest number of tasks of a job that is read.
        max_tasks: Largest number of tasks of a job that is read, or ``None``
            for no limit.
        work: How the work of a task is derived, one of :data:`WORK_RULES`.
            ``"duration"`` takes the recorded span of the task and
            ``"core_seconds"`` the core-seconds its instances request over
            that span. The span of a task with dependencies includes the time
            it waited for them, so both overstate the work of such a task,
            about twofold along the longest path of a job with dependencies.
            ``"after_dependencies"`` takes only the part of the span after
            the tasks it depends on have ended, and skips a job whose times
            contradict a dependency. The module documentation defines the
            three rules.
        edge_data: Data volume of every dependency, in the data unit of the
            platform's bandwidths.
        on_skip: Called with the job name and the reason, one of
            :data:`SKIP_REASONS`, for every job that is skipped, in the order
            of the source. An exception it raises reaches the caller of the
            generator and ends the iteration: the jobs after it are not read.

    Returns:
        A generator of one problem per job that is read, in the order of the
        source. Its ``close`` method ends the iteration and closes the file of
        a source that was given as a path.

    Raises:
        ConfigError: If an argument is invalid: an unknown or repeated
            objective, status, or tier, an objective that needs due dates, a
            string given for ``tiers`` or ``statuses``, a tier without a
            node, a node with less than one core, a task limit below 1,
            ``max_tasks`` below ``min_tasks``, an unknown work rule, a
            negative ``edge_data``, a positive ``edge_data`` on a platform
            whose network lacks a link between two eligible nodes, or a
            source that is neither a path nor an iterable of lines: bytes,
            an object that is not iterable, or a path that is not text,
            holds a null character, or cannot be written in the encoding of
            file names.
        SerializationError: While iterating, if a row is malformed, a job
            name returns among the last 1024 jobs, the source is not UTF-8
            text, a job that is otherwise read has a task whose core-seconds
            are too large for a floating-point number, or a job cannot be
            built on the platform.
        OSError: While iterating, if the file cannot be read.
    """
    try:
        names = check_objectives(objectives, due_dates=False, no_due_dates=READ_WITHOUT_DUE_DATES)
        allowed = _check_statuses(statuses)
        smallest = check_int("min_tasks", min_tasks)
        largest = None if max_tasks is None else check_int("max_tasks", max_tasks, smallest)
        if not isinstance(work, str) or work not in WORK_RULES:
            raise ValueError(f"work must be one of {', '.join(WORK_RULES)}, got {work!r}")
        data = check_number("edge_data", edge_data)
        _check_callback(on_skip)
        _check_source(source)
        nodes = eligible_nodes(platform, tiers)
        if data > 0.0:
            _check_links(platform, nodes)
    except ValueError as error:
        raise ConfigError(f"invalid arguments for iter_alibaba_batch: {error}") from error
    lines = _lines(source)
    jobs = _scan(lines, allowed, WORK_RULES.index(work), smallest, largest)
    return _problems(lines, jobs, platform, nodes, names, work, data, on_skip)


# ------------------------------------------------------------------ arguments


def _check_statuses(value: object) -> frozenset[str]:
    """Return the admitted statuses as a set.

    Raises:
        ValueError: If the statuses are a string, not a sequence, empty,
            repeated, or not of :data:`STATUSES`.
    """
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence) or not value:
        hint = "; write one status as a list of one name" if isinstance(value, str) else ""
        raise ValueError(
            f"statuses must be a non-empty sequence of task statuses, got {value!r}{hint}"
        )
    for name in value:
        if not isinstance(name, str) or name not in STATUSES:
            raise ValueError(f"unknown status {name!r}; statuses: {', '.join(STATUSES)}")
    if len(set(value)) != len(value):
        raise ValueError(f"statuses repeat a name: {list(value)}")
    return frozenset(value)


def _check_callback(value: object) -> None:
    if value is not None and not callable(value):
        raise ValueError(f"on_skip must be callable or None, got {value!r}")


def _check_source(value: object) -> None:
    """Refuse a source that is neither a usable path nor an iterable of lines.

    Raises:
        ValueError: If the source is bytes, is not iterable, or is a path
            that :func:`~dispatchatlas.benchmarks._common.check_source_path`
            refuses.
    """
    if isinstance(value, str | bytes | bytearray | os.PathLike) or not isinstance(value, Iterable):
        check_source_path(value, "a path or an iterable of text lines")


def _check_links(platform: Platform, nodes: Sequence[Resource]) -> None:
    """Refuse a network without a link that a transfer between two of ``nodes`` needs.

    Raises:
        ValueError: If two of the nodes are in tiers that no link joins, or
            share a tier that has no link to itself.
    """
    counts = Counter(node.endpoint for node in nodes)
    endpoints = sorted(counts)
    for index, a in enumerate(endpoints):
        for b in endpoints[index:]:
            if (a != b or counts[a] > 1) and platform.network.link(a, b) is None:
                raise ValueError(
                    f"edge_data needs a link between {a!r} and {b!r}, "
                    "which the platform network lacks"
                )


# ----------------------------------------------------------------------- rows


def _lines(source: str | os.PathLike[str] | Iterable[str]) -> Generator[str, None, None]:
    """Yield the lines of ``source``.

    A path is opened here, and its file is closed when the generator is
    exhausted or closed. Any other source belongs to the caller: closing the
    generator leaves it as it is.
    """
    if isinstance(source, str | os.PathLike):
        with Path(source).open(encoding="utf-8", newline="") as handle:
            yield from handle
    else:
        for line in source:  # noqa: UP028 - yield from would close the caller's source
            yield line


def _integer(text: str) -> int | None:
    """Return an integer field as a number, or ``None`` when it is empty.

    Raises:
        ValueError: If the field is not at most 18 ASCII digits after an
            optional minus sign.
    """
    if text.isdigit():
        if len(text) <= _INTEGER_DIGITS and text.isascii():
            return int(text)
        raise ValueError(text)
    if not text:
        return None
    digits = text[1:]
    if text[0] == "-" and digits.isdigit() and digits.isascii() and len(digits) <= _INTEGER_DIGITS:
        return int(text)
    raise ValueError(text)


def _number(text: str) -> float | None:
    """Return a decimal field as a number, or ``None`` when it is empty.

    Raises:
        ValueError: If the field is not a finite decimal number in the form
            that :func:`~dispatchatlas.benchmarks._common.decimal_number`
            reads: ASCII digits with an optional minus sign, fraction, and
            exponent.
    """
    if not text:
        return None
    value = decimal_number(text)
    if not math.isfinite(value):
        raise ValueError(text)
    return value


def _plain(text: str) -> bool:
    """Return whether ``text`` is printable ASCII without a quote character."""
    return text.isascii() and text.isprintable() and '"' not in text and "'" not in text


def _shown(text: str) -> str:
    """Return ``text`` quoted for an error message, shortened when it is long."""
    return repr(text) if len(text) <= _SHOWN else f"{text[:_SHOWN]!r}... ({len(text)} characters)"


def _jobs(lines: Iterable[str]) -> Iterator[tuple[str, list[_Row]]]:
    """Yield each run of consecutive rows that name one job, with the job name.

    Raises:
        SerializationError: If a row is malformed, a job name returns among
            the last remembered jobs, or the lines are not UTF-8 text.
    """
    current: str | None = None
    rows: list[_Row] = []
    recent: deque[str] = deque()
    remembered: set[str] = set()
    number = 0
    width = len(COLUMNS)
    try:
        for number, line in enumerate(lines, start=1):
            try:
                if number == 1:
                    line = line.removeprefix(_BYTE_ORDER_MARK)  # noqa: PLW2901
                fields = line.rstrip("\r\n").split(",")
            except (AttributeError, TypeError):
                raise SerializationError(f"row {number} is not a line of text") from None
            if len(fields) != width:
                if len(fields) == 1 and not fields[0].strip():
                    continue
                raise SerializationError(
                    f"row {number} has {len(fields)} fields, expected the {width} "
                    f"of batch_task.csv: {', '.join(COLUMNS)}"
                )
            name, instances, job, kind, status, start, end, cpu, memory = fields
            if not name or not job or name != name.strip() or job != job.strip():
                raise SerializationError(
                    f"row {number} must name its task and its job without surrounding spaces"
                )
            if not _plain(name) or (job != current and not _plain(job)):
                column, text = ("job_name", job) if _plain(name) else ("task_name", name)
                raise SerializationError(
                    f"row {number}: {column} {_shown(text)} has a quote or a character outside "
                    "printable ASCII; the table is UTF-8 text without quoted fields"
                )
            try:
                row = _Row(
                    name,
                    _integer(instances),
                    kind,
                    status,
                    _integer(start),
                    _integer(end),
                    _number(cpu),
                    _number(memory),
                )
            except ValueError:
                raise _number_error(number, fields) from None
            if job != current:
                if current is not None:
                    yield current, rows
                    if len(recent) == _REMEMBERED_JOBS:
                        remembered.discard(recent.popleft())
                    recent.append(current)
                    remembered.add(current)
                if job in remembered:
                    raise SerializationError(
                        f"row {number}: job {_shown(job)} returns after other jobs, "
                        "and the rows of a job must be consecutive"
                    )
                current, rows = job, []
            rows.append(row)
    except UnicodeDecodeError as error:
        raise SerializationError(f"the source is not UTF-8 text after row {number}") from error
    if current is not None:
        yield current, rows


def _number_error(number: int, fields: Sequence[str]) -> SerializationError:
    """Return the error for the first numeric field of row ``number`` that is not a number."""
    kinds: dict[str, tuple[Callable[[str], int | float | None], str]] = {
        "instance_num": (_integer, f"an integer of at most {_INTEGER_DIGITS} digits"),
        "start_time": (_integer, f"an integer of at most {_INTEGER_DIGITS} digits"),
        "end_time": (_integer, f"an integer of at most {_INTEGER_DIGITS} digits"),
        "plan_cpu": (_number, "a finite decimal number"),
        "plan_mem": (_number, "a finite decimal number"),
    }
    for column, text in zip(COLUMNS, fields, strict=True):
        if column not in kinds:
            continue
        convert, expected = kinds[column]
        try:
            convert(text)
        except ValueError:
            return SerializationError(
                f"row {number}: {column} must be {expected}, got {_shown(text)}"
            )
    return SerializationError(f"row {number} has a malformed number")  # pragma: no cover


# ----------------------------------------------------------------------- jobs


def _classify(name: str) -> tuple[int, int, tuple[int, ...]]:
    """Return how a task name is read and, for a dependency code, what it encodes.

    Returns:
        The kind (``_CODED``, ``_INDEPENDENT``, ``_CUT``, or ``_UNKNOWN``),
        the task number, and the numbers the task depends on, in the order of
        the name and each once. The number is 0 and the dependencies are
        empty unless the kind is ``_CODED``.
    """
    if len(name) < _NAME_LIMIT:
        return _classify_short(name)
    return (_CUT if _CODE_START.match(name) else _INDEPENDENT), 0, ()


@lru_cache(maxsize=_REMEMBERED_NAMES)
def _classify_short(name: str) -> tuple[int, int, tuple[int, ...]]:
    """Classify a name shorter than the trace's limit, as :func:`_classify` does."""
    match = _CODE.fullmatch(name)
    if match is None:
        return (_UNKNOWN if _CODE_START.match(name) else _INDEPENDENT), 0, ()
    if match.group(3):
        return _CUT, 0, ()
    listed = match.group(2)
    if not listed:
        return _CODED, int(match.group(1)), ()
    needed = tuple(dict.fromkeys(int(part) for part in listed[1:].split("_")))
    return _CODED, int(match.group(1)), needed


def _scan(
    lines: Iterable[str],
    statuses: frozenset[str],
    rule: int,
    min_tasks: int,
    max_tasks: int | None,
) -> Iterator[_Job]:
    """Yield every job of ``lines``, read or with the reason it is skipped.

    Args:
        lines: The lines of the table.
        statuses: The statuses a task may have.
        rule: The work rule, as an index of :data:`WORK_RULES`.
        min_tasks: Smallest number of tasks of a job that is read.
        max_tasks: Largest number of tasks of a job that is read, or ``None``.

    Raises:
        SerializationError: If a row is malformed, a job name returns among
            the last remembered jobs, or a job that no reason skips has a
            task whose core-seconds are too large for a floating-point number.
    """
    for name, rows in _jobs(lines):
        found, edges, work, arrival = _inspect(rows, statuses, rule)
        if found == _READABLE:
            if len(rows) < min_tasks:
                found = _TOO_FEW
            elif max_tasks is not None and len(rows) > max_tasks:
                found = _TOO_MANY
            elif rule == _CORES and math.inf in work:
                raise SerializationError(
                    f"job {_shown(name)} has a task whose work is too large to represent"
                )
        if found == _READABLE:
            yield _Job(name, rows, None, edges, work, arrival)
        else:
            yield _Job(name, rows, SKIP_REASONS[found], (), (), 0)


def _inspect(
    rows: Sequence[_Row], statuses: frozenset[str], rule: int
) -> tuple[int, tuple[tuple[int, int], ...], tuple[float, ...], int]:
    """Check the rows of one job against every rule except the size limits.

    Returns:
        The index in :data:`SKIP_REASONS` of the first reason to skip the job,
        or ``_READABLE``; and, for a readable job, its dependencies as sorted
        pairs of row positions, the work of each row's task under ``rule``,
        and the earliest start time. Core-seconds that are too large for a
        floating-point number are infinite.
    """
    found = _READABLE
    arrival = 0
    work: list[float] = []
    times: list[tuple[int, int]] = []
    positions: dict[int, int] = {}
    independent: set[str] = set()
    wanted: list[tuple[int, tuple[int, ...]]] = []
    for position, (name, instances, _, status, start, end, cpu, _) in enumerate(rows):
        if status not in statuses:
            return _STATUS, (), (), 0
        if start is None or end is None or start <= 0 or end < start:
            found = _TIME
            continue
        if arrival == 0 or start < arrival:
            arrival = start
        if rule == _SPAN:
            work.append(float(end - start))
        elif rule == _AFTER:
            times.append((start, end))
        elif instances is None or cpu is None or instances <= 0 or cpu <= 0.0:
            found = min(found, _DEMAND)
        else:
            amount = (end - start) * instances * cpu / 100.0
            if amount == math.inf:
                # The product is too large although the core-seconds may not be.
                amount = (end - start) * instances * (cpu / 100.0)
            work.append(amount)
        if found <= _DEMAND:
            continue
        kind, number, needed = _classify(name)
        if kind == _CODED:
            if number in positions:
                found = min(found, _DUPLICATE)
            positions[number] = position
            if needed:
                wanted.append((position, needed))
        elif kind == _INDEPENDENT:
            if name in independent:
                found = min(found, _DUPLICATE)
            independent.add(name)
        else:
            found = min(found, _TRUNCATED_NAME if kind == _CUT else _UNKNOWN_NAME)
    if found != _READABLE:
        return found, (), (), 0
    pairs: set[tuple[int, int]] = set()
    for position, needed in wanted:
        for number in needed:
            before = positions.get(number)
            if before is None:
                return _MISSING, (), (), 0
            pairs.add((before, position))
    edges = tuple(sorted(pairs))
    if _has_cycle(len(rows), edges):
        return _CYCLE, (), (), 0
    if rule == _AFTER:
        # A task may run from its start time or the end of the last task it depends on.
        free = [start for start, _ in times]
        for before, after in edges:
            free[after] = max(free[after], times[before][1])
        for (_, end), since in zip(times, free, strict=True):
            if end < since:
                return _DEPENDENCY_TIME, (), (), 0
            work.append(float(end - since))
    return _READABLE, edges, tuple(work), arrival


def _has_cycle(count: int, edges: Sequence[tuple[int, int]]) -> bool:
    """Return whether ``edges`` over ``count`` nodes contain a cycle, by Kahn's algorithm."""
    if not edges:
        return False
    waiting = [0] * count
    following: list[list[int]] = [[] for _ in range(count)]
    for before, after in edges:
        waiting[after] += 1
        following[before].append(after)
    ready = [node for node in range(count) if not waiting[node]]
    ordered = 0
    while ready:
        ordered += 1
        for after in following[ready.pop()]:
            waiting[after] -= 1
            if not waiting[after]:
                ready.append(after)
    return ordered != count


# ------------------------------------------------------------------- problems


def _problems(
    lines: Generator[str, None, None],
    jobs: Iterable[_Job],
    platform: Platform,
    nodes: Sequence[Resource],
    objectives: tuple[str, ...],
    work: str,
    edge_data: float,
    on_skip: Callable[[str, str], None] | None,
) -> Generator[Problem, None, None]:
    """Yield the problem of every job that is read and report the others.

    ``lines`` are the lines that ``jobs`` are read from. They are closed when
    this generator is exhausted, raises, or is closed, which closes the file
    of a source that is a path.
    """
    try:
        for job in jobs:
            if job.reason is None:
                yield _problem(job, platform, nodes, objectives, work, edge_data)
            elif on_skip is not None:
                on_skip(job.name, job.reason)
    finally:
        lines.close()


def _problem(
    job: _Job,
    platform: Platform,
    nodes: Sequence[Resource],
    objectives: tuple[str, ...],
    work: str,
    edge_data: float,
) -> Problem:
    """Build the problem of one job that is read.

    Raises:
        SerializationError: If the model refuses the problem, as it does when
            the duration of a task on a node is not finite.
    """
    rows = job.rows
    try:
        tasks = tuple(
            Task(row.name, platform.modes(amount, nodes), group=job.name, metadata=_recorded(row))
            for row, amount in zip(rows, job.work, strict=True)
        )
        dependencies = tuple(
            Dependency(rows[before].name, rows[after].name, data=edge_data)
            for before, after in job.edges
        )
        return Problem(
            id=job.name,
            tasks=tasks,
            resources=platform.resources,
            dependencies=dependencies,
            objectives=objectives,
            network=platform.network,
            metadata={"source": _SOURCE, "job": job.name, "arrival": job.arrival, "work": work},
        )
    except (ModelError, OverflowError) as error:
        raise SerializationError(
            f"job {_shown(job.name)} cannot be built on the platform: {error}"
        ) from error


def _recorded(row: _Row) -> dict[str, Scalar]:
    """Return the metadata of a row's task: its fields, without the empty numbers."""
    fields: dict[str, Scalar | None] = {
        "instance_num": row.instances,
        "plan_cpu": row.cpu,
        "plan_mem": row.memory,
        "task_type": row.kind,
        "status": row.status,
        "start_time": row.start,
        "end_time": row.end,
    }
    return {key: value for key, value in fields.items() if value is not None}
