"""Reading WfCommons workflow instances onto a continuum platform.

WfCommons (Coleman et al. 2022) publishes executions of real scientific
workflows in the WfFormat JSON format. :func:`read_wfcommons` maps one
instance onto a :class:`~dispatchatlas.benchmarks.continuum.platform.Platform`:

- Every workflow task becomes a task with one mode per eligible node. Its
  recorded runtime is taken as its work, so it runs ``runtime / speed``
  seconds on one core of the chosen node, whatever number of cores the
  execution records for it.
- Every parent-child pair becomes a dependency that carries the bytes of the
  files the parent writes and the child reads, divided by ``bytes_per_unit``
  to express them in the platform's data unit (megabytes by default). The
  network then prices the transfer whenever the two tasks run on different
  nodes. A file that several children read is carried by the dependency of
  each of them, a file that a task lists more than once counts once, and a
  pair that shares no file is a dependency without data.
- All tasks form one group, the workflow, and are released at time 0.

A file is represented only where it passes from a task to one of the declared
children of that task. Three kinds of file lie on no dependency and are left
out, whatever their size: files that no task of the workflow writes (its
inputs), files that no task reads (its results), and files that one task
writes and another reads when the two are not declared parent and child. The
machines, core counts, memory, and start times that an execution records are
not read.

The reader consumes a few fields and checks each of them; anything missing
or malformed raises :class:`~dispatchatlas.core.errors.SerializationError`
naming the field:

- Runtimes are never invented: a task without a recorded runtime is refused,
  and so is an instance of schema ``1.5`` or ``1.6`` that has no ``execution``
  section. Zero runtimes are accepted, as recorded measurements.
- A file size is a non-negative whole number of bytes, written as a JSON
  integer or as a real number without a fractional part, such as ``2048.0``
  or ``1e6``.
- Task identifiers are kept as written. One that starts or ends with white
  space is refused, since the data model does not hold such an id.
- A document that contradicts itself is refused: a task or an execution entry
  listed twice, an execution entry for a task that the specification does not
  list, a parent or child that is not a task, a file listed with two sizes, a
  task that depends on itself, or dependencies that form a cycle.
- A runtime, a duration, or the bytes of a dependency that a floating-point
  number cannot hold are refused.
- A file is UTF-8 text, with or without a byte-order mark. A file in which a
  JSON object repeats a key is refused.

Supported ``schemaVersion`` values and the fields read:

- ``1.4``: ``workflow.tasks[]`` with ``name`` (the task's identity),
  ``parents`` and ``children`` (task names), ``runtimeInSeconds``, and
  ``files[]`` entries with ``name``, ``sizeInBytes``, and ``link``
  (``"input"`` or ``"output"``).
- ``1.5`` and ``1.6``: ``workflow.specification.tasks[]`` with ``id``,
  ``parents``, ``children``, ``inputFiles``, and ``outputFiles``;
  ``workflow.specification.files[]`` with ``id`` and ``sizeInBytes``; and
  ``workflow.execution.tasks[]`` with ``id`` and ``runtimeInSeconds``.

The lists of parents, children, and files are optional in every version: a
task without one has none. Dependencies are the union of the ``parents`` and
``children`` lists, each pair counted once, so a pair that only one of its two
tasks declares is a dependency.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dispatchatlas.benchmarks._common import check_number, check_objectives, eligible_nodes
from dispatchatlas.benchmarks.continuum.platform import Platform
from dispatchatlas.core.errors import ConfigError, ModelError, SerializationError
from dispatchatlas.core.model import Dependency, Problem, Resource, Task

__all__ = ["SCHEMA_VERSIONS", "read_wfcommons"]

SCHEMA_VERSIONS: tuple[str, ...] = ("1.4", "1.5", "1.6")
"""WfFormat schema versions :func:`read_wfcommons` reads."""


@dataclass(frozen=True, slots=True)
class _Row:
    """The fields of one workflow task that the reader uses."""

    key: str
    parents: tuple[str, ...]
    children: tuple[str, ...]
    inputs: frozenset[str]
    outputs: frozenset[str]
    runtime: float


def read_wfcommons(
    source: Mapping[str, Any] | str | os.PathLike[str],
    platform: Platform,
    *,
    tiers: Sequence[str] | None = None,
    problem_id: str | None = None,
    objectives: str | Sequence[str] = ("makespan",),
    bytes_per_unit: float = 1e6,
) -> Problem:
    """Read a WfCommons WfFormat instance as a problem on ``platform``.

    Args:
        source: The parsed JSON document, as a mapping, or the path of a JSON
            file. A string is always a path, never JSON text. The numbers of
            a mapping may be of any type registered with :mod:`numbers`, such
            as NumPy scalars.
        platform: Nodes and network the workflow runs on.
        tiers: Tiers whose nodes may run tasks, as a sequence of names other
            than a string, each tier once and each with a node. Defaults to
            every tier of the platform except ``"device"``.
        problem_id: Id of the problem, a non-empty string without surrounding
            spaces. Defaults to the file name without its extension when
            ``source`` is a path, and to the instance ``name`` when it is a
            mapping. In either default, characters outside letters, digits,
            ``.``, ``_``, and ``-`` are replaced by ``-`` and leading
            punctuation is dropped, so that the id starts with a letter or
            digit. A file name that leaves nothing gives way to the ``name``,
            and a ``name`` that is absent or leaves nothing to
            ``"wfcommons"``. The ``name`` is kept in the metadata of the
            problem, under ``"name"``.
        objectives: One objective name or several, primary first. The problem
            has no due dates, so an objective that needs them is refused.
        bytes_per_unit: Bytes per data unit of the platform's bandwidths.

    Returns:
        The problem.

    Raises:
        SerializationError: If the document is not a supported WfFormat
            instance or its content is inconsistent.
        ConfigError: If an objective is unknown, repeated, or needs due
            dates, ``tiers`` is a string, repeats a tier, or names a tier
            without a node, a node has less than one core,
            ``bytes_per_unit`` is not a positive number, ``problem_id`` is
            not a non-empty string without surrounding spaces, ``platform``
            is not a platform, or ``source`` is neither a mapping nor a path.
        OSError: If the file cannot be read.
    """
    try:
        names = check_objectives(objectives, due_dates=False)
        unit = check_number("bytes_per_unit", bytes_per_unit, positive=True)
        _check_arguments(platform, tiers, problem_id)
    except ValueError as error:
        raise ConfigError(f"invalid arguments for read_wfcommons: {error}") from error
    nodes = _nodes(platform, tiers)
    document, stem = _load(source)
    version = document.get("schemaVersion")
    if version not in SCHEMA_VERSIONS:
        raise SerializationError(
            f"WfFormat schemaVersion must be one of {', '.join(SCHEMA_VERSIONS)}, found {version!r}"
        )
    workflow = _mapping(document.get("workflow"), "workflow")
    rows, sizes = _flat_rows(workflow) if version == "1.4" else _split_rows(workflow)
    edges = _edges(rows)
    name = document.get("name")
    label = name if isinstance(name, str) else ""
    identifier = problem_id if problem_id is not None else stem or _slug(label) or "wfcommons"
    by_key = {row.key: row for row in rows}
    tasks = tuple(_task(row, platform, nodes, identifier) for row in rows)
    dependencies = tuple(
        _dependency(by_key[before], by_key[after], sizes, unit) for before, after in edges
    )
    metadata: dict[str, str] = {"source": "wfcommons", "schema_version": str(version)}
    if label:
        metadata["name"] = label
    try:
        return Problem(
            id=identifier,
            tasks=tasks,
            resources=platform.resources,
            dependencies=dependencies,
            objectives=names,
            network=platform.network,
            metadata=metadata,
        )
    except ModelError as error:
        raise SerializationError(f"WfFormat instance {identifier!r} is invalid: {error}") from error


def _check_arguments(platform: object, tiers: object, problem_id: object) -> None:
    """Check the platform, the tiers, and the problem id, refusing them with ``ValueError``."""
    if not isinstance(platform, Platform):
        raise ValueError(f"platform must be a Platform, got {type(platform).__name__}")
    if tiers is not None:
        if isinstance(tiers, str) or not isinstance(tiers, Sequence):
            raise ValueError(f"tiers must be a sequence of tier names, got {tiers!r}")
        if not all(isinstance(tier, str) for tier in tiers):
            raise ValueError(f"tiers must be a sequence of tier names, got {list(tiers)}")
        if len(set(tiers)) != len(tiers):
            raise ValueError(f"tiers repeat a name: {list(tiers)}")
    if problem_id is not None and (
        not isinstance(problem_id, str) or not problem_id or problem_id != problem_id.strip()
    ):
        raise ValueError(
            f"problem_id must be a non-empty string without surrounding spaces, got {problem_id!r}"
        )


def _load(source: object) -> tuple[Mapping[str, Any], str]:
    """Return the document and, for a file, the id that its name gives."""
    if isinstance(source, Mapping):
        return source, ""
    if not isinstance(source, (str, os.PathLike)):
        raise ConfigError(
            "invalid arguments for read_wfcommons: source must be a mapping or a path, "
            f"got {type(source).__name__}"
        )
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise SerializationError(
            f"invalid JSON in {path}: the file is not UTF-8 text ({error})"
        ) from error
    try:
        document = json.loads(text, object_pairs_hook=_unique_keys)
    except RecursionError as error:
        raise SerializationError(
            f"invalid JSON in {path}: the document is nested too deeply"
        ) from error
    except ValueError as error:
        raise SerializationError(f"invalid JSON in {path}: {error}") from error
    return _mapping(document, "document"), _slug(path.stem)


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object from its members, refusing a repeated key with ``ValueError``."""
    members = dict(pairs)
    if len(members) != len(pairs):
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise ValueError(f"an object repeats the key {key!r}")
            seen.add(key)
    return members


def _split_rows(workflow: Mapping[str, Any]) -> tuple[list[_Row], dict[str, int]]:
    """Read the tasks and file sizes of the schema 1.5 and 1.6 layout."""
    specification = _mapping(workflow.get("specification"), "workflow.specification")
    execution = _mapping(workflow.get("execution"), "workflow.execution")
    sizes: dict[str, int] = {}
    for number, raw in enumerate(_sequence(specification.get("files", []), "files")):
        entry = _mapping(raw, f"file {number}")
        file_id = _text(entry.get("id"), f"file {number} id")
        _record_size(sizes, file_id, entry.get("sizeInBytes"))
    runtimes: dict[str, float] = {}
    for number, raw in enumerate(_sequence(execution.get("tasks"), "workflow.execution.tasks")):
        entry = _mapping(raw, f"execution task {number}")
        task_id = _identifier(entry.get("id"), f"execution task {number} id")
        if task_id in runtimes:
            raise SerializationError(f"execution task {task_id!r} is listed twice")
        runtimes[task_id] = _runtime(entry.get("runtimeInSeconds"), task_id)
    rows: list[_Row] = []
    specified: set[str] = set()
    for number, raw in enumerate(
        _sequence(specification.get("tasks"), "workflow.specification.tasks")
    ):
        entry = _mapping(raw, f"specification task {number}")
        key = _identifier(entry.get("id"), f"specification task {number} id")
        if key in specified:
            raise SerializationError(f"task {key!r} is listed twice")
        specified.add(key)
        if key not in runtimes:
            raise SerializationError(f"task {key!r} has no execution runtime")
        rows.append(
            _Row(
                key=key,
                parents=_texts(entry.get("parents", []), f"task {key!r} parents"),
                children=_texts(entry.get("children", []), f"task {key!r} children"),
                inputs=frozenset(_texts(entry.get("inputFiles", []), f"task {key!r} inputFiles")),
                outputs=frozenset(
                    _texts(entry.get("outputFiles", []), f"task {key!r} outputFiles")
                ),
                runtime=runtimes[key],
            )
        )
    for task_id in runtimes:
        if task_id not in specified:
            raise SerializationError(
                f"execution task {task_id!r} is not a task of the specification"
            )
    return rows, sizes


def _flat_rows(workflow: Mapping[str, Any]) -> tuple[list[_Row], dict[str, int]]:
    """Read the tasks and file sizes of the schema 1.4 layout."""
    sizes: dict[str, int] = {}
    rows: list[_Row] = []
    named: set[str] = set()
    for number, raw in enumerate(_sequence(workflow.get("tasks"), "workflow.tasks")):
        entry = _mapping(raw, f"task {number}")
        key = _identifier(entry.get("name"), f"task {number} name")
        if key in named:
            raise SerializationError(f"task {key!r} is listed twice")
        named.add(key)
        inputs: set[str] = set()
        outputs: set[str] = set()
        for index, raw_file in enumerate(_sequence(entry.get("files", []), f"task {key!r} files")):
            item = _mapping(raw_file, f"file {index} of task {key!r}")
            file_name = _text(item.get("name"), f"file {index} of task {key!r} name")
            _record_size(sizes, file_name, item.get("sizeInBytes"))
            link = item.get("link")
            if link == "input":
                inputs.add(file_name)
            elif link == "output":
                outputs.add(file_name)
            else:
                raise SerializationError(
                    f"file {file_name!r} of task {key!r} has link {link!r}, "
                    "expected 'input' or 'output'"
                )
        if "runtimeInSeconds" not in entry:
            raise SerializationError(f"task {key!r} has no runtimeInSeconds")
        rows.append(
            _Row(
                key=key,
                parents=_texts(entry.get("parents", []), f"task {key!r} parents"),
                children=_texts(entry.get("children", []), f"task {key!r} children"),
                inputs=frozenset(inputs),
                outputs=frozenset(outputs),
                runtime=_runtime(entry.get("runtimeInSeconds"), key),
            )
        )
    return rows, sizes


def _edges(rows: Sequence[_Row]) -> list[tuple[str, str]]:
    """Return the parent-child pairs, ordered by parent and then child position."""
    if not rows:
        raise SerializationError("the workflow has no task")
    position = {row.key: number for number, row in enumerate(rows)}
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        for parent in row.parents:
            pairs.add((_known(parent, position, row.key, "parent"), row.key))
        for child in row.children:
            pairs.add((row.key, _known(child, position, row.key, "child")))
    for before, after in pairs:
        if before == after:
            raise SerializationError(f"task {before!r} depends on itself")
    return sorted(pairs, key=lambda pair: (position[pair[0]], position[pair[1]]))


def _known(key: str, position: Mapping[str, int], owner: str, role: str) -> str:
    if key not in position:
        raise SerializationError(f"task {owner!r} names unknown {role} {key!r}")
    return key


def _task(row: _Row, platform: Platform, nodes: Sequence[Resource], group: str) -> Task:
    """Return the task of ``row``, with one mode per node."""
    try:
        return Task(row.key, platform.modes(row.runtime, nodes), group=group)
    except ModelError as error:
        raise SerializationError(f"task {row.key!r} cannot be represented: {error}") from error


def _dependency(
    parent: _Row, child: _Row, sizes: Mapping[str, int], bytes_per_unit: float
) -> Dependency:
    """Return the dependency that carries the files ``parent`` passes to ``child``."""
    shared = _shared_bytes(parent, child, sizes)
    try:
        data = shared / bytes_per_unit
    except OverflowError:
        data = math.inf
    if not math.isfinite(data):
        raise SerializationError(
            f"the files passed from {parent.key!r} to {child.key!r} are too large to express "
            "in units of bytes_per_unit"
        )
    return Dependency(parent.key, child.key, data=data)


def _shared_bytes(parent: _Row, child: _Row, sizes: Mapping[str, int]) -> int:
    """Return the bytes of the files ``parent`` writes and ``child`` reads."""
    total = 0
    for name in sorted(parent.outputs & child.inputs):
        if name not in sizes:
            raise SerializationError(
                f"file {name!r} passed from {parent.key!r} to {child.key!r} has no size"
            )
        total += sizes[name]
    return total


def _nodes(platform: Platform, tiers: Sequence[str] | None) -> tuple[Resource, ...]:
    try:
        return eligible_nodes(platform, tiers)
    except ValueError as error:
        raise ConfigError(f"invalid arguments for read_wfcommons: {error}") from error


def _slug(text: str) -> str:
    """Return ``text`` as an id that starts with a letter or digit, empty if nothing is left."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).lstrip("-._").rstrip("-.")


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SerializationError(f"{where} must be a JSON object")
    return value


def _sequence(value: object, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise SerializationError(f"{where} must be a JSON array")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SerializationError(f"{where} must be a non-empty string")
    return value


def _identifier(value: object, where: str) -> str:
    """Return a task identifier, which becomes the id of a task as it is written."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise SerializationError(f"{where} must be a non-empty string without surrounding spaces")
    return value


def _texts(value: object, where: str) -> tuple[str, ...]:
    return tuple(_text(item, where) for item in _sequence(value, where))


def _record_size(sizes: dict[str, int], name: str, value: object) -> None:
    """Record the size of file ``name``, a whole number of bytes."""
    size = _whole(value)
    if size is None or size < 0:
        raise SerializationError(f"file {name!r} sizeInBytes must be a non-negative integer")
    if sizes.setdefault(name, size) != size:
        raise SerializationError(f"file {name!r} is listed with two different sizes")


def _whole(value: object) -> int | None:
    """Return ``value`` as an integer if it is a whole number, else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        try:
            number = float(value)
        except OverflowError:
            return None
        return int(number) if number.is_integer() else None
    return None


def _runtime(value: object, task: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise SerializationError(f"task {task!r} runtimeInSeconds must be a number")
    try:
        runtime = float(value)
    except OverflowError:
        runtime = math.inf
    if not math.isfinite(runtime) or runtime < 0.0:
        raise SerializationError(f"task {task!r} runtimeInSeconds must be finite and non-negative")
    return runtime + 0.0
