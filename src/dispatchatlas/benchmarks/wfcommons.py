"""Reading WfCommons workflow instances onto a continuum platform.

WfCommons (Coleman et al. 2022) publishes executions of real scientific
workflows in the WfFormat JSON format. :func:`read_wfcommons` maps one
instance onto a :class:`~dispatchatlas.benchmarks.continuum.platform.Platform`:

- Every workflow task becomes a task with one mode per eligible node. Its
  recorded runtime is taken as its work, so it runs ``runtime / speed``
  seconds on one core of the chosen node.
- Every parent-child pair becomes a dependency that carries the bytes of the
  files the parent writes and the child reads, divided by ``bytes_per_unit``
  to express them in the platform's data unit (megabytes by default). The
  network then prices the transfer whenever the two tasks run on different
  nodes.
- All tasks form one group, the workflow, and are released at time 0.

The reader consumes a few fields and checks each of them; anything missing
or malformed raises :class:`~dispatchatlas.core.errors.SerializationError`
naming the field. Runtimes are never invented: a task without a recorded
runtime is refused. Zero runtimes are accepted, as recorded measurements.

Supported ``schemaVersion`` values and the fields read:

- ``1.4``: ``workflow.tasks[]`` with ``name`` (the task's identity),
  ``parents`` and optional ``children`` (task names), ``runtimeInSeconds``,
  and ``files[]`` entries with ``name``, ``sizeInBytes``, and ``link``
  (``"input"`` or ``"output"``).
- ``1.5`` and ``1.6``: ``workflow.specification.tasks[]`` with ``id``,
  ``parents``, ``children``, ``inputFiles``, and ``outputFiles``;
  ``workflow.specification.files[]`` with ``id`` and ``sizeInBytes``; and
  ``workflow.execution.tasks[]`` with ``id`` and ``runtimeInSeconds``.

Dependencies are the union of the ``parents`` and ``children`` lists, each
pair counted once.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dispatchatlas.benchmarks._common import check_number, check_objectives
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
        source: The parsed JSON document, or the path of a JSON file.
        platform: Nodes and network the workflow runs on.
        tiers: Tiers whose nodes may run tasks. Defaults to every tier of the
            platform except ``"device"``.
        problem_id: Id of the problem. Defaults to the instance ``name`` with
            characters outside letters, digits, ``.``, ``_``, and ``-``
            replaced, or else the file name without its extension.
        objectives: One objective name or several, primary first.
        bytes_per_unit: Bytes per data unit of the platform's bandwidths.

    Returns:
        The problem.

    Raises:
        SerializationError: If the document is not a supported WfFormat
            instance or its content is inconsistent.
        ConfigError: If an objective is unknown or repeated, no node of
            ``tiers`` exists, or ``bytes_per_unit`` is not a positive number.
        OSError: If the file cannot be read.
    """
    try:
        names = check_objectives(objectives)
        check_number("bytes_per_unit", bytes_per_unit, positive=True)
    except ValueError as error:
        raise ConfigError(f"invalid arguments for read_wfcommons: {error}") from error
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
    identifier = problem_id or _slug(name if isinstance(name, str) else "") or stem or "wfcommons"
    nodes = _nodes(platform, tiers)
    by_key = {row.key: row for row in rows}
    tasks = tuple(
        Task(row.key, platform.modes(row.runtime, nodes), group=identifier) for row in rows
    )
    dependencies = []
    for before, after in edges:
        shared = _shared_bytes(by_key[before], by_key[after], sizes)
        dependencies.append(Dependency(before, after, data=shared / bytes_per_unit))
    metadata: dict[str, str] = {"source": "wfcommons", "schema_version": str(version)}
    if isinstance(name, str) and name:
        metadata["name"] = name
    try:
        return Problem(
            id=identifier,
            tasks=tasks,
            resources=platform.resources,
            dependencies=tuple(dependencies),
            objectives=names,
            network=platform.network,
            metadata=metadata,
        )
    except ModelError as error:
        raise SerializationError(f"WfFormat instance {identifier!r} is invalid: {error}") from error


def _load(source: Mapping[str, Any] | str | os.PathLike[str]) -> tuple[Mapping[str, Any], str]:
    """Return the document and, for a file, its name without extension."""
    if isinstance(source, Mapping):
        return source, ""
    path = Path(source)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SerializationError(f"invalid JSON in {path}: {error}") from error
    return _mapping(document, "document"), _slug(path.stem)


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
        task_id = _text(entry.get("id"), f"execution task {number} id")
        runtimes[task_id] = _runtime(entry.get("runtimeInSeconds"), task_id)
    rows: list[_Row] = []
    for number, raw in enumerate(
        _sequence(specification.get("tasks"), "workflow.specification.tasks")
    ):
        entry = _mapping(raw, f"specification task {number}")
        key = _text(entry.get("id"), f"specification task {number} id")
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
    return rows, sizes


def _flat_rows(workflow: Mapping[str, Any]) -> tuple[list[_Row], dict[str, int]]:
    """Read the tasks and file sizes of the schema 1.4 layout."""
    sizes: dict[str, int] = {}
    rows: list[_Row] = []
    for number, raw in enumerate(_sequence(workflow.get("tasks"), "workflow.tasks")):
        entry = _mapping(raw, f"task {number}")
        key = _text(entry.get("name"), f"task {number} name")
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
    if len(position) != len(rows):
        raise SerializationError("workflow task identifiers must be unique")
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
    chosen = [tier for tier in platform.tiers if tier != "device"] if tiers is None else tiers
    nodes = tuple(node for tier in chosen for node in platform.nodes(tier))
    if not nodes:
        raise ConfigError(f"the platform has no node in tiers {list(chosen)}")
    return nodes


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")


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


def _texts(value: object, where: str) -> tuple[str, ...]:
    return tuple(_text(item, where) for item in _sequence(value, where))


def _record_size(sizes: dict[str, int], name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SerializationError(f"file {name!r} sizeInBytes must be a non-negative integer")
    if sizes.setdefault(name, value) != value:
        raise SerializationError(f"file {name!r} is listed with two different sizes")


def _runtime(value: object, task: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SerializationError(f"task {task!r} runtimeInSeconds must be a number")
    runtime = float(value)
    if not math.isfinite(runtime) or runtime < 0.0:
        raise SerializationError(f"task {task!r} runtimeInSeconds must be finite and non-negative")
    return runtime
