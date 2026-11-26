"""Events and traces: what happens to an online scheduler, and when.

Three kinds of event drive an online run:

- :class:`Arrival`: a :class:`Workflow`, a graph of tasks, becomes known at
  its arrival time. Its tasks may start from then on.
- :class:`Failure`: a node stops. The tasks running on it are lost and the
  node takes no task until it is repaired.
- :class:`Repair`: a failed node is available again.

A :class:`Trace` is the sorted, immutable list of the events of one run. It is
an input, built before any method runs, so every method compared on a trace
faces the same arrivals and failures: common random numbers by construction.
A trace is serialisable and carries a content hash, :attr:`Trace.digest`, the
SHA-256 digest of its canonical JSON document, which run records cite instead
of repeating the workflows.

Events are ordered by time; at one instant repairs come before failures and
failures before arrivals, then by subject (node or workflow id). Events at
one instant form one decision point of the runner.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any

from dispatchatlas.core.errors import ModelError, SerializationError
from dispatchatlas.core.io import atomic_write_text
from dispatchatlas.core.model import Dependency, Problem, Task
from dispatchatlas.core.serialization import (
    _dependency_from_dict,
    _dependency_to_dict,
    _task_from_dict,
    _task_to_dict,
    canonical_json,
    digest,
)

__all__ = [
    "TIME_DECIMALS",
    "TRACE_FORMAT",
    "Arrival",
    "Event",
    "EventKind",
    "Failure",
    "Repair",
    "Trace",
    "Workflow",
    "workflows_from_problem",
]

TRACE_FORMAT = "dispatchatlas.online-trace"
"""Value of the ``format`` field of a trace document."""

TIME_DECIMALS = 6
"""A trace holds event times rounded to this many decimals: microseconds."""

_VERSION = 1


def round_time(time: float) -> float:
    """Return ``time`` as a float rounded to :data:`TIME_DECIMALS`, never ``-0.0``."""
    return round(float(time), TIME_DECIMALS) + 0.0


class EventKind(StrEnum):
    """The kind of an event, in the order events at one instant are applied."""

    REPAIR = "repair"
    FAILURE = "failure"
    ARRIVAL = "arrival"


_RANK = {EventKind.REPAIR: 0, EventKind.FAILURE: 1, EventKind.ARRIVAL: 2}


@dataclass(frozen=True)
class Workflow:
    """A graph of tasks that arrives as one unit.

    Task modes name the resources of the platform the workflow will run on.
    A task's ``release``, ``due``, and ``deadline`` are offsets from the
    workflow's arrival; the release is usually 0. Its ``group`` is replaced by
    the workflow id when the runner builds its problem. Task ids must be unique
    across every workflow of a trace.

    Attributes:
        id: Unique identifier of the workflow.
        tasks: Its tasks.
        dependencies: Dependencies between its own tasks only.
    """

    id: str
    tasks: tuple[Task, ...]
    dependencies: tuple[Dependency, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or self.id != self.id.strip():
            raise ModelError("workflow id must be a non-empty string without surrounding spaces")
        tasks = tuple(self.tasks)
        if not tasks or not all(isinstance(task, Task) for task in tasks):
            raise ModelError(f"workflow {self.id!r} needs at least one Task")
        ids = [task.id for task in tasks]
        if len(set(ids)) != len(ids):
            raise ModelError(f"workflow {self.id!r} lists a task id twice")
        known = set(ids)
        dependencies = tuple(self.dependencies)
        for dependency in dependencies:
            if dependency.before not in known or dependency.after not in known:
                raise ModelError(
                    f"workflow {self.id!r} has a dependency {dependency.before!r} -> "
                    f"{dependency.after!r} outside its tasks"
                )
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "dependencies", dependencies)

    def to_dict(self) -> dict[str, Any]:
        """Return the workflow as a JSON-ready dictionary."""
        document: dict[str, Any] = {"id": self.id, "tasks": [_task_to_dict(t) for t in self.tasks]}
        if self.dependencies:
            document["dependencies"] = [_dependency_to_dict(d) for d in self.dependencies]
        return document

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Workflow:
        """Rebuild a workflow from :meth:`to_dict` output."""
        return cls(
            id=document["id"],
            tasks=tuple(_task_from_dict(item) for item in document["tasks"]),
            dependencies=tuple(
                _dependency_from_dict(item) for item in document.get("dependencies", ())
            ),
        )


@dataclass(frozen=True)
class Arrival:
    """A workflow becomes known at ``time``."""

    time: float
    workflow: Workflow

    @property
    def kind(self) -> EventKind:
        """:attr:`EventKind.ARRIVAL`."""
        return EventKind.ARRIVAL

    @property
    def subject(self) -> str:
        """The workflow id."""
        return self.workflow.id


@dataclass(frozen=True)
class Failure:
    """Node ``node`` stops at ``time``."""

    time: float
    node: str

    @property
    def kind(self) -> EventKind:
        """:attr:`EventKind.FAILURE`."""
        return EventKind.FAILURE

    @property
    def subject(self) -> str:
        """The node id."""
        return self.node


@dataclass(frozen=True)
class Repair:
    """Node ``node`` is available again at ``time``."""

    time: float
    node: str

    @property
    def kind(self) -> EventKind:
        """:attr:`EventKind.REPAIR`."""
        return EventKind.REPAIR

    @property
    def subject(self) -> str:
        """The node id."""
        return self.node


type Event = Arrival | Failure | Repair
"""One event of a trace."""


def _key(event: Event) -> tuple[float, int, str]:
    return (event.time, _RANK[event.kind], event.subject)


@dataclass(frozen=True)
class Trace:
    """The sorted, immutable events of one online run.

    Building a trace sorts its events and checks them: times are finite and
    non-negative, workflow ids and task ids are unique, and each node
    alternates between failure and repair, starting up. Times are stored as
    floats rounded to microseconds (:data:`TIME_DECIMALS`), so ``Arrival(0, w)``
    and ``Arrival(0.0, w)`` are the same event with the same digest, and two
    times closer than half a microsecond are one instant.

    Attributes:
        events: The events, sorted.
    """

    events: tuple[Event, ...]

    def __post_init__(self) -> None:
        rounded: list[Event] = []
        for event in self.events:
            if not isinstance(event, (Arrival, Failure, Repair)):
                raise ModelError(f"a trace holds events, got {event!r}")
            if isinstance(event.time, bool) or not (
                isinstance(event.time, (int, float)) and math.isfinite(event.time)
            ):
                raise ModelError(f"event time must be a finite number, got {event.time!r}")
            if event.time < 0.0:
                raise ModelError(f"event time must be non-negative, got {event.time!r}")
            rounded.append(replace(event, time=round_time(event.time)))
        events = tuple(sorted(rounded, key=_key))
        workflows: set[str] = set()
        tasks: set[str] = set()
        down: set[str] = set()
        for event in events:
            if isinstance(event, Arrival):
                if event.workflow.id in workflows:
                    raise ModelError(f"workflow {event.workflow.id!r} arrives twice")
                workflows.add(event.workflow.id)
                for task in event.workflow.tasks:
                    if task.id in tasks:
                        raise ModelError(f"task id {task.id!r} appears in two workflows")
                    tasks.add(task.id)
            elif isinstance(event, Failure):
                if event.node in down:
                    raise ModelError(f"node {event.node!r} fails at {event.time} while down")
                down.add(event.node)
            else:
                if event.node not in down:
                    raise ModelError(f"node {event.node!r} is repaired at {event.time} while up")
                down.discard(event.node)
        object.__setattr__(self, "events", events)

    @classmethod
    def of(cls, events: Iterable[Event]) -> Trace:
        """Build a trace from any iterable of events."""
        return cls(tuple(events))

    def merged(self, *others: Trace) -> Trace:
        """Return a trace holding the events of this trace and of ``others``."""
        return Trace(self.events + tuple(e for other in others for e in other.events))

    @property
    def arrivals(self) -> tuple[Arrival, ...]:
        """The arrivals, in order."""
        return tuple(event for event in self.events if isinstance(event, Arrival))

    @property
    def nodes(self) -> frozenset[str]:
        """Every node that a failure or repair names."""
        return frozenset(e.node for e in self.events if isinstance(e, (Failure, Repair)))

    def __len__(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        """Return the trace as a JSON-ready document."""
        rows: list[dict[str, Any]] = []
        for event in self.events:
            if isinstance(event, Arrival):
                rows.append(
                    {"time": event.time, "kind": "arrival", "workflow": event.workflow.to_dict()}
                )
            else:
                rows.append({"time": event.time, "kind": event.kind.value, "node": event.node})
        return {"format": TRACE_FORMAT, "version": _VERSION, "events": rows}

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> Trace:
        """Rebuild a trace from :meth:`to_dict` output.

        Raises:
            SerializationError: If the document is not a trace document.
        """
        if document.get("format") != TRACE_FORMAT or document.get("version") != _VERSION:
            raise SerializationError(f"not a version {_VERSION} {TRACE_FORMAT} document")
        events: list[Event] = []
        try:
            for row in document["events"]:
                kind = row["kind"]
                if kind == "arrival":
                    events.append(Arrival(row["time"], Workflow.from_dict(row["workflow"])))
                elif kind == "failure":
                    events.append(Failure(row["time"], row["node"]))
                elif kind == "repair":
                    events.append(Repair(row["time"], row["node"]))
                else:
                    raise SerializationError(f"unknown event kind {kind!r}")
        except (KeyError, TypeError) as error:
            raise SerializationError(f"malformed trace document: {error}") from error
        return cls(tuple(events))

    @cached_property
    def digest(self) -> str:
        """SHA-256 hex digest of the canonical JSON of :meth:`to_dict`."""
        return digest(self.to_dict())

    def dumps(self) -> str:
        """Return the trace as canonical JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def loads(cls, text: str) -> Trace:
        """Read a trace from JSON text."""
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            raise SerializationError(f"invalid trace JSON: {error}") from error
        return cls.from_dict(document)

    def save(self, path: str | Path) -> Path:
        """Write the trace to ``path`` atomically, as canonical JSON. Returns the path."""
        return atomic_write_text(Path(path), self.dumps() + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        """Read a trace written by :meth:`save`."""
        return cls.loads(Path(path).read_text(encoding="utf-8"))


def workflows_from_problem(problem: Problem) -> list[tuple[Workflow, float]]:
    """Split a static problem into workflows with their submit times.

    Each group of the problem becomes a workflow named after the group; a task
    without a group becomes a workflow of its own, named after the task. A
    workflow's submit time is the earliest release of its tasks, and each
    task's release, due date, and deadline become offsets from that time. Dependencies between two
    groups are refused, since workflows arrive independently.

    Args:
        problem: A problem whose groups are its workflows, such as one of the
            continuum families.

    Returns:
        ``(workflow, submit time)`` pairs in the order the groups first appear.

    Raises:
        ModelError: If a dependency joins two workflows.
    """
    members: dict[str, list[Task]] = {}
    for task in problem.tasks:
        members.setdefault(task.group if task.group is not None else task.id, []).append(task)
    owner = {task.id: name for name, tasks in members.items() for task in tasks}
    edges: dict[str, list[Dependency]] = {name: [] for name in members}
    for dependency in problem.dependencies:
        name = owner[dependency.before]
        if owner[dependency.after] != name:
            raise ModelError(
                f"dependency {dependency.before!r} -> {dependency.after!r} joins two workflows"
            )
        edges[name].append(dependency)
    result: list[tuple[Workflow, float]] = []
    for name, tasks in members.items():
        submit = min(task.release for task in tasks)
        shifted = tuple(_shifted(task, submit) for task in tasks)
        result.append((Workflow(name, shifted, tuple(edges[name])), submit))
    return result


def _shifted(task: Task, offset: float) -> Task:
    if offset == 0.0:
        return task
    return Task(
        task.id,
        task.modes,
        release=task.release - offset,
        due=None if task.due is None else task.due - offset,
        deadline=None if task.deadline is None else task.deadline - offset,
        weight=task.weight,
        group=task.group,
        metadata=task.metadata,
    )
