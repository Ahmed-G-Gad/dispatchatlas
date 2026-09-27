"""Applications that devices issue, and the problems they form on a platform.

An :class:`Application` is the unit of work a device issues: a request made
of one task, a chain of stages, or a whole workflow. :func:`build_problem`
turns applications into a scheduling problem in which placement is a
decision:

- Every compute task gets one mode per eligible node, with duration
  ``work / speed`` rounded to microseconds and one core of that node, so
  choosing a mode places the task.
- Every application gets a zero-duration **source** task and **sink** task
  located on its device. The source sends the input data to the entry tasks,
  the exit tasks send their results to the sink, and the network prices both
  transfers. Offloading therefore costs upload and download time, and the
  response time of the application runs from its release to its sink.
- All tasks of an application share its release time, weight, and group, so
  completion-time objectives such as ``mean_response_time`` are computed per
  application.
- The due date and the optional hard deadline sit on the sink, at the release
  plus a factor times the application's contention-free response time,
  rounded up to microseconds.

The contention-free response time is the sink's finish time when resources
never make a task wait: each task starts once the data of its predecessors
have arrived, and each dependency is priced at the predecessor's best node
independently. For a chain, such as a request or a pipeline, this is exactly
the best response time the application can have on its own. For a general
workflow it is a lower bound on it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dispatchatlas.benchmarks.continuum.platform import TIERS, Platform
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.model import Dependency, Mode, Problem, Resource, Scalar, Task

__all__ = ["Application", "build_problem", "eligible_nodes", "response_bound"]


@dataclass(frozen=True, slots=True)
class Application:
    """A request, pipeline, or workflow that a device issues.

    Compute tasks are numbered from 0 in a topological order: every edge goes
    from a lower to a higher number.

    Attributes:
        name: Identifier of the application. It names the group of its tasks
            and prefixes their ids: ``<name>-source``, ``<name>-t<k>``, and
            ``<name>-sink``.
        device: Id of the node that issues the application. Its source and
            sink tasks are located there.
        release: Time the application is issued.
        work: Work of each compute task, in work units.
        edges: ``(before, after, data)`` dependencies between compute tasks,
            with ``before < after`` and the data volume the first sends.
        upload: Data the device sends to each compute task. Empty means none.
            Tasks without predecessors always depend on the source, even
            without data.
        download: Data each compute task returns to the device. Empty means
            none. Tasks without successors always precede the sink.
        tiers: Tiers each compute task may run on. The ``"device"`` tier
            stands for the issuing device only. Empty means every tier of
            :data:`~dispatchatlas.benchmarks.continuum.platform.TIERS` for
            every task.
        weight: Importance of the application in weighted objectives.
    """

    name: str
    device: str
    release: float
    work: tuple[float, ...]
    edges: tuple[tuple[int, int, float], ...] = ()
    upload: tuple[float, ...] = ()
    download: tuple[float, ...] = ()
    tiers: tuple[tuple[str, ...], ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        where = f"application {self.name!r}"
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ConfigError("an application needs a non-empty name without surrounding spaces")
        count = len(self.work)
        if count == 0:
            raise ConfigError(f"{where} has no compute task")
        object.__setattr__(self, "work", tuple(self.work))
        edges = tuple(tuple(edge) for edge in self.edges)
        if any(len(edge) != 3 for edge in edges):
            raise ConfigError(f"{where} edges must be (before, after, data) triples")
        object.__setattr__(self, "edges", edges)
        for name in ("upload", "download"):
            values = tuple(getattr(self, name)) or (0.0,) * count
            if len(values) != count:
                raise ConfigError(f"{where} needs one {name} value per compute task")
            object.__setattr__(self, name, values)
        tiers = tuple(tuple(item) for item in self.tiers) or (TIERS,) * count
        if len(tiers) != count or not all(tiers):
            raise ConfigError(f"{where} needs a non-empty tier list per compute task")
        object.__setattr__(self, "tiers", tiers)
        numbers = [self.release, self.weight, *self.work, *self.upload, *self.download]
        numbers += [data for _, _, data in self.edges]
        if not all(math.isfinite(value) and value >= 0.0 for value in numbers):
            raise ConfigError(f"{where} has a negative or non-finite number")
        pairs: set[tuple[int, int]] = set()
        for before, after, _ in self.edges:
            if not (isinstance(before, int) and isinstance(after, int)):
                raise ConfigError(f"{where} edges must join task numbers")
            if not 0 <= before < after < count:
                raise ConfigError(f"{where} has edge {before} -> {after} out of order")
            if (before, after) in pairs:
                raise ConfigError(f"{where} declares edge {before} -> {after} twice")
            pairs.add((before, after))

    def entries(self) -> tuple[int, ...]:
        """Compute tasks that depend on the source."""
        has_predecessor = {after for _, after, _ in self.edges}
        return tuple(
            k for k in range(len(self.work)) if k not in has_predecessor or self.upload[k] > 0.0
        )

    def exits(self) -> tuple[int, ...]:
        """Compute tasks that precede the sink."""
        has_successor = {before for before, _, _ in self.edges}
        return tuple(
            k for k in range(len(self.work)) if k not in has_successor or self.download[k] > 0.0
        )


def eligible_nodes(platform: Platform, application: Application) -> list[tuple[Resource, ...]]:
    """Return the nodes each compute task of ``application`` may run on.

    Raises:
        ConfigError: If the device is unknown or a task has no eligible node.
    """
    issuer = platform.resource(application.device)
    result: list[tuple[Resource, ...]] = []
    for k, tiers in enumerate(application.tiers):
        nodes: list[Resource] = []
        for tier in tiers:
            for node in (issuer,) if tier == "device" else platform.nodes(tier):
                if node not in nodes:
                    nodes.append(node)
        if not nodes:
            raise ConfigError(
                f"task {k} of application {application.name!r} has no node in tiers {tiers}"
            )
        result.append(tuple(nodes))
    return result


class _Transfers:
    """Transfer times between the nodes of a platform, cached per volume and tier pair."""

    __slots__ = ("_cache", "_platform")

    def __init__(self, platform: Platform) -> None:
        self._platform = platform
        self._cache: dict[tuple[float, str, str], float] = {}

    def __call__(self, data: float, source: Resource, target: Resource) -> float:
        if data <= 0.0 or source.id == target.id:
            return 0.0
        key = (data, source.endpoint, target.endpoint)
        time = self._cache.get(key)
        if time is None:
            time = self._cache[key] = self._platform.transfer_time(data, source.id, target.id)
        return time


def response_bound(platform: Platform, application: Application) -> float:
    """Return the contention-free response time of ``application``.

    Dynamic programming over the compute tasks in index order: the finish
    time of task ``k`` on node ``n`` is its data-ready time on ``n`` plus
    ``work / speed(n)``, where the data of each predecessor arrive from the
    node that makes them arrive first. Durations are those of
    :meth:`~dispatchatlas.benchmarks.continuum.platform.Platform.duration`.
    The result is the time, after the release, at which the last result
    reaches the device. Exact for chains, a lower bound for general workflows
    (see the module description).
    """
    return _response_bound(
        platform, application, eligible_nodes(platform, application), _Transfers(platform)
    )


def _response_bound(
    platform: Platform,
    application: Application,
    nodes: Sequence[tuple[Resource, ...]],
    transfer: _Transfers,
) -> float:
    device = platform.resource(application.device)
    predecessors: list[list[tuple[int, float]]] = [[] for _ in application.work]
    for before, after, data in application.edges:
        predecessors[after].append((before, data))
    entries = set(application.entries())
    finish: list[list[float]] = []
    for k, work in enumerate(application.work):
        row: list[float] = []
        for node in nodes[k]:
            ready = transfer(application.upload[k], device, node) if k in entries else 0.0
            for before, data in predecessors[k]:
                arrival = min(
                    time + transfer(data, source, node)
                    for time, source in zip(finish[before], nodes[before], strict=True)
                )
                ready = max(ready, arrival)
            row.append(ready + platform.duration(work, node.id))
        finish.append(row)
    response = 0.0
    for k in application.exits():
        back = min(
            time + transfer(application.download[k], node, device)
            for time, node in zip(finish[k], nodes[k], strict=True)
        )
        response = max(response, back)
    return response


def build_problem(
    problem_id: str,
    platform: Platform,
    applications: Sequence[Application],
    *,
    objectives: Sequence[str] = ("mean_response_time", "energy", "cost"),
    due_factor: float | None = None,
    deadline_factor: float | None = None,
    family: str | None = None,
    metadata: Mapping[str, Scalar] | None = None,
) -> Problem:
    """Build the scheduling problem of ``applications`` on ``platform``.

    Tasks are listed application by application, each as its source, its
    compute tasks in index order, and its sink. See the module description
    for the modes, dependencies, and due dates this creates.

    Args:
        problem_id: Id of the problem.
        platform: Nodes and network the applications run on.
        applications: The applications, with distinct names.
        objectives: Objectives of the problem, primary first.
        due_factor: Due date of each sink as a multiple of the application's
            contention-free response time, counted from its release. ``None``
            sets no due date.
        deadline_factor: Hard deadline of each sink, as a multiple of the
            same response time. ``None`` sets no deadline.
        family: Id of the generating family, if any.
        metadata: Annotations of the problem.

    Returns:
        The problem.

    Raises:
        ConfigError: If a factor is below 1, two applications share a name,
            or an application refers to an unknown node or has a task without
            an eligible node.
    """
    for name, factor in (("due_factor", due_factor), ("deadline_factor", deadline_factor)):
        if factor is not None and not (math.isfinite(factor) and factor >= 1.0):
            raise ConfigError(f"{name} must be a finite number of at least 1, got {factor!r}")
    names = [application.name for application in applications]
    if len(set(names)) != len(names):
        raise ConfigError("applications must have distinct names")
    builder = _Builder(platform, due_factor, deadline_factor)
    tasks: list[Task] = []
    dependencies: list[Dependency] = []
    for application in applications:
        tasks.extend(builder.tasks(application))
        dependencies.extend(_application_dependencies(application))
    return Problem(
        id=problem_id,
        tasks=tuple(tasks),
        resources=platform.resources,
        dependencies=tuple(dependencies),
        objectives=tuple(objectives),
        network=platform.network,
        family=family,
        metadata=dict(metadata or {}),
    )


class _Builder:
    """Builds the tasks of applications, reusing modes and bounds they share.

    Applications that differ only in name, release, and weight, such as the
    frames of one stream, get the same modes and the same response bound.
    """

    def __init__(
        self, platform: Platform, due_factor: float | None, deadline_factor: float | None
    ) -> None:
        self.platform = platform
        self.due_factor = due_factor
        self.deadline_factor = deadline_factor
        self.transfer = _Transfers(platform)
        self.modes: dict[tuple[float, tuple[str, ...]], tuple[Mode, ...]] = {}
        self.bounds: dict[tuple[object, ...], float] = {}

    def tasks(self, application: Application) -> list[Task]:
        """Return the source, compute, and sink tasks of ``application``."""
        name, release, weight = application.name, application.release, application.weight
        nodes = eligible_nodes(self.platform, application)
        bound = 0.0
        if self.due_factor is not None or self.deadline_factor is not None:
            shape = (
                application.device,
                application.work,
                application.edges,
                application.upload,
                application.download,
                application.tiers,
            )
            if shape not in self.bounds:
                self.bounds[shape] = _response_bound(
                    self.platform, application, nodes, self.transfer
                )
            bound = self.bounds[shape]
        here = (Mode(0.0, location=application.device),)
        tasks = [Task(f"{name}-source", here, release=release, weight=weight, group=name)]
        for k, work in enumerate(application.work):
            key = (work, tuple(node.id for node in nodes[k]))
            if key not in self.modes:
                self.modes[key] = self.platform.modes(work, nodes[k])
            modes = self.modes[key]
            tasks.append(Task(f"{name}-t{k}", modes, release=release, weight=weight, group=name))
        due, deadline = self.due_factor, self.deadline_factor
        tasks.append(
            Task(
                f"{name}-sink",
                here,
                release=release,
                due=None if due is None else _round_up(release + due * bound),
                deadline=None if deadline is None else _round_up(release + deadline * bound),
                weight=weight,
                group=name,
            )
        )
        return tasks


def _round_up(time: float) -> float:
    """Return ``time`` rounded up to microseconds, ignoring floating-point noise."""
    return math.ceil(time * 1e6 - 1e-6) / 1e6


def _application_dependencies(application: Application) -> list[Dependency]:
    name = application.name
    items = [
        Dependency(f"{name}-source", f"{name}-t{k}", data=application.upload[k])
        for k in application.entries()
    ]
    items += [
        Dependency(f"{name}-t{before}", f"{name}-t{after}", data=data)
        for before, after, data in application.edges
    ]
    items += [
        Dependency(f"{name}-t{k}", f"{name}-sink", data=application.download[k])
        for k in application.exits()
    ]
    return items
