"""The scheduling data model: resources, tasks, modes, dependencies, and problems.

One model covers classic machine scheduling, project scheduling, and task
placement across a computing continuum:

- A **resource** has a capacity. A machine is a resource of capacity 1. A compute
  node with eight cores is a resource of capacity 8.
- A **task** runs in exactly one of its **modes**. A mode fixes the duration and
  the amount of each resource the task holds while it runs. Alternative
  machines (flexible job shops, unrelated parallel machines, placement on
  heterogeneous nodes) are alternative modes.
- A **dependency** forces a task to start after another one finishes, plus an
  optional time lag and an optional data transfer whose duration a
  :class:`Network` derives from where the two tasks run.
- A **problem** bundles tasks, resources, dependencies, the objectives to
  optimise, and optional setup times, network, and permutation constraint.

A mode of zero duration occupies an instant. On an **exclusive** resource, one
where every demand takes the whole capacity (a machine, typically), no other
task may run across that instant, as in the disjunctive graph of shop
scheduling. On other resources a zero-duration task holds nothing, as in
project scheduling. In a permutation problem it also takes its place in each
machine's group order, which reproduces the completion-time recurrence of the
permutation flow shop for instances with zero processing times. Zero-duration
modes may not use machines with setup times.

Every object is immutable. A :class:`Problem` validates its whole structure when
it is built, so code that receives one can rely on it being consistent.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

from dispatchatlas.core._frozen import FrozenMap
from dispatchatlas.core.errors import ModelError

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "FEATURES",
    "SHAPES",
    "Dependency",
    "Link",
    "Metadata",
    "Mode",
    "Network",
    "Problem",
    "Resource",
    "Scalar",
    "SetupTimes",
    "Task",
]

type Scalar = str | int | float | bool
type Metadata = Mapping[str, Scalar]

FEATURES: Mapping[str, str] = FrozenMap(
    {
        "precedence": "Some tasks must wait for others to finish.",
        "lags": "Some dependencies carry a minimum time lag.",
        "modes": "Some tasks can run in more than one mode.",
        "multi-resource": "Some modes hold more than one resource at once.",
        "cumulative": "Some resources can run several tasks at once.",
        "release": "Some tasks cannot start before a release time.",
        "due": "Some tasks carry a soft due date.",
        "deadline": "Some tasks carry a hard deadline.",
        "setups": "Some resources need sequence-dependent setup times.",
        "network": "Some dependencies transfer data across a network.",
        "permutation": "Every resource must process the task groups in one order.",
        "groups": "Tasks are grouped into jobs or applications.",
    }
)
"""Names of the structural features a problem can exhibit, with their meaning.

Solvers declare the features they model in :attr:`SolverInfo.supports`, and
campaigns refuse to pair a solver with a problem that has an unsupported one.
"""

SHAPES: Mapping[str, str] = FrozenMap(
    {
        "job-shop": (
            "The tasks form chains, the jobs, and every task has one mode that runs it on "
            "one machine."
        ),
    }
)
"""Names of the structural shapes a problem can have, with their meaning.

A shape describes the whole problem, where a feature (see :data:`FEATURES`) marks one
property of it. :attr:`Problem.shapes` lists the shapes a problem has. A solver that
works on one shape only names it in :attr:`SolverInfo.shape`, and campaigns refuse to
pair it with a problem that lacks the shape. A feature that the definition of a shape
does not name, such as release times or lags, does not decide the shape; a solver
leaves the features it does not model out of :attr:`SolverInfo.supports`.
"""


def _check_id(value: object, what: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ModelError(f"{what} must be a non-empty string without surrounding spaces")
    return value


def _check_number(
    value: object,
    what: str,
    *,
    minimum: float | None = 0.0,
    allow_infinite: bool = False,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{what} must be a number, got {value!r}")
    number = float(value)
    if math.isnan(number) or (math.isinf(number) and not allow_infinite):
        raise ModelError(f"{what} must be finite, got {value!r}")
    if strictly_positive and number <= 0.0:
        raise ModelError(f"{what} must be positive, got {value!r}")
    if minimum is not None and number < minimum:
        raise ModelError(f"{what} must be at least {minimum}, got {value!r}")
    return number


def _freeze_metadata(value: Mapping[str, Any], what: str) -> Mapping[str, Scalar]:
    if not isinstance(value, Mapping):
        raise ModelError(f"{what} metadata must be a mapping")
    if not all(isinstance(key, str) and key for key in value):
        raise ModelError(f"{what} metadata keys must be non-empty strings")
    frozen: dict[str, Scalar] = {}
    for key in sorted(value):
        item = value[key]
        if not isinstance(item, (str, int, float, bool)):
            raise ModelError(f"{what} metadata value for {key!r} must be a string, number, or bool")
        if isinstance(item, float) and not math.isfinite(item):
            raise ModelError(f"{what} metadata value for {key!r} must be finite, got {item!r}")
        frozen[key] = item
    return FrozenMap(frozen)


@dataclass(frozen=True, slots=True)
class Resource:
    """A renewable resource that tasks hold while they run.

    Attributes:
        id: Unique identifier within the problem.
        capacity: Units available at any instant. 1 models a machine that runs
            one task at a time. Larger values model pools such as the cores of a
            compute node.
        kind: Free-form label such as ``"machine"``, ``"node"``, or ``"device"``.
        tier: Network endpoint group, for example ``"edge"``, ``"fog"``, or
            ``"cloud"``. Links of a :class:`Network` join tiers. When absent, the
            resource is its own endpoint.
        idle_power: Power drawn while powered on, in watts.
        busy_power: Power drawn at full utilisation, in watts. A task holding
            ``u`` of ``capacity`` units adds ``(busy_power - idle_power) * u /
            capacity`` watts while it runs.
        cost_rate: Monetary cost per second of full utilisation. A task holding
            ``u`` units for ``d`` seconds costs ``cost_rate * d * u / capacity``.
        metadata: Free-form annotations. Never read by solvers.
    """

    id: str
    capacity: float = 1.0
    kind: str = "machine"
    tier: str | None = None
    idle_power: float = 0.0
    busy_power: float = 0.0
    cost_rate: float = 0.0
    metadata: Metadata = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _check_id(self.id, "resource id")
        where = f"resource {self.id!r}"
        object.__setattr__(
            self,
            "capacity",
            _check_number(self.capacity, f"{where} capacity", strictly_positive=True),
        )
        _check_id(self.kind, f"{where} kind")
        if self.tier is not None:
            _check_id(self.tier, f"{where} tier")
        idle = _check_number(self.idle_power, f"{where} idle_power")
        busy = _check_number(self.busy_power, f"{where} busy_power")
        if busy < idle:
            raise ModelError(f"{where} busy_power must be at least idle_power")
        object.__setattr__(self, "idle_power", idle)
        object.__setattr__(self, "busy_power", busy)
        object.__setattr__(self, "cost_rate", _check_number(self.cost_rate, f"{where} cost_rate"))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata, where))

    @property
    def endpoint(self) -> str:
        """The network endpoint of this resource: its tier, or its own id."""
        return self.tier if self.tier is not None else self.id

    @property
    def unary(self) -> bool:
        """Whether the resource runs at most one unit-demand task at a time."""
        return self.capacity == 1.0


@dataclass(frozen=True, slots=True)
class Mode:
    """One way of executing a task.

    Attributes:
        duration: Processing time, in seconds or any consistent time unit.
        demands: Units of each resource held for the whole duration, keyed by
            resource id. Empty for a task that holds nothing, such as a dummy
            start or end task.
        location: Resource whose network endpoint is the task's location when a
            dependency transfers data. Defaults to the demanded resource when
            the mode demands exactly one.
    """

    duration: float
    demands: Mapping[str, float] = field(default_factory=dict, hash=False)
    location: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "duration", _check_number(self.duration, "mode duration"))
        if not isinstance(self.demands, Mapping):
            raise ModelError("mode demands must be a mapping of resource id to units")
        for resource_id in self.demands:
            _check_id(resource_id, "demanded resource id")
        demands: dict[str, float] = {}
        for resource_id in sorted(self.demands):
            demands[resource_id] = _check_number(
                self.demands[resource_id],
                f"demand on {resource_id!r}",
                strictly_positive=True,
            )
        object.__setattr__(self, "demands", FrozenMap(demands))
        if self.location is None and len(demands) == 1:
            object.__setattr__(self, "location", next(iter(demands)))
        elif self.location is not None:
            _check_id(self.location, "mode location")


@dataclass(frozen=True, slots=True)
class Task:
    """A unit of work that runs once, in one of its modes.

    Attributes:
        id: Unique identifier within the problem.
        modes: Alternative ways of running the task. At least one.
        release: Earliest start time.
        due: Soft due date. Finishing later adds tardiness, which objectives
            such as ``total_tardiness`` penalise.
        deadline: Hard deadline. A schedule that finishes the task later is
            infeasible.
        weight: Importance used by weighted objectives. Tasks in one group must
            share a weight.
        group: Job, application, or tenant the task belongs to. Completion-time
            objectives are computed per group.
        metadata: Free-form annotations. Never read by solvers.
    """

    id: str
    modes: tuple[Mode, ...]
    release: float = 0.0
    due: float | None = None
    deadline: float | None = None
    weight: float = 1.0
    group: str | None = None
    metadata: Metadata = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _check_id(self.id, "task id")
        where = f"task {self.id!r}"
        modes = tuple(self.modes)
        if not modes or not all(isinstance(mode, Mode) for mode in modes):
            raise ModelError(f"{where} needs at least one Mode")
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "release", _check_number(self.release, f"{where} release"))
        if self.due is not None:
            object.__setattr__(self, "due", _check_number(self.due, f"{where} due"))
        if self.deadline is not None:
            object.__setattr__(self, "deadline", _check_number(self.deadline, f"{where} deadline"))
        object.__setattr__(self, "weight", _check_number(self.weight, f"{where} weight"))
        if self.group is not None:
            _check_id(self.group, f"{where} group")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata, where))

    @classmethod
    def on(
        cls,
        task_id: str,
        resource: str,
        duration: float,
        *,
        demand: float = 1.0,
        **attributes: Any,
    ) -> Self:
        """Build a single-mode task that holds ``demand`` units of one resource.

        Args:
            task_id: Task identifier.
            resource: Identifier of the resource the task runs on.
            duration: Processing time.
            demand: Units of the resource held while running.
            **attributes: Further :class:`Task` fields, such as ``release``.

        Returns:
            The task.
        """
        return cls(task_id, (Mode(duration, {resource: demand}),), **attributes)

    @property
    def min_duration(self) -> float:
        """The shortest duration over the task's modes."""
        return min(mode.duration for mode in self.modes)


@dataclass(frozen=True, slots=True)
class Dependency:
    """A finish-to-start precedence between two tasks.

    The ``after`` task may start once the ``before`` task has finished, the
    ``lag`` has elapsed, and ``data`` units have crossed the network between
    the two tasks' locations.

    Attributes:
        before: Id of the predecessor.
        after: Id of the successor.
        lag: Minimum time between the predecessor's finish and the successor's
            start.
        data: Volume the predecessor sends to the successor. Transfer time is
            derived from the problem's :class:`Network`. Ignored without one.
    """

    before: str
    after: str
    lag: float = 0.0
    data: float = 0.0

    def __post_init__(self) -> None:
        _check_id(self.before, "dependency 'before' id")
        _check_id(self.after, "dependency 'after' id")
        if self.before == self.after:
            raise ModelError(f"task {self.before!r} cannot depend on itself")
        where = f"dependency {self.before!r} -> {self.after!r}"
        object.__setattr__(self, "lag", _check_number(self.lag, f"{where} lag"))
        object.__setattr__(self, "data", _check_number(self.data, f"{where} data"))


@dataclass(frozen=True, slots=True)
class Link:
    """A symmetric network link between two endpoints.

    Endpoints are resource tiers, or resource ids for resources without a
    tier. A link from an endpoint to itself describes transfers between two
    different resources of the same tier.

    Attributes:
        a: First endpoint.
        b: Second endpoint.
        bandwidth: Data units per second. Infinite bandwidth leaves only latency.
        latency: Fixed delay added to every transfer, in seconds.
        energy_per_unit: Energy spent per data unit transferred, in joules.
        cost_per_unit: Monetary cost per data unit transferred.
    """

    a: str
    b: str
    bandwidth: float = math.inf
    latency: float = 0.0
    energy_per_unit: float = 0.0
    cost_per_unit: float = 0.0

    def __post_init__(self) -> None:
        _check_id(self.a, "link endpoint")
        _check_id(self.b, "link endpoint")
        where = f"link {self.a!r} <-> {self.b!r}"
        object.__setattr__(
            self,
            "bandwidth",
            _check_number(
                self.bandwidth,
                f"{where} bandwidth",
                allow_infinite=True,
                strictly_positive=True,
            ),
        )
        for name in ("latency", "energy_per_unit", "cost_per_unit"):
            object.__setattr__(self, name, _check_number(getattr(self, name), f"{where} {name}"))

    def transfer_time(self, data: float) -> float:
        """Return the time to move ``data`` units across this link.

        Args:
            data: Volume to transfer. Zero means nothing is sent.

        Returns:
            ``latency + data / bandwidth``, or zero when ``data`` is zero.
        """
        if data <= 0.0:
            return 0.0
        if math.isinf(self.bandwidth):
            return self.latency
        return self.latency + data / self.bandwidth


@dataclass(frozen=True)
class Network:
    """The links that carry data between task locations.

    Two tasks placed on the same resource exchange data instantly. Otherwise
    the transfer uses the link between the endpoints of the two resources.
    """

    links: tuple[Link, ...] = ()

    def __post_init__(self) -> None:
        links = tuple(self.links)
        seen: set[tuple[str, str]] = set()
        for link in links:
            if not isinstance(link, Link):
                raise ModelError("network links must be Link objects")
            key = (min(link.a, link.b), max(link.a, link.b))
            if key in seen:
                raise ModelError(f"network defines the link {key[0]!r} <-> {key[1]!r} twice")
            seen.add(key)
        object.__setattr__(self, "links", links)

    @cached_property
    def _by_endpoints(self) -> Mapping[tuple[str, str], Link]:
        index: dict[tuple[str, str], Link] = {}
        for link in self.links:
            index[(link.a, link.b)] = link
            index[(link.b, link.a)] = link
        return FrozenMap(index)

    def link(self, a: str, b: str) -> Link | None:
        """Return the link joining endpoints ``a`` and ``b``, if one exists."""
        return self._by_endpoints.get((a, b))

    @property
    def endpoints(self) -> frozenset[str]:
        """Every endpoint named by a link."""
        return frozenset(name for link in self.links for name in (link.a, link.b))


@dataclass(frozen=True)
class SetupTimes:
    """Sequence-dependent setup times on machines.

    ``times[(resource, previous, following)]`` is the idle time ``resource``
    needs after finishing ``previous`` before it can start ``following``. A
    ``previous`` of ``None`` gives the setup before the first task on the
    resource. Missing entries mean no setup. Setups apply to resources of
    capacity 1 only.
    """

    times: Mapping[tuple[str, str | None, str], float] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.times, Mapping):
            raise ModelError("setup times must be a mapping")
        frozen: dict[tuple[str, str | None, str], float] = {}
        for key, value in self.times.items():
            if not (isinstance(key, tuple) and len(key) == 3):
                raise ModelError("setup keys must be (resource, previous, following)")
            resource, previous, following = key
            _check_id(resource, "setup resource")
            if previous is not None:
                _check_id(previous, "setup previous task")
            _check_id(following, "setup following task")
            number = _check_number(value, f"setup time {key}")
            if number > 0.0:
                frozen[(resource, previous, following)] = number
        object.__setattr__(self, "times", FrozenMap(frozen))

    def get(self, resource: str, previous: str | None, following: str) -> float:
        """Return the setup time for ``following`` after ``previous`` on ``resource``."""
        return self.times.get((resource, previous, following), 0.0)

    @cached_property
    def resources(self) -> frozenset[str]:
        """Resources that have at least one non-zero setup time."""
        return frozenset(key[0] for key in self.times)


@dataclass(frozen=True)
class Problem:
    """A complete scheduling problem.

    Building a problem validates its whole structure. Invalid input raises
    :class:`~dispatchatlas.core.errors.ModelError` listing every defect.

    Attributes:
        id: Problem identifier, used in file names and result records.
        tasks: The tasks to schedule. Their order is the task index order that
            solvers use.
        resources: The resources tasks hold.
        dependencies: Finish-to-start precedences. Must form an acyclic graph.
        objectives: Names of the objectives to optimise, primary first. See
            :mod:`dispatchatlas.core.objectives` for the registry.
        network: Links used to price data transfers along dependencies.
        setups: Sequence-dependent setup times on machines.
        permutation: Whether every resource must process the task groups in
            one common order, as in the permutation flow shop. A task without
            a group forms a group of its own. Permutation problems use
            capacity-1 machines held whole, and each group uses a machine in
            at most one of its tasks.
        family: Id of the benchmark family that generated the problem, if any.
        metadata: Free-form annotations such as generator parameters. Never read
            by solvers.
    """

    id: str
    tasks: tuple[Task, ...]
    resources: tuple[Resource, ...]
    dependencies: tuple[Dependency, ...] = ()
    objectives: tuple[str, ...] = ("makespan",)
    network: Network | None = None
    setups: SetupTimes | None = None
    permutation: bool = False
    family: str | None = None
    metadata: Metadata = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _check_id(self.id, "problem id")
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        objectives: object = self.objectives
        if isinstance(objectives, str):
            objectives = (objectives,)
        object.__setattr__(self, "objectives", tuple(objectives))  # type: ignore[arg-type]
        if self.family is not None:
            _check_id(self.family, "problem family")
        object.__setattr__(
            self, "metadata", _freeze_metadata(self.metadata, f"problem {self.id!r}")
        )
        issues = _structural_issues(self)
        if issues:
            raise ModelError(f"problem {self.id!r} is invalid", tuple(issues))

    # ------------------------------------------------------------------ lookups

    @cached_property
    def task_index(self) -> Mapping[str, int]:
        """Map from task id to its position in :attr:`tasks`."""
        return FrozenMap({task.id: i for i, task in enumerate(self.tasks)})

    @cached_property
    def resource_index(self) -> Mapping[str, int]:
        """Map from resource id to its position in :attr:`resources`."""
        return FrozenMap({resource.id: i for i, resource in enumerate(self.resources)})

    def task(self, task_id: str) -> Task:
        """Return the task with the given id."""
        return self.tasks[self.task_index[task_id]]

    def resource(self, resource_id: str) -> Resource:
        """Return the resource with the given id."""
        return self.resources[self.resource_index[resource_id]]

    @cached_property
    def exclusive_resources(self) -> frozenset[str]:
        """Ids of the demanded resources where every demand takes the whole capacity.

        Tasks never overlap on such a resource, and a zero-duration task
        occupies an instant that no other task on it may run across.
        """
        capacity = {resource.id: resource.capacity for resource in self.resources}
        demanded: set[str] = set()
        shared: set[str] = set()
        for task in self.tasks:
            for mode in task.modes:
                for resource_id, units in mode.demands.items():
                    demanded.add(resource_id)
                    if units != capacity[resource_id]:
                        shared.add(resource_id)
        return frozenset(demanded - shared)

    @cached_property
    def incoming(self) -> tuple[tuple[Dependency, ...], ...]:
        """Dependencies ending at each task, indexed like :attr:`tasks`."""
        lists: list[list[Dependency]] = [[] for _ in self.tasks]
        for dependency in self.dependencies:
            lists[self.task_index[dependency.after]].append(dependency)
        return tuple(tuple(items) for items in lists)

    @cached_property
    def predecessors(self) -> tuple[tuple[int, ...], ...]:
        """Indices of each task's direct predecessors."""
        index = self.task_index
        return tuple(
            tuple(index[dependency.before] for dependency in items) for items in self.incoming
        )

    @cached_property
    def successors(self) -> tuple[tuple[int, ...], ...]:
        """Indices of each task's direct successors."""
        lists: list[list[int]] = [[] for _ in self.tasks]
        index = self.task_index
        for dependency in self.dependencies:
            lists[index[dependency.before]].append(index[dependency.after])
        return tuple(tuple(items) for items in lists)

    @cached_property
    def topological_order(self) -> tuple[int, ...]:
        """Task indices in a precedence-feasible order, lowest index first on ties."""
        order = _topological_order(len(self.tasks), self.successors, self.predecessors)
        if order is None:  # pragma: no cover - rejected at construction
            raise ModelError(f"problem {self.id!r} has a dependency cycle")
        return order

    @cached_property
    def groups(self) -> Mapping[str, tuple[int, ...]]:
        """Task indices of each named group, in task order."""
        members: dict[str, list[int]] = {}
        for i, task in enumerate(self.tasks):
            if task.group is not None:
                members.setdefault(task.group, []).append(i)
        return FrozenMap({name: tuple(items) for name, items in members.items()})

    @cached_property
    def features(self) -> frozenset[str]:
        """The structural features the problem exhibits. See :data:`FEATURES`."""
        found: set[str] = set()
        if self.dependencies:
            found.add("precedence")
        if any(dependency.lag > 0.0 for dependency in self.dependencies):
            found.add("lags")
        for task in self.tasks:
            if len(task.modes) > 1:
                found.add("modes")
            if any(len(mode.demands) > 1 for mode in task.modes):
                found.add("multi-resource")
            if task.release > 0.0:
                found.add("release")
            if task.due is not None:
                found.add("due")
            if task.deadline is not None:
                found.add("deadline")
            if task.group is not None:
                found.add("groups")
        if any(not resource.unary for resource in self.resources):
            found.add("cumulative")
        if self.setups is not None and self.setups.times:
            found.add("setups")
        if self.network is not None and any(
            dependency.data > 0.0 for dependency in self.dependencies
        ):
            found.add("network")
        if self.permutation:
            found.add("permutation")
        return frozenset(found)

    @cached_property
    def shapes(self) -> frozenset[str]:
        """The structural shapes the problem has. See :data:`SHAPES`.

        A problem has ``"job-shop"`` when its dependencies join the tasks into disjoint
        chains, the jobs, so that every task has at most one predecessor and at most one
        successor, and every task has a single mode that demands a single resource, which
        is a machine: one of the :attr:`exclusive_resources`, where tasks never overlap.
        Release times, due dates, groups, and the other features do not decide the shape.
        """
        found: set[str] = set()
        if _is_job_shop(self):
            found.add("job-shop")
        return frozenset(found)

    @property
    def size(self) -> int:
        """Number of tasks."""
        return len(self.tasks)

    def transfer_time(
        self, dependency: Dependency, source: str | None, target: str | None
    ) -> float:
        """Return the data-transfer time of a dependency between two locations.

        Args:
            dependency: The dependency whose data moves.
            source: Resource id where the predecessor ran, or ``None``.
            target: Resource id where the successor runs, or ``None``.

        Returns:
            Zero when there is no network, no data, no known location, or both
            tasks share a resource. Otherwise the link's transfer time.
        """
        if (
            self.network is None
            or dependency.data <= 0.0
            or source is None
            or target is None
            or source == target
        ):
            return 0.0
        link = self.network.link(self.resource(source).endpoint, self.resource(target).endpoint)
        if link is None:  # pragma: no cover - rejected at construction
            raise ModelError(f"no network link between {source!r} and {target!r}")
        return link.transfer_time(dependency.data)

    def with_objectives(self, *objectives: str) -> Problem:
        """Return a copy that optimises the given objectives, primary first."""
        return replace(self, objectives=tuple(objectives))

    def renamed(self, problem_id: str) -> Problem:
        """Return a copy with another id.

        The structure is unchanged, so the copy skips validation, which matters
        for large generated problems.
        """
        _check_id(problem_id, "problem id")
        clone = copy.copy(self)
        object.__setattr__(clone, "id", problem_id)
        return clone

    def summary(self) -> dict[str, Any]:
        """Return headline counts that describe the problem's size and shape."""
        return {
            "id": self.id,
            "family": self.family,
            "tasks": len(self.tasks),
            "resources": len(self.resources),
            "dependencies": len(self.dependencies),
            "modes": sum(len(task.modes) for task in self.tasks),
            "groups": len(self.groups),
            "objectives": list(self.objectives),
            "features": sorted(self.features),
        }


def _topological_order(
    count: int,
    successors: Sequence[Sequence[int]],
    predecessors: Sequence[Sequence[int]],
) -> tuple[int, ...] | None:
    """Kahn's algorithm with a min-heap, so ties resolve to the lowest index."""
    import heapq  # noqa: PLC0415 - small, local helper

    remaining = [len(items) for items in predecessors]
    ready = [i for i in range(count) if remaining[i] == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        current = heapq.heappop(ready)
        order.append(current)
        for following in successors[current]:
            remaining[following] -= 1
            if remaining[following] == 0:
                heapq.heappush(ready, following)
    return tuple(order) if len(order) == count else None


def _is_job_shop(problem: Problem) -> bool:
    """Return whether the tasks form chains in which every task runs on one machine."""
    machines = problem.exclusive_resources
    return all(
        len(before) <= 1 and len(after) <= 1 and _runs_on_one_machine(task, machines)
        for task, before, after in zip(
            problem.tasks, problem.predecessors, problem.successors, strict=True
        )
    )


def _runs_on_one_machine(task: Task, machines: frozenset[str]) -> bool:
    """Return whether ``task`` has one mode that demands one resource, which is a machine."""
    if len(task.modes) != 1:
        return False
    demands = task.modes[0].demands
    return len(demands) == 1 and next(iter(demands)) in machines


def _structural_issues(problem: Problem) -> list[str]:
    """Return every structural defect of ``problem`` as a readable line."""
    issues: list[str] = []
    if not problem.tasks:
        issues.append("the problem has no tasks")
    if not all(isinstance(task, Task) for task in problem.tasks):
        return [*issues, "every task must be a Task"]
    if not all(isinstance(resource, Resource) for resource in problem.resources):
        return [*issues, "every resource must be a Resource"]
    if not all(isinstance(item, Dependency) for item in problem.dependencies):
        return [*issues, "every dependency must be a Dependency"]

    task_ids = _duplicates(task.id for task in problem.tasks)
    if task_ids:
        issues.append(f"duplicate task ids: {sorted(task_ids)}")
    resource_ids = _duplicates(resource.id for resource in problem.resources)
    if resource_ids:
        issues.append(f"duplicate resource ids: {sorted(resource_ids)}")
    resources = {resource.id: resource for resource in problem.resources}

    for task in problem.tasks:
        for number, mode in enumerate(task.modes):
            for resource_id, amount in mode.demands.items():
                resource = resources.get(resource_id)
                if resource is None:
                    issues.append(
                        f"task {task.id!r} mode {number} demands unknown resource {resource_id!r}"
                    )
                elif amount > resource.capacity * (1.0 + 1e-9):
                    issues.append(
                        f"task {task.id!r} mode {number} demands {amount} of "
                        f"{resource_id!r}, above its capacity {resource.capacity}"
                    )
            if mode.location is not None and mode.location not in resources:
                issues.append(
                    f"task {task.id!r} mode {number} is located on unknown resource "
                    f"{mode.location!r}"
                )

    known = {task.id for task in problem.tasks}
    pairs: set[tuple[str, str]] = set()
    for dependency in problem.dependencies:
        for name in (dependency.before, dependency.after):
            if name not in known:
                issues.append(f"dependency refers to unknown task {name!r}")
        pair = (dependency.before, dependency.after)
        if pair in pairs:
            issues.append(f"dependency {pair[0]!r} -> {pair[1]!r} is declared twice")
        pairs.add(pair)
    # These checks need no valid references; the ones below index by them.
    issues.extend(_objective_issues(problem))
    issues.extend(_group_issues(problem))
    issues.extend(_setup_issues(problem, resources, known))
    if issues:
        return issues

    index = {task.id: i for i, task in enumerate(problem.tasks)}
    successors: list[list[int]] = [[] for _ in problem.tasks]
    predecessors: list[list[int]] = [[] for _ in problem.tasks]
    for dependency in problem.dependencies:
        successors[index[dependency.before]].append(index[dependency.after])
        predecessors[index[dependency.after]].append(index[dependency.before])
    if _topological_order(len(problem.tasks), successors, predecessors) is None:
        issues.append("the dependencies form a cycle")

    issues.extend(_permutation_issues(problem, resources))
    issues.extend(_network_issues(problem, resources))
    return issues


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        (repeated if value in seen else seen).add(value)
    return repeated


def _objective_issues(problem: Problem) -> list[str]:
    from dispatchatlas.core.objectives import is_registered  # noqa: PLC0415

    issues: list[str] = []
    if not problem.objectives:
        issues.append("the problem declares no objective")
    if len(set(problem.objectives)) != len(problem.objectives):
        issues.append("an objective is declared twice")
    for name in problem.objectives:
        if not isinstance(name, str) or not is_registered(name):
            issues.append(f"unknown objective {name!r}")
    return issues


def _group_issues(problem: Problem) -> list[str]:
    weights: dict[str, float] = {}
    issues: list[str] = []
    for task in problem.tasks:
        if task.group is None:
            continue
        first = weights.setdefault(task.group, task.weight)
        if first != task.weight:
            issues.append(f"tasks of group {task.group!r} have different weights")
            weights[task.group] = task.weight
    return sorted(set(issues))


def _setup_issues(
    problem: Problem, resources: Mapping[str, Resource], tasks: set[str]
) -> list[str]:
    if problem.setups is None:
        return []
    issues: list[str] = []
    for resource_id, previous, following in problem.setups.times:
        resource = resources.get(resource_id)
        if resource is None:
            issues.append(f"setup refers to unknown resource {resource_id!r}")
        elif not resource.unary:
            issues.append(f"setups are allowed on capacity-1 resources only: {resource_id!r}")
        for name in (previous, following):
            if name is not None and name not in tasks:
                issues.append(f"setup refers to unknown task {name!r}")
    setup_resources = {key[0] for key in problem.setups.times}
    for task in problem.tasks:
        for mode in task.modes:
            for resource_id, units in mode.demands.items():
                if resource_id not in setup_resources:
                    continue
                if units != 1.0:
                    issues.append(
                        f"task {task.id!r} must hold the whole of machine {resource_id!r}, "
                        "which has setup times"
                    )
                if mode.duration <= 0.0:
                    issues.append(
                        f"task {task.id!r} has a zero-duration mode on machine "
                        f"{resource_id!r}, which has setup times"
                    )
    return sorted(set(issues))


def _permutation_issues(problem: Problem, resources: Mapping[str, Resource]) -> list[str]:
    if not problem.permutation:
        return []
    issues: list[str] = []
    users: dict[tuple[str, str], str] = {}
    for task in problem.tasks:
        group = task.group if task.group is not None else task.id
        used: set[str] = set()
        for mode in task.modes:
            for resource_id, units in mode.demands.items():
                if not resources[resource_id].unary or units != 1.0:
                    issues.append(
                        f"task {task.id!r} must hold the whole of capacity-1 machine "
                        f"{resource_id!r} in a permutation problem"
                    )
                used.add(resource_id)
        for resource_id in sorted(used):
            first = users.setdefault((group, resource_id), task.id)
            if first != task.id:
                issues.append(
                    f"tasks {first!r} and {task.id!r} of group {group!r} can both use machine "
                    f"{resource_id!r}; in a permutation problem a group uses a machine at most once"
                )
    return sorted(set(issues))


def _network_issues(problem: Problem, resources: Mapping[str, Resource]) -> list[str]:
    if problem.network is None:
        return []
    needed: set[tuple[str, str]] = set()
    for dependency in problem.dependencies:
        if dependency.data <= 0.0:
            continue
        sources = {m.location for m in problem.tasks[problem.task_index[dependency.before]].modes}
        targets = {m.location for m in problem.tasks[problem.task_index[dependency.after]].modes}
        for source in sources:
            for target in targets:
                if source is None or target is None or source == target:
                    continue
                a, b = resources[source].endpoint, resources[target].endpoint
                needed.add((min(a, b), max(a, b)))
    return [
        f"the network has no link between endpoints {a!r} and {b!r}"
        for a, b in sorted(needed)
        if problem.network.link(a, b) is None
    ]
