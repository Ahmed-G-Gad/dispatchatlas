"""Resource-constrained project scheduling, single-mode and multi-mode.

Both families follow ProGen, the generator of Kolisch, Sprecher and Drexl
(1995) that produced the PSPLIB instances (Kolisch and Sprecher 1997), and take
its three parameters:

- **Network complexity** ``NC``: the average number of non-redundant arcs per
  node of the activity-on-node network, counting a dummy source and sink. A
  network of ``J`` activities has ``ceil(NC * (J + 2))`` arcs, the smallest
  count that reaches ``NC``, of which the ``S`` arcs from the source to the
  start activities and the ``F`` arcs from the finish activities to the sink
  stay implicit. The PSPLIB J30 networks have exactly these counts: 42, 52, and
  62 explicit arcs for ``NC`` 1.5, 1.8, and 2.1.
- **Resource factor** ``RF``: the average share of the ``K`` resources an
  activity requests.
- **Resource strength** ``RS``: each capacity is
  ``K_min + round(RS * (K_max - K_min))``, where ``K_min`` is the smallest
  capacity that lets every activity run and ``K_max`` is the peak demand of the
  earliest-start schedule that ignores resources. With ``RS = 1`` that schedule
  is feasible.

**Network.** Activities are indexed in a precedence-feasible order and arcs run
from lower to higher indices. As in ProGen:

1. The first ``S`` activities are the start activities and the last ``F`` the
   finish activities. Start activities get no predecessor and finish
   activities no successor.
2. Every other activity, in increasing order, gets one predecessor drawn
   uniformly from the earlier non-finish activities.
3. Every non-finish activity still without a successor gets one drawn
   uniformly from the later non-start activities.
4. Arcs drawn uniformly from the remaining admissible pairs are added until the
   network has its target number of arcs.

An arc is admissible when it is not redundant (no other path already leads
from its tail to its head), makes no existing arc redundant, and leaves every
activity at most ``max_degree`` predecessors and successors. A network that
leaves an activity of step 3 without an admissible successor, or does not end
with exactly the target number of arcs, is discarded and drawn again, up to 100
times. If no draw succeeds, the last one is kept, with the limits of step 3
relaxed for the activities that needed it.

**Requests.** As in ProGen, every activity first requests one resource drawn
uniformly; then (activity, resource) pairs not yet requested are drawn
uniformly, without replacement, until ``ceil(RF * J * K)`` pairs are requested.

**Differences from ProGen.** ProGen draws the numbers of start and finish
activities from ranges, runs step 3 in increasing order, and accepts an arc
count within a tolerance of the target. Here the numbers are fixed, step 3
runs from the last activity back to the first, which gives the late
activities, with few later activities to choose from, the first choice and so
fails far less often, and the arc count must match the target. ProGen lets the
minimum and maximum number of requests per activity vary; here they are one
and ``K``. There are no dummy source and sink activities and no nonrenewable
resources. :class:`MultiModeRcpsp` lists its own differences.

Activities are tasks ``a1``, ``a2``, ... and resources ``r1``, ``r2``, ...,
zero-padded to one width, following the PSPLIB numbering.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.benchmarks._common import (
    ceil_tolerant,
    check_int,
    check_int_range,
    check_number,
    check_objectives,
    instance_id,
    instance_metadata,
    numbered_ids,
    round_half_up,
)
from dispatchatlas.benchmarks.registry import FamilyInfo, FamilyRegistry
from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Dependency, Mode, Problem, Resource, Scalar, Task

__all__ = ["MultiModeRcpsp", "Rcpsp", "project_problem", "register"]

type ModeData = tuple[float, Sequence[float]]
"""A mode as ``(duration, demand on each resource)``."""

NETWORK_ATTEMPTS = 100
"""Networks drawn before the generator keeps one that misses its targets."""


def project_problem(
    modes: Sequence[Sequence[ModeData]],
    arcs: Iterable[tuple[int, int]],
    capacities: Sequence[float],
    *,
    activity_ids: Sequence[str] | None = None,
    problem_id: str = "project",
    objectives: Sequence[str] = ("makespan",),
    family: str | None = None,
    metadata: Mapping[str, Scalar] | None = None,
) -> Problem:
    """Build a project from activity modes, precedence arcs, and resource capacities.

    Args:
        modes: ``modes[i]`` lists activity ``i``'s modes as ``(duration,
            demands)``, where ``demands[k]`` is the amount of resource ``k``
            held while the activity runs.
        arcs: Precedence arcs ``(i, j)``: activity ``j`` starts after ``i``
            finishes.
        capacities: Capacity of each renewable resource ``r1``, ``r2``, ....
        activity_ids: Task id of each activity. Defaults to ``a1``, ``a2``, ...
        problem_id: Id of the problem.
        objectives: Objectives to optimise, primary first.
        family: Id of the family that generated the data, if any.
        metadata: Annotations stored on the problem.

    Returns:
        The problem.

    Raises:
        ModelError: If a demand vector or the id list has the wrong length, or
            an arc names an unknown activity.
    """
    count = len(modes)
    if count == 0:
        raise ModelError("a project needs at least one activity")
    ids = list(activity_ids) if activity_ids is not None else numbered_ids("a", count, start=1)
    if len(ids) != count:
        raise ModelError(f"activity_ids lists {len(ids)} ids for {count} activities")
    resources = numbered_ids("r", len(capacities), start=1)
    tasks = []
    for task_id, options in zip(ids, modes, strict=True):
        built = []
        for duration, demands in options:
            if len(demands) != len(resources):
                raise ModelError(
                    f"activity {task_id!r} lists {len(demands)} demands for "
                    f"{len(resources)} resources"
                )
            used = {
                resource: float(amount)
                for resource, amount in zip(resources, demands, strict=True)
                if amount
            }
            built.append(Mode(duration, used))
        tasks.append(Task(task_id, tuple(built)))
    dependencies = []
    for before, after in arcs:
        if not (0 <= before < count and 0 <= after < count):
            raise ModelError(f"arc ({before}, {after}) names an unknown activity")
        dependencies.append(Dependency(ids[before], ids[after]))
    return Problem(
        problem_id,
        tasks=tuple(tasks),
        resources=tuple(
            Resource(resource, capacity=capacity, kind="renewable")
            for resource, capacity in zip(resources, capacities, strict=True)
        ),
        dependencies=tuple(dependencies),
        objectives=tuple(objectives),
        family=family,
        metadata=metadata or {},
    )


@dataclass(frozen=True)
class Rcpsp:
    """Single-mode projects with ProGen's network complexity, resource factor, and strength.

    Draws, in order: the network (see the module description), the duration of
    every activity, the requests, and the demand of every request, activity by
    activity and resource by resource. Capacities follow from the drawn data,
    with ``K_min`` the largest demand on the resource.

    Attributes:
        activities: Number of activities ``J``.
        resources: Number of renewable resources ``K``.
        network_complexity: ``NC``. PSPLIB uses 1.5, 1.8, and 2.1.
        resource_factor: ``RF`` in ``[0, 1]``. PSPLIB uses 0.25, 0.5, 0.75,
            and 1.
        resource_strength: ``RS`` in ``[0, 1]``.
        min_duration: Shortest duration.
        max_duration: Longest duration.
        min_demand: Smallest demand of a request.
        max_demand: Largest demand of a request.
        start_activities: Number ``S`` of activities without predecessors.
        finish_activities: Number ``F`` of activities without successors.
        max_degree: Most predecessors, and most successors, of an activity.
        objectives: Objectives the problems declare, primary first.
    """

    activities: int = 30
    resources: int = 4
    network_complexity: float = 1.5
    resource_factor: float = 0.5
    resource_strength: float = 0.5
    min_duration: int = 1
    max_duration: int = 10
    min_demand: int = 1
    max_demand: int = 10
    start_activities: int = 3
    finish_activities: int = 3
    max_degree: int = 3
    objectives: tuple[str, ...] = ("makespan",)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="rcpsp",
        name="Resource-constrained project",
        problem_class="rcpsp",
        summary="Activities with precedence arcs share renewable resources, generated "
        "with the network complexity, resource factor, and resource strength of "
        "Kolisch, Sprecher and Drexl (1995).",
        references=("kolisch1995characterization", "kolisch1997psplib"),
    )
    _tags: ClassVar[Mapping[str, str]] = {
        "min_duration": "dmin",
        "max_duration": "dmax",
        "min_demand": "qmin",
        "max_demand": "qmax",
        "start_activities": "start",
        "finish_activities": "finish",
        "max_degree": "deg",
        "objectives": "",
    }

    def __post_init__(self) -> None:
        _check_network(self)
        check_int_range("min_duration", self.min_duration, "max_duration", self.max_duration)
        check_int_range("min_demand", self.min_demand, "max_demand", self.max_demand)
        object.__setattr__(self, "objectives", check_objectives(self.objectives, due_dates=False))

    def generate(self, seed: int = 0) -> Problem:
        """Return the project for ``seed``."""
        problem_id = instance_id(self, _headline(self), seed, self._tags)
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        arcs = _network(rng, self)
        durations = [
            rng.randint(self.min_duration, self.max_duration) for _ in range(self.activities)
        ]
        requests = _requests(rng, self.activities, self.resources, self.resource_factor)
        modes: list[list[ModeData]] = []
        for duration, requested in zip(durations, requests, strict=True):
            demands = [0.0] * self.resources
            for resource in requested:
                demands[resource] = float(rng.randint(self.min_demand, self.max_demand))
            modes.append([(float(duration), demands)])
        return project_problem(
            modes,
            arcs,
            _capacities(modes, arcs, self.resources, self.resource_strength),
            problem_id=problem_id,
            objectives=self.objectives,
            family=self.info.id,
            metadata=metadata,
        )


@dataclass(frozen=True)
class MultiModeRcpsp:
    """Multi-mode projects whose modes trade duration against resource demand.

    The network, the requests, and the resource strength follow :class:`Rcpsp`.
    An activity's modes all use the resources the activity requests and are
    listed from shortest to longest: durations strictly increase, the demand on
    one requested resource strictly decreases, and the demands on the others
    never increase, so no mode is at least as long and as demanding as another.

    Draws, in order: the network; the number of modes and their distinct
    durations, activity by activity; the requests; then, activity by activity,
    the resource whose demand strictly decreases, and the demands of each
    requested resource in resource order, drawn distinct for that resource and
    freely for the others, and sorted from largest to smallest.

    Unlike ProGen, whose modes may request different resources and whose
    demands may also stay constant as the duration grows, every mode here
    requests the same resources. ``K_min`` is the largest demand of any mode,
    because the model requires every mode to fit its resources. ``K_max`` is
    the peak of the earliest-start schedule with every activity in its
    shortest mode, which is also its most demanding one.

    Attributes:
        activities: Number of activities ``J``.
        resources: Number of renewable resources ``K``.
        network_complexity: ``NC``.
        resource_factor: ``RF`` in ``(0, 1]``.
        resource_strength: ``RS`` in ``[0, 1]``.
        min_modes: Fewest modes of an activity.
        max_modes: Most modes of an activity.
        min_duration: Shortest duration.
        max_duration: Longest duration.
        min_demand: Smallest demand of a request.
        max_demand: Largest demand of a request.
        start_activities: Number ``S`` of activities without predecessors.
        finish_activities: Number ``F`` of activities without successors.
        max_degree: Most predecessors, and most successors, of an activity.
        objectives: Objectives the problems declare, primary first.
    """

    activities: int = 20
    resources: int = 2
    network_complexity: float = 1.5
    resource_factor: float = 0.5
    resource_strength: float = 0.5
    min_modes: int = 2
    max_modes: int = 3
    min_duration: int = 1
    max_duration: int = 10
    min_demand: int = 1
    max_demand: int = 10
    start_activities: int = 3
    finish_activities: int = 3
    max_degree: int = 3
    objectives: tuple[str, ...] = ("makespan",)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="multi-mode-rcpsp",
        name="Multi-mode resource-constrained project",
        problem_class="multi-mode-rcpsp",
        summary="Activities with precedence arcs choose among modes that trade "
        "duration against renewable-resource demand, on networks generated as in "
        "Kolisch, Sprecher and Drexl (1995).",
        references=("kolisch1995characterization", "kolisch1997psplib"),
    )
    _tags: ClassVar[Mapping[str, str]] = {
        "min_modes": "mmin",
        "max_modes": "mmax",
        "min_duration": "dmin",
        "max_duration": "dmax",
        "min_demand": "qmin",
        "max_demand": "qmax",
        "start_activities": "start",
        "finish_activities": "finish",
        "max_degree": "deg",
        "objectives": "",
    }

    def __post_init__(self) -> None:
        _check_network(self)
        check_int_range("min_modes", self.min_modes, "max_modes", self.max_modes)
        check_int_range("min_duration", self.min_duration, "max_duration", self.max_duration)
        check_int_range("min_demand", self.min_demand, "max_demand", self.max_demand)
        if self.max_duration - self.min_duration + 1 < self.max_modes:
            raise ValueError("the duration range must hold max_modes distinct values")
        if self.max_demand - self.min_demand + 1 < self.max_modes:
            raise ValueError("the demand range must hold max_modes distinct values")
        if self.resource_factor <= 0.0:
            raise ValueError("resource_factor must be positive, so that modes differ in demand")
        object.__setattr__(self, "objectives", check_objectives(self.objectives, due_dates=False))

    def generate(self, seed: int = 0) -> Problem:
        """Return the multi-mode project for ``seed``."""
        problem_id = instance_id(self, _headline(self), seed, self._tags)
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        arcs = _network(rng, self)
        durations = [
            sorted(
                rng.sample(
                    range(self.min_duration, self.max_duration + 1),
                    rng.randint(self.min_modes, self.max_modes),
                )
            )
            for _ in range(self.activities)
        ]
        requests = _requests(rng, self.activities, self.resources, self.resource_factor)
        levels = range(self.min_demand, self.max_demand + 1)
        modes: list[list[ModeData]] = []
        for activity_durations, requested in zip(durations, requests, strict=True):
            count = len(activity_durations)
            decreasing = rng.choice(requested)
            columns = {
                resource: sorted(
                    rng.sample(levels, count)
                    if resource == decreasing
                    else [rng.randint(self.min_demand, self.max_demand) for _ in range(count)],
                    reverse=True,
                )
                for resource in requested
            }
            activity: list[ModeData] = []
            for position, duration in enumerate(activity_durations):
                demands = [0.0] * self.resources
                for resource, column in columns.items():
                    demands[resource] = float(column[position])
                activity.append((float(duration), demands))
            modes.append(activity)
        return project_problem(
            modes,
            arcs,
            _capacities(modes, arcs, self.resources, self.resource_strength),
            problem_id=problem_id,
            objectives=self.objectives,
            family=self.info.id,
            metadata=metadata,
        )


def register(registry: FamilyRegistry) -> None:
    """Add the project-scheduling families to ``registry``."""
    registry.register(Rcpsp)
    registry.register(MultiModeRcpsp)


# ------------------------------------------------------------------ helpers


def _check_network(generator: Rcpsp | MultiModeRcpsp) -> None:
    """Check the network and resource parameters shared by both families."""
    check_int("activities", generator.activities)
    check_int("resources", generator.resources)
    check_int("start_activities", generator.start_activities)
    check_int("finish_activities", generator.finish_activities)
    check_int("max_degree", generator.max_degree)
    if generator.start_activities + generator.finish_activities > generator.activities:
        raise ValueError(
            f"start_activities + finish_activities cannot exceed activities "
            f"({generator.activities})"
        )
    complexity = check_number("network_complexity", generator.network_complexity, positive=True)
    object.__setattr__(generator, "network_complexity", complexity)
    for name in ("resource_factor", "resource_strength"):
        object.__setattr__(
            generator, name, check_number(name, getattr(generator, name), maximum=1.0)
        )


def _headline(generator: Rcpsp | MultiModeRcpsp) -> str:
    """Return the size and ProGen parameters that open every problem id."""
    return (
        f"{generator.activities}x{generator.resources}"
        f"-nc{generator.network_complexity:g}"
        f"-rf{generator.resource_factor:g}"
        f"-rs{generator.resource_strength:g}"
    )


def _network(rng: random.Random, generator: Rcpsp | MultiModeRcpsp) -> list[tuple[int, int]]:
    """Draw networks until one meets its limits and arc target; see the module docstring."""
    count = generator.activities
    target = ceil_tolerant(generator.network_complexity * (count + 2))
    target -= generator.start_activities + generator.finish_activities
    arcs: list[tuple[int, int]] = []
    for _ in range(NETWORK_ATTEMPTS):
        arcs, within_limits = _draw_network(rng, generator, target)
        if within_limits and len(arcs) == target:
            break
    return arcs


def _draw_network(
    rng: random.Random, generator: Rcpsp | MultiModeRcpsp, target: int
) -> tuple[list[tuple[int, int]], bool]:
    """Draw one network; return its arcs and whether steps 2 and 3 kept every limit."""
    count = generator.activities
    starts = generator.start_activities
    first_finish = count - generator.finish_activities
    graph = _Network(count, generator.max_degree)
    within_limits = True
    for after in range(starts, count):
        options = range(min(after, first_finish))
        candidates = [before for before in options if graph.admissible(before, after)]
        if not candidates:
            within_limits = False
            candidates = list(options)
        graph.add(rng.choice(candidates), after)
    for before in range(first_finish - 1, -1, -1):
        if graph.successors[before]:
            continue
        options = range(max(before + 1, starts), count)
        candidates = [after for after in options if graph.admissible(before, after)]
        if not candidates:
            within_limits = False
            relaxed = [after for after in options if graph.admissible(before, after, degree=False)]
            candidates = relaxed or list(options)
        graph.add(before, rng.choice(candidates))
    pool: list[tuple[int, int]] | None = None
    while len(graph.arcs) < target:
        if pool is None:
            # Draw uniform pairs while admissible arcs are common, then switch
            # to an explicit pool: admissibility only shrinks as arcs are added.
            arc = _sample_arc(rng, graph, starts, first_finish, attempts=4 * count)
            if arc is not None:
                graph.add(*arc)
                continue
            pool = [
                (before, after)
                for before in range(first_finish)
                for after in range(max(before + 1, starts), count)
                if graph.admissible(before, after)
            ]
        if not pool:
            break
        index = rng.randrange(len(pool))
        before, after = pool[index]
        pool[index] = pool[-1]
        pool.pop()
        if graph.admissible(before, after):
            graph.add(before, after)
    return graph.arcs, within_limits


def _sample_arc(
    rng: random.Random, graph: _Network, starts: int, first_finish: int, attempts: int
) -> tuple[int, int] | None:
    """Draw pairs uniformly among non-finish tails and later non-start heads."""
    for _ in range(attempts):
        before = rng.randrange(first_finish)
        after = rng.randrange(starts, graph.count)
        if before < after and graph.admissible(before, after):
            return before, after
    return None


class _Network:
    """An acyclic activity network with reachability kept as bit sets."""

    def __init__(self, count: int, max_degree: int) -> None:
        self.count = count
        self.max_degree = max_degree
        self.arcs: list[tuple[int, int]] = []
        self.successors = [0] * count
        self.predecessor_count = [0] * count
        self.successor_count = [0] * count
        self.descendants = [0] * count
        self.ancestors = [0] * count

    def admissible(self, before: int, after: int, *, degree: bool = True) -> bool:
        """Return whether arc ``before -> after`` keeps the network non-redundant."""
        if self.successors[before] >> after & 1:
            return False
        if degree and (
            self.successor_count[before] >= self.max_degree
            or self.predecessor_count[after] >= self.max_degree
        ):
            return False
        if self.descendants[before] >> after & 1:
            return False
        # The new arc would make an existing arc (a, b) redundant when a leads
        # to ``before`` and ``after`` leads to b.
        heads = self.descendants[after] | 1 << after
        return not any(
            self.successors[tail] & heads for tail in _bits(self.ancestors[before] | 1 << before)
        )

    def add(self, before: int, after: int) -> None:
        """Add arc ``before -> after`` and update reachability."""
        self.arcs.append((before, after))
        self.successors[before] |= 1 << after
        self.successor_count[before] += 1
        self.predecessor_count[after] += 1
        heads = self.descendants[after] | 1 << after
        tails = self.ancestors[before] | 1 << before
        for tail in _bits(tails):
            self.descendants[tail] |= heads
        for head in _bits(heads):
            self.ancestors[head] |= tails


def _bits(value: int) -> list[int]:
    """Return the positions of the set bits of ``value``, lowest first."""
    positions = []
    while value:
        low = value & -value
        positions.append(low.bit_length() - 1)
        value ^= low
    return positions


def _requests(
    rng: random.Random, activities: int, resources: int, factor: float
) -> list[list[int]]:
    """Return the resources each activity requests, following ProGen's two steps.

    Every activity first requests one resource drawn uniformly. Then pairs of
    an activity and a resource it does not request yet are drawn uniformly
    without replacement until ``ceil(factor * activities * resources)`` pairs
    are requested. A factor of 0 requests nothing.
    """
    if factor <= 0.0:
        return [[] for _ in range(activities)]
    chosen = [{rng.randrange(resources)} for _ in range(activities)]
    missing = ceil_tolerant(factor * activities * resources) - activities
    if missing > 0:
        free = [
            (activity, resource)
            for activity in range(activities)
            for resource in range(resources)
            if resource not in chosen[activity]
        ]
        for activity, resource in rng.sample(free, min(missing, len(free))):
            chosen[activity].add(resource)
    return [sorted(items) for items in chosen]


def _capacities(
    modes: Sequence[Sequence[ModeData]],
    arcs: Sequence[tuple[int, int]],
    resources: int,
    strength: float,
) -> list[float]:
    """Return ``K_min + round(strength * (K_max - K_min))`` for every resource.

    ``K_min`` is the largest demand of any mode. ``K_max`` is the peak demand
    of the earliest-start schedule that ignores resources, with every activity
    in its first mode. A resource nobody requests gets capacity 1.
    """
    predecessors: list[list[int]] = [[] for _ in modes]
    for before, after in arcs:
        predecessors[after].append(before)
    starts = [0.0] * len(modes)
    for activity in range(len(modes)):  # arcs run from lower to higher indices
        starts[activity] = max(
            (starts[before] + modes[before][0][0] for before in predecessors[activity]),
            default=0.0,
        )
    capacities = []
    for resource in range(resources):
        k_min = max(demands[resource] for options in modes for _, demands in options)
        events = []
        for activity, options in enumerate(modes):
            duration, demands = options[0]
            if demands[resource] and duration > 0:
                events.append((starts[activity], demands[resource]))
                events.append((starts[activity] + duration, -demands[resource]))
        k_max = level = 0.0
        for _, change in sorted(events):  # at equal times, releases sort first
            level += change
            k_max = max(k_max, level)
        capacity = k_min + round_half_up(strength * (k_max - k_min))
        capacities.append(max(capacity, 1.0))
    return capacities
