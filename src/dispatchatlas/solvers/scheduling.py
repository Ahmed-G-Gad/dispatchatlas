"""The schedule builder shared by every order-based solver.

Most heuristics and metaheuristics search over **task orders** (priority lists)
and, for tasks with several modes, over **mode choices**. :class:`Decoder`
turns an order into a schedule with the serial schedule-generation scheme
(Kolisch 1996, doi:10.1016/0377-2217(95)00357-6): it takes tasks one at a
time, in the order given as far as precedence allows, and starts each at the
earliest time its predecessors, data transfers, release time, and resources
permit.

Details that make every decoded schedule feasible for release times,
precedence with lags and transfers, capacities, and setup times:

- **Precedence repair.** At each step the decoder takes the earliest-listed
  task whose predecessors are all scheduled, so any permutation decodes.
  :attr:`Decoder.repairs` counts these passes.
- **Insertion.** A task may fill an idle gap left earlier on its resources.
  Resources with setup times, and every resource of a permutation problem,
  append only, so their processing sequence equals the decoding sequence.
- **Mode choice.** Without explicit modes, or where a task's mode entry is
  negative, a task takes the mode that finishes it earliest, the lowest mode
  index on ties.
- **Permutation problems.** The group (job) sequence is the order in which
  groups first appear in the list, and each group's tasks are decoded
  together, which yields a permutation schedule.
- **Zero-duration tasks** occupy an instant that no task on an exclusive
  resource may run across, and hold nothing on other resources, as the data
  model defines. On append-only resources they also take their place in the
  sequence.

Hard deadlines are not enforced here. The evaluator prices their excess.
"""

from __future__ import annotations

import heapq
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Problem
from dispatchatlas.core.schedule import Assignment, Schedule

__all__ = ["Builder", "Decoded", "Decoder", "precedence_feasible"]

EPSILON = 1e-9
"""Tolerance of the builder's time and capacity comparisons."""


@dataclass(frozen=True, slots=True)
class Decoded:
    """A decoded schedule in array form, indexed by task index.

    Attributes:
        sequence: Task indices in the order they were placed.
        modes: Mode index of each task. ``-1`` for tasks not decoded.
        starts: Start time of each task.
        ends: Finish time of each task.
    """

    sequence: tuple[int, ...]
    modes: tuple[int, ...]
    starts: tuple[float, ...]
    ends: tuple[float, ...]

    @property
    def makespan(self) -> float:
        """Latest finish time over the decoded tasks."""
        return max((self.ends[i] for i in self.sequence), default=0.0)

    def schedule(self, problem: Problem) -> Schedule:
        """Return the decoded tasks as a :class:`~dispatchatlas.core.schedule.Schedule`."""
        tasks = problem.tasks
        return Schedule.of(
            Assignment(tasks[i].id, self.modes[i], self.starts[i], self.ends[i])
            for i in self.sequence
        )


def precedence_feasible(problem: Problem, order: Sequence[int]) -> list[int]:
    """Return the precedence-feasible sequence closest to a priority list.

    At every step the earliest-listed task whose listed predecessors are all
    placed comes next. Tasks absent from ``order`` are left out, so ``order``
    may cover a predecessor-closed subset of the tasks.

    Args:
        problem: The problem.
        order: Task indices, each at most once.

    Returns:
        The repaired sequence.

    Raises:
        ModelError: If ``order`` repeats a task, names an unknown index, or
            lists a task without one of its predecessors.
    """
    count = problem.size
    position = [-1] * count
    for rank, task in enumerate(order):
        if not 0 <= task < count:
            raise ModelError(f"task index {task} is out of range")
        if position[task] >= 0:
            raise ModelError(f"task index {task} appears twice in the order")
        position[task] = rank
    predecessors = problem.predecessors
    successors = problem.successors
    waiting = [0] * count
    ready: list[tuple[int, int]] = []
    for task in order:
        for before in predecessors[task]:
            if position[before] < 0:
                raise ModelError(
                    f"task {problem.tasks[task].id!r} is listed without its predecessor "
                    f"{problem.tasks[before].id!r}"
                )
        waiting[task] = len(predecessors[task])
        if waiting[task] == 0:
            ready.append((position[task], task))
    heapq.heapify(ready)
    sequence: list[int] = []
    while ready:
        _, task = heapq.heappop(ready)
        sequence.append(task)
        for following in successors[task]:
            if position[following] >= 0:
                waiting[following] -= 1
                if waiting[following] == 0:
                    heapq.heappush(ready, (position[following], following))
    return sequence


class _Gaps:
    """Busy intervals of a resource where every task holds its full capacity."""

    __slots__ = ("ends", "starts")

    def __init__(self) -> None:
        self.starts: list[float] = []
        self.ends: list[float] = []

    def fit(self, ready: float, duration: float) -> float:
        index = bisect_right(self.ends, ready + EPSILON)
        time = ready
        starts, ends = self.starts, self.ends
        while index < len(starts):
            if time + duration <= starts[index] + EPSILON:
                return time
            time = max(time, ends[index])
            index += 1
        return time

    def reserve(self, start: float, end: float) -> None:
        index = bisect_right(self.starts, start)
        # An instant sorts before a busy interval that starts with it, which
        # keeps the ends sorted too.
        while index > 0 and self.starts[index - 1] == start and self.ends[index - 1] > end:
            index -= 1
        self.starts.insert(index, start)
        self.ends.insert(index, end)


class _Profile:
    """Step function of the units in use on a cumulative resource."""

    __slots__ = ("capacity", "levels", "times")

    def __init__(self, capacity: float) -> None:
        self.capacity = capacity
        self.times: list[float] = [0.0]
        self.levels: list[float] = [0.0]

    def fit(self, ready: float, duration: float, demand: float) -> float:
        times, levels = self.times, self.levels
        limit = self.capacity + EPSILON - demand
        time = ready
        while True:
            index = bisect_right(times, time) - 1
            end = time + duration
            conflict = -1
            probe = index
            while probe < len(times) and (probe == index or times[probe] < end - EPSILON):
                if levels[probe] > limit:
                    conflict = probe
                    break
                probe += 1
            if conflict < 0:
                return time
            # The conflicting segment ends at the next breakpoint; the last
            # segment is always empty, so a later start always exists.
            time = times[conflict + 1]

    def reserve(self, start: float, end: float, demand: float) -> None:
        if end <= start:
            return
        for moment in (start, end):
            index = bisect_right(self.times, moment) - 1
            if self.times[index] != moment:
                self.times.insert(index + 1, moment)
                self.levels.insert(index + 1, self.levels[index])
        first = bisect_right(self.times, start) - 1
        last = bisect_right(self.times, end) - 1
        for index in range(first, last):
            self.levels[index] += demand


class Decoder:
    """Builds schedules from task orders and optional mode choices.

    A decoder precomputes everything that does not depend on the order, so
    create one per problem and reuse it for every evaluation. Constructive
    heuristics that place tasks one at a time use :meth:`builder` instead.

    Args:
        problem: The problem to decode for.
        insertion: Let tasks fill idle gaps on their resources. Ignored on
            resources with setup times and in permutation problems.

    Attributes:
        repairs: How many times :meth:`decode` has called
            :func:`precedence_feasible`, which it does once per call. Tasks
            placed through :meth:`builder` are never repaired.
    """

    def __init__(self, problem: Problem, *, insertion: bool = True) -> None:
        self.problem = problem
        self.insertion = insertion and not problem.permutation
        self.repairs = 0
        resource_index = problem.resource_index
        self.capacity = [resource.capacity for resource in problem.resources]
        setup_resources = problem.setups.resources if problem.setups is not None else frozenset()
        self.setups = problem.setups.times if problem.setups is not None else {}
        self.resource_ids = [resource.id for resource in problem.resources]
        self.task_ids = [task.id for task in problem.tasks]
        self.release = [task.release for task in problem.tasks]
        # Per task and mode: (duration, ((resource index, units), ...), location index).
        self.modes: list[list[tuple[float, tuple[tuple[int, float], ...], int]]] = []
        for task in problem.tasks:
            options = []
            for mode in task.modes:
                demands = tuple(
                    (resource_index[resource_id], units)
                    for resource_id, units in mode.demands.items()
                )
                location = resource_index[mode.location] if mode.location is not None else -1
                options.append((mode.duration, demands, location))
            self.modes.append(options)
        # Exclusive resources use the fast interval structure; the rest use a
        # capacity profile. Machines with setup times, and every exclusive
        # resource when insertion is off, append only.
        exclusive = problem.exclusive_resources
        full_units = [resource.id in exclusive for resource in problem.resources]
        self.interval = full_units
        self.append_only = [
            full_units[r] and (not self.insertion or resource.id in setup_resources)
            for r, resource in enumerate(problem.resources)
        ]
        task_index = problem.task_index
        self.incoming: list[list[tuple[int, float, int]]] = []
        self.dependencies = problem.dependencies
        dependency_number = {id(d): n for n, d in enumerate(problem.dependencies)}
        for items in problem.incoming:
            self.incoming.append(
                [(task_index[d.before], d.lag, dependency_number[id(d)]) for d in items]
            )
        self.has_network = "network" in problem.features
        self._transfer_cache: dict[tuple[int, int, int], float] = {}
        self.group_of = [
            task.group if task.group is not None else task.id for task in problem.tasks
        ]

    def builder(self) -> Builder:
        """Return an empty partial schedule to place tasks into one by one."""
        return Builder(self)

    def decode(self, order: Sequence[int], modes: Sequence[int] | None = None) -> Decoded:
        """Decode a priority list into a schedule.

        Args:
            order: Task indices in priority order. Covering every task decodes
                the whole problem. A predecessor-closed subset decodes a
                partial schedule, as used by insertion heuristics.
            modes: Mode index per task index. ``None``, or a negative entry
                for a task, picks the mode that finishes the task earliest
                when it is placed.

        Returns:
            The decoded schedule.
        """
        problem = self.problem
        if problem.permutation:
            order = self.group_major(order)
        builder = Builder(self)
        self.repairs += 1
        for task in precedence_feasible(problem, order):
            mode = -1 if modes is None else modes[task]
            if mode < 0:
                start, _, mode = builder.best(task)
            else:
                start, _ = builder.earliest(task, mode)
            builder.place(task, mode, start)
        return builder.decoded()

    def schedule(self, order: Sequence[int], modes: Sequence[int] | None = None) -> Schedule:
        """Decode ``order`` and return the result as a schedule."""
        return self.decode(order, modes).schedule(self.problem)

    def group_major(self, order: Sequence[int]) -> list[int]:
        """Reorder so each group's tasks follow the group's first appearance."""
        rank: dict[str, int] = {}
        for task in order:
            rank.setdefault(self.group_of[task], len(rank))
        position = {task: n for n, task in enumerate(order)}
        topological = {task: n for n, task in enumerate(self.problem.topological_order)}
        return sorted(
            order,
            key=lambda task: (rank[self.group_of[task]], topological[task], position[task]),
        )

    def transfer(self, number: int, source: int, target: int) -> float:
        """Return the transfer time of dependency ``number`` between two resources."""
        if source < 0 or target < 0 or source == target:
            return 0.0
        key = (number, source, target)
        cached = self._transfer_cache.get(key)
        if cached is None:
            cached = self.problem.transfer_time(
                self.dependencies[number], self.resource_ids[source], self.resource_ids[target]
            )
            self._transfer_cache[key] = cached
        return cached


class Builder:
    """A partial schedule that grows one task at a time.

    Obtain one from :meth:`Decoder.builder`. A task can be placed once all of
    its predecessors are placed.
    """

    __slots__ = (
        "_decoder",
        "_gaps",
        "_last_end",
        "_last_task",
        "_profiles",
        "ends",
        "modes",
        "sequence",
        "starts",
    )

    def __init__(self, decoder: Decoder) -> None:
        count = decoder.problem.size
        self._decoder = decoder
        self.sequence: list[int] = []
        self.modes = [-1] * count
        self.starts = [0.0] * count
        self.ends = [0.0] * count
        self._gaps: dict[int, _Gaps] = {}
        self._profiles: dict[int, _Profile] = {}
        self._last_end: dict[int, float] = {}
        self._last_task: dict[int, int] = {}

    def placed(self, task: int) -> bool:
        """Return whether ``task`` has been placed."""
        return self.modes[task] >= 0

    def eligible(self, task: int) -> bool:
        """Return whether ``task`` is unplaced and all its predecessors are placed."""
        if self.modes[task] >= 0:
            return False
        return all(self.modes[before] >= 0 for before, _, _ in self._decoder.incoming[task])

    def earliest(self, task: int, mode: int) -> tuple[float, float]:
        """Return the earliest ``(start, finish)`` of ``task`` in ``mode`` now.

        Raises:
            ModelError: If the mode is invalid or a predecessor is unplaced.
        """
        decoder = self._decoder
        options = decoder.modes[task]
        if not 0 <= mode < len(options):
            raise ModelError(f"mode {mode} is invalid for task {decoder.task_ids[task]!r}")
        duration, demands, location = options[mode]
        ready = decoder.release[task]
        for before, lag, number in decoder.incoming[task]:
            if self.modes[before] < 0:
                raise ModelError(
                    f"task {decoder.task_ids[task]!r} placed before its predecessor "
                    f"{decoder.task_ids[before]!r}"
                )
            time = self.ends[before] + lag
            if decoder.has_network:
                source = decoder.modes[before][self.modes[before]][2]
                time += decoder.transfer(number, source, location)
            ready = max(ready, time)
        start = self._fit(task, ready, duration, demands)
        return start, start + duration

    def best(self, task: int, allowed: Sequence[int] | None = None) -> tuple[float, float, int]:
        """Return ``(start, finish, mode)`` of the mode that finishes ``task`` first.

        Args:
            task: Task index.
            allowed: Candidate modes. Defaults to every mode.
        """
        candidates = range(len(self._decoder.modes[task])) if allowed is None else allowed
        best: tuple[float, float, int] | None = None
        for mode in candidates:
            start, finish = self.earliest(task, mode)
            if best is None or finish < best[1] - EPSILON:
                best = (start, finish, mode)
        if best is None:
            raise ModelError("no candidate mode given")
        return best

    def place(self, task: int, mode: int, start: float | None = None) -> tuple[float, float]:
        """Commit ``task`` in ``mode`` at ``start``, or as early as possible.

        Returns:
            The ``(start, finish)`` of the placed task.
        """
        if self.modes[task] >= 0:
            raise ModelError(f"task {self._decoder.task_ids[task]!r} is already placed")
        if start is None:
            start, _ = self.earliest(task, mode)
        duration, demands, _ = self._decoder.modes[task][mode]
        finish = start + duration
        self.modes[task] = mode
        self.starts[task] = start
        self.ends[task] = finish
        self.sequence.append(task)
        decoder = self._decoder
        for resource, units in demands:
            if decoder.append_only[resource]:
                self._last_end[resource] = finish
                self._last_task[resource] = task
            elif decoder.interval[resource]:
                self._gaps.setdefault(resource, _Gaps()).reserve(start, finish)
            elif finish > start:
                self._profile(resource).reserve(start, finish, units)
        return start, finish

    def copy(self) -> Builder:
        """Return an independent copy, for look-ahead heuristics."""
        clone = Builder.__new__(Builder)
        clone._decoder = self._decoder
        clone.sequence = list(self.sequence)
        clone.modes = list(self.modes)
        clone.starts = list(self.starts)
        clone.ends = list(self.ends)
        clone._gaps = {}
        for resource, gaps in self._gaps.items():
            copy = _Gaps()
            copy.starts, copy.ends = list(gaps.starts), list(gaps.ends)
            clone._gaps[resource] = copy
        clone._profiles = {}
        for resource, profile in self._profiles.items():
            copied = _Profile(profile.capacity)
            copied.times, copied.levels = list(profile.times), list(profile.levels)
            clone._profiles[resource] = copied
        clone._last_end = dict(self._last_end)
        clone._last_task = dict(self._last_task)
        return clone

    @property
    def makespan(self) -> float:
        """Latest finish time of the placed tasks."""
        return max((self.ends[i] for i in self.sequence), default=0.0)

    def decoded(self) -> Decoded:
        """Return the placed tasks as a :class:`Decoded` schedule."""
        return Decoded(
            tuple(self.sequence), tuple(self.modes), tuple(self.starts), tuple(self.ends)
        )

    def _profile(self, resource: int) -> _Profile:
        profile = self._profiles.get(resource)
        if profile is None:
            profile = self._profiles[resource] = _Profile(self._decoder.capacity[resource])
        return profile

    def _fit(
        self, task: int, ready: float, duration: float, demands: tuple[tuple[int, float], ...]
    ) -> float:
        if not demands:
            return ready
        if len(demands) == 1:
            resource, units = demands[0]
            return self._earliest_on(task, resource, units, ready, duration)
        time = ready
        moved = True
        while moved:
            moved = False
            for resource, units in demands:
                earliest = self._earliest_on(task, resource, units, time, duration)
                if earliest > time + EPSILON:
                    time = earliest
                    moved = True
        return time

    def _earliest_on(
        self, task: int, resource: int, units: float, time: float, duration: float
    ) -> float:
        decoder = self._decoder
        if decoder.append_only[resource]:
            previous = self._last_task.get(resource)
            setup = decoder.setups.get(
                (
                    decoder.resource_ids[resource],
                    None if previous is None else decoder.task_ids[previous],
                    decoder.task_ids[task],
                ),
                0.0,
            )
            return max(time, self._last_end.get(resource, 0.0) + setup)
        if decoder.interval[resource]:
            return self._gaps.setdefault(resource, _Gaps()).fit(time, duration)
        if duration <= 0.0:
            return time
        return self._profile(resource).fit(time, duration, units)
