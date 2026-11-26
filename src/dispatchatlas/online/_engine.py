"""Placement routines shared by the runner and the planning context.

Every routine places tasks into a :class:`~dispatchatlas.solvers.scheduling.Builder`
that already holds the committed tasks, never before ``now``, and only in
modes whose resources are all up. One call of ``Builder.earliest`` is one
**placement**, the unit in which planning is charged.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Collection, Mapping, Sequence

from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Problem
from dispatchatlas.solvers.scheduling import EPSILON, Builder, Decoder

TIME_TOLERANCE = 1e-9
"""Times within this of each other are one instant: the runner batches events by it and the
recovery measures match forecasts to events by it."""

type Allowed = Callable[[int], Sequence[int]]


class Model:
    """What the runner and the planners share about the run's problem.

    Args:
        problem: The run's problem: every task of every workflow of the trace.
        priority: A value per task; higher goes first among tasks that fall
            due at one instant.
        arrivals: Arrival time of each workflow, in workflow order.
    """

    def __init__(
        self, problem: Problem, priority: Sequence[float], arrivals: Sequence[float]
    ) -> None:
        self.problem = problem
        self.decoder = Decoder(problem)
        self.size = problem.size
        self.predecessors = problem.predecessors
        self.successors = problem.successors
        self.priority = tuple(float(value) for value in priority)
        if len(self.priority) != problem.size:
            raise ModelError("the priority needs one value per task")
        groups = problem.groups
        self.workflows = tuple(groups)
        self.workflow_index = {name: n for n, name in enumerate(self.workflows)}
        self.members = tuple(groups[name] for name in self.workflows)
        self.workflow_of = [0] * problem.size
        for n, members in enumerate(self.members):
            for task in members:
                self.workflow_of[task] = n
        self.arrivals = tuple(arrivals)
        self.mode_nodes = tuple(
            tuple(frozenset(mode.demands) for mode in task.modes) for task in problem.tasks
        )
        self.durations = tuple(
            tuple(mode.duration for mode in task.modes) for task in problem.tasks
        )

    def allowed(self, task: int, down: Collection[str]) -> tuple[int, ...]:
        """Return the modes of ``task`` whose resources are all up."""
        if not down:
            return tuple(range(len(self.mode_nodes[task])))
        return tuple(m for m, nodes in enumerate(self.mode_nodes[task]) if nodes.isdisjoint(down))

    def key(self, task: int, due: float) -> tuple[float, float, int]:
        """The dispatch key of a task due at ``due``: earliest due, then priority, then index."""
        return (due, -self.priority[task], task)


def earliest_finish(
    builder: Builder, task: int, allowed: Sequence[int], now: float
) -> tuple[float, float, int]:
    """Return ``(start, finish, mode)`` of the allowed mode that finishes ``task`` first.

    Ties go to the lowest mode index, as in ``Builder.best``. Costs one
    placement per allowed mode.
    """
    best: tuple[float, float, int] | None = None
    for mode in allowed:
        start, finish = builder.earliest(task, mode, not_before=now)
        if best is None or finish < best[1] - EPSILON:
            best = (start, finish, mode)
    if best is None:
        raise ModelError("no allowed mode")
    return best


def dispatch_rollout(
    model: Model,
    builder: Builder,
    tasks: Collection[int],
    ready_since: Mapping[int, float],
    allowed: Allowed,
    now: float,
) -> list[tuple[int, int]]:
    """Place ``tasks`` by the earliest-finish dispatch rule; return ``(task, mode)`` pairs.

    Tasks are decided in the order they fall due, as the dispatcher decides
    them: a ready task at the instant in ``ready_since``, any other one when
    its last predecessor finishes in the rollout; ties by priority, then by
    index. Each takes the allowed mode that finishes it earliest. ``tasks``
    must contain every unplaced predecessor of each of its members.
    """
    waiting: dict[int, int] = {}
    heap: list[tuple[tuple[float, float, int], int]] = []
    for task in tasks:
        left = sum(1 for before in model.predecessors[task] if not builder.placed(before))
        waiting[task] = left
        if left == 0:
            due = ready_since.get(task)
            if due is None:
                due = max((builder.ends[b] for b in model.predecessors[task]), default=now)
            heap.append((model.key(task, due), task))
    heapq.heapify(heap)
    placed: list[tuple[int, int]] = []
    ends = builder.ends
    while heap:
        _, task = heapq.heappop(heap)
        start, _, mode = earliest_finish(builder, task, allowed(task), now)
        builder.place(task, mode, start)
        placed.append((task, mode))
        for after in model.successors[task]:
            if after in waiting:
                waiting[after] -= 1
                if waiting[after] == 0:
                    due = max(ends[before] for before in model.predecessors[after])
                    heapq.heappush(heap, (model.key(after, due), after))
    if len(placed) != len(waiting):
        raise ModelError("a planned task has a predecessor that is neither placed nor planned")
    return placed


def place_in_order(
    model: Model,
    builder: Builder,
    order: Sequence[int],
    modes: Sequence[int],
    allowed: Allowed,
    now: float,
) -> list[tuple[int, int]]:
    """Place tasks by the serial scheme in ``order``; return ``(task, mode)`` pairs.

    At every step the earliest-listed task whose predecessors are placed comes
    next. A task takes its mode from ``modes``, aligned with ``order``, or the
    allowed mode that finishes it earliest where the entry is negative.
    """
    position = {task: n for n, task in enumerate(order)}
    waiting: dict[int, int] = {}
    heap: list[tuple[int, int]] = []
    for task, n in position.items():
        left = sum(1 for before in model.predecessors[task] if not builder.placed(before))
        waiting[task] = left
        if left == 0:
            heap.append((n, task))
    heapq.heapify(heap)
    placed: list[tuple[int, int]] = []
    while heap:
        n, task = heapq.heappop(heap)
        mode = modes[n]
        if mode < 0:
            start, _, mode = earliest_finish(builder, task, allowed(task), now)
        else:
            start, _ = builder.earliest(task, mode, not_before=now)
        builder.place(task, mode, start)
        placed.append((task, mode))
        for after in model.successors[task]:
            if after in waiting:
                waiting[after] -= 1
                if waiting[after] == 0:
                    heapq.heappush(heap, (position[after], after))
    if len(placed) != len(waiting):
        raise ModelError("a planned task has a predecessor that is neither placed nor planned")
    return placed


def predicted_finishes(
    model: Model, builder: Builder, workflows: Collection[int]
) -> dict[int, float | None]:
    """Return the latest finish of each workflow's tasks, ``None`` if one is unplaced."""
    finishes: dict[int, float | None] = {}
    ends = builder.ends
    for workflow in workflows:
        finish = 0.0
        for task in model.members[workflow]:
            if not builder.placed(task):
                finishes[workflow] = None
                break
            finish = max(finish, ends[task])
        else:
            finishes[workflow] = finish
    return finishes
