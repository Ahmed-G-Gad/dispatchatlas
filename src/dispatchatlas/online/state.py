"""The state of an online run at one instant, as a planner sees it.

Every task of a run is in one :class:`TaskStatus`:

- ``unreleased``: its workflow has not arrived.
- ``released``: its workflow has arrived and some predecessor has not
  finished.
- ``ready``: every predecessor has finished and it has not started.
- ``running``: it has started on a node and not finished.
- ``finished``: it ran to completion.

A task **commits** when it starts: from that instant its node, start, and
finish are a reservation that nothing moves. ``running`` and ``finished``
tasks are the committed ones; every other task is uncommitted, and only
uncommitted tasks are planned. A node failure ends every attempt running on
it: the attempt is **lost**, kept in the record as an :class:`Attempt` with
outcome ``lost``, and its task is ready again and restarts from scratch.
Finished work is never lost, so the outputs of a finished task stay
available after its node fails.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["Attempt", "AttemptOutcome", "SystemState", "TaskStatus"]


class TaskStatus(StrEnum):
    """Where a task is in its life."""

    UNRELEASED = "unreleased"
    RELEASED = "released"
    READY = "ready"
    RUNNING = "running"
    FINISHED = "finished"

    @property
    def committed(self) -> bool:
        """Whether the task has started: running or finished."""
        return self in (TaskStatus.RUNNING, TaskStatus.FINISHED)


class AttemptOutcome(StrEnum):
    """How an attempt ended."""

    RUNNING = "running"
    FINISHED = "finished"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class Attempt:
    """One execution of a task, from its commitment.

    Attributes:
        task: Task index in the run's problem.
        mode: Mode index of the task.
        start: Commitment time, when the attempt started.
        end: Finish time: the planned one while it runs, the failure time if
            it was lost.
        outcome: How it ended.
    """

    task: int
    mode: int
    start: float
    end: float
    outcome: AttemptOutcome


@dataclass(frozen=True)
class SystemState:
    """A snapshot of a run at a decision point.

    Attributes:
        now: The current time.
        status: Status of each task, by task index.
        committed: The current attempt of each committed task, by task index.
        down: Ids of the nodes that are down.
        ready_since: For each ready task, the instant it became ready: when
            its last predecessor finished, its workflow arrived, or its last
            attempt was lost.
        known: Ids of the workflows that have arrived, in arrival order.
        losses: Number of lost attempts of each task that has had one.
    """

    now: float
    status: tuple[TaskStatus, ...]
    committed: Mapping[int, Attempt]
    down: frozenset[str]
    ready_since: Mapping[int, float]
    known: tuple[str, ...]
    losses: Mapping[int, int]

    def tasks_in(self, status: TaskStatus) -> tuple[int, ...]:
        """Return the indices of the tasks in ``status``, in index order."""
        return tuple(i for i, s in enumerate(self.status) if s is status)

    @property
    def uncommitted(self) -> tuple[int, ...]:
        """Indices of the released and ready tasks."""
        return tuple(
            i for i, s in enumerate(self.status) if s in (TaskStatus.RELEASED, TaskStatus.READY)
        )
