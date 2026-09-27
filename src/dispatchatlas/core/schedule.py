"""Schedules: when each task runs and in which mode."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property

from dispatchatlas.core._frozen import FrozenMap
from dispatchatlas.core.errors import ModelError

__all__ = ["Assignment", "Schedule"]


@dataclass(frozen=True, slots=True)
class Assignment:
    """The execution of one task.

    Attributes:
        task: Id of the task.
        mode: Index of the mode the task runs in, within ``Task.modes``.
        start: Start time.
        end: Finish time. Equals ``start`` plus the mode's duration in a
            valid schedule.
    """

    task: str
    mode: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        """Time between start and finish."""
        return self.end - self.start


@dataclass(frozen=True)
class Schedule:
    """A set of task assignments, ordered by start time and then task id.

    A schedule records decisions only. Whether it respects a problem's
    constraints is checked by :func:`dispatchatlas.core.validation.check_schedule`.
    """

    assignments: tuple[Assignment, ...]

    def __post_init__(self) -> None:
        items = sorted(self.assignments, key=lambda item: (item.start, item.task))
        object.__setattr__(self, "assignments", tuple(items))

    @classmethod
    def of(cls, assignments: Iterable[Assignment]) -> Schedule:
        """Build a schedule from any iterable of assignments."""
        return cls(tuple(assignments))

    @cached_property
    def by_task(self) -> Mapping[str, Assignment]:
        """Assignments keyed by task id.

        Raises:
            ModelError: If a task is assigned more than once.
        """
        index: dict[str, Assignment] = {}
        for item in self.assignments:
            if item.task in index:
                raise ModelError(f"task {item.task!r} is assigned more than once")
            index[item.task] = item
        return FrozenMap(index)

    @property
    def makespan(self) -> float:
        """Latest finish time, or zero for an empty schedule."""
        return max((item.end for item in self.assignments), default=0.0)

    def __len__(self) -> int:
        return len(self.assignments)
