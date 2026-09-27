"""Pieces the learning solvers share: due dates, rewards, and policy values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dispatchatlas.solvers.priorities import effective_due_dates, min_durations

if TYPE_CHECKING:
    from dispatchatlas.core.model import Problem
    from dispatchatlas.solvers.evaluation import Score
    from dispatchatlas.solvers.search import Candidate, Search

__all__ = ["Improvement", "due_dates", "horizon", "policy_value"]


def horizon(problem: Problem) -> float:
    """Return the latest release time plus the sum of the shortest task durations.

    Running every task in its shortest mode one after another from the last
    release, ignoring lags and transfers, ends at this time. The learners use
    it as the due date of tasks that have none.
    """
    return max(task.release for task in problem.tasks) + math.fsum(min_durations(problem))


def due_dates(problem: Problem) -> list[float]:
    """Return the due date that the learners rank each task by.

    A task's due date is the earlier of its soft due date, inherited from its
    group when it has none (see
    :func:`~dispatchatlas.solvers.priorities.effective_due_dates`), and its hard
    deadline. A task with neither gets the problem's :func:`horizon`, so that
    its slack stays finite and large.
    """
    end = horizon(problem)
    dues: list[float] = []
    for task, soft in zip(problem.tasks, effective_due_dates(problem), strict=True):
        due = soft if task.deadline is None else min(soft, task.deadline)
        dues.append(end if math.isinf(due) else due)
    return dues


@dataclass(frozen=True, slots=True)
class Improvement:
    """The reward of a schedule: its improvement over a reference schedule, without units.

    The reward of a schedule with objective ``v``, in minimisation form, and
    total deadline overrun ``o`` is ``(v0 - v) / max(|v0|, 1) + (o0 - o) /
    max(m0, 1)``, where ``v0``, ``o0``, and ``m0`` are the objective, overrun,
    and makespan of the reference. The first term is the relative gain in the
    objective, the second the overrun removed, counted in reference makespans.
    The reference itself scores zero.

    Attributes:
        value: Objective of the reference, in minimisation form.
        overrun: Total deadline overrun of the reference.
        value_scale: Divisor of objective differences, ``max(|v0|, 1)``.
        time_scale: Divisor of overrun differences, ``max(m0, 1)``.
    """

    value: float
    overrun: float
    value_scale: float
    time_scale: float

    @classmethod
    def over(cls, reference: Candidate) -> Improvement:
        """Return the reward that measures schedules against an evaluated reference."""
        score = reference.score
        return cls(
            score.value,
            score.violation,
            max(abs(score.value), 1.0),
            max(reference.decoded.makespan, 1.0),
        )

    def __call__(self, score: Score) -> float:
        """Return the reward of a schedule with the given score."""
        gain = (self.value - score.value) / self.value_scale
        return gain + (self.overrun - score.violation) / self.time_scale


def policy_value(search: Search, latest: Candidate | None) -> float:
    """Return the objective of the latest greedy-policy schedule a learner evaluated.

    The learners evaluate the greedy policy of their untrained start first.
    When that evaluation reached the target and ended the run before
    returning, the incumbent is that schedule.

    Args:
        search: The run.
        latest: The latest greedy-policy schedule whose evaluation returned.

    Returns:
        The objective with its natural sign, or NaN when nothing was evaluated.
    """
    candidate = latest if latest is not None else search.best
    if candidate is None:
        return math.nan
    return search.evaluator.objective.minimized(candidate.score.value)
