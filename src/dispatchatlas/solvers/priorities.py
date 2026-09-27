"""Static task attributes that priority rules and list heuristics rank by.

Every function returns one value per task index. Durations of multi-mode
tasks use a summary over the modes (shortest or mean), stated per function.
"""

from __future__ import annotations

import math
from statistics import fmean

from dispatchatlas.core.model import Problem

__all__ = [
    "effective_due_dates",
    "longest_tails",
    "mean_durations",
    "min_durations",
    "remaining_group_work",
    "upward_ranks",
]


def min_durations(problem: Problem) -> list[float]:
    """Shortest duration of each task over its modes."""
    return [task.min_duration for task in problem.tasks]


def mean_durations(problem: Problem) -> list[float]:
    """Mean duration of each task over its modes."""
    return [fmean(mode.duration for mode in task.modes) for task in problem.tasks]


def effective_due_dates(problem: Problem) -> list[float]:
    """Due date of each task, inherited from its group when it has none.

    A task without its own due date takes the earliest due date in its group.
    Tasks with neither get infinity.
    """
    group_due: dict[str, float] = {}
    for task in problem.tasks:
        if task.group is not None and task.due is not None:
            group_due[task.group] = min(group_due.get(task.group, math.inf), task.due)
    return [
        task.due
        if task.due is not None
        else group_due.get(task.group, math.inf)
        if task.group is not None
        else math.inf
        for task in problem.tasks
    ]


def longest_tails(problem: Problem) -> list[float]:
    """Length of the longest path from each task's start to the end of the problem.

    The path length counts the shortest duration of every task on it and the
    lags between them, the classic critical-path bound.
    """
    durations = min_durations(problem)
    tails = [0.0] * problem.size
    incoming_lags: dict[tuple[int, int], float] = {}
    index = problem.task_index
    for dependency in problem.dependencies:
        incoming_lags[(index[dependency.before], index[dependency.after])] = dependency.lag
    for task in reversed(problem.topological_order):
        best = 0.0
        for following in problem.successors[task]:
            best = max(best, incoming_lags[(task, following)] + tails[following])
        tails[task] = durations[task] + best
    return tails


def remaining_group_work(problem: Problem) -> list[float]:
    """Work left in each task's group from that task on.

    The work is the task's shortest duration plus those of the tasks of its
    group that it reaches through dependencies between tasks of that group.
    For a job made of a chain of operations this is the classic "most work
    remaining" quantity. A task without a group counts only itself.
    """
    durations = min_durations(problem)
    group_of = [task.group for task in problem.tasks]
    remaining = [0.0] * problem.size
    for task in reversed(problem.topological_order):
        descendants: set[int] = set()
        stack = list(problem.successors[task])
        while stack:
            following = stack.pop()
            if following in descendants:
                continue
            if group_of[task] is not None and group_of[following] == group_of[task]:
                descendants.add(following)
                stack.extend(problem.successors[following])
        remaining[task] = durations[task] + sum(durations[i] for i in descendants)
    return remaining


def upward_ranks(problem: Problem) -> list[float]:
    """Upward rank of each task, as in HEFT (Topcuoglu, Hariri and Wu 2002).

    ``rank(i) = w(i) + max over successors j of (c(i, j) + rank(j))``, where
    ``w`` is the mean duration over modes and ``c`` is the lag plus the mean
    transfer time over every pair of modes of ``i`` and ``j``.
    """
    durations = mean_durations(problem)
    index = problem.task_index
    communication: dict[tuple[int, int], float] = {}
    for dependency in problem.dependencies:
        before = problem.tasks[index[dependency.before]]
        after = problem.tasks[index[dependency.after]]
        transfers = [
            problem.transfer_time(dependency, source.location, target.location)
            for source in before.modes
            for target in after.modes
        ]
        communication[(index[dependency.before], index[dependency.after])] = dependency.lag + fmean(
            transfers
        )
    ranks = [0.0] * problem.size
    for task in reversed(problem.topological_order):
        best = 0.0
        for following in problem.successors[task]:
            best = max(best, communication[(task, following)] + ranks[following])
        ranks[task] = durations[task] + best
    return ranks
