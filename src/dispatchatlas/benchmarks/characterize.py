"""Descriptive features of problem instances.

:func:`features` summarises a problem as numbers that describe its size,
structure, flexibility, resources, communication, and timing, so instance
sets can be compared, stratified, and related to solver performance. Every
feature is a deterministic function of the problem alone; :data:`DEFINITIONS`
states each one.

Conventions shared by the definitions:

- A task's *shortest* and *mean* duration are taken over its modes.
- *Jobs* are groups of tasks, and every ungrouped task is a job of its own,
  as in :mod:`dispatchatlas.core.objectives`. A job's release is the earliest
  release of its tasks.
- The *critical path* is the longest dependency path when every task takes
  its shortest duration and every dependency its lag, ignoring transfers,
  releases, and resource limits.
- The *earliest finish* of a task is its finish time in a relaxation without
  resource limits: every task starts at its release or once each
  predecessor's data have arrived, where each dependency is priced at the
  predecessor mode that delivers first (see :func:`earliest_finish_times`).
- A ratio whose denominator is zero is reported as 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import fmean, pstdev
from types import MappingProxyType

from dispatchatlas.core.model import Problem

__all__ = ["DEFINITIONS", "earliest_finish_times", "features"]

DEFINITIONS: Mapping[str, str] = MappingProxyType(
    {
        "tasks": "Number of tasks.",
        "resources": "Number of resources.",
        "dependencies": "Number of dependencies.",
        "jobs": "Number of jobs: groups plus ungrouped tasks.",
        "objectives": "Number of declared objectives.",
        "dependency_density": "Dependencies divided by the tasks(tasks - 1) / 2 pairs.",
        "depth": "Tasks on the longest chain of dependencies.",
        "critical_path": "Length of the critical path.",
        "total_work": "Sum of the tasks' shortest durations.",
        "parallelism": "Total work divided by the critical path, the mean parallelism.",
        "mean_duration": "Mean over tasks of their mean duration.",
        "duration_cv": "Coefficient of variation of the tasks' mean durations.",
        "mean_modes": "Mean number of modes per task, the flexibility.",
        "mode_spread": "Mean over tasks with several modes of (longest - shortest) / longest "
        "duration, the heterogeneity of the alternatives.",
        "pinned_share": "Share of tasks whose modes all lie at one network endpoint, so their "
        "location is fixed.",
        "capacity_mean": "Mean resource capacity.",
        "capacity_max": "Largest resource capacity.",
        "unary_share": "Share of resources of capacity 1.",
        "setup_share": "Share of resources with a non-zero setup time.",
        "area_bound": "Sum over tasks of the smallest product, over modes, of duration and "
        "units held, divided by the total capacity: a lower bound on the length of any "
        "schedule.",
        "ccr": "Communication-to-computation ratio: the mean over dependencies of their mean "
        "transfer time between modes on different resources, divided by the mean duration "
        "of the tasks whose mean duration is positive; 0 without a network.",
        "data_share": "Share of dependencies that carry data; 0 without a network.",
        "release_span": "Latest release minus earliest release.",
        "release_spread": "Release span divided by the area bound; large values mean the "
        "work arrives thinly spread over time.",
        "due_share": "Share of tasks with a due date.",
        "due_tightness": "Mean over tasks with a due date of (due date - job release) / "
        "(earliest finish - job release), skipping tasks whose earliest finish is their "
        "job's release; 1 means no slack.",
        "deadline_share": "Share of tasks with a hard deadline.",
        "deadline_tightness": "Like due_tightness, for hard deadlines.",
    }
)
"""The name and definition of every feature :func:`features` returns, in order."""


def features(problem: Problem) -> dict[str, float]:
    """Return every feature of :data:`DEFINITIONS` for ``problem``.

    Args:
        problem: The problem to describe.

    Returns:
        Feature values keyed by name, in the order of :data:`DEFINITIONS`.
    """
    tasks = problem.tasks
    count = len(tasks)
    shortest = [task.min_duration for task in tasks]
    means = [fmean(mode.duration for mode in task.modes) for task in tasks]
    capacities = [resource.capacity for resource in problem.resources]
    total_capacity = math.fsum(capacities)
    critical_path, depth = _critical_path(problem)
    total_work = math.fsum(shortest)
    occupied = math.fsum(
        min(mode.duration * math.fsum(mode.demands.values()) for mode in task.modes)
        for task in tasks
    )
    area_bound = _ratio(occupied, total_capacity)
    releases = [task.release for task in tasks]
    release_span = max(releases) - min(releases)
    mean_duration = fmean(means)
    finishes = earliest_finish_times(problem)
    job_release = _job_releases(problem)

    values: dict[str, float] = {
        "tasks": count,
        "resources": len(problem.resources),
        "dependencies": len(problem.dependencies),
        "jobs": len(problem.groups) + sum(1 for task in tasks if task.group is None),
        "objectives": len(problem.objectives),
        "dependency_density": _ratio(len(problem.dependencies), count * (count - 1) / 2),
        "depth": depth,
        "critical_path": critical_path,
        "total_work": total_work,
        "parallelism": _ratio(total_work, critical_path),
        "mean_duration": mean_duration,
        "duration_cv": _ratio(pstdev(means), mean_duration),
        "mean_modes": fmean(len(task.modes) for task in tasks),
        "mode_spread": _mode_spread(problem),
        "pinned_share": _pinned_share(problem),
        "capacity_mean": fmean(capacities) if capacities else 0.0,
        "capacity_max": max(capacities, default=0.0),
        "unary_share": _ratio(sum(1 for c in capacities if c == 1.0), len(capacities)),
        "setup_share": _ratio(
            len(problem.setups.resources) if problem.setups is not None else 0,
            len(capacities),
        ),
        "area_bound": area_bound,
        "ccr": _ccr(problem, means),
        "data_share": _ratio(
            sum(1 for d in problem.dependencies if d.data > 0.0)
            if problem.network is not None
            else 0,
            len(problem.dependencies),
        ),
        "release_span": release_span,
        "release_spread": _ratio(release_span, area_bound),
        "due_share": _ratio(sum(1 for task in tasks if task.due is not None), count),
        "due_tightness": _tightness(
            [(i, task.due) for i, task in enumerate(tasks) if task.due is not None],
            finishes,
            job_release,
        ),
        "deadline_share": _ratio(sum(1 for task in tasks if task.deadline is not None), count),
        "deadline_tightness": _tightness(
            [(i, task.deadline) for i, task in enumerate(tasks) if task.deadline is not None],
            finishes,
            job_release,
        ),
    }
    return {name: float(values[name]) for name in DEFINITIONS}


def earliest_finish_times(problem: Problem) -> tuple[float, ...]:
    """Return each task's earliest finish time when resources never make it wait.

    Dynamic programming in topological order over every task and mode. A
    task in a given mode starts at its release, or once every predecessor's
    data have arrived, whichever is later. The data of a predecessor arrive
    at its finish plus the dependency lag plus the transfer time to the
    mode's location, minimised over the predecessor's modes independently for
    each dependency. A task's earliest finish is the minimum over its modes.
    The result is a lower bound on the task's finish in any feasible schedule.
    When every task has at most one predecessor, it is also the earliest
    finish that some placement reaches when resources never make a task wait.

    Args:
        problem: The problem.

    Returns:
        The earliest finish of each task, indexed like ``problem.tasks``.
    """
    tasks = problem.tasks
    index = problem.task_index
    by_mode: list[list[float]] = [[] for _ in tasks]
    for i in problem.topological_order:
        task = tasks[i]
        row: list[float] = []
        for mode in task.modes:
            start = task.release
            for dependency in problem.incoming[i]:
                before = index[dependency.before]
                arrival = min(
                    finish + problem.transfer_time(dependency, other.location, mode.location)
                    for finish, other in zip(by_mode[before], tasks[before].modes, strict=True)
                )
                start = max(start, arrival + dependency.lag)
            row.append(start + mode.duration)
        by_mode[i] = row
    return tuple(min(row) for row in by_mode)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _critical_path(problem: Problem) -> tuple[float, int]:
    """Return the critical-path length and the number of tasks on the longest chain."""
    index = problem.task_index
    finish = [0.0] * problem.size
    chain = [0] * problem.size
    for i in problem.topological_order:
        start, links = 0.0, 0
        for dependency in problem.incoming[i]:
            before = index[dependency.before]
            start = max(start, finish[before] + dependency.lag)
            links = max(links, chain[before])
        finish[i] = start + problem.tasks[i].min_duration
        chain[i] = links + 1
    return max(finish, default=0.0), max(chain, default=0)


def _mode_spread(problem: Problem) -> float:
    spreads = []
    for task in problem.tasks:
        durations = [mode.duration for mode in task.modes]
        if len(durations) > 1 and max(durations) > 0.0:
            spreads.append((max(durations) - min(durations)) / max(durations))
    return fmean(spreads) if spreads else 0.0


def _pinned_share(problem: Problem) -> float:
    pinned = 0
    for task in problem.tasks:
        endpoints = {
            None if mode.location is None else problem.resource(mode.location).endpoint
            for mode in task.modes
        }
        pinned += len(endpoints) == 1
    return _ratio(pinned, problem.size)


def _ccr(problem: Problem, means: list[float]) -> float:
    if problem.network is None or not problem.dependencies:
        return 0.0
    tasks, index = problem.tasks, problem.task_index
    communication = []
    for dependency in problem.dependencies:
        before, after = tasks[index[dependency.before]], tasks[index[dependency.after]]
        times = [
            problem.transfer_time(dependency, a.location, b.location)
            for a in before.modes
            for b in after.modes
            if a.location is not None and b.location is not None and a.location != b.location
        ]
        communication.append(fmean(times) if times else 0.0)
    computation = [mean for mean in means if mean > 0.0]
    return _ratio(fmean(communication), fmean(computation) if computation else 0.0)


def _job_releases(problem: Problem) -> list[float]:
    """Return the release of each task's job, indexed like ``problem.tasks``."""
    releases = [task.release for task in problem.tasks]
    for members in problem.groups.values():
        earliest = min(releases[i] for i in members)
        for i in members:
            releases[i] = earliest
    return releases


def _tightness(
    limits: list[tuple[int, float]], finishes: tuple[float, ...], releases: list[float]
) -> float:
    ratios = [
        (limit - releases[i]) / (finishes[i] - releases[i])
        for i, limit in limits
        if finishes[i] > releases[i]
    ]
    return fmean(ratios) if ratios else 0.0
