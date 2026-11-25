"""Checking an executed online schedule against the problem and the trace.

:func:`check_run` is independent of the runner's bookkeeping: it rebuilds
what happened from the run's attempts and the trace's events, and reports
every rule an execution must keep:

- every attempt runs a valid mode of a task whose workflow had arrived, and
  starts no earlier than the task's release;
- an attempt starts after every predecessor's finished attempt, plus the
  lag and the transfer between their resources;
- a finished attempt runs exactly its mode's duration; a lost one ends at a
  failure of one of its resources, before its duration is out;
- no attempt runs on a resource while it is down;
- at every instant the attempts on a resource, lost ones included up to
  their loss, hold no more than its capacity;
- every task has at most one finished attempt, and its attempts do not
  overlap;
- and, when every task finished, the finished attempts form a schedule that
  :func:`~dispatchatlas.core.validation.check_schedule` accepts.

Commitments never move by construction: an attempt is written once, at its
start, and a later record can only end it at a failure, which the checks of
lost attempts cover.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

from dispatchatlas.core.model import Problem
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.validation import DEFAULT_TOLERANCE, check_schedule
from dispatchatlas.online.events import Failure, Repair, Trace
from dispatchatlas.online.runner import OnlineResult

__all__ = ["check_run"]


def check_run(
    problem: Problem, trace: Trace, result: OnlineResult, *, tolerance: float = DEFAULT_TOLERANCE
) -> tuple[str, ...]:
    """Return every rule the run's execution breaks, empty when it keeps them all.

    Args:
        problem: The run's problem, from :func:`~dispatchatlas.online.runner.online_problem`.
        trace: The run's trace.
        result: The run.
        tolerance: Absolute slack of time and capacity comparisons.
    """
    issues: list[str] = []
    index = problem.task_index
    arrivals = {arrival.workflow.id: arrival.time for arrival in trace.arrivals}
    outages: dict[str, list[tuple[float, float]]] = defaultdict(list)
    failures: dict[str, list[float]] = defaultdict(list)
    open_failure: dict[str, float] = {}
    for event in trace.events:
        if isinstance(event, Failure):
            open_failure[event.node] = event.time
            failures[event.node].append(event.time)
        elif isinstance(event, Repair):
            outages[event.node].append((open_failure.pop(event.node), event.time))
    for node, began in open_failure.items():
        outages[node].append((began, float("inf")))

    finished: dict[str, tuple[int, float, float]] = {}
    by_task: dict[str, list[tuple[float, float]]] = defaultdict(list)
    modes_of: dict[str, list[tuple[int, float]]] = defaultdict(list)
    usage: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for task_id, mode, start, end, outcome in result.attempts:
        if task_id not in index:
            issues.append(f"unknown task {task_id!r}")
            continue
        task = problem.tasks[index[task_id]]
        where = f"task {task_id!r} attempt at {start}"
        if not 0 <= mode < len(task.modes):
            issues.append(f"{where}: invalid mode {mode}")
            continue
        chosen = task.modes[mode]
        if task.group is None or start < arrivals.get(task.group, float("inf")) - tolerance:
            issues.append(f"{where}: starts before its workflow arrives")
        if start < task.release - tolerance:
            issues.append(f"{where}: starts before its release {task.release}")
        if outcome == "finished":
            if abs(end - start - chosen.duration) > tolerance:
                issues.append(f"{where}: runs {end - start} instead of {chosen.duration}")
            if task_id in finished:
                issues.append(f"task {task_id!r} finished twice")
            finished[task_id] = (mode, start, end)
        elif outcome == "lost":
            if end > start + chosen.duration + tolerance:
                issues.append(f"{where}: lost after its duration was out")
            if not any(
                abs(end - time) <= tolerance for node in chosen.demands for time in failures[node]
            ):
                issues.append(f"{where}: lost at {end}, when none of its resources failed")
        else:
            issues.append(f"{where}: still {outcome} when the run ended")
        for node in chosen.demands:
            for down, up in outages[node]:
                if start < up - tolerance and end > down + tolerance:
                    issues.append(f"{where}: runs on {node!r} while it is down")
        by_task[task_id].append((start, end))
        modes_of[task_id].append((mode, start))
        for node, units in chosen.demands.items():
            usage[node].append((start, end, units))

    for task_id, spans in by_task.items():
        spans.sort()
        for (_, end), (start, _) in pairwise(spans):
            if start < end - tolerance:
                issues.append(f"task {task_id!r} has overlapping attempts")

    capacity = {resource.id: resource.capacity for resource in problem.resources}
    for node, held in usage.items():
        changes = sorted(
            [(start, units) for start, end, units in held if end > start]
            + [(end, -units) for start, end, units in held if end > start],
            key=lambda item: (item[0], item[1]),
        )
        level = 0.0
        for at, delta in changes:
            level += delta
            if level > capacity[node] + tolerance:
                issues.append(f"resource {node!r} holds {level} units at {at}")
                break

    for task_id, attempts in modes_of.items():
        task = problem.tasks[index[task_id]]
        for dependency in problem.incoming[index[task_id]]:
            before = finished.get(dependency.before)
            for mode, start in attempts:
                if before is None:
                    issues.append(f"task {task_id!r} ran before {dependency.before!r} finished")
                    continue
                before_mode, _, before_end = before
                source = problem.task(dependency.before).modes[before_mode].location
                target = task.modes[mode].location
                ready = before_end + dependency.lag
                ready += problem.transfer_time(dependency, source, target)
                if start < ready - tolerance:
                    issues.append(
                        f"task {task_id!r} starts at {start} before the data of "
                        f"{dependency.before!r} arrive at {ready}"
                    )

    if len(finished) == problem.size:
        schedule = Schedule.of(
            Assignment(task_id, mode, start, end)
            for task_id, (mode, start, end) in finished.items()
        )
        for violation in check_schedule(problem, schedule, tolerance=tolerance):
            issues.append(f"schedule: {violation.message}")
    return tuple(issues)
