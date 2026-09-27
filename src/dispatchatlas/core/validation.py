"""Feasibility checking of schedules against problems.

:func:`check_schedule` reports every constraint a schedule breaks. It is
independent of the schedule builders in :mod:`dispatchatlas.solvers`, so tests
use it as an oracle for them.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dispatchatlas.core.model import Problem
    from dispatchatlas.core.schedule import Assignment, Schedule

__all__ = ["DEFAULT_TOLERANCE", "Violation", "check_schedule", "deadline_excess", "is_feasible"]

DEFAULT_TOLERANCE = 1e-6
"""Absolute slack allowed in time and capacity comparisons."""


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken constraint.

    Attributes:
        code: Machine-readable kind, for example ``"precedence"``.
        message: Human-readable description.
        task: Id of the task involved, when there is one.
        resource: Id of the resource involved, when there is one.
        amount: Size of the violation in the constraint's unit.
    """

    code: str
    message: str
    task: str | None = None
    resource: str | None = None
    amount: float = 0.0


def check_schedule(
    problem: Problem, schedule: Schedule, *, tolerance: float = DEFAULT_TOLERANCE
) -> tuple[Violation, ...]:
    """Return every constraint of ``problem`` that ``schedule`` violates.

    Checked constraints: every task assigned exactly once in a valid mode, the
    mode's duration respected, release times, dependencies with lags and
    transfer times, resource capacities (including the instants zero-duration
    tasks occupy on exclusive resources), setup times, hard deadlines, and the
    permutation constraint.

    Args:
        problem: The problem.
        schedule: The schedule to check.
        tolerance: Absolute slack for time and capacity comparisons.

    Returns:
        The violations, empty when the schedule is feasible.
    """
    found: list[Violation] = []
    assigned: dict[str, Assignment] = {}
    for item in schedule.assignments:
        if item.task not in problem.task_index:
            found.append(Violation("unknown-task", f"unknown task {item.task!r}", item.task))
        elif item.task in assigned:
            found.append(
                Violation("duplicate-task", f"task {item.task!r} assigned twice", item.task)
            )
        else:
            assigned[item.task] = item
    for task in problem.tasks:
        if task.id not in assigned:
            found.append(Violation("missing-task", f"task {task.id!r} is not assigned", task.id))
    if found:
        return tuple(found)

    for task in problem.tasks:
        item = assigned[task.id]
        if not 0 <= item.mode < len(task.modes):
            found.append(
                Violation("invalid-mode", f"task {task.id!r} uses mode {item.mode}", task.id)
            )
            continue
        expected = task.modes[item.mode].duration
        # Negated so that a non-finite start or end, whose duration is NaN, fails.
        if not abs((item.end - item.start) - expected) <= tolerance:
            found.append(
                Violation(
                    "wrong-duration",
                    f"task {task.id!r} runs {item.end - item.start} instead of {expected}",
                    task.id,
                    amount=abs(item.end - item.start - expected),
                )
            )
        if item.start < task.release - tolerance:
            found.append(
                Violation(
                    "before-release",
                    f"task {task.id!r} starts at {item.start} before release {task.release}",
                    task.id,
                    amount=task.release - item.start,
                )
            )
        if task.deadline is not None and item.end > task.deadline + tolerance:
            found.append(
                Violation(
                    "deadline",
                    f"task {task.id!r} ends at {item.end} after deadline {task.deadline}",
                    task.id,
                    amount=item.end - task.deadline,
                )
            )
    if any(violation.code == "invalid-mode" for violation in found):
        return tuple(found)

    found.extend(_precedence_violations(problem, assigned, tolerance))
    found.extend(_capacity_violations(problem, schedule, tolerance))
    found.extend(_instant_violations(problem, schedule, tolerance))
    found.extend(_setup_violations(problem, assigned, tolerance))
    if problem.permutation:
        found.extend(_permutation_violations(problem, schedule, tolerance))
    return tuple(found)


def is_feasible(
    problem: Problem, schedule: Schedule, *, tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    """Return whether ``schedule`` violates no constraint of ``problem``."""
    return not check_schedule(problem, schedule, tolerance=tolerance)


def deadline_excess(problem: Problem, schedule: Schedule) -> float:
    """Return the total time by which tasks overrun their hard deadlines."""
    total = 0.0
    by_task = schedule.by_task
    for task in problem.tasks:
        if task.deadline is not None:
            total += max(0.0, by_task[task.id].end - task.deadline)
    return total


def _precedence_violations(
    problem: Problem, assigned: dict[str, Assignment], tolerance: float
) -> list[Violation]:
    found: list[Violation] = []
    for dependency in problem.dependencies:
        before = assigned[dependency.before]
        after = assigned[dependency.after]
        source = problem.task(dependency.before).modes[before.mode].location
        target = problem.task(dependency.after).modes[after.mode].location
        earliest = before.end + dependency.lag + problem.transfer_time(dependency, source, target)
        if after.start < earliest - tolerance:
            found.append(
                Violation(
                    "precedence",
                    f"task {dependency.after!r} starts at {after.start} before "
                    f"{dependency.before!r} allows it at {earliest}",
                    dependency.after,
                    amount=earliest - after.start,
                )
            )
    return found


def _capacity_violations(problem: Problem, schedule: Schedule, tolerance: float) -> list[Violation]:
    events: dict[str, list[tuple[float, int, float, float]]] = defaultdict(list)
    for item in schedule.assignments:
        if item.end - item.start <= 0.0:
            continue
        mode = problem.task(item.task).modes[item.mode]
        for resource_id, units in mode.demands.items():
            # Acquisitions count from ``tolerance`` after the start, and
            # releases sort before acquisitions at equal times, so back-to-back
            # tasks do not overlap even when rounding leaves a sliver.
            events[resource_id].append((item.start + tolerance, 1, units, item.start))
            events[resource_id].append((item.end, 0, -units, item.end))
    found: list[Violation] = []
    for resource_id in sorted(events):
        capacity = problem.resource(resource_id).capacity
        level = 0.0
        worst = 0.0
        when = 0.0
        for _, _, change, time in sorted(events[resource_id]):
            level += change
            if level - capacity > worst:
                worst, when = level - capacity, time
        if worst > tolerance:
            found.append(
                Violation(
                    "capacity",
                    f"resource {resource_id!r} is over capacity by {worst} at time {when}",
                    resource=resource_id,
                    amount=worst,
                )
            )
    return found


def _setup_violations(
    problem: Problem, assigned: dict[str, Assignment], tolerance: float
) -> list[Violation]:
    if problem.setups is None or not problem.setups.times:
        return []
    sequences: dict[str, list[Assignment]] = defaultdict(list)
    for task in problem.tasks:
        item = assigned[task.id]
        for resource_id in task.modes[item.mode].demands:
            if resource_id in problem.setups.resources:
                sequences[resource_id].append(item)
    found: list[Violation] = []
    for resource_id, items in sorted(sequences.items()):
        items.sort(key=lambda item: (item.start, item.end, item.task))
        previous: Assignment | None = None
        for item in items:
            ready = 0.0 if previous is None else previous.end
            setup = problem.setups.get(
                resource_id, None if previous is None else previous.task, item.task
            )
            if item.start < ready + setup - tolerance:
                found.append(
                    Violation(
                        "setup",
                        f"task {item.task!r} starts at {item.start} on {resource_id!r} "
                        f"before its setup completes at {ready + setup}",
                        item.task,
                        resource_id,
                        amount=ready + setup - item.start,
                    )
                )
            previous = item
    return found


def _instant_violations(problem: Problem, schedule: Schedule, tolerance: float) -> list[Violation]:
    """Report zero-duration tasks that another task runs across on an exclusive resource."""
    exclusive = problem.exclusive_resources
    busy: dict[str, list[tuple[float, float]]] = defaultdict(list)
    instants: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for item in schedule.assignments:
        for resource_id in problem.task(item.task).modes[item.mode].demands:
            if resource_id not in exclusive:
                continue
            if item.end - item.start > 0.0:
                busy[resource_id].append((item.start, item.end))
            else:
                instants[resource_id].append((item.start, item.task))
    found: list[Violation] = []
    for resource_id in sorted(instants):
        intervals = sorted(busy.get(resource_id, ()))
        starts = [start for start, _ in intervals]
        # reach[k] is the latest end among the first k + 1 intervals by start.
        reach = list(accumulate((end for _, end in intervals), max))
        for time, task_id in sorted(instants[resource_id]):
            count = bisect_left(starts, time - tolerance)
            if count and reach[count - 1] > time + tolerance:
                found.append(
                    Violation(
                        "capacity",
                        f"zero-duration task {task_id!r} at {time} falls inside a task "
                        f"running on exclusive resource {resource_id!r}",
                        task_id,
                        resource_id,
                        amount=reach[count - 1] - time,
                    )
                )
    return found


def _permutation_violations(
    problem: Problem, schedule: Schedule, tolerance: float
) -> list[Violation]:
    """Check that one group order fits the processing sequence of every machine.

    A machine's sequence is its tasks sorted by start and finish. Zero-duration
    tasks at the same instant may run in either order. The orders the machines
    imply between groups must not contradict each other. Overlaps are the
    capacity check's concern.
    """
    entries: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for item in schedule.assignments:
        task = problem.task(item.task)
        group = task.group if task.group is not None else task.id
        for resource_id in task.modes[item.mode].demands:
            entries[resource_id].append((item.start, item.end, group))
    later: dict[str, set[str]] = defaultdict(set)
    for resource_id in sorted(entries):
        blocks: list[list[tuple[float, float, str]]] = []
        for start, end, group in sorted(entries[resource_id]):
            head = blocks[-1][0] if blocks else None
            if (
                head is not None
                and end <= start
                and head[1] <= head[0]
                and abs(start - head[0]) <= tolerance
            ):
                blocks[-1].append((start, end, group))
            else:
                blocks.append([(start, end, group)])
        for block, following in pairwise(blocks):
            for _, _, before in block:
                later[before].update(after for _, _, after in following)
    cycle = _cyclic_groups(later)
    if not cycle:
        return []
    shown = ", ".join(repr(group) for group in cycle[:5])
    return [
        Violation(
            "permutation",
            f"the machines process groups in contradictory orders, involving {shown}",
        )
    ]


def _cyclic_groups(later: dict[str, set[str]]) -> list[str]:
    """Return, sorted, the groups Kahn's algorithm cannot order: on or after a cycle."""
    nodes = set(later) | {group for items in later.values() for group in items}
    waiting = dict.fromkeys(nodes, 0)
    for items in later.values():
        for group in items:
            waiting[group] += 1
    ready = [group for group, count in waiting.items() if count == 0]
    while ready:
        group = ready.pop()
        for following in later.get(group, ()):
            waiting[following] -= 1
            if waiting[following] == 0:
                ready.append(following)
    return sorted(group for group, count in waiting.items() if count > 0)
