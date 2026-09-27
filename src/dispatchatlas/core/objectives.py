"""Objective functions and their registry.

Every objective maps a problem and a schedule to one number. Solvers,
campaigns, and analyses look objectives up by name, so a plugin can add one
with :func:`register_objective` and use it everywhere.

Completion-time objectives are computed per **job**: the tasks that share a
``group``, or a single task without one. Due-date objectives are computed per
task, so a job's due date belongs on its final task.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from dispatchatlas.core.errors import ModelError

if TYPE_CHECKING:
    from dispatchatlas.core.model import Problem
    from dispatchatlas.core.schedule import Schedule

__all__ = [
    "Objective",
    "evaluate",
    "get_objective",
    "is_built_in",
    "is_registered",
    "register_objective",
    "registered_objectives",
    "unregister_objective",
]

TARDY_TOLERANCE = 1e-9
"""A task counts as tardy only when it finishes this much after its due date."""


@dataclass(frozen=True, slots=True)
class Objective:
    """A named objective function.

    Attributes:
        name: Registry key, for example ``"makespan"``.
        sense: ``"min"`` to minimise or ``"max"`` to maximise.
        unit: Unit of the value, for example ``"time"``.
        description: One sentence stating what is measured.
        function: Computes the value of a schedule for a problem.
    """

    name: str
    sense: Literal["min", "max"]
    unit: str
    description: str
    function: Callable[[Problem, Schedule], float]

    def minimized(self, value: float) -> float:
        """Return ``value`` oriented so that smaller is better."""
        return value if self.sense == "min" else -value


_REGISTRY: dict[str, Objective] = {}


def register_objective(objective: Objective, *, replace: bool = False) -> None:
    """Add an objective to the registry.

    Args:
        objective: The objective to add.
        replace: Allow replacing an objective registered under the same name.

    Raises:
        ModelError: If the name is taken and ``replace`` is false, or the sense
            is not ``"min"`` or ``"max"``.
    """
    if objective.sense not in ("min", "max"):
        raise ModelError(f"objective {objective.name!r} has invalid sense {objective.sense!r}")
    if objective.name in _REGISTRY and not replace:
        raise ModelError(f"objective {objective.name!r} is already registered")
    _REGISTRY[objective.name] = objective


def unregister_objective(name: str) -> None:
    """Remove an objective that a plugin registered.

    Removing a plugin's replacement of a built-in objective restores the
    library's definition. The library's own definitions cannot be removed.

    Raises:
        ModelError: If no objective of that name is registered, or it is a
            built-in objective with the library's definition.
    """
    if name not in _REGISTRY:
        raise ModelError(f"unknown objective {name!r}")
    if name not in _BUILT_IN:
        del _REGISTRY[name]
    elif is_built_in(name):
        raise ModelError(f"objective {name!r} is built in and cannot be removed")
    else:
        unit, description, function = _BUILT_IN[name]
        _REGISTRY[name] = Objective(name, "min", unit, description, function)


def is_registered(name: str) -> bool:
    """Return whether an objective of that name is registered."""
    return name in _REGISTRY


def get_objective(name: str) -> Objective:
    """Return the registered objective called ``name``.

    Raises:
        ModelError: If no such objective is registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ModelError(f"unknown objective {name!r}; known: {known}") from None


def registered_objectives() -> tuple[Objective, ...]:
    """Return every registered objective, sorted by name."""
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def is_built_in(name: str) -> bool:
    """Return whether ``name`` is registered with the library's own definition."""
    built_in = _BUILT_IN.get(name)
    registered = _REGISTRY.get(name)
    return built_in is not None and registered is not None and registered.function is built_in[2]


def evaluate(
    problem: Problem, schedule: Schedule, names: Iterable[str] | None = None
) -> dict[str, float]:
    """Compute objective values of a schedule.

    Args:
        problem: The problem the schedule solves.
        schedule: The schedule to evaluate.
        names: Objectives to compute. Defaults to the problem's objectives.

    Returns:
        Objective values keyed by name, in the requested order.
    """
    selected = problem.objectives if names is None else tuple(names)
    return {name: get_objective(name).function(problem, schedule) for name in selected}


# ------------------------------------------------------------------ helpers


def _ends(problem: Problem, schedule: Schedule) -> list[float]:
    by_task = schedule.by_task
    try:
        return [by_task[task.id].end for task in problem.tasks]
    except KeyError as missing:
        raise ModelError(f"schedule does not assign task {missing.args[0]!r}") from None


def _jobs(problem: Problem) -> list[tuple[float, float, list[int]]]:
    """Return ``(weight, release, task indices)`` for every job."""
    jobs: list[tuple[float, float, list[int]]] = []
    for members in problem.groups.values():
        tasks = [problem.tasks[i] for i in members]
        jobs.append((tasks[0].weight, min(t.release for t in tasks), list(members)))
    for i, task in enumerate(problem.tasks):
        if task.group is None:
            jobs.append((task.weight, task.release, [i]))
    return jobs


def _transfers(problem: Problem, schedule: Schedule) -> Iterable[tuple[float, str, str]]:
    """Yield ``(data, source endpoint, target endpoint)`` for every remote transfer."""
    if problem.network is None:
        return
    by_task = schedule.by_task
    for dependency in problem.dependencies:
        if dependency.data <= 0.0:
            continue
        before = problem.task(dependency.before)
        after = problem.task(dependency.after)
        source = before.modes[by_task[before.id].mode].location
        target = after.modes[by_task[after.id].mode].location
        if source is None or target is None or source == target:
            continue
        yield (
            dependency.data,
            problem.resource(source).endpoint,
            problem.resource(target).endpoint,
        )


def _busy_terms(problem: Problem, schedule: Schedule) -> Iterable[tuple[str, float, float]]:
    """Yield ``(resource id, utilised fraction, duration)`` for every demand."""
    for item in schedule.assignments:
        mode = problem.task(item.task).modes[item.mode]
        for resource_id, units in mode.demands.items():
            capacity = problem.resource(resource_id).capacity
            yield resource_id, units / capacity, item.end - item.start


# --------------------------------------------------------------- objectives


def _makespan(problem: Problem, schedule: Schedule) -> float:
    return float(max(_ends(problem, schedule), default=0.0))


def _total_completion_time(problem: Problem, schedule: Schedule) -> float:
    ends = _ends(problem, schedule)
    return sum(weight * max(ends[i] for i in members) for weight, _, members in _jobs(problem))


def _total_flow_time(problem: Problem, schedule: Schedule) -> float:
    ends = _ends(problem, schedule)
    return sum(
        weight * (max(ends[i] for i in members) - release)
        for weight, release, members in _jobs(problem)
    )


def _mean_response_time(problem: Problem, schedule: Schedule) -> float:
    ends = _ends(problem, schedule)
    jobs = _jobs(problem)
    total = sum(max(ends[i] for i in members) - release for _, release, members in jobs)
    return total / len(jobs)


def _total_tardiness(problem: Problem, schedule: Schedule) -> float:
    ends = _ends(problem, schedule)
    return sum(
        task.weight * max(0.0, ends[i] - task.due)
        for i, task in enumerate(problem.tasks)
        if task.due is not None
    )


def _max_lateness(problem: Problem, schedule: Schedule) -> float:
    ends = _ends(problem, schedule)
    values = [ends[i] - task.due for i, task in enumerate(problem.tasks) if task.due is not None]
    return max(values, default=0.0)


def _tardy_tasks(problem: Problem, schedule: Schedule) -> float:
    ends = _ends(problem, schedule)
    return float(
        sum(
            1
            for i, task in enumerate(problem.tasks)
            if task.due is not None and ends[i] > task.due + TARDY_TOLERANCE
        )
    )


def _energy(problem: Problem, schedule: Schedule) -> float:
    horizon = _makespan(problem, schedule)
    idle = sum(resource.idle_power for resource in problem.resources) * horizon
    busy = 0.0
    for resource_id, fraction, duration in _busy_terms(problem, schedule):
        resource = problem.resource(resource_id)
        busy += (resource.busy_power - resource.idle_power) * fraction * duration
    network = 0.0
    if problem.network is not None:
        for data, a, b in _transfers(problem, schedule):
            link = problem.network.link(a, b)
            if link is not None:
                network += link.energy_per_unit * data
    return idle + busy + network


def _cost(problem: Problem, schedule: Schedule) -> float:
    total = 0.0
    for resource_id, fraction, duration in _busy_terms(problem, schedule):
        total += problem.resource(resource_id).cost_rate * fraction * duration
    if problem.network is not None:
        for data, a, b in _transfers(problem, schedule):
            link = problem.network.link(a, b)
            if link is not None:
                total += link.cost_per_unit * data
    return total


_BUILT_IN: Mapping[str, tuple[str, str, Callable[[Problem, Schedule], float]]] = {
    "makespan": ("time", "Finish time of the last task.", _makespan),
    "total_completion_time": (
        "time",
        "Weighted sum of job completion times.",
        _total_completion_time,
    ),
    "total_flow_time": (
        "time",
        "Weighted sum of job completion times minus job release times.",
        _total_flow_time,
    ),
    "mean_response_time": (
        "time",
        "Mean over jobs of completion time minus release time.",
        _mean_response_time,
    ),
    "total_tardiness": (
        "time",
        "Weighted sum of how late tasks finish after their due dates.",
        _total_tardiness,
    ),
    "max_lateness": (
        "time",
        "Largest finish time minus due date over tasks with a due date.",
        _max_lateness,
    ),
    "tardy_tasks": ("count", "Number of tasks that finish after their due date.", _tardy_tasks),
    "energy": (
        "joule",
        "Idle power over the makespan plus busy power while tasks run plus "
        "network transfer energy.",
        _energy,
    ),
    "cost": (
        "currency",
        "Resource cost of the time tasks hold resources plus network transfer cost.",
        _cost,
    ),
}

for _name, (_unit, _description, _function) in _BUILT_IN.items():
    register_objective(Objective(_name, "min", _unit, _description, _function))
