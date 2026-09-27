"""Scoring decoded schedules.

An :class:`Evaluator` decodes orders with a :class:`~dispatchatlas.solvers.scheduling.Decoder`
and scores the result. Scores compare lexicographically: first the total
hard-deadline overrun, then the primary objective in minimisation form. Every
search therefore prefers any schedule that meets all deadlines over one that
does not, and among equals prefers the better objective.

The evaluator computes every built-in objective directly from the decoded
arrays, which is much faster than building a schedule object. Objectives
registered by plugins fall back to their registered function. A partial
schedule, as insertion heuristics decode, is scored over its placed tasks:
jobs with no placed task, and transfers from or to unplaced tasks, count
nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dispatchatlas.core.objectives import TARDY_TOLERANCE, get_objective, is_built_in
from dispatchatlas.solvers.scheduling import Decoded, Decoder

if TYPE_CHECKING:
    from dispatchatlas.core.model import Problem
    from dispatchatlas.core.objectives import Objective

__all__ = ["Evaluator", "Score"]


@dataclass(frozen=True, slots=True, order=True)
class Score:
    """The quality of a schedule. Smaller is better.

    Attributes:
        violation: Total time by which tasks overrun their hard deadlines.
            Zero for a feasible schedule.
        value: Primary objective in minimisation form.
    """

    violation: float
    value: float

    @property
    def feasible(self) -> bool:
        """Whether every hard deadline is met."""
        return self.violation <= 0.0


class Evaluator:
    """Decodes orders and scores them for one problem.

    Args:
        problem: The problem.
        objective: Primary objective. Defaults to the problem's first one.
        insertion: Passed to the :class:`Decoder`.

    Attributes:
        problem: The problem.
        decoder: The shared schedule builder.
        objective: The primary objective.
        evaluations: Candidates scored so far: calls to :meth:`evaluate`, and
            schedules a :class:`~dispatchatlas.solvers.search.Search` received
            through ``submit``.
    """

    def __init__(
        self, problem: Problem, objective: str | None = None, *, insertion: bool = True
    ) -> None:
        self.problem = problem
        self.decoder = Decoder(problem, insertion=insertion)
        self.objective: Objective = get_objective(objective or problem.objectives[0])
        self.evaluations = 0
        self._objectives = tuple(get_objective(name) for name in problem.objectives)
        self._deadlines = [
            (i, task.deadline) for i, task in enumerate(problem.tasks) if task.deadline is not None
        ]
        self._fast = _FastObjectives(problem)

    def evaluate(
        self, order: Sequence[int], modes: Sequence[int] | None = None
    ) -> tuple[Score, Decoded]:
        """Decode and score one candidate. Counts as one evaluation."""
        self.evaluations += 1
        decoded = self.decoder.decode(order, modes)
        return self.score(decoded), decoded

    def score(self, decoded: Decoded) -> Score:
        """Score an already decoded schedule. Does not count as an evaluation."""
        ends = decoded.ends
        violation = 0.0
        for task, deadline in self._deadlines:
            if ends[task] > deadline:
                violation += ends[task] - deadline
        value = self.objective.minimized(self.value(decoded, self.objective))
        return Score(violation, value)

    def value(self, decoded: Decoded, objective: Objective) -> float:
        """Return one objective of a decoded schedule, as the registry defines it."""
        fast = self._fast.get(objective.name)
        if fast is not None:
            return fast(decoded)
        return objective.function(self.problem, decoded.schedule(self.problem))

    def values(self, decoded: Decoded) -> dict[str, float]:
        """Return every problem objective of a decoded schedule, keyed by name."""
        return {objective.name: self.value(decoded, objective) for objective in self._objectives}

    def vector(self, decoded: Decoded) -> tuple[float, ...]:
        """Return every problem objective in minimisation form, in problem order."""
        return tuple(
            objective.minimized(self.value(decoded, objective)) for objective in self._objectives
        )


class _FastObjectives:
    """Array implementations of the built-in objectives."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        jobs: list[tuple[float, float, tuple[int, ...]]] = []
        for members in problem.groups.values():
            first = problem.tasks[members[0]]
            jobs.append((first.weight, min(problem.tasks[i].release for i in members), members))
        for i, task in enumerate(problem.tasks):
            if task.group is None:
                jobs.append((task.weight, task.release, (i,)))
        self.jobs = jobs
        self.dues = [
            (i, task.due, task.weight)
            for i, task in enumerate(problem.tasks)
            if task.due is not None
        ]
        self.idle_power = sum(resource.idle_power for resource in problem.resources)
        # Busy energy and cost of each (task, mode) do not depend on timing.
        index = problem.resource_index
        self.busy_energy: list[list[float]] = []
        self.busy_cost: list[list[float]] = []
        for task in problem.tasks:
            energy_row, cost_row = [], []
            for mode in task.modes:
                energy = cost = 0.0
                for resource_id, units in mode.demands.items():
                    resource = problem.resources[index[resource_id]]
                    fraction = units / resource.capacity
                    energy += (resource.busy_power - resource.idle_power) * fraction * mode.duration
                    cost += resource.cost_rate * fraction * mode.duration
                energy_row.append(energy)
                cost_row.append(cost)
            self.busy_energy.append(energy_row)
            self.busy_cost.append(cost_row)
        self.transfers: list[tuple[int, int, float]] = []
        if problem.network is not None:
            task_index = problem.task_index
            for dependency in problem.dependencies:
                if dependency.data > 0.0:
                    self.transfers.append(
                        (
                            task_index[dependency.before],
                            task_index[dependency.after],
                            dependency.data,
                        )
                    )
        self.functions: dict[str, Callable[[Decoded], float]] = {
            "makespan": self.makespan,
            "total_completion_time": self.total_completion_time,
            "total_flow_time": self.total_flow_time,
            "mean_response_time": self.mean_response_time,
            "total_tardiness": self.total_tardiness,
            "max_lateness": self.max_lateness,
            "tardy_tasks": self.tardy_tasks,
            "energy": self.energy,
            "cost": self.cost,
        }

    def get(self, name: str) -> Callable[[Decoded], float] | None:
        # A plugin that re-registers a built-in name gets its own function.
        return self.functions.get(name) if is_built_in(name) else None

    def _jobs(self, decoded: Decoded) -> list[tuple[float, float, tuple[int, ...]]]:
        """Jobs with their placed members, leaving out jobs with none placed."""
        if len(decoded.sequence) == self.problem.size:
            return self.jobs
        modes = decoded.modes
        jobs = []
        for weight, release, members in self.jobs:
            placed = tuple(i for i in members if modes[i] >= 0)
            if placed:
                jobs.append((weight, release, placed))
        return jobs

    def _dues(self, decoded: Decoded) -> list[tuple[int, float, float]]:
        """Due-date entries of the placed tasks."""
        if len(decoded.sequence) == self.problem.size:
            return self.dues
        modes = decoded.modes
        return [entry for entry in self.dues if modes[entry[0]] >= 0]

    def makespan(self, decoded: Decoded) -> float:
        return float(max(decoded.ends, default=0.0))

    def total_completion_time(self, decoded: Decoded) -> float:
        ends = decoded.ends
        return sum(
            weight * max(ends[i] for i in members) for weight, _, members in self._jobs(decoded)
        )

    def total_flow_time(self, decoded: Decoded) -> float:
        ends = decoded.ends
        return sum(
            weight * (max(ends[i] for i in members) - release)
            for weight, release, members in self._jobs(decoded)
        )

    def mean_response_time(self, decoded: Decoded) -> float:
        ends = decoded.ends
        jobs = self._jobs(decoded)
        total = sum(max(ends[i] for i in members) - release for _, release, members in jobs)
        return total / len(jobs) if jobs else 0.0

    def total_tardiness(self, decoded: Decoded) -> float:
        ends = decoded.ends
        return sum(weight * max(0.0, ends[i] - due) for i, due, weight in self._dues(decoded))

    def max_lateness(self, decoded: Decoded) -> float:
        ends = decoded.ends
        return max((ends[i] - due for i, due, _ in self._dues(decoded)), default=0.0)

    def tardy_tasks(self, decoded: Decoded) -> float:
        ends = decoded.ends
        return float(sum(1 for i, due, _ in self._dues(decoded) if ends[i] > due + TARDY_TOLERANCE))

    def energy(self, decoded: Decoded) -> float:
        busy = sum(self.busy_energy[i][decoded.modes[i]] for i in decoded.sequence)
        return self.idle_power * self.makespan(decoded) + busy + self._network(decoded, "energy")

    def cost(self, decoded: Decoded) -> float:
        busy = sum(self.busy_cost[i][decoded.modes[i]] for i in decoded.sequence)
        return busy + self._network(decoded, "cost")

    def _network(self, decoded: Decoded, kind: str) -> float:
        problem = self.problem
        network = problem.network
        if network is None:
            return 0.0
        total = 0.0
        modes = decoded.modes
        for before, after, data in self.transfers:
            if modes[before] < 0 or modes[after] < 0:
                continue
            source = problem.tasks[before].modes[modes[before]].location
            target = problem.tasks[after].modes[modes[after]].location
            if source is None or target is None or source == target:
                continue
            link = network.link(
                problem.resource(source).endpoint, problem.resource(target).endpoint
            )
            if link is not None:
                total += (link.energy_per_unit if kind == "energy" else link.cost_per_unit) * data
        return total
