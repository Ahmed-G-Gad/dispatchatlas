"""Hybrid solvers: heuristic search around exact subproblem solves.

``restricted-neighbourhood-search`` improves one schedule by large neighbourhood
search (Shaw 1998; Pisinger and Ropke 2010) whose neighbourhoods are solved
exactly. Each iteration takes the items of the incumbent schedule, tasks or, in
permutation problems, jobs, in order of their first start, and frees a window
of ``window`` consecutive ones at a random place. Every other task keeps its
decisions: its mode, and its order on each resource it uses relative to the
other kept tasks. The restricted problem is the CP-SAT interval model of the
``cp-sat`` solver (see :class:`~dispatchatlas.solvers.exact._cpsat.IntervalModel`)
with those decisions added as constraints and the incumbent as a hint, so
CP-SAT places the freed tasks anywhere, in any mode, and retimes the others
around them, which the incumbent's start times alone would not allow.

The solution CP-SAT returns is turned back into a task order, by start time,
and its modes, and decoded by the shared schedule builder. For a regular
objective the decoded schedule starts no task later than CP-SAT's, since the
serial schedule-generation scheme never delays a task beyond a feasible start
of the same order. It replaces the incumbent when it is no worse, so the run
never moves to a worse schedule.

Budgets: every exact call is one iteration and one evaluation, the decoding
of its solution, or, when it finds none, a probe of the incumbent that keeps
evaluation limits binding. A call runs on one worker and stops after
``call_limit`` units of CP-SAT's deterministic time, or earlier at the run's
time limit, so runs without a time limit replay exactly from their seed. When
the window covers every task and CP-SAT proves the window optimal, the
schedule is optimal and the run ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, ClassVar

from dispatchatlas.core.errors import MissingDependencyError
from dispatchatlas.core.model import FEATURES, Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria, StopReason
from dispatchatlas.solvers.constructive import Items
from dispatchatlas.solvers.exact.common import (
    decimal_scale,
    primary_objective,
    scaled,
    serial_horizon,
    time_values,
)
from dispatchatlas.solvers.exact.cpsat import OBJECTIVES
from dispatchatlas.solvers.priorities import upward_ranks
from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.scheduling import EPSILON
from dispatchatlas.solvers.search import Candidate, Exhausted, Search

if TYPE_CHECKING:
    from types import ModuleType

    from dispatchatlas.solvers.scheduling import Decoded

__all__ = ["RestrictedNeighbourhoodSearch", "Window", "register"]

_DUE_DATE_OBJECTIVES = frozenset({"total_tardiness", "max_lateness", "tardy_tasks"})


@dataclass(frozen=True)
class Window:
    """The solution of one restricted problem.

    Attributes:
        order: Task indices by start time, then finish time, then index.
        modes: Mode of every task.
        proven: Whether CP-SAT proved the restricted problem optimal.
    """

    order: list[int]
    modes: list[int]
    proven: bool


def _load_cpsat() -> tuple[ModuleType, ModuleType]:
    """Import OR-Tools CP-SAT and the interval model, or explain what is missing."""
    try:
        from ortools.sat.python import cp_model  # noqa: PLC0415 - optional dependency

        from dispatchatlas.solvers.exact import _cpsat  # noqa: PLC0415 - loads OR-Tools
    except ModuleNotFoundError as error:
        if (error.name or "").partition(".")[0] != "ortools":
            raise
        raise MissingDependencyError(
            "ortools", "exact", "The restricted-neighbourhood-search solver"
        ) from error
    return cp_model, _cpsat


@dataclass(frozen=True)
class RestrictedNeighbourhoodSearch:
    """Large neighbourhood search whose neighbourhoods CP-SAT solves exactly.

    The first schedule decodes the tasks in decreasing upward rank, each in its
    earliest-finishing mode.

    Attributes:
        window: Items, tasks or jobs, freed per exact call.
        call_limit: CP-SAT deterministic time per exact call, in its own units
            of roughly one second.
    """

    window: int = 8
    call_limit: float = 1.0

    info: ClassVar[SolverInfo] = SolverInfo(
        id="restricted-neighbourhood-search",
        name="Restricted neighbourhood search",
        family="matheuristic",
        summary="Frees the tasks that start in a random window of the incumbent schedule, "
        "keeps every other task's mode and resource order, solves the restricted problem "
        "exactly with the CP-SAT interval model, and keeps the result when it is no worse.",
        references=("shaw1998using", "pisinger2010large"),
        supports=frozenset(FEATURES),
        requires=("exact",),
    )

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if not self.call_limit > 0.0:
            raise ValueError("call_limit must be positive")

    def warm_up(self) -> None:
        """Import OR-Tools and the model, which the first call in a process pays for.

        Raises:
            MissingDependencyError: If OR-Tools is not installed.
        """
        _load_cpsat()

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Improve the incumbent window by window until a stop criterion fires.

        Raises:
            UnsupportedProblemError: If the problem has an unsupported feature,
                an objective the interval model does not state, or time values
                it cannot scale to integers exactly.
            MissingDependencyError: If OR-Tools is not installed.
        """
        objective = primary_objective(problem, OBJECTIVES, self.info.id)
        cp_model, _cpsat = _load_cpsat()
        search = Search(problem, stop, seed, self.info)
        dues = objective in _DUE_DATE_OBJECTIVES
        scale = decimal_scale(time_values(problem, dues=dues), "time values of the problem")
        horizon = serial_horizon(problem, scale)
        ranks = upward_ranks(problem)
        items = Items(problem, jobs=problem.permutation)
        size = min(self.window, items.count)
        calls = improvements = 0
        with search:
            order = sorted(range(problem.size), key=lambda task: (-ranks[task], task))
            current = search.assess(order)
            while True:
                limit = search.stop.max_evaluations
                if limit is not None and search.evaluations >= limit:
                    raise Exhausted(StopReason.EVALUATION_LIMIT)
                search.next_iteration()
                starts = current.decoded.starts
                begins = [min(starts[task] for task in tasks) for tasks in items.members]
                by_start = sorted(range(items.count), key=lambda item: (begins[item], item))
                first = search.rng.randrange(items.count - size + 1)
                free = frozenset(items.tasks(by_start[first : first + size]))
                calls += 1
                found = self._solve_window(
                    (cp_model, _cpsat),
                    problem,
                    current,
                    free,
                    (scale, horizon, objective),
                    search.remaining_time(),
                )
                if found is None:
                    search.probe(current.order, current.modes)
                    continue
                candidate = search.assess(found.order, found.modes)
                if candidate.score < current.score:
                    improvements += 1
                if candidate.score <= current.score:
                    current = candidate
                if found.proven and size == items.count:
                    break
        return search.result(extras={"exact_calls": calls, "improvements": improvements})

    def _solve_window(
        self,
        modules: tuple[ModuleType, ModuleType],
        problem: Problem,
        current: Candidate,
        free: frozenset[int],
        statement: tuple[int, int, str],
        remaining: float | None,
    ) -> Window | None:
        """Solve the restricted problem in which only the tasks in ``free`` are free.

        Args:
            modules: The CP-SAT module and the interval-model module.
            problem: The problem.
            current: The incumbent, whose modes and resource orders the other
                tasks keep.
            free: Indices of the freed tasks.
            statement: Time scale, horizon, and name of the objective.
            remaining: Seconds left before the run's time limit, if any.

        Returns:
            The solution, or ``None`` when CP-SAT found none within its limits.
        """
        cp_model, _cpsat = modules
        scale, horizon, objective_name = statement
        model = _cpsat.IntervalModel(problem, scale, horizon)
        model.model.minimize(model.objective(objective_name).expression)
        _restrict(model, problem, current.decoded, free, scale)
        solver = cp_model.CpSolver()
        parameters = solver.parameters
        parameters.num_workers = 1
        parameters.max_deterministic_time = self.call_limit
        if remaining is not None:
            parameters.max_time_in_seconds = remaining
        status = solver.solve(model.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        starts = [solver.value(variable) for variable in model.starts]
        ends = [solver.value(variable) for variable in model.ends]
        modes = [
            next(option for option, literal in enumerate(chosen) if solver.boolean_value(literal))
            for chosen in model.chosen
        ]
        order = sorted(range(problem.size), key=lambda task: (starts[task], ends[task], task))
        return Window(order, modes, status == cp_model.OPTIMAL)


def _restrict(
    model: Any, problem: Problem, decoded: Decoded, free: frozenset[int], scale: int
) -> None:
    """Hint the incumbent and keep the decisions of every task outside ``free``.

    A kept task keeps its mode and, on every resource its mode uses, its order
    relative to the kept tasks that start after it finishes.
    """
    cp = model.model
    users: dict[str, list[tuple[float, float, int]]] = {}
    for task, spec in enumerate(problem.tasks):
        chosen = decoded.modes[task]
        cp.add_hint(model.starts[task], scaled(decoded.starts[task], scale))
        for option, literal in enumerate(model.chosen[task]):
            cp.add_hint(literal, option == chosen)
        if task in free:
            continue
        cp.add(model.chosen[task][chosen] == 1)
        for resource in spec.modes[chosen].demands:
            users.setdefault(resource, []).append((decoded.starts[task], decoded.ends[task], task))
    for entries in users.values():
        entries.sort()
        for (_, finish, before), (begin, _, after) in pairwise(entries):
            if finish <= begin + EPSILON:
                cp.add(model.ends[before] <= model.starts[after])


def register(registry: SolverRegistry) -> None:
    """Add the hybrid solvers to ``registry``."""
    registry.register(RestrictedNeighbourhoodSearch)
