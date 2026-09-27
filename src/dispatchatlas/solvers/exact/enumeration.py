"""The ``enumeration`` solver: exhaustive search over task orders and mode vectors.

It needs no third-party package and serves as the reference that the model
based solvers are tested against.

**Why it is exact.** Fix an optimal schedule and list its tasks by start time,
zero-duration tasks first among equal starts, other ties broken along the
dependencies. Decoding that list with the schedule's own modes by the serial
schedule builder, which places each task at its earliest feasible start and
may fill idle gaps, starts every task no later than the optimal schedule
does: when a task is placed, the tasks placed before it started no later than
in the optimal schedule and therefore occupy each resource at any later
moment at most as much, the instants of zero-duration tasks placed before it
lie no later than its own start, and its predecessors finish no later. Every
objective this solver accepts is regular, never better when a task finishes
later, so the decoded schedule is optimal as well, and it meets every hard
deadline the optimal schedule meets. Enumerating every
precedence-feasible order and every mode vector therefore reaches an optimal
schedule (Kolisch 1996; Sprecher, Kolisch and Drexl 1995). The argument covers
release times, lags, transfer times, modes, and cumulative resources, but not
sequence-dependent setup times, so problems with setups are refused.

**Permutation problems.** The schedule builder decodes one group after
another, appending every task to its machines, zero-duration tasks included.
The solver therefore enumerates group orders and mode vectors. When every
dependency stays within a group, each machine then processes the groups in
the enumerated order, and each task starts as early as that order allows,
which is the best schedule of the order. Permutation problems with a
dependency between two groups are refused.

**Pruning.** Two orders that reach the same partial schedule have the same
completions, so the search expands each distinct partial schedule once. A
partial schedule in which a placed task already misses its deadline is not
expanded, because placed tasks never move.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from dispatchatlas.core.errors import UnsupportedProblemError
from dispatchatlas.core.model import FEATURES, Problem, Scalar
from dispatchatlas.core.solver import SolveResult, SolverInfo, Status, StopCriteria, StopReason
from dispatchatlas.solvers.exact.common import primary_objective
from dispatchatlas.solvers.scheduling import Builder
from dispatchatlas.solvers.search import Exhausted, Search

__all__ = ["OBJECTIVES", "Enumeration"]

OBJECTIVES = frozenset(
    {
        "makespan",
        "total_completion_time",
        "total_flow_time",
        "mean_response_time",
        "total_tardiness",
        "max_lateness",
        "tardy_tasks",
        "energy",
        "cost",
    }
)
"""The built-in objectives, all regular once the modes are fixed."""


@dataclass(frozen=True)
class Enumeration:
    """Find an optimal schedule by decoding every task order and mode vector.

    Statuses: ``OPTIMAL`` once the enumeration is complete, with ``bound``
    equal to the value; ``INFEASIBLE`` when it is complete and no schedule
    meets every hard deadline. A run cut short by a stop criterion reports
    the best schedule so far as ``FEASIBLE``, as ``VIOLATED`` if it misses a
    deadline, or ``UNKNOWN`` without one, never ``OPTIMAL``.

    The solver is not iterative: without stop criteria it runs until the
    enumeration is complete. It honours any criteria it is given. Each
    complete schedule it scores is one evaluation and one iteration. Its
    extras record ``states``, the number of distinct partial schedules
    expanded, and ``complete``.

    Attributes:
        max_tasks: Largest number of tasks accepted. The work grows with the
            factorial of the task count, so larger problems are refused.
    """

    max_tasks: int = 8

    info: ClassVar[SolverInfo] = SolverInfo(
        id="enumeration",
        name="Exhaustive enumeration",
        family="exact",
        summary=(
            "Decodes every precedence-feasible task order under every mode vector with the "
            "serial schedule builder, expanding each distinct partial schedule once, and "
            "returns the best schedule, which is optimal for regular objectives."
        ),
        references=("kolisch1996serial", "sprecher1995semiactive"),
        stochastic=False,
        iterative=False,
        supports=frozenset(FEATURES) - {"setups"},
    )

    def __post_init__(self) -> None:
        value = self.max_tasks
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"max_tasks must be a positive integer, got {value!r}")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Enumerate the schedules of ``problem`` and return the best.

        Args:
            problem: The problem.
            stop: Optional stop criteria. Without any, the run is complete.
            seed: Seed recorded by the search; the enumeration uses no
                randomness.

        Returns:
            The result.

        Raises:
            UnsupportedProblemError: If the problem has setups, more than
                ``max_tasks`` tasks, or an objective that is not built in, or
                is a permutation problem with a dependency between groups.
        """
        search = Search(problem, stop, seed, self.info)
        primary_objective(problem, OBJECTIVES, self.info.id)
        if problem.size > self.max_tasks:
            raise UnsupportedProblemError(
                f"solver {self.info.id!r} accepts at most {self.max_tasks} tasks; "
                f"problem {problem.id!r} has {problem.size}"
            )
        if problem.permutation:
            _check_permutation(problem)
        tree = _Tree(problem, search)
        complete = False
        with search:
            tree.expand(search.evaluator.decoder.builder())
            complete = True
        return _result(search, tree, complete)


def _group_keys(problem: Problem) -> list[str]:
    return [task.group if task.group is not None else task.id for task in problem.tasks]


def _check_permutation(problem: Problem) -> None:
    """Refuse a permutation problem whose group-major decoding may break the group order.

    A dependency between two groups can make the builder place a task behind
    one of a later group. The data model already guarantees the other
    conditions: capacity-1 machines held whole, each used at most once per group.
    """
    keys = _group_keys(problem)
    index = problem.task_index
    for dependency in problem.dependencies:
        if keys[index[dependency.before]] != keys[index[dependency.after]]:
            raise UnsupportedProblemError(
                f"dependency {dependency.before!r} -> {dependency.after!r} joins two groups; "
                "the enumeration of group orders needs dependencies within groups"
            )


type _State = tuple[tuple[int, ...], tuple[float, ...]]


class _Tree:
    """Depth-first expansion of partial schedules, one placed task per level."""

    def __init__(self, problem: Problem, search: Search) -> None:
        self.problem = problem
        self.search = search
        self.seen: set[_State] = set()
        self.deadlines = [task.deadline for task in problem.tasks]
        self.modes = [len(task.modes) for task in problem.tasks]
        keys = _group_keys(problem)
        rank = {task: n for n, task in enumerate(problem.topological_order)}
        members: dict[str, list[int]] = {}
        for task in sorted(range(problem.size), key=rank.__getitem__):
            members.setdefault(keys[task], []).append(task)
        self.keys = keys
        self.members = members

    def expand(self, builder: Builder) -> None:
        """Place every candidate next task in every mode, then recurse."""
        if len(builder.sequence) == self.problem.size:
            self.search.next_iteration()
            self.search.submit(builder.decoded())
            return
        for task in self._candidates(builder):
            deadline = self.deadlines[task]
            for mode in range(self.modes[task]):
                self._check_time()
                child = builder.copy()
                _, end = child.place(task, mode)
                if deadline is not None and end > deadline:
                    continue
                state = (tuple(child.modes), tuple(child.starts))
                if state in self.seen:
                    continue
                self.seen.add(state)
                self.expand(child)

    def _candidates(self, builder: Builder) -> list[int]:
        """Return the tasks that may be placed next."""
        if not self.problem.permutation:
            return [task for task in range(self.problem.size) if builder.eligible(task)]
        # Groups are placed whole, their tasks in topological order.
        if builder.sequence:
            current = self.members[self.keys[builder.sequence[-1]]]
            waiting = [task for task in current if not builder.placed(task)]
            if waiting:
                return waiting[:1]
        return [tasks[0] for tasks in self.members.values() if not builder.placed(tasks[0])]

    def _check_time(self) -> None:
        limit = self.search.stop.time_limit
        if limit is not None and self.search.elapsed >= limit:
            raise Exhausted(StopReason.TIME_LIMIT)


def _result(search: Search, tree: _Tree, complete: bool) -> SolveResult:
    extras: dict[str, Scalar] = {"states": len(tree.seen), "complete": complete}
    best = search.best
    if best is None or (complete and not best.score.feasible):
        return SolveResult(
            solver=search.info.id,
            problem=search.problem.id,
            objective=search.evaluator.objective.name,
            status=Status.INFEASIBLE if complete else Status.UNKNOWN,
            schedule=None,
            objectives={},
            stop_reason=search.reason or StopReason.COMPLETED,
            elapsed=search.elapsed,
            evaluations=search.evaluations,
            iterations=search.iterations,
            extras=extras,
        )
    result = search.result(extras)
    if complete:
        return replace(result, status=Status.OPTIMAL, bound=result.value)
    return result
