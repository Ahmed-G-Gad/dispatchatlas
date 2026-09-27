"""Priority rules and random sampling.

A priority rule ranks the tasks by one static attribute and decodes that order
once with the shared schedule builder, which also picks each task's
earliest-finishing mode. Rules are fast, deterministic baselines. Surveys of
the classic rules: Panwalkar and Iskander (1977).

``random-search`` is the stochastic baseline every metaheuristic should beat:
it decodes random precedence-feasible orders until the budget runs out.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.operators import random_modes, random_topological_order
from dispatchatlas.solvers.priorities import (
    effective_due_dates,
    longest_tails,
    min_durations,
    remaining_group_work,
    upward_ranks,
)
from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.search import Search

__all__ = [
    "EarliestDueDate",
    "Heft",
    "InputOrder",
    "LeastSlack",
    "LongestProcessingTime",
    "LongestRemainingPath",
    "MostWorkRemaining",
    "RandomSearch",
    "ShortestProcessingTime",
    "register",
]


def _rule_info(solver_id: str, name: str, summary: str, *references: str) -> SolverInfo:
    return SolverInfo(
        id=solver_id,
        name=name,
        family="dispatching-rule",
        summary=summary,
        references=references or ("panwalkar1977survey", "kolisch1996serial"),
        stochastic=False,
        iterative=False,
    )


def _solve_by_keys(
    solver: object,
    info: SolverInfo,
    problem: Problem,
    stop: StopCriteria,
    seed: int,
    keys: Callable[[Problem], Sequence[float]],
) -> SolveResult:
    del solver
    search = Search(problem, stop, seed, info)
    values = keys(problem)
    with search:
        search.evaluate(sorted(range(problem.size), key=lambda i: (values[i], i)))
    return search.result()


@dataclass(frozen=True)
class InputOrder:
    """Schedule tasks by release time, then in the order the problem lists them."""

    info: ClassVar[SolverInfo] = _rule_info(
        "input-order",
        "Input order",
        "Decodes tasks by release time, then in their listed order, a first-come "
        "first-served baseline.",
        "kolisch1996serial",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Decode the input order."""
        return _solve_by_keys(
            self, self.info, problem, stop, seed, lambda p: [t.release for t in p.tasks]
        )


@dataclass(frozen=True)
class ShortestProcessingTime:
    """Schedule shorter tasks first."""

    info: ClassVar[SolverInfo] = _rule_info(
        "shortest-processing-time",
        "Shortest processing time",
        "Decodes tasks in increasing order of their shortest duration.",
        "smith1956various",
        "panwalkar1977survey",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Decode tasks by increasing duration."""
        return _solve_by_keys(self, self.info, problem, stop, seed, min_durations)


@dataclass(frozen=True)
class LongestProcessingTime:
    """Schedule longer tasks first."""

    info: ClassVar[SolverInfo] = _rule_info(
        "longest-processing-time",
        "Longest processing time",
        "Decodes tasks in decreasing order of their shortest duration, which "
        "balances load on parallel resources.",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Decode tasks by decreasing duration."""
        return _solve_by_keys(
            self, self.info, problem, stop, seed, lambda p: [-d for d in min_durations(p)]
        )


@dataclass(frozen=True)
class EarliestDueDate:
    """Schedule tasks with earlier due dates first."""

    info: ClassVar[SolverInfo] = _rule_info(
        "earliest-due-date",
        "Earliest due date",
        "Decodes tasks by increasing due date, inherited from the task's group "
        "when the task has none.",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Decode tasks by due date."""
        return _solve_by_keys(self, self.info, problem, stop, seed, effective_due_dates)


@dataclass(frozen=True)
class MostWorkRemaining:
    """Schedule tasks whose job has the most work left first."""

    info: ClassVar[SolverInfo] = _rule_info(
        "most-work-remaining",
        "Most work remaining",
        "Decodes tasks by decreasing work left in their group from the task on.",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Decode tasks by decreasing remaining group work."""
        return _solve_by_keys(
            self, self.info, problem, stop, seed, lambda p: [-w for w in remaining_group_work(p)]
        )


@dataclass(frozen=True)
class LongestRemainingPath:
    """Schedule tasks with the longest path to the end of the problem first."""

    info: ClassVar[SolverInfo] = _rule_info(
        "longest-remaining-path",
        "Longest remaining path",
        "Decodes tasks by decreasing length of the longest dependency path from "
        "their start to the end of the problem, the critical-path rule.",
        "kolisch1996serial",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Decode tasks by decreasing tail length."""
        return _solve_by_keys(
            self, self.info, problem, stop, seed, lambda p: [-t for t in longest_tails(p)]
        )


@dataclass(frozen=True)
class LeastSlack:
    """Schedule tasks with the latest start before their due date first."""

    info: ClassVar[SolverInfo] = _rule_info(
        "least-slack",
        "Least slack",
        "Decodes tasks by increasing slack, the due date minus the longest "
        "remaining path, which approximates the latest start time.",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Decode tasks by increasing slack."""

        def slack(p: Problem) -> list[float]:
            dues = effective_due_dates(p)
            tails = longest_tails(p)
            return [
                due - tail if not math.isinf(due) else math.inf
                for due, tail in zip(dues, tails, strict=True)
            ]

        return _solve_by_keys(self, self.info, problem, stop, seed, slack)


@dataclass(frozen=True)
class Heft:
    """Heterogeneous earliest finish time list scheduling."""

    info: ClassVar[SolverInfo] = SolverInfo(
        id="heft",
        name="Heterogeneous earliest finish time",
        family="list-scheduling",
        summary="Ranks tasks by upward rank, the mean duration plus the longest "
        "mean-communication path to the end, then places each on the mode that "
        "finishes it earliest, filling idle gaps.",
        references=("topcuoglu2002performance",),
        stochastic=False,
        iterative=False,
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Decode tasks by decreasing upward rank with earliest-finish mode choice."""
        return _solve_by_keys(
            self, self.info, problem, stop, seed, lambda p: [-r for r in upward_ranks(p)]
        )


@dataclass(frozen=True)
class RandomSearch:
    """Decode random precedence-feasible orders and keep the best.

    Attributes:
        random_modes: Draw every task's mode at random too, instead of letting
            the builder pick the earliest-finishing mode.
    """

    random_modes: bool = False

    info: ClassVar[SolverInfo] = SolverInfo(
        id="random-search",
        name="Random search",
        family="random-sampling",
        summary="Decodes uniformly random precedence-feasible orders until the "
        "budget runs out and returns the best, the baseline any search must beat.",
        references=("kolisch1996serial",),
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Sample random orders within the stop criteria."""
        search = Search(problem, stop, seed, self.info)
        with search:
            while True:
                search.next_iteration()
                modes = random_modes(problem, search.rng) if self.random_modes else None
                search.evaluate(random_topological_order(problem, search.rng), modes)
        return search.result()


def register(registry: SolverRegistry) -> None:
    """Add the rules and the random-search baseline to ``registry``."""
    for factory in (
        InputOrder,
        ShortestProcessingTime,
        LongestProcessingTime,
        EarliestDueDate,
        MostWorkRemaining,
        LongestRemainingPath,
        LeastSlack,
        Heft,
        RandomSearch,
    ):
        registry.register(factory)
