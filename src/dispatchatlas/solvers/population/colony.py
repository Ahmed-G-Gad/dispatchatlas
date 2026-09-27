"""Ant colony optimisation: the MAX-MIN Ant System (Stützle and Hoos 2000)."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.constructive import Items
from dispatchatlas.solvers.population._construction import construct, roulette
from dispatchatlas.solvers.priorities import (
    effective_due_dates,
    longest_tails,
    min_durations,
    upward_ranks,
)
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory.space import SequenceSpace, Solution

__all__ = ["HEURISTICS", "AntColonyOptimization", "Trails", "desirability", "heuristic_rule"]

HEURISTICS: Mapping[str, Callable[[Problem], list[float]]] = MappingProxyType(
    {
        "upward-rank": upward_ranks,
        "longest-tail": longest_tails,
        "earliest-due-date": lambda problem: [-due for due in effective_due_dates(problem)],
        "shortest-processing-time": lambda problem: [-time for time in min_durations(problem)],
    }
)
"""Task priorities the ants' desirability can follow, larger first, from
:mod:`~dispatchatlas.solvers.priorities`: the upward rank of HEFT, the longest
path to the end, the earliest due date, and the shortest duration."""

_BY_OBJECTIVE: Mapping[str, str] = MappingProxyType(
    {
        "total_tardiness": "earliest-due-date",
        "max_lateness": "earliest-due-date",
        "tardy_tasks": "earliest-due-date",
        "total_completion_time": "shortest-processing-time",
        "total_flow_time": "shortest-processing-time",
        "mean_response_time": "shortest-processing-time",
    }
)


def heuristic_rule(name: str, objective: str) -> str | None:
    """Return the priority rule of :data:`HEURISTICS` that ``name`` selects, or ``None``.

    ``"auto"`` picks a rule suited to the primary ``objective``: the earliest due
    date for due-date objectives, the shortest duration for completion-time
    objectives, and the upward rank otherwise. ``"none"`` selects no rule.
    """
    if name == "none":
        return None
    if name == "auto":
        return _BY_OBJECTIVE.get(objective, "upward-rank")
    return name


class Trails:
    """Pheromone trails with the bounds of the MAX-MIN Ant System.

    Each row holds the trails of one decision's options: the positions an item
    may take, or the modes of a task. Trails start at the upper bound. An
    update evaporates every trail by the factor ``1 - evaporation``, adds one
    unit to the options of the depositing solution, and clips every trail to
    ``[lower, upper]``.

    With a unit deposit a trail reinforced at every update tends to
    ``upper = 1 / evaporation``, the ``tau_max`` of Stützle and Hoos (2000) for
    this deposit. The lower bound follows their rule. Once the trails have
    converged, those of one solution's options at ``upper`` and all others at
    ``lower``, an ant guided by the trails alone rebuilds that solution with
    probability ``best_probability``, given ``decisions`` decisions of
    ``choices`` options on average. This gives
    ``lower = upper (1 - r) / ((choices - 1) r)`` with
    ``r = best_probability ** (1 / decisions)``. When that exceeds ``upper``, as
    for very few decisions, ``lower = upper`` and the trails carry no
    information. A unit deposit, rather than the published inverse objective
    value, keeps the bounds independent of the objective's scale and sign.

    Args:
        widths: Options of each decision.
        evaporation: Fraction of every trail lost per update, in (0, 1).
        best_probability: The ``p_best`` of the lower bound, in (0, 1).
        decisions: Decisions that build one solution.
        choices: Mean options per decision.

    Attributes:
        values: The trails, by decision, then option.
        lower: Lower bound of every trail.
        upper: Upper bound of every trail.
    """

    def __init__(
        self,
        widths: Sequence[int],
        evaporation: float,
        best_probability: float,
        decisions: int,
        choices: float,
    ) -> None:
        self.evaporation = evaporation
        self.upper = 1.0 / evaporation
        root = best_probability ** (1.0 / max(1, decisions))
        if choices > 1.0:
            self.lower = min(self.upper, self.upper * (1.0 - root) / ((choices - 1.0) * root))
        else:
            self.lower = self.upper
        self.values = [[self.upper] * width for width in widths]

    def reinforce(self, options: Iterable[tuple[int, int]]) -> None:
        """Evaporate every trail, deposit one unit on each ``(decision, option)``, and clip."""
        keep = 1.0 - self.evaporation
        for row in self.values:
            for option, value in enumerate(row):
                row[option] = value * keep
        for decision, option in options:
            self.values[decision][option] += 1.0
        for row in self.values:
            for option, value in enumerate(row):
                row[option] = min(self.upper, max(self.lower, value))


def desirability(items: Items, priority: Sequence[float]) -> list[float]:
    """Return the heuristic desirability of each item from a task priority.

    An item's priority is the largest priority of its tasks. The desirability
    is ``1 - h / k`` for an item with ``h`` of the ``k`` items strictly ahead of
    it: 1 for the items of highest priority down to ``1 / k``. Using ranks
    rather than raw priorities keeps the desirability positive and independent
    of the time unit.
    """
    item_priority = [max(priority[task] for task in members) for members in items.members]
    ordered = sorted(item_priority)
    count = len(item_priority)
    return [bisect_right(ordered, value) / count for value in item_priority]


@dataclass(frozen=True)
class AntColonyOptimization:
    """The MAX-MIN Ant System of Stützle and Hoos (2000) over task sequences.

    Ants build item sequences (tasks, or whole jobs in permutation problems)
    one position at a time, as in the ant colony optimisation of Merkle,
    Middendorf and Schmeck (2002) for project scheduling, which builds on the
    ant system of Dorigo, Maniezzo and Colorni (1996). At position ``j`` an ant
    draws the next item among the eligible ones, whose predecessors are
    placed, with probability proportional to ``tau(i, j)^alpha eta(i)^beta``,
    where ``tau(i, j)`` is the pheromone of item ``i`` at position ``j`` and
    ``eta(i)`` its :func:`desirability` under a priority rule (see
    :func:`heuristic_rule`), or 1 without one. The default rule depends on the
    primary objective, as heuristic information does in the ant colony methods
    of the scheduling literature: the earliest due date for due-date
    objectives, the shortest duration for completion-time objectives, and the
    upward rank of HEFT (Topcuoglu, Hariri and Wu 2002) otherwise. The
    desirability spans ranks from ``1 / k`` to 1, so the default ``beta = 1``
    lets the pheromone, not the rule, dominate once trails differ.

    Multi-mode tasks run in the mode the builder finds earliest-finishing
    (``modes="builder"``), or in a mode each ant draws with probability
    proportional to ``tau(t, m)^alpha``, a pheromone on (task, mode)
    (``modes="pheromone"``).

    After all ants of an iteration are evaluated, one solution deposits
    pheromone on its (item, position) pairs and, with mode pheromone, on its
    (task, mode) pairs: the iteration's best, or the best found so far. The
    trails obey the bounds of :class:`Trails`, which keep every option
    possible. One iteration is one colony of ``ants`` solutions.

    Attributes:
        ants: Solutions built per iteration.
        alpha: Weight ``alpha`` of the pheromone.
        beta: Weight ``beta`` of the desirability.
        heuristic: The priority rule of the desirability: ``"auto"``, ``"none"``,
            or a key of :data:`HEURISTICS`.
        evaporation: Fraction of pheromone lost per iteration.
        best_probability: The ``p_best`` that sets the lower trail bound.
        deposit: ``"iteration-best"`` or ``"best-so-far"``: whose solution
            deposits pheromone.
        modes: ``"builder"`` or ``"pheromone"``: how multi-mode tasks get their modes.
    """

    ants: int = 10
    alpha: float = 1.0
    beta: float = 1.0
    heuristic: str = "auto"
    evaporation: float = 0.1
    best_probability: float = 0.05
    deposit: Literal["iteration-best", "best-so-far"] = "iteration-best"
    modes: Literal["builder", "pheromone"] = "builder"

    info: ClassVar[SolverInfo] = SolverInfo(
        id="ant-colony-optimization",
        name="MAX-MIN Ant System",
        family="population",
        summary="Builds task sequences position by position from bounded pheromone trails on "
        "(task, position) pairs and a priority-rule desirability, and lets the best "
        "solution reinforce its trails.",
        references=(
            "stutzle2000max",
            "dorigo1996ant",
            "merkle2002ant",
            "topcuoglu2002performance",
        ),
    )

    def __post_init__(self) -> None:
        if self.ants < 1:
            raise ValueError("ants must be at least 1")
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError("alpha and beta must be non-negative")
        if self.heuristic not in ("auto", "none", *HEURISTICS):
            raise ValueError(f"unknown heuristic {self.heuristic!r}")
        if not 0.0 < self.evaporation < 1.0:
            raise ValueError("evaporation must lie in (0, 1)")
        if not 0.0 < self.best_probability < 1.0:
            raise ValueError("best_probability must lie in (0, 1)")
        if self.deposit not in ("iteration-best", "best-so-far"):
            raise ValueError("deposit must be 'iteration-best' or 'best-so-far'")
        if self.modes not in ("builder", "pheromone"):
            raise ValueError("modes must be 'builder' or 'pheromone'")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Build and reinforce until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space = SequenceSpace(search)
            if space.movable:
                self._search(space)
            else:
                space.random_start()
        return search.result()

    def position_trails(self, items: Items) -> Trails:
        """Return fresh (item, position) trails for ``items``."""
        count = items.count
        return Trails([count] * count, self.evaporation, self.best_probability, count, count / 2)

    def mode_trails(self, problem: Problem) -> Trails | None:
        """Return fresh (task, mode) trails, or ``None`` when modes come from the builder."""
        flexible = [len(task.modes) for task in problem.tasks if len(task.modes) > 1]
        if self.modes == "builder" or not flexible:
            return None
        widths = [len(task.modes) for task in problem.tasks]
        choices = sum(flexible) / len(flexible)
        return Trails(widths, self.evaporation, self.best_probability, len(flexible), choices)

    def _search(self, space: SequenceSpace) -> None:
        search = space.search
        problem, rng = search.problem, search.rng
        items = space.items
        rule = heuristic_rule(self.heuristic, search.evaluator.objective.name)
        if rule is None:
            eta = [1.0] * items.count
        else:
            priority = HEURISTICS[rule](problem)
            eta = [math.pow(value, self.beta) for value in desirability(items, priority)]
        trails = self.position_trails(items)
        mode_trails = self.mode_trails(problem)
        flexible = set(space.flexible)
        alpha = self.alpha

        def weight(item: int, position: int) -> float:
            return math.pow(trails.values[item][position], alpha) * eta[item]

        best: Solution | None = None
        while True:
            search.next_iteration()
            colony: list[Solution] = []
            for _ in range(self.ants):
                sequence = construct(items, weight, rng)
                if mode_trails is None:
                    modes = space.automatic
                else:
                    modes = tuple(
                        roulette([math.pow(trail, alpha) for trail in row], rng)
                        if task in flexible
                        else 0
                        for task, row in enumerate(mode_trails.values)
                    )
                colony.append(space.evaluate(sequence, modes))
            leader = min(colony, key=lambda solution: solution.score)
            if best is None or leader.score < best.score:
                best = leader
            depositor = leader if self.deposit == "iteration-best" else best
            trails.reinforce((item, position) for position, item in enumerate(depositor.sequence))
            if mode_trails is not None:
                running = depositor.decoded.modes
                mode_trails.reinforce((task, running[task]) for task in space.flexible)
