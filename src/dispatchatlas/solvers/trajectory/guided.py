"""Guided local search (Voudouris and Tsang 1999).

Guided local search escapes local optima by changing the objective the local
search minimises. A solution exhibits **features**, each with a fixed cost:

- an item (a task, or a job in permutation problems) at a sequence position,
  whose cost is the item's mean work;
- a multi-mode task running in a mode, whose cost is that mode's duration.

The local search minimises the augmented objective ``f + λ · Σ p_i`` over the
features ``i`` present in the solution, where ``p_i`` counts how often feature
``i`` has been penalised. At each local optimum the features present with the
highest utility ``cost / (1 + p_i)`` gain one penalty unit, so costly features
that keep appearing in local optima are driven out, and each new penalty on a
feature matters less than the last. The best solution under the true objective
is what the run reports.

The local search is the fast local search of the same authors: the
neighbourhood is split into one sub-neighbourhood per item, the exchanges of
the item with every other item it may swap with plus the other mode settings
of its tasks, each with an activation bit. Active items are scanned in index
order. The moves of an item's sub-neighbourhood are tried in random order, and
the first that improves the augmented objective is made and activates the items
it involves; an item whose moves all fail is deactivated. The descent ends when
no item is active. After a penalty update, only the items of the penalised
features are activated.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory.space import Move, SequenceSpace, Solution

__all__ = ["Feature", "GuidedLocalSearch", "Penalties"]

type Feature = tuple[int, int, int]
"""``(0, item, position)`` for an item at a position, ``(1, task, mode)`` for a task in a mode."""


class Penalties:
    """Feature penalties and the augmented objective of one run.

    Args:
        space: The space the features describe.

    Attributes:
        weight: The penalty weight ``λ``.
        counts: Penalty count of every penalised feature.
    """

    def __init__(self, space: SequenceSpace) -> None:
        self._space = space
        self._work = space.items.work
        self._durations = [
            [mode.duration for mode in task.modes] for task in space.search.problem.tasks
        ]
        self.weight = 0.0
        self.counts: dict[Feature, int] = {}

    @property
    def size(self) -> int:
        """Number of features present in every solution."""
        return self._space.items.count + len(self._space.flexible)

    def features(self, solution: Solution) -> list[tuple[Feature, float]]:
        """Return the features present in ``solution`` with their costs."""
        present: list[tuple[Feature, float]] = [
            ((0, item, position), self._work[item])
            for position, item in enumerate(solution.sequence)
        ]
        running = solution.decoded.modes
        present += [
            ((1, task, running[task]), self._durations[task][running[task]])
            for task in self._space.flexible
        ]
        return present

    def total(self, solution: Solution) -> int:
        """Return the sum of the penalties of the features present in ``solution``."""
        if not self.counts:
            return 0
        return sum(self.counts.get(feature, 0) for feature, _ in self.features(solution))

    def augmented(self, solution: Solution) -> Score:
        """Return the score whose objective carries ``λ`` times the solution's penalties."""
        score = solution.score
        return Score(score.violation, score.value + self.weight * self.total(solution))

    def penalise(self, solution: Solution) -> list[int]:
        """Penalise the features of maximum utility in ``solution``.

        Returns:
            The items the penalised features belong to, in index order.
        """
        present = self.features(solution)
        utilities = [cost / (1.0 + self.counts.get(feature, 0)) for feature, cost in present]
        top = max(utilities)
        item_of = self._space.items.item_of
        touched: set[int] = set()
        for (feature, _), utility in zip(present, utilities, strict=True):
            if math.isclose(utility, top, rel_tol=1e-12, abs_tol=0.0):
                self.counts[feature] = self.counts.get(feature, 0) + 1
                touched.add(feature[1] if feature[0] == 0 else item_of[feature[1]])
        return sorted(touched)


def fast_descent(
    space: SequenceSpace, penalties: Penalties, current: Solution, active: Iterable[int]
) -> Solution:
    """Descend on the augmented objective by fast local search from ``current``.

    Args:
        space: The space of the run.
        penalties: The penalties that define the augmented objective.
        current: The starting solution.
        active: Items whose sub-neighbourhoods start active.

    Returns:
        A solution that no move in the sub-neighbourhood of any item scanned
        since its last change improves under the augmented objective.
    """
    queue = sorted(set(active))
    queued = set(queue)
    value = penalties.augmented(current)
    while queue:
        item = heapq.heappop(queue)
        queued.discard(item)
        step = _first_improvement(space, penalties, current, value, item)
        if step is None:
            continue
        current, partner, value = step
        for touched in (item, partner):
            if touched is not None and touched not in queued:
                heapq.heappush(queue, touched)
                queued.add(touched)
    return current


def _first_improvement(
    space: SequenceSpace, penalties: Penalties, current: Solution, value: Score, item: int
) -> tuple[Solution, int | None, Score] | None:
    """Scan the sub-neighbourhood of ``item`` in random order for an improving move.

    Returns:
        The first neighbour whose augmented score beats ``value``, the item it
        swapped with (``None`` for a mode move), and its augmented score, or
        ``None`` when no move of the sub-neighbourhood improves.
    """
    sequence, modes = current.sequence, current.modes
    index = sequence.index(item)
    moves = [
        Move("swap", min(index, other), max(index, other)) for other in space.swaps(sequence, index)
    ]
    if modes is not None:
        tasks = space.search.problem.tasks
        for task in space.items.members[item]:
            if len(tasks[task].modes) > 1:
                running = current.decoded.modes[task]
                moves += [
                    Move("mode", task, setting)
                    for setting in space.settings(task, modes[task], running)
                ]
    space.search.rng.shuffle(moves)
    for move in moves:
        candidate = space.evaluate(*space.apply(sequence, modes, move))
        augmented = penalties.augmented(candidate)
        if augmented < value:
            partner = None
            if move.kind == "swap":
                partner = sequence[move.second if move.first == index else move.first]
            return candidate, partner, augmented
    return None


@dataclass(frozen=True)
class GuidedLocalSearch:
    """Guided local search (Voudouris and Tsang 1999) with fast local search.

    The search starts from a random precedence-feasible sequence and descends
    with every item active and no penalty. The penalty weight is then fixed at
    ``λ = alpha · |f(s*)| / F``, where ``s*`` is that first local optimum and
    ``F`` the number of features present in a solution, so one penalty unit is
    worth ``alpha`` times the objective per feature. Each iteration penalises
    the features of maximum utility in the current local optimum and descends
    again from it.

    Attributes:
        alpha: Coefficient of the penalty weight.
    """

    alpha: float = 0.3

    info: ClassVar[SolverInfo] = SolverInfo(
        id="guided-local-search",
        name="Guided local search",
        family="local-search",
        summary="Descends on the objective plus penalties on solution features, items at "
        "positions and tasks in modes, and at each local optimum penalises the features "
        "whose cost over one plus their penalty is highest.",
        references=("voudouris1999guided",),
    )

    def __post_init__(self) -> None:
        if not self.alpha > 0.0:
            raise ValueError("alpha must be positive")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Descend, penalise, and descend again until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space = SequenceSpace(search)
            current = space.random_start()
            if space.movable:
                penalties = Penalties(space)
                current = fast_descent(space, penalties, current, range(space.items.count))
                penalties.weight = self.alpha * abs(current.score.value) / penalties.size
                while True:
                    search.next_iteration()
                    touched = penalties.penalise(current)
                    current = fast_descent(space, penalties, current, touched)
        return search.result()
