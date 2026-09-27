"""Scatter search over task sequences (Glover 1998; Martí, Laguna and Glover 2006)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.constructive import Items
from dispatchatlas.solvers.population._construction import construct
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory.space import SequenceSpace, Solution

__all__ = ["ScatterSearch", "combine", "distance"]


def distance(first: Solution, second: Solution) -> int:
    """Return how far apart two solutions are.

    The sum over items of the difference between their positions in the two
    sequences (Spearman's footrule), plus the number of tasks whose mode
    settings differ.
    """
    position = [0] * len(second.sequence)
    for index, item in enumerate(second.sequence):
        position[item] = index
    gap = sum(abs(index - position[item]) for index, item in enumerate(first.sequence))
    if first.modes is not None and second.modes is not None:
        gap += sum(a != b for a, b in zip(first.modes, second.modes, strict=True))
    return gap


def combine(
    items: Items, first: Solution, second: Solution, rng: random.Random
) -> tuple[list[int], tuple[int, ...] | None]:
    """Combine two solutions into a new sequence and mode settings.

    Scatter search combines reference solutions linearly and rounds the
    result back to a solution. Here the combined vector is the convex
    combination ``w p_first + (1 - w) p_second`` of the two position vectors,
    with ``w`` uniform in [0, 1), and the rounding sorts the items by it, ties
    by their position in ``first``. Every precedence relation holds in both
    parents, hence in the combination. Each task takes its mode setting from
    ``first`` with probability ``w``.
    """
    weight = rng.random()
    first_position = [0] * items.count
    second_position = [0] * items.count
    for index, item in enumerate(first.sequence):
        first_position[item] = index
    for index, item in enumerate(second.sequence):
        second_position[item] = index
    sequence = sorted(
        range(items.count),
        key=lambda item: (
            weight * first_position[item] + (1.0 - weight) * second_position[item],
            first_position[item],
        ),
    )
    if first.modes is None or second.modes is None:
        return sequence, None
    modes = tuple(
        a if rng.random() < weight else b for a, b in zip(first.modes, second.modes, strict=True)
    )
    return sequence, modes


@dataclass(frozen=True)
class ScatterSearch:
    """Scatter search (Glover 1998) in the basic design of Martí, Laguna and Glover (2006).

    Solutions are the item sequences and mode settings of
    :class:`~dispatchatlas.solvers.trajectory.space.SequenceSpace`. The search
    combines five methods:

    1. **diversification**: new sequences are built position by position, each
       next item drawn among the eligible ones with probability proportional
       to ``1 / (1 + f)``, where ``f`` counts how often the item has held that
       position in the sequences built so far;
    2. **improvement**: the first-improvement climb of
       :meth:`~dispatchatlas.solvers.trajectory.space.SequenceSpace.climb`, which
       stops after ``max_failures`` consecutive random neighbours fail to
       improve;
    3. **reference set**: ``reference_size`` solutions, the ``quality_size``
       best of the improved population and then, one at a time, the solution
       whose smallest :func:`distance` to those already chosen is largest;
    4. **subset generation**: every pair of reference solutions of which at
       least one entered the set since the last round;
    5. **combination**: :func:`combine`, whose result is improved in turn.

    Each round combines the new pairs, then keeps the ``reference_size`` best
    distinct solutions among the reference set and the improved combinations.
    When a round adds nothing, the set is rebuilt: its ``quality_size`` best
    stay, and a newly diversified and improved population supplies the rest by
    the distance rule. One round is one iteration.

    Attributes:
        population_size: Solutions diversified and improved to build or
            rebuild the reference set.
        reference_size: Solutions in the reference set, at least 2.
        quality_size: Reference solutions chosen by quality, the rest by
            diversity; at least 1 and at most ``reference_size``.
        max_failures: Consecutive non-improving neighbours that end an improvement.
    """

    population_size: int = 20
    reference_size: int = 8
    quality_size: int = 4
    max_failures: int = 50

    info: ClassVar[SolverInfo] = SolverInfo(
        id="scatter-search",
        name="Scatter search",
        family="population",
        summary="Keeps a reference set of good and mutually distant task sequences, combines "
        "pairs of them by averaging item positions, improves each combination by local "
        "search, and rebuilds the set from new diverse solutions when it stops changing.",
        references=("glover1998template", "marti2006principles"),
    )

    def __post_init__(self) -> None:
        if self.reference_size < 2:
            raise ValueError("reference_size must be at least 2")
        if not 1 <= self.quality_size <= self.reference_size:
            raise ValueError("quality_size must lie in [1, reference_size]")
        if self.population_size < self.reference_size:
            raise ValueError("population_size must be at least reference_size")
        if self.max_failures < 1:
            raise ValueError("max_failures must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Combine and improve reference solutions until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space = SequenceSpace(search)
            if space.movable:
                self._search(space)
            else:
                space.random_start()
        return search.result()

    def _search(self, space: SequenceSpace) -> None:
        rng = space.search.rng
        count = space.items.count
        frequency = [[0] * count for _ in range(count)]
        reference = self.reference_set(self._diversify(space, frequency), [])
        fresh = {solution.key for solution in reference}
        while True:
            space.search.next_iteration()
            known = {solution.key for solution in reference}
            trials: list[Solution] = []
            for first, second in combinations(reference, 2):
                if first.key not in fresh and second.key not in fresh:
                    continue
                sequence, modes = combine(space.items, first, second, rng)
                if (tuple(sequence), modes) in known:
                    continue
                trials.append(space.climb(space.evaluate(sequence, modes), self.max_failures))
            updated = self._best_distinct([*reference, *trials])
            fresh = {solution.key for solution in updated} - known
            if fresh:
                reference = updated
            else:
                kept = reference[: self.quality_size]
                reference = self.reference_set(self._diversify(space, frequency), kept)
                fresh = {solution.key for solution in reference} - known

    def reference_set(
        self, candidates: Sequence[Solution], kept: Sequence[Solution]
    ) -> list[Solution]:
        """Complete ``kept`` into a reference set, first by quality, then by diversity.

        Returns the set sorted from best to worst. It is smaller than
        ``reference_size`` when there are too few distinct solutions.
        """
        chosen = list(kept)
        keys = {solution.key for solution in chosen}
        pool: list[Solution] = []
        for solution in sorted(candidates, key=lambda member: member.score):
            if solution.key not in keys:
                keys.add(solution.key)
                pool.append(solution)
        while len(chosen) < self.quality_size and pool:
            chosen.append(pool.pop(0))
        while len(chosen) < self.reference_size and pool:
            spreads = [min(distance(solution, member) for member in chosen) for solution in pool]
            chosen.append(pool.pop(spreads.index(max(spreads))))
        return sorted(chosen, key=lambda member: member.score)

    def _best_distinct(self, solutions: Sequence[Solution]) -> list[Solution]:
        best: list[Solution] = []
        keys: set[tuple[tuple[int, ...], tuple[int, ...] | None]] = set()
        for solution in sorted(solutions, key=lambda member: member.score):
            if solution.key not in keys:
                keys.add(solution.key)
                best.append(solution)
            if len(best) == self.reference_size:
                break
        return best

    def _diversify(self, space: SequenceSpace, frequency: list[list[int]]) -> list[Solution]:
        """Build, count, and improve ``population_size`` new solutions."""
        rng = space.search.rng
        solutions: list[Solution] = []
        for _ in range(self.population_size):
            sequence = construct(
                space.items, lambda item, position: 1.0 / (1.0 + frequency[item][position]), rng
            )
            for position, item in enumerate(sequence):
                frequency[item][position] += 1
            solutions.append(space.climb(space.start(sequence), self.max_failures))
        return solutions
