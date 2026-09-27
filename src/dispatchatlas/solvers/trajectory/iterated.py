"""Perturbation and local search: iterated local search and variable neighbourhood search.

Both methods alternate a random perturbation of a local optimum with the local
search :meth:`~dispatchatlas.solvers.trajectory.space.SequenceSpace.climb`, a
first-improvement descent over random insert, swap, and mode neighbours that ends
after ``max_failures`` consecutive neighbours fail to improve. A perturbation of
strength ``k`` applies ``k`` random moves in a row. One iteration is one
perturbation followed by its local search. Both start from the local optimum of
a random precedence-feasible sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory.space import SequenceSpace

__all__ = ["IteratedLocalSearch", "VariableNeighbourhoodSearch"]


@dataclass(frozen=True)
class IteratedLocalSearch:
    """Iterated local search (Lourenço, Martin and Stützle 2003).

    Each iteration perturbs the current local optimum with ``strength`` random
    moves, climbs to a new local optimum, and keeps it when it is no worse than
    the current one: the *better* acceptance criterion, with ties accepted so
    that the walk can cross plateaus of equal cost.

    Attributes:
        strength: Random moves per perturbation.
        max_failures: Consecutive non-improving neighbours that end a local search.
    """

    strength: int = 3
    max_failures: int = 100

    info: ClassVar[SolverInfo] = SolverInfo(
        id="iterated-local-search",
        name="Iterated local search",
        family="local-search",
        summary="Perturbs the current local optimum with a few random moves, climbs to a "
        "new local optimum by first improvement, and keeps it when it is no worse.",
        references=("lourenco2003iterated",),
    )

    def __post_init__(self) -> None:
        if self.strength < 1:
            raise ValueError("strength must be at least 1")
        if self.max_failures < 1:
            raise ValueError("max_failures must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Perturb and climb until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space = SequenceSpace(search)
            current = space.climb(space.random_start(), self.max_failures)
            while space.movable:
                search.next_iteration()
                candidate = space.climb(space.perturb(current, self.strength), self.max_failures)
                if candidate.score <= current.score:
                    current = candidate
        return search.result()


@dataclass(frozen=True)
class VariableNeighbourhoodSearch:
    """Basic variable neighbourhood search (Mladenović and Hansen 1997).

    Neighbourhood ``N_k`` holds the solutions ``k`` random moves away. Each
    iteration shakes the current local optimum to a random point of ``N_k``,
    climbs from it, and moves there when the result is strictly better,
    returning to ``k = 1``; otherwise ``k`` grows by one, wrapping to 1 after
    ``max_neighbourhood``.

    Attributes:
        max_neighbourhood: The largest neighbourhood index ``k_max``.
        max_failures: Consecutive non-improving neighbours that end a local search.
    """

    max_neighbourhood: int = 5
    max_failures: int = 100

    info: ClassVar[SolverInfo] = SolverInfo(
        id="variable-neighbourhood-search",
        name="Variable neighbourhood search",
        family="local-search",
        summary="Shakes the current local optimum by k random moves, climbs by first "
        "improvement, and moves on improvement with k reset to 1, otherwise tries the "
        "next larger k.",
        references=("mladenovic1997variable",),
    )

    def __post_init__(self) -> None:
        if self.max_neighbourhood < 1:
            raise ValueError("max_neighbourhood must be at least 1")
        if self.max_failures < 1:
            raise ValueError("max_failures must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Shake, climb, and change neighbourhoods until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space = SequenceSpace(search)
            current = space.climb(space.random_start(), self.max_failures)
            neighbourhood = 1
            while space.movable:
                search.next_iteration()
                candidate = space.climb(space.perturb(current, neighbourhood), self.max_failures)
                if candidate.score < current.score:
                    current, neighbourhood = candidate, 1
                else:
                    neighbourhood = neighbourhood % self.max_neighbourhood + 1
        return search.result()
