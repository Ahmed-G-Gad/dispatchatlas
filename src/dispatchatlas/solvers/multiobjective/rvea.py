"""RVEA: reference vectors and the angle-penalised distance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.multiobjective._loop import (
    Evolution,
    GenerationalSelection,
    Member,
    MultiObjectiveEvolution,
)
from dispatchatlas.solvers.multiobjective.selection import (
    adapt_vectors,
    das_dennis,
    ideal_point,
    nadir_point,
    rvea_survival,
    simplex_divisions,
    smallest_angles,
    unit_vectors,
)

__all__ = ["Rvea"]


class _ReferenceVectorSelection(GenerationalSelection):
    """Random mating, one survivor per reference vector, and periodic vector adaptation."""

    def __init__(self, evolution: Evolution, size: int, alpha: float, frequency: float) -> None:
        count = evolution.objectives
        initial = unit_vectors(das_dennis(simplex_divisions(count, size), count))
        super().__init__(evolution, len(initial))
        self.initial = initial
        self.vectors = list(initial)
        self.gammas = smallest_angles(self.vectors)
        self.alpha = alpha
        self.frequency = frequency
        self.next_adaptation = frequency

    def survive(self, population: list[Member], offspring: list[Member]) -> list[Member]:
        pool = [*population, *offspring]
        progress = self.evolution.progress()
        chosen = rvea_survival(
            [member.vector for member in pool],
            self.vectors,
            self.gammas,
            progress,
            self.alpha,
            [member.violation for member in pool],
        )
        kept = [pool[index] for index in chosen]
        if progress >= self.next_adaptation:
            vectors = [member.vector for member in kept]
            self.vectors = adapt_vectors(self.initial, ideal_point(vectors), nadir_point(vectors))
            self.gammas = smallest_angles(self.vectors)
            while self.next_adaptation <= progress:
                self.next_adaptation += self.frequency
        return kept


@dataclass(frozen=True)
class Rvea(MultiObjectiveEvolution):
    """The reference vector guided evolutionary algorithm, RVEA.

    The method of Cheng, Jin, Olhofer and Sendhoff (2016). The reference
    vectors are the Das and Dennis (1998) lattice with the most
    divisions that give at most ``population_size`` points, scaled to unit
    length. Each generation breeds one child per vector from pairs of parents
    drawn at random. Parents and children are translated by their ideal point
    and grouped by the vector at the smallest angle, and each group keeps the
    member of smallest angle-penalised distance
    ``(1 + M * (t / t_max) ** alpha * theta / gamma) * |f'|``, where ``theta``
    is the member's angle to its vector and ``gamma`` the vector's smallest
    angle to another vector. The fraction ``t / t_max`` is the fraction of the
    budget used, so the selection moves from convergence to spread along the
    vectors as the run proceeds. Each time a further ``adaptation_frequency``
    of the budget is used, the vectors are rescaled to the ranges of the
    objectives in the population. The population therefore holds at most one
    member per vector. Deadlines are handled by constrained domination within
    each group: a group keeps its feasible member of smallest distance or,
    without one, its member of smallest deadline overrun.

    With one objective there is one vector and the method is a (1 + 1)
    evolutionary algorithm that keeps the better of parent and child.

    Attributes:
        alpha: Rate at which the angle penalty grows with the budget used; 2 in
            the paper.
        adaptation_frequency: Fraction of the budget between vector
            adaptations; 0.1 in the paper.
    """

    alpha: float = 2.0
    adaptation_frequency: float = 0.1

    info: ClassVar[SolverInfo] = SolverInfo(
        id="rvea",
        name="RVEA",
        family="multi-objective",
        summary="Breeds task sequences and modes from random parents and keeps, for each "
        "reference vector, the member of smallest angle-penalised distance, a penalty that "
        "shifts selection from convergence to spread as the budget is used, rescaling the "
        "vectors to the objectives' ranges at fixed budget fractions.",
        references=("cheng2016reference", "das1998normalboundary", "davis1985applying"),
        multi_objective=True,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (self.alpha >= 0.0 and math.isfinite(self.alpha)):
            raise ValueError("alpha must be a non-negative number")
        if not 0.0 < self.adaptation_frequency <= 1.0:
            raise ValueError("adaptation_frequency must lie in (0, 1]")

    def selection(self, evolution: Evolution) -> GenerationalSelection:
        """Return the reference-vector selection for one run."""
        return _ReferenceVectorSelection(
            evolution, self.population_size, self.alpha, self.adaptation_frequency
        )
