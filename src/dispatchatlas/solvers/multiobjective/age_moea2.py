"""AGE-MOEA-II: survival by the estimated geometry of the first front."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.multiobjective._loop import (
    Evolution,
    GenerationalSelection,
    Member,
    MultiObjectiveEvolution,
    feasible_first,
    keep,
    tournament,
)
from dispatchatlas.solvers.multiobjective.selection import age_survival

__all__ = ["AgeMoea2"]


class _GeometrySelection(GenerationalSelection):
    """Tournaments on rank and survival score, and survival by the estimated front geometry."""

    def prepare(self, population: list[Member]) -> list[Member]:
        return self.survive([], population)

    def parents(self, population: Sequence[Member], index: int) -> tuple[Member, Member]:
        del index
        return tournament(population, self.rng), tournament(population, self.rng)

    def survive(self, population: list[Member], offspring: list[Member]) -> list[Member]:
        return feasible_first([*population, *offspring], self.size, _geometry)


def _geometry(members: list[Member], size: int) -> list[Member]:
    return keep(members, age_survival([member.vector for member in members], size))


@dataclass(frozen=True)
class AgeMoea2(MultiObjectiveEvolution):
    """The adaptive geometry estimation MOEA of Panichella (2022), AGE-MOEA-II.

    Each generation breeds ``population_size`` children from parents chosen by
    binary tournaments on non-dominated rank, then on survival score. Parents
    and children are sorted into non-dominated fronts and whole fronts survive
    in rank order. The first front is normalised by its ideal point and the
    intercepts of its extreme points, and Newton's method fits the exponent
    ``p`` of the surface ``sum f_i ** p = 1`` through its central point, which
    tells a linear front from a convex or a concave one. Its extreme points
    score infinity, and the others are scored greedily by the geodesic
    distance, on the fitted surface, to their two nearest already scored
    points (Panichella 2019; 2022). Later fronts score their proximity to the
    ideal point in the ``p``-norm. The front that does not fit keeps its best
    scores. Deadline overruns are handled by constrained domination.

    With one objective the first front holds the members of best value and
    later fronts are ordered by value, so the method keeps the best members.
    """

    info: ClassVar[SolverInfo] = SolverInfo(
        id="age-moea-ii",
        name="AGE-MOEA-II",
        family="multi-objective",
        summary="Breeds task sequences and modes by binary tournaments on rank and survival "
        "score, fits the curvature of the first non-dominated front by Newton's method, and "
        "keeps members by their geodesic spread on the fitted surface and, in later fronts, by "
        "proximity to the ideal point.",
        references=(
            "panichella2022improved",
            "panichella2019adaptive",
            "deb2002fast",
            "davis1985applying",
        ),
        multi_objective=True,
    )

    def selection(self, evolution: Evolution) -> GenerationalSelection:
        """Return the geometry-based selection for one run."""
        return _GeometrySelection(evolution, self.population_size)
