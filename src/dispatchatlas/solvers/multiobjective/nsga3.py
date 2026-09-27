"""NSGA-III: non-dominated sorting with niching around reference points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.multiobjective._loop import (
    Evolution,
    GenerationalSelection,
    Member,
    MultiObjectiveEvolution,
    feasible_first,
)
from dispatchatlas.solvers.multiobjective.selection import (
    das_dennis,
    niching_survival,
    simplex_divisions,
)

__all__ = ["Nsga3"]


class _NichingSelection(GenerationalSelection):
    """Random mating and survival by fronts and reference-point niching."""

    def __init__(self, evolution: Evolution, size: int) -> None:
        super().__init__(evolution, size)
        count = evolution.objectives
        self.references = das_dennis(simplex_divisions(count, size), count)

    def survive(self, population: list[Member], offspring: list[Member]) -> list[Member]:
        return feasible_first([*population, *offspring], self.size, self._niche)

    def _niche(self, members: list[Member], size: int) -> list[Member]:
        vectors = [member.vector for member in members]
        return [members[i] for i in niching_survival(vectors, size, self.references, self.rng)]


@dataclass(frozen=True)
class Nsga3(MultiObjectiveEvolution):
    """The reference-point-based NSGA-III of Deb and Jain (2014).

    The reference points are the Das and Dennis (1998) lattice of the unit
    simplex with the most divisions that give at most ``population_size``
    points. Each generation breeds ``population_size`` children from pairs of
    parents drawn at random, as the paper prescribes. Parents and children are
    sorted into non-dominated fronts, whole fronts survive in rank order, and
    the front that does not fit is filled by niching: objectives are
    normalised by the ideal point and the intercepts of the hyperplane through
    the extreme points, every member joins its nearest reference line, and the
    least represented reference points receive members first. Deadline
    overruns are handled by constrained domination.

    With one objective the lattice is a single reference point and the method
    keeps the best members, breaking ties among equal values at random.
    """

    info: ClassVar[SolverInfo] = SolverInfo(
        id="nsga-iii",
        name="NSGA-III",
        family="multi-objective",
        summary="Breeds task sequences and modes from random parents by order crossover and "
        "insert, swap, and mode mutation, then keeps whole non-dominated fronts and fills the "
        "last one by niching normalised objective vectors around Das and Dennis reference "
        "points.",
        references=(
            "deb2014evolutionary",
            "das1998normalboundary",
            "deb2002fast",
            "davis1985applying",
        ),
        multi_objective=True,
    )

    def selection(self, evolution: Evolution) -> GenerationalSelection:
        """Return the niching selection for one run."""
        return _NichingSelection(evolution, self.population_size)
