"""NSGA-II: non-dominated sorting with crowding distance."""

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
from dispatchatlas.solvers.multiobjective.selection import crowding_survival

__all__ = ["Nsga2"]


class _CrowdingSelection(GenerationalSelection):
    """Crowded binary tournaments and survival by fronts and crowding distance."""

    def prepare(self, population: list[Member]) -> list[Member]:
        return self.survive([], population)

    def parents(self, population: Sequence[Member], index: int) -> tuple[Member, Member]:
        del index
        return tournament(population, self.rng), tournament(population, self.rng)

    def survive(self, population: list[Member], offspring: list[Member]) -> list[Member]:
        return feasible_first([*population, *offspring], self.size, _crowding)


def _crowding(members: list[Member], size: int) -> list[Member]:
    return keep(members, crowding_survival([member.vector for member in members], size))


@dataclass(frozen=True)
class Nsga2(MultiObjectiveEvolution):
    """The non-dominated sorting genetic algorithm II (Deb, Pratap, Agarwal and Meyarivan 2002).

    Each generation breeds ``population_size`` children from parents chosen
    by binary tournaments under the crowded comparison: the lower
    non-dominated rank wins, and on equal ranks the larger crowding distance.
    Parents and children are then sorted into non-dominated fronts, whole
    fronts survive in rank order, and the front that does not fit keeps its
    members of largest crowding distance, the boundary points of every
    objective first. Deadline overruns are handled by the constrained
    domination of the same paper (see :mod:`dispatchatlas.solvers.multiobjective`).

    With one objective the fronts are groups of equal values, and the method
    is a (mu + mu) evolutionary algorithm that keeps the best members.
    """

    info: ClassVar[SolverInfo] = SolverInfo(
        id="nsga-ii",
        name="NSGA-II",
        family="multi-objective",
        summary="Breeds task sequences and modes by crowded binary tournaments, order "
        "crossover, and insert, swap, and mode mutation, then keeps the parents and children "
        "of lowest non-dominated rank and, in the last front, of largest crowding distance.",
        references=("deb2002fast", "davis1985applying"),
        multi_objective=True,
    )

    def selection(self, evolution: Evolution) -> GenerationalSelection:
        """Return the crowding-distance selection for one run."""
        return _CrowdingSelection(evolution, self.population_size)
