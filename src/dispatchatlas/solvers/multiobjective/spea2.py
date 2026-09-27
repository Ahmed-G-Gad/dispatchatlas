"""SPEA2: strength fitness, nearest-neighbour density, and archive truncation."""

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
from dispatchatlas.solvers.multiobjective.selection import spea2_survival

__all__ = ["Spea2"]


class _StrengthSelection(GenerationalSelection):
    """Binary tournaments on SPEA2 fitness and the SPEA2 archive update."""

    def prepare(self, population: list[Member]) -> list[Member]:
        return self.survive([], population)

    def parents(self, population: Sequence[Member], index: int) -> tuple[Member, Member]:
        del index
        return tournament(population, self.rng), tournament(population, self.rng)

    def survive(self, population: list[Member], offspring: list[Member]) -> list[Member]:
        return feasible_first([*population, *offspring], self.size, _strength)


def _strength(members: list[Member], size: int) -> list[Member]:
    return keep(members, spea2_survival([member.vector for member in members], size))


@dataclass(frozen=True)
class Spea2(MultiObjectiveEvolution):
    """The improved strength Pareto evolutionary algorithm of Zitzler, Laumanns and Thiele (2001).

    The population is the archive, of ``population_size`` members, and each
    generation breeds as many children from parents chosen by binary
    tournaments on fitness. Every member of the archive and the children
    receives the SPEA2 fitness: the summed strengths of the members that
    dominate it, the strength being the number of members a member dominates,
    plus a density ``1 / (sigma + 2)`` from the distance to the k-th nearest
    member, ``k`` the square root of their number. All non-dominated members
    enter the next archive; a shortfall is filled with the dominated members of
    best fitness, and a surplus is removed one member at a time, always the one
    closest to its nearest remaining neighbour, ties by the next nearest.
    Distances are measured after scaling each objective to the range of the
    members, so that they do not depend on the objectives' units; the paper
    uses raw objective values. Deadline overruns are handled by constrained
    domination.

    With one objective the fitness orders members by value, and the method
    keeps the best members.
    """

    info: ClassVar[SolverInfo] = SolverInfo(
        id="spea2",
        name="SPEA2",
        family="multi-objective",
        summary="Breeds task sequences and modes by binary tournaments on strength fitness, the "
        "summed strengths of dominating members plus a nearest-neighbour density, and keeps an "
        "archive of the non-dominated members, truncated by distance to the nearest neighbours.",
        references=("zitzler2001spea2", "deb2002fast", "davis1985applying"),
        multi_objective=True,
    )

    def selection(self, evolution: Evolution) -> GenerationalSelection:
        """Return the strength-fitness selection for one run."""
        return _StrengthSelection(evolution, self.population_size)
