"""MOEA/D: decomposition into Tchebycheff subproblems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.multiobjective._loop import (
    Evolution,
    Member,
    MultiObjectiveEvolution,
    SteadyStateSelection,
)
from dispatchatlas.solvers.multiobjective.selection import (
    das_dennis,
    ideal_point,
    nadir_point,
    simplex_divisions,
    tchebycheff,
    weight_neighbourhoods,
)

__all__ = ["MoeaD"]


class _DecompositionSelection(SteadyStateSelection):
    """One incumbent per weight vector, mating and replacement within neighbourhoods."""

    def __init__(self, evolution: Evolution, size: int, neighbourhood: int) -> None:
        count = evolution.objectives
        weights = das_dennis(simplex_divisions(count, size), count)
        super().__init__(evolution, len(weights))
        self.weights = weights
        self.neighbours = weight_neighbourhoods(weights, min(neighbourhood, len(weights)))
        self.ideal: list[float] = []

    def prepare(self, population: list[Member]) -> list[Member]:
        self.ideal = list(ideal_point([member.vector for member in population]))
        return population

    def parents(self, population: Sequence[Member], index: int) -> tuple[Member, Member]:
        near = self.neighbours[index]
        if len(near) < 2:
            return population[near[0]], population[near[0]]
        first, second = self.rng.sample(near, 2)
        return population[first], population[second]

    def admit(self, population: list[Member], child: Member, index: int) -> list[Member]:
        self.ideal = [min(low, value) for low, value in zip(self.ideal, child.vector, strict=True)]
        nadir = nadir_point([*(member.vector for member in population), child.vector])
        spans = [
            high - low if high > low else 1.0 for low, high in zip(self.ideal, nadir, strict=True)
        ]
        for neighbour in self.neighbours[index]:
            if self._key(child, neighbour, spans) <= self._key(
                population[neighbour], neighbour, spans
            ):
                population[neighbour] = child
        return population

    def _key(self, member: Member, subproblem: int, spans: list[float]) -> tuple[float, float]:
        value = tchebycheff(member.vector, self.weights[subproblem], self.ideal, spans)
        return member.violation, value


@dataclass(frozen=True)
class MoeaD(MultiObjectiveEvolution):
    """The multi-objective evolutionary algorithm based on decomposition (Zhang and Li 2007).

    The weight vectors are the Das and Dennis (1998) lattice with the most
    divisions that give at most ``population_size`` points, and each defines
    a Tchebycheff subproblem with one incumbent schedule. Objectives are
    normalised by the ideal point and the largest value in the population, the
    normalisation the paper applies to disparately scaled objectives, and
    zero weights count as :data:`~dispatchatlas.solvers.multiobjective.selection.WEIGHT_FLOOR`.
    In every generation each subproblem in turn breeds one child from two
    distinct incumbents of its neighbourhood, the ``neighbourhood_size``
    nearest weight vectors; the child updates the ideal point and replaces
    every neighbourhood incumbent whose scalarised value is no better than its
    own. With deadlines, a child replaces an incumbent when it overruns them by
    less, or by as much with a no larger scalarised value. The run's archive
    of non-dominated schedules plays the role of the paper's external
    population.

    With one objective the lattice holds a single weight vector, and the
    method is a (1 + 1) evolutionary algorithm in which a child replaces its
    parent when it is no worse.

    Attributes:
        neighbourhood_size: Weight vectors in each neighbourhood, the vector
            itself included; 20 in the paper's experiments.
    """

    neighbourhood_size: int = 20

    info: ClassVar[SolverInfo] = SolverInfo(
        id="moea-d",
        name="MOEA/D",
        family="multi-objective",
        summary="Decomposes the objectives into Tchebycheff subproblems along Das and Dennis "
        "weight vectors, each holding one schedule, and lets a child bred from a subproblem's "
        "neighbourhood replace every neighbour that it scalarises no worse.",
        references=("zhang2007moead", "das1998normalboundary", "deb2002fast", "davis1985applying"),
        multi_objective=True,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.neighbourhood_size, bool) or self.neighbourhood_size < 1:
            raise ValueError("neighbourhood_size must be at least 1")

    def selection(self, evolution: Evolution) -> SteadyStateSelection:
        """Return the decomposition selection for one run."""
        return _DecompositionSelection(evolution, self.population_size, self.neighbourhood_size)
