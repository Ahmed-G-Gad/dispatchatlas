"""IBEA: fitness from the additive epsilon indicator."""

from __future__ import annotations

import math
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
from dispatchatlas.solvers.multiobjective.selection import IBEA_KAPPA, ibea_survival

__all__ = ["Ibea"]


class _IndicatorSelection(GenerationalSelection):
    """Binary tournaments on indicator fitness and removal of the least fit."""

    def __init__(self, evolution: Evolution, size: int, kappa: float) -> None:
        super().__init__(evolution, size)
        self.kappa = kappa

    def prepare(self, population: list[Member]) -> list[Member]:
        return self.survive([], population)

    def parents(self, population: Sequence[Member], index: int) -> tuple[Member, Member]:
        del index
        return tournament(population, self.rng), tournament(population, self.rng)

    def survive(self, population: list[Member], offspring: list[Member]) -> list[Member]:
        return feasible_first([*population, *offspring], self.size, self._indicator)

    def _indicator(self, members: list[Member], size: int) -> list[Member]:
        vectors = [member.vector for member in members]
        return keep(members, ibea_survival(vectors, size, self.kappa))


@dataclass(frozen=True)
class Ibea(MultiObjectiveEvolution):
    """The indicator-based evolutionary algorithm of Zitzler and Künzli (2004).

    Each generation breeds ``population_size`` children from parents chosen by
    binary tournaments on fitness. Objectives of parents and children are
    scaled to their range, and each member ``x`` receives the fitness
    ``sum over y != x of -exp(-I(y, x) / (c * kappa))``, where ``I`` is the
    additive epsilon indicator and ``c`` its largest absolute value, the
    adaptive variant of the paper. The least fit member is removed and the
    others' fitness is updated to exclude it, until ``population_size``
    members remain. Deadline overruns are handled by constrained domination.

    With one objective the indicator is the difference of values, and the
    method removes the worst members.

    Attributes:
        kappa: Scaling factor of the fitness; 0.05 in the paper.
    """

    kappa: float = IBEA_KAPPA

    info: ClassVar[SolverInfo] = SolverInfo(
        id="ibea",
        name="IBEA",
        family="multi-objective",
        summary="Breeds task sequences and modes by binary tournaments on fitness from the "
        "additive epsilon indicator between members, and removes the least fit member one at "
        "a time, updating the others' fitness, until the population fits.",
        references=("zitzler2004indicatorbased", "deb2002fast", "davis1985applying"),
        multi_objective=True,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (self.kappa > 0.0 and math.isfinite(self.kappa)):
            raise ValueError("kappa must be a positive number")

    def selection(self, evolution: Evolution) -> GenerationalSelection:
        """Return the indicator selection for one run."""
        return _IndicatorSelection(evolution, self.population_size, self.kappa)
