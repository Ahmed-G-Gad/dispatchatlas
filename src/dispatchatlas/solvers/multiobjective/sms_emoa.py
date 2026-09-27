"""SMS-EMOA: steady-state selection by hypervolume contribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.multiobjective._loop import (
    Evolution,
    Member,
    MultiObjectiveEvolution,
    SteadyStateSelection,
)
from dispatchatlas.solvers.multiobjective.selection import hypervolume_discard

__all__ = ["SmsEmoa"]


class _HypervolumeSelection(SteadyStateSelection):
    """Random mating and removal of the smallest hypervolume contributor."""

    def admit(self, population: list[Member], child: Member, index: int) -> list[Member]:
        del index
        pool = [*population, child]
        infeasible = [k for k, member in enumerate(pool) if not member.feasible]
        if infeasible:
            discard = max(infeasible, key=lambda k: (pool[k].violation, k))
        else:
            discard = hypervolume_discard([member.vector for member in pool], self.rng)
        del pool[discard]
        return pool


@dataclass(frozen=True)
class SmsEmoa(MultiObjectiveEvolution):
    """The S-metric selection EMOA of Beume, Naujoks and Emmerich (2007).

    A steady-state (mu + 1) algorithm: each step breeds one child from two
    parents drawn at random, sorts the parents and the child into
    non-dominated fronts, and removes from the last front the member whose
    exclusive hypervolume is smallest. The contributions are exact. The best
    member of the last front in each objective is always kept, which Emmerich,
    Beume and Naujoks (2005) note makes a reference point unnecessary with
    two objectives; with more, the reference point lies beyond the front's
    worst values by a tenth of its range. When a member misses a deadline, the
    member with the largest overrun is removed instead, the last front under
    constrained domination. ``population_size`` steps make one iteration.

    With one objective the last front holds the members of worst value, and
    the method is a steady-state evolutionary algorithm that removes one of
    them per step.
    """

    info: ClassVar[SolverInfo] = SolverInfo(
        id="sms-emoa",
        name="SMS-EMOA",
        family="multi-objective",
        summary="Breeds one child at a time from random parents by order crossover and insert, "
        "swap, and mode mutation, then removes the member of the last non-dominated front with "
        "the smallest exact hypervolume contribution, keeping the front's best member in each "
        "objective.",
        references=("beume2007smsemoa", "emmerich2005emo", "deb2002fast", "davis1985applying"),
        multi_objective=True,
    )

    def selection(self, evolution: Evolution) -> SteadyStateSelection:
        """Return the hypervolume selection for one run."""
        return _HypervolumeSelection(evolution, self.population_size)
