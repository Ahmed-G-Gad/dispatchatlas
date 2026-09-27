"""Teaching-Learning-Based Optimization (Rao, Savsani and Vakharia 2011).

Each iteration runs two greedy phases over the population:

1. Every member ``x`` proposes ``x + r (b - F m)``, where ``b`` is the best
   member and ``m`` the mean of the population, both taken when the phase
   starts, ``r`` is a uniform scalar and ``F``, 1 or 2 with equal probability,
   sets how much of the mean is subtracted.
2. Every member ``x`` then draws another member ``y`` and proposes
   ``x + r (x - y)`` if ``x`` is better than ``y``, otherwise ``x + r (y - x)``:
   a step away from worse members and towards better ones.

A proposal replaces the member only if it is better, so each iteration costs
two evaluations per member. ``r`` and ``F`` are drawn once per proposal.
Črepinšek, Liu and Mernik (2012) report that the published description omits a
step of the authors' implementation, gives wrong formulae for the number of
evaluations, and wrongly presents the method as free of parameters of its own.
This port implements the two published phases and counts every evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import Swarm, SwarmMethod, centroid, method_info

__all__ = ["TeachingLearningBasedOptimization"]


@dataclass(frozen=True)
class TeachingLearningBasedOptimization(SwarmMethod):
    """Teaching-Learning-Based Optimization on priority vectors."""

    info: ClassVar[SolverInfo] = method_info(
        "teaching-learning-based-optimization",
        "Teaching-Learning-Based Optimization",
        "Moves each vector towards the best member and away from the population mean, "
        "then towards a better or away from a worse random member, keeping each move "
        "only if it improves the vector.",
        "rao2011teaching",
    )

    def step(self, swarm: Swarm) -> None:
        """Run the phase guided by the best member, then the pairwise phase."""
        rng = swarm.rng
        best = swarm.positions[swarm.leader()].copy()
        mean = centroid(swarm.positions)
        for index in range(swarm.size):
            factor = 1 + int(rng.integers(2))
            swarm.offer(index, swarm.positions[index] + rng.random() * (best - factor * mean))
        for index in range(swarm.size):
            other = int(rng.integers(swarm.size - 1))
            other += other >= index
            position, partner = swarm.positions[index], swarm.positions[other]
            if swarm.scores[index] < swarm.scores[other]:
                direction = position - partner
            else:
                direction = partner - position
            swarm.offer(index, position + rng.random() * direction)
