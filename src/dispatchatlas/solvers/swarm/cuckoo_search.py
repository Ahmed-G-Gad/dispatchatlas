"""Cuckoo Search (Yang and Deb 2009).

Each iteration has two phases:

1. Every member ``x`` proposes ``x + alpha * l`` (Eq. 1), where ``l`` is a
   vector of Lévy steps of index ``beta = 1.5`` drawn by Mantegna's algorithm,
   and keeps it unless it is worse.
2. The worst fraction ``pa`` of the members, at least one, is replaced by
   uniformly random vectors whatever their scores, the simplified form of its
   replacement rule that the paper states. ``pa = 0.25`` as in the paper.

The paper sets ``alpha = 1`` and compares each proposal with a randomly chosen
member. This port follows the authors' implementation on both points instead:
``alpha = 0.01 (x - b)``, with ``b`` the best vector found so far, is a
hundredth of the offset from ``b`` and leaves the best member in place, and
each proposal competes with the member it came from. That implementation also
multiplies every step by a standard normal draw, which this port leaves out,
and replaces the second phase by a random walk along differences of members,
where this port follows the paper. An iteration costs one evaluation per member
plus one per replaced member.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import (
    Swarm,
    SwarmMethod,
    check_fraction,
    check_positive,
    levy,
    method_info,
)

__all__ = ["CuckooSearch"]

LEVY_BETA = 1.5
"""Index ``beta`` of the Lévy steps."""


@dataclass(frozen=True)
class CuckooSearch(SwarmMethod):
    """Cuckoo Search on priority vectors.

    Attributes:
        step_size: Factor of the offset from the best vector in the step size.
        replaced_fraction: The fraction ``pa`` of worst members replaced every
            iteration.
    """

    step_size: float = 0.01
    replaced_fraction: float = 0.25

    info: ClassVar[SolverInfo] = method_info(
        "cuckoo-search",
        "Cuckoo Search",
        "Perturbs each vector with Lévy steps scaled by its offset from the best vector, "
        "keeps every move that is not worse, and replaces a fixed fraction of the worst "
        "vectors with random ones.",
        "yang2009cuckoo",
        "mantegna1994fast",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        check_positive("step_size", self.step_size)
        check_fraction("replaced_fraction", self.replaced_fraction)

    def step(self, swarm: Swarm) -> None:
        """Take a Lévy step from every member, then replace the worst members."""
        rng = swarm.rng
        best = swarm.best
        for index in range(swarm.size):
            position = swarm.positions[index]
            flight = levy(rng, swarm.dimension, LEVY_BETA)
            swarm.offer(index, position + self.step_size * (position - best) * flight, ties=True)
        count = max(1, round(self.replaced_fraction * swarm.size))
        for index in swarm.ranking()[swarm.size - count :]:
            swarm.replace(index, rng.random(swarm.dimension))
