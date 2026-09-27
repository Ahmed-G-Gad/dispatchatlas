"""Firefly Algorithm (Yang 2009).

Each iteration moves every vector ``x_i`` towards every member ``x_j`` that
scored better at the start of the iteration, in member order:

    x_i <- x_i + beta0 e^(-gamma r^2) (x_j - x_i) + alpha_t (u - 1/2)

where ``r`` is the current distance between the two vectors, ``u`` a fresh
uniform vector, ``beta0 = 1`` the attraction at zero distance and ``gamma`` how
fast it decays with distance. A member that no other member beats takes the
random term alone, the paper's rule for the best one. ``gamma = 1`` and
``alpha_0 = 0.25`` are the defaults of Yang's reference implementation, and the
noise weight falls geometrically, ``alpha_t = alpha_0 theta^t`` at iteration
``t`` counted from 0 with ``theta = 0.97``, within the range Yang (2014) gives
for this reduction. Distances and noise are in units of the unit hypercube, the
scale of the search space as the method prescribes.

Every vector is evaluated once per iteration, after all its moves, as in
Yang's reference implementation: comparisons use the scores and positions of
the previous iteration, and a moved vector replaces the old one whatever its
score. The cost per iteration is therefore one evaluation per member plus a
number of vector operations quadratic in the population size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import (
    Swarm,
    SwarmMethod,
    check_fraction,
    check_nonnegative,
    method_info,
)

__all__ = ["FireflyAlgorithm"]


@dataclass(frozen=True)
class FireflyAlgorithm(SwarmMethod):
    """Firefly Algorithm on priority vectors.

    Attributes:
        attraction: Attraction ``beta0`` between two vectors at zero distance.
        attraction_decay: The coefficient ``gamma`` of the squared distance in
            the attraction's exponential decay.
        noise: Initial weight ``alpha_0`` of the uniform random term.
        noise_decay: Factor ``theta`` applied to the noise weight every iteration.
    """

    attraction: float = 1.0
    attraction_decay: float = 1.0
    noise: float = 0.25
    noise_decay: float = 0.97

    info: ClassVar[SolverInfo] = method_info(
        "firefly-algorithm",
        "Firefly Algorithm",
        "Pulls each vector towards every better vector of the previous iteration with "
        "a strength that decays exponentially with the squared distance, adding "
        "uniform noise that shrinks geometrically.",
        "yang2009firefly",
        "yang2014firefly",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        check_nonnegative("attraction", self.attraction)
        check_nonnegative("attraction_decay", self.attraction_decay)
        check_nonnegative("noise", self.noise)
        check_fraction("noise_decay", self.noise_decay)

    def step(self, swarm: Swarm) -> None:
        """Move every vector towards the better vectors of the previous iteration."""
        rng = swarm.rng
        noise = self.noise * self.noise_decay ** (swarm.iteration - 1)
        previous = swarm.positions.copy()
        scores = list(swarm.scores)
        for index in range(swarm.size):
            moved = previous[index].copy()
            better = [other for other in range(swarm.size) if scores[other] < scores[index]]
            if not better:
                moved += noise * (rng.random(swarm.dimension) - 0.5)
            for other in better:
                gap = previous[other] - moved
                weight = self.attraction * math.exp(-self.attraction_decay * float(gap @ gap))
                moved += weight * gap + noise * (rng.random(swarm.dimension) - 0.5)
            swarm.replace(index, moved)
