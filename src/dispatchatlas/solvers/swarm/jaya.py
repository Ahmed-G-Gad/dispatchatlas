"""Jaya (Rao 2016).

Each iteration moves every member ``x`` towards the best member ``b`` and away
from the worst member ``w`` of the population, both taken when the iteration
starts:

    x' = x + r1 * (b - |x|) - r2 * (w - |x|)

with fresh uniform vectors ``r1`` and ``r2`` (Eq. 1). ``x'`` replaces ``x`` only
if it is better, so each iteration costs one evaluation per member. The
absolute values are the paper's; in the unit hypercube they change nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import Swarm, SwarmMethod, method_info

__all__ = ["Jaya"]


@dataclass(frozen=True)
class Jaya(SwarmMethod):
    """Jaya on priority vectors."""

    info: ClassVar[SolverInfo] = method_info(
        "jaya",
        "Jaya",
        "Moves each vector towards the best and away from the worst member of the "
        "population by random amounts per component, keeping the move only if it "
        "improves the vector.",
        "rao2016jaya",
    )

    def step(self, swarm: Swarm) -> None:
        """Move every vector towards the best and away from the worst member."""
        rng = swarm.rng
        best = swarm.positions[swarm.leader()].copy()
        worst = swarm.positions[swarm.laggard()].copy()
        for index in range(swarm.size):
            position = swarm.positions[index]
            magnitude = np.abs(position)
            r1 = rng.random(swarm.dimension)
            r2 = rng.random(swarm.dimension)
            swarm.offer(index, position + r1 * (best - magnitude) - r2 * (worst - magnitude))
