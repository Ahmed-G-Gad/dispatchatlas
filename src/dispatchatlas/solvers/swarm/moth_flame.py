"""Moth-Flame Optimization (Mirjalili 2015).

The method keeps a sorted memory of the best vectors ever evaluated, as many as
the population (the paper merges the moved population with the memory and
keeps the best, which amounts to the same set). Each iteration uses the first
``F = round(N - t (N - 1) / T)`` entries of that memory, a count falling
linearly from the population size ``N`` to 1 (Eq. 3.14). Member ``i`` moves
along a logarithmic spiral around entry ``m``, the one at position
``min(i, F - 1)`` (Eqs. 3.12 and 3.13),

    x <- |m - x| e^(b s) cos(2 pi s) + m

with shape ``b`` and, per component, ``s = (r - 1) u + 1`` for a fresh uniform
``u``, so ``s`` lies in ``[r, 1]`` where ``r`` falls linearly from -1 to -2
over the budget. As ``F`` shrinks, more members share the best entries. The
moved vectors replace the old ones whatever their scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import Swarm, SwarmMethod, check_positive, method_info

__all__ = ["MothFlameOptimization"]


@dataclass(frozen=True)
class MothFlameOptimization(SwarmMethod):
    """Moth-Flame Optimization on priority vectors.

    One evaluation per member and iteration.

    Attributes:
        spiral_shape: The constant ``b`` of the logarithmic spiral (Eq. 3.12).
    """

    spiral_shape: float = 1.0

    info: ClassVar[SolverInfo] = method_info(
        "moth-flame-optimization",
        "Moth-Flame Optimization",
        "Moves each vector along a logarithmic spiral around one of the best vectors "
        "found so far, sharing a number of those vectors that shrinks to one over the "
        "budget.",
        "mirjalili2015moth",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        check_positive("spiral_shape", self.spiral_shape)

    def step(self, swarm: Swarm) -> None:
        """Spiral every vector around its assigned entry of the memory of best vectors."""
        rng = swarm.rng
        progress = swarm.progress
        count = max(1, round(swarm.size - progress * (swarm.size - 1)))
        low = -1.0 - progress
        memory = swarm.elite(count)
        for index in range(swarm.size):
            centre = memory[min(index, len(memory) - 1)]
            turn = (low - 1.0) * rng.random(swarm.dimension) + 1.0
            radius = np.exp(self.spiral_shape * turn) * np.cos(2.0 * math.pi * turn)
            swarm.replace(index, np.abs(centre - swarm.positions[index]) * radius + centre)
