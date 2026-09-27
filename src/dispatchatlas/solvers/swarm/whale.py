"""Whale Optimization Algorithm (Mirjalili and Lewis 2016).

Each iteration moves every vector ``x`` by one of two rules, chosen by a fair
coin ``p``:

- ``p < 0.5``: the point ``g - A * |C * g - x|`` (Eqs. 2.1, 2.2, 2.7, 2.8) with
  ``A = 2 a r1 - a`` and ``C = 2 r2``. The guide ``g`` is the best vector found
  so far when ``|A| < 1`` and a uniformly drawn member otherwise, so large
  coefficients explore around random members and small ones contract around
  the best vector;
- ``p >= 0.5``: the logarithmic spiral ``|x* - x| e^(b l) cos(2 pi l) + x*``
  around the best vector ``x*``, with ``l`` uniform in ``[-1, 1]`` and shape
  ``b`` (Eq. 2.5).

``a`` falls linearly from 2 to 0 over the budget. ``A`` and ``C`` are scalars
drawn once per member and iteration, and the same ``A`` both selects the guide
and scales the move, as in the authors' implementation (the paper writes them
as vectors but tests ``|A| < 1`` as a single decision). That implementation
also draws a random member per component, and draws ``l`` from ``[a2, 1]``
with ``a2`` falling from -1 to -2 over the run; this port follows the paper on
both points. The new vectors replace the old ones whatever their scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import Swarm, SwarmMethod, check_positive, method_info

__all__ = ["WhaleOptimizationAlgorithm"]


@dataclass(frozen=True)
class WhaleOptimizationAlgorithm(SwarmMethod):
    """Whale Optimization Algorithm on priority vectors.

    One evaluation per member and iteration.

    Attributes:
        spiral_shape: The constant ``b`` of the logarithmic spiral (Eq. 2.5).
    """

    spiral_shape: float = 1.0

    info: ClassVar[SolverInfo] = method_info(
        "whale-optimization-algorithm",
        "Whale Optimization Algorithm",
        "Moves each vector by a shrinking random multiple of its distance to the best "
        "vector or to a random member, or along a logarithmic spiral around the best "
        "vector, choosing between the two by a fair coin.",
        "mirjalili2016whale",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        check_positive("spiral_shape", self.spiral_shape)

    def step(self, swarm: Swarm) -> None:
        """Move every vector relative to the best vector or a random member, or along a spiral."""
        rng = swarm.rng
        a = 2.0 * (1.0 - swarm.progress)
        best = swarm.best
        for index in range(swarm.size):
            position = swarm.positions[index]
            r1, r2, coin = rng.random(3)
            spread = a * (2.0 * r1 - 1.0)
            stretch = 2.0 * r2
            if coin < 0.5:
                guide = best if abs(spread) < 1.0 else swarm.positions[rng.integers(swarm.size)]
                target = guide - spread * np.abs(stretch * guide - position)
            else:
                turn = rng.uniform(-1.0, 1.0)
                radius = math.exp(self.spiral_shape * turn) * math.cos(2.0 * math.pi * turn)
                target = np.abs(best - position) * radius + best
            swarm.replace(index, target)
