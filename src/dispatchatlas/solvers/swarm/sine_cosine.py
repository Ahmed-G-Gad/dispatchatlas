"""Sine Cosine Algorithm (Mirjalili 2016).

Each iteration moves every component of every vector by

    x <- x + r1 * w(r2) * |r3 * p - x|

where ``p`` is the best vector found so far, ``w`` is the sine or, with equal
probability, the cosine (Eq. 3.3), and fresh uniform draws per component give
``r2`` in ``[0, 2 pi)`` and ``r3`` in ``[0, 2)``. The amplitude ``r1 = a (1 - t/T)``
falls linearly from ``a`` to 0 over the budget (Eq. 3.4), so steps longer than
the distance to ``p`` are common early and the vectors settle late. The moved
vectors replace the old ones whatever their scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import Swarm, SwarmMethod, check_positive, method_info

__all__ = ["SineCosineAlgorithm"]


@dataclass(frozen=True)
class SineCosineAlgorithm(SwarmMethod):
    """Sine Cosine Algorithm on priority vectors.

    One evaluation per member and iteration.

    Attributes:
        amplitude: The constant ``a``, the amplitude at the start (Eq. 3.4).
    """

    amplitude: float = 2.0

    info: ClassVar[SolverInfo] = method_info(
        "sine-cosine-algorithm",
        "Sine Cosine Algorithm",
        "Moves each vector component by a sine- or cosine-weighted random multiple of "
        "its distance to a random rescaling of the best vector, with an amplitude that "
        "shrinks linearly.",
        "mirjalili2016sca",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        check_positive("amplitude", self.amplitude)

    def step(self, swarm: Swarm) -> None:
        """Move every vector by a sine or cosine step relative to the best vector."""
        rng = swarm.rng
        amplitude = self.amplitude * (1.0 - swarm.progress)
        best = swarm.best
        dimension = swarm.dimension
        for index in range(swarm.size):
            position = swarm.positions[index]
            angle = 2.0 * math.pi * rng.random(dimension)
            stretch = 2.0 * rng.random(dimension)
            wave = np.where(rng.random(dimension) < 0.5, np.sin(angle), np.cos(angle))
            swarm.replace(index, position + amplitude * wave * np.abs(stretch * best - position))
