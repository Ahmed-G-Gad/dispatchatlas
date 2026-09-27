"""Salp Swarm Algorithm (Mirjalili, Gandomi, Mirjalili, Saremi, Faris and Mirjalili 2017).

The population is a chain. Each iteration first resamples the head of the
chain around the best vector ``b`` found so far, component by component,

    x_0 <- b +/- c1 ((UB - LB) c2 + LB)

with ``c2`` uniform and the sign drawn by a second uniform ``c3`` (Eq. 3.1), and
``c1 = 2 e^(-(4 t/T)^2)`` falling from 2 to nearly 0 over the budget (Eq. 3.2).
Every other member then moves halfway to the member ahead of it,
``x_i <- (x_i + x_(i-1)) / 2`` (Eq. 3.4), in chain order, so each follows the
already moved one. The new vectors replace the old ones whatever their scores.

The paper draws the minus sign when ``c3 < 0``, which a uniform ``c3`` never
is; as in the authors' implementation, each sign has probability one half
here. Unlike that implementation, which treats the first half of the chain as
heads, this port follows the paper and has a single head. Castelli, Manzoni,
Mariot, Nobile and Tangherloni (2022) review both discrepancies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import Swarm, SwarmMethod, method_info

__all__ = ["SalpSwarmAlgorithm"]


@dataclass(frozen=True)
class SalpSwarmAlgorithm(SwarmMethod):
    """Salp Swarm Algorithm on priority vectors.

    One evaluation per member and iteration.
    """

    info: ClassVar[SolverInfo] = method_info(
        "salp-swarm-algorithm",
        "Salp Swarm Algorithm",
        "Resamples the first vector of a chain around the best vector with a spread "
        "that shrinks over the budget, then moves every other vector halfway to the "
        "one ahead of it.",
        "mirjalili2017salp",
    )

    def step(self, swarm: Swarm) -> None:
        """Resample the head of the chain, then pull every member to its predecessor."""
        rng = swarm.rng
        spread = 2.0 * math.exp(-((4.0 * swarm.progress) ** 2))
        sign = np.where(rng.random(swarm.dimension) < 0.5, 1.0, -1.0)
        offset = rng.random(swarm.dimension)  # LB + c2 (UB - LB) is c2 in [0, 1]
        swarm.replace(0, swarm.best + sign * spread * offset)
        for index in range(1, swarm.size):
            swarm.replace(index, (swarm.positions[index] + swarm.positions[index - 1]) / 2.0)
