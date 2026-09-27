"""Grey Wolf Optimizer (Mirjalili, Mirjalili and Lewis 2014).

Each iteration moves every vector to the mean of three points, one drawn around
each of the three best vectors evaluated so far (Eqs. 3.5 to 3.7). The point
drawn around such a leader ``L`` is ``L - A * |C * L - x|`` for the member ``x``,
with ``A = 2 a r1 - a`` and ``C = 2 r2`` for fresh uniform vectors ``r1`` and
``r2`` (Eqs. 3.3 and 3.4), drawn anew for every leader. The coefficient ``a``
falls linearly from 2 to 0 over the budget, so the points spread beyond the
leaders early and contract around them late. The new vectors replace the old
ones whatever their scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import Swarm, SwarmMethod, method_info

__all__ = ["GreyWolfOptimizer"]

LEADERS = 3
"""Number of best vectors that guide every move (Eq. 3.7)."""


@dataclass(frozen=True)
class GreyWolfOptimizer(SwarmMethod):
    """Grey Wolf Optimizer on priority vectors.

    One evaluation per member and iteration.
    """

    info: ClassVar[SolverInfo] = method_info(
        "grey-wolf-optimizer",
        "Grey Wolf Optimizer",
        "Moves every vector to the mean of three random points drawn around the three "
        "best vectors found so far, with a spread that shrinks linearly to zero over "
        "the budget.",
        "mirjalili2014grey",
    )
    minimum_population: ClassVar[int] = LEADERS

    def step(self, swarm: Swarm) -> None:
        """Move every vector towards the three best vectors found so far."""
        rng = swarm.rng
        a = 2.0 * (1.0 - swarm.progress)
        leaders = np.stack(swarm.elite(LEADERS))
        shape = leaders.shape
        for index in range(swarm.size):
            spread = a * (2.0 * rng.random(shape) - 1.0)
            stretch = 2.0 * rng.random(shape)
            points = leaders - spread * np.abs(stretch * leaders - swarm.positions[index])
            swarm.replace(index, points.mean(axis=0))
