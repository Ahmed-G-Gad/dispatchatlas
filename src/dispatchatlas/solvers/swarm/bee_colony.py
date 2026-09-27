"""Artificial Bee Colony (Karaboga and Basturk 2007).

Each iteration has three phases:

1. Every member ``x`` proposes a copy of itself with one uniformly drawn
   component ``k`` moved to ``x_k + phi (x_k - y_k)``, where ``y`` is another
   uniformly drawn member and ``phi`` is uniform in ``[-1, 1]``, and keeps it
   only if it is better.
2. As many members as the population holds are drawn, with replacement and
   with probability ``p_i = f_i / sum_j f_j``, and each makes the same kind of
   proposal. The quality is ``f = 1 / (1 + v)`` for a number ``v >= 0`` and
   ``f = 1 + |v|`` otherwise, where ``v`` is the member's
   :func:`~dispatchatlas.solvers.swarm._engine.magnitudes` value, and the
   probabilities are fixed when the phase starts.
3. The member whose proposals failed most often in a row, the lowest index on
   ties, is replaced by a uniformly random vector if that count has reached
   ``limit``. At most one member is replaced per iteration.

An iteration therefore costs two evaluations per member, plus one when a member
is replaced. ``limit`` defaults to the number of members times the vector
length, the setting commonly recommended for the method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import Swarm, SwarmMethod, magnitudes, method_info

__all__ = ["ArtificialBeeColony"]


@dataclass(frozen=True)
class ArtificialBeeColony(SwarmMethod):
    """Artificial Bee Colony on priority vectors.

    Attributes:
        limit: Failed proposals in a row after which a member is replaced by a
            random vector. ``None`` uses the population size times the vector
            length.
    """

    limit: int | None = None

    info: ClassVar[SolverInfo] = method_info(
        "artificial-bee-colony",
        "Artificial Bee Colony",
        "Moves one random component of each vector towards or away from a random member, "
        "repeats this for members drawn in proportion to their quality, keeps only "
        "improvements, and restarts the vector that failed to improve longest.",
        "karaboga2007powerful",
        "deb2000efficient",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        limit = self.limit
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)):
            raise ValueError(f"limit must be an integer or None, got {limit!r}")
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit!r}")

    def step(self, swarm: Swarm) -> None:
        """Run the three phases: every member, members drawn by quality, restart."""
        rng = swarm.rng
        size = swarm.size
        for index in range(size):
            self._propose(swarm, index)
        quality = np.array(
            [1.0 / (1.0 + v) if v >= 0.0 else 1.0 - v for v in magnitudes(swarm.scores)]
        )
        chosen = rng.choice(size, size=size, p=quality / quality.sum())
        for index in chosen:
            self._propose(swarm, int(index))
        stalest = max(range(size), key=swarm.trials.__getitem__)
        limit = self.limit if self.limit is not None else size * swarm.dimension
        if swarm.trials[stalest] >= limit:
            swarm.replace(stalest, rng.random(swarm.dimension))

    @staticmethod
    def _propose(swarm: Swarm, index: int) -> None:
        """Move one random component of member ``index`` relative to another member."""
        rng = swarm.rng
        other = int(rng.integers(swarm.size - 1))
        other += other >= index
        component = int(rng.integers(swarm.dimension))
        target = swarm.positions[index].copy()
        target[component] += rng.uniform(-1.0, 1.0) * (
            target[component] - swarm.positions[other, component]
        )
        swarm.offer(index, target)
