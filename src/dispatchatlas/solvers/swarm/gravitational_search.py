"""Gravitational Search Algorithm (Rashedi, Nezamabadi-pour and Saryazdi 2009).

Each iteration gives every member a weight from its quality: with ``v`` the
:func:`~dispatchatlas.solvers.swarm._engine.magnitudes` values,
``w_i = (v_worst - v_i) / (v_worst - v_best)``, normalised to sum to one, or
equal weights when all values tie. Only the ``K`` heaviest members attract,
where ``K`` falls linearly from the population size to 1 over the budget, the
elitism the paper calls Kbest. Member ``i`` accelerates by

    a_i = sum over attracting j != i of  rand_ij G w_j (x_j - x_i) / (R_ij + eps)

with ``R_ij`` the Euclidean distance, a fresh uniform scalar ``rand_ij`` per
pair, and the gain ``G = G0 e^(-alpha t/T)`` with ``G0 = 100`` and
``alpha = 20`` as in the paper. The velocity and the position then follow
``v_i <- rand_i v_i + a_i`` and ``x_i <- x_i + v_i``, with a uniform scalar
``rand_i`` per member, and the new vectors replace the old ones whatever their
scores. All accelerations use the population as it stood when the iteration
began. ``eps`` is the machine epsilon.

``G0 = 100`` is large for the unit hypercube: until the gain has decayed, steps
overshoot the bounds many times over and the folding back into the hypercube
scatters the vectors almost at random. Because ``x_j - x_i`` is not normalised,
the division by ``R_ij`` cancels its length, so the pull of a member does not
depend on its distance, as Gauci, Dodd and Groß (2012) point out; the physical
law the method is named after would make it fall with the squared distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import (
    EPSILON,
    Swarm,
    SwarmMethod,
    check_nonnegative,
    check_positive,
    distances,
    magnitudes,
    method_info,
)

__all__ = ["GravitationalSearchAlgorithm"]


@dataclass(frozen=True)
class GravitationalSearchAlgorithm(SwarmMethod):
    """Gravitational Search Algorithm on priority vectors.

    One evaluation per member and iteration.

    Attributes:
        gain: The initial gain ``G0``.
        gain_decay: The rate ``alpha`` of the gain's exponential decay.
    """

    gain: float = 100.0
    gain_decay: float = 20.0

    info: ClassVar[SolverInfo] = method_info(
        "gravitational-search-algorithm",
        "Gravitational Search Algorithm",
        "Accelerates each vector towards the members of highest quality, weighted by "
        "quality over distance, with a gain that decays exponentially, a number of "
        "attracting members that shrinks to one, and random momentum.",
        "rashedi2009gsa",
        "deb2000efficient",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        check_positive("gain", self.gain)
        check_nonnegative("gain_decay", self.gain_decay)

    def step(self, swarm: Swarm) -> None:
        """Accelerate every vector towards the heaviest members and move it."""
        rng = swarm.rng
        size = swarm.size
        progress = swarm.progress
        gain = self.gain * math.exp(-self.gain_decay * progress)
        values = magnitudes(swarm.scores)
        best, worst = min(values), max(values)
        raw = (
            [(worst - value) / (worst - best) for value in values] if worst > best else [1.0] * size
        )
        weights = np.array(raw) / sum(raw)
        count = max(1, round(size - progress * (size - 1)))
        attractors = sorted(range(size), key=lambda member: -weights[member])[:count]
        positions = swarm.positions.copy()
        spans = distances(positions)
        acceleration = np.zeros_like(positions)
        for index in range(size):
            for other in attractors:
                if other != index:
                    pull = rng.random() * gain * weights[other] / (spans[index, other] + EPSILON)
                    acceleration[index] += pull * (positions[other] - positions[index])
        swarm.velocity = rng.random((size, 1)) * swarm.velocity + acceleration
        for index in range(size):
            swarm.replace(index, positions[index] + swarm.velocity[index])
