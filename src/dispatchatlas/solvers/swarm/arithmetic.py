"""Arithmetic Optimization Algorithm (Abualigah, Diabat, Mirjalili, Abd Elaziz and Gandomi 2021).

Each iteration resamples every component of every vector around the matching
component ``b`` of the best vector found so far; the member's own position
plays no part. With ``w = (UB - LB) mu + LB``, which is ``mu`` in the unit
hypercube, a component becomes

- when a uniform draw exceeds ``MOA`` (Eq. 3): ``b / (MOP + eps) * w`` or
  ``b * MOP * w``, with equal probability;
- otherwise (Eq. 5): ``b - MOP * w`` or ``b + MOP * w``, with equal probability.

``MOA = MOA_min + (t/T) (MOA_max - MOA_min)`` rises linearly (Eq. 2), so the
wide multiplicative moves give way to additive ones, and
``MOP = 1 - (t/T)^(1/alpha)`` falls from 1 to 0 (Eq. 4), so the additive steps
shrink. The constants are the paper's: ``alpha = 5`` and ``mu = 0.5`` (its
Table 1) and ``MOA`` rising from 0.2 to 0.9 (its Section 2.5); ``eps`` is the
machine epsilon. The new vectors replace the old ones whatever their scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import (
    EPSILON,
    Swarm,
    SwarmMethod,
    check_fraction,
    check_positive,
    method_info,
)

__all__ = ["ArithmeticOptimizationAlgorithm"]


@dataclass(frozen=True)
class ArithmeticOptimizationAlgorithm(SwarmMethod):
    """Arithmetic Optimization Algorithm on priority vectors.

    One evaluation per member and iteration.

    Attributes:
        moa_min: Value of ``MOA`` at the start (Eq. 2).
        moa_max: Value of ``MOA`` at the end of the budget (Eq. 2).
        alpha: Exponent ``alpha`` of the ``MOP`` schedule (Eq. 4).
        mu: Control parameter ``mu`` of the step width (Eqs. 3 and 5).
    """

    moa_min: float = 0.2
    moa_max: float = 0.9
    alpha: float = 5.0
    mu: float = 0.5

    info: ClassVar[SolverInfo] = method_info(
        "arithmetic-optimization-algorithm",
        "Arithmetic Optimization Algorithm",
        "Resamples each vector component around the best vector by multiplying or "
        "dividing early and by adding or subtracting late, with a step that shrinks "
        "over the budget.",
        "abualigah2021arithmetic",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        check_fraction("moa_min", self.moa_min, allow_zero=True)
        check_fraction("moa_max", self.moa_max, allow_zero=True)
        if self.moa_min > self.moa_max:
            raise ValueError("moa_min must not exceed moa_max")
        check_positive("alpha", self.alpha)
        check_positive("mu", self.mu)

    def step(self, swarm: Swarm) -> None:
        """Resample every vector around the best vector."""
        rng = swarm.rng
        progress = swarm.progress
        accelerator = self.moa_min + progress * (self.moa_max - self.moa_min)
        probability = 1.0 - progress ** (1.0 / self.alpha)
        width = self.mu  # (UB - LB) mu + LB with LB = 0 and UB = 1
        best = swarm.best
        spread = best / (probability + EPSILON) * width, best * probability * width
        shift = best - probability * width, best + probability * width
        shape = (swarm.dimension,)
        for index in range(swarm.size):
            explore = rng.random(shape) > accelerator
            wide = np.where(rng.random(shape) < 0.5, *spread)
            narrow = np.where(rng.random(shape) < 0.5, *shift)
            swarm.replace(index, np.where(explore, wide, narrow))
