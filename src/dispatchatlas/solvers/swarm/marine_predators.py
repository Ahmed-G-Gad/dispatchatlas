"""Marine Predators Algorithm (Faramarzi, Heidarinejad, Mirjalili and Gandomi 2020).

Each iteration moves every member ``x`` relative to the best vector ``e`` found
so far with random vectors: ``g`` standard normal, ``l`` Lévy steps with
``beta = 1.5`` scaled by 0.05 (the scale of the authors' implementation), and
``u`` uniform. With ``P = 0.5`` and ``CF = (1 - t/T)^(2 t/T)``, the step that
depends on the budget used:

- first third (Eq. 12): ``x + P u * g * (e - g * x)``;
- second third (Eqs. 13 and 14): ``x + P u * l * (e - l * x)`` for the first
  half of the population, ``e + P CF g * (g * e - x)`` for the second half;
- last third (Eq. 15): ``e + P CF l * (l * e - x)``.

A second perturbation follows (Eq. 16). For each member a uniform ``r`` is
drawn: if ``r <= FADs``, the member moves by ``CF (LB + v (UB - LB)) * m``, with
``v`` uniform and the mask ``m`` zero where its own uniform draw is below
``FADs`` and one elsewhere, as the paper defines it (the authors' code uses the
opposite mask); otherwise it moves by ``(FADs (1 - r) + r) (x_1 - x_2)``, the
difference of two random members, scaled with the same ``r``. ``FADs = 0.2``.
The authors' code makes this choice once per iteration for the whole
population and scales the differences with a second, independent draw; this
port applies Eq. 16 to each member with a single ``r``.

After each of the two moves a member keeps its new vector unless the vector is
worse, which costs two evaluations per member and iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import (
    Swarm,
    SwarmMethod,
    check_fraction,
    check_positive,
    levy,
    method_info,
)

__all__ = ["MarinePredatorsAlgorithm"]

LEVY_BETA = 1.5
"""Index ``beta`` of the Lévy steps."""

LEVY_SCALE = 0.05
"""Factor of the Lévy steps in the authors' implementation."""


@dataclass(frozen=True)
class MarinePredatorsAlgorithm(SwarmMethod):
    """Marine Predators Algorithm on priority vectors.

    Attributes:
        step_scale: The constant ``P`` that scales every step of Eqs. 12 to 15.
        jump_probability: The constant ``FADs`` of Eq. 16: the probability of a
            masked uniform jump instead of a scaled member difference.
    """

    step_scale: float = 0.5
    jump_probability: float = 0.2

    info: ClassVar[SolverInfo] = method_info(
        "marine-predators-algorithm",
        "Marine Predators Algorithm",
        "Moves vectors relative to the best vector with Gaussian steps early, Lévy "
        "steps late and a mix of both in between, then applies masked random jumps or "
        "scaled member differences, keeping every change that is not worse.",
        "faramarzi2020marine",
        "mantegna1994fast",
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        check_positive("step_scale", self.step_scale)
        check_fraction("jump_probability", self.jump_probability, allow_zero=True)

    def step(self, swarm: Swarm) -> None:
        """Apply the step of the current third of the budget, then the perturbation."""
        rng = swarm.rng
        size, dimension = swarm.size, swarm.dimension
        progress = swarm.progress
        factor = (1.0 - progress) ** (2.0 * progress)
        scale = self.step_scale
        best = swarm.best
        for index in range(size):
            position = swarm.positions[index]
            uniform = rng.random(dimension)
            if progress < 1.0 / 3.0:
                normal = rng.standard_normal(dimension)
                target = position + scale * uniform * normal * (best - normal * position)
            elif progress < 2.0 / 3.0 and 2 * (index + 1) <= size:
                flight = LEVY_SCALE * levy(rng, dimension, LEVY_BETA)
                target = position + scale * uniform * flight * (best - flight * position)
            elif progress < 2.0 / 3.0:
                normal = rng.standard_normal(dimension)
                target = best + scale * factor * normal * (normal * best - position)
            else:
                flight = LEVY_SCALE * levy(rng, dimension, LEVY_BETA)
                target = best + scale * factor * flight * (flight * best - position)
            swarm.offer(index, target, ties=True)
        members = swarm.positions.copy()
        jump = self.jump_probability
        for index in range(size):
            r = rng.random()
            if r <= jump:
                mask = rng.random(dimension) >= jump
                target = members[index] + factor * rng.random(dimension) * mask
            else:
                first, second = rng.integers(size, size=2)
                target = members[index] + (jump * (1.0 - r) + r) * (
                    members[first] - members[second]
                )
            swarm.offer(index, target, ties=True)
