"""Comprehensive learning particle swarm optimisation (Liang, Qin, Suganthan and Baskar 2006)."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.continuous.core import Ordered, Run, initial_population
from dispatchatlas.solvers.continuous.encoded import EncodedSolver

__all__ = ["Clpso", "learning_probabilities", "select_exemplars"]


def learning_probabilities(size: int) -> list[float]:
    """Return the learning probability of each of ``size`` particles.

    Particle ``i`` (from 0) learns from others with probability
    ``0.05 + 0.45 (exp(10 i / (N - 1)) - 1) / (exp(10) - 1)``, rising from 0.05
    for the first particle to 0.5 for the last, so the swarm mixes particles
    that mostly follow their own best with particles that mostly follow others.
    """
    if size < 2:
        return [0.05] * size
    scale = math.exp(10.0) - 1.0
    return [0.05 + 0.45 * (math.exp(10.0 * i / (size - 1)) - 1.0) / scale for i in range(size)]


def select_exemplars[F: Ordered](
    particle: int,
    best_fitness: Sequence[F],
    probability: float,
    dimension: int,
    rng: random.Random,
) -> list[int]:
    """Choose, for each coordinate, whose best position ``particle`` learns from.

    With probability ``probability`` a coordinate follows the winner of a
    tournament between two other particles drawn uniformly, the one with the
    better best position (the first drawn on ties); otherwise it follows the
    particle's own best. When every coordinate chose the particle itself, one
    coordinate drawn uniformly follows a tournament winner instead. Needs at
    least three particles.
    """
    exemplars = [
        _tournament(particle, best_fitness, rng) if rng.random() < probability else particle
        for _ in range(dimension)
    ]
    if all(choice == particle for choice in exemplars):
        exemplars[rng.randrange(dimension)] = _tournament(particle, best_fitness, rng)
    return exemplars


def _tournament[F: Ordered](particle: int, best_fitness: Sequence[F], rng: random.Random) -> int:
    first, second = (k + (k >= particle) for k in rng.sample(range(len(best_fitness) - 1), 2))
    return first if best_fitness[first] <= best_fitness[second] else second


@dataclass(frozen=True)
class Clpso(EncodedSolver):
    """The comprehensive learning particle swarm optimiser of Liang et al. (2006).

    Each coordinate of each particle learns from a possibly different
    particle's best position, its *exemplar*, so a particle follows a
    composite of the swarm's experience rather than one leader:

    ``v_d = w v_d + c r_d (p_{f(d), d} - x_d)`` and then ``x_d = x_d + v_d``,

    with ``r_d`` uniform in [0, 1) and ``f(d)`` the exemplar of coordinate
    ``d`` (see :func:`select_exemplars`), chosen with the particle's learning
    probability (see :func:`learning_probabilities`). A particle whose best
    position has not improved for ``refresh_gap`` evaluations draws new
    exemplars. The inertia weight decreases linearly from ``inertia_start`` to
    ``inertia_end`` as the budget is spent, and each velocity coordinate is
    limited to ``velocity_limit`` times the box width, 20 % in the original.
    Velocities start uniform within their limits.

    Positions are never clipped. On a strict box, the original bound handling
    applies: a particle outside the box is neither evaluated nor allowed to
    update its best position, and its exemplars, all inside the box, draw it
    back. Random keys decode anywhere, so there every particle is evaluated
    after every move.

    The defaults are the published settings; Liang et al. use 10 particles for
    10-dimensional problems and 40 for 30-dimensional ones.

    Attributes:
        population_size: Particles, at least 3.
        inertia_start: Inertia weight at the start of the run.
        inertia_end: Inertia weight at the end of the run.
        acceleration: The acceleration ``c``.
        refresh_gap: Evaluations without improvement before new exemplars, ``m``.
        velocity_limit: Velocity limit as a fraction of the box width.
    """

    population_size: int = 40
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    acceleration: float = 1.49445
    refresh_gap: int = 7
    velocity_limit: float = 0.2

    info: ClassVar[SolverInfo] = SolverInfo(
        id="clpso",
        name="Comprehensive learning particle swarm optimizer",
        family="swarm",
        summary="Moves random-key vectors toward exemplars whose coordinates come from "
        "different particles' best positions, drawing new exemplars for a particle that "
        "stops improving.",
        references=("liang2006comprehensive", "bean1994genetic"),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.population_size < 3:
            raise ValueError("population_size must be at least 3")
        if not (0.0 <= self.inertia_end <= self.inertia_start):
            raise ValueError("the inertia weight must satisfy 0 <= inertia_end <= inertia_start")
        if not self.acceleration > 0.0:
            raise ValueError("acceleration must be positive")
        if self.refresh_gap < 1:
            raise ValueError("refresh_gap must be at least 1")
        if not self.velocity_limit > 0.0:
            raise ValueError("velocity_limit must be positive")

    def optimize[F: Ordered](self, run: Run[F], rng: random.Random) -> None:
        """Move the swarm until the run is exhausted."""
        box = run.box
        size, dimension = self.population_size, box.dimension
        probabilities = learning_probabilities(size)
        positions, fitness = initial_population(run, size, rng)
        limits = [self.velocity_limit * width for width in box.widths]
        velocities = [[rng.uniform(-limit, limit) for limit in limits] for _ in range(size)]
        bests = [list(position) for position in positions]
        best_fitness = list(fitness)
        exemplars = [
            select_exemplars(particle, best_fitness, probabilities[particle], dimension, rng)
            for particle in range(size)
        ]
        stalled = [0] * size
        span = self.inertia_start - self.inertia_end
        while True:
            weight = self.inertia_start - span * run.progress
            run.next_iteration()
            for particle in range(size):
                if stalled[particle] >= self.refresh_gap:
                    exemplars[particle] = select_exemplars(
                        particle, best_fitness, probabilities[particle], dimension, rng
                    )
                    stalled[particle] = 0
                position, velocity = positions[particle], velocities[particle]
                for j, (source, limit) in enumerate(zip(exemplars[particle], limits, strict=True)):
                    step = weight * velocity[j] + self.acceleration * rng.random() * (
                        bests[source][j] - position[j]
                    )
                    velocity[j] = min(limit, max(-limit, step))
                    position[j] += velocity[j]
                if box.strict and not box.contains(position):
                    continue
                value = run.evaluate(position)
                if value < best_fitness[particle]:
                    bests[particle], best_fitness[particle] = list(position), value
                    stalled[particle] = 0
                else:
                    stalled[particle] += 1
