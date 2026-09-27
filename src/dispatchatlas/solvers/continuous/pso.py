"""Particle swarm optimisation with a decreasing inertia weight."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.continuous.core import Ordered, Run, initial_population
from dispatchatlas.solvers.continuous.encoded import EncodedSolver

__all__ = ["ParticleSwarmOptimization"]


@dataclass(frozen=True)
class ParticleSwarmOptimization(EncodedSolver):
    """Global-best particle swarm optimisation with a decreasing inertia weight.

    Each particle has a position ``x``, a velocity ``v``, and the best position
    it has visited, ``p``; ``g`` is the best position of the whole swarm. Every
    generation moves each particle in turn, coordinate by coordinate:

    ``v = w v + c1 r1 (p - x) + c2 r2 (g - x)`` and then ``x = x + v``,

    with ``r1`` and ``r2`` uniform in [0, 1), drawn per coordinate (Kennedy and
    Eberhart 1995). The inertia weight ``w`` of Shi and Eberhart (1998)
    decreases linearly from ``inertia_start`` to ``inertia_end`` as the budget
    is spent (Shi and Eberhart 1999). Each velocity coordinate is limited to
    ``velocity_limit`` times the box width. After each move the particle is
    evaluated, updates ``p`` when strictly better, and ``g`` at once when better
    than the swarm's best. On a strict box positions are clipped to the box;
    random keys leave it free. Velocities start uniform within their limits.

    Attributes:
        population_size: Particles.
        inertia_start: Inertia weight at the start of the run.
        inertia_end: Inertia weight at the end of the run.
        cognitive: Acceleration ``c1`` toward the particle's own best.
        social: Acceleration ``c2`` toward the swarm's best.
        velocity_limit: Velocity limit as a fraction of the box width; 1 limits
            each step to the dynamic range of its coordinate.
    """

    population_size: int = 30
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    cognitive: float = 2.0
    social: float = 2.0
    velocity_limit: float = 1.0

    info: ClassVar[SolverInfo] = SolverInfo(
        id="particle-swarm-optimization",
        name="Particle swarm optimization",
        family="swarm",
        summary="Moves random-key vectors with velocities that combine inertia, a pull "
        "toward each particle's best position, and a pull toward the swarm's best, "
        "with the inertia weight decreasing over the budget.",
        references=(
            "kennedy1995particle",
            "shi1998modified",
            "shi1999empirical",
            "bean1994genetic",
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.population_size < 1:
            raise ValueError("population_size must be at least 1")
        if not (0.0 <= self.inertia_end <= self.inertia_start):
            raise ValueError("the inertia weight must satisfy 0 <= inertia_end <= inertia_start")
        if self.cognitive < 0.0 or self.social < 0.0:
            raise ValueError("the accelerations must be non-negative")
        if not self.velocity_limit > 0.0:
            raise ValueError("velocity_limit must be positive")

    def inertia(self, progress: float) -> float:
        """Return the inertia weight after ``progress`` of the budget, a fraction in [0, 1]."""
        return self.inertia_start - (self.inertia_start - self.inertia_end) * progress

    def optimize[F: Ordered](self, run: Run[F], rng: random.Random) -> None:
        """Move the swarm until the run is exhausted."""
        box = run.box
        size = self.population_size
        positions, fitness = initial_population(run, size, rng)
        limits = [self.velocity_limit * width for width in box.widths]
        velocities = [[rng.uniform(-limit, limit) for limit in limits] for _ in range(size)]
        bests = [list(position) for position in positions]
        best_fitness = list(fitness)
        leader = min(range(size), key=lambda particle: best_fitness[particle])
        while True:
            weight = self.inertia(run.progress)
            run.next_iteration()
            for particle in range(size):
                position, velocity = positions[particle], velocities[particle]
                own, swarm = bests[particle], bests[leader]
                for j, limit in enumerate(limits):
                    step = (
                        weight * velocity[j]
                        + self.cognitive * rng.random() * (own[j] - position[j])
                        + self.social * rng.random() * (swarm[j] - position[j])
                    )
                    velocity[j] = min(limit, max(-limit, step))
                    position[j] += velocity[j]
                if box.strict:
                    position[:] = box.clip(position)
                value = run.evaluate(position)
                if value < best_fitness[particle]:
                    bests[particle], best_fitness[particle] = list(position), value
                    if value < best_fitness[leader]:
                        leader = particle
