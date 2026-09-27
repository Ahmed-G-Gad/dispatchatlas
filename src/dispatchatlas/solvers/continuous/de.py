"""Differential evolution: DE/rand/1/bin of Storn and Price (1997)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.continuous.core import (
    Box,
    Ordered,
    Run,
    initial_population,
    midpoint_repair,
)
from dispatchatlas.solvers.continuous.encoded import EncodedSolver

__all__ = ["DifferentialEvolution"]


@dataclass(frozen=True)
class DifferentialEvolution(EncodedSolver):
    """DE/rand/1/bin, the differential evolution of Storn and Price (1997).

    Each generation builds one trial vector per member, the *target* ``x_i``,
    from the population at the start of the generation:

    1. **mutation**: ``v = x_a + F (x_b - x_c)`` for three distinct members
       ``a``, ``b``, ``c`` other than ``i``, drawn uniformly;
    2. **binomial crossover**: the trial takes each coordinate from ``v`` with
       probability ``CR`` and from ``x_i`` otherwise; one coordinate drawn
       uniformly always comes from ``v``;
    3. **selection**: the trial replaces ``x_i`` when it is no worse.

    On a strict box a mutated coordinate that leaves the box is moved halfway
    back to the target's coordinate, the repair of JADE (Zhang and Sanderson
    2009); random keys need no repair. The population starts uniform in the
    box, with the seed, if any, as one member.

    Attributes:
        population_size: Members ``NP``, at least 4. ``None`` uses ``5 D`` for
            a ``D``-dimensional box, the lower end of the 5 D to 10 D that Storn
            and Price recommend, which leaves more generations in a budget.
        scale_factor: Difference weight ``F``, in (0, 2]; they recommend 0.5.
        crossover_rate: Crossover probability ``CR``, in [0, 1]; 0.1 is their
            first choice, and 0.9 or 1 may speed up convergence.
    """

    population_size: int | None = None
    scale_factor: float = 0.5
    crossover_rate: float = 0.1

    info: ClassVar[SolverInfo] = SolverInfo(
        id="differential-evolution",
        name="Differential evolution",
        family="population",
        summary="Adds a scaled difference of two random members to a third, crosses the "
        "result with each member coordinate by coordinate, and keeps the trial when it "
        "is no worse, over random-key vectors.",
        references=("storn1997differential", "bean1994genetic", "zhang2009jade"),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.population_size is not None and self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if not 0.0 < self.scale_factor <= 2.0:
            raise ValueError("scale_factor must lie in (0, 2]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must lie in [0, 1]")

    def optimize[F: Ordered](self, run: Run[F], rng: random.Random) -> None:
        """Evolve the population until the run is exhausted."""
        box = run.box
        size = self.population_size or max(4, 5 * box.dimension)
        population, fitness = initial_population(run, size, rng)
        while True:
            run.next_iteration()
            trials = [self._trial(population, target, box, rng) for target in range(size)]
            for target, trial in enumerate(trials):
                value = run.evaluate(trial)
                if value <= fitness[target]:
                    population[target], fitness[target] = trial, value

    def _trial(
        self, population: list[list[float]], target: int, box: Box, rng: random.Random
    ) -> list[float]:
        """Build the trial vector of ``target`` by DE/rand/1 mutation and binomial crossover."""
        others = rng.sample(range(len(population) - 1), 3)
        base, first, second = (population[k + (k >= target)] for k in others)
        parent = population[target]
        forced = rng.randrange(box.dimension)
        trial = list(parent)
        for j in range(box.dimension):
            if j == forced or rng.random() < self.crossover_rate:
                value = base[j] + self.scale_factor * (first[j] - second[j])
                if box.strict:
                    value = midpoint_repair(value, parent[j], box.lower[j], box.upper[j])
                trial[j] = value
        return trial
