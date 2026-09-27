"""The biased random-key genetic algorithm (Gonçalves and Resende 2011)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.continuous.core import Ordered, Run, initial_population
from dispatchatlas.solvers.continuous.encoded import EncodedSolver

__all__ = ["Brkga"]


@dataclass(frozen=True)
class Brkga(EncodedSolver):
    """The biased random-key genetic algorithm of Gonçalves and Resende (2011).

    A chromosome is a vector of random keys in the unit cube, decoded into a
    schedule as random-key genetic algorithms do (Bean 1994). Each generation
    ranks the population and builds the next one from three parts:

    1. the **elite**, the ``elite_fraction`` best members, copied unchanged and
       never evaluated again;
    2. **mutants**, ``mutant_fraction`` of new uniform random chromosomes;
    3. **offspring** for the remaining places, each crossing a parent drawn
       uniformly from the elite with one drawn uniformly from the non-elite
       members: every key comes from the elite parent with probability
       ``inheritance`` and from the other parent otherwise.

    Each part keeps at least one member. The published recommendations are a
    population of ``a n`` members for a chromosome of ``n`` keys and some
    ``a >= 1``, an elite of 10 to 25 %, mutants of 10 to 30 %, and an
    inheritance probability above 0.5, up to 0.8. With ``seeded`` the initial
    population holds the heuristic schedule's keys.

    Attributes:
        population_size: Members per generation, at least 3. ``None`` uses the
            chromosome length ``n`` (``a = 1``), and at least 3.
        elite_fraction: Fraction of the population kept as the elite.
        mutant_fraction: Fraction of the population replaced by mutants.
        inheritance: Probability that a key of a child comes from its elite parent.
    """

    population_size: int | None = None
    elite_fraction: float = 0.2
    mutant_fraction: float = 0.15
    inheritance: float = 0.7

    info: ClassVar[SolverInfo] = SolverInfo(
        id="brkga",
        name="Biased random-key genetic algorithm",
        family="population",
        summary="Keeps the elite random-key chromosomes, adds random mutants, and fills the "
        "population with crossings of an elite and a non-elite parent that favour the "
        "elite parent's keys.",
        references=("goncalves2011biased", "bean1994genetic"),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.population_size is not None and self.population_size < 3:
            raise ValueError("population_size must be at least 3")
        if not 0.0 < self.elite_fraction < 1.0:
            raise ValueError("elite_fraction must lie in (0, 1)")
        if not 0.0 <= self.mutant_fraction < 1.0:
            raise ValueError("mutant_fraction must lie in [0, 1)")
        if self.elite_fraction + self.mutant_fraction >= 1.0:
            raise ValueError("elite_fraction plus mutant_fraction must be below 1")
        if not 0.5 <= self.inheritance < 1.0:
            raise ValueError("inheritance must lie in [0.5, 1)")

    def partition(self, size: int) -> tuple[int, int]:
        """Return the elite and mutant counts of a population of ``size`` members.

        The elite keeps at least one member and the offspring at least one place.
        """
        elite = min(size - 1, max(1, round(self.elite_fraction * size)))
        mutants = min(size - elite - 1, round(self.mutant_fraction * size))
        return elite, mutants

    def optimize[F: Ordered](self, run: Run[F], rng: random.Random) -> None:
        """Evolve the chromosomes until the run is exhausted."""
        box = run.box
        size = self.population_size or max(3, box.dimension)
        elite_count, mutant_count = self.partition(size)
        population, fitness = initial_population(run, size, rng)
        while True:
            run.next_iteration()
            ranked = sorted(range(size), key=lambda member: fitness[member])
            elite, others = ranked[:elite_count], ranked[elite_count:]
            members = [population[member] for member in elite]
            values = [fitness[member] for member in elite]
            fresh = [box.sample(rng) for _ in range(mutant_count)]
            for _ in range(size - elite_count - mutant_count):
                favoured = population[elite[rng.randrange(elite_count)]]
                other = population[others[rng.randrange(len(others))]]
                fresh.append(
                    [
                        key if rng.random() < self.inheritance else alternative
                        for key, alternative in zip(favoured, other, strict=True)
                    ]
                )
            for chromosome in fresh:
                values.append(run.evaluate(chromosome))
                members.append(chromosome)
            population, fitness = members, values
