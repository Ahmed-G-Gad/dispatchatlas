"""The continuous univariate marginal distribution algorithm with Gaussian marginals."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import ClassVar

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.continuous.core import Ordered, Run, initial_population
from dispatchatlas.solvers.continuous.encoded import EncodedSolver

__all__ = ["Umda"]


@dataclass(frozen=True)
class Umda(EncodedSolver):
    """UMDA with Gaussian marginals, an estimation of distribution algorithm.

    The univariate marginal distribution algorithm of Mühlenbein and Paaß
    (1996) in its continuous form, UMDA_c with normal densities (Larrañaga and
    Lozano 2002). The search model is a product of independent normal
    distributions, one per coordinate. Each generation:

    1. **selection**: the ``selection_fraction`` best members are selected
       (truncation selection);
    2. **estimation**: each coordinate's normal distribution gets the maximum
       likelihood mean and standard deviation of the selected members;
    3. **sampling**: the next population holds the ``elites`` best members,
       kept without a new evaluation, and new members sampled from the model.

    The model ignores interactions between coordinates, and its spread shrinks
    with every truncation, so the search concentrates quickly around good
    regions. On a strict box samples are clipped to the box; random keys need
    no clipping. The initial population is uniform in the box. The mean
    standard deviation of the last model appears in the result's extras as
    ``spread``.

    Attributes:
        population_size: Members per generation, at least 2.
        selection_fraction: Fraction of the population that estimates the model.
        elites: Best members kept in the next population, fewer than
            ``population_size``.
    """

    population_size: int = 100
    selection_fraction: float = 0.5
    elites: int = 1

    info: ClassVar[SolverInfo] = SolverInfo(
        id="umda",
        name="Univariate marginal distribution algorithm",
        family="population",
        summary="Fits an independent normal distribution per random key to the best members "
        "and samples the next population from their product.",
        references=("muhlenbein1996recombination", "larranaga2002estimation", "bean1994genetic"),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0.0 < self.selection_fraction <= 1.0:
            raise ValueError("selection_fraction must lie in (0, 1]")
        if not 0 <= self.elites < self.population_size:
            raise ValueError("elites must lie in [0, population_size)")

    def optimize[F: Ordered](self, run: Run[F], rng: random.Random) -> None:
        """Estimate and sample the model until the run is exhausted."""
        box = run.box
        size = self.population_size
        chosen = min(size, max(2, round(self.selection_fraction * size)))
        population, fitness = initial_population(run, size, rng)
        while True:
            run.next_iteration()
            ranked = sorted(range(size), key=lambda member: fitness[member])
            selected = [population[member] for member in ranked[:chosen]]
            columns = list(zip(*selected, strict=True))
            means = [fmean(column) for column in columns]
            deviations = [
                math.sqrt(fmean([(value - mean) ** 2 for value in column]))
                for column, mean in zip(columns, means, strict=True)
            ]
            run.note("spread", fmean(deviations))
            members = [population[member] for member in ranked[: self.elites]]
            values = [fitness[member] for member in ranked[: self.elites]]
            samples = [
                [
                    rng.gauss(mean, deviation)
                    for mean, deviation in zip(means, deviations, strict=True)
                ]
                for _ in range(size - self.elites)
            ]
            for sample in samples:
                vector = box.clip(sample) if box.strict else sample
                values.append(run.evaluate(vector))
                members.append(vector)
            population, fitness = members, values
