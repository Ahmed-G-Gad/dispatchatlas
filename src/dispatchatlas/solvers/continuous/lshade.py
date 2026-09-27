"""L-SHADE: success-history adaptive differential evolution with linear population reduction."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
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

__all__ = ["LShade", "SuccessMemory", "lehmer_mean", "planned_population"]


def lehmer_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Return the weighted Lehmer mean ``sum(w s^2) / sum(w s)``.

    Falls back to the arithmetic mean when the denominator is not positive.
    """
    denominator = sum(w * s for w, s in zip(weights, values, strict=True))
    if denominator <= 0.0:
        return sum(values) / len(values)
    return sum(w * s * s for w, s in zip(weights, values, strict=True)) / denominator


def planned_population(initial: int, minimum: int, progress: float) -> int:
    """Return the population size of linear population size reduction.

    ``round(initial + (minimum - initial) * progress)``: the size falls on a
    straight line from ``initial`` to ``minimum`` as the budget is spent.
    Tanabe and Fukunaga (2014) key the line to the evaluations spent;
    ``progress`` follows whichever limit binds.
    """
    return max(minimum, round(initial + (minimum - initial) * progress))


class SuccessMemory:
    """The success-history memory of SHADE (Tanabe and Fukunaga 2013) as L-SHADE uses it.

    ``H`` slots hold a crossover-rate mean ``M_CR`` and a scale-factor mean
    ``M_F``, both 0.5 at first. A trial draws a slot uniformly, then
    ``CR ~ N(M_CR, 0.1)`` clipped to [0, 1], or exactly 0 when the slot holds the
    *terminal* value, and ``F ~ Cauchy(M_F, 0.1)``, drawn again while not
    positive and capped at 1. After a generation with successful trials, the
    slot under the cursor receives the weighted Lehmer means of the successful
    ``CR`` and ``F`` values, weighted by their fitness improvements, and the
    cursor advances. The slot's ``M_CR`` becomes terminal for good when it
    already was, or when every successful ``CR`` was 0.

    Args:
        size: Slots ``H``.

    Attributes:
        crossover: ``M_CR`` of each slot, meaningless for terminal slots.
        scale: ``M_F`` of each slot.
        terminal: Whether each slot's ``M_CR`` holds the terminal value.
        cursor: The slot the next update writes.
    """

    def __init__(self, size: int) -> None:
        self.crossover = [0.5] * size
        self.scale = [0.5] * size
        self.terminal = [False] * size
        self.cursor = 0

    def sample(self, rng: random.Random) -> tuple[float, float]:
        """Draw the ``(CR, F)`` pair of one trial."""
        slot = rng.randrange(len(self.crossover))
        if self.terminal[slot]:
            rate = 0.0
        else:
            rate = min(1.0, max(0.0, rng.gauss(self.crossover[slot], 0.1)))
        scale = _cauchy(self.scale[slot], rng)
        while scale <= 0.0:
            scale = _cauchy(self.scale[slot], rng)
        return rate, min(1.0, scale)

    def update(
        self, rates: Sequence[float], scales: Sequence[float], improvements: Sequence[float]
    ) -> None:
        """Fold a generation's successful trials into the slot under the cursor.

        A generation without success changes nothing, not even the cursor.
        Improvements weigh the means; when they do not sum to a positive
        number, the successes weigh equally.
        """
        if not rates:
            return
        total = sum(improvements)
        count = len(rates)
        weights = (
            [value / total for value in improvements] if total > 0.0 else [1.0 / count] * count
        )
        slot = self.cursor
        if self.terminal[slot] or max(rates) <= 0.0:
            self.terminal[slot] = True
        else:
            self.crossover[slot] = lehmer_mean(rates, weights)
        self.scale[slot] = lehmer_mean(scales, weights)
        self.cursor = (slot + 1) % len(self.crossover)

    def mean_crossover(self) -> float:
        """Return the mean ``M_CR`` over the slots, a terminal slot counting as 0."""
        values = [
            0.0 if flag else value
            for value, flag in zip(self.crossover, self.terminal, strict=True)
        ]
        return sum(values) / len(values)


def _cauchy(location: float, rng: random.Random) -> float:
    """Return a Cauchy sample of scale 0.1 around ``location``, by inverse transform."""
    return location + 0.1 * math.tan(math.pi * (rng.random() - 0.5))


@dataclass(frozen=True)
class LShade(EncodedSolver):
    """L-SHADE, the adaptive differential evolution of Tanabe and Fukunaga (2014).

    Every generation builds one trial per member ``x_i`` from the population
    at the start of the generation, with its own ``CR`` and ``F`` drawn from a
    :class:`SuccessMemory`:

    1. **current-to-pbest/1 mutation** (Zhang and Sanderson 2009):
       ``v = x_i + F (x_pbest - x_i) + F (x_r1 - x_r2)``, where ``x_pbest`` is
       drawn uniformly from the ``max(2, round(p N))`` best members, ``x_r1``
       from the population and ``x_r2`` from the population together with the
       archive, with ``i``, ``r1``, and ``r2`` distinct positions;
    2. **binomial crossover** with rate ``CR``, one uniformly drawn coordinate
       always taken from ``v``;
    3. **repair**: a coordinate outside the box moves halfway back to that of
       ``x_i``;
    4. **selection**: the trial replaces ``x_i`` when no worse. When strictly
       better, ``x_i`` enters the archive and ``(CR, F)`` counts as a success,
       weighted by the improvement (for schedules, see
       :func:`~dispatchatlas.solvers.continuous.encoded.score_improvement`).

    After the generation the memory absorbs the successes, the population
    shrinks to :func:`planned_population` by dropping its worst members, and
    random archive members are dropped until at most ``round(archive_rate N)``
    remain. The population thus reaches ``min_population`` at the end of the
    run, whichever limit binds.

    The defaults are the published settings: ``N_init = 18 D``, ``N_min = 4``,
    ``H = 6``, ``p = 0.11``, and an archive rate of 2.6. The final population
    size and the mean memories appear in the result's extras.

    Attributes:
        population_size: Initial population ``N_init``. ``None`` uses ``18 D``
            for a ``D``-dimensional box.
        min_population: Final population ``N_min``, at least 4.
        memory_size: Memory slots ``H``.
        best_fraction: Fraction ``p`` of the population ``x_pbest`` is drawn from.
        archive_rate: Archive capacity as a multiple of the population size.
    """

    population_size: int | None = None
    min_population: int = 4
    memory_size: int = 6
    best_fraction: float = 0.11
    archive_rate: float = 2.6

    info: ClassVar[SolverInfo] = SolverInfo(
        id="lshade",
        name="L-SHADE",
        family="population",
        summary="Evolves random-key vectors by current-to-pbest differential mutation with "
        "an archive, draws each trial's crossover rate and scale factor from a memory of "
        "successful values, and shrinks the population linearly over the budget.",
        references=(
            "tanabe2014improving",
            "tanabe2013success",
            "zhang2009jade",
            "bean1994genetic",
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_population < 4:
            raise ValueError("min_population must be at least 4")
        if self.population_size is not None and self.population_size < self.min_population:
            raise ValueError("population_size must be at least min_population")
        if self.memory_size < 1:
            raise ValueError("memory_size must be at least 1")
        if not 0.0 < self.best_fraction <= 1.0:
            raise ValueError("best_fraction must lie in (0, 1]")
        if self.archive_rate < 0.0:
            raise ValueError("archive_rate must be non-negative")

    def optimize[F: Ordered](self, run: Run[F], rng: random.Random) -> None:
        """Evolve the population until the run is exhausted."""
        box = run.box
        initial = max(self.population_size or 18 * box.dimension, self.min_population)
        population, fitness = initial_population(run, initial, rng)
        memory = SuccessMemory(self.memory_size)
        archive: list[list[float]] = []
        while True:
            run.next_iteration()
            size = len(population)
            ranked = sorted(range(size), key=lambda member: fitness[member])
            leaders = ranked[: max(2, round(self.best_fraction * size))]
            controls = [memory.sample(rng) for _ in range(size)]
            trials = [
                self._trial(box, population, archive, target, leaders, controls[target], rng)
                for target in range(size)
            ]
            rates: list[float] = []
            scales: list[float] = []
            improvements: list[float] = []
            for target, trial in enumerate(trials):
                value = run.evaluate(trial)
                if value <= fitness[target]:
                    if value < fitness[target]:
                        archive.append(population[target])
                        rates.append(controls[target][0])
                        scales.append(controls[target][1])
                        improvements.append(run.improvement(fitness[target], value))
                    population[target], fitness[target] = trial, value
            memory.update(rates, scales, improvements)
            planned = planned_population(initial, self.min_population, run.progress)
            if planned < size:
                survivors = sorted(
                    sorted(range(size), key=lambda member: fitness[member])[:planned]
                )
                population = [population[member] for member in survivors]
                fitness = [fitness[member] for member in survivors]
            capacity = round(self.archive_rate * len(population))
            while len(archive) > capacity:
                archive.pop(rng.randrange(len(archive)))
            run.note("population", len(population))
            run.note("memory_cr", memory.mean_crossover())
            run.note("memory_f", sum(memory.scale) / len(memory.scale))

    @staticmethod
    def _trial(
        box: Box,
        population: list[list[float]],
        archive: list[list[float]],
        target: int,
        leaders: list[int],
        control: tuple[float, float],
        rng: random.Random,
    ) -> list[float]:
        """Build the trial of ``target``: current-to-pbest/1 mutation, crossover, and repair."""
        rate, scale = control
        size = len(population)
        parent = population[target]
        leader = population[leaders[rng.randrange(len(leaders))]]
        first = rng.randrange(size - 1)
        first += first >= target
        pool = size + len(archive)
        second = rng.randrange(pool)
        while second in (target, first):
            second = rng.randrange(pool)
        partner = population[first]
        other = population[second] if second < size else archive[second - size]
        forced = rng.randrange(box.dimension)
        trial = list(parent)
        for j in range(box.dimension):
            if j == forced or rng.random() < rate:
                value = (
                    parent[j] + scale * (leader[j] - parent[j]) + scale * (partner[j] - other[j])
                )
                trial[j] = midpoint_repair(value, parent[j], box.lower[j], box.upper[j])
        return trial
