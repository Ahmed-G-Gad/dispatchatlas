"""Genetic and memetic algorithms over task sequences and mode settings."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.constructive import Items
from dispatchatlas.solvers.operators import order_crossover, uniform_mode_crossover
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory.space import SequenceSpace, Solution

__all__ = ["GeneticAlgorithm", "MemeticAlgorithm", "recombine"]


def recombine(
    items: Items, first: Solution, second: Solution, rng: random.Random
) -> tuple[list[int], tuple[int, ...] | None]:
    """Cross two solutions: order crossover of the sequences, uniform crossover of the modes.

    The order crossover of Davis (1985) copies a random slice of ``first``'s
    sequence and fills the other positions with the missing items in the order
    they appear in ``second``, from the end of the slice onwards. The result is
    repaired so that every item follows its predecessors, keeping the crossed
    order as far as precedence allows. Each task takes its mode setting from
    either parent with equal probability.
    """
    sequence = items.feasible(order_crossover(first.sequence, second.sequence, rng))
    if first.modes is None or second.modes is None:
        return sequence, None
    return sequence, tuple(uniform_mode_crossover(first.modes, second.modes, rng))


@dataclass(frozen=True)
class GeneticAlgorithm:
    """A generational genetic algorithm over task sequences (Holland 1992).

    Solutions are those of the trajectory methods'
    :class:`~dispatchatlas.solvers.trajectory.space.SequenceSpace`: a
    precedence-feasible sequence of items (tasks, or whole jobs in permutation
    problems) and a mode setting for every multi-mode task, either a fixed mode
    or ``AUTO``, the earliest-finishing mode when the task is placed. This is
    the activity-list representation that Hartmann (1998) evolves with the
    serial schedule-generation scheme. The initial population holds random
    feasible sequences with every multi-mode task on ``AUTO``. Each generation
    builds a new population:

    1. **elitism**: the ``elites`` best members pass unchanged, without a new
       evaluation;
    2. **selection**: each further child has two parents, each the best of
       ``tournament_size`` members drawn uniformly with replacement;
    3. **crossover**: with probability ``crossover_rate``, the order crossover
       of Davis (1985) and a uniform crossover of the mode settings (see
       :func:`recombine`); otherwise a copy of the first parent;
    4. **mutation**: with probability ``mutation_rate``, one random move of the
       space: an item inserted elsewhere, two items swapped, or a task's mode
       setting changed, always keeping precedence.

    A child identical to one of its parents inherits the parent's score without
    a new evaluation. One generation is one iteration.

    Attributes:
        population_size: Members per generation, at least 2.
        crossover_rate: Probability that a child is a crossover of its parents.
        mutation_rate: Probability that a child receives one random move.
        tournament_size: Members drawn per tournament.
        elites: Best members copied into the next generation, fewer than
            ``population_size``.
    """

    population_size: int = 30
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament_size: int = 2
    elites: int = 2

    info: ClassVar[SolverInfo] = SolverInfo(
        id="genetic-algorithm",
        name="Genetic algorithm",
        family="population",
        summary="Evolves task sequences and mode settings by tournament selection, order "
        "crossover, uniform mode crossover, and insert, swap, or mode mutation, keeping "
        "the best members from one generation to the next.",
        references=("holland1992adaptation", "davis1985applying", "hartmann1998competitive"),
    )

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must lie in [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must lie in [0, 1]")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")
        if not 0 <= self.elites < self.population_size:
            raise ValueError("elites must lie in [0, population_size)")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Evolve the population until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space = SequenceSpace(search)
            if space.movable:
                self._evolve(space)
            else:
                space.random_start()
        return search.result()

    def _evolve(self, space: SequenceSpace) -> None:
        population = [space.random_start() for _ in range(self.population_size)]
        while True:
            space.search.next_iteration()
            population = self._generation(space, population)

    def _generation(self, space: SequenceSpace, population: list[Solution]) -> list[Solution]:
        rng = space.search.rng
        ranked = sorted(population, key=lambda member: member.score)
        children = ranked[: self.elites]
        while len(children) < self.population_size:
            first = self._select(population, rng)
            second = self._select(population, rng)
            children.append(self._offspring(space, first, second))
        return children

    def _select(self, population: Sequence[Solution], rng: random.Random) -> Solution:
        """Return the best of ``tournament_size`` members drawn with replacement."""
        winner = population[rng.randrange(len(population))]
        for _ in range(self.tournament_size - 1):
            rival = population[rng.randrange(len(population))]
            if rival.score < winner.score:
                winner = rival
        return winner

    def _offspring(self, space: SequenceSpace, first: Solution, second: Solution) -> Solution:
        rng = space.search.rng
        sequence: Sequence[int]
        modes: Sequence[int] | None
        if rng.random() < self.crossover_rate:
            sequence, modes = recombine(space.items, first, second, rng)
        else:
            sequence, modes = first.sequence, first.modes
        if rng.random() < self.mutation_rate:
            move = space.sample(sequence, modes)
            if move is not None:
                sequence, modes = space.apply(sequence, modes, move)
        key = (tuple(sequence), None if modes is None else tuple(modes))
        for parent in (first, second):
            if parent.key == key:
                return parent
        return self._improve(space, space.evaluate(sequence, modes))

    def _improve(self, space: SequenceSpace, child: Solution) -> Solution:
        """Return the child as it enters the population."""
        del space
        return child


@dataclass(frozen=True)
class MemeticAlgorithm(GeneticAlgorithm):
    """A memetic algorithm: the genetic algorithm with local search on offspring (Moscato 1989).

    Every evaluated child is improved, with probability
    ``local_search_rate``, by the insertion local search of
    :meth:`~dispatchatlas.solvers.trajectory.space.SequenceSpace.descend`, the
    iterative improvement of Ruiz and Stützle (2007) extended to modes: each
    item is reinserted at its best position and each of its multi-mode tasks
    tries its other mode settings, passes repeating until one changes nothing.
    The population therefore holds local optima, and crossover and mutation
    move between their basins. The rest follows :class:`GeneticAlgorithm`.

    Attributes:
        local_search_rate: Probability that an evaluated child is improved.
    """

    population_size: int = 20
    local_search_rate: float = 1.0

    info: ClassVar[SolverInfo] = SolverInfo(
        id="memetic-algorithm",
        name="Memetic algorithm",
        family="population",
        summary="Evolves task sequences and mode settings like the genetic algorithm and "
        "improves every child by an insertion local search, so that the population "
        "holds local optima.",
        references=("moscato1989evolution", "davis1985applying", "ruiz2007simple"),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.local_search_rate <= 1.0:
            raise ValueError("local_search_rate must lie in [0, 1]")

    def _improve(self, space: SequenceSpace, child: Solution) -> Solution:
        """Improve the child by insertion local search with probability ``local_search_rate``."""
        if space.search.rng.random() < self.local_search_rate:
            return space.descend(child)
        return child
