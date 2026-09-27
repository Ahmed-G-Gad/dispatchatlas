"""The evolutionary loop that every multi-objective solver of this package runs.

A genome is an item sequence, as :class:`~dispatchatlas.solvers.constructive.Items`
defines it (tasks, or whole jobs in permutation problems), together with an
explicit mode for every task. Every genome is decoded by the shared schedule
builder through :meth:`~dispatchatlas.solvers.search.Search.assess` and scored
on every objective the problem declares through
:meth:`~dispatchatlas.solvers.search.Search.vector`, so each candidate counts
against the budget and enters the run's archive of non-dominated feasible
schedules. That archive is the front the result reports, so every method
returns the best trade-offs it ever evaluated, not only its last population.

Variation is the same for every method. With probability ``crossover_rate``
a child takes the order crossover of Davis (1985) of its parents' sequences,
repaired to respect precedence, and each task's mode from either parent with
equal probability; otherwise it copies its first parent. With probability
``mutation_rate`` it then moves one item to another position or swaps two
items, again repaired, and each multi-mode task redraws its mode with
probability ``mode_mutation`` divided by the number of such tasks. A child
identical to a parent receives further single moves, so that the budget is
not spent on copies.

Constraint handling follows the constrained domination of Deb, Pratap,
Agarwal and Meyarivan (2002): a schedule that meets every hard deadline is
better than one that does not, and of two that do not, the one with the
smaller total overrun is better. Environmental selection therefore applies a
method's rule to the feasible members and fills any remaining places with the
infeasible members of smallest overrun, and every mating key starts with the
overrun.

Methods differ only in their selection, which subclasses of
:class:`GenerationalSelection` and :class:`SteadyStateSelection` implement.
One generation, of as many children as the method's population size, is one
iteration.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.constructive import Items
from dispatchatlas.solvers.multiobjective.selection import Survivors
from dispatchatlas.solvers.operators import (
    mutate_modes,
    order_crossover,
    random_insert,
    random_modes,
    random_swap,
    uniform_mode_crossover,
)
from dispatchatlas.solvers.search import Search

__all__ = [
    "Evolution",
    "GenerationalSelection",
    "Member",
    "MultiObjectiveEvolution",
    "Selection",
    "SteadyStateSelection",
    "feasible_first",
    "keep",
    "tournament",
]

type Genome = tuple[tuple[int, ...], tuple[int, ...]]

RETRIES = 8
"""Most extra moves a child identical to a parent receives before it is evaluated anyway."""


class Member:
    """An evaluated genome.

    Attributes:
        sequence: Item indices in sequence order.
        modes: Mode index of every task.
        vector: Every problem objective in minimisation form.
        violation: Total hard-deadline overrun, zero when feasible.
        fitness: Mating key set by the method's selection; a binary tournament
            prefers the smaller key. It starts with the overrun, so feasible
            members always win against infeasible ones.
    """

    __slots__ = ("fitness", "modes", "sequence", "vector", "violation")

    def __init__(
        self,
        sequence: tuple[int, ...],
        modes: tuple[int, ...],
        vector: tuple[float, ...],
        violation: float,
    ) -> None:
        self.sequence = sequence
        self.modes = modes
        self.vector = vector
        self.violation = violation
        self.fitness: tuple[float, ...] = (violation,)

    @property
    def feasible(self) -> bool:
        """Whether the schedule meets every hard deadline."""
        return self.violation <= 0.0

    @property
    def genome(self) -> Genome:
        """The decisions of the member: its sequence and its modes."""
        return self.sequence, self.modes


class Evolution:
    """Genomes, variation, and budget of one run.

    Args:
        search: The run.
        crossover_rate: Probability that a child is a crossover of its parents.
        mutation_rate: Probability that a child's sequence receives one move.
        mode_mutation: Expected number of multi-mode tasks whose mode a child
            redraws.

    Attributes:
        search: The run.
        rng: The run's random generator.
        problem: The problem.
        items: The items that sequences order.
        flexible: Indices of the tasks with more than one mode.
        objectives: Number of problem objectives.
    """

    def __init__(
        self,
        search: Search,
        crossover_rate: float,
        mutation_rate: float,
        mode_mutation: float,
    ) -> None:
        problem = search.problem
        self.search = search
        self.rng = search.rng
        self.problem = problem
        self.items = Items(problem, jobs=problem.permutation)
        self.flexible = tuple(i for i, task in enumerate(problem.tasks) if len(task.modes) > 1)
        self.objectives = len(problem.objectives)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mode_rate = min(1.0, mode_mutation / len(self.flexible)) if self.flexible else 0.0

    @property
    def movable(self) -> bool:
        """Whether more than one genome exists: a free sequence or a multi-mode task."""
        return not self.items.fixed or bool(self.flexible)

    def progress(self) -> float:
        """Return the fraction of the budget used, in ``[0, 1]``.

        The search's deterministic progress: the larger of the used fractions
        of the evaluation and iteration limits, or, when neither is set, of
        the time limit, so that runs with a deterministic budget progress
        deterministically.
        """
        return self.search.progress()

    # ------------------------------------------------------------ genomes

    def evaluate(self, sequence: Sequence[int], modes: Sequence[int]) -> Member:
        """Decode and score a genome. Counts as one evaluation."""
        candidate = self.search.assess(self.items.tasks(sequence), modes)
        return Member(
            tuple(sequence),
            tuple(modes),
            self.search.vector(candidate),
            candidate.score.violation,
        )

    def random_member(self) -> Member:
        """Evaluate a random precedence-feasible sequence with random modes."""
        return self.evaluate(
            self.items.random_sequence(self.rng), random_modes(self.problem, self.rng)
        )

    def offspring(self, first: Member, second: Member) -> Member:
        """Vary two parents into a child and evaluate it."""
        sequence, modes = self.vary(first, second)
        return self.evaluate(sequence, modes)

    def vary(self, first: Member, second: Member) -> tuple[list[int], list[int]]:
        """Return the sequence and modes of a child of two parents, as the module describes."""
        rng = self.rng
        sequence: list[int]
        modes: list[int]
        if rng.random() < self.crossover_rate:
            sequence = self.items.feasible(order_crossover(first.sequence, second.sequence, rng))
            modes = uniform_mode_crossover(first.modes, second.modes, rng)
        else:
            sequence, modes = list(first.sequence), list(first.modes)
        if not self.items.fixed and rng.random() < self.mutation_rate:
            sequence = self._move(sequence)
        if self.mode_rate > 0.0:
            modes = mutate_modes(self.problem, modes, rng, self.mode_rate)
        parents = (first.genome, second.genome)
        for _ in range(RETRIES):
            if (tuple(sequence), tuple(modes)) not in parents:
                break
            sequence, modes = self._perturb(sequence, modes)
        return sequence, modes

    def _move(self, sequence: Sequence[int]) -> list[int]:
        """Insert one item elsewhere or swap two items, then restore precedence."""
        operator = random_insert if self.rng.random() < 0.5 else random_swap
        return self.items.feasible(operator(sequence, self.rng))

    def _perturb(self, sequence: list[int], modes: list[int]) -> tuple[list[int], list[int]]:
        """Apply one sequence move or give one multi-mode task another mode."""
        rng = self.rng
        if not self.flexible or (not self.items.fixed and rng.random() < 0.5):
            return self._move(sequence), modes
        task = self.flexible[rng.randrange(len(self.flexible))]
        choice = rng.randrange(len(self.problem.tasks[task].modes) - 1)
        changed = list(modes)
        changed[task] = choice if choice < modes[task] else choice + 1
        return sequence, changed

    # --------------------------------------------------------------- loop

    def run(self, selection: Selection) -> None:
        """Evolve until a stop criterion fires.

        A problem that admits a single genome is evaluated once and the run
        completes.
        """
        if not self.movable:
            self.random_member()
            return
        population = selection.prepare([self.random_member() for _ in range(selection.size)])
        while True:
            self.search.next_iteration()
            population = selection.generation(population)


class Selection(ABC):
    """The mating and environmental selection of one method during one run.

    Args:
        evolution: The run.
        size: Population size the method maintains.

    Attributes:
        evolution: The run.
        rng: The run's random generator.
        size: Population size.
    """

    def __init__(self, evolution: Evolution, size: int) -> None:
        self.evolution = evolution
        self.rng = evolution.rng
        self.size = size

    def prepare(self, population: list[Member]) -> list[Member]:
        """Return the evaluated initial population ready for mating. Unchanged by default."""
        return population

    def parents(self, population: Sequence[Member], index: int) -> tuple[Member, Member]:
        """Return the parents of the ``index``-th child of a generation.

        By default two distinct members drawn uniformly at random.
        """
        del index
        if len(population) < 2:
            return population[0], population[0]
        first, second = self.rng.sample(range(len(population)), 2)
        return population[first], population[second]

    @abstractmethod
    def generation(self, population: list[Member]) -> list[Member]:
        """Breed one generation and return the population that survives it."""


class GenerationalSelection(Selection):
    """A method that breeds a whole generation, then selects among parents and children."""

    def generation(self, population: list[Member]) -> list[Member]:
        """Breed ``size`` children, then apply :meth:`survive`."""
        evolution = self.evolution
        offspring = [
            evolution.offspring(*self.parents(population, index)) for index in range(self.size)
        ]
        return self.survive(population, offspring)

    @abstractmethod
    def survive(self, population: list[Member], offspring: list[Member]) -> list[Member]:
        """Return the next population, chosen from the parents and their children."""


class SteadyStateSelection(Selection):
    """A method that lets each child compete for a place as soon as it is born."""

    def generation(self, population: list[Member]) -> list[Member]:
        """Breed ``size`` children one at a time, applying :meth:`admit` after each."""
        evolution = self.evolution
        for index in range(self.size):
            child = evolution.offspring(*self.parents(population, index))
            population = self.admit(population, child, index)
        return population

    @abstractmethod
    def admit(self, population: list[Member], child: Member, index: int) -> list[Member]:
        """Return the population after ``child``, bred for slot ``index``, competed for a place."""


def tournament(population: Sequence[Member], rng: random.Random) -> Member:
    """Return the member of smaller fitness key among two drawn uniformly with replacement."""
    first = population[rng.randrange(len(population))]
    second = population[rng.randrange(len(population))]
    return second if second.fitness < first.fitness else first


def keep(members: Sequence[Member], survivors: Survivors) -> list[Member]:
    """Return the selected members with their mating keys, prefixed by a zero overrun."""
    kept: list[Member] = []
    for index, key in zip(survivors.indices, survivors.keys, strict=True):
        member = members[index]
        member.fitness = (0.0, *key)
        kept.append(member)
    return kept


def feasible_first(
    pool: Sequence[Member],
    size: int,
    select: Callable[[list[Member], int], list[Member]],
) -> list[Member]:
    """Apply constrained domination to an environmental selection.

    ``select(feasible, count)`` chooses ``count`` of the feasible members, at
    most their number. When fewer than ``size`` members are feasible, the
    infeasible members of smallest overrun fill the remaining places.
    """
    feasible = [member for member in pool if member.feasible]
    kept = select(feasible, min(size, len(feasible))) if feasible else []
    if len(kept) < size:
        infeasible = sorted(
            (member for member in pool if not member.feasible),
            key=lambda member: member.violation,
        )
        for member in infeasible[: size - len(kept)]:
            member.fitness = (member.violation,)
            kept.append(member)
    return kept


@dataclass(frozen=True)
class MultiObjectiveEvolution(ABC):
    """Parameters and run of the multi-objective evolutionary algorithms.

    Subclasses provide :attr:`info` and :meth:`selection`.

    Attributes:
        population_size: Members of the population, at least 2. Methods that
            decompose the objective space with the Das and Dennis lattice use
            the largest lattice with at most this many points.
        crossover_rate: Probability that a child is a crossover of its parents.
        mutation_rate: Probability that a child's sequence receives one insert
            or swap move.
        mode_mutation: Expected number of multi-mode tasks whose mode a child
            redraws.
    """

    population_size: int = 40
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    mode_mutation: float = 1.0

    info: ClassVar[SolverInfo]

    def __post_init__(self) -> None:
        if isinstance(self.population_size, bool) or self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        for name in ("crossover_rate", "mutation_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not self.mode_mutation >= 0.0:
            raise ValueError("mode_mutation must be non-negative")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Evolve a population until a stop criterion fires and return the archive as the front."""
        search = Search(problem, stop, seed, self.info)
        with search:
            evolution = Evolution(
                search, self.crossover_rate, self.mutation_rate, self.mode_mutation
            )
            selection = self.selection(evolution)
            search.extras["population"] = selection.size
            evolution.run(selection)
        return search.result()

    @abstractmethod
    def selection(self, evolution: Evolution) -> Selection:
        """Return the method's selection for one run."""
