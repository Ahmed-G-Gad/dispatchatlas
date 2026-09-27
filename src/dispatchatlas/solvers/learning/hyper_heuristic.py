"""A genetic-programming hyper-heuristic that evolves priority rules.

A hyper-heuristic searches a space of heuristics instead of a space of
schedules (Burke et al. 2013). This one evolves priority rules, the expression
trees of :mod:`~dispatchatlas.solvers.learning.trees`, by genetic programming
(Koza 1992), and scores each rule by the schedule it builds on the problem:

- **Decoding.** A rule gives every task a priority; a priority that is not a
  number counts as minus infinity. The list schedule takes, at each step, the
  eligible task of highest priority (:func:`pick`): priorities within
  :data:`TIE_TOLERANCE` of the highest tie, and ties go to the shortest task,
  then the lowest index. The schedule builder places each task at its earliest
  start in its earliest-finishing mode. Every rule applied to the problem is
  one evaluation.
- **Initialisation.** ``population`` rules by ramped half-and-half.
- **Breeding.** A generation breeds ``offspring_multiplier * population``
  children, each from winners of tournaments of ``tournament_size`` members:
  subtree crossover of two winners with probability ``crossover_rate``,
  subtree mutation of one with probability ``mutation_rate``, otherwise a copy
  of one. A child deeper than ``max_depth`` is replaced by a new ramped
  half-and-half tree.
- **Surrogate screening** (Hildebrandt and Branke 2015). A rule's phenotypic
  characterisation lists its decision in each of ``situations`` decision
  situations: sets of 2 to ``situation_size`` tasks, drawn uniformly once at the
  start of the run. The decision is the rank, by shortest duration then lowest
  index, of the task that the rule picks in the situation. Rules with equal
  characterisations count as equivalent, so a child whose characterisation
  occurs in the population or in an earlier child is dropped. The others are
  ranked by their estimated score, the mean score of the ``neighbours`` members
  of the current population whose characterisations are nearest in Euclidean
  distance, and evaluated in that order until, together with the ``elites``
  best members carried over unchanged, they fill the next population. New
  random rules fill any remaining places. Zhang et al. (2021) screen offspring
  of scheduling rules with this surrogate as well.

One generation is one iteration. The result extras hold the rule of the best
schedule as text.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.learning.trees import (
    INITIAL_DEPTHS,
    Tree,
    crossover,
    depth,
    evaluate,
    mutate,
    ramped_half_and_half,
    terminal_columns,
    text,
)
from dispatchatlas.solvers.priorities import min_durations
from dispatchatlas.solvers.search import Search

if TYPE_CHECKING:
    from dispatchatlas.solvers.learning.trees import Column

__all__ = [
    "TIE_TOLERANCE",
    "GpHyperHeuristic",
    "Member",
    "Rule",
    "characterise",
    "decision_situations",
    "list_order",
    "pick",
]

TIE_TOLERANCE = 1e-9
"""Priorities this close to the highest one count as ties."""


def pick(candidates: Sequence[int], priorities: Sequence[float], durations: Sequence[float]) -> int:
    """Return the candidate of highest priority.

    Candidates within :data:`TIE_TOLERANCE` of the highest priority tie, and
    ties go to the shortest duration, then the lowest index.
    """
    top = max(priorities[task] for task in candidates)
    return min(
        (task for task in candidates if priorities[task] >= top - TIE_TOLERANCE),
        key=lambda task: (durations[task], task),
    )


def list_order(
    problem: Problem, priorities: Sequence[float], durations: Sequence[float]
) -> list[int]:
    """Return the list-scheduling order of a rule's priorities.

    At each step the eligible task (all its predecessors already listed) that
    :func:`pick` chooses comes next.
    """
    waiting = [len(before) for before in problem.predecessors]
    eligible = [task for task, count in enumerate(waiting) if count == 0]
    order: list[int] = []
    while eligible:
        chosen = pick(eligible, priorities, durations)
        order.append(chosen)
        eligible.remove(chosen)
        for following in problem.successors[chosen]:
            waiting[following] -= 1
            if waiting[following] == 0:
                eligible.append(following)
    return order


def decision_situations(
    tasks: int, count: int, largest: int, rng: random.Random
) -> list[tuple[int, ...]]:
    """Draw the decision situations of the phenotypic characterisation.

    Each situation is a sorted set of distinct tasks whose size is drawn
    uniformly from 2 to ``min(largest, tasks)``. A single-task problem has
    only the situation made of its task.

    Args:
        tasks: Number of tasks of the problem.
        count: Number of situations.
        largest: Largest number of tasks in a situation, at least 2.
        rng: Random generator.
    """
    if tasks < 2:
        return [(0,)] * count
    high = min(largest, tasks)
    return [tuple(sorted(rng.sample(range(tasks), rng.randint(2, high)))) for _ in range(count)]


def characterise(
    priorities: Sequence[float],
    situations: Sequence[tuple[int, ...]],
    durations: Sequence[float],
) -> tuple[int, ...]:
    """Return the phenotypic characterisation of a rule given by its priorities.

    Entry ``i`` is the rank, by shortest duration then lowest index, of the
    task that :func:`pick` chooses in situation ``i``.
    """
    decisions: list[int] = []
    for situation in situations:
        chosen = pick(situation, priorities, durations)
        reference = sorted(situation, key=lambda task: (durations[task], task))
        decisions.append(reference.index(chosen))
    return tuple(decisions)


@dataclass(frozen=True, slots=True)
class Rule:
    """A rule with its priorities and phenotypic characterisation.

    Attributes:
        tree: The expression tree.
        priorities: Priority of every task.
        characterisation: Rank decisions in the run's decision situations.
    """

    tree: Tree
    priorities: tuple[float, ...]
    characterisation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Member:
    """An evaluated rule of the population.

    Attributes:
        rule: The rule.
        score: Score of the schedule it builds.
    """

    rule: Rule
    score: Score


@dataclass(frozen=True)
class GpHyperHeuristic:
    """Genetic programming of priority rules with surrogate screening of the offspring.

    A population of priority-rule trees evolves by tournament selection,
    subtree crossover, and subtree mutation. Every generation breeds a surplus
    of children, and a nearest-neighbour surrogate over the rules' decisions in
    sample situations chooses which of them to evaluate. The module docstring
    gives the details.

    Attributes:
        population: Rules per generation, at least 1.
        max_depth: Largest depth of a bred tree, at least the largest initial
            depth, 3.
        crossover_rate: Probability that a child is a subtree crossover.
        mutation_rate: Probability that a child is a subtree mutation. With
            ``crossover_rate`` it sums to at most 1; the rest is copying.
        tournament_size: Members drawn, with replacement, per tournament.
        elites: Best members carried into the next generation, fewer than
            ``population``.
        offspring_multiplier: Children bred per generation, as a multiple of
            ``population``.
        neighbours: Nearest population members whose mean score estimates a
            child's score.
        situations: Decision situations of the phenotypic characterisation.
        situation_size: Largest number of tasks in a decision situation, at
            least 2.
    """

    population: int = 24
    max_depth: int = 6
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    tournament_size: int = 5
    elites: int = 2
    offspring_multiplier: int = 2
    neighbours: int = 1
    situations: int = 12
    situation_size: int = 7

    info: ClassVar[SolverInfo] = SolverInfo(
        id="gp-hyper-heuristic",
        name="Genetic programming hyper-heuristic",
        family="hyper-heuristic",
        summary="Evolves priority-rule expression trees over task attributes by tournament "
        "selection, subtree crossover, and subtree mutation, evaluating only the bred "
        "rules that a nearest-neighbour surrogate over their decisions in sample "
        "situations ranks best, each by the list schedule it builds.",
        references=(
            "koza1992genetic",
            "hildebrandt2015using",
            "zhang2021surrogateassisted",
            "zaki2024cooperative",
            "burke2013hyper",
        ),
    )

    def __post_init__(self) -> None:
        if self.population < 1:
            raise ValueError("population must be at least 1")
        if self.max_depth < max(INITIAL_DEPTHS):
            raise ValueError(f"max_depth must be at least {max(INITIAL_DEPTHS)}")
        for name in ("crossover_rate", "mutation_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.crossover_rate + self.mutation_rate > 1.0:
            raise ValueError("crossover_rate and mutation_rate must sum to at most 1")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")
        if not 0 <= self.elites < self.population:
            raise ValueError("elites must lie in [0, population)")
        for name in ("offspring_multiplier", "neighbours", "situations"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.situation_size < 2:
            raise ValueError("situation_size must be at least 2")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Evolve rules within the stop criteria and return the best schedule built.

        The result extras hold ``rule``, the rule of the best schedule as text.
        """
        search = Search(problem, stop, seed, self.info)
        evolution = _Evolution(search, self)
        with search:
            evolution.run()
        best = evolution.best
        return search.result(extras={"rule": "" if best is None else text(best.rule.tree)})


class _Evolution:
    """The populations, the surrogate, and the evaluations of one run."""

    def __init__(self, search: Search, settings: GpHyperHeuristic) -> None:
        problem = search.problem
        self.search = search
        self.settings = settings
        self.rng = search.rng
        self.columns: dict[str, Column] = terminal_columns(problem)
        self.durations = min_durations(problem)
        self.situations = decision_situations(
            problem.size, settings.situations, settings.situation_size, self.rng
        )
        self.best: Member | None = None

    def run(self) -> None:
        """Evolve generations until the run raises ``Exhausted``."""
        population = [
            self.evaluate(self.rule(ramped_half_and_half(self.rng)))
            for _ in range(self.settings.population)
        ]
        while True:
            self.search.next_iteration()
            population = self.next_population(population, self.breed(population))

    def rule(self, tree: Tree) -> Rule:
        """Compute a tree's priorities and characterisation."""
        values = evaluate(tree, self.columns)
        priorities = tuple(np.where(np.isnan(values), -math.inf, values).tolist())
        return Rule(tree, priorities, characterise(priorities, self.situations, self.durations))

    def evaluate(self, rule: Rule) -> Member:
        """Build and score the list schedule of a rule: one evaluation."""
        order = list_order(self.search.problem, rule.priorities, self.durations)
        member = Member(rule, self.search.evaluate(order))
        if self.best is None or member.score < self.best.score:
            self.best = member
        return member

    def breed(self, population: Sequence[Member]) -> list[Rule]:
        """Breed the generation's children, without characterisations already present."""
        children: list[Rule] = []
        seen = {member.rule.characterisation for member in population}
        for _ in range(self.settings.offspring_multiplier * self.settings.population):
            self.search.check()
            child = self.rule(self.child(population))
            if child.characterisation not in seen:
                seen.add(child.characterisation)
                children.append(child)
        return children

    def child(self, population: Sequence[Member]) -> Tree:
        """Breed one child tree by crossover, mutation, or copying."""
        settings = self.settings
        rng = self.rng
        roll = rng.random()
        if roll < settings.crossover_rate:
            parent = self.tournament(population)
            tree = crossover(parent, self.tournament(population), rng)
        elif roll < settings.crossover_rate + settings.mutation_rate:
            tree = mutate(self.tournament(population), rng)
        else:
            tree = self.tournament(population)
        if depth(tree) > settings.max_depth:
            return ramped_half_and_half(rng)
        return tree

    def tournament(self, population: Sequence[Member]) -> Tree:
        """Return the tree of the best of ``tournament_size`` members drawn with replacement."""
        rng = self.rng
        winner = population[rng.randrange(len(population))]
        for _ in range(self.settings.tournament_size - 1):
            rival = population[rng.randrange(len(population))]
            if rival.score < winner.score:
                winner = rival
        return winner.rule.tree

    def estimate(self, rule: Rule, population: Sequence[Member]) -> Score:
        """Return the mean score of the members nearest to ``rule`` in characterisation."""
        nearest = sorted(
            population,
            key=lambda member: math.dist(member.rule.characterisation, rule.characterisation),
        )[: self.settings.neighbours]
        return Score(
            math.fsum(member.score.violation for member in nearest) / len(nearest),
            math.fsum(member.score.value for member in nearest) / len(nearest),
        )

    def next_population(
        self, population: Sequence[Member], children: Sequence[Rule]
    ) -> list[Member]:
        """Carry the elites over and evaluate the children in surrogate order until full."""
        size = self.settings.population
        members = sorted(population, key=lambda member: member.score)[: self.settings.elites]
        for child in sorted(children, key=lambda item: self.estimate(item, population)):
            if len(members) >= size:
                break
            members.append(self.evaluate(child))
        while len(members) < size:
            members.append(self.evaluate(self.rule(ramped_half_and_half(self.rng))))
        return members
