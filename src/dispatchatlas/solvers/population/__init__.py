"""Population metaheuristics: methods that evolve many solutions at once.

Two representations are used:

- **Sequences.** The genetic, memetic, scatter-search, and ant-colony methods
  work on the precedence-feasible item sequences and mode settings of
  :class:`~dispatchatlas.solvers.trajectory.space.SequenceSpace`, the space the
  trajectory methods move through, and share its moves and local searches.
- **Random keys.** The biased random-key genetic algorithm and the univariate
  marginal distribution algorithm search real vectors decoded by the
  random-key rule, through
  :class:`~dispatchatlas.solvers.continuous.encoded.EncodedSolver`, like the
  continuous optimisers of :mod:`dispatchatlas.solvers.continuous`.

Solver ids:

- ``genetic-algorithm``, :class:`GeneticAlgorithm`: tournament selection, order
  crossover, and insert, swap, or mode mutation (Holland 1992; Davis 1985);
- ``memetic-algorithm``, :class:`MemeticAlgorithm`: the genetic algorithm with
  an insertion local search on every child (Moscato 1989);
- ``brkga``, :class:`Brkga`: elite, mutants, and biased crossover of random-key
  chromosomes (Gonçalves and Resende 2011);
- ``umda``, :class:`Umda`: an independent normal model per key, estimated from
  the best members (Mühlenbein and Paaß 1996; Larrañaga and Lozano 2002);
- ``scatter-search``, :class:`ScatterSearch`: a reference set of good and
  diverse solutions, combined pairwise and improved (Glover 1998; Martí,
  Laguna and Glover 2006);
- ``ant-colony-optimization``, :class:`AntColonyOptimization`: the MAX-MIN Ant
  System with pheromone on (task, position) pairs (Stützle and Hoos 2000).
"""

from __future__ import annotations

from dispatchatlas.solvers.population.brkga import Brkga
from dispatchatlas.solvers.population.colony import AntColonyOptimization, Trails
from dispatchatlas.solvers.population.genetic import GeneticAlgorithm, MemeticAlgorithm
from dispatchatlas.solvers.population.scatter import ScatterSearch
from dispatchatlas.solvers.population.umda import Umda
from dispatchatlas.solvers.registry import SolverRegistry

__all__ = [
    "AntColonyOptimization",
    "Brkga",
    "GeneticAlgorithm",
    "MemeticAlgorithm",
    "ScatterSearch",
    "Trails",
    "Umda",
    "register",
]


def register(registry: SolverRegistry) -> None:
    """Add the population metaheuristics to ``registry``."""
    for factory in (
        GeneticAlgorithm,
        MemeticAlgorithm,
        Brkga,
        Umda,
        ScatterSearch,
        AntColonyOptimization,
    ):
        registry.register(factory)
