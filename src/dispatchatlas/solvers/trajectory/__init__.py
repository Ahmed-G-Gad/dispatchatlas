"""Trajectory metaheuristics: methods that improve one solution at a time.

All of them search item sequences and, for multi-mode tasks, explicit mode
choices, through the moves and the local search of
:class:`~dispatchatlas.solvers.trajectory.space.SequenceSpace`. They differ in
how they move and what they accept:

- :mod:`~dispatchatlas.solvers.trajectory.acceptance`: one random neighbour per
  iteration, accepted by a rule. ``hill-climbing``, ``simulated-annealing``,
  ``threshold-accepting``, ``great-deluge``, ``record-to-record-travel``, and
  ``late-acceptance-hill-climbing``.
- :mod:`~dispatchatlas.solvers.trajectory.tabu`: the best admissible neighbour of a
  sample, with a tabu list. ``tabu-search``.
- :mod:`~dispatchatlas.solvers.trajectory.iterated`: perturbation plus local search.
  ``iterated-local-search`` and ``variable-neighbourhood-search``.
- :mod:`~dispatchatlas.solvers.trajectory.greedy`: greedy construction plus local
  search. ``iterated-greedy`` and ``grasp``.
- :mod:`~dispatchatlas.solvers.trajectory.alns`: adaptive destroy and repair.
  ``alns``.
- :mod:`~dispatchatlas.solvers.trajectory.guided`: descent on an objective
  augmented with feature penalties. ``guided-local-search``.
- :mod:`~dispatchatlas.solvers.trajectory.hyper`: learned choice among low-level
  heuristics with late acceptance. ``selection-hyper-heuristic``.
- :mod:`~dispatchatlas.solvers.trajectory.mode_tabu`: tabu search over mode
  vectors alone. ``mode-assignment-tabu``.
"""

from __future__ import annotations

from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.trajectory.acceptance import (
    GreatDeluge,
    HillClimbing,
    LateAcceptanceHillClimbing,
    RecordToRecordTravel,
    SimulatedAnnealing,
    ThresholdAccepting,
)
from dispatchatlas.solvers.trajectory.alns import Alns
from dispatchatlas.solvers.trajectory.greedy import Grasp, IteratedGreedy
from dispatchatlas.solvers.trajectory.guided import GuidedLocalSearch
from dispatchatlas.solvers.trajectory.hyper import SelectionHyperHeuristic
from dispatchatlas.solvers.trajectory.iterated import (
    IteratedLocalSearch,
    VariableNeighbourhoodSearch,
)
from dispatchatlas.solvers.trajectory.mode_tabu import ModeAssignmentTabu
from dispatchatlas.solvers.trajectory.space import SequenceSpace, Solution
from dispatchatlas.solvers.trajectory.tabu import TabuSearch

__all__ = [
    "Alns",
    "Grasp",
    "GreatDeluge",
    "GuidedLocalSearch",
    "HillClimbing",
    "IteratedGreedy",
    "IteratedLocalSearch",
    "LateAcceptanceHillClimbing",
    "ModeAssignmentTabu",
    "RecordToRecordTravel",
    "SelectionHyperHeuristic",
    "SequenceSpace",
    "SimulatedAnnealing",
    "Solution",
    "TabuSearch",
    "ThresholdAccepting",
    "VariableNeighbourhoodSearch",
    "register",
]


def register(registry: SolverRegistry) -> None:
    """Add the trajectory metaheuristics to ``registry``."""
    for factory in (
        HillClimbing,
        SimulatedAnnealing,
        ThresholdAccepting,
        GreatDeluge,
        RecordToRecordTravel,
        LateAcceptanceHillClimbing,
        TabuSearch,
        IteratedLocalSearch,
        VariableNeighbourhoodSearch,
        IteratedGreedy,
        Grasp,
        Alns,
        GuidedLocalSearch,
        SelectionHyperHeuristic,
        ModeAssignmentTabu,
    ):
        registry.register(factory)
