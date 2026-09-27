"""Solvers that learn how to build schedules while they solve.

Each learner starts from scratch on the problem it is given, spends the run's
budget on learning, and returns the best schedule it evaluated. What it learns
is a way of building schedules, which is itself evaluated as it improves:

- :mod:`~dispatchatlas.solvers.learning.temporal_difference`:
  ``q-learning-dispatching`` and ``sarsa-dispatching``, tabular agents that
  learn which of four dispatching rules picks the next task in each state of
  the construction.
- :mod:`~dispatchatlas.solvers.learning.policy_gradient`:
  ``reinforce-priority-policy``, a softmax policy over linear priorities of
  task features, trained by the REINFORCE policy gradient.
- :mod:`~dispatchatlas.solvers.learning.hyper_heuristic`:
  ``gp-hyper-heuristic``, genetic programming of priority-rule expression
  trees (:mod:`~dispatchatlas.solvers.learning.trees`) with a surrogate that
  chooses which bred rules to evaluate.

All of them use only the standard library and NumPy.
"""

from __future__ import annotations

from dispatchatlas.solvers.learning import temporal_difference
from dispatchatlas.solvers.learning.hyper_heuristic import GpHyperHeuristic
from dispatchatlas.solvers.learning.policy_gradient import ReinforcePriorityPolicy
from dispatchatlas.solvers.learning.temporal_difference import TemporalDifferenceDispatching
from dispatchatlas.solvers.registry import SolverRegistry

__all__ = [
    "GpHyperHeuristic",
    "ReinforcePriorityPolicy",
    "TemporalDifferenceDispatching",
    "register",
]


def register(registry: SolverRegistry) -> None:
    """Add the learning solvers and the genetic-programming hyper-heuristic to ``registry``."""
    temporal_difference.register(registry)
    registry.register(ReinforcePriorityPolicy)
    registry.register(GpHyperHeuristic)
