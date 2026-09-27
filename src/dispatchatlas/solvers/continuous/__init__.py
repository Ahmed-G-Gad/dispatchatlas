"""Continuous optimisers and their use on scheduling problems.

Each optimiser searches real vectors and needs only a box, a fitness it can
compare, and a budget (see :mod:`~dispatchatlas.solvers.continuous.core`). The same
object minimises a plain function through :func:`minimize` and solves a
scheduling problem as a registered solver, through the random-key encoding of
:mod:`~dispatchatlas.solvers.continuous.encoded`:

- ``differential-evolution``, :class:`DifferentialEvolution`: DE/rand/1/bin
  (Storn and Price 1997);
- ``particle-swarm-optimization``, :class:`ParticleSwarmOptimization`:
  global-best particle swarm with a decreasing inertia weight (Kennedy and
  Eberhart 1995; Shi and Eberhart 1998);
- ``clpso``, :class:`Clpso`: comprehensive learning particle swarm (Liang et
  al. 2006);
- ``lshade``, :class:`LShade`: success-history adaptive differential evolution
  with linear population size reduction (Tanabe and Fukunaga 2014);
- ``cma-es``, :class:`CmaEs`: covariance matrix adaptation evolution strategy
  (Hansen and Ostermeier 2001).
"""

from __future__ import annotations

from dispatchatlas.solvers.continuous.clpso import Clpso
from dispatchatlas.solvers.continuous.cmaes import CmaEs
from dispatchatlas.solvers.continuous.core import (
    Box,
    FunctionRun,
    Minimum,
    Optimizer,
    Ordered,
    Run,
    minimize,
)
from dispatchatlas.solvers.continuous.de import DifferentialEvolution
from dispatchatlas.solvers.continuous.encoded import EncodedSolver, ScheduleRun, score_improvement
from dispatchatlas.solvers.continuous.lshade import LShade
from dispatchatlas.solvers.continuous.pso import ParticleSwarmOptimization
from dispatchatlas.solvers.registry import SolverRegistry

__all__ = [
    "Box",
    "Clpso",
    "CmaEs",
    "DifferentialEvolution",
    "EncodedSolver",
    "FunctionRun",
    "LShade",
    "Minimum",
    "Optimizer",
    "Ordered",
    "ParticleSwarmOptimization",
    "Run",
    "ScheduleRun",
    "minimize",
    "register",
    "score_improvement",
]


def register(registry: SolverRegistry) -> None:
    """Add the continuous optimisers to ``registry``."""
    for factory in (DifferentialEvolution, ParticleSwarmOptimization, Clpso, LShade, CmaEs):
        registry.register(factory)
