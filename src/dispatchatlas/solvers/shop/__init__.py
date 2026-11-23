"""Solvers for the job shop.

A job shop is a problem whose tasks form chains, the jobs, and whose tasks each
have one mode that runs them on one machine, the shape ``"job-shop"`` of
:data:`~dispatchatlas.core.model.SHAPES`. A solver of this package names the
shape in :attr:`~dispatchatlas.core.solver.SolverInfo.shape`, so the search
raises :class:`~dispatchatlas.core.errors.UnsupportedProblemError` for a
problem without it, and a campaign excludes the pairing and records the
missing shape.

:func:`register` adds the solvers of the package to a registry.
"""

from __future__ import annotations

from collections.abc import Callable

from dispatchatlas.core.solver import Solver
from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.shop.annealing import ShopAnnealing
from dispatchatlas.solvers.shop.genetic import ShopSteadyStateGa
from dispatchatlas.solvers.shop.sampling import ShopRandomSampling
from dispatchatlas.solvers.shop.swarm import ShopParticleSwarm

__all__ = ["register"]

# The solver classes of the package, in the order they register.
_SOLVERS: tuple[Callable[..., Solver], ...] = (
    ShopRandomSampling,
    ShopAnnealing,
    ShopSteadyStateGa,
    ShopParticleSwarm,
)


def register(registry: SolverRegistry) -> None:
    """Add the job-shop solvers to ``registry``."""
    for factory in _SOLVERS:
        registry.register(factory)
