"""Multi-objective evolutionary algorithms.

Eight published methods share one evolutionary loop and differ only in their
selection:

- ``nsga-ii``, :class:`Nsga2`: non-dominated fronts and crowding distance
  (Deb, Pratap, Agarwal and Meyarivan 2002);
- ``nsga-iii``, :class:`Nsga3`: non-dominated fronts and niching around
  reference points (Deb and Jain 2014);
- ``moea-d``, :class:`MoeaD`: Tchebycheff subproblems with neighbourhood
  replacement (Zhang and Li 2007);
- ``spea2``, :class:`Spea2`: strength fitness, density, and archive truncation
  (Zitzler, Laumanns and Thiele 2001);
- ``ibea``, :class:`Ibea`: additive epsilon indicator fitness (Zitzler and
  Künzli 2004);
- ``sms-emoa``, :class:`SmsEmoa`: steady-state removal of the smallest
  hypervolume contributor (Beume, Naujoks and Emmerich 2007);
- ``rvea``, :class:`Rvea`: reference vectors and the angle-penalised distance
  (Cheng, Jin, Olhofer and Sendhoff 2016);
- ``age-moea-ii``, :class:`AgeMoea2`: survival by the estimated geometry of the
  first front (Panichella 2022).

**Genomes and variation.** A genome is a precedence-feasible item sequence
(tasks, or whole jobs in permutation problems) and an explicit mode for every
task. Children come from the order crossover of Davis (1985) and a uniform
crossover of the modes, followed by an insert or swap move and a redraw of
some modes; sequences are repaired to respect precedence. Every child is
decoded by the shared schedule builder, counts against the budget, and is
scored on every objective the problem declares.

**Deadlines.** Hard deadlines are handled by the constrained domination of
Deb, Pratap, Agarwal and Meyarivan (2002): a schedule that meets every
deadline is better than one that does not, and of two that do not, the one
with the smaller total overrun is better. Each method applies its own rule to
the feasible members and fills the rest of the population with the members of
smallest overrun.

**Result.** The front of a result is the run's archive: the non-dominated
feasible schedules among all evaluated ones, over the problem's objectives,
with values equal to those :func:`~dispatchatlas.core.objectives.evaluate`
computes. The result's schedule is the best one for the primary objective.

The pure selection rules live in
:mod:`~dispatchatlas.solvers.multiobjective.selection`.
"""

from __future__ import annotations

from dispatchatlas.solvers.multiobjective.age_moea2 import AgeMoea2
from dispatchatlas.solvers.multiobjective.ibea import Ibea
from dispatchatlas.solvers.multiobjective.moead import MoeaD
from dispatchatlas.solvers.multiobjective.nsga2 import Nsga2
from dispatchatlas.solvers.multiobjective.nsga3 import Nsga3
from dispatchatlas.solvers.multiobjective.rvea import Rvea
from dispatchatlas.solvers.multiobjective.sms_emoa import SmsEmoa
from dispatchatlas.solvers.multiobjective.spea2 import Spea2
from dispatchatlas.solvers.registry import SolverRegistry

__all__ = [
    "AgeMoea2",
    "Ibea",
    "MoeaD",
    "Nsga2",
    "Nsga3",
    "Rvea",
    "SmsEmoa",
    "Spea2",
    "register",
]


def register(registry: SolverRegistry) -> None:
    """Add the multi-objective evolutionary algorithms to ``registry``."""
    for factory in (Nsga2, Nsga3, MoeaD, Spea2, Ibea, SmsEmoa, Rvea, AgeMoea2):
        registry.register(factory)
