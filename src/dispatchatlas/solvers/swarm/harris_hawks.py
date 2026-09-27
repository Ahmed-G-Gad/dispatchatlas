"""Harris Hawks Optimization (Heidari, Mirjalili, Faris, Aljarah, Mafarja and Chen 2019).

Each iteration draws, per member ``x``, a coefficient ``E = 2 E0 (1 - t/T)``
with ``E0`` uniform in ``[-1, 1]`` (Eq. 3), so its range shrinks from
``[-2, 2]`` to 0 over the budget, and a factor ``J = 2 (1 - u)`` for a uniform
``u``. With ``b`` the best vector found so far, ``m`` the mean of the
population and ``y`` a uniformly drawn member, all taken at the start of the
iteration, and fresh uniform scalars ``q``, ``r`` and ``r1`` to ``r4``:

- ``|E| >= 1`` (Eq. 1): ``y - r1 |y - 2 r2 x|`` when ``q >= 0.5``, otherwise
  ``(b - m) - r3 (LB + r4 (UB - LB))``;
- ``|E| < 1`` and ``r >= 0.5``: ``(b - x) - E |J b - x|`` when ``|E| >= 0.5``
  (Eqs. 4 and 5), otherwise ``b - E |b - x|`` (Eq. 6);
- ``|E| < 1`` and ``r < 0.5``: two greedy trials. The first is
  ``Y = b - E |J b - c|`` with ``c = x`` when ``|E| >= 0.5`` (Eq. 7) and
  ``c = m`` otherwise (Eq. 12); the second is ``Z = Y + s * LF``, a uniform
  vector ``s`` times Lévy steps with ``beta = 1.5`` scaled by 0.01 (Eqs. 8, 9,
  and 13). The member moves to ``Y`` if it beats the member, otherwise to
  ``Z`` if that does, otherwise it stays (Eqs. 10 and 11).

The first two cases replace the member whatever its new score. As ``E``
shrinks, the population passes from the first case to the others. Every trial
vector is one evaluation, so an iteration costs between one and two
evaluations per member, and a member that keeps its position is not evaluated
again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from dispatchatlas.core.solver import SolverInfo
from dispatchatlas.solvers.swarm._engine import (
    Swarm,
    SwarmMethod,
    Vector,
    centroid,
    levy,
    method_info,
)

__all__ = ["HarrisHawksOptimization"]

LEVY_BETA = 1.5
"""Index ``beta`` of the Lévy steps (Eq. 9)."""

LEVY_SCALE = 0.01
"""Factor of the Lévy steps (Eq. 9)."""


@dataclass(frozen=True)
class HarrisHawksOptimization(SwarmMethod):
    """Harris Hawks Optimization on priority vectors."""

    info: ClassVar[SolverInfo] = method_info(
        "harris-hawks-optimization",
        "Harris Hawks Optimization",
        "Moves each vector by a rule chosen with a random coefficient whose range shrinks "
        "over the budget: jumps relative to random members and the population mean while "
        "it is large, contractions towards the best vector and greedy trials with Lévy "
        "perturbations while it is small.",
        "heidari2019harris",
        "mantegna1994fast",
    )

    def step(self, swarm: Swarm) -> None:
        """Move every vector by the rule its coefficient ``E`` selects."""
        rng = swarm.rng
        envelope = 2.0 * (1.0 - swarm.progress)
        best = swarm.best
        members = swarm.positions.copy()
        mean = centroid(members)
        for index in range(swarm.size):
            position = members[index]
            e = envelope * (2.0 * rng.random() - 1.0)
            stretch = 2.0 * (1.0 - rng.random())
            if abs(e) >= 1.0:
                if rng.random() >= 0.5:
                    other = members[rng.integers(swarm.size)]
                    r1, r2 = rng.random(2)
                    target = other - r1 * np.abs(other - 2.0 * r2 * position)
                else:
                    r3, r4 = rng.random(2)
                    target = (best - mean) - r3 * r4  # LB + r4 (UB - LB) is r4 in [0, 1]
                swarm.replace(index, target)
            elif rng.random() >= 0.5:
                if abs(e) >= 0.5:
                    target = (best - position) - e * np.abs(stretch * best - position)
                else:
                    target = best - e * np.abs(best - position)
                swarm.replace(index, target)
            else:
                anchor = position if abs(e) >= 0.5 else mean
                self._trials(swarm, index, best - e * np.abs(stretch * best - anchor))

    @staticmethod
    def _trials(swarm: Swarm, index: int, trial: Vector) -> None:
        """Move member ``index`` to ``trial`` or its Lévy perturbation if either is better."""
        vector, score = swarm.evaluate(trial)
        if score < swarm.scores[index]:
            swarm.accept(index, vector, score)
            return
        flight = LEVY_SCALE * levy(swarm.rng, swarm.dimension, LEVY_BETA)
        vector, score = swarm.evaluate(trial + swarm.rng.random(swarm.dimension) * flight)
        if score < swarm.scores[index]:
            swarm.accept(index, vector, score)
