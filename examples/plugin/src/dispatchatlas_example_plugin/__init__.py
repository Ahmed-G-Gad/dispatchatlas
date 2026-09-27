"""An example solver plugin for DispatchAtlas: insertion descent with restarts.

Installing this package makes the solver ``insertion-descent`` available
wherever DispatchAtlas looks solvers up: ``dispatchatlas solvers list``,
:func:`dispatchatlas.solvers.solve`, and campaign files. The package declares
one entry point in its ``pyproject.toml``::

    [project.entry-points."dispatchatlas.solvers"]
    example = "dispatchatlas_example_plugin:register"

Install it next to DispatchAtlas and use it by id::

    pip install ./examples/plugin
    dispatchatlas solvers show insertion-descent
    dispatchatlas solve problem.json --solver insertion-descent --max-evaluations 5000

The solver follows the contract of the built-in ones. Its parameters are
dataclass fields checked in ``__post_init__``, its metadata is a class-level
:class:`~dispatchatlas.core.SolverInfo`, and its loop runs inside a
:class:`~dispatchatlas.solvers.Search`, which makes it honour every stop
criterion, count evaluations exactly, and draw randomness only from the seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core import Problem, SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers import Search, SolverRegistry
from dispatchatlas.solvers.operators import random_insert, random_topological_order

__all__ = ["InsertionDescent", "register"]


@dataclass(frozen=True)
class InsertionDescent:
    """Improve one task order by random insertion moves, restarting when stuck.

    Each iteration moves one task of the current order to another position and
    keeps the new order when its schedule scores no worse. After
    ``restart_after`` rejected moves in a row, the walk restarts from a new
    random precedence-feasible order. The best schedule over all restarts is
    returned.

    Attributes:
        restart_after: Consecutive rejected moves after which the walk
            restarts.
    """

    restart_after: int = 100

    info: ClassVar[SolverInfo] = SolverInfo(
        id="insertion-descent",
        name="Insertion descent with restarts",
        family="local-search",
        summary=(
            "Moves one task of the current order to a random position, keeps the move when "
            "the schedule scores no worse, and restarts from a random order after a run of "
            "rejected moves."
        ),
        references=("kolisch1996serial",),
    )

    def __post_init__(self) -> None:
        value = self.restart_after
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"restart_after must be a positive integer, got {value!r}")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Run the descent within ``stop`` using randomness derived from ``seed``."""
        search = Search(problem, stop, seed, self.info)
        with search:
            order = random_topological_order(problem, search.rng)
            score = search.evaluate(order)
            rejected = 0
            while True:
                search.next_iteration()
                if rejected >= self.restart_after:
                    order = random_topological_order(problem, search.rng)
                    score = search.evaluate(order)
                    rejected = 0
                    continue
                candidate = random_insert(order, search.rng)
                candidate_score = search.evaluate(candidate)
                if candidate_score <= score:
                    order, score, rejected = candidate, candidate_score, 0
                else:
                    rejected += 1
        return search.result()


def register(registry: SolverRegistry) -> None:
    """Add the plugin's solvers to ``registry``; DispatchAtlas calls it on discovery."""
    registry.register(InsertionDescent)
