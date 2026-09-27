"""Solving scheduling problems with methods that search real vectors.

:class:`EncodedSolver` is the base of every solver that searches a real vector
space: the continuous methods of this package and the random-key population
methods of :mod:`dispatchatlas.solvers.population`. A subclass implements
``optimize(run, rng)`` on any :class:`~dispatchatlas.solvers.continuous.core.Run`;
the base class supplies :meth:`EncodedSolver.solve`, which hands the method a
:class:`ScheduleRun`:

- vectors decode through :class:`~dispatchatlas.solvers.encoding.VectorEncoding`
  with the random-key rule of Bean (1994): tasks run in ascending order of their
  keys, as far as precedence allows; each multi-mode task either has one more
  key that selects its mode or runs in the mode the schedule builder finds
  earliest-finishing (see :attr:`EncodedSolver.modes`);
- every fitness call is one evaluation of the run's
  :class:`~dispatchatlas.solvers.search.Search`, so each method honours every
  stop criterion and reports exact counts;
- the search box is the unit cube, marked non-strict because every real vector
  decodes: methods designed for unbounded spaces leave it freely.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from dispatchatlas.core.model import Problem, Scalar
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.constructive import Items
from dispatchatlas.solvers.continuous.core import Box, Ordered, Run
from dispatchatlas.solvers.encoding import VectorEncoding
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.priorities import upward_ranks
from dispatchatlas.solvers.search import Search

__all__ = [
    "EncodedSolver",
    "ScheduleRun",
    "heuristic_seed",
    "score_improvement",
    "single_schedule",
]


def score_improvement(before: Score, after: Score) -> float:
    """Return how much better ``after`` is than ``before``, in the first component that differs.

    Scores compare lexicographically, so the improvement is measured in the
    leading component where they differ: the reduction of the deadline overrun
    when the overruns differ, otherwise the reduction of the objective. It is
    positive exactly when ``after < before``.
    """
    if after.violation != before.violation:
        return before.violation - after.violation
    return before.value - after.value


def heuristic_seed(search: Search, encoding: VectorEncoding) -> tuple[list[float], Score]:
    """Evaluate a heuristic schedule and return its vector and score.

    The schedule decodes the tasks in decreasing upward rank, the priority of
    HEFT (Topcuoglu, Hariri and Wu 2002), with each task in its
    earliest-finishing mode. It costs one evaluation. The returned vector,
    from :meth:`~dispatchatlas.solvers.encoding.VectorEncoding.encode`, decodes to
    the same order and modes, hence to the same schedule.
    """
    ranks = upward_ranks(search.problem)
    order = sorted(range(search.problem.size), key=lambda task: (-ranks[task], task))
    candidate = search.assess(order)
    return encoding.encode(order, candidate.decoded.modes), candidate.score


def single_schedule(encoding: VectorEncoding) -> bool:
    """Return whether every vector decodes to the same schedule.

    That is the case when precedence admits one task sequence, that of the
    jobs in permutation problems, and no key selects a mode.
    """
    if encoding.flexible:
        return False
    problem = encoding.problem
    return Items(problem, jobs=problem.permutation).fixed


class ScheduleRun:
    """A run whose fitness is the score of the schedule a vector decodes to.

    Args:
        search: The solver run that pays for every evaluation.
        encoding: The vector encoding of the problem.
        seeds: Starting points with their scores.
    """

    def __init__(
        self,
        search: Search,
        encoding: VectorEncoding,
        seeds: Sequence[tuple[Sequence[float], Score]] = (),
    ) -> None:
        self.search = search
        self.encoding = encoding
        self.box = Box.cube(encoding.dimension, 0.0, 1.0, strict=False)
        self.seeds = list(seeds)

    @property
    def progress(self) -> float:
        """Fraction of the tightest limit spent, in [0, 1]: the search's progress."""
        return self.search.progress()

    def evaluate(self, vector: Sequence[float]) -> Score:
        """Decode ``vector`` and return the score of its schedule."""
        order, modes = self.encoding.decode(vector)
        return self.search.evaluate(order, modes)

    def next_iteration(self) -> None:
        """Start a generation of the search."""
        self.search.next_iteration()

    def improvement(self, before: Score, after: Score) -> float:
        """Return :func:`score_improvement` of the two scores."""
        return score_improvement(before, after)

    def note(self, key: str, value: Scalar) -> None:
        """Record a fact in the result's extras."""
        self.search.extras[key] = value


@dataclass(frozen=True)
class EncodedSolver:
    """Base of the solvers that search real vectors decoded into schedules.

    Subclasses define their parameters as dataclass fields, their
    :class:`~dispatchatlas.core.solver.SolverInfo`, and :meth:`optimize`, the method
    itself, which works on any run: a plain function through
    :func:`~dispatchatlas.solvers.continuous.core.minimize`, or a scheduling problem
    through :meth:`solve`.

    When every vector decodes to the same schedule (see
    :func:`single_schedule`), :meth:`solve` evaluates that schedule and
    completes.

    Attributes:
        seeded: Start from a heuristic schedule (see :func:`heuristic_seed`),
            which joins the initial population, or centres the first
            distribution of CMA-ES.
        modes: Where the modes of multi-mode tasks come from. ``"keys"``: one
            more key per multi-mode task selects its mode, so the search covers
            every mode choice, including slower modes that save energy or cost.
            ``"builder"``: the vector holds task keys only and the schedule
            builder gives each task its earliest-finishing mode, which suits
            time-based objectives such as the makespan or tardiness and leaves
            a smaller space to search.
    """

    seeded: bool = False
    modes: Literal["keys", "builder"] = "keys"

    info: ClassVar[SolverInfo]

    def __post_init__(self) -> None:
        if self.modes not in ("keys", "builder"):
            raise ValueError("modes must be 'keys' or 'builder'")

    def optimize[F: Ordered](self, run: Run[F], rng: random.Random) -> None:
        """Search ``run`` until it raises :class:`~dispatchatlas.solvers.search.Exhausted`."""
        raise NotImplementedError

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Search the random-key space of ``problem`` until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            encoding = VectorEncoding(problem, modes=self.modes)
            if single_schedule(encoding):
                search.evaluate(problem.topological_order)
            else:
                seeds = [heuristic_seed(search, encoding)] if self.seeded else []
                self.optimize(ScheduleRun(search, encoding, seeds), search.rng)
        return search.result()
