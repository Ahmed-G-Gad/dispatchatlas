"""The run-time every order-based solver uses.

:class:`Search` owns what all heuristics and metaheuristics share: the random
generator, the evaluator, the stop criteria, the incumbent, the convergence
trace, and the construction of the :class:`~dispatchatlas.core.solver.SolveResult`.
Solvers only express their search logic, so every one honours every stop
criterion, counts evaluations exactly, and scores candidates with the same
schedule builder that produces the reported schedule.

Typical use inside a solver::

    def solve(self, problem, stop, seed=0):
        search = Search(problem, stop, seed, self.info)
        with search:
            order = list(range(problem.size))
            search.evaluate(order)
            while True:
                search.next_iteration()
                ...  # propose candidates and call search.evaluate(...)
        return search.result()

``evaluate`` and ``next_iteration`` raise :class:`Exhausted` when a stop
criterion fires; leaving the ``with`` block swallows it. Constructive solvers
that build a schedule task by task hand it to :meth:`Search.submit` instead of
decoding its order again, and schedules that change a parameter over the run
read :meth:`Search.progress`.

A search created inside :func:`listening` also reports every new incumbent to
a listener, as an :class:`Improvement`. The campaign engine listens in this
way to keep what a run had found when the run is killed.
"""

from __future__ import annotations

import contextlib
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from types import TracebackType
from typing import TYPE_CHECKING, Self

from dispatchatlas.core.clock import clock_start
from dispatchatlas.core.errors import SolverError, UnsupportedProblemError
from dispatchatlas.core.pareto import dominates
from dispatchatlas.core.solver import (
    FrontPoint,
    SolveResult,
    Status,
    StopCriteria,
    StopReason,
    TracePoint,
)
from dispatchatlas.solvers.evaluation import Evaluator, Score

if TYPE_CHECKING:
    import numpy as np

    from dispatchatlas.core.model import Problem, Scalar
    from dispatchatlas.core.solver import SolverInfo
    from dispatchatlas.solvers.scheduling import Decoded

__all__ = ["Candidate", "Exhausted", "Improvement", "Listener", "Search", "listening"]


class Exhausted(Exception):  # noqa: N818 - it signals an end, not an error
    """Raised by :class:`Search` when a stop criterion fires."""

    def __init__(self, reason: StopReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class Candidate:
    """An evaluated order with its modes, decoded schedule, and score."""

    __slots__ = ("_vector", "decoded", "modes", "order", "score")

    def __init__(
        self, order: list[int], modes: list[int] | None, score: Score, decoded: Decoded
    ) -> None:
        self.order = order
        self.modes = modes
        self.score = score
        self.decoded = decoded
        self._vector: tuple[float, ...] | None = None


class Improvement:
    """A new incumbent of a search, as a listener receives it.

    Attributes:
        point: The trace point the search records for the incumbent: when the
            search found it, its value of the primary objective, and whether
            it is feasible.
    """

    __slots__ = ("_candidate", "_search", "point")

    def __init__(self, search: Search, candidate: Candidate, point: TracePoint) -> None:
        self._search = search
        self._candidate = candidate
        self.point = point

    @property
    def objective(self) -> str:
        """Name of the objective the search optimises."""
        return self._search.evaluator.objective.name

    def objectives(self) -> dict[str, float]:
        """Return every problem objective of the incumbent, keyed by name.

        The values are computed from the incumbent's decoded schedule when
        this is called, so a listener that drops an improvement pays nothing
        for them. The schedule of an evaluated candidate does not change, so
        the call is valid from another thread while the search goes on.
        """
        return self._search.evaluator.values(self._candidate.decoded)


type Listener = Callable[[Improvement], None]
"""What a search calls with each new incumbent."""

_LISTENER: ContextVar[Listener | None] = ContextVar("dispatchatlas-search-listener", default=None)


@contextlib.contextmanager
def listening(listener: Listener) -> Iterator[None]:
    """Report the new incumbents of the searches created inside the block.

    A :class:`Search` created inside the block, in the thread that entered
    it, calls ``listener`` each time its incumbent improves, for as long as
    the search runs. The call is made inside the search and its time counts
    as the solver's, so a listener returns at once and leaves any work for
    later, such as :meth:`Improvement.objectives`. An exception it raises
    ends the solver's run. A solver that does not run inside a
    :class:`Search` reports nothing.

    Args:
        listener: Called with every :class:`Improvement`.
    """
    token = _LISTENER.set(listener)
    try:
        yield
    finally:
        _LISTENER.reset(token)


class Search:
    """Budget, incumbent, and trace of one solver run.

    Args:
        problem: The problem being solved.
        stop: Stop criteria. Must be bounded when ``info.iterative`` is true.
        seed: Seed of the run's random generator.
        info: Metadata of the solver, used for the result and for feature checks.
        objective: Primary objective. Defaults to the problem's first one.
        archive: Keep the non-dominated set of feasible candidates over every
            problem objective, for multi-objective solvers.
        insertion: Passed to the schedule builder.

    Raises:
        UnsupportedProblemError: If the problem has features the solver does not
            declare in ``info.supports``.
        SolverError: If an iterative solver receives unbounded stop criteria.
    """

    def __init__(
        self,
        problem: Problem,
        stop: StopCriteria,
        seed: int,
        info: SolverInfo,
        *,
        objective: str | None = None,
        archive: bool = False,
        insertion: bool = True,
    ) -> None:
        missing = info.unsupported(problem)
        if missing:
            raise UnsupportedProblemError(
                f"solver {info.id!r} does not model features {sorted(missing)} "
                f"of problem {problem.id!r}"
            )
        if info.iterative and not stop.bounded:
            raise SolverError(f"solver {info.id!r} needs a time, evaluation, or iteration limit")
        self.problem = problem
        self.stop = stop
        self.seed = seed
        self.info = info
        self.rng = random.Random(seed)
        self.evaluator = Evaluator(problem, objective, insertion=insertion)
        self.iterations = 0
        self.best: Candidate | None = None
        self.reason: StopReason | None = None
        self.extras: dict[str, Scalar] = {}
        self._archive_enabled = archive or info.multi_objective
        self._archive: list[Candidate] = []
        self._trace: list[TracePoint] = []
        self._last_improvement = 0
        self._schedule_read = False
        self._schedule_override: bool | None = None
        self._target = (
            None if stop.target is None else self.evaluator.objective.minimized(stop.target)
        )
        self._listener = _LISTENER.get()
        self._started = clock_start()

    # ------------------------------------------------------------ properties

    @property
    def evaluations(self) -> int:
        """Candidates evaluated so far."""
        return self.evaluator.evaluations

    @property
    def elapsed(self) -> float:
        """Seconds since the origin of the run's clock.

        The origin is the instant the search was created, or, inside
        :func:`~dispatchatlas.core.clock.clock_origin`, the origin set there.
        The time limit counts from the same instant.
        """
        return time.perf_counter() - self._started

    def numpy_rng(self) -> np.random.Generator:
        """Return a NumPy generator seeded from the run seed."""
        import numpy as np  # noqa: PLC0415

        return np.random.default_rng(self.seed)

    def remaining_time(self) -> float | None:
        """Seconds left before the time limit, or ``None`` without one."""
        if self.stop.time_limit is None:
            return None
        return max(0.0, self.stop.time_limit - self.elapsed)

    def progress(self, *, deterministic: bool | None = None) -> float:
        """Return the fraction of the binding budget used so far, in [0, 1].

        Solvers whose parameters follow a schedule over the run, such as a
        falling inertia weight, read it to place the current iteration on
        that schedule. It is :meth:`StopCriteria.progress
        <dispatchatlas.core.solver.StopCriteria.progress>` of this run's
        evaluations, started iterations, and elapsed time: the largest
        fraction of a limit that the criteria's ``schedule`` counts.

        A run that reads it records in the extras of its result which clock
        its schedule followed, see :meth:`result`.

        Args:
            deterministic: Overrides the criteria's ``schedule`` for this
                reading: ``True`` counts the time limit only when neither an
                evaluation nor an iteration limit is set, so the schedule
                replays exactly on any machine, and ``False`` counts every
                set limit. ``None``, the default, follows the criteria.
        """
        self._schedule_read = True
        self._schedule_override = deterministic
        return self.stop.progress(
            self.evaluations, self.iterations, self.elapsed, deterministic=deterministic
        )

    # --------------------------------------------------------------- control

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if isinstance(error, Exhausted):
            self.reason = error.reason
            return True
        return False

    def check(self) -> None:
        """Raise :class:`Exhausted` if the time limit has passed.

        Call it inside expensive loops that evaluate nothing, so they cannot
        overrun the time limit.
        """
        limit = self.stop.time_limit
        if limit is not None and self.evaluator.evaluations > 0 and self.elapsed >= limit:
            raise Exhausted(StopReason.TIME_LIMIT)

    def next_iteration(self) -> None:
        """Start a new main-loop iteration.

        Raises:
            Exhausted: When the iteration limit, the patience, or the time limit
                is reached.
        """
        stop = self.stop
        if stop.max_iterations is not None and self.iterations >= stop.max_iterations:
            raise Exhausted(StopReason.ITERATION_LIMIT)
        if stop.patience is not None and self.iterations - self._last_improvement >= stop.patience:
            raise Exhausted(StopReason.STAGNATION)
        self.check()
        self.iterations += 1

    # ------------------------------------------------------------ evaluation

    def evaluate(self, order: Sequence[int], modes: Sequence[int] | None = None) -> Score:
        """Evaluate a candidate and update the incumbent.

        Args:
            order: Task indices in priority order.
            modes: Mode index per task, or ``None`` to let the builder choose.

        Returns:
            The candidate's score.

        Raises:
            Exhausted: When the evaluation or time limit is reached before the
                evaluation, or the target is reached by it.
        """
        return self.assess(order, modes).score

    def assess(self, order: Sequence[int], modes: Sequence[int] | None = None) -> Candidate:
        """Evaluate a candidate like :meth:`evaluate` and return all its details."""
        self._check_budget()
        score, decoded = self.evaluator.evaluate(order, modes)
        candidate = Candidate(list(order), None if modes is None else list(modes), score, decoded)
        return self._consider(candidate)

    def submit(
        self,
        decoded: Decoded,
        order: Sequence[int] | None = None,
        modes: Sequence[int] | None = None,
    ) -> Candidate:
        """Score a complete schedule the solver built itself, like :meth:`assess`.

        Constructive solvers that place tasks one at a time with a builder of
        this search's decoder (``search.evaluator.decoder.builder()``) submit
        the result instead of decoding its order again. The schedule counts as
        one evaluation and updates the incumbent, the trace, and the archive
        exactly as :meth:`assess` does. The builder keeps capacities,
        precedence, and setup times; a solver that accepts permutation
        problems must also place the tasks of each group together, as the
        decoder does.

        Args:
            decoded: The complete schedule, as ``builder.decoded()`` returns it.
            order: A priority list that yields the schedule, kept with the
                candidate for later variation. Defaults to the placement order.
            modes: Mode of every task. Defaults to the modes of the schedule.

        Returns:
            The candidate.

        Raises:
            SolverError: If the schedule leaves a task unplaced.
            Exhausted: When the evaluation or time limit is reached before the
                schedule is scored, or the target is reached by it.
        """
        missing = self.problem.size - len(decoded.sequence)
        if missing:
            raise SolverError(
                f"solver {self.info.id!r} submitted a schedule that leaves {missing} of "
                f"{self.problem.size} tasks unplaced"
            )
        self._check_budget()
        self.evaluator.evaluations += 1
        candidate = Candidate(
            list(decoded.sequence if order is None else order),
            list(decoded.modes if modes is None else modes),
            self.evaluator.score(decoded),
            decoded,
        )
        return self._consider(candidate)

    def _check_budget(self) -> None:
        """Raise :class:`Exhausted` when no evaluation may start."""
        stop = self.stop
        if stop.max_evaluations is not None and self.evaluations >= stop.max_evaluations:
            raise Exhausted(StopReason.EVALUATION_LIMIT)
        self.check()

    def _consider(self, candidate: Candidate) -> Candidate:
        """Update the incumbent, the trace, and the archive with a scored candidate."""
        score = candidate.score
        if self.best is None or score < self.best.score:
            self.best = candidate
            self._last_improvement = self.iterations
            point = TracePoint(
                self.elapsed,
                self.evaluations,
                self.iterations,
                self._reported(score),
                score.feasible,
            )
            self._trace.append(point)
            if self._listener is not None:
                self._listener(Improvement(self, candidate, point))
        if self._archive_enabled and score.feasible:
            self._update_archive(candidate)
        if (
            self._target is not None
            and self.best.score.feasible
            and self.best.score.value <= self._target
        ):
            raise Exhausted(StopReason.TARGET_REACHED)
        return candidate

    def probe(
        self, order: Sequence[int], modes: Sequence[int] | None = None
    ) -> tuple[Score, Decoded]:
        """Decode and score an order without letting it become the incumbent.

        Constructive heuristics probe partial orders to compare insertion
        positions. A probe counts as one evaluation and obeys the evaluation
        and time limits like :meth:`evaluate`, but it never changes the
        incumbent, the trace, or the archive. Evaluate a complete candidate
        before probing, so that a run stopped by a probe has a schedule to
        report.

        Args:
            order: Task indices in priority order, possibly a
                predecessor-closed subset of the tasks.
            modes: Mode index per task, negative or ``None`` to let the
                builder choose.

        Returns:
            The score and the decoded, possibly partial, schedule.

        Raises:
            Exhausted: When the evaluation or time limit is reached.
        """
        self._check_budget()
        return self.evaluator.evaluate(order, modes)

    def vector(self, candidate: Candidate) -> tuple[float, ...]:
        """Return a candidate's objective vector in minimisation form."""
        if candidate._vector is None:
            candidate._vector = self.evaluator.vector(candidate.decoded)
        return candidate._vector

    # --------------------------------------------------------------- results

    def front(self) -> list[Candidate]:
        """Return the current non-dominated feasible candidates."""
        return list(self._archive)

    def result(self, extras: Mapping[str, Scalar] | None = None) -> SolveResult:
        """Build the result of the run from the incumbent.

        A run that read :meth:`progress` records in its extras which clock
        its parameter schedule followed: ``schedule_clock`` is
        ``"evaluations"``, ``"iterations"``, or ``"time"``, the limit that
        places the end of the run on the schedule, and ``schedule_progress``
        is that place, in [0, 1]. A value below 1 says that the schedule did
        not reach its end.

        Args:
            extras: Solver-specific facts to record, merged with :attr:`extras`.

        Raises:
            SolverError: If no candidate was ever evaluated.
        """
        if self.best is None:
            raise SolverError(f"solver {self.info.id!r} evaluated no candidate")
        best = self.best
        schedule = best.decoded.schedule(self.problem)
        values = self.evaluator.values(best.decoded)
        elapsed = self.elapsed
        trace = list(self._trace)
        final = TracePoint(
            elapsed,
            self.evaluations,
            self.iterations,
            self._reported(best.score),
            best.score.feasible,
        )
        if not trace or trace[-1] != final:
            trace.append(final)
        front = tuple(
            FrontPoint(
                tuple(self.evaluator.values(item.decoded).values()),
                item.decoded.schedule(self.problem),
            )
            for item in sorted(self._archive, key=self.vector)
        )
        merged: dict[str, Scalar] = dict(self.extras)
        if self._schedule_read:
            counts = (self.evaluations, self.iterations, elapsed)
            clock = self.stop.schedule_clock(*counts, deterministic=self._schedule_override)
            if clock is not None:
                merged["schedule_clock"] = clock
                merged["schedule_progress"] = self.stop.progress(
                    *counts, deterministic=self._schedule_override
                )
        if extras:
            merged.update(extras)
        return SolveResult(
            solver=self.info.id,
            problem=self.problem.id,
            objective=self.evaluator.objective.name,
            status=Status.FEASIBLE if best.score.feasible else Status.VIOLATED,
            schedule=schedule,
            objectives=values,
            stop_reason=self.reason or StopReason.COMPLETED,
            elapsed=elapsed,
            evaluations=self.evaluations,
            iterations=self.iterations,
            trace=tuple(trace),
            front=front,
            extras=merged,
        )

    # --------------------------------------------------------------- helpers

    def _reported(self, score: Score) -> float:
        """Return a score's objective value with its natural sign."""
        return self.evaluator.objective.minimized(score.value)

    def _update_archive(self, candidate: Candidate) -> None:
        vector = self.vector(candidate)
        kept: list[Candidate] = []
        for member in self._archive:
            other = self.vector(member)
            if other == vector or dominates(other, vector):
                return
            if not dominates(vector, other):
                kept.append(member)
        kept.append(candidate)
        self._archive = kept
