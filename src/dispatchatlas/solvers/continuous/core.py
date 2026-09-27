"""Search boxes, optimisation runs, and the ``minimize`` entry point.

The optimisers of this package search real vectors. Each implements
``optimize(run, rng)``: it draws vectors, asks ``run.evaluate`` for their
fitness, calls ``run.next_iteration`` at the top of every generation, and keeps
going until the run raises :class:`~dispatchatlas.solvers.search.Exhausted`. A
:class:`Run` bundles the search box, the fitness, and the budget, so one
optimiser serves two purposes:

- :func:`minimize` runs it on a plain function of real vectors, which is how
  the methods are checked on analytic test functions;
- :class:`~dispatchatlas.solvers.continuous.encoded.ScheduleRun` scores a vector
  by the schedule it decodes to, which is how they solve scheduling problems.

Fitness values only need an order. Functions return floats; schedules return a
:class:`~dispatchatlas.solvers.evaluation.Score`, compared by deadline overrun
first and objective second. Every comparison uses that order. The one method
that weighs *how much* a candidate improved, L-SHADE, asks
:meth:`Run.improvement`.

Methods that adapt a parameter over the run, such as a decreasing inertia
weight or a shrinking population, follow :attr:`Run.progress`: the fraction of
the tightest time, evaluation, or iteration limit spent so far. A schedule
therefore reaches its end whichever limit stops the run.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from dispatchatlas.core.errors import SolverError
from dispatchatlas.core.model import Scalar
from dispatchatlas.core.solver import StopCriteria, StopReason
from dispatchatlas.solvers.search import Exhausted

__all__ = [
    "Box",
    "FunctionRun",
    "Minimum",
    "Optimizer",
    "Ordered",
    "Run",
    "initial_population",
    "midpoint_repair",
    "minimize",
]


class Ordered(Protocol):
    """A fitness value: ordered by ``<`` and ``<=``, smaller is better."""

    def __lt__(self, other: Any, /) -> bool: ...

    def __le__(self, other: Any, /) -> bool: ...


@dataclass(frozen=True)
class Box:
    """An axis-aligned search box.

    Attributes:
        lower: Lower bound of each coordinate.
        upper: Upper bound of each coordinate, above the lower bound.
        strict: Whether the fitness is defined only inside the box. When false,
            as for random keys, where every real vector decodes, the box only
            marks where the search starts, and methods designed for unbounded
            spaces may leave it.
    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    strict: bool = True

    def __post_init__(self) -> None:
        lower = tuple(float(value) for value in self.lower)
        upper = tuple(float(value) for value in self.upper)
        if not lower or len(lower) != len(upper):
            raise ValueError("a box needs bounds of equal, non-zero length")
        for low, high in zip(lower, upper, strict=True):
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("every lower bound must be finite and below its upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def cube(cls, dimension: int, lower: float, upper: float, *, strict: bool = True) -> Box:
        """Return the box ``[lower, upper]`` in every one of ``dimension`` coordinates."""
        if dimension < 1:
            raise ValueError("a box needs at least one dimension")
        return cls((lower,) * dimension, (upper,) * dimension, strict)

    @property
    def dimension(self) -> int:
        """Number of coordinates."""
        return len(self.lower)

    @property
    def widths(self) -> tuple[float, ...]:
        """Upper minus lower bound of each coordinate."""
        return tuple(high - low for low, high in zip(self.lower, self.upper, strict=True))

    def sample(self, rng: random.Random) -> list[float]:
        """Return a point drawn uniformly from the box."""
        return [rng.uniform(low, high) for low, high in zip(self.lower, self.upper, strict=True)]

    def contains(self, vector: Sequence[float]) -> bool:
        """Return whether every coordinate of ``vector`` lies within its bounds."""
        return all(
            low <= value <= high
            for value, low, high in zip(vector, self.lower, self.upper, strict=True)
        )

    def clip(self, vector: Sequence[float]) -> list[float]:
        """Return ``vector`` with each coordinate moved to its nearest bound when outside."""
        return [
            min(high, max(low, value))
            for value, low, high in zip(vector, self.lower, self.upper, strict=True)
        ]


class Run[F: Ordered](Protocol):
    """What an optimiser sees of the task it minimises.

    Attributes:
        box: The search box.
        seeds: Starting points with a known fitness. Population methods place
            them in their initial population; CMA-ES centres its first
            distribution on the best one.
    """

    @property
    def box(self) -> Box:
        """The search box."""
        ...

    @property
    def seeds(self) -> Sequence[tuple[Sequence[float], F]]:
        """Starting points with their fitness."""
        ...

    @property
    def progress(self) -> float:
        """Fraction of the tightest time, evaluation, or iteration limit spent, in [0, 1]."""
        ...

    def evaluate(self, vector: Sequence[float]) -> F:
        """Return the fitness of ``vector``, counting one evaluation.

        Raises:
            Exhausted: When a stop criterion fires.
        """
        ...

    def next_iteration(self) -> None:
        """Start a generation.

        Raises:
            Exhausted: When a stop criterion fires.
        """
        ...

    def improvement(self, before: F, after: F) -> float:
        """Return how much better ``after`` is than ``before``, positive when it is better."""
        ...

    def note(self, key: str, value: Scalar) -> None:
        """Record a fact about the search, such as the final population size."""
        ...


class Optimizer(Protocol):
    """A method that minimises the fitness of a :class:`Run`."""

    def optimize[F: Ordered](self, run: Run[F], rng: random.Random) -> None:
        """Search until ``run`` raises :class:`~dispatchatlas.solvers.search.Exhausted`."""
        ...


def initial_population[F: Ordered](
    run: Run[F], size: int, rng: random.Random
) -> tuple[list[list[float]], list[F]]:
    """Return a population of ``size`` members and their fitness.

    The run's seeds come first, without a new evaluation; uniform samples of
    the box fill the remaining places, each evaluated.
    """
    members = [list(vector) for vector, _ in run.seeds[:size]]
    fitness = [value for _, value in run.seeds[:size]]
    fresh = [run.box.sample(rng) for _ in range(size - len(members))]
    for vector in fresh:
        fitness.append(run.evaluate(vector))
        members.append(vector)
    return members, fitness


def midpoint_repair(value: float, parent: float, lower: float, upper: float) -> float:
    """Move a coordinate that left ``[lower, upper]`` halfway back to its parent's value.

    The repair of JADE (Zhang and Sanderson 2009) that L-SHADE adopts: a value
    below the box becomes ``(lower + parent) / 2``, a value above it
    ``(upper + parent) / 2``. It stays inside whenever the parent is inside.
    """
    if value < lower:
        return (lower + parent) / 2.0
    if value > upper:
        return (upper + parent) / 2.0
    return value


@dataclass(frozen=True)
class Minimum:
    """The outcome of :func:`minimize`.

    Attributes:
        vector: The best point evaluated.
        value: Its function value.
        evaluations: Function evaluations performed.
        iterations: Generations started.
        reason: The stop criterion that ended the run.
        elapsed: Wall-clock seconds.
        extras: Facts the method recorded, such as its final population size.
    """

    vector: tuple[float, ...]
    value: float
    evaluations: int
    iterations: int
    reason: StopReason
    elapsed: float
    extras: Mapping[str, Scalar] = field(default_factory=dict)


class FunctionRun:
    """A run over a plain function of real vectors.

    Its budget behaves like that of :class:`~dispatchatlas.solvers.search.Search`:
    the evaluation and time limits are checked before each evaluation, the
    iteration limit and the patience at the start of each generation, and the
    target after each evaluation. A NaN function value counts as infinity.

    Args:
        function: The function to minimise.
        box: The search box.
        stop: Stop criteria. Must include a time, evaluation, or iteration limit.

    Raises:
        SolverError: If ``stop`` is unbounded.
    """

    def __init__(
        self, function: Callable[[Sequence[float]], float], box: Box, stop: StopCriteria
    ) -> None:
        if not stop.bounded:
            raise SolverError("minimize needs a time, evaluation, or iteration limit")
        self.function = function
        self.box = box
        self.stop = stop
        self.seeds: list[tuple[Sequence[float], float]] = []
        self.evaluations = 0
        self.iterations = 0
        self.best: tuple[tuple[float, ...], float] | None = None
        self.reason: StopReason | None = None
        self.extras: dict[str, Scalar] = {}
        self._last_improvement = 0
        self._started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Seconds since the run started."""
        return time.perf_counter() - self._started

    @property
    def progress(self) -> float:
        """Fraction of the tightest limit spent, in [0, 1], by ``StopCriteria.progress``."""
        return self.stop.progress(self.evaluations, self.iterations, self.elapsed)

    def check(self) -> None:
        """Raise :class:`Exhausted` once the time limit has passed."""
        limit = self.stop.time_limit
        if limit is not None and self.evaluations > 0 and self.elapsed >= limit:
            raise Exhausted(StopReason.TIME_LIMIT)

    def next_iteration(self) -> None:
        """Start a generation, or raise :class:`Exhausted` when a limit is reached."""
        stop = self.stop
        if stop.max_iterations is not None and self.iterations >= stop.max_iterations:
            raise Exhausted(StopReason.ITERATION_LIMIT)
        if stop.patience is not None and self.iterations - self._last_improvement >= stop.patience:
            raise Exhausted(StopReason.STAGNATION)
        self.check()
        self.iterations += 1

    def evaluate(self, vector: Sequence[float]) -> float:
        """Return the function value of ``vector``, counting one evaluation."""
        stop = self.stop
        if stop.max_evaluations is not None and self.evaluations >= stop.max_evaluations:
            raise Exhausted(StopReason.EVALUATION_LIMIT)
        self.check()
        value = float(self.function(vector))
        if math.isnan(value):
            value = math.inf
        self.evaluations += 1
        if self.best is None or value < self.best[1]:
            self.best = (tuple(vector), value)
            self._last_improvement = self.iterations
        if stop.target is not None and self.best[1] <= stop.target:
            raise Exhausted(StopReason.TARGET_REACHED)
        return value

    def improvement(self, before: float, after: float) -> float:
        """Return ``before - after``."""
        return before - after

    def note(self, key: str, value: Scalar) -> None:
        """Record a fact about the search."""
        self.extras[key] = value

    def result(self) -> Minimum:
        """Return the best point found and the run's counts.

        Raises:
            SolverError: If nothing was evaluated.
        """
        if self.best is None:
            raise SolverError("the optimiser evaluated no point")
        return Minimum(
            vector=self.best[0],
            value=self.best[1],
            evaluations=self.evaluations,
            iterations=self.iterations,
            reason=self.reason or StopReason.COMPLETED,
            elapsed=self.elapsed,
            extras=dict(self.extras),
        )


def minimize(
    optimizer: Optimizer,
    function: Callable[[Sequence[float]], float],
    box: Box,
    stop: StopCriteria,
    *,
    seed: int = 0,
    initial: Sequence[Sequence[float]] = (),
) -> Minimum:
    """Minimise a function of real vectors over a box.

    Example::

        from dispatchatlas.core import StopCriteria
        from dispatchatlas.solvers.continuous import Box, CmaEs, minimize

        def sphere(x):
            return sum(value * value for value in x)

        box = Box.cube(10, -5.0, 5.0)
        best = minimize(CmaEs(), sphere, box, StopCriteria(max_evaluations=5000))

    Args:
        optimizer: The method, for example :class:`~dispatchatlas.solvers.continuous.CmaEs`.
        function: The function to minimise.
        box: The search box.
        stop: Stop criteria. Must include a time, evaluation, or iteration limit.
        seed: Seed of the run's random generator.
        initial: Points evaluated first and handed to the method as seeds.

    Returns:
        The best point evaluated, its value, and the run's counts.
    """
    run = FunctionRun(function, box, stop)
    rng = random.Random(seed)
    try:
        for point in initial:
            run.seeds.append((list(point), run.evaluate(point)))
        optimizer.optimize(run, rng)
    except Exhausted as exhausted:
        run.reason = exhausted.reason
    return run.result()
