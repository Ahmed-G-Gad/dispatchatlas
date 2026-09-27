"""The priority-vector engine shared by the population methods of this package.

A :class:`Swarm` holds a population of real vectors in the unit hypercube, one
row per member, with the score of each. A vector decodes through
:class:`~dispatchatlas.solvers.encoding.VectorEncoding` into a task order and,
unless the schedule builder chooses them, the modes of the tasks with several
modes, and the run's :class:`~dispatchatlas.solvers.search.Search` scores it.
Every decoded vector is therefore one evaluation: it counts against the budget,
obeys every stop criterion, and becomes the incumbent when it is the best so
far.

Methods subclass :class:`SwarmMethod` and implement only
:meth:`SwarmMethod.step`, one iteration of their update rule, with these tools:

- member updates: :meth:`Swarm.replace` moves a member whatever the new score,
  :meth:`Swarm.offer` only when the new vector is better (or, on request, no
  worse), and :meth:`Swarm.evaluate` with :meth:`Swarm.accept` handle trial
  vectors that are compared before one is kept;
- memories: :attr:`Swarm.best` and :meth:`Swarm.elite` hold the best vectors ever
  evaluated, :attr:`Swarm.velocity` a momentum per member, and
  :attr:`Swarm.trials` how many proposals in a row failed to replace each member;
- schedules: :attr:`Swarm.progress`, the fraction of the budget used;
- numerics: :func:`levy`, :func:`distance`, :func:`distances`, :func:`centroid`,
  and :func:`magnitudes`.

**Bounds.** Initial vectors are uniform in the unit hypercube. Every proposed
vector is folded back into it before decoding: a component beyond a bound is
reflected at that bound, repeatedly if needed, so ``-0.2`` becomes ``0.2`` and
``1.3`` becomes ``0.7``. The reference implementations of most of these methods
clamp to the nearest bound instead, but clamping piles keys up on the bounds,
where they tie and decode in task-index order, and a component clamped to 0
stops moving under rules whose steps scale with ``|c * g - x|`` for a guide
``g`` that is 0 there too. Reflection spreads such components back over the
range instead. Values are kept
below 1 so that a mode key never wraps to the first mode, and a non-finite
value, which only an overflowing heavy-tailed step produces, becomes the
middle of the range.

**Budget progress.** Several update rules follow a schedule over the iteration
budget ``T`` of their paper, such as a coefficient falling from 2 to 0. Here
``t / T`` is :attr:`Swarm.progress`: the fraction of the budget used since the
initial population was evaluated, measured when the iteration starts. It is the
largest of the fractions of the iteration, evaluation, and time limits that are
set, so the schedules work with any single bounded stop criterion. Under a time
limit it depends on the clock, like the run itself.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from bisect import insort
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.encoding import VectorEncoding
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.search import Search

__all__ = [
    "EPSILON",
    "LOWER",
    "MODE_SOURCES",
    "ORDER_RULES",
    "UPPER",
    "ModeSource",
    "OrderRule",
    "Swarm",
    "SwarmMethod",
    "Vector",
    "bound",
    "centroid",
    "check_fraction",
    "check_nonnegative",
    "check_positive",
    "distance",
    "distances",
    "iterate",
    "levy",
    "magnitudes",
    "mantegna_sigma",
    "method_info",
]

type Vector = NDArray[np.float64]
type OrderRule = Literal["random-key", "largest-value"]
type ModeSource = Literal["keys", "builder"]

ORDER_RULES: tuple[OrderRule, ...] = ("random-key", "largest-value")
"""The ways a vector's task keys become a task order."""

MODE_SOURCES: tuple[ModeSource, ...] = ("keys", "builder")
"""Where the modes of tasks with several modes come from."""

LOWER = 0.0
"""Lower bound of every component."""

UPPER = math.nextafter(1.0, 0.0)
"""Upper bound of every component: the largest float below 1."""

EPSILON = float(np.finfo(np.float64).eps)
"""Machine epsilon, the small constant that update rules add to a divisor."""

_TINY = float(np.finfo(np.float64).tiny)
_UNSCORED = Score(math.inf, math.inf)


def bound(vector: ArrayLike) -> Vector:
    """Return a copy of ``vector`` folded into ``[LOWER, UPPER]``.

    Components in ``[0, 1)`` are kept exactly. Any other finite component is
    mapped by the triangle wave ``|((x + 1) mod 2) - 1|``, a reflection at 0 and
    1 repeated as often as needed, and a result of exactly 1 becomes
    :data:`UPPER`. Non-finite components become 0.5.
    """
    values = np.asarray(vector, dtype=np.float64)
    finite = np.where(np.isfinite(values), values, 0.5)
    folded = np.minimum(np.abs(np.mod(finite + 1.0, 2.0) - 1.0), UPPER)
    inside = (finite >= LOWER) & (finite < 1.0)
    return np.asarray(np.where(inside, finite, folded), dtype=np.float64)


def mantegna_sigma(beta: float) -> float:
    """Return the scale of the numerator draw in Mantegna's algorithm.

    ``sigma = (Gamma(1 + beta) sin(pi beta / 2) / (Gamma((1 + beta) / 2) beta
    2^((beta - 1) / 2)))^(1 / beta)``, about 0.6966 for ``beta = 1.5`` and exactly
    1 for ``beta = 1``.
    """
    numerator = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    denominator = math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0)
    return float((numerator / denominator) ** (1.0 / beta))


def levy(rng: np.random.Generator, size: int, beta: float = 1.5) -> Vector:
    """Draw ``size`` Lévy-stable steps of index ``beta`` by Mantegna's algorithm.

    Each step is ``u / |v| ** (1 / beta)`` with ``u`` normal of standard
    deviation :func:`mantegna_sigma` and ``v`` standard normal, the first stage
    of Mantegna's (1994) algorithm and the form the cited papers use. The steps
    are symmetric about zero and their tails follow the Lévy-stable law of
    index ``beta`` and unit scale; for ``beta = 1`` they are exactly standard
    Cauchy. Callers multiply by their paper's step scale. A ``v`` of exactly
    zero is replaced by the smallest positive float, so the step is huge but
    finite.

    Raises:
        ValueError: If ``beta`` is not in ``(0, 2)``.
    """
    if not 0.0 < beta < 2.0:
        raise ValueError(f"beta must lie in (0, 2), got {beta!r}")
    numerator = rng.normal(0.0, mantegna_sigma(beta), size)
    denominator = np.maximum(np.abs(rng.standard_normal(size)), _TINY) ** (1.0 / beta)
    return np.asarray(numerator / denominator, dtype=np.float64)


def distance(first: Vector, second: Vector) -> float:
    """Return the Euclidean distance between two vectors."""
    return float(np.linalg.norm(first - second))


def distances(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the matrix of Euclidean distances between the rows of ``positions``."""
    gaps = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.asarray(np.sqrt(np.einsum("ijk,ijk->ij", gaps, gaps)), dtype=np.float64)


def centroid(positions: NDArray[np.float64]) -> Vector:
    """Return the component-wise mean of the rows of ``positions``."""
    return np.asarray(positions.mean(axis=0), dtype=np.float64)


def magnitudes(scores: Sequence[Score]) -> list[float]:
    """Turn a population's scores into numbers, smaller being better.

    Methods that weigh members by *how good* they are (attraction weights,
    selection probabilities) need one number per member, while a
    :class:`~dispatchatlas.solvers.evaluation.Score` orders deadline overrun
    first and the objective second. This applies the feasibility rule of Deb
    (2000): a member that meets every deadline counts its objective value, and
    any other counts the worst such value in the population plus its overrun,
    or its overrun alone when no member meets every deadline. Every member that
    meets the deadlines thus ranks ahead of every one that does not, and
    without deadlines the numbers are the objective values themselves.
    """
    feasible = [score.value for score in scores if score.feasible]
    worst = max(feasible, default=0.0)
    return [score.value if score.feasible else worst + score.violation for score in scores]


def check_positive(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is a finite number above zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")


def check_nonnegative(name: str, value: float) -> None:
    """Raise ``ValueError`` unless ``value`` is a finite number of at least zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")


def check_fraction(name: str, value: float, *, allow_zero: bool = False) -> None:
    """Raise ``ValueError`` unless ``value`` lies in ``(0, 1]``, or ``[0, 1]`` if allowed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    low_ok = value >= 0.0 if allow_zero else value > 0.0
    if not (low_ok and value <= 1.0):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must lie in {interval}, got {value!r}")


def method_info(solver_id: str, name: str, summary: str, *references: str) -> SolverInfo:
    """Return the metadata of a method of this package.

    Every method cites its sources and the random-key representation of Bean
    (1994) it searches in.
    """
    return SolverInfo(
        id=solver_id,
        name=name,
        family="swarm",
        summary=summary,
        references=(*references, "bean1994genetic"),
    )


class Swarm:
    """A population of priority vectors bound to one run.

    Args:
        search: The run. Every decoded vector is evaluated through it.
        size: Number of members.
        rule: How task keys become an order: ``"random-key"`` sorts tasks by
            ascending key, ``"largest-value"`` by descending key.
        modes: Where the modes of tasks with several modes come from:
            ``"keys"``, one more key per such task, or ``"builder"``, the
            schedule builder's earliest-finishing mode.

    Attributes:
        search: The run.
        encoding: The decoder from vectors to orders and modes.
        rng: The run's NumPy generator, the only source of randomness.
        size: Number of members.
        dimension: Length of the vectors: one key per task, plus one per task
            with several modes when modes come from keys.
        positions: The members, one row each.
        scores: The score of each member.
        velocity: A momentum per member, zero until a method sets it.
        trials: For each member, the proposals in a row that failed to replace it.
        best: The best vector evaluated so far, read-only.
        best_score: Its score.
        progress: Fraction of the budget used when the current iteration began.

    Raises:
        ValueError: If ``size`` is below 1.
    """

    def __init__(
        self,
        search: Search,
        size: int,
        *,
        rule: OrderRule = "random-key",
        modes: ModeSource = "keys",
    ) -> None:
        if size < 1:
            raise ValueError(f"a swarm needs at least one member, got {size}")
        self.search = search
        self.encoding = VectorEncoding(search.problem, rule=rule, modes=modes)
        self.rng = search.numpy_rng()
        self.size = size
        self.dimension = self.encoding.dimension
        self.positions: NDArray[np.float64] = np.zeros((size, self.dimension))
        self.scores: list[Score] = [_UNSCORED] * size
        self.velocity: NDArray[np.float64] = np.zeros((size, self.dimension))
        self.trials: list[int] = [0] * size
        self.best: Vector = np.zeros(self.dimension)
        self.best_score: Score = _UNSCORED
        self.progress = 0.0
        self._elite: list[tuple[Score, int, Vector]] = []
        self._serial = 0
        self._origin = (0, 0.0)

    @property
    def iteration(self) -> int:
        """Number of the current iteration, 1 for the first after initialisation."""
        return self.search.iterations

    # ------------------------------------------------------------- lifecycle

    def initialize(self) -> None:
        """Draw every member uniformly from the unit hypercube and evaluate it."""
        for index in range(self.size):
            self.replace(index, self.rng.random(self.dimension))
        self._origin = (self.search.evaluations, self.search.elapsed)

    def advance(self) -> None:
        """Start the next iteration and measure :attr:`progress`.

        Raises:
            Exhausted: When the iteration limit, the patience, or the time
                limit is reached.
        """
        self.search.next_iteration()
        self.progress = self._budget_used()

    # ------------------------------------------------------------ evaluation

    def score(self, position: Vector) -> Score:
        """Decode ``position`` and score it through the run, as one evaluation.

        This is the only place where vectors meet the problem. A subclass may
        override it to optimise another function of the vectors, as the tests
        do with analytic functions.
        """
        order, modes = self.encoding.decode(position.tolist())
        return self.search.evaluate(order, modes)

    def evaluate(self, vector: ArrayLike) -> tuple[Vector, Score]:
        """Bound, decode, and score a vector without changing the population.

        Returns:
            The bounded vector, read-only, and its score.

        Raises:
            Exhausted: When a stop criterion fires before or at the evaluation.
        """
        position = bound(vector)
        position.setflags(write=False)
        score = self.score(position)
        self._remember(position, score)
        return position, score

    def accept(self, index: int, position: Vector, score: Score) -> None:
        """Make an evaluated vector the position of member ``index``."""
        self.positions[index] = position
        self.scores[index] = score
        self.trials[index] = 0

    def replace(self, index: int, vector: ArrayLike) -> Score:
        """Evaluate ``vector`` and make it member ``index``'s position, whatever its score."""
        position, score = self.evaluate(vector)
        self.accept(index, position, score)
        return score

    def offer(self, index: int, vector: ArrayLike, *, ties: bool = False) -> bool:
        """Evaluate ``vector`` and keep it for member ``index`` only if it is better.

        Args:
            index: The member.
            vector: The proposed position.
            ties: Also keep a vector that scores exactly as well.

        Returns:
            Whether the member moved. A refused proposal adds one to the
            member's :attr:`trials`.
        """
        position, score = self.evaluate(vector)
        current = self.scores[index]
        if score < current or (ties and score == current):
            self.accept(index, position, score)
            return True
        self.trials[index] += 1
        return False

    # -------------------------------------------------------------- memories

    def elite(self, count: int) -> list[Vector]:
        """Return the ``count`` best vectors evaluated so far, best first.

        The memory holds up to :attr:`size` vectors. Among equal scores the one
        evaluated first ranks first. The vectors are read-only.
        """
        return [entry[2] for entry in self._elite[:count]]

    def leader(self) -> int:
        """Return the index of the best member, the lowest on ties."""
        return min(range(self.size), key=self.scores.__getitem__)

    def laggard(self) -> int:
        """Return the index of the worst member, the lowest on ties."""
        return max(range(self.size), key=self.scores.__getitem__)

    def ranking(self) -> list[int]:
        """Return the member indices from best to worst, lower index first on ties."""
        return sorted(range(self.size), key=self.scores.__getitem__)

    # --------------------------------------------------------------- helpers

    def _remember(self, position: Vector, score: Score) -> None:
        self._serial += 1
        if score < self.best_score:
            self.best, self.best_score = position, score
        elite = self._elite
        if len(elite) == self.size and not score < elite[-1][0]:
            return
        insort(elite, (score, self._serial, position), key=lambda entry: entry[:2])
        if len(elite) > self.size:
            elite.pop()

    def _budget_used(self) -> float:
        search = self.search
        stop = search.stop
        evaluations, elapsed = self._origin
        fractions: list[float] = []
        if stop.max_iterations is not None:
            fractions.append((search.iterations - 1) / stop.max_iterations)
        if stop.max_evaluations is not None:
            remaining = stop.max_evaluations - evaluations
            spent = search.evaluations - evaluations
            fractions.append(spent / remaining if remaining > 0 else 1.0)
        if stop.time_limit is not None:
            remaining_time = stop.time_limit - elapsed
            spent_time = search.elapsed - elapsed
            fractions.append(spent_time / remaining_time if remaining_time > 0.0 else 1.0)
        return min(1.0, max(0.0, *fractions)) if fractions else 0.0


@dataclass(frozen=True)
class SwarmMethod(ABC):
    """Parameters and run loop shared by the methods of this package.

    A subclass sets :attr:`info`, adds its own parameters, and implements
    :meth:`step`. :meth:`solve` evaluates a uniformly random initial population
    and then calls :meth:`step` once per iteration until a stop criterion fires.

    Attributes:
        population_size: Number of vectors the method keeps.
        rule: How a vector's task keys become a task order: ``"random-key"``
            (ascending keys, Bean 1994) or ``"largest-value"`` (descending
            keys, the largest-order-value rule).
        modes: Where the modes of tasks with several modes come from.
            ``"keys"``: one more key per such task selects its mode, so the
            search covers every mode choice, including slower modes that save
            energy or cost. ``"builder"``: the vectors hold task keys only and
            the schedule builder gives each task its earliest-finishing mode,
            which suits time-based objectives such as the makespan or
            tardiness and leaves a smaller space to search.
    """

    population_size: int = 20
    rule: OrderRule = "random-key"
    modes: ModeSource = "keys"

    info: ClassVar[SolverInfo]
    minimum_population: ClassVar[int] = 2
    """Smallest population the update rule works with."""

    def __post_init__(self) -> None:
        size = self.population_size
        if isinstance(size, bool) or not isinstance(size, int) or size < self.minimum_population:
            raise ValueError(
                f"population_size must be an integer of at least {self.minimum_population}, "
                f"got {size!r}"
            )
        if self.rule not in ORDER_RULES:
            raise ValueError(f"rule must be one of {ORDER_RULES}, got {self.rule!r}")
        if self.modes not in MODE_SOURCES:
            raise ValueError(f"modes must be one of {MODE_SOURCES}, got {self.modes!r}")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Search priority vectors until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            swarm = Swarm(search, self.population_size, rule=self.rule, modes=self.modes)
            swarm.initialize()
            iterate(self, swarm)
        return search.result()

    @abstractmethod
    def step(self, swarm: Swarm) -> None:
        """Apply one iteration of the update rule to ``swarm``."""


def iterate(method: SwarmMethod, swarm: Swarm) -> None:
    """Apply ``method``'s update rule to an initialised swarm until a stop criterion fires.

    Floating-point overflow is expected from heavy-tailed steps and divisions
    by small numbers, and :func:`bound` absorbs it, so NumPy's warnings about
    it are silenced here.

    Raises:
        Exhausted: When a stop criterion fires. ``Search`` swallows it.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        while True:
            swarm.advance()
            method.step(swarm)
