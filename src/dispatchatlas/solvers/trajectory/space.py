"""The solution space the trajectory methods move through.

A solution is an item sequence (see :class:`~dispatchatlas.solvers.constructive.Items`)
and, when some task has several modes, a **mode setting** for every task: a fixed
mode, or :data:`~dispatchatlas.solvers.constructive.AUTO`, which lets the builder
give the task its earliest-finishing mode when it is placed. Items are jobs in
permutation problems, whose schedules depend only on the job sequence, and single
tasks otherwise. Searches start with every multi-mode task on ``AUTO``, the
greedy mapping of list scheduling, and fix modes only where that pays. Three
moves change a solution, each keeping the sequence precedence-feasible:

- **insert** moves one item to another position between its predecessors and
  successors;
- **swap** exchanges two items whose exchange keeps precedence;
- **mode** changes one multi-mode task's setting: an ``AUTO`` task gets a fixed
  mode other than the one it runs in, a fixed task gets ``AUTO`` or another
  fixed mode, drawn uniformly.

A random neighbour draws the move kind uniformly among the kinds the problem
allows. Solutions compare by :class:`~dispatchatlas.solvers.evaluation.Score`:
deadline overrun first, then the objective. Rules that weigh *how much* worse a
candidate is use :func:`worsening`, which is infinite when the overruns differ,
so a walk always accepts a smaller overrun and never a larger one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from dispatchatlas.solvers.constructive import AUTO, Items
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.operators import insert, swap
from dispatchatlas.solvers.scheduling import EPSILON, Decoded
from dispatchatlas.solvers.search import Search

__all__ = ["Move", "MoveKind", "SequenceSpace", "Solution", "move_scale", "worsening"]

type MoveKind = Literal["insert", "swap", "mode"]


@dataclass(frozen=True, slots=True)
class Solution:
    """An evaluated point of the space.

    Attributes:
        sequence: Item indices in sequence order.
        modes: Mode setting of every task, or ``None`` when no task has several modes.
        score: The score of the decoded schedule.
        decoded: The decoded schedule, whose ``modes`` are the modes the tasks run in.
    """

    sequence: tuple[int, ...]
    modes: tuple[int, ...] | None
    score: Score
    decoded: Decoded

    @property
    def key(self) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """A hashable identity of the solution's decisions."""
        return self.sequence, self.modes


@dataclass(frozen=True, slots=True)
class Move:
    """A change to a solution.

    Attributes:
        kind: ``"insert"``, ``"swap"``, or ``"mode"``.
        first: The source position (insert), the lower position (swap), or the task (mode).
        second: The target position (insert), the upper position (swap), or the
            new setting (mode).
    """

    kind: MoveKind
    first: int
    second: int


def worsening(candidate: Score, current: Score) -> float:
    """Return how much worse ``candidate`` is than ``current``, in objective units.

    Returns:
        The objective difference when both overrun their deadlines by the same
        amount, minus infinity when the candidate's overrun is smaller, and
        plus infinity when it is larger.
    """
    gap = candidate.violation - current.violation
    if gap > EPSILON:
        return math.inf
    if gap < -EPSILON:
        return -math.inf
    return candidate.value - current.value


def move_scale(deltas: Sequence[float]) -> float:
    """Return the typical size of a worsening move among sampled deltas.

    The mean of the finite positive deltas, or, when no sampled move worsens
    the solution, the mean size of the finite negative ones, or zero when no
    move changed the objective.
    """
    worse = [delta for delta in deltas if 0.0 < delta < math.inf]
    if worse:
        return sum(worse) / len(worse)
    better = [-delta for delta in deltas if -math.inf < delta < 0.0]
    return sum(better) / len(better) if better else 0.0


class SequenceSpace:
    """Solutions, moves, and local search for one run.

    Every evaluation goes through the run's :class:`~dispatchatlas.solvers.search.Search`,
    so it counts against the budget and updates the incumbent.

    Args:
        search: The run.
        items: The items to sequence. Defaults to jobs in permutation problems
            and tasks otherwise.

    Attributes:
        search: The run.
        items: The items the sequences order.
        flexible: Indices of the tasks with more than one mode.
    """

    def __init__(self, search: Search, items: Items | None = None) -> None:
        problem = search.problem
        self.search = search
        self.items = items if items is not None else Items(problem, jobs=problem.permutation)
        self.flexible = tuple(
            task for task, spec in enumerate(problem.tasks) if len(spec.modes) > 1
        )
        self._mode_counts = tuple(len(spec.modes) for spec in problem.tasks)
        kinds: list[MoveKind] = []
        if not self.items.fixed:
            kinds += ["insert", "swap"]
        if self.flexible:
            kinds.append("mode")
        self._kinds = tuple(kinds)

    @property
    def movable(self) -> bool:
        """Whether any move exists, that is, whether the space holds more than one solution."""
        return bool(self._kinds)

    @property
    def kinds(self) -> tuple[MoveKind, ...]:
        """The move kinds this problem allows.

        Insert and swap, unless precedence fixes the sequence, and mode when
        some task has several modes.
        """
        return self._kinds

    @property
    def automatic(self) -> tuple[int, ...] | None:
        """The starting mode settings: ``AUTO`` for multi-mode tasks, mode 0 for the others."""
        if not self.flexible:
            return None
        return tuple(AUTO if count > 1 else 0 for count in self._mode_counts)

    # ------------------------------------------------------------ evaluation

    def evaluate(self, sequence: Sequence[int], modes: Sequence[int] | None) -> Solution:
        """Decode and score a sequence under the given mode settings."""
        settings = None if modes is None else tuple(modes)
        candidate = self.search.assess(self.items.tasks(sequence), settings)
        return Solution(tuple(sequence), settings, candidate.score, candidate.decoded)

    def start(self, sequence: Sequence[int]) -> Solution:
        """Evaluate a sequence with every multi-mode task on ``AUTO``."""
        return self.evaluate(sequence, self.automatic)

    def random_start(self) -> Solution:
        """Evaluate a random precedence-feasible sequence with ``AUTO`` modes."""
        return self.start(self.items.random_sequence(self.search.rng))

    def settings(self, task: int, current: int, running: int) -> list[int]:
        """Return the settings a mode move or a mode descent may give ``task``.

        Args:
            task: The task.
            current: Its current setting.
            running: The mode it runs in now.

        Returns:
            For an ``AUTO`` task, every mode but the one it runs in; for a fixed
            task, ``AUTO`` and every other mode.
        """
        others = [mode for mode in range(self._mode_counts[task]) if mode != running]
        return others if current == AUTO else [AUTO, *others]

    # ----------------------------------------------------------------- moves

    def sample(
        self,
        sequence: Sequence[int],
        modes: Sequence[int] | None,
        running: Sequence[int] | None = None,
    ) -> Move | None:
        """Draw a random move, or return ``None`` when no move exists.

        Args:
            sequence: The item sequence.
            modes: The mode settings.
            running: The modes the tasks run in, when known, so that a mode move
                never picks the mode an ``AUTO`` task already runs in.
        """
        if not self._kinds:
            return None
        return self.sample_kind(self.search.rng.choice(self._kinds), sequence, modes, running)

    def sample_kind(
        self,
        kind: MoveKind,
        sequence: Sequence[int],
        modes: Sequence[int] | None,
        running: Sequence[int] | None = None,
    ) -> Move:
        """Draw a random move of the given kind, which must be one of :attr:`kinds`.

        Args:
            kind: ``"insert"``, ``"swap"``, or ``"mode"``.
            sequence: The item sequence.
            modes: The mode settings.
            running: The modes the tasks run in, as for :meth:`sample`.
        """
        if kind not in self._kinds:
            raise ValueError(f"the space allows no {kind} move")
        if kind == "mode":
            if modes is None:  # pragma: no cover - flexible spaces always carry settings
                raise ValueError("a mode move needs mode settings")
            return self._mode_move(modes, running)
        if kind == "insert":
            return self._insert_move(sequence)
        return self._swap_move(sequence)

    def swaps(self, sequence: Sequence[int], index: int) -> list[int]:
        """Return every position whose item can exchange places with the one at ``index``.

        An exchange is allowed when both items stay between their direct
        predecessors and successors, which keeps the sequence precedence-feasible.
        """
        position = self._positions(sequence)
        low, high = self._bounds(sequence, position, index)
        partners = []
        for other in range(low, high + 1):
            if other == index:
                continue
            other_low, other_high = self._bounds(sequence, position, other)
            if other_low <= index <= other_high:
                partners.append(other)
        return partners

    @staticmethod
    def apply(
        sequence: Sequence[int], modes: Sequence[int] | None, move: Move
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Return the sequence and mode settings that ``move`` produces."""
        settings = None if modes is None else tuple(modes)
        if move.kind == "insert":
            return tuple(insert(sequence, move.first, move.second)), settings
        if move.kind == "swap":
            return tuple(swap(sequence, move.first, move.second)), settings
        if settings is None:  # pragma: no cover - flexible spaces always carry settings
            raise ValueError("a mode move needs mode settings")
        return tuple(sequence), (*settings[: move.first], move.second, *settings[move.first + 1 :])

    def neighbour(self, solution: Solution) -> tuple[Solution, Move] | None:
        """Evaluate a random neighbour of ``solution``, with the move that made it."""
        move = self.sample(solution.sequence, solution.modes, solution.decoded.modes)
        if move is None:
            return None
        sequence, modes = self.apply(solution.sequence, solution.modes, move)
        return self.evaluate(sequence, modes), move

    def perturb(self, solution: Solution, moves: int) -> Solution:
        """Apply ``moves`` random moves in a row and evaluate the result once."""
        sequence: Sequence[int] = solution.sequence
        modes: Sequence[int] | None = solution.modes
        for _ in range(moves):
            move = self.sample(sequence, modes, solution.decoded.modes)
            if move is None:
                break
            sequence, modes = self.apply(sequence, modes, move)
        return self.evaluate(sequence, modes)

    def deltas(self, solution: Solution, samples: int) -> list[float]:
        """Evaluate random neighbours and return how much worse each is than ``solution``."""
        found: list[float] = []
        for _ in range(samples):
            step = self.neighbour(solution)
            if step is None:
                break
            found.append(worsening(step[0].score, solution.score))
        return found

    # ---------------------------------------------------------- local search

    def climb(self, solution: Solution, failures: int) -> Solution:
        """Improve ``solution`` by first-improvement over random neighbours.

        Random neighbours are evaluated one at a time and each strictly better
        one becomes the current solution. The climb ends after ``failures``
        consecutive neighbours fail to improve, a sampled test of local
        optimality whose cost does not grow with the neighbourhood.
        """
        current = solution
        misses = 0
        while misses < failures:
            step = self.neighbour(current)
            if step is None:
                break
            if step[0].score < current.score:
                current, misses = step[0], 0
            else:
                misses += 1
        return current

    def descend(self, solution: Solution) -> Solution:
        """Improve ``solution`` to a local optimum of the insertion and mode moves.

        The iterative-improvement insertion search of Ruiz and Stützle (2007),
        extended to modes: items are visited in random order, and each is
        removed and reinserted at the feasible position that scores best, then
        each of its multi-mode tasks tries every other setting. A change is kept
        only when it strictly improves the solution. Passes repeat until one
        changes nothing.
        """
        rng = self.search.rng
        count = self.items.count
        current = solution
        improved = True
        while improved:
            improved = False
            for item in rng.sample(range(count), count):
                better = self.reinsert(current, item)
                if better is not None:
                    current, improved = better, True
                if current.modes is None:
                    continue
                for task in self.items.members[item]:
                    if self._mode_counts[task] > 1:
                        better = self._remode(current, task)
                        if better is not None:
                            current, improved = better, True
        return current

    def reinsert(self, current: Solution, item: int) -> Solution | None:
        """Move ``item`` to its best feasible position, if that improves ``current``.

        Every other feasible position is evaluated. Returns the best of them
        when it is strictly better than ``current``, and ``None`` otherwise.
        """
        sequence = list(current.sequence)
        origin = sequence.index(item)
        del sequence[origin]
        best: Solution | None = None
        for position in self.items.insertion_range(sequence, item):
            if position == origin:
                continue
            candidate = self.evaluate(
                [*sequence[:position], item, *sequence[position:]], current.modes
            )
            if best is None or candidate.score < best.score:
                best = candidate
        return best if best is not None and best.score < current.score else None

    def _remode(self, current: Solution, task: int) -> Solution | None:
        modes = current.modes
        if modes is None:  # pragma: no cover - called only with mode settings
            return None
        best: Solution | None = None
        for setting in self.settings(task, modes[task], current.decoded.modes[task]):
            changed = (*modes[:task], setting, *modes[task + 1 :])
            candidate = self.evaluate(current.sequence, changed)
            if best is None or candidate.score < best.score:
                best = candidate
        return best if best is not None and best.score < current.score else None

    # --------------------------------------------------------------- helpers

    def _bounds(
        self, sequence: Sequence[int], position: Sequence[int], index: int
    ) -> tuple[int, int]:
        """Return the lowest and highest positions the item at ``index`` can move to."""
        item = sequence[index]
        low = max((position[before] for before in self.items.predecessors[item]), default=-1)
        high = min(
            (position[after] for after in self.items.successors[item]), default=len(sequence)
        )
        return low + 1, high - 1

    @staticmethod
    def _positions(sequence: Sequence[int]) -> list[int]:
        position = [0] * len(sequence)
        for index, item in enumerate(sequence):
            position[item] = index
        return position

    def _insert_move(self, sequence: Sequence[int]) -> Move:
        rng = self.search.rng
        position = self._positions(sequence)
        count = len(sequence)
        chosen: tuple[int, int, int] | None = None
        for _ in range(count):
            index = rng.randrange(count)
            low, high = self._bounds(sequence, position, index)
            if high > low:
                chosen = (index, low, high)
                break
        if chosen is None:
            movable = []
            for index in range(count):
                low, high = self._bounds(sequence, position, index)
                if high > low:
                    movable.append((index, low, high))
            chosen = rng.choice(movable)
        index, low, high = chosen
        target = rng.randrange(low, high)
        return Move("insert", index, target + 1 if target >= index else target)

    def _swap_move(self, sequence: Sequence[int]) -> Move:
        rng = self.search.rng
        position = self._positions(sequence)
        count = len(sequence)
        for _ in range(count):
            index = rng.randrange(count)
            low, high = self._bounds(sequence, position, index)
            if high == low:
                continue
            other = rng.randrange(low, high)
            if other >= index:
                other += 1
            other_low, other_high = self._bounds(sequence, position, other)
            if other_low <= index <= other_high:
                return Move("swap", min(index, other), max(index, other))
        successors = self.items.successors
        adjacent = [
            index
            for index in range(count - 1)
            if sequence[index + 1] not in successors[sequence[index]]
        ]
        index = rng.choice(adjacent)
        return Move("swap", index, index + 1)

    def _mode_move(self, modes: Sequence[int], running: Sequence[int] | None) -> Move:
        rng = self.search.rng
        task = rng.choice(self.flexible)
        now = running[task] if running is not None else modes[task]
        return Move("mode", task, rng.choice(self.settings(task, modes[task], now)))
