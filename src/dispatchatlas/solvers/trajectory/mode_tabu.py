"""Tabu search over mode assignments (De Reyck and Herroelen 1999).

The search space is the **mode vector**: one mode for every task that has more
than one. The schedule of a mode vector is the serial decoding of one fixed
priority list, the tasks in decreasing upward rank (see
:func:`~dispatchatlas.solvers.priorities.upward_ranks`), so the search changes
durations and resource demands and never the priority order. The published
method solves this inner fixed-mode problem with a truncated branch and bound.

Following the published tabu search:

- The search starts with every task in its shortest mode, the published start
  when no nonrenewable resource constrains the modes.
- The neighbourhood holds every mode vector in which exactly one task has
  another mode. Every neighbour is evaluated and the best admissible one is
  taken, even when it is worse (steepest descent, mildest ascent).
- After a move from mode ``x`` to mode ``y`` of a task, moves that give that
  task mode ``x`` again are tabu. The tabu list keeps the attributes of the
  last ``L`` moves. ``L`` starts at ``2√m`` for ``m`` multi-mode tasks and, after
  every ``adjust_after`` iterations without a new best solution, grows or
  shrinks by one, or stays, with equal probability, within ``[√m, 3√m]``.
- A tabu move aspires, globally, when it beats the best solution found and,
  regionally, when it beats the best evaluated solution with that task in that
  mode. Aspiring moves have a pending status: they are admissible only when no
  non-tabu move improves the current solution.
- When no admissible move improves the current solution, the choice is
  diversified by transition frequencies: a move on a task that has changed mode
  often is penalised in proportion to its count relative to the highest one.
  The penalty is ``diversification`` times the mean worsening of the evaluated
  neighbours, so it is measured in the units of the moves themselves.

The published phases of intensification and diversification by residence
frequencies, and the aspiration by influence, by search direction, and by strong
admissibility are not implemented.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import fmean
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.priorities import upward_ranks
from dispatchatlas.solvers.search import Candidate, Search
from dispatchatlas.solvers.trajectory.space import worsening

__all__ = ["ModeAssignmentTabu", "Neighbour", "tabu_length_bounds"]


@dataclass(frozen=True, slots=True)
class Neighbour:
    """An evaluated mode vector one move away from the current one.

    Attributes:
        task: The task whose mode changes.
        mode: Its new mode.
        candidate: The evaluated schedule.
    """

    task: int
    mode: int
    candidate: Candidate

    @property
    def score(self) -> Score:
        """The neighbour's score."""
        return self.candidate.score

    @property
    def rank(self) -> tuple[Score, int, int]:
        """Sort key: score, then task, then mode."""
        return self.candidate.score, self.task, self.mode


def tabu_length_bounds(flexible: int) -> tuple[int, int, int]:
    """Return the lowest, starting, and highest tabu list length for ``flexible`` tasks.

    The published bounds ``√m`` and ``3√m`` and start ``2√m``, rounded, and at least 1.
    """
    root = math.sqrt(flexible)
    low = max(1, round(root))
    high = max(low, round(3.0 * root))
    return low, min(high, max(low, round(2.0 * root))), high


def _update_records(
    record: dict[tuple[int, int], Score],
    neighbours: list[Neighbour],
    modes: list[int],
    flexible: list[int],
) -> None:
    """Record, for every task and mode, the best evaluated solution with that task in it.

    A neighbour holds its own move's attribute and every attribute of the
    current solution except the moved task's, so each current attribute takes
    the best neighbour that moves another task.
    """
    best_by_task: dict[int, Score] = {}
    for neighbour in neighbours:
        key = (neighbour.task, neighbour.mode)
        if key not in record or neighbour.score < record[key]:
            record[key] = neighbour.score
        if neighbour.task not in best_by_task or neighbour.score < best_by_task[neighbour.task]:
            best_by_task[neighbour.task] = neighbour.score
    ranked = sorted(best_by_task.items(), key=lambda entry: (entry[1], entry[0]))
    for task in flexible:
        others = [score for moved, score in ranked[:2] if moved != task]
        if not others:
            continue
        key = (task, modes[task])
        if key not in record or others[0] < record[key]:
            record[key] = others[0]


@dataclass(frozen=True)
class ModeAssignmentTabu:
    """Tabu search over the mode vector of a multi-mode problem.

    One iteration evaluates the whole neighbourhood, one evaluation per
    neighbour, and makes one move. A problem without multi-mode tasks has a
    single mode vector, which the run evaluates once.

    Attributes:
        adjust_after: Iterations without a new best solution between two random
            changes of the tabu list length.
        diversification: Weight of the transition-frequency penalty, in mean
            worsenings of the neighbourhood.
    """

    adjust_after: int = 10
    diversification: float = 0.5

    info: ClassVar[SolverInfo] = SolverInfo(
        id="mode-assignment-tabu",
        name="Mode-assignment tabu search",
        family="local-search",
        summary="Moves each iteration to the best admissible mode vector that changes one "
        "task's mode, decoding a fixed priority list, with a tabu list of recently "
        "vacated modes, aspiration by objective, and frequency-based diversification.",
        references=("dereyck1999multi",),
    )

    def __post_init__(self) -> None:
        if self.adjust_after < 1:
            raise ValueError("adjust_after must be at least 1")
        if not self.diversification >= 0.0:
            raise ValueError("diversification must be non-negative")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Search mode vectors until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        ranks = upward_ranks(problem)
        order = sorted(range(problem.size), key=lambda task: (-ranks[task], task))
        modes = []
        for task in problem.tasks:
            durations = [mode.duration for mode in task.modes]
            modes.append(durations.index(min(durations)))
        flexible = [task for task, spec in enumerate(problem.tasks) if len(spec.modes) > 1]
        with search:
            current = search.assess(order, modes)
            if flexible:
                self._search(search, order, modes, flexible, current)
        return search.result()

    def _search(
        self,
        search: Search,
        order: list[int],
        modes: list[int],
        flexible: list[int],
        current: Candidate,
    ) -> None:
        tasks = search.problem.tasks
        rng = search.rng
        low, length, high = tabu_length_bounds(len(flexible))
        tabu: deque[tuple[int, int]] = deque()
        frequency = [0] * search.problem.size
        record = {(task, modes[task]): current.score for task in flexible}
        stale = 0
        while True:
            search.next_iteration()
            if search.best is None:  # pragma: no cover - the start was evaluated
                raise ValueError("mode-assignment tabu search needs an evaluated start")
            best = search.best.score
            neighbours = []
            for task in flexible:
                for mode in range(len(tasks[task].modes)):
                    if mode != modes[task]:
                        changed = [*modes[:task], mode, *modes[task + 1 :]]
                        neighbours.append(Neighbour(task, mode, search.assess(order, changed)))
            chosen = self._choose(neighbours, current.score, best, set(tabu), record, frequency)
            _update_records(record, neighbours, modes, flexible)
            tabu.append((chosen.task, modes[chosen.task]))
            modes[chosen.task] = chosen.mode
            frequency[chosen.task] += 1
            current = chosen.candidate
            if search.best.score < best:
                stale = 0
            else:
                stale += 1
                if stale % self.adjust_after == 0:
                    length = min(high, max(low, length + rng.choice((-1, 0, 1))))
            while len(tabu) > length:
                tabu.popleft()

    def _choose(
        self,
        neighbours: list[Neighbour],
        current: Score,
        best: Score,
        tabu: set[tuple[int, int]],
        record: dict[tuple[int, int], Score],
        frequency: list[int],
    ) -> Neighbour:
        """Pick the move to make among the evaluated neighbours."""
        free = [n for n in neighbours if (n.task, n.mode) not in tabu]
        improving = [n for n in free if n.score < current]
        if improving:
            return min(improving, key=lambda n: n.rank)
        aspiring = [
            n
            for n in neighbours
            if (n.task, n.mode) in tabu
            and (n.score < best or n.score < record.get((n.task, n.mode), n.score))
        ]
        admissible = free + aspiring
        if not admissible:
            return min(neighbours, key=lambda n: n.rank)
        if any(n.score < current for n in admissible):
            return min(admissible, key=lambda n: n.rank)
        deltas = [worsening(n.score, current) for n in admissible]
        finite = [delta for delta in deltas if 0.0 < delta < math.inf]
        scale = self.diversification * fmean(finite) if finite else 0.0
        most = max(frequency[n.task] for n in admissible) or 1

        def key(pair: tuple[Neighbour, float]) -> tuple[float, Score, int, int]:
            neighbour, delta = pair
            return (delta + scale * frequency[neighbour.task] / most, *neighbour.rank)

        return min(zip(admissible, deltas, strict=True), key=key)[0]
