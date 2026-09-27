"""Tabu search over sampled insert, swap, and mode moves.

Tabu search (Glover 1986, 1989) moves every iteration to the best admissible
neighbour, even when it is worse than the current solution, and forbids moves
that would undo recent ones. Here the neighbourhood is a sample of
``candidates`` random moves (a candidate-list strategy). A move's **attributes**
record where it puts things: an insert or swap puts an item at a position, a
mode move puts a task in a mode. After a move, the attributes it removed (the
item's old position, the task's old mode) are tabu for ``tenure`` iterations:
a candidate that would restore one is inadmissible, unless it beats the best
score found before the iteration (aspiration by objective). When every sampled
candidate is inadmissible, the best of them is taken anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory.space import Move, SequenceSpace, Solution

__all__ = ["TabuSearch", "attributes"]

type Attribute = tuple[int, int, int]
"""``(0, item, position)`` for an item at a position, ``(1, task, mode)`` for a task in a mode."""


def attributes(solution: Solution, move: Move) -> tuple[list[Attribute], list[Attribute]]:
    """Return the attributes a move adds to and removes from ``solution``.

    Returns:
        The attributes the neighbour gains, and those the current solution loses.
    """
    if move.kind == "mode":
        if solution.modes is None:  # pragma: no cover - mode moves need explicit modes
            raise ValueError("a mode move needs explicit modes")
        task = move.first
        return [(1, task, move.second)], [(1, task, solution.modes[task])]
    sequence = solution.sequence
    if move.kind == "insert":
        item = sequence[move.first]
        return [(0, item, move.second)], [(0, item, move.first)]
    first, second = sequence[move.first], sequence[move.second]
    return (
        [(0, first, move.second), (0, second, move.first)],
        [(0, first, move.first), (0, second, move.second)],
    )


@dataclass(frozen=True)
class TabuSearch:
    """Tabu search with a sampled neighbourhood and aspiration by objective.

    One iteration evaluates ``candidates`` random neighbours and moves to the
    best admissible one. The search starts from a random precedence-feasible
    sequence.

    Attributes:
        tenure: Iterations during which a removed attribute stays tabu.
        candidates: Neighbours sampled per iteration.
    """

    tenure: int = 7
    candidates: int = 20

    info: ClassVar[SolverInfo] = SolverInfo(
        id="tabu-search",
        name="Tabu search",
        family="local-search",
        summary="Moves to the best of a sample of neighbours while forbidding, for a "
        "fixed tenure, moves that restore recently removed item positions or task "
        "modes, unless they beat the best solution found.",
        references=("glover1986future", "glover1989tabu"),
    )

    def __post_init__(self) -> None:
        if self.tenure < 1:
            raise ValueError("tenure must be at least 1")
        if self.candidates < 1:
            raise ValueError("candidates must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Search until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space = SequenceSpace(search)
            current = space.random_start()
            expiry: dict[Attribute, int] = {}
            while space.movable:
                search.next_iteration()
                current = self._step(space, current, expiry)
        return search.result()

    def _step(
        self, space: SequenceSpace, current: Solution, expiry: dict[Attribute, int]
    ) -> Solution:
        """Sample the neighbourhood, move to the best admissible neighbour, update the list."""
        search = space.search
        now = search.iterations
        if search.best is None:  # pragma: no cover - the start was evaluated
            raise ValueError("tabu search needs an evaluated start")
        record = search.best.score
        chosen: tuple[Solution, Move] | None = None
        fallback: tuple[Solution, Move] | None = None
        for _ in range(self.candidates):
            step = space.neighbour(current)
            if step is None:  # pragma: no cover - a movable space always has a move
                break
            candidate, move = step
            if fallback is None or candidate.score < fallback[0].score:
                fallback = step
            gained, _ = attributes(current, move)
            tabu = any(expiry.get(attribute, 0) >= now for attribute in gained)
            if tabu and not candidate.score < record:
                continue
            if chosen is None or candidate.score < chosen[0].score:
                chosen = step
        selected = chosen if chosen is not None else fallback
        if selected is None:  # pragma: no cover - at least one candidate was sampled
            return current
        _, lost = attributes(current, selected[1])
        for attribute in lost:
            expiry[attribute] = now + self.tenure
        return selected[0]
