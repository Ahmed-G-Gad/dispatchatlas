"""A selection hyper-heuristic: learned heuristic choice with late acceptance.

A selection hyper-heuristic searches over heuristics rather than directly over
solutions (Burke et al. 2013). At each step it chooses one of several
**low-level heuristics**, applies it to the current solution, and lets a
move-acceptance rule decide whether the result becomes the current solution.
Here:

- The low-level heuristics are ``insert``, ``swap``, and ``mode``, each one random
  move of that kind (see :class:`~dispatchatlas.solvers.trajectory.space.SequenceSpace`);
  ``ruin_recreate``, which removes a few random items and reinserts each at its
  best position (see :func:`~dispatchatlas.solvers.trajectory.greedy.rebuild`);
  and ``reinsert``, which moves one random item to its best position when that
  improves the solution. Only those the problem allows take part.
- The heuristic is chosen by reinforcement learning with bounded scores
  (Nareyek 2003), in the form the survey describes: every heuristic starts at
  score 0, one whose result improves the current solution gains 1, any other
  loses 1, scores stay within ``[0, max_score]``, and the heuristic with the
  highest score is applied, ties broken uniformly at random.
- The result is accepted by late acceptance (Burke and Bykov 2017), the
  combination studied by Özcan et al. (2009): a result no worse than the current
  solution, or than the current solution ``history`` steps earlier, is kept.

One iteration is one application of one heuristic, whatever it costs in
evaluations. The run starts from a random precedence-feasible sequence.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory.acceptance import LateAcceptance
from dispatchatlas.solvers.trajectory.greedy import rebuild
from dispatchatlas.solvers.trajectory.space import MoveKind, SequenceSpace, Solution

__all__ = ["Heuristic", "HeuristicScores", "SelectionHyperHeuristic", "low_level_heuristics"]

type Heuristic = Callable[[SequenceSpace, Solution], Solution]
"""A low-level heuristic: evaluates and returns a new solution made from the current one."""


def _single_move(kind: MoveKind) -> Heuristic:
    def apply(space: SequenceSpace, current: Solution) -> Solution:
        move = space.sample_kind(kind, current.sequence, current.modes, current.decoded.modes)
        return space.evaluate(*space.apply(current.sequence, current.modes, move))

    return apply


def _ruin_recreate(removed: int) -> Heuristic:
    def apply(space: SequenceSpace, current: Solution) -> Solution:
        return rebuild(space, current, min(removed, space.items.count - 1))

    return apply


def _reinsert(space: SequenceSpace, current: Solution) -> Solution:
    item = space.search.rng.randrange(space.items.count)
    better = space.reinsert(current, item)
    return current if better is None else better


def low_level_heuristics(space: SequenceSpace, removed: int) -> dict[str, Heuristic]:
    """Return the low-level heuristics the problem allows, by name.

    Args:
        space: The space of the run.
        removed: Items the ``ruin_recreate`` heuristic removes.
    """
    heuristics = {str(kind): _single_move(kind) for kind in space.kinds}
    if not space.items.fixed:
        heuristics["ruin_recreate"] = _ruin_recreate(removed)
        heuristics["reinsert"] = _reinsert
    return heuristics


class HeuristicScores:
    """Bounded reinforcement scores and the max strategy that reads them.

    Args:
        count: Number of heuristics.
        maximum: Upper bound of a score. The lower bound is 0, where scores start.

    Attributes:
        values: The score of every heuristic.
    """

    def __init__(self, count: int, maximum: int) -> None:
        self.values = [0] * count
        self._maximum = maximum

    def choose(self, rng: random.Random) -> int:
        """Return a heuristic of highest score, drawn uniformly among ties."""
        top = max(self.values)
        leaders = [index for index, value in enumerate(self.values) if value == top]
        return leaders[0] if len(leaders) == 1 else leaders[rng.randrange(len(leaders))]

    def update(self, index: int, *, improved: bool) -> None:
        """Raise a heuristic's score by one after an improvement, else lower it by one."""
        value = self.values[index] + (1 if improved else -1)
        self.values[index] = min(self._maximum, max(0, value))


@dataclass(frozen=True)
class SelectionHyperHeuristic:
    """Reinforcement-learning heuristic selection with late acceptance.

    Attributes:
        history: Length of the late-acceptance cost history, in heuristic
            applications. They cost several evaluations each, so the default
            is shorter than that of ``late-acceptance-hill-climbing``.
        max_score: Upper bound of a heuristic's score.
        removed: Items the ``ruin_recreate`` heuristic removes.
    """

    history: int = 5
    max_score: int = 10
    removed: int = 2

    info: ClassVar[SolverInfo] = SolverInfo(
        id="selection-hyper-heuristic",
        name="Selection hyper-heuristic",
        family="hyper-heuristic",
        summary="Applies the low-level heuristic with the highest reinforcement score, "
        "raised by one when it improves the current solution and lowered by one "
        "otherwise, and keeps its result by late acceptance.",
        references=(
            "burke2013hyper",
            "nareyek2003choosing",
            "ozcan2009examination",
            "burke2017late",
        ),
    )

    def __post_init__(self) -> None:
        if self.history < 1:
            raise ValueError("history must be at least 1")
        if self.max_score < 1:
            raise ValueError("max_score must be at least 1")
        if self.removed < 1:
            raise ValueError("removed must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Choose, apply, and accept heuristics until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        calls: dict[str, int] = {}
        with search:
            space = SequenceSpace(search)
            heuristics = low_level_heuristics(space, self.removed)
            names = list(heuristics)
            calls.update(dict.fromkeys(names, 0))
            current = space.random_start()
            rule = LateAcceptance([current.score] * self.history)
            scores = HeuristicScores(len(names), self.max_score)
            rng = search.rng
            while names:
                search.next_iteration()
                chosen = scores.choose(rng)
                calls[names[chosen]] += 1
                candidate = heuristics[names[chosen]](space, current)
                scores.update(chosen, improved=candidate.score < current.score)
                accepted = rule.accept(candidate.score, current.score, rng)
                if accepted:
                    current = candidate
                rule.update(current.score, accepted=accepted)
        return search.result(extras={f"calls_{name}": count for name, count in calls.items()})
