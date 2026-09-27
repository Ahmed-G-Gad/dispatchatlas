"""Greedy construction inside a search: iterated greedy and GRASP.

Iterated greedy (Ruiz and Stützle 2007) repeatedly removes a few items from the
current sequence and reinserts them greedily, NEH style. GRASP (Feo and Resende
1995) repeatedly builds a schedule with a randomised greedy rule and improves it
by local search, the first-improvement climb of
:meth:`~dispatchatlas.solvers.trajectory.space.SequenceSpace.climb` for GRASP and
the insertion search of
:meth:`~dispatchatlas.solvers.trajectory.space.SequenceSpace.descend` for
iterated greedy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.constructive import Items, insert_best, neh
from dispatchatlas.solvers.priorities import mean_durations
from dispatchatlas.solvers.scheduling import EPSILON
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory.space import SequenceSpace, Solution, worsening

__all__ = ["Grasp", "IteratedGreedy", "rebuild"]


def rebuild(space: SequenceSpace, current: Solution, count: int) -> Solution:
    """Remove ``count`` random items and reinsert each at its best position.

    The destruction and construction phases of iterated greedy (Ruiz and
    Stützle 2007): the items are drawn uniformly, then reinserted one at a time,
    ancestors first and otherwise in the order drawn, each at the position whose
    partial schedule scores best. The mode settings are kept.

    Args:
        space: The space of the run.
        current: The solution to rebuild.
        count: Items to remove, at least 1 and fewer than the number of items.
    """
    search, items = space.search, space.items
    removed = search.rng.sample(current.sequence, count)
    taken = set(removed)
    partial = [item for item in current.sequence if item not in taken]
    order = items.feasible(removed)
    scored = None
    for index, item in enumerate(order):
        partial, scored = insert_best(
            search, items, partial, item, current.modes, complete=index == len(order) - 1
        )
    if scored is None:  # pragma: no cover - the last reinsertion is always scored
        raise ValueError("the rebuilt sequence was not scored")
    return Solution(tuple(partial), current.modes, scored[0], scored[1])


@dataclass(frozen=True)
class IteratedGreedy:
    """Iterated greedy (Ruiz and Stützle 2007), from an NEH start.

    The search starts from the NEH sequence (see
    :func:`~dispatchatlas.solvers.constructive.neh`), improved by local search
    when ``local_search`` is set. Each iteration then:

    1. **destroys**: removes ``destruction`` items drawn at random;
    2. **constructs**: reinserts them, in the order drawn, each at the position
       whose partial schedule scores best; an item whose removed ancestor has
       not been reinserted yet waits for it;
    3. improves the result by local search, when ``local_search`` is set;
    4. **accepts** it when it is better than the current solution, and
       otherwise with probability ``exp(-Δ/temperature)``, where the constant
       temperature is ``temperature * mean task duration / 10``, the published
       ``T · Σp / (n · m · 10)`` with each task as one operation.

    NEH inserts whole jobs in grouped problems. The search then keeps moving
    whole jobs in permutation problems, whose schedules depend only on the job
    sequence, and single tasks otherwise. The temperature is in time units, as
    calibrated for makespan.

    Attributes:
        destruction: Items removed per iteration, the published ``d``.
        temperature: The published temperature factor ``T``.
        local_search: Apply the insertion and mode local search after NEH and
            after every construction.
    """

    destruction: int = 4
    temperature: float = 0.4
    local_search: bool = True

    info: ClassVar[SolverInfo] = SolverInfo(
        id="iterated-greedy",
        name="Iterated greedy",
        family="local-search",
        summary="Removes a few random tasks or jobs from the current sequence, reinserts "
        "each at its best position, and accepts the result at a constant temperature, "
        "starting from NEH.",
        references=("ruiz2007simple", "nawaz1983heuristic"),
    )

    def __post_init__(self) -> None:
        if self.destruction < 1:
            raise ValueError("destruction must be at least 1")
        if not self.temperature >= 0.0:
            raise ValueError("temperature must be non-negative")

    def acceptance_temperature(self, problem: Problem) -> float:
        """Return the constant acceptance temperature for ``problem``.

        ``temperature`` times the mean task duration over ten; a task's
        duration is the mean over its modes.
        """
        return self.temperature * fmean(mean_durations(problem)) / 10.0

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Destroy, reconstruct, and accept until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        temperature = self.acceptance_temperature(problem)
        with search:
            seeds = Items(problem, jobs=True)
            space = SequenceSpace(search)
            items = space.items
            sequence, score, decoded = neh(search, seeds)
            start = items.sequence_of(seeds.tasks(sequence))
            current = Solution(tuple(start), space.automatic, score, decoded)
            if self.local_search:
                current = space.descend(current)
            while space.movable and items.count > 1:
                search.next_iteration()
                candidate = rebuild(space, current, min(self.destruction, items.count - 1))
                if self.local_search:
                    candidate = space.descend(candidate)
                if candidate.score < current.score:
                    current = candidate
                    continue
                delta = worsening(candidate.score, current.score)
                if delta <= 0.0 or (
                    temperature > 0.0
                    and not math.isinf(delta)
                    and search.rng.random() <= math.exp(-delta / temperature)
                ):
                    current = candidate
        return search.result()


@dataclass(frozen=True)
class Grasp:
    """Greedy randomised adaptive search procedure (Feo and Resende 1995).

    Each iteration builds a schedule task by task. The greedy value of an
    eligible task is its earliest completion time over its modes, recomputed
    after every placement. The restricted candidate list holds the eligible
    tasks whose value is at most ``min + alpha · (max - min)``; one is drawn
    uniformly and placed in its best mode. The schedule is then improved by a
    first-improvement climb over random insert, swap, and mode neighbours. The
    best schedule over all iterations is returned.

    Attributes:
        alpha: Greediness of the candidate list, from 0 (pure greedy) to 1
            (uniformly random among eligible tasks).
        max_failures: Consecutive non-improving neighbours that end a local search.
    """

    alpha: float = 0.3
    max_failures: int = 100

    info: ClassVar[SolverInfo] = SolverInfo(
        id="grasp",
        name="GRASP",
        family="local-search",
        summary="Builds schedules by drawing each next task from those whose earliest "
        "completion time is within a fraction of the best, then improves each by a "
        "first-improvement climb.",
        references=("feo1995greedy",),
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if self.max_failures < 1:
            raise ValueError("max_failures must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Construct and improve until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        with search:
            space = SequenceSpace(search)
            while True:
                search.next_iteration()
                start = space.start(space.items.sequence_of(self._construct(space)))
                if not space.movable:
                    break
                space.climb(start, self.max_failures)
        return search.result()

    def _construct(self, space: SequenceSpace) -> list[int]:
        """Build one randomised greedy schedule and return its task order.

        Decoding that order with every multi-mode task on ``AUTO`` reproduces
        the schedule, since each task was placed in its earliest-finishing mode.
        """
        search = space.search
        problem = search.problem
        builder = search.evaluator.decoder.builder()
        waiting = [len(items) for items in problem.predecessors]
        ready = [task for task, count in enumerate(waiting) if count == 0]
        while ready:
            search.check()
            ready.sort()
            options = [(builder.best(task), task) for task in ready]
            finishes = [option[0][1] for option in options]
            low, high = min(finishes), max(finishes)
            limit = low + self.alpha * (high - low) + EPSILON
            candidates = [option for option in options if option[0][1] <= limit]
            (start, _, mode), task = candidates[search.rng.randrange(len(candidates))]
            builder.place(task, mode, start)
            ready.remove(task)
            for following in problem.successors[task]:
                waiting[following] -= 1
                if waiting[following] == 0:
                    ready.append(following)
        return builder.sequence
