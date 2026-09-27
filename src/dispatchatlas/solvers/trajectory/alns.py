"""Adaptive large neighbourhood search (Ropke and Pisinger 2006).

Each iteration removes ``q`` items from the current solution with a **destroy**
operator and reinserts them with a **repair** operator; ``q`` is drawn uniformly
between ``min_removed`` and ``max_removed``. The operators are drawn by roulette
wheel in proportion to their weights, destroy and repair independently.

Destroy operators, on the current decoded schedule:

- **random**: ``q`` items drawn uniformly;
- **worst**: items whose tasks waited longest after they were ready, the waiting
  time standing in for the cost an item adds;
- **related**: the relatedness removal of Shaw (1998): a random item, then items
  related to a random already-removed one, relatedness being closeness of start
  times plus a unit penalty when the two items share no resource.

Worst and related removal draw from their ranked candidate list with the
randomised rule of Ropke and Pisinger: position ``floor(y^p · |L|)`` for a uniform
``y`` in ``[0, 1)`` and determinism ``p``.

Repair operators insert the removed items into the partial sequence one at a
time. A removed task with several modes loses its mode setting, as a removed
request loses its route: every feasible position is scored with the task on
``AUTO``, and at the best position each other mode is scored too. The operators
differ in which item goes next:

- **greedy**: the item whose best insertion scores best goes first;
- **regret-2**: the item that would lose most by missing its best insertion,
  the gap between its second-best and best insertion scores, goes first.

A new solution is accepted by the simulated-annealing test with a temperature
that starts where a solution ``start_worsening`` (as a fraction) worse than the
initial one is accepted with probability 0.5, and is multiplied by ``cooling``
every iteration. Every ``segment`` iterations each operator's weight becomes
``w (1 - reaction) + reaction · π / θ``, where ``θ`` counts its uses in the
segment and ``π`` sums its rewards: ``sigma1`` for a new best solution,
``sigma2`` for a better solution not accepted before, ``sigma3`` for a worse one
not accepted before but accepted now. The defaults of those parameters are the
published ones.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.constructive import AUTO, Items, score_sequence
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.trajectory.space import SequenceSpace, Solution, worsening

__all__ = ["Alns"]

type _Destroy = Callable[[SequenceSpace, Solution, int], list[int]]
type _Option = tuple[Score, list[int], tuple[int, ...] | None, Decoded]
"""A scored insertion: score, sequence, modes, decoded schedule."""


def _pick(ranked: Sequence[int], rng: random.Random, determinism: float) -> int:
    """Draw from a ranked list, favouring its head: index ``floor(y^p · len)``."""
    return ranked[int(rng.random() ** determinism * len(ranked))]


def _random_removal(space: SequenceSpace, current: Solution, count: int) -> list[int]:
    return space.search.rng.sample(current.sequence, count)


def _worst_removal(determinism: float) -> _Destroy:
    def destroy(space: SequenceSpace, current: Solution, count: int) -> list[int]:
        waits = _waiting_times(space, current.decoded)
        items = space.items
        cost = [sum(waits[task] for task in items.members[item]) for item in range(items.count)]
        ranked = sorted(current.sequence, key=lambda item: (-cost[item], item))
        removed: list[int] = []
        for _ in range(count):
            item = _pick(ranked, space.search.rng, determinism)
            ranked.remove(item)
            removed.append(item)
        return removed

    return destroy


def _related_removal(determinism: float) -> _Destroy:
    def destroy(space: SequenceSpace, current: Solution, count: int) -> list[int]:
        rng = space.search.rng
        items = space.items
        decoded = current.decoded
        decoder = space.search.evaluator.decoder
        horizon = max(decoded.makespan, 1e-9)
        starts = [min(decoded.starts[task] for task in tasks) for tasks in items.members]
        resources = [
            {
                resource
                for task in tasks
                for resource, _ in decoder.modes[task][decoded.modes[task]][1]
            }
            for tasks in items.members
        ]
        removed = [rng.choice(current.sequence)]
        remaining = [item for item in current.sequence if item != removed[0]]
        while len(removed) < count:
            anchor = rng.choice(removed)

            def relatedness(item: int, anchor: int = anchor) -> tuple[float, int]:
                shared = 0.0 if resources[item] & resources[anchor] else 1.0
                return abs(starts[item] - starts[anchor]) / horizon + shared, item

            remaining.sort(key=relatedness)
            item = _pick(remaining, rng, determinism)
            remaining.remove(item)
            removed.append(item)
        return removed

    return destroy


def _waiting_times(space: SequenceSpace, decoded: Decoded) -> list[float]:
    """Return how long each task waited after its predecessors and release allowed it to start."""
    decoder = space.search.evaluator.decoder
    waits: list[float] = []
    for task, incoming in enumerate(decoder.incoming):
        ready = decoder.release[task]
        location = decoder.modes[task][decoded.modes[task]][2]
        for before, lag, number in incoming:
            time = decoded.ends[before] + lag
            if decoder.has_network:
                source = decoder.modes[before][decoded.modes[before]][2]
                time += decoder.transfer(number, source, location)
            ready = max(ready, time)
        waits.append(max(0.0, decoded.starts[task] - ready))
    return waits


def _options(
    space: SequenceSpace,
    partial: Sequence[int],
    item: int,
    modes: tuple[int, ...] | None,
    *,
    complete: bool,
) -> list[_Option]:
    """Score the insertions of ``item``: every feasible position, then modes at the best one.

    A removed single task with several modes loses its setting: every position
    is scored with the task on ``AUTO``, and at the best position the task then
    tries each mode other than the one ``AUTO`` gave it. A job keeps its tasks'
    settings.
    """
    items = space.items
    task = items.members[item][0]
    count = len(space.search.problem.tasks[task].modes)
    flexible = modes is not None and len(items.members[item]) == 1 and count > 1
    base = _with(modes, task, AUTO) if flexible and modes is not None else modes
    found: list[_Option] = []
    for position in items.insertion_range(partial, item):
        sequence = [*partial[:position], item, *partial[position:]]
        score, decoded = score_sequence(space.search, items, sequence, base, complete=complete)
        found.append((score, sequence, base, decoded))
    if flexible and modes is not None:
        _, sequence, _, decoded = min(found, key=lambda option: option[0])
        for mode in range(count):
            if mode != decoded.modes[task]:
                chosen = _with(modes, task, mode)
                score, other = score_sequence(
                    space.search, items, sequence, chosen, complete=complete
                )
                found.append((score, sequence, chosen, other))
    return found


def _with(modes: tuple[int, ...], task: int, setting: int) -> tuple[int, ...]:
    return (*modes[:task], setting, *modes[task + 1 :])


def _repair(
    space: SequenceSpace, current: Solution, removed: list[int], *, regret: bool
) -> Solution:
    """Reinsert ``removed`` into the rest of ``current`` by greedy or regret-2 insertion.

    Only items whose removed ancestors are back compete at each step, so every
    candidate item appears in the partial schedule it is judged on.
    """
    items = space.items
    taken = set(removed)
    partial = [item for item in current.sequence if item not in taken]
    modes = current.modes
    pending = list(removed)
    last: _Option | None = None
    while pending:
        complete = len(pending) == 1
        best_key: tuple[float, Score, int] | None = None
        best: tuple[int, _Option] | None = None
        for item in pending:
            if any(items.is_ancestor(other, item) for other in pending):
                continue
            options = sorted(
                _options(space, partial, item, modes, complete=complete), key=lambda o: o[0]
            )
            top = options[0]
            gap = worsening(options[1][0], top[0]) if len(options) > 1 else math.inf
            key = (-gap if regret else 0.0, top[0], item)
            if best_key is None or key < best_key:
                best_key, best = key, (item, top)
        if best is None:  # pragma: no cover - some pending item has no pending ancestor
            break
        item, last = best
        pending.remove(item)
        partial, modes = last[1], last[2]
    if last is None:  # pragma: no cover - at least one item is removed
        return current
    return Solution(tuple(last[1]), last[2], last[0], last[3])


def _roulette(weights: Sequence[float], rng: random.Random) -> int:
    """Draw an index with probability proportional to its weight."""
    threshold = rng.random() * sum(weights)
    total = 0.0
    for index, weight in enumerate(weights):
        total += weight
        if threshold < total:
            return index
    return len(weights) - 1


@dataclass(frozen=True)
class Alns:
    """Adaptive large neighbourhood search (Ropke and Pisinger 2006).

    The search starts from a random precedence-feasible sequence. One
    iteration is one destroy, repair, and acceptance step. Removal sizes are
    smaller than the published 4 to 40 percent of the requests because every
    insertion probe decodes a schedule.

    Attributes:
        min_removed: Fewest items removed per iteration.
        max_removed: Most items removed per iteration.
        segment: Iterations between weight updates.
        reaction: Reaction factor ``r`` of the weight update.
        sigma1: Reward for a new best solution.
        sigma2: Reward for a better solution not accepted before.
        sigma3: Reward for an accepted worse solution not accepted before.
        start_worsening: A solution this fraction worse than the initial one
            is accepted with probability 0.5 at the start.
        cooling: Temperature factor per iteration.
        determinism: Exponent ``p`` of the randomised choice in worst and
            related removal.
    """

    min_removed: int = 1
    max_removed: int = 4
    segment: int = 100
    reaction: float = 0.1
    sigma1: float = 33.0
    sigma2: float = 9.0
    sigma3: float = 13.0
    start_worsening: float = 0.05
    cooling: float = 0.99975
    determinism: float = 6.0

    info: ClassVar[SolverInfo] = SolverInfo(
        id="alns",
        name="Adaptive large neighbourhood search",
        family="local-search",
        summary="Removes and greedily reinserts groups of tasks with destroy and repair "
        "operators drawn by adaptive roulette weights, accepting results by "
        "simulated annealing.",
        references=("ropke2006adaptive", "shaw1998using"),
    )

    def __post_init__(self) -> None:
        if not 1 <= self.min_removed <= self.max_removed:
            raise ValueError("removal sizes must satisfy 1 <= min_removed <= max_removed")
        if self.segment < 1:
            raise ValueError("segment must be at least 1")
        if not 0.0 <= self.reaction <= 1.0:
            raise ValueError("reaction must lie in [0, 1]")
        if min(self.sigma1, self.sigma2, self.sigma3) < 0.0:
            raise ValueError("rewards must be non-negative")
        if not self.start_worsening >= 0.0:
            raise ValueError("start_worsening must be non-negative")
        if not 0.0 < self.cooling <= 1.0:
            raise ValueError("cooling must lie in (0, 1]")
        if not self.determinism >= 1.0:
            raise ValueError("determinism must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Destroy, repair, and adapt until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        destroyers: list[_Destroy] = [
            _random_removal,
            _worst_removal(self.determinism),
            _related_removal(self.determinism),
        ]
        weights = [[1.0] * len(destroyers), [1.0, 1.0]]
        with search:
            space = SequenceSpace(search, Items(problem, jobs=problem.permutation))
            rng = search.rng
            current = space.random_start()
            temperature = self.start_worsening * abs(current.score.value) / math.log(2.0)
            accepted_before = {current.key}
            rewards = [[0.0] * len(destroyers), [0.0, 0.0]]
            uses = [[0] * len(destroyers), [0, 0]]
            count = space.items.count
            low, high = min(self.min_removed, count), min(self.max_removed, count)
            while space.movable:
                search.next_iteration()
                if search.best is None:  # pragma: no cover - the start was evaluated
                    break
                record = search.best.score
                chosen = (_roulette(weights[0], rng), _roulette(weights[1], rng))
                removed = destroyers[chosen[0]](space, current, rng.randint(low, high))
                candidate = _repair(space, current, removed, regret=chosen[1] == 1)
                delta = worsening(candidate.score, current.score)
                accepted = delta <= 0.0 or (
                    temperature > 0.0
                    and not math.isinf(delta)
                    and rng.random() < math.exp(-delta / temperature)
                )
                new = candidate.key not in accepted_before
                reward = 0.0
                if candidate.score < record:
                    reward = self.sigma1
                elif new and accepted and candidate.score < current.score:
                    reward = self.sigma2
                elif new and accepted and delta > 0.0:
                    reward = self.sigma3
                if accepted:
                    current = candidate
                    accepted_before.add(candidate.key)
                for kind in (0, 1):
                    rewards[kind][chosen[kind]] += reward
                    uses[kind][chosen[kind]] += 1
                temperature *= self.cooling
                if search.iterations % self.segment == 0:
                    self._adapt(weights, rewards, uses)
        return search.result()

    def _adapt(
        self, weights: list[list[float]], rewards: list[list[float]], uses: list[list[int]]
    ) -> None:
        """Blend each used operator's mean reward into its weight and reset the segment."""
        for kind in (0, 1):
            for index, used in enumerate(uses[kind]):
                if used:
                    weights[kind][index] = (
                        weights[kind][index] * (1.0 - self.reaction)
                        + self.reaction * rewards[kind][index] / used
                    )
                rewards[kind][index] = 0.0
                uses[kind][index] = 0
