"""Monte Carlo tree search over sequence prefixes.

The tree's root is the empty sequence, and each node is a sequence prefix whose
children append one more eligible item: a task, or a job in permutation
problems. Each iteration is one playout of UCT (Kocsis and Szepesvári 2006):

1. **selection**: from the root, while the node has no untried item, descend to
   the child with the highest UCB1 value (Auer, Cesa-Bianchi and Fischer 2002),
   ``mean reward + c · sqrt(ln N / n)`` for a node visited ``N`` times and a
   child visited ``n`` times;
2. **expansion**: append one untried eligible item, drawn uniformly, as a new
   child;
3. **simulation**: complete the sequence by drawing each next item uniformly
   among the eligible ones, and evaluate it, each multi-mode task in its
   earliest-finishing mode;
4. **backpropagation**: add the reward to every node on the path.

UCB1 needs rewards in ``[0, 1]``. The reward of a playout is the fraction of the
earlier playouts whose schedule it beats, ties counting half, which maps any
objective and deadline overrun to that interval. A node whose whole subtree has
been enumerated is closed and never selected again, so on a small problem the
search ends by itself once every sequence has been evaluated, and the best of
them is optimal for the schedule builder.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass, field
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.constructive import Items
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.search import Search

__all__ = ["MonteCarloTreeSearch", "RankReward"]


@dataclass(slots=True)
class _Node:
    """A sequence prefix: its untried items, children, visits, and reward sum."""

    untried: list[int]
    children: dict[int, _Node] = field(default_factory=dict)
    visits: int = 0
    reward: float = 0.0
    closed: bool = False


class RankReward:
    """Rewards a score by the fraction of the earlier scores it beats, ties counting half."""

    def __init__(self) -> None:
        self._scores: list[Score] = []

    def __call__(self, score: Score) -> float:
        """Return the reward of ``score`` and remember it."""
        scores = self._scores
        if not scores:
            scores.append(score)
            return 0.5
        below = bisect_left(scores, score)
        above = bisect_right(scores, score)
        reward = (len(scores) - above + 0.5 * (above - below)) / len(scores)
        insort(scores, score)
        return reward


@dataclass(frozen=True)
class MonteCarloTreeSearch:
    """UCT over sequence prefixes with uniformly random rollouts.

    Attributes:
        exploration: The constant ``c`` of the UCB1 bound. UCB1 itself uses
            ``√2``; rank rewards cluster around one half, so the mean rewards
            of siblings differ little and a smaller constant lets them count.
    """

    exploration: float = 0.3

    info: ClassVar[SolverInfo] = SolverInfo(
        id="monte-carlo-tree-search",
        name="Monte Carlo tree search",
        family="constructive",
        summary="Grows a tree of sequence prefixes by UCB1 selection and expansion of one "
        "eligible task per playout, completes each playout with random eligible tasks, "
        "and backs up rank-normalised rewards.",
        references=("kocsis2006bandit", "auer2002finite"),
    )

    def __post_init__(self) -> None:
        if not self.exploration >= 0.0:
            raise ValueError("exploration must be non-negative")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Play out until a stop criterion fires or the tree is enumerated."""
        search = Search(problem, stop, seed, self.info)
        items = Items(problem, jobs=problem.permutation)
        waiting = [len(before) for before in items.predecessors]
        eligible = [item for item, count in enumerate(waiting) if count == 0]
        root = _Node(list(eligible))
        reward = RankReward()
        nodes = 1
        with search:
            while not root.closed:
                search.next_iteration()
                nodes += self._playout(search, items, root, (waiting, eligible), reward)
        return search.result(extras={"nodes": nodes})

    def _playout(
        self,
        search: Search,
        items: Items,
        root: _Node,
        start: tuple[list[int], list[int]],
        reward: RankReward,
    ) -> int:
        """Run one selection, expansion, simulation, and backpropagation.

        Args:
            search: The run.
            items: The items.
            root: The root of the tree.
            start: Unplaced predecessor counts of every item, and the eligible
                items, of the empty sequence.
            reward: The reward function.

        Returns:
            The number of nodes added to the tree.
        """
        rng = search.rng
        waiting, ready = list(start[0]), list(start[1])
        sequence: list[int] = []

        def place(item: int) -> None:
            sequence.append(item)
            ready.remove(item)
            for after in items.successors[item]:
                waiting[after] -= 1
                if waiting[after] == 0:
                    ready.append(after)

        node, path = root, [root]
        while not node.untried and node.children:
            item = self._select(node)
            place(item)
            node = node.children[item]
            path.append(node)
        added = 0
        if node.untried:
            item = node.untried.pop(rng.randrange(len(node.untried)))
            place(item)
            child = _Node(sorted(ready))
            node.children[item] = child
            node = child
            path.append(child)
            added = 1
        while ready:
            place(ready[rng.randrange(len(ready))])
        gain = reward(search.assess(items.tasks(sequence)).score)
        for visited in path:
            visited.visits += 1
            visited.reward += gain
        for visited in reversed(path):
            if visited.untried or not all(child.closed for child in visited.children.values()):
                break
            visited.closed = True
        return added

    def _select(self, node: _Node) -> int:
        """Return the open child item of ``node`` with the highest UCB1 value."""
        spread = math.log(node.visits)
        best: tuple[float, int] | None = None
        for item in sorted(node.children):
            child = node.children[item]
            if child.closed:
                continue
            value = child.reward / child.visits + self.exploration * math.sqrt(
                spread / child.visits
            )
            if best is None or value > best[0]:
                best = (value, item)
        if best is None:  # pragma: no cover - a node with only closed children is closed
            raise ValueError("no open child to select")
        return best[1]
