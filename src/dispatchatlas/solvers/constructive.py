"""Constructive heuristics: insertion, beam and tree search, batch mapping, and SWO.

The first five build one schedule step by step and finish on their own, so none
needs a stop criterion, yet each honours the criteria it is given:

- ``neh`` (Nawaz, Enscore and Ham 1983) takes items in decreasing order of total
  work and inserts each at the position of the growing sequence whose partial
  schedule scores best.
- ``beam-search`` (Ow and Morton 1988) keeps the best few partial schedules,
  extends each by one eligible item, and ranks the extensions by a look-ahead:
  the score of the complete schedule a priority rule makes of each.
- ``min-min`` and ``max-min`` (Ibarra and Kim 1977; Braun et al. 2001) place, among
  the eligible tasks, the one whose earliest completion time is smallest, resp.
  largest, in the mode that achieves it.
- ``sufferage`` (Maheswaran et al. 1999) gives each resource, in passes, to the
  eligible task that would lose the most by not getting its best mode.

Two more repeat their construction until a stop criterion fires:

- ``squeaky-wheel-optimization`` (Joslin and Clements 1999) decodes a priority
  sequence, blames each item for its excess cost in the schedule, and moves
  blamed items forward before the next construction.
- ``monte-carlo-tree-search`` (Kocsis and Szepesvári 2006), in
  :mod:`~dispatchatlas.solvers.tree_search`, grows a search tree over sequence
  prefixes and completes each new prefix at random.

Insertion heuristics order **items**: single tasks, or whole jobs (the tasks of a
group) when the problem is grouped. :class:`Items` expands an item sequence into
a task order and keeps every sequence precedence-feasible. A partial sequence
decodes into a partial schedule, which :func:`score_sequence` scores with
:meth:`~dispatchatlas.solvers.search.Search.probe`: one evaluation that never
becomes the run's incumbent, since it does not schedule every task.

Candidates carry **mode settings**: one per task, either a mode index or
:data:`AUTO`, which lets the builder give the task its earliest-finishing mode
when the task is placed.
"""

from __future__ import annotations

import heapq
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar

from dispatchatlas.core.model import FEATURES, Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.priorities import effective_due_dates, mean_durations, upward_ranks
from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.scheduling import Builder, Decoded
from dispatchatlas.solvers.search import Search

__all__ = [
    "AUTO",
    "BeamSearch",
    "Items",
    "MaxMin",
    "MinMin",
    "Neh",
    "SqueakyWheel",
    "Sufferage",
    "insert_best",
    "neh",
    "neh_order",
    "register",
    "score_sequence",
    "task_blame",
]

AUTO = -1
"""Mode setting that lets the builder choose the task's earliest-finishing mode.

The decoder treats every negative mode entry this way.
"""

_MAPPING_FEATURES = frozenset(FEATURES) - {"permutation"}
"""Mapping heuristics interleave jobs freely, which a permutation problem forbids."""


class Items:
    """Tasks or whole jobs: the elements that sequence heuristics order.

    An item sequence expands to a task order by listing each item's tasks in
    topological order. Item ``a`` precedes item ``b`` when some task of ``a``
    precedes some task of ``b``. Every sequence built through this class keeps
    each item after its ancestors, so its expansion is precedence-feasible and
    any prefix of it decodes as a partial schedule.

    Args:
        problem: The problem.
        jobs: Order whole jobs rather than single tasks. Tasks without a group
            stay single items. Ignored when the problem has no groups, or when
            dependencies between groups form a cycle, since jobs then cannot be
            sequenced as wholes.

    Attributes:
        problem: The problem.
        jobs: Whether the items are jobs.
        members: Task indices of each item, in topological order.
        item_of: Item index of each task.
        predecessors: Direct predecessor items of each item.
        successors: Direct successor items of each item.
        topological: Item indices in a precedence-feasible order, lowest first on ties.
        work: Total mean duration of each item's tasks.
        fixed: Whether precedence admits a single item sequence.
    """

    def __init__(self, problem: Problem, *, jobs: bool) -> None:
        self.problem = problem
        rank = [0] * problem.size
        for position, task in enumerate(problem.topological_order):
            rank[task] = position
        members = _job_members(problem, rank) if jobs and problem.groups else None
        graph = None if members is None else _item_graph(problem, members)
        if members is None or graph is None:
            members = [[task] for task in range(problem.size)]
            graph = _item_graph(problem, members)
            if graph is None:  # pragma: no cover - problems are acyclic by construction
                raise ValueError(f"problem {problem.id!r} has a dependency cycle")
        predecessors, successors, topological = graph
        self.jobs = len(members) < problem.size
        self.members = tuple(tuple(tasks) for tasks in members)
        item_of = [0] * problem.size
        for item, tasks in enumerate(members):
            for task in tasks:
                item_of[task] = item
        self.item_of = tuple(item_of)
        self.predecessors = predecessors
        self.successors = successors
        self.topological = topological
        durations = mean_durations(problem)
        self.work = tuple(sum(durations[task] for task in tasks) for tasks in members)
        self._ancestors = [0] * len(members)
        for item in topological:
            mask = 0
            for before in predecessors[item]:
                mask |= self._ancestors[before] | (1 << before)
            self._ancestors[item] = mask
        self._descendants = [0] * len(members)
        for item in reversed(topological):
            mask = 0
            for after in successors[item]:
                mask |= self._descendants[after] | (1 << after)
            self._descendants[item] = mask
        self.fixed = all(after in successors[before] for before, after in pairwise(topological))

    @property
    def count(self) -> int:
        """Number of items."""
        return len(self.members)

    def tasks(self, sequence: Sequence[int]) -> list[int]:
        """Expand an item sequence into the task order it stands for."""
        if not self.jobs:
            return list(sequence)
        return [task for item in sequence for task in self.members[item]]

    def closed_tasks(self, sequence: Sequence[int]) -> list[int]:
        """Expand the items of ``sequence`` whose ancestors are all in it.

        A partial sequence may hold an item whose ancestor was removed. Its tasks
        cannot be decoded before that ancestor returns, so they are left out.
        """
        present = 0
        for item in sequence:
            present |= 1 << item
        ancestors = self._ancestors
        available = [item for item in sequence if ancestors[item] & ~present == 0]
        return self.tasks(available)

    def insertion_range(self, sequence: Sequence[int], item: int) -> range:
        """Return the positions where ``item`` can enter ``sequence``.

        The item must follow every ancestor and precede every descendant in the
        sequence, including relations that run through items not in it.
        ``sequence`` must itself respect those relations, as every sequence built
        from a feasible one by removals and feasible insertions does.
        """
        ancestors = self._ancestors[item]
        descendants = self._descendants[item]
        low, high = 0, len(sequence)
        for position, other in enumerate(sequence):
            bit = 1 << other
            if ancestors & bit:
                low = position + 1
            elif descendants & bit and high == len(sequence):
                high = position
        return range(low, high + 1)

    def feasible(self, priority: Sequence[int]) -> list[int]:
        """Return the feasible order of the given items closest to their priority.

        At each step the earliest-listed item whose listed ancestors are all
        placed comes next.

        Args:
            priority: Distinct item indices, highest priority first.
        """
        chosen = 0
        for item in priority:
            chosen |= 1 << item
        waiting = {item: (self._ancestors[item] & chosen).bit_count() for item in priority}
        rank = {item: position for position, item in enumerate(priority)}
        ready = [(rank[item], item) for item in priority if waiting[item] == 0]
        heapq.heapify(ready)
        order: list[int] = []
        while ready:
            _, item = heapq.heappop(ready)
            order.append(item)
            descendants = self._descendants[item]
            for other in priority:
                if descendants >> other & 1:
                    waiting[other] -= 1
                    if waiting[other] == 0:
                        heapq.heappush(ready, (rank[other], other))
        return order

    def sequence_of(self, order: Sequence[int]) -> list[int]:
        """Return the item sequence of a task order: items by first appearance, repaired."""
        seen: set[int] = set()
        firsts: list[int] = []
        for task in order:
            item = self.item_of[task]
            if item not in seen:
                seen.add(item)
                firsts.append(item)
        return self.feasible(firsts)

    def random_sequence(self, rng: random.Random) -> list[int]:
        """Return a random feasible sequence, drawing uniformly among ready items."""
        waiting = [len(items) for items in self.predecessors]
        ready = [item for item, count in enumerate(waiting) if count == 0]
        sequence: list[int] = []
        while ready:
            item = ready.pop(rng.randrange(len(ready)))
            sequence.append(item)
            for after in self.successors[item]:
                waiting[after] -= 1
                if waiting[after] == 0:
                    ready.append(after)
        return sequence

    def is_ancestor(self, before: int, after: int) -> bool:
        """Return whether item ``before`` must precede item ``after``."""
        return bool(self._ancestors[after] >> before & 1)


def _job_members(problem: Problem, rank: Sequence[int]) -> list[list[int]]:
    """Group task indices by job, in order of each job's first task."""
    index: dict[tuple[str, str | int], int] = {}
    members: list[list[int]] = []
    for task, spec in enumerate(problem.tasks):
        key: tuple[str, str | int] = (
            ("group", spec.group) if spec.group is not None else ("task", task)
        )
        if key not in index:
            index[key] = len(members)
            members.append([])
        members[index[key]].append(task)
    for tasks in members:
        tasks.sort(key=lambda task: rank[task])
    return members


type _Graph = tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[int, ...]]


def _item_graph(problem: Problem, members: Sequence[Sequence[int]]) -> _Graph | None:
    """Return item-level predecessors, successors, and topological order, or ``None`` on a cycle."""
    item_of = [0] * problem.size
    for item, tasks in enumerate(members):
        for task in tasks:
            item_of[task] = item
    before: list[set[int]] = [set() for _ in members]
    after: list[set[int]] = [set() for _ in members]
    for task, successors in enumerate(problem.successors):
        for following in successors:
            a, b = item_of[task], item_of[following]
            if a != b:
                after[a].add(b)
                before[b].add(a)
    waiting = [len(items) for items in before]
    ready = [item for item, count in enumerate(waiting) if count == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        item = heapq.heappop(ready)
        order.append(item)
        for following in sorted(after[item]):
            waiting[following] -= 1
            if waiting[following] == 0:
                heapq.heappush(ready, following)
    if len(order) < len(members):
        return None
    return (
        tuple(tuple(sorted(items)) for items in before),
        tuple(tuple(sorted(items)) for items in after),
        tuple(order),
    )


# ---------------------------------------------------------------- scoring


def score_sequence(
    search: Search,
    items: Items,
    sequence: Sequence[int],
    modes: Sequence[int] | None = None,
    *,
    complete: bool,
) -> tuple[Score, Decoded]:
    """Score an item sequence: as a candidate when complete, as a probe otherwise.

    A probe scores the partial schedule of the items whose ancestors are all in
    the sequence (see :meth:`Items.closed_tasks`). It counts as an evaluation
    but never becomes the incumbent, so evaluate a complete candidate first.

    Args:
        search: The run that pays for the evaluation.
        items: The items.
        sequence: The item sequence.
        modes: Mode setting of every task, or ``None`` for :data:`AUTO` everywhere.
        complete: Whether ``sequence`` holds every item.

    Returns:
        The score and the decoded schedule.
    """
    if complete:
        candidate = search.assess(items.tasks(sequence), modes)
        return candidate.score, candidate.decoded
    return search.probe(items.closed_tasks(sequence), modes)


def insert_best(
    search: Search,
    items: Items,
    sequence: Sequence[int],
    item: int,
    modes: Sequence[int] | None = None,
    *,
    complete: bool = False,
) -> tuple[list[int], tuple[Score, Decoded] | None]:
    """Insert ``item`` at the feasible position whose schedule scores best.

    The earliest of equally good positions wins. A partial insertion with a
    single feasible position is made without scoring it.

    Args:
        search: The run that pays for the evaluations.
        items: The items.
        sequence: The sequence to insert into. It does not contain ``item``.
        item: The item to insert.
        modes: Mode setting of every task, or ``None`` for :data:`AUTO` everywhere.
        complete: Whether the result holds every item, so that each candidate
            is a complete schedule evaluated by the search.

    Returns:
        The new sequence and, when scored, its score and decoded schedule.
    """
    positions = items.insertion_range(sequence, item)
    if len(positions) == 1 and not complete:
        position = positions[0]
        return [*sequence[:position], item, *sequence[position:]], None
    best: tuple[list[int], tuple[Score, Decoded]] | None = None
    for position in positions:
        candidate = [*sequence[:position], item, *sequence[position:]]
        scored = score_sequence(search, items, candidate, modes, complete=complete)
        if best is None or scored[0] < best[1][0]:
            best = (candidate, scored)
    if best is None:  # pragma: no cover - a feasible sequence always has a position
        raise ValueError("no feasible insertion position")
    return best


def neh(
    search: Search, items: Items, modes: Sequence[int] | None = None
) -> tuple[list[int], Score, Decoded]:
    """Build a sequence with the NEH insertion procedure of Nawaz, Enscore and Ham (1983).

    Items are taken in decreasing order of total work, ties by index, repaired
    so that each follows its ancestors. The first is the initial sequence; each
    further item is inserted at the best feasible position of the sequence built
    so far, judged on the partial schedule. The priority order itself is first
    evaluated as a complete candidate, so a run stopped during the insertions
    still returns a schedule.

    Args:
        search: The run that pays for the evaluations.
        items: The items to sequence.
        modes: Mode setting of every task, or ``None`` for :data:`AUTO` everywhere.

    Returns:
        The final sequence, its score, and its decoded schedule.
    """
    priority = sorted(range(items.count), key=lambda item: (-items.work[item], item))
    order = items.feasible(priority)
    initial = search.assess(items.tasks(order), modes)
    if items.count == 1:
        return order, initial.score, initial.decoded
    sequence = [order[0]]
    scored: tuple[Score, Decoded] | None = None
    for step, item in enumerate(order[1:], start=2):
        sequence, scored = insert_best(
            search, items, sequence, item, modes, complete=step == items.count
        )
    if scored is None:  # pragma: no cover - the last insertion is always scored
        raise ValueError("the complete NEH sequence was not scored")
    return sequence, scored[0], scored[1]


def neh_order(problem: Problem) -> list[int]:
    """Return the task order the NEH heuristic builds for ``problem``.

    Useful to seed a metaheuristic. Runs without a budget, and scores partial
    schedules on the problem's primary objective.
    """
    search = Search(problem, StopCriteria(), 0, Neh.info)
    items = Items(problem, jobs=True)
    sequence, _, _ = neh(search, items)
    return items.tasks(sequence)


# ---------------------------------------------------------------- solvers


@dataclass(frozen=True)
class Neh:
    """The NEH insertion heuristic of Nawaz, Enscore and Ham (1983).

    Grouped problems insert whole jobs, as in the permutation flow shop the
    heuristic was designed for; other problems insert single tasks. The builder
    picks each task's earliest-finishing mode.
    """

    info: ClassVar[SolverInfo] = SolverInfo(
        id="neh",
        name="NEH insertion",
        family="constructive",
        summary="Takes tasks, or whole jobs, in decreasing order of total work and "
        "inserts each at the position of the growing sequence whose partial "
        "schedule scores best.",
        references=("nawaz1983heuristic",),
        stochastic=False,
        iterative=False,
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Build the NEH sequence and return its schedule."""
        search = Search(problem, stop, seed, self.info)
        items = Items(problem, jobs=True)
        with search:
            neh(search, items)
        return search.result()


@dataclass(frozen=True)
class _Node:
    """A partial schedule of the beam: the item prefix and the builder that placed it."""

    prefix: tuple[int, ...]
    placed: int
    builder: Builder


@dataclass(frozen=True)
class BeamSearch:
    """Filtered beam search over partial schedules (Ow and Morton 1988).

    Level ``k`` holds up to ``width`` partial schedules of ``k`` items. Each is
    extended by its ``filter_width`` eligible items of highest priority, the
    upward rank of HEFT (for a job, its highest task rank). Every extension is
    scored by a look-ahead: the complete schedule obtained by appending the
    remaining items in priority order. The ``width`` best extensions form the
    next level; extensions that yield an identical partial schedule count once.
    One level is one iteration, so iteration and evaluation limits truncate the
    search, which then returns the best look-ahead found. The root look-ahead is
    the priority rule's own schedule, so the result is never worse than it.
    Permutation problems extend whole jobs.

    Attributes:
        width: Partial schedules kept at each level.
        filter_width: Extensions scored per partial schedule.
    """

    width: int = 3
    filter_width: int = 5

    info: ClassVar[SolverInfo] = SolverInfo(
        id="beam-search",
        name="Filtered beam search",
        family="constructive",
        summary="Keeps the best partial schedules, extends each by its highest-priority "
        "eligible items, and ranks the extensions by the score of their completion "
        "under the upward-rank rule.",
        references=("ow1988filtered", "sabuncuoglu1999job", "topcuoglu2002performance"),
        stochastic=False,
        iterative=False,
    )

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be at least 1")
        if self.filter_width < 1:
            raise ValueError("filter_width must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Run the beam from the empty schedule to complete ones."""
        search = Search(problem, stop, seed, self.info)
        items = Items(problem, jobs=problem.permutation)
        ranks = upward_ranks(problem)
        priority = [max(ranks[task] for task in tasks) for tasks in items.members]
        ranked = sorted(range(items.count), key=lambda item: (-priority[item], item))

        def lookahead(node: _Node) -> list[int]:
            rest = [item for item in ranked if not node.placed >> item & 1]
            return items.tasks([*node.prefix, *rest])

        with search:
            beam = [_Node((), 0, search.evaluator.decoder.builder())]
            search.evaluate(lookahead(beam[0]))
            for _ in range(items.count):
                search.next_iteration()
                children: list[tuple[Score, tuple[int, ...], _Node]] = []
                seen: set[tuple[tuple[int, int, float], ...]] = set()
                for node in beam:
                    eligible = [
                        item
                        for item in ranked
                        if not node.placed >> item & 1
                        and all(node.placed >> before & 1 for before in items.predecessors[item])
                    ]
                    for item in eligible[: self.filter_width]:
                        child = self._extend(node, item, items)
                        signature = tuple(
                            (task, child.builder.modes[task], child.builder.starts[task])
                            for task in sorted(child.builder.sequence)
                        )
                        if signature in seen:
                            continue
                        seen.add(signature)
                        score = search.evaluate(lookahead(child))
                        children.append((score, child.prefix, child))
                children.sort(key=lambda entry: (entry[0], entry[1]))
                beam = [child for _, _, child in children[: self.width]]
        return search.result(extras={"width": self.width, "filter_width": self.filter_width})

    @staticmethod
    def _extend(node: _Node, item: int, items: Items) -> _Node:
        builder = node.builder.copy()
        for task in items.members[item]:
            start, _, mode = builder.best(task)
            builder.place(task, mode, start)
        return _Node((*node.prefix, item), node.placed | 1 << item, builder)


type _Choice = Callable[[Builder, list[int]], list[tuple[int, int]]]
"""Picks, from the builder state and the eligible tasks, the ``(task, mode)`` pairs to place."""


def _map_tasks(problem: Problem, search: Search, choose: _Choice) -> None:
    """Place tasks in the batches ``choose`` returns until all are placed, then submit."""
    builder = search.evaluator.decoder.builder()
    waiting = [len(items) for items in problem.predecessors]
    ready = [task for task, count in enumerate(waiting) if count == 0]
    while ready:
        ready.sort()
        for task, mode in choose(builder, ready):
            builder.place(task, mode)
            ready.remove(task)
            for following in problem.successors[task]:
                waiting[following] -= 1
                if waiting[following] == 0:
                    ready.append(following)
    search.submit(builder.decoded())


def _extreme_completion(
    builder: Builder, ready: list[int], *, largest: bool
) -> list[tuple[int, int]]:
    """Choose the ready task whose best completion time is smallest, or largest.

    ``ready`` is sorted, and only a strictly better time replaces the choice, so
    ties go to the lowest task index.
    """
    chosen = (ready[0], *builder.best(ready[0])[1:])
    for task in ready[1:]:
        _, finish, mode = builder.best(task)
        if finish > chosen[1] if largest else finish < chosen[1]:
            chosen = (task, finish, mode)
    return [(chosen[0], chosen[2])]


def _mapping_info(solver_id: str, name: str, summary: str, *references: str) -> SolverInfo:
    return SolverInfo(
        id=solver_id,
        name=name,
        family="list-scheduling",
        summary=summary,
        references=references,
        stochastic=False,
        iterative=False,
        supports=_MAPPING_FEATURES,
    )


@dataclass(frozen=True)
class MinMin:
    """The min-min mapping heuristic (Ibarra and Kim 1977; Braun et al. 2001).

    Among the eligible tasks, whose predecessors are all placed, each step places
    the task with the smallest earliest completion time over its modes, in the
    mode that achieves it. Ties go to the lowest task, then the lowest mode.
    """

    info: ClassVar[SolverInfo] = _mapping_info(
        "min-min",
        "Min-min",
        "Repeatedly places, among the eligible tasks, the one whose earliest "
        "completion time over its modes is smallest, in the mode that achieves it.",
        "braun2001comparison",
        "ibarra1977heuristic",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Map every task and return the schedule."""
        search = Search(problem, stop, seed, self.info)
        with search:
            _map_tasks(problem, search, lambda b, r: _extreme_completion(b, r, largest=False))
        return search.result()


@dataclass(frozen=True)
class MaxMin:
    """The max-min mapping heuristic (Ibarra and Kim 1977; Braun et al. 2001).

    Like :class:`MinMin`, but each step places the eligible task whose earliest
    completion time is *largest*, so long tasks claim their best resources
    before short ones fill in around them.
    """

    info: ClassVar[SolverInfo] = _mapping_info(
        "max-min",
        "Max-min",
        "Repeatedly places, among the eligible tasks, the one whose earliest "
        "completion time over its modes is largest, in the mode that achieves it.",
        "braun2001comparison",
        "ibarra1977heuristic",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Map every task and return the schedule."""
        search = Search(problem, stop, seed, self.info)
        with search:
            _map_tasks(problem, search, lambda b, r: _extreme_completion(b, r, largest=True))
        return search.result()


@dataclass(frozen=True)
class Sufferage:
    """The sufferage mapping heuristic of Maheswaran et al. (1999).

    Tasks are mapped in passes. In a pass, the eligible tasks are visited in
    index order. A task's sufferage is its second-best completion time over its
    modes minus its best, or zero with a single mode. The task tentatively
    claims the resources of its best mode; if another task of the pass already
    holds one of them, the newcomer takes them over only when its sufferage is
    larger than every holder's, and the displaced tasks wait for the next pass.
    At the end of the pass the claims are placed. Resources play the role of the
    machines of the original meta-task mapping.
    """

    info: ClassVar[SolverInfo] = _mapping_info(
        "sufferage",
        "Sufferage",
        "Maps tasks in passes where each resource goes to the claiming task that "
        "would lose the most, second-best minus best completion time, without its "
        "best mode.",
        "maheswaran1999dynamic",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Map every task and return the schedule."""
        search = Search(problem, stop, seed, self.info)
        demands = [
            [tuple(resource for resource, _ in option[1]) for option in options]
            for options in search.evaluator.decoder.modes
        ]

        def one_pass(builder: Builder, ready: list[int]) -> list[tuple[int, int]]:
            claims: dict[int, tuple[int, float]] = {}  # task -> (mode, sufferage)
            holder: dict[int, int] = {}  # resource -> task
            for task in ready:
                finishes = [builder.earliest(task, mode)[1] for mode in range(len(demands[task]))]
                mode = min(range(len(finishes)), key=lambda m: (finishes[m], m))
                others = [finish for m, finish in enumerate(finishes) if m != mode]
                sufferage = min(others) - finishes[mode] if others else 0.0
                rivals = {holder[r] for r in demands[task][mode] if r in holder}
                if any(claims[rival][1] >= sufferage for rival in rivals):
                    continue
                for rival in rivals:
                    rival_mode = claims.pop(rival)[0]
                    for resource in demands[rival][rival_mode]:
                        del holder[resource]
                claims[task] = (mode, sufferage)
                for resource in demands[task][mode]:
                    holder[resource] = task
            return [(task, claims[task][0]) for task in ready if task in claims]

        with search:
            _map_tasks(problem, search, one_pass)
        return search.result()


_MAKESPAN_OBJECTIVES = frozenset({"makespan", "energy"})
"""Objectives whose dependence on start times is through the makespan alone."""

_DUE_DATE_OBJECTIVES = frozenset({"total_tardiness", "max_lateness", "tardy_tasks"})
"""Objectives whose cost lies in missed due dates."""


def task_blame(problem: Problem, objective: str) -> Callable[[Sequence[float]], list[float]]:
    """Return the analyser of squeaky wheel optimization for ``problem``.

    The blame of a task is its excess cost, the cost it incurs in a schedule
    minus the least cost it could incur, where the cost follows the objective:

    - for makespan and energy, whose time-dependent part is the makespan, the
      end of schedule that the task's finish forces, its finish plus the
      shortest chain of successors after it; the least cost is the
      critical-path lower bound of the makespan, so tasks with slack get none;
    - for the due-date objectives, the task's weighted tardiness against its
      due date, or its group's; the least cost is the tardiness of its earliest
      possible finish;
    - for any other objective, the task's weighted completion time; the least
      cost is its earliest possible finish.

    Earliest finishes and successor chains use each task's shortest mode, its
    release, and the lags, and ignore resources.

    Args:
        problem: The problem.
        objective: Name of the objective the run minimises.

    Returns:
        A function from the tasks' finish times to their blame.
    """
    index = problem.task_index
    earliest = [0.0] * problem.size
    for task in problem.topological_order:
        spec = problem.tasks[task]
        ready = spec.release
        for dependency in problem.incoming[task]:
            ready = max(ready, earliest[index[dependency.before]] + dependency.lag)
        earliest[task] = ready + spec.min_duration
    weights = [task.weight for task in problem.tasks]
    if objective in _MAKESPAN_OBJECTIVES:
        outgoing: list[list[tuple[int, float]]] = [[] for _ in range(problem.size)]
        for dependency in problem.dependencies:
            outgoing[index[dependency.before]].append((index[dependency.after], dependency.lag))
        tails = [0.0] * problem.size
        for task in reversed(problem.topological_order):
            for after, lag in outgoing[task]:
                chain = lag + problem.tasks[after].min_duration + tails[after]
                tails[task] = max(tails[task], chain)
        bound = max(low + tail for low, tail in zip(earliest, tails, strict=True))

        def excess_makespan(ends: Sequence[float]) -> list[float]:
            return [max(0.0, end + tail - bound) for end, tail in zip(ends, tails, strict=True)]

        return excess_makespan
    if objective in _DUE_DATE_OBJECTIVES:
        dues = effective_due_dates(problem)
        floors = [max(0.0, low - due) for low, due in zip(earliest, dues, strict=True)]

        def excess_tardiness(ends: Sequence[float]) -> list[float]:
            return [
                w * (max(0.0, end - due) - floor)
                for w, end, due, floor in zip(weights, ends, dues, floors, strict=True)
            ]

        return excess_tardiness

    def excess_completion(ends: Sequence[float]) -> list[float]:
        return [w * (end - low) for w, end, low in zip(weights, ends, earliest, strict=True)]

    return excess_completion


@dataclass(frozen=True)
class SqueakyWheel:
    """Squeaky wheel optimization (Joslin and Clements 1999).

    Each iteration is one construct, analyse, prioritise cycle:

    1. **construct**: decode the priority sequence of items, each task in its
       earliest-finishing mode;
    2. **analyse**: blame every item for its excess cost in that schedule (see
       :func:`task_blame`; an item's blame is its highest task blame);
    3. **prioritise**: move every blamed item forward by ``step`` times the
       number of items times its blame over the highest blame, less a random
       jitter of up to ``jitter`` positions, and repair precedence.

    An item thus needs high blame over several cycles to travel from the back
    of the sequence to the front, the published "sticky sort", and items that
    stop receiving blame sink as others overtake them. The first sequence ranks
    items by upward rank. After ``restart_after`` cycles without a new best
    schedule the cycle restarts from a random sequence, as the authors restart
    periodically. Items are jobs in permutation problems and tasks otherwise.

    Attributes:
        step: Largest forward move per cycle, as a fraction of the sequence length.
        jitter: Largest random forward shift, in positions, that breaks cycles.
        restart_after: Cycles without a new best schedule before a restart.
    """

    step: float = 0.5
    jitter: float = 1.0
    restart_after: int = 50

    info: ClassVar[SolverInfo] = SolverInfo(
        id="squeaky-wheel-optimization",
        name="Squeaky wheel optimization",
        family="constructive",
        summary="Repeatedly decodes a priority sequence, blames each task for its cost "
        "beyond the least it could incur, and moves blamed tasks forward by distances "
        "that grow with their blame.",
        references=("joslin1999squeaky",),
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.step <= 1.0:
            raise ValueError("step must lie in (0, 1]")
        if not self.jitter >= 0.0:
            raise ValueError("jitter must be non-negative")
        if self.restart_after < 1:
            raise ValueError("restart_after must be at least 1")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Construct, analyse, and prioritise until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        items = Items(problem, jobs=problem.permutation)
        blame = task_blame(problem, search.evaluator.objective.name)
        ranks = upward_ranks(problem)
        priority = [max(ranks[task] for task in tasks) for tasks in items.members]
        sequence = items.feasible(sorted(range(items.count), key=lambda i: (-priority[i], i)))
        stale = 0
        with search:
            while True:
                search.next_iteration()
                record = None if search.best is None else search.best.score
                candidate = search.assess(items.tasks(sequence))
                stale = 0 if record is None or candidate.score < record else stale + 1
                if stale >= self.restart_after:
                    sequence, stale = items.random_sequence(search.rng), 0
                    continue
                tasks = blame(candidate.decoded.ends)
                faults = [max(tasks[task] for task in members) for members in items.members]
                sequence = self.prioritise(items, sequence, faults, search.rng)
        return search.result()

    def prioritise(
        self, items: Items, sequence: Sequence[int], blame: Sequence[float], rng: random.Random
    ) -> list[int]:
        """Move blamed items forward and return the repaired sequence.

        Args:
            items: The items.
            sequence: The current priority sequence.
            blame: Blame of every item.
            rng: Source of the jitter.
        """
        top = max(blame)
        reach = self.step * len(sequence) / top if top > 0.0 else 0.0
        keys = {
            item: position - reach * max(0.0, blame[item]) - self.jitter * rng.random()
            for position, item in enumerate(sequence)
        }
        return items.feasible(sorted(sequence, key=lambda item: keys[item]))


def register(registry: SolverRegistry) -> None:
    """Add the constructive heuristics to ``registry``."""
    from dispatchatlas.solvers.tree_search import (  # noqa: PLC0415 - it imports this module
        MonteCarloTreeSearch,
    )

    for factory in (Neh, BeamSearch, MinMin, MaxMin, Sufferage, SqueakyWheel):
        registry.register(factory)
    registry.register(MonteCarloTreeSearch)
