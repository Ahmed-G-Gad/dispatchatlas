"""The insertion heuristic of Werner and Winkler for the job shop, in its basic form.

Werner and Winkler (1995, doi:10.1016/0166-218X(93)E0127-K) build a job-shop
schedule by inserting operations one at a time into a partial schedule, in
analogy to the insertion algorithm of Nawaz, Enscore and Ham for the
permutation flow shop (p. 196). :class:`WernerWinklerInsertion` is the basic
form of that constructive phase: a beam of width one, with **no beam search**,
which keeps the ``k`` best partial schedules of each step (p. 198), and **no
iterative phase**, the path search that improves the constructed schedule
(Section 4, which begins on p. 198).

**Partial schedule.** A partial schedule fixes a machine order for the
operations inserted so far. Its disjunctive graph has every operation as a
vertex, an arc from each operation to the next operation of its job, inserted
or not, and an arc between consecutive inserted operations of a machine (the job
arcs "are inserted at the beginning", p. 195). The length of an arc is the
processing time ``t`` of the operation it leaves. The head ``r`` of an operation
is the length of the longest path into it, and its tail ``q`` the length of the
longest path out of it, each without the operation's own ``t`` (p. 195).

**The heuristic**, as the paper gives it:

1. The job with the greatest total processing time is inserted first (p. 196).
2. The other operations are inserted one at a time by non-increasing processing
   time (p. 196).
3. An operation is inserted into the order of its machine at a position that
   keeps every job's route and creates no cycle in the graph (conditions (a) to
   (c), pp. 196-197).
4. The cost of a position is the length of the longest path through the
   operation, ``r + t + q`` (p. 197). It is not the longest path of the whole
   graph, which the authors found worse, and it need not rise from one insertion
   to the next (p. 197).
5. The position of least cost is taken.

**Choices of this implementation**, where the paper leaves a detail open or does
not treat it:

- Ties are broken by lowest index: between jobs of equal total processing time,
  the lower job index is inserted first; between operations of equal processing
  time, the lower job index and then the lower operation index; between
  positions of equal cost, the earliest position in the machine's order.
- The schedule returned is the semi-active schedule of the final machine orders:
  every operation starts at its head, so its makespan is the length of the
  critical path of the graph, which the paper takes as the makespan of a
  schedule (p. 194). The paper does not say whether the schedules it constructs
  are active.
- A release time starts a path, so the head of an operation is at least its
  release time, and the costs follow. A time limit cannot interrupt the
  construction, which has no schedule to report before the end.
- The paper's jobs visit each machine at most once (p. 193). A job that revisits
  a machine is treated by the same rules: the operations of the first job are
  placed on each machine in their route order, and a position is feasible only
  where the job's route is kept.
- The cost is the longest path, the paper's makespan criterion, whatever
  objective the problem scores; the evaluator scores the schedule returned.

**Evaluations.** A run evaluates one schedule, the one it returns, through the
search. The cost of a position is a path length in the partial graph, not a
schedule, so it is not an evaluation; the extras of the result record how many
position costs were computed. Each insertion makes a pass over the graph, so a
run takes time proportional to the square of the number of operations.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from dispatchatlas.core.errors import ModelError, SolverError
from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.graph import decoded_from_heads, heads_tails
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop

__all__ = ["WernerWinklerInsertion", "insertion_costs"]


def insertion_costs(shop: JobShop, orders: Sequence[Sequence[int]], task: int) -> dict[int, float]:
    """Return the cost of each feasible position at which an operation can join its machine.

    ``orders`` is a partial schedule: one sequence of task indices per machine,
    listing the operations inserted so far in processing order, each on its own
    machine. ``task`` is an operation that is not among them. Position ``p`` puts
    ``task`` into the order of its machine before the operation now at index ``p``,
    so the positions run from 0, first, to the length of the order, last. A
    position is feasible when the machine arcs it adds leave the graph of the
    partial schedule acyclic, which keeps every job's route.

    The cost of a position is the length of the longest path through ``task``
    once it is inserted there (Werner and Winkler 1995, p. 197): its head, its
    processing time, and its tail. The head is the larger of the head of ``task``
    along its job, which includes its release time, and the head of the operation
    before it in the machine order plus that operation's processing time. The
    tail is the larger of the tail of ``task`` along its job and the tail of the
    operation after it in the machine order plus that operation's processing
    time. A feasible position changes none of the heads and tails these
    formulas read, so they are read from the partial schedule without ``task``.

    Args:
        shop: The job shop.
        orders: The machine orders of the operations inserted so far.
        task: The operation to insert.

    Returns:
        The cost of each feasible position, keyed by position, in increasing
        order of position.

    Raises:
        ModelError: If ``task`` is not an operation of ``shop``, ``orders`` does
            not list operations once each on their own machines, ``orders`` lists
            ``task``, or ``orders`` and the jobs form a cycle.
    """
    if not 0 <= task < shop.size:
        raise ModelError(f"task {task} is not an operation of problem {shop.problem.id!r}")
    before, after = _neighbours(shop, orders, task)
    paths = _longest_paths(shop, before, after)
    if paths is None:
        raise ModelError("the machine orders and the jobs form a cycle")
    heads, tails = paths
    chain = orders[shop.machine[task]]
    precede = _reached(task, shop.predecessor, before)
    follow = _reached(task, shop.successor, after)
    # The operations of the machine that precede the task in the graph form a prefix of its
    # order, and those that follow it a suffix: the task goes between them.
    low = 0
    while low < len(chain) and precede[chain[low]]:
        low += 1
    high = len(chain)
    while high > low and follow[chain[high - 1]]:
        high -= 1
    duration = shop.duration
    costs: dict[int, float] = {}
    for position in range(low, high + 1):
        head, tail = heads[task], tails[task]
        if position > 0:
            head = max(head, heads[chain[position - 1]] + duration[chain[position - 1]])
        if position < len(chain):
            tail = max(tail, tails[chain[position]] + duration[chain[position]])
        costs[position] = head + duration[task] + tail
    return costs


def _neighbours(
    shop: JobShop, orders: Sequence[Sequence[int]], task: int
) -> tuple[list[int], list[int]]:
    """Return each operation's machine predecessor and successor in ``orders``, ``-1`` if none.

    Raises:
        ModelError: If ``orders`` does not list operations once each on their own
            machines, or lists ``task``.
    """
    if len(orders) != shop.machines:
        raise ModelError(f"expected {shop.machines} machine orders, got {len(orders)}")
    size = shop.size
    machine = shop.machine
    before = [-1] * size
    after = [-1] * size
    listed = [False] * size
    for number, order in enumerate(orders):
        previous = -1
        for entry in order:
            member = operator.index(entry)
            if not 0 <= member < size or listed[member] or machine[member] != number:
                raise ModelError(
                    f"machine order {number} lists task {member}, which is unknown, listed "
                    "twice, or runs on another machine"
                )
            listed[member] = True
            before[member] = previous
            if previous >= 0:
                after[previous] = member
            previous = member
    if listed[task]:
        raise ModelError(f"task {task} is already in the machine orders")
    return before, after


def _longest_paths(
    shop: JobShop, before: Sequence[int], after: Sequence[int]
) -> tuple[list[float], list[float]] | None:
    """Return the heads and tails of every operation, or ``None`` when the graph has a cycle.

    The graph holds every job arc and the machine arcs given by ``before`` and
    ``after``, so the operations not yet inserted are vertices without machine
    arcs.
    """
    predecessor, successor, duration = shop.predecessor, shop.successor, shop.duration
    size = shop.size
    # Kahn's algorithm: an operation is ready once its job and machine predecessors,
    # where it has them, are both ordered.
    waiting = [(predecessor[task] >= 0) + (before[task] >= 0) for task in range(size)]
    ready = [task for task in range(size) if not waiting[task]]
    order: list[int] = []
    while ready:
        task = ready.pop()
        order.append(task)
        for following in (successor[task], after[task]):
            if following >= 0:
                waiting[following] -= 1
                if not waiting[following]:
                    ready.append(following)
    if len(order) < size:
        return None
    heads = list(shop.release)
    for task in order:
        end = heads[task] + duration[task]
        for following in (successor[task], after[task]):
            if following >= 0 and heads[following] < end:
                heads[following] = end
    tails = [0.0] * size
    for task in reversed(order):
        length = tails[task] + duration[task]
        for previous in (predecessor[task], before[task]):
            if previous >= 0 and tails[previous] < length:
                tails[previous] = length
    return heads, tails


def _reached(task: int, along_job: Sequence[int], along_machine: Sequence[int]) -> list[bool]:
    """Return which operations a path from ``task`` reaches, following the given arcs.

    ``along_job`` and ``along_machine`` give the next operation on each arc, or
    ``-1``: the job and machine successors reach what follows ``task``, and the
    job and machine predecessors reach what precedes it.
    """
    reached = [False] * len(along_job)
    stack = [task]
    while stack:
        current = stack.pop()
        for following in (along_job[current], along_machine[current]):
            if following >= 0 and not reached[following]:
                reached[following] = True
                stack.append(following)
    return reached


def _insert(shop: JobShop) -> tuple[Decoded, int]:
    """Build the machine orders of the basic insertion, and return their semi-active schedule.

    Returns:
        The schedule and the number of position costs computed.
    """
    duration = shop.duration
    work = [sum(duration[task] for task in route) for route in shop.operations]
    first = min(range(shop.jobs), key=lambda job: (-work[job], job))
    # The sort is stable, so operations of equal processing time keep the order of their job
    # index and then of their place in the job.
    rest = [task for job, route in enumerate(shop.operations) if job != first for task in route]
    rest.sort(key=lambda task: -duration[task])
    orders: list[list[int]] = [[] for _ in range(shop.machines)]
    for task in shop.operations[first]:
        orders[shop.machine[task]].append(task)
    computed = 0
    for task in rest:
        costs = insertion_costs(shop, orders, task)
        # The dictionary lists the positions in increasing order, and ``min`` keeps the first
        # of equal costs: the earliest position.
        position = min(costs, key=costs.__getitem__)
        orders[shop.machine[task]].insert(position, task)
        computed += len(costs)
    paths = heads_tails(shop, orders)
    if paths is None:  # pragma: no cover - every inserted position keeps the graph acyclic
        raise SolverError("the insertion left a cycle in the machine orders")
    return decoded_from_heads(shop, paths[0]), computed


@dataclass(frozen=True)
class WernerWinklerInsertion:
    """The basic insertion heuristic of Werner and Winkler (1995), with no beam search.

    The run inserts the operations of the job shop one at a time into a partial
    schedule, the job with the greatest total processing time first and then the
    other operations by non-increasing processing time, each at the position of
    its machine's order with the least longest path through it, and returns the
    semi-active schedule of the final machine orders. This is the constructive
    phase with a beam of width one: **it has no beam search and no iterative
    phase** of the paper. The module documents the rules and the choices made
    where the paper is open. A run evaluates one schedule, and the seed has no
    effect. The extras of the result hold ``position_costs``, the number of
    position costs computed.
    """

    info: ClassVar[SolverInfo] = SolverInfo(
        id="werner-winkler-insertion",
        name="Werner-Winkler basic insertion",
        family="constructive",
        summary="Inserts the operations of a job shop one at a time, the job with the greatest "
        "total processing time first and then single operations by non-increasing processing "
        "time, each at the position of its machine's order with the least longest path through "
        "it, without beam search or an iterative phase.",
        references=("werner1995insertion",),
        stochastic=False,
        iterative=False,
        supports=JOB_SHOP_FEATURES,
        shape="job-shop",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Build the schedule of the basic insertion and evaluate it once."""
        search = Search(problem, stop, seed, self.info)
        shop = JobShop.from_problem(problem)

        def build() -> Decoded:
            decoded, computed = _insert(shop)
            search.extras["position_costs"] = computed
            return decoded

        with search:
            search.assess_built(build)
        return search.result()
