"""Tabu search over the machine orders of a job shop, with the N5 and N7 neighbourhoods.

:class:`ShopTabuSearch` is a tabu search over the block neighbourhoods of Nowicki
and Smutnicki (1996) and of Zhang et al. (2007), with the control set out below,
which is the library's own. It searches the **machine orders** of a schedule, the
order in which each machine processes its operations
(:func:`~dispatchatlas.solvers.shop.graph.machine_orders`), not a job sequence or
random keys. A move reverses the order of operations that follow each other on one
machine, and the schedule of the new orders is their semi-active one, which
:func:`~dispatchatlas.solvers.shop.graph.decoded_from_heads` builds from the heads.

**Start.** The first evaluation of a run is one random job sequence, the first draw
of the run's operator stream as in every method of this package, decoded by the
active builder (:func:`~dispatchatlas.solvers.shop.decoding.giffler_thompson`). Its
machine orders are where the walk begins. No dispatching rule or greedy pass seeds it.

**Iteration.** The search traces one critical path of the current orders
(:func:`~dispatchatlas.solvers.shop.graph.critical_blocks`), lists the moves of the
neighbourhood on its blocks, and applies the best admissible one, tabu or not
(see below), even when it makes the makespan worse. The path is the one
``critical_blocks`` traces, which prefers a machine arc where both arcs are tight.

**Neighbourhoods.** :func:`n5_moves` lists the swaps of Nowicki and Smutnicki, which
:func:`swap_estimate` prices without recomputing the schedule. :func:`n7_moves` adds
the insertions of Zhang et al., each priced exactly.

**Tabu list.** A move that reverses arcs of a machine order, an arc being a pair of
operations of one machine of which the first is processed before the second, adds
the reversed arcs, as they stand after the move, to the tabu list for a tenure drawn
for that move. A move is tabu while it would reverse an arc of the list. A tabu move
is admitted when its estimate, in N7 its exact makespan, is below the best makespan
found. When every move is tabu, the move whose tabu entry is oldest is applied. After
``max_stagnation`` moves without a new best makespan the walk restarts from the best
machine orders, after a few random swaps on a critical path, with an empty tabu list.

**Counting.** Only an applied move counts as an evaluation: the search recomputes
its heads exactly, builds the schedule with
:func:`~dispatchatlas.solvers.shop.graph.decoded_from_heads`, and counts it through
:meth:`~dispatchatlas.solvers.search.Search.assess_built`. The start and the orders a
restart begins from count one evaluation each. The estimates and the
exact scorings of the candidates of N7 are not counted, so an evaluation limit
does not bound the work this method does between two evaluations, and in the study
the wall-clock time of a run limits it. :func:`critical_block_descent` counts the
same way, for the local search of other methods, or, on request, counts every
neighbour it evaluates exactly.
"""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, NamedTuple

import numpy as np

from dispatchatlas.core.errors import ModelError, SolverError
from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria, StopReason
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.search import Exhausted, Search
from dispatchatlas.solvers.shop.family import make_representation
from dispatchatlas.solvers.shop.graph import (
    critical_blocks,
    decoded_from_heads,
    heads_tails,
    machine_orders,
)
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from dispatchatlas.solvers.shop.observers import TrajectoryObserver
from dispatchatlas.solvers.shop.run import Evaluated, ShopRun

__all__ = [
    "NEIGHBOURHOODS",
    "Descent",
    "Move",
    "NeighbourhoodName",
    "ShopTabuSearch",
    "critical_block_descent",
    "n5_moves",
    "n7_moves",
    "swap_estimate",
]

type Move = tuple[int, int, int]
"""A move of machine orders: the machine, the place of the operation to move, and its new place."""

type NeighbourhoodName = Literal["n5", "n7"]
"""The name of a neighbourhood, one of :data:`NEIGHBOURHOODS`."""

NEIGHBOURHOODS: tuple[NeighbourhoodName, ...] = ("n5", "n7")
"""The neighbourhoods of :class:`ShopTabuSearch`."""


class Descent(NamedTuple):
    """The result of :func:`critical_block_descent`.

    Attributes:
        orders: The machine orders the descent ended in.
        decoded: The semi-active schedule of ``orders``.
        makespan: The makespan of ``decoded``.
        moves: The swaps applied, each counted as one evaluation.
    """

    orders: tuple[tuple[int, ...], ...]
    decoded: Decoded
    makespan: float
    moves: int


def n5_moves(
    shop: JobShop, orders: Sequence[Sequence[int]], blocks: Sequence[Sequence[int]]
) -> list[Move]:
    """List the N5 moves of the blocks of a critical path, after Nowicki and Smutnicki (1996).

    A block is a maximal run of at least two operations of the path on one machine,
    which are adjacent in the machine's order. A move swaps two adjacent operations
    of a block, and is a triple ``(machine, place, place + 1)`` of
    :data:`Move`. The rule, which the library reads from the paper:

    - the swap of the first two operations of a block is a move unless the block is
      the first block of the path;
    - the swap of the last two operations of a block is a move unless the block is
      the last block of the path;
    - a block of two operations has one swap, which is both its first pair and its
      last pair, and which a move keeps when either rule keeps it, so the first
      block of two keeps it as its last pair and the last block of two as its first;
    - a path of one block has no move.

    The first pair of the first block and the last pair of the last block are
    left out because swapping them cannot shorten the path when every release time
    is zero, as in the benchmark instances: the path leaves the first block by its
    last operation and enters the last block by its first. With release times a
    swap of the excluded pairs can shorten it, and the neighbourhood still leaves
    them out. A swap of two operations that the same job also joins is left out as
    well, since it would close a cycle; it arises only in a job that returns to a
    machine at once.

    Args:
        shop: The job shop.
        orders: The machine orders the blocks belong to.
        blocks: The blocks of a critical path of ``orders``, as
            :func:`~dispatchatlas.solvers.shop.graph.critical_blocks` returns them.

    Returns:
        The moves, block by block in path order, the first pair of a block before
        its last pair.
    """
    last = len(blocks) - 1
    moves: list[Move] = []
    for index, block in enumerate(blocks):
        if len(block) < 2:
            continue
        machine = shop.machine[block[0]]
        order = orders[machine]
        start = order.index(block[0])
        places = []
        if index > 0:
            places.append(start)
        if index < last:
            places.append(start + len(block) - 2)
        for place in dict.fromkeys(places):
            if shop.successor[order[place]] != order[place + 1]:
                moves.append((machine, place, place + 1))
    return moves


def n7_moves(
    shop: JobShop, orders: Sequence[Sequence[int]], blocks: Sequence[Sequence[int]]
) -> list[Move]:
    """List the N7 moves of the blocks of a critical path, after Zhang et al. (2007).

    N7 extends N5 with insertions, in the manner of the neighbourhood of Balas and
    Vazacopoulos (1998). In a block of ``k`` operations, a move takes the operation
    at one place of the block and inserts it at another, the operations between
    them shifting by one place, and it is a move when either place is the first or
    the last of the block. So the first operation moves to just after any other,
    any other moves to just before the first, the last operation moves to just before
    any other, and any other moves to just after the last. A swap of two adjacent
    operations is one move, not two. As in N5, a first block takes only the moves that
    change its last operation, a last block only those that change its first, and a
    path of one block has none, so the N5 moves are among the N7 moves.

    The moves are not checked for cycles here, which needs the heads of the new
    orders; the search keeps only the moves that leave the graph acyclic.

    Certainty: moderate, about 60 percent. The set is rebuilt from the published
    description of N6 and N7 without the papers at hand. The two insertion kinds and
    the end-of-path rule, which is the library's carry-over from N5, are the
    uncertain parts: the paper may restrict the destinations of an insertion further,
    or admit interior insertions that this set leaves out.

    Args:
        shop: The job shop.
        orders: The machine orders the blocks belong to.
        blocks: The blocks of a critical path of ``orders``, as
            :func:`~dispatchatlas.solvers.shop.graph.critical_blocks` returns them.

    Returns:
        The moves, block by block in path order, each by the place of the operation
        moved and the place it takes.
    """
    last = len(blocks) - 1
    moves: list[Move] = []
    if last == 0:
        return moves
    for index, block in enumerate(blocks):
        size = len(block)
        if size < 2:
            continue
        machine = shop.machine[block[0]]
        start = orders[machine].index(block[0])
        ends = (0, size - 1)
        for source in range(size):
            for target in range(size):
                if source == target or target == source - 1:
                    continue
                if source not in ends and target not in ends:
                    continue
                if index == 0 and ends[1] not in (source, target):
                    continue
                if index == last and ends[0] not in (source, target):
                    continue
                moves.append((machine, start + source, start + target))
    return moves


def _moved(orders: Sequence[Sequence[int]], move: Move) -> list[Sequence[int]]:
    """Return the machine orders after a move, sharing the machines it leaves alone."""
    machine, source, target = move
    moved = list(orders[machine])
    moved.insert(target, moved.pop(source))
    changed = list(orders)
    changed[machine] = moved
    return changed


def _crossed(order: Sequence[int], source: int, target: int) -> tuple[tuple[int, int], ...]:
    """Return the arcs a move reverses, each as the pair of its operations before the move."""
    moved = order[source]
    if source < target:
        return tuple((moved, passed) for passed in order[source + 1 : target + 1])
    return tuple((passed, moved) for passed in order[target:source])


def swap_estimate(
    shop: JobShop,
    order: Sequence[int],
    place: int,
    heads: Sequence[float],
    tails: Sequence[float],
) -> float:
    """Estimate the makespan after swapping two adjacent operations of one machine.

    The operation ``u`` at ``place`` and the operation ``v`` after it trade places, so
    that ``v`` comes first. The estimate recomputes the heads and tails of ``u`` and
    ``v`` alone, from the heads and tails of their neighbours before the swap, and
    so takes constant time.

    In this library a head is the earliest start of an operation, the longest path
    into it with its release time counted, and a tail is the longest path that
    leaves it, from its completion. Neither includes the operation's own processing
    time ``p``, so head, ``p`` and tail add up to the longest path through the
    operation. Write ``r`` for a head and ``q`` for a tail, PJ and SJ for the job
    predecessor and successor, PM for the machine predecessor of ``u``, SM for the
    machine successor of ``v``, and let a neighbour that does not exist add nothing.
    Then, with the release time of ``v`` and ``u`` as a floor on ``r'(v)`` and
    ``r'(u)``:

    - ``r'(v) = max(r(PJ(v)) + p(PJ(v)), r(PM(u)) + p(PM(u)))``,
    - ``r'(u) = max(r(PJ(u)) + p(PJ(u)), r'(v) + p(v))``,
    - ``q'(u) = max(q(SJ(u)) + p(SJ(u)), q(SM(v)) + p(SM(v)))``,
    - ``q'(v) = max(q(SJ(v)) + p(SJ(v)), q'(u) + p(u))``,
    - the estimate is ``max(r'(v) + p(v) + q'(v), r'(u) + p(u) + q'(u))``.

    When the swapped orders have no cycle, these are the exact heads and tails of
    ``u`` and ``v`` after the swap, so the estimate is the length of the longest path
    through either of them and never exceeds the makespan of the new orders. It
    falls short of it when a longest path avoids both. Swapping adjacent operations
    of a critical path leaves no cycle unless a job arc joins them.

    Args:
        shop: The job shop.
        order: The machine order that holds ``u`` and ``v``.
        place: The place of ``u`` in ``order``; ``v`` is at ``place + 1``.
        heads: The heads of the operations before the swap, as
            :func:`~dispatchatlas.solvers.shop.graph.heads_tails` returns them.
        tails: The tails of the operations before the swap.

    Returns:
        The estimate, or infinity when the job of ``u`` also leads straight to
        ``v``, so that the swap would close a cycle.

    Raises:
        ValueError: If ``place`` and ``place + 1`` are not both places of ``order``.
    """
    if not 0 <= place < len(order) - 1:
        raise ValueError(f"place {place} has no operation after it in an order of {len(order)}")
    u, v = order[place], order[place + 1]
    successor = shop.successor
    if successor[u] == v:
        return math.inf
    duration, release, predecessor = shop.duration, shop.release, shop.predecessor
    time_u, time_v = duration[u], duration[v]
    head_v = release[v]
    for before in (predecessor[v], order[place - 1] if place else -1):
        if before >= 0:
            head_v = max(head_v, heads[before] + duration[before])
    head_u = max(release[u], head_v + time_v)
    if predecessor[u] >= 0:
        head_u = max(head_u, heads[predecessor[u]] + duration[predecessor[u]])
    tail_u = 0.0
    for after in (successor[u], order[place + 2] if place + 2 < len(order) else -1):
        if after >= 0:
            tail_u = max(tail_u, tails[after] + duration[after])
    tail_v = tail_u + time_u
    if successor[v] >= 0:
        tail_v = max(tail_v, tails[successor[v]] + duration[successor[v]])
    return max(head_v + time_v + tail_v, head_u + time_u + tail_u)


def critical_block_descent(
    shop: JobShop,
    orders: Sequence[Sequence[int]],
    run: ShopRun[Any],
    *,
    parent: Evaluated[Any] | None = None,
    count_neighbours: bool = False,
) -> Descent:
    """Improve machine orders by best-improvement descent over the N5 neighbourhood.

    Each step traces one critical path of the orders, ranks its N5 swaps
    (:func:`n5_moves`) by their estimates (:func:`swap_estimate`), and recomputes the
    exact heads of the swaps in that order. The estimate never exceeds the exact
    makespan, so a swap whose estimate is not below the best makespan so far cannot
    beat it, and the scan stops there. The step applies the swap of the lowest exact
    makespan when that is below the makespan of the current orders, and the descent
    ends when no swap is. Every applied swap has been checked exactly.

    Each applied swap counts as one evaluation, as in :class:`ShopTabuSearch`: its
    schedule is built with :func:`~dispatchatlas.solvers.shop.graph.decoded_from_heads`
    and counted through the run's search, which updates the incumbent and the trace,
    and the run's observer sees it as a move of the orders before it. The orders the
    descent starts from are not counted, since the caller has counted the schedule
    they came from, and neither are the estimates and the exact recomputations of
    swaps that were ranked and not applied. A run at its budget raises
    :class:`~dispatchatlas.solvers.search.Exhausted` from the first swap the descent
    would apply, as any evaluation does.

    With ``count_neighbours`` the descent counts every neighbour it evaluates instead:
    each swap whose exact heads it recomputes is built and counted as one evaluation
    through the run's search, which updates the incumbent and the trace with it, and the
    observer sees it as a move of the orders the step starts from. The swap a step
    applies is the counted neighbour of the lowest makespan, which is not counted again.
    The swaps it applies are the same as without the option; only the counting differs.
    The estimates are not counted, and neither is a swap whose exact heads reveal a
    cycle, which builds no schedule. A run at its budget raises
    :class:`~dispatchatlas.solvers.search.Exhausted` from the first neighbour the descent
    would evaluate, so an evaluation limit bounds the work of the descent.

    Args:
        shop: The job shop.
        orders: The machine orders to improve, as
            :func:`~dispatchatlas.solvers.shop.graph.machine_orders` returns them.
        run: The run whose search counts the evaluations; it must belong to ``shop``.
        parent: The evaluation that ``orders`` come from, which the observer compares
            the first applied swap, or every neighbour of the first step, with, or
            ``None``.
        count_neighbours: Count every neighbour evaluated exactly, not only the
            swaps applied.

    Returns:
        The final orders, their semi-active schedule, its makespan, and the number of
        swaps applied, which without ``count_neighbours`` is the number of evaluations
        counted. With no swap applied, the schedule is built from the heads of
        ``orders``, uncounted.

    Raises:
        ValueError: If ``run`` belongs to another job shop.
        ModelError: If ``orders`` do not list every operation once on its machine,
            or form a cycle with the jobs.
        Exhausted: When the budget of the run ends before a swap is counted.
    """
    if run.shop is not shop:
        raise ValueError("the run belongs to another job shop")
    position = _position(shop, orders)
    if position is None:
        raise ModelError("the machine orders and the jobs form a cycle")
    moves = 0
    evaluated: Evaluated[Any] | None = None
    if count_neighbours:
        while (counted := _best_counted_swap(run, position, parent)) is not None:
            position, evaluated = counted
            parent, moves = evaluated, moves + 1
    else:
        while (better := _best_swap(shop, position)) is not None:
            evaluated = _count(run, better, parent)
            parent, position, moves = evaluated, better, moves + 1
    decoded = decoded_from_heads(shop, position.heads) if evaluated is None else evaluated.decoded
    return Descent(
        tuple(tuple(order) for order in position.orders), decoded, position.makespan, moves
    )


@dataclass(frozen=True, slots=True)
class _Position:
    """Machine orders with their exact heads, tails and makespan.

    The orders are never changed in place: a move builds new orders, which share the
    machines it leaves alone.
    """

    orders: Sequence[Sequence[int]]
    heads: tuple[float, ...]
    tails: tuple[float, ...]
    makespan: float


def _position(shop: JobShop, orders: Sequence[Sequence[int]]) -> _Position | None:
    """Return the exact position of machine orders, or ``None`` when they form a cycle."""
    result = heads_tails(shop, orders)
    if result is None:
        return None
    heads, tails = result
    return _Position(orders, heads, tails, max(map(operator.add, heads, shop.duration)))


def _evaluated(
    shop: JobShop, position: _Position, score: Score, decoded: Decoded
) -> Evaluated[Any]:
    """Return a counted schedule as the evaluation of its machine orders, for the observer."""
    jobs = np.asarray(shop.job, dtype=np.intp)[np.asarray(decoded.sequence, dtype=np.intp)]
    return Evaluated(position.orders, jobs, score, decoded)


def _observe(run: ShopRun[Any], evaluated: Evaluated[Any], parent: Evaluated[Any] | None) -> None:
    """Show an evaluation to the run's observer, and time the observer apart from the search."""
    observer = run.observer
    if observer is not None:
        started = time.perf_counter()
        observer.observe(evaluated, parent)
        run.observer_seconds += time.perf_counter() - started


def _count(run: ShopRun[Any], position: _Position, parent: Evaluated[Any] | None) -> Evaluated[Any]:
    """Count the schedule of a position as one evaluation and show it to the observer.

    The search checks its budget first, so a run at its limit builds nothing, and the
    time of the build counts in the run's ``decode_seconds``. A schedule that reaches
    the target is counted before the search stops, and the observer sees it too.
    """
    shop, search = run.shop, run.search
    built: list[Decoded] = []

    def build() -> Decoded:
        started = time.perf_counter()
        decoded = decoded_from_heads(shop, position.heads)
        run.decode_seconds += time.perf_counter() - started
        built.append(decoded)
        return decoded

    try:
        candidate = search.assess_built(build)
    except Exhausted as stop:
        best = search.best
        if stop.reason is StopReason.TARGET_REACHED and built and best is not None:
            _observe(run, _evaluated(shop, position, best.score, best.decoded), parent)
        raise
    evaluated = _evaluated(shop, position, candidate.score, candidate.decoded)
    _observe(run, evaluated, parent)
    return evaluated


class _Candidate(NamedTuple):
    """A move of the neighbourhood, with its value and the arcs it would reverse.

    The value is the estimate of an N5 move and the exact makespan of an N7 move,
    whose position is then known already.
    """

    move: Move
    value: float
    position: _Position | None
    arcs: tuple[tuple[int, int], ...]


class _TabuList:
    """The arcs that recent moves created, each forbidden to reverse for its tenure.

    An entry is the arc as it stands after a move, the pair of its operations in the
    order the machine now processes them, stamped with the step of the move.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[int, int], tuple[int, int]] = {}

    def forbid(self, crossed: Sequence[tuple[int, int]], step: int, tenure: int) -> None:
        """Add the reversed arcs of the move of ``step``, tabu for the next ``tenure`` steps."""
        for before, after in crossed:
            self._entries[after, before] = (step + tenure, step)

    def oldest(self, crossed: Sequence[tuple[int, int]], step: int) -> int | None:
        """Return the step of the oldest live entry among the arcs a move would reverse."""
        stamps = [
            entry[1]
            for arc in crossed
            if (entry := self._entries.get(arc)) is not None and entry[0] >= step
        ]
        return min(stamps) if stamps else None

    def expire(self, step: int) -> None:
        """Drop the entries that are no longer live at ``step``."""
        self._entries = {arc: entry for arc, entry in self._entries.items() if entry[0] >= step}

    def clear(self) -> None:
        """Forget every entry."""
        self._entries.clear()


type _Kind = Literal["free", "aspired", "forced"]
"""Why a move is applied: it is not tabu, aspiration admits it, or every move is tabu."""


def _start_position(shop: JobShop, decoded: Decoded) -> _Position:
    """Return the position of the machine orders of a schedule."""
    position = _position(shop, machine_orders(shop, decoded))
    if position is None:
        raise SolverError("the machine orders of a schedule of a builder hold a cycle")
    return position


def _critical_swaps(shop: JobShop, position: _Position) -> list[Move]:
    """List the swaps of adjacent operations inside the blocks of a critical path."""
    swaps: list[Move] = []
    for block in critical_blocks(shop, position.orders, position.heads):
        machine = shop.machine[block[0]]
        order = position.orders[machine]
        start = order.index(block[0])
        for place in range(start, start + len(block) - 1):
            if shop.successor[order[place]] != order[place + 1]:
                swaps.append((machine, place, place + 1))
    return swaps


def _ranked_swaps(shop: JobShop, position: _Position) -> list[tuple[float, Move]]:
    """Return the N5 swaps of a critical path of a position with their estimates, lowest first."""
    orders = position.orders
    ranked: list[tuple[float, Move]] = []
    for move in n5_moves(shop, orders, critical_blocks(shop, orders, position.heads)):
        machine, place, _ = move
        ranked.append(
            (swap_estimate(shop, orders[machine], place, position.heads, position.tails), move)
        )
    ranked.sort()
    return ranked


def _best_swap(shop: JobShop, position: _Position) -> _Position | None:
    """Return the orders of the N5 swap that lowers the makespan most, or ``None``.

    The swaps are ranked by their estimates, and a swap is recomputed exactly only
    while its estimate is below the best makespan found, the current one at first.
    The first of equal swaps in that order wins.
    """
    best: _Position | None = None
    bound = position.makespan
    for estimate, move in _ranked_swaps(shop, position):
        if estimate >= bound:
            break
        following = _position(shop, _moved(position.orders, move))
        if following is not None and following.makespan < bound:
            best, bound = following, following.makespan
    return best


def _best_counted_swap(
    run: ShopRun[Any], position: _Position, parent: Evaluated[Any] | None
) -> tuple[_Position, Evaluated[Any]] | None:
    """Return the swap that :func:`_best_swap` picks, counting every neighbour it recomputes.

    Each swap recomputed exactly, in the order and with the bound of :func:`_best_swap`,
    is counted as one evaluation and shown to the observer as a move of ``parent``.

    Returns:
        The orders of the best swap with its counted evaluation, or ``None`` when no
        swap lowers the makespan.
    """
    shop = run.shop
    best: tuple[_Position, Evaluated[Any]] | None = None
    bound = position.makespan
    for estimate, move in _ranked_swaps(shop, position):
        if estimate >= bound:
            break
        following = _position(shop, _moved(position.orders, move))
        if following is None:
            continue
        evaluated = _count(run, following, parent)
        if following.makespan < bound:
            best, bound = (following, evaluated), following.makespan
    return best


def _check_integer(name: str, value: object, least: int) -> None:
    """Raise :class:`ValueError` unless ``value`` is an integer of at least ``least``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ValueError(f"{name} must be an integer of at least {least}, got {value!r}")


@dataclass(slots=True)
class _Tally:
    """What a walk did, for the extras of the result."""

    moves: int = 0
    restarts: int = 0
    aspirations: int = 0
    forced: int = 0


@dataclass(frozen=True)
class ShopTabuSearch:
    """Walk the machine orders of a job shop by tabu search, from one random start.

    The run evaluates one random job sequence with the active builder and walks from
    its machine orders, as the module describes: each iteration applies the best
    admissible move of the neighbourhood, which counts as one evaluation. Every
    default below is the library's choice, not a value from a paper.

    The start and the random swaps of a restart come from the run's operator stream,
    and the tenures and the tie-breaks from its control stream. Moves of equal
    value, an estimate in N5 and an exact makespan in N7, are told apart by one draw
    from the control stream. The search steers by the makespan whatever the
    objective of the problem, and the result is the incumbent the search keeps by
    that objective. The result's extras add ``moves``, the moves applied,
    ``restarts``, ``aspirations``, the tabu moves that aspiration admitted and
    applied, and ``forced_moves``, the moves applied because every move was tabu.
    The run's ``decode_seconds`` counts the schedules built from heads; the
    estimates, the exact recomputation of the heads of a move and the scoring of
    the moves of N7 are search time. A shop of one job has one schedule, which the
    run evaluates once.

    Attributes:
        neighbourhood: ``"n5"`` (the default), the swaps at the ends of the blocks of
            one critical path, ranked by their estimates (:func:`n5_moves`), or
            ``"n7"``, which adds insertions and scores every move exactly
            (:func:`n7_moves`).
        tenure_low: The least tenure of a tabu entry, in moves; 8 by default.
        tenure_high: The greatest tenure; 12 by default. Each applied move draws its
            tenure uniformly from ``tenure_low`` to ``tenure_high``, both included.
        max_stagnation: The moves without a new best makespan after which the walk
            restarts from the best orders; 200 by default.
        restart_swaps: The random swaps on a critical path that a restart applies to
            the best orders; 3 by default. When no swap exists, the restart evaluates a
            random job sequence instead.
    """

    neighbourhood: NeighbourhoodName = "n5"
    tenure_low: int = 8
    tenure_high: int = 12
    max_stagnation: int = 200
    restart_swaps: int = 3

    info: ClassVar[SolverInfo] = SolverInfo(
        id="shop-tabu-search",
        name="Job-shop tabu search",
        family="local-search",
        summary="Moves between the machine orders of a job shop by swapping operations at the "
        "ends of the blocks of a critical path, forbids the swaps that undo recent ones, and "
        "restarts from the best orders after a stagnation.",
        references=("nowicki1996fast", "zhang2007tabu", "balas1998guided"),
        supports=JOB_SHOP_FEATURES,
        shape="job-shop",
    )

    def __post_init__(self) -> None:
        if self.neighbourhood not in NEIGHBOURHOODS:
            raise ValueError(
                f"unknown neighbourhood {self.neighbourhood!r}; expected one of "
                f"{list(NEIGHBOURHOODS)}"
            )
        _check_integer("tenure_low", self.tenure_low, 1)
        _check_integer("tenure_high", self.tenure_high, self.tenure_low)
        _check_integer("max_stagnation", self.max_stagnation, 1)
        _check_integer("restart_swaps", self.restart_swaps, 1)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Walk the machine orders of a random start until a stop criterion fires."""
        search = Search(problem, stop, seed, self.info)
        shop = JobShop.from_problem(problem)
        representation = make_representation(shop, "sequence")
        run = ShopRun(search, shop, "active", representation, TrajectoryObserver(shop))
        tally = _Tally()
        with search:
            start = run.evaluate(representation.initial(run.operators))
            if shop.jobs > 1:
                self._walk(run, start, tally)
        run.finish()
        search.extras.update(
            moves=tally.moves,
            restarts=tally.restarts,
            aspirations=tally.aspirations,
            forced_moves=tally.forced,
        )
        return search.result()

    def _walk(self, run: ShopRun[Any], start: Evaluated[Any], tally: _Tally) -> None:
        """Walk from the machine orders of ``start`` until the search is exhausted."""
        shop, search, control = run.shop, run.search, run.control
        current = _start_position(shop, start.decoded)
        parent, best, best_parent = start, current, start
        tabu = _TabuList()
        stagnation = step = 0
        while True:
            search.next_iteration()
            step += 1
            tabu.expire(step)
            chosen = self._choose(run, current, best.makespan, tabu, step)
            if chosen is None:
                stagnation = self.max_stagnation
            else:
                candidate, following, kind = chosen
                parent = _count(run, following, parent)
                tenure = int(control.integers(self.tenure_low, self.tenure_high + 1))
                tabu.forbid(candidate.arcs, step, tenure)
                tally.moves += 1
                tally.aspirations += kind == "aspired"
                tally.forced += kind == "forced"
                current = following
                if current.makespan < best.makespan:
                    best, best_parent, stagnation = current, parent, 0
                else:
                    stagnation += 1
            if stagnation >= self.max_stagnation:
                current, parent = self._restart(run, best, best_parent)
                tabu.clear()
                tally.restarts += 1
                stagnation = 0
                if current.makespan < best.makespan:
                    best, best_parent = current, parent

    def _candidates(self, run: ShopRun[Any], current: _Position) -> list[_Candidate]:
        """List the moves of the neighbourhood at ``current``, each with its value.

        An N5 move is valued by its estimate, and one whose swap a job arc blocks is
        left out. An N7 move is valued by the makespan of the orders it leads to, and
        one that leaves a cycle is left out.
        """
        shop, orders = run.shop, current.orders
        blocks = critical_blocks(shop, orders, current.heads)
        candidates: list[_Candidate] = []
        if self.neighbourhood == "n5":
            for move in n5_moves(shop, orders, blocks):
                machine, place, target = move
                value = swap_estimate(shop, orders[machine], place, current.heads, current.tails)
                if value < math.inf:
                    arcs = _crossed(orders[machine], place, target)
                    candidates.append(_Candidate(move, value, None, arcs))
            return candidates
        for move in n7_moves(shop, orders, blocks):
            run.search.check()
            position = _position(shop, _moved(orders, move))
            if position is not None:
                arcs = _crossed(orders[move[0]], move[1], move[2])
                candidates.append(_Candidate(move, position.makespan, position, arcs))
        return candidates

    @staticmethod
    def _select(
        candidates: Sequence[_Candidate],
        best_makespan: float,
        tabu: _TabuList,
        step: int,
        control: np.random.Generator,
    ) -> tuple[int, _Kind]:
        """Return the index of the move to apply and why it is admitted.

        The move of least value among those that are not tabu, or tabu with a value
        below the best makespan, wins, and a draw from ``control`` breaks a tie of
        values. When no move is admitted, the move whose oldest tabu entry is the
        oldest wins, the first of them in list order.
        """
        ages = [tabu.oldest(candidate.arcs, step) for candidate in candidates]
        admitted = [
            index
            for index, candidate in enumerate(candidates)
            if ages[index] is None or candidate.value < best_makespan
        ]
        if not admitted:
            oldest = min((age, index) for index, age in enumerate(ages) if age is not None)
            return oldest[1], "forced"
        least = min(candidates[index].value for index in admitted)
        ties = [index for index in admitted if candidates[index].value == least]
        chosen = ties[0] if len(ties) == 1 else ties[int(control.integers(len(ties)))]
        return chosen, "free" if ages[chosen] is None else "aspired"

    def _choose(
        self,
        run: ShopRun[Any],
        current: _Position,
        best_makespan: float,
        tabu: _TabuList,
        step: int,
    ) -> tuple[_Candidate, _Position, _Kind] | None:
        """Return the move to apply, the orders it leads to, and why it is admitted.

        A move whose exact orders turn out to hold a cycle, which only operations
        of no duration allow on a critical path, is dropped and the next best is
        taken. The result is ``None`` when no move is left.
        """
        candidates = self._candidates(run, current)
        while candidates:
            index, kind = self._select(candidates, best_makespan, tabu, step, run.control)
            candidate = candidates.pop(index)
            position = candidate.position
            if position is None:
                position = _position(run.shop, _moved(current.orders, candidate.move))
            if position is not None:
                return candidate, position, kind
        return None

    def _restart(
        self, run: ShopRun[Any], best: _Position, best_parent: Evaluated[Any]
    ) -> tuple[_Position, Evaluated[Any]]:
        """Return the position that a restart starts from, counted as one evaluation.

        It is the best orders after ``restart_swaps`` swaps of adjacent operations of
        a critical path, each drawn uniformly from the operator stream, or a random
        job sequence of the run when the best orders offer no swap.
        """
        shop, operators = run.shop, run.operators
        position = best
        for _ in range(self.restart_swaps):
            swaps = _critical_swaps(shop, position)
            if not swaps:
                break
            following = _position(
                shop, _moved(position.orders, swaps[int(operators.integers(len(swaps)))])
            )
            if following is not None:
                position = following
        if position is best:
            fresh = run.evaluate(run.representation.initial(operators))
            return _start_position(shop, fresh.decoded), fresh
        return position, _count(run, position, best_parent)
