"""The disjunctive graph of a job-shop schedule: machine orders, heads, tails, critical blocks.

The **machine orders** of a schedule list, for each machine, its operations in
processing order. Together with the jobs they fix the disjunctive graph: an arc
leads from every operation to the next operation of its job and to the next
operation on its machine, and its length is the processing time of the
operation it leaves. The orders are feasible exactly when this graph has no
cycle.

- The **head** of an operation is the length of the longest path into it, where
  a path also counts the release time of the operation it starts at: the
  earliest start the orders allow. Starting every operation at its head gives
  the semi-active schedule of the orders, whose makespan is the largest head
  plus processing time.
- The **tail** of an operation is the length of the longest path that leaves it,
  from its completion to the end of the schedule. Head, processing time, and
  tail add up to the makespan exactly for the operations on a longest path, a
  **critical path**.
- A critical path splits into **blocks**: maximal runs of its operations that
  follow each other on one machine. The makespan can fall only when new orders
  reverse two operations of one block, since otherwise the path, or a longer
  one through the same operations, remains.

:func:`machine_orders` reads the orders of a schedule, :func:`heads_tails`
computes the heads and tails of orders, or finds a cycle, :func:`critical_blocks`
splits a critical path, and :func:`decoded_from_heads` turns heads back into a
:class:`~dispatchatlas.solvers.scheduling.Decoded` schedule. Operations are task
indices, as in :class:`~dispatchatlas.solvers.shop.instance.JobShop`.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from itertools import chain

import numpy as np

from dispatchatlas.core.errors import ModelError
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.shop.instance import JobShop

__all__ = ["critical_blocks", "decoded_from_heads", "heads_tails", "machine_orders"]


def machine_orders(shop: JobShop, decoded: Decoded) -> tuple[tuple[int, ...], ...]:
    """Return the order in which each machine processes its operations in a schedule.

    The operations of a machine are listed by start time, then by completion time,
    so an operation that takes no time comes before one that starts at the same
    instant, and then in the order the schedule placed them.

    Args:
        shop: The job shop.
        decoded: A schedule that places every operation of ``shop``.

    Returns:
        One tuple of task indices per machine, in machine order.

    Raises:
        ModelError: If the schedule leaves an operation unplaced.
    """
    size = shop.size
    if len(decoded.sequence) != size:
        raise ModelError(
            f"the schedule places {len(decoded.sequence)} of the {size} operations of "
            f"problem {shop.problem.id!r}"
        )
    placed = np.empty(size, dtype=np.intp)
    placed[np.asarray(decoded.sequence, dtype=np.intp)] = np.arange(size, dtype=np.intp)
    machine = np.asarray(shop.machine, dtype=np.intp)
    order = np.lexsort((placed, np.asarray(decoded.ends), np.asarray(decoded.starts), machine))
    bounds = np.cumsum(np.bincount(machine, minlength=shop.machines))[:-1]
    return tuple(tuple(part.tolist()) for part in np.split(order, bounds))


def heads_tails(
    shop: JobShop, orders: Sequence[Sequence[int]]
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Return the heads and tails of every operation under machine orders.

    Args:
        shop: The job shop.
        orders: One sequence of task indices per machine, in processing order,
            which together list every operation once, each on its own machine.

    Returns:
        The heads and the tails, indexed by task index, or ``None`` when the
        orders and the jobs form a cycle, so that no schedule follows them.

    Raises:
        ModelError: If ``orders`` does not list every operation once on its
            machine.
    """
    before, after = _machine_neighbours(shop, orders)
    predecessor, successor = shop.predecessor, shop.successor
    duration = shop.duration
    size = shop.size
    # Kahn's algorithm: an operation is ready once its job and machine
    # predecessors, where it has them, are both in the order.
    waiting = [
        (job >= 0) + (machine >= 0) for job, machine in zip(predecessor, before, strict=True)
    ]
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
        for previous in (predecessor[task], before[task]):
            if previous >= 0:
                heads[task] = max(heads[task], heads[previous] + duration[previous])
    tails = [0.0] * size
    for task in reversed(order):
        for following in (successor[task], after[task]):
            if following >= 0:
                tails[task] = max(tails[task], tails[following] + duration[following])
    return tuple(heads), tuple(tails)


def critical_blocks(
    shop: JobShop, orders: Sequence[Sequence[int]], heads: Sequence[float]
) -> tuple[tuple[int, ...], ...]:
    """Return the blocks of a critical path of machine orders, in path order.

    The path is traced back from the end of the schedule. It ends at the
    operation of lowest task index among those that complete at the makespan.
    Each operation on it is preceded by its machine predecessor when that one
    completes exactly at the operation's head, and otherwise by its job
    predecessor when that one does; the path starts at an operation whose head
    no predecessor sets, which starts at its release time. A block is a maximal
    run of consecutive operations on the path that machine arcs join, so
    consecutive operations of a block follow each other on their machine, and
    consecutive blocks are joined by a job arc. A block may hold one operation.

    Args:
        shop: The job shop.
        orders: The machine orders, as :func:`heads_tails` takes them.
        heads: The heads of the operations under ``orders``, as
            :func:`heads_tails` returns them.

    Returns:
        The blocks, each a tuple of task indices in processing order, from the
        start of the path to its end.

    Raises:
        ModelError: If ``orders`` does not list every operation once on its
            machine, ``heads`` does not hold one head per operation, or the path
            runs into a cycle, which heads of acyclic orders never let it do.
    """
    before, _ = _machine_neighbours(shop, orders)
    size = shop.size
    if len(heads) != size:
        raise ModelError(f"expected {size} heads, got {len(heads)}")
    predecessor, duration = shop.predecessor, shop.duration
    ends = [heads[task] + duration[task] for task in range(size)]
    task = ends.index(max(ends))
    path = [task]
    # Whether each operation of the path is reached from the one before it by a
    # machine arc, traced from the end of the path.
    by_machine: list[bool] = []
    while True:
        start = heads[task]
        previous = before[task]
        if previous >= 0 and heads[previous] + duration[previous] == start:
            by_machine.append(True)
        else:
            previous = predecessor[task]
            if previous < 0 or heads[previous] + duration[previous] != start:
                break
            by_machine.append(False)
        task = previous
        path.append(task)
        if len(path) > size:
            raise ModelError("the machine orders and the jobs form a cycle")
    path.reverse()
    by_machine.reverse()
    blocks = [[path[0]]]
    for task, joined in zip(path[1:], by_machine, strict=True):
        if joined:
            blocks[-1].append(task)
        else:
            blocks.append([task])
    return tuple(tuple(block) for block in blocks)


def decoded_from_heads(shop: JobShop, heads: Sequence[float]) -> Decoded:
    """Return the schedule that starts every operation at its head.

    Given the heads of feasible machine orders, as :func:`heads_tails` returns
    them, the schedule is the semi-active schedule of the orders. Its sequence
    lists the operations by start time, then by completion time, and then job
    by job in processing order, so that every operation comes after its job
    predecessor.

    Args:
        shop: The job shop.
        heads: The start time of each operation, indexed by task index.

    Returns:
        The schedule.

    Raises:
        ModelError: If ``heads`` does not hold one start time per operation.
    """
    size = shop.size
    starts = np.asarray(heads, dtype=np.float64)
    if starts.shape != (size,):
        raise ModelError(f"expected {size} heads, got shape {starts.shape}")
    ends = starts + np.asarray(shop.duration, dtype=np.float64)
    tasks = np.fromiter(chain.from_iterable(shop.operations), dtype=np.intp, count=size)
    sequence = tasks[np.lexsort((ends[tasks], starts[tasks]))]
    return Decoded(
        tuple(sequence.tolist()), (0,) * size, tuple(starts.tolist()), tuple(ends.tolist())
    )


def _machine_neighbours(
    shop: JobShop, orders: Sequence[Sequence[int]]
) -> tuple[list[int], list[int]]:
    """Return each operation's machine predecessor and successor, ``-1`` where none.

    Raises:
        ModelError: If ``orders`` does not list every operation once on its
            machine.
    """
    if len(orders) != shop.machines:
        raise ModelError(f"expected {shop.machines} machine orders, got {len(orders)}")
    size = shop.size
    machine = shop.machine
    before = [-1] * size
    after = [-1] * size
    listed = [False] * size
    count = 0
    for number, order in enumerate(orders):
        previous = -1
        for entry in order:
            task = operator.index(entry)
            if not 0 <= task < size or listed[task] or machine[task] != number:
                raise ModelError(
                    f"machine order {number} lists task {task}, which is unknown, listed "
                    "twice, or runs on another machine"
                )
            listed[task] = True
            before[task] = previous
            if previous >= 0:
                after[previous] = task
            previous = task
            count += 1
    if count != size:
        raise ModelError(f"the machine orders list {count} of the {size} operations")
    return before, after
