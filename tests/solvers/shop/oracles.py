"""Naive reference implementations that the tests of the job-shop rules compare the library with.

They follow the published procedures line by line and ignore speed: the
Giffler-Thompson oracle scans every candidate at every step, and the insertion
oracle tries every position of every operation and recomputes every longest
path from scratch. They share no code with the library's builders and graph
functions beyond the job-shop view. :func:`job_shop` builds the small job shops
the tests write out by hand.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from dispatchatlas.core import Dependency, Problem, Resource, Task
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.shop.instance import JobShop


def naive_giffler_thompson(
    shop: JobShop, rank: Sequence[float], *, non_delay: bool = False
) -> Decoded:
    """Build the active, or non-delay, schedule of the Giffler-Thompson procedure for ranks.

    A lower rank comes first. Of the operations that can be scheduled next and
    attain the earliest completion time C*, the one of lowest rank fixes the
    machine M*; the conflict set holds it and the operations on M* that can
    start before C*; the one of lowest rank is scheduled at its earliest start.
    With ``non_delay``, the operation of lowest rank among those that attain the
    earliest start S* is scheduled at S*.
    """
    next_step = [0] * shop.jobs
    job_free = [0.0] * shop.jobs
    machine_free = [0.0] * shop.machines
    starts = [0.0] * shop.size
    ends = [0.0] * shop.size
    placed: list[int] = []
    for _ in range(shop.size):
        candidates: list[tuple[int, float]] = []
        for job, route in enumerate(shop.operations):
            if next_step[job] < len(route):
                task = route[next_step[job]]
                start = max(job_free[job], machine_free[shop.machine[task]], shop.release[task])
                candidates.append((task, start))
        if non_delay:
            earliest = min(start for _, start in candidates)
            chosen = min(
                (task for task, start in candidates if start == earliest), key=rank.__getitem__
            )
        else:
            completion = min(start + shop.duration[task] for task, start in candidates)
            fixing = min(
                (task for task, start in candidates if start + shop.duration[task] == completion),
                key=rank.__getitem__,
            )
            conflict = [
                task
                for task, start in candidates
                if shop.machine[task] == shop.machine[fixing]
                and (start < completion or task == fixing)
            ]
            chosen = min(conflict, key=rank.__getitem__)
        start = dict(candidates)[chosen]
        end = start + shop.duration[chosen]
        starts[chosen], ends[chosen] = start, end
        job_free[shop.job[chosen]] = machine_free[shop.machine[chosen]] = end
        next_step[shop.job[chosen]] += 1
        placed.append(chosen)
    return Decoded(tuple(placed), (0,) * shop.size, tuple(starts), tuple(ends))


def _arcs(shop: JobShop, orders: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Return the job arcs of every operation and the machine arcs of consecutive ones."""
    arcs = [(task, shop.successor[task]) for task in range(shop.size) if shop.successor[task] >= 0]
    for order in orders:
        arcs.extend(itertools.pairwise(order))
    return arcs


def naive_heads_tails(
    shop: JobShop, orders: Sequence[Sequence[int]]
) -> tuple[list[float], list[float]] | None:
    """Return every head and tail of the graph of ``orders`` by repeated relaxation, or ``None``.

    The graph holds all job arcs and the machine arcs of ``orders``, which may
    list only some of the operations. A cycle keeps the relaxation from settling.
    """
    arcs = _arcs(shop, orders)
    heads = list(shop.release)
    tails = [0.0] * shop.size
    for _ in range(shop.size + 1):
        changed = False
        for before, after in arcs:
            if heads[before] + shop.duration[before] > heads[after]:
                heads[after] = heads[before] + shop.duration[before]
                changed = True
            if tails[after] + shop.duration[after] > tails[before]:
                tails[before] = tails[after] + shop.duration[after]
                changed = True
        if not changed:
            return heads, tails
    return None


def naive_insertion_costs(
    shop: JobShop, orders: Sequence[Sequence[int]], task: int
) -> dict[int, float]:
    """Return the cost of each position at which ``task`` can join its machine, by brute force.

    For every position in the order of the task's machine, the task is inserted
    there and the heads and tails of the whole graph are recomputed. The position
    is feasible when the graph is acyclic, and its cost is the head, the
    processing time and the tail of the task.
    """
    used = shop.machine[task]
    costs: dict[int, float] = {}
    for position in range(len(orders[used]) + 1):
        trial = [list(order) for order in orders]
        trial[used].insert(position, task)
        paths = naive_heads_tails(shop, trial)
        if paths is not None:
            heads, tails = paths
            costs[position] = heads[task] + shop.duration[task] + tails[task]
    return costs


type Step = tuple[list[list[int]], int, dict[int, float]]
"""One insertion: the machine orders before it, the operation, and the cost of each position."""


def naive_insertion(shop: JobShop) -> tuple[list[list[int]], list[Step]]:
    """Run the basic Werner-Winkler insertion by brute force.

    The job with the greatest total processing time comes first, the lowest job
    index among equals, with its operations in route order on their machines.
    The other operations follow by non-increasing processing time, then job
    index, then operation index. Each goes into the feasible position of its
    machine with the least cost, the earliest position among equals.

    Returns:
        The final machine orders, and every insertion after the first job.
    """
    work = [sum(shop.duration[task] for task in route) for route in shop.operations]
    first = min(range(shop.jobs), key=lambda job: (-work[job], job))
    rest = sorted(
        (task for task in range(shop.size) if shop.job[task] != first),
        key=lambda task: (
            -shop.duration[task],
            shop.job[task],
            shop.operations[shop.job[task]].index(task),
        ),
    )
    orders: list[list[int]] = [[] for _ in range(shop.machines)]
    for task in shop.operations[first]:
        orders[shop.machine[task]].append(task)
    steps: list[Step] = []
    for task in rest:
        costs = naive_insertion_costs(shop, orders, task)
        steps.append(([list(order) for order in orders], task, costs))
        best = min(costs.values())
        orders[shop.machine[task]].insert(min(p for p, cost in costs.items() if cost == best), task)
    return orders, steps


def job_shop(name: str, routes: Sequence[Sequence[tuple[int, float]]], **release: float) -> Problem:
    """Build a job shop: each route lists a job's ``(machine, time)`` pairs in order.

    The task of a job's step is called ``j{job}-o{step}`` and takes the release
    time given for that name, if any.
    """
    tasks: list[Task] = []
    dependencies: list[Dependency] = []
    for job, route in enumerate(routes):
        for step, (machine, time) in enumerate(route):
            task = f"j{job}-o{step}"
            tasks.append(Task.on(task, f"m{machine}", time, release=release.get(task, 0.0)))
            if step:
                dependencies.append(Dependency(f"j{job}-o{step - 1}", task))
    machines = 1 + max(machine for route in routes for machine, _ in route)
    return Problem(
        name,
        tasks=tuple(tasks),
        resources=tuple(Resource(f"m{machine}") for machine in range(machines)),
        dependencies=tuple(dependencies),
    )
