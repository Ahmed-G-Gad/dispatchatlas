"""Variation operators on task orders and mode vectors.

Every operator returns a new list and leaves its inputs untouched. Orders need
not respect precedence: the schedule builder repairs them.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from dispatchatlas.core.model import Problem

__all__ = [
    "insert",
    "mutate_modes",
    "order_crossover",
    "partially_mapped_crossover",
    "random_insert",
    "random_modes",
    "random_order",
    "random_swap",
    "random_topological_order",
    "reverse",
    "swap",
    "uniform_mode_crossover",
]


def swap(order: Sequence[int], i: int, j: int) -> list[int]:
    """Return ``order`` with the elements at positions ``i`` and ``j`` exchanged."""
    result = list(order)
    result[i], result[j] = result[j], result[i]
    return result


def insert(order: Sequence[int], i: int, j: int) -> list[int]:
    """Return ``order`` with the element at position ``i`` moved to position ``j``."""
    result = list(order)
    result.insert(j, result.pop(i))
    return result


def reverse(order: Sequence[int], i: int, j: int) -> list[int]:
    """Return ``order`` with the segment between positions ``i`` and ``j`` reversed."""
    low, high = min(i, j), max(i, j)
    result = list(order)
    result[low : high + 1] = reversed(result[low : high + 1])
    return result


def random_swap(order: Sequence[int], rng: random.Random) -> list[int]:
    """Exchange two distinct random positions."""
    if len(order) < 2:
        return list(order)
    i, j = rng.sample(range(len(order)), 2)
    return swap(order, i, j)


def random_insert(order: Sequence[int], rng: random.Random) -> list[int]:
    """Move a random element to a different random position."""
    if len(order) < 2:
        return list(order)
    i, j = rng.sample(range(len(order)), 2)
    return insert(order, i, j)


def random_order(problem: Problem, rng: random.Random) -> list[int]:
    """Return a uniformly random permutation of the task indices."""
    order = list(range(problem.size))
    rng.shuffle(order)
    return order


def random_topological_order(problem: Problem, rng: random.Random) -> list[int]:
    """Return a random precedence-feasible order.

    At each step a task is drawn uniformly from those whose predecessors are
    all placed.
    """
    waiting = [len(items) for items in problem.predecessors]
    ready = [i for i, count in enumerate(waiting) if count == 0]
    order: list[int] = []
    while ready:
        task = ready.pop(rng.randrange(len(ready)))
        order.append(task)
        for following in problem.successors[task]:
            waiting[following] -= 1
            if waiting[following] == 0:
                ready.append(following)
    return order


def order_crossover(first: Sequence[int], second: Sequence[int], rng: random.Random) -> list[int]:
    """Order crossover (OX) of Davis (1985).

    A random slice is copied from ``first``; the remaining positions receive
    the missing elements in the order they appear in ``second``, starting after
    the slice and wrapping around.
    """
    count = len(first)
    if count < 2:
        return list(first)
    i, j = sorted(rng.sample(range(count), 2))
    child: list[int | None] = [None] * count
    child[i : j + 1] = first[i : j + 1]
    kept = set(first[i : j + 1])
    fill = [gene for gene in list(second[j + 1 :]) + list(second[: j + 1]) if gene not in kept]
    position = (j + 1) % count
    for gene in fill:
        child[position] = gene
        position = (position + 1) % count
    return [gene for gene in child if gene is not None]


def partially_mapped_crossover(
    first: Sequence[int], second: Sequence[int], rng: random.Random
) -> list[int]:
    """Partially mapped crossover (PMX) of Goldberg and Lingle (1985)."""
    count = len(first)
    if count < 2:
        return list(first)
    i, j = sorted(rng.sample(range(count), 2))
    child = list(second)
    child[i : j + 1] = first[i : j + 1]
    mapping = {first[k]: second[k] for k in range(i, j + 1)}
    segment = set(first[i : j + 1])
    for k in list(range(i)) + list(range(j + 1, count)):
        gene = second[k]
        while gene in segment:
            gene = mapping[gene]
        child[k] = gene
    return child


def random_modes(problem: Problem, rng: random.Random) -> list[int]:
    """Return a uniformly random mode index for every task."""
    return [rng.randrange(len(task.modes)) for task in problem.tasks]


def mutate_modes(
    problem: Problem, modes: Sequence[int], rng: random.Random, rate: float
) -> list[int]:
    """Redraw each task's mode with probability ``rate``, among its other modes."""
    result = list(modes)
    for i, task in enumerate(problem.tasks):
        count = len(task.modes)
        if count > 1 and rng.random() < rate:
            choice = rng.randrange(count - 1)
            result[i] = choice if choice < result[i] else choice + 1
    return result


def uniform_mode_crossover(
    first: Sequence[int], second: Sequence[int], rng: random.Random
) -> list[int]:
    """Take each task's mode from either parent with equal probability."""
    return [a if rng.random() < 0.5 else b for a, b in zip(first, second, strict=True)]
