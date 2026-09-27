"""Probabilistic construction of item sequences, shared by ant colony and scatter search."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from dispatchatlas.solvers.constructive import Items

__all__ = ["construct", "roulette"]


def roulette(weights: Sequence[float], rng: random.Random) -> int:
    """Return an index drawn with probability proportional to its weight.

    Draws uniformly when the weights do not sum to a positive finite number.
    """
    total = sum(weights)
    if not (total > 0.0 and math.isfinite(total)):
        return rng.randrange(len(weights))
    threshold = rng.random() * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += weight
        if threshold < running:
            return index
    return len(weights) - 1


def construct(items: Items, weight: Callable[[int, int], float], rng: random.Random) -> list[int]:
    """Build a precedence-feasible item sequence one position at a time.

    At each position the next item is drawn among the *eligible* items, those
    whose predecessors are all placed, with probability proportional to
    ``weight(item, position)``.
    """
    waiting = [len(before) for before in items.predecessors]
    ready = [item for item, count in enumerate(waiting) if count == 0]
    sequence: list[int] = []
    for position in range(items.count):
        item = ready.pop(roulette([weight(candidate, position) for candidate in ready], rng))
        sequence.append(item)
        for after in items.successors[item]:
            waiting[after] -= 1
            if waiting[after] == 0:
                ready.append(after)
    return sequence
