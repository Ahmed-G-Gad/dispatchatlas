"""Representations that map search vectors to task orders and mode choices.

Continuous optimisers (differential evolution, particle swarms, CMA-ES, and
the population methods in :mod:`dispatchatlas.solvers.swarm`) search over
real-valued vectors. These helpers turn a vector into a task order and mode
choices, which the shared schedule builder then decodes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from dispatchatlas.core.model import Problem

__all__ = [
    "VectorEncoding",
    "keys_from_order",
    "largest_value_order",
    "modes_from_keys",
    "random_key_order",
]


def random_key_order(keys: Sequence[float]) -> list[int]:
    """Return task indices sorted by ascending key, ties by index.

    This is the random-key representation of Bean (1994),
    doi:10.1287/ijoc.6.2.154, also called the smallest-position-value rule.
    """
    return sorted(range(len(keys)), key=lambda i: (keys[i], i))


def largest_value_order(keys: Sequence[float]) -> list[int]:
    """Return task indices sorted by descending key, ties by index.

    This is the largest-order-value rule used by many swarm scheduling methods.
    """
    return sorted(range(len(keys)), key=lambda i: (-keys[i], i))


def keys_from_order(order: Sequence[int]) -> list[float]:
    """Return keys in ``[0, 1)`` whose ascending sort reproduces ``order``.

    Useful to seed a continuous population with a heuristic order.
    """
    count = len(order)
    keys = [0.0] * count
    for rank, task in enumerate(order):
        keys[task] = rank / count
    return keys


def modes_from_keys(problem: Problem, keys: Sequence[float]) -> list[int]:
    """Map one key in ``[0, 1)`` per task to a mode index.

    Key ``k`` selects mode ``floor(k * m)`` of a task with ``m`` modes. Keys
    outside ``[0, 1)`` are wrapped into it first, so any real vector is valid.
    """
    modes: list[int] = []
    for task, key in zip(problem.tasks, keys, strict=True):
        fraction = key - math.floor(key)
        count = len(task.modes)
        modes.append(min(count - 1, int(fraction * count)))
    return modes


class VectorEncoding:
    """Decodes real vectors into task orders and mode choices for one problem.

    A vector holds one key per task, then, when modes come from keys, one key
    per task with several modes. The task keys give the order: ascending
    under the ``"random-key"`` rule (:func:`random_key_order`), descending
    under the ``"largest-value"`` rule (:func:`largest_value_order`). Each mode
    key selects its task's mode as :func:`modes_from_keys` does. With
    ``modes="builder"`` the vector holds task keys only and the schedule
    builder gives each task its earliest-finishing mode, which suits
    time-based objectives but ignores modes that save energy or cost. Search
    in the unit hypercube: every real vector decodes, because orders only
    compare keys and mode keys wrap.

    Args:
        problem: The problem.
        rule: How task keys become an order.
        modes: ``"keys"`` to search mode choices, ``"builder"`` to leave them
            to the schedule builder.

    Attributes:
        problem: The problem.
        rule: The order rule.
        modes: Where mode choices come from.
        dimension: Length of the vectors.
        flexible: Indices of the tasks whose mode a key selects, in key order.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        rule: Literal["random-key", "largest-value"] = "random-key",
        modes: Literal["keys", "builder"] = "keys",
    ) -> None:
        if rule not in ("random-key", "largest-value"):
            raise ValueError(f"unknown order rule {rule!r}")
        if modes not in ("keys", "builder"):
            raise ValueError(f"unknown mode source {modes!r}")
        self.problem = problem
        self.rule = rule
        self.modes = modes
        self.flexible: tuple[int, ...] = (
            tuple(i for i, task in enumerate(problem.tasks) if len(task.modes) > 1)
            if modes == "keys"
            else ()
        )
        self.dimension = problem.size + len(self.flexible)

    def decode(self, vector: Sequence[float]) -> tuple[list[int], list[int] | None]:
        """Return the task order and the modes, ``None`` when the builder chooses them.

        Raises:
            ValueError: If the vector length differs from :attr:`dimension`.
        """
        if len(vector) != self.dimension:
            raise ValueError(f"expected a vector of length {self.dimension}, got {len(vector)}")
        size = self.problem.size
        keys = vector[:size]
        order = random_key_order(keys) if self.rule == "random-key" else largest_value_order(keys)
        if not self.flexible:
            return order, None
        modes = [0] * size
        tasks = self.problem.tasks
        for task, key in zip(self.flexible, vector[size:], strict=True):
            fraction = key - math.floor(key)
            count = len(tasks[task].modes)
            modes[task] = min(count - 1, int(fraction * count))
        return order, modes

    def encode(self, order: Sequence[int], modes: Sequence[int] | None = None) -> list[float]:
        """Return a vector in the unit hypercube that decodes to ``order`` and ``modes``.

        Useful to seed a population with a heuristic solution. A task whose
        mode is ``None`` or negative gets the key of mode 0. Modes are ignored
        when the builder chooses them.
        """
        keys = keys_from_order(order)
        if self.rule == "largest-value":
            keys = [1.0 - key for key in keys]
        tasks = self.problem.tasks
        for task in self.flexible:
            count = len(tasks[task].modes)
            mode = 0 if modes is None else max(0, modes[task])
            keys.append((mode + 0.5) / count)
        return keys
