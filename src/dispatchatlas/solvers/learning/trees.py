"""Priority rules as expression trees: the primitive set and the variation operators.

A rule is an expression tree whose leaves are **terminals**, attributes of a
task, and whose inner nodes are binary **functions**. Evaluated on a task, it
gives the task's priority. The function set, including the protected division,
follows the task-scheduling rules that Zaki et al. (2024) evolve, as do the
duration and successor-count terminals:

- ``DUR``: the shortest duration over the task's modes.
- ``SUCC``: the number of the task's direct successors.
- ``REL``: the task's release time.
- ``SLACK``: the due date from
  :func:`~dispatchatlas.solvers.learning.common.due_dates` minus the release
  time minus the shortest duration.
- ``DEPTH``: the number of dependencies on the longest chain of dependencies
  that ends at the task (:func:`task_depths`).
- ``ADD``, ``SUB``, ``MUL``, ``MAX``, ``MIN``, and ``DIV``, the protected
  division, whose value is 1 wherever the divisor is 0.

Trees are evaluated on every task at once, over NumPy columns. The operators
are those of Koza (1992): ramped half-and-half initialisation, subtree
crossover, and subtree mutation. Nodes are addressed by their pre-order index.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from dispatchatlas.solvers.learning.common import due_dates
from dispatchatlas.solvers.priorities import min_durations

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dispatchatlas.core.model import Problem

__all__ = [
    "FUNCTIONS",
    "GROW_TERMINAL_RATE",
    "INITIAL_DEPTHS",
    "OPERATIONS",
    "TERMINALS",
    "Column",
    "Tree",
    "crossover",
    "depth",
    "evaluate",
    "grow",
    "mutate",
    "protected_division",
    "ramped_half_and_half",
    "replace",
    "size",
    "subtree",
    "task_depths",
    "terminal_columns",
    "text",
]

type Tree = str | tuple[str, Tree, Tree]
"""A terminal name, or a function name with its two argument trees."""

type Column = NDArray[np.float64]
"""One value per task."""

TERMINALS: tuple[str, ...] = ("DUR", "SUCC", "REL", "SLACK", "DEPTH")
"""Terminal names, in the order random draws index them."""

FUNCTIONS: tuple[str, ...] = ("ADD", "SUB", "MUL", "DIV", "MAX", "MIN")
"""Function names, in the order random draws index them."""

INITIAL_DEPTHS: tuple[int, ...] = (2, 3)
"""Depths of the trees that ramped half-and-half draws from."""

GROW_TERMINAL_RATE = 0.3
"""Probability that the grow method ends a branch above the depth limit."""


def protected_division(numerator: Column, divisor: Column) -> Column:
    """Divide element-wise, giving 1 wherever the divisor is 0."""
    result = np.ones_like(numerator)
    np.divide(numerator, divisor, out=result, where=divisor != 0.0)
    return result


OPERATIONS: Mapping[str, Callable[[Column, Column], Column]] = MappingProxyType(
    {
        "ADD": np.add,
        "SUB": np.subtract,
        "MUL": np.multiply,
        "DIV": protected_division,
        "MAX": lambda first, second: np.where(first >= second, first, second),
        "MIN": lambda first, second: np.where(first <= second, first, second),
    }
)
"""The element-wise operation of every function."""


def task_depths(problem: Problem) -> list[int]:
    """Return, per task, the number of dependencies on the longest chain ending at it.

    One pass over the tasks in topological order relaxes every dependency, so
    each task's depth is one more than the largest depth of its predecessors,
    and zero without predecessors.
    """
    depths = [0] * problem.size
    for task in problem.topological_order:
        for following in problem.successors[task]:
            depths[following] = max(depths[following], depths[task] + 1)
    return depths


def terminal_columns(problem: Problem) -> dict[str, Column]:
    """Return the value of every terminal for every task, keyed by terminal name."""
    durations = min_durations(problem)
    releases = [task.release for task in problem.tasks]
    slacks = [
        due - release - duration
        for due, release, duration in zip(due_dates(problem), releases, durations, strict=True)
    ]
    columns = {
        "DUR": durations,
        "SUCC": [float(len(items)) for items in problem.successors],
        "REL": releases,
        "SLACK": slacks,
        "DEPTH": [float(value) for value in task_depths(problem)],
    }
    return {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


def evaluate(tree: Tree, columns: Mapping[str, Column]) -> Column:
    """Return the value of ``tree`` for every task, from the terminal columns.

    Overflow gives infinities and undefined operations give NaN, without
    warnings.
    """
    with np.errstate(all="ignore"):
        return _evaluate(tree, columns)


def _evaluate(tree: Tree, columns: Mapping[str, Column]) -> Column:
    if isinstance(tree, str):
        return columns[tree]
    name, left, right = tree
    return OPERATIONS[name](_evaluate(left, columns), _evaluate(right, columns))


def grow(limit: int, full: bool, rng: random.Random) -> Tree:
    """Return a random tree of depth at most ``limit``.

    With ``full``, every branch reaches ``limit``. Otherwise a branch above the
    limit ends in a terminal with probability :data:`GROW_TERMINAL_RATE`.
    Terminals and functions are drawn uniformly.
    """
    if limit <= 1 or (not full and rng.random() < GROW_TERMINAL_RATE):
        return rng.choice(TERMINALS)
    name = rng.choice(FUNCTIONS)
    return (name, grow(limit - 1, full, rng), grow(limit - 1, full, rng))


def ramped_half_and_half(rng: random.Random) -> Tree:
    """Return a tree of a depth drawn from :data:`INITIAL_DEPTHS`, full or grown with equal odds."""
    limit = rng.choice(INITIAL_DEPTHS)
    full = rng.random() < 0.5
    return grow(limit, full, rng)


def size(tree: Tree) -> int:
    """Return the number of nodes of ``tree``."""
    if isinstance(tree, str):
        return 1
    return 1 + size(tree[1]) + size(tree[2])


def depth(tree: Tree) -> int:
    """Return the depth of ``tree``; a single terminal has depth 1."""
    if isinstance(tree, str):
        return 1
    return 1 + max(depth(tree[1]), depth(tree[2]))


def subtree(tree: Tree, position: int) -> Tree:
    """Return the subtree rooted at the node of pre-order index ``position``."""
    while position:
        if isinstance(tree, str):
            raise IndexError("node position out of range")
        _, left, right = tree
        left_size = size(left)
        if position <= left_size:
            tree, position = left, position - 1
        else:
            tree, position = right, position - 1 - left_size
    return tree


def replace(tree: Tree, position: int, replacement: Tree) -> Tree:
    """Return a copy of ``tree`` with the subtree at pre-order index ``position`` replaced."""
    if position == 0:
        return replacement
    if isinstance(tree, str):
        raise IndexError("node position out of range")
    name, left, right = tree
    left_size = size(left)
    if position <= left_size:
        return (name, replace(left, position - 1, replacement), right)
    return (name, left, replace(right, position - 1 - left_size, replacement))


def crossover(parent: Tree, donor: Tree, rng: random.Random) -> Tree:
    """Return ``parent`` with a random subtree replaced by a random subtree of ``donor``."""
    graft = subtree(donor, rng.randrange(size(donor)))
    return replace(parent, rng.randrange(size(parent)), graft)


def mutate(tree: Tree, rng: random.Random) -> Tree:
    """Return ``tree`` with a random subtree replaced by a new grown tree.

    The new tree has at most the smallest depth of :data:`INITIAL_DEPTHS`.
    """
    fresh = grow(min(INITIAL_DEPTHS), False, rng)
    return replace(tree, rng.randrange(size(tree)), fresh)


def text(tree: Tree) -> str:
    """Return a compact text form such as ``ADD(DUR,DIV(SLACK,DEPTH))``."""
    if isinstance(tree, str):
        return tree
    name, left, right = tree
    return f"{name}({text(left)},{text(right)})"
