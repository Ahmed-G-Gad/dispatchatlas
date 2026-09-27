"""Ranks, ties, and the orientation of objective values.

Every rank-based statistic in this package opens the same way: replace values
by their ranks, give tied values the mean of the positions they span
(midranks), and orient the ranks so that rank 1 is the best value. Two values
tie when they are equal within :data:`TIE_TOLERANCE`, relatively or
absolutely, so a difference left by floating-point summation is not read as a
win. Infinite values are allowed and rank last or first: the analysis
pipeline scores a failed run as the worst possible value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from dispatchatlas.core.errors import ConfigError

__all__ = [
    "SENSES",
    "TIE_TOLERANCE",
    "Sense",
    "check_sense",
    "finite_or_infinite",
    "midranks",
    "oriented_ranks",
    "rank_block",
    "strictly_better",
    "ties",
]

type Sense = Literal["min", "max"]
"""Whether smaller (``"min"``) or larger (``"max"``) values are better."""

SENSES: tuple[Sense, ...] = ("min", "max")

TIE_TOLERANCE = 1e-12
"""Two values tie when they differ by at most this much, relatively or absolutely."""


def check_sense(sense: object) -> Sense:
    """Return ``sense`` if it is ``"min"`` or ``"max"``.

    Raises:
        ConfigError: For any other value.
    """
    if sense == "min":
        return "min"
    if sense == "max":
        return "max"
    raise ConfigError(f"sense must be 'min' or 'max', got {sense!r}")


def finite_or_infinite(values: Iterable[float], what: str) -> list[float]:
    """Return ``values`` as floats, refusing NaN.

    Args:
        values: The values.
        what: Name of the quantity, for the error message.

    Raises:
        ConfigError: If a value is NaN or not a number.
    """
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{what} must be numbers, got {value!r}")
        number = float(value)
        if math.isnan(number):
            raise ConfigError(f"{what} must not contain NaN")
        result.append(number)
    return result


def ties(left: float, right: float) -> bool:
    """Return whether two values are equal within :data:`TIE_TOLERANCE`.

    Equal infinities tie; infinities of opposite sign do not.
    """
    return math.isclose(left, right, rel_tol=TIE_TOLERANCE, abs_tol=TIE_TOLERANCE)


def strictly_better(left: float, right: float, sense: Sense) -> bool:
    """Return whether ``left`` beats ``right`` under ``sense``, ties excluded."""
    if ties(left, right):
        return False
    return left < right if sense == "min" else left > right


def midranks(values: Sequence[float]) -> tuple[float, ...]:
    """Return ascending ranks, from 1, with tied values sharing their mean rank.

    Values are sorted, and each following value joins the current tie group
    while it ties with the group's first value, so groups never chain. The
    ranks are returned in the order of the input.
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and ties(values[order[end]], values[order[start]]):
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return tuple(ranks)


def oriented_ranks(values: Sequence[float], sense: Sense) -> tuple[float, ...]:
    """Return midranks oriented so that rank 1 is the best value under ``sense``."""
    if sense == "max":
        return midranks([-value for value in values])
    return midranks(values)


def rank_block(block: Mapping[str, float], sense: Sense) -> dict[str, float]:
    """Rank the entries of one block, such as the solvers on one problem.

    Args:
        block: Value of each entry.
        sense: Whether smaller or larger values are better.

    Returns:
        The oriented midrank of each entry, keyed like ``block``.
    """
    keys = list(block)
    ranks = oriented_ranks([block[key] for key in keys], sense)
    return dict(zip(keys, ranks, strict=True))
