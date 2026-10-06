"""The instant from which a solver run counts its time.

A solver measures its elapsed time, and checks its time limit, on a clock it
creates during the run: the one of :class:`~dispatchatlas.solvers.search.Search`,
of the exact solvers, or of the continuous optimisers. Each such clock starts
at :func:`clock_start`. Called on its own, a solver therefore counts from the
instant it creates its clock, and what it does before, such as importing a
backend, is outside its elapsed time and its time limit.

Code that calls a solver can fix the origin first, with :func:`clock_origin`.
Every clock created inside the block then starts at that origin, so the same
time limit covers the same work for every solver: everything the solver does
from the origin on. The campaign engine sets the origin when an attempt of a
run starts.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Iterator
from contextvars import ContextVar

__all__ = ["clock_origin", "clock_start"]

_ORIGIN: ContextVar[float | None] = ContextVar("dispatchatlas-clock-origin", default=None)


@contextlib.contextmanager
def clock_origin(origin: float | None = None) -> Iterator[float]:
    """Start the solver clocks created inside the block at one instant.

    The origin holds for the clocks created in the thread that entered the
    block, until the block ends. A block inside another one replaces the
    origin for its own duration.

    A clock created inside the block counts from the origin, not from its own
    creation, and so does its time limit. A solver that runs several searches
    in turn, or gives one a share of its time limit, therefore opens a block
    of its own around each: ``with clock_origin():`` starts the clocks created
    inside it at the instant it is entered.

    Args:
        origin: The instant, as :func:`time.perf_counter` counts. Defaults to
            the instant the block is entered.

    Yields:
        The origin.

    Raises:
        ValueError: If ``origin`` is not a finite number, or is later than the
            current reading of :func:`time.perf_counter`, as a reading of
            another clock would be.
    """
    now = time.perf_counter()
    instant = now if origin is None else float(origin)
    if not math.isfinite(instant) or instant > now:
        raise ValueError(
            f"a clock origin must be a past reading of time.perf_counter (now {now!r}), "
            f"got {origin!r}"
        )
    token = _ORIGIN.set(instant)
    try:
        yield instant
    finally:
        _ORIGIN.reset(token)


def clock_start() -> float:
    """Return the instant a solver clock created now starts at.

    Returns:
        The origin set by :func:`clock_origin`, or the current reading of
        :func:`time.perf_counter` when none is set.
    """
    origin = _ORIGIN.get()
    return time.perf_counter() if origin is None else origin
