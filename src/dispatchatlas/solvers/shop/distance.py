"""Distances between job sequences, and the calibration of the keys' step size.

The distance between two job sequences is the **Kendall tau distance** (Kendall
1938, doi:10.1093/biomet/30.1-2.81) between the orders in which they place the
operations: the number of pairs of operations they place in different orders.
The ``k``-th occurrence of job ``j`` in a sequence stands for the ``k``-th
operation of ``j``, whose slot is the ``k``-th slot of ``j`` in
:attr:`~dispatchatlas.solvers.shop.instance.JobShop.slot_job`.
:func:`occurrence_permutation` lists the slot of each position of a sequence,
and :func:`kendall_tau` counts the pairs of slots that two such lists order
differently. Swapping two adjacent positions that hold different jobs moves a
sequence by one, and two key vectors that sort into the same job sequence are at
distance zero, whichever slots of a job hold which of its keys.

:func:`move_distances` measures how far one move of a representation takes
random solutions. :func:`calibrate_step` sets the step of the keys' move so that
its median distance equals a target, the median distance of the move of the job
sequences, so that the two representations of a family move by steps of the
same median size. Both draw from streams derived from their own seed, never from
those of a run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dispatchatlas.core.seeds import derive_seed

if TYPE_CHECKING:
    from dispatchatlas.solvers.shop.decoding import JobSequence
    from dispatchatlas.solvers.shop.representation import Representation

__all__ = [
    "MOVE_STREAM",
    "SOLUTION_STREAM",
    "calibrate_step",
    "kendall_tau",
    "move_distances",
    "occurrence_permutation",
]

SOLUTION_STREAM = "shop:distance:solutions"
"""The namespace of the stream that :func:`move_distances` draws its random solutions from."""

MOVE_STREAM = "shop:distance:moves"
"""The namespace of the stream that :func:`move_distances` draws its moves from."""

# Rows of the pair comparisons that kendall_tau makes at once, which bounds its memory.
_BLOCK = 256

# Bisection steps after which calibrate_step gives up.
_BISECTIONS = 200


def kendall_tau(first: ArrayLike, second: ArrayLike) -> int:
    """Return the Kendall tau distance between two orderings of the same items.

    Each ordering lists the same distinct integers, each once, in an order. The
    distance is the number of pairs of items that the two list in different
    orders: 0 for equal orderings and ``n (n - 1) / 2`` for an ordering and its
    reverse. It is computed with NumPy, by comparing every pair of positions in
    blocks of rows, which takes time quadratic in the number of items and memory
    linear in it.

    Args:
        first: The first ordering.
        second: The second ordering, of the same items.

    Returns:
        The number of discordant pairs.

    Raises:
        ValueError: If the orderings are not rows of the same distinct integers.
    """
    one = np.asarray(first)
    other = np.asarray(second)
    if one.ndim != 1 or one.shape != other.shape:
        raise ValueError(
            f"expected two orderings of one size, got shapes {one.shape} and {other.shape}"
        )
    if one.size and (one.dtype.kind not in "iu" or other.dtype.kind not in "iu"):
        raise ValueError(f"orderings hold integers, got {one.dtype} and {other.dtype}")
    items = np.sort(other)
    if (items[1:] == items[:-1]).any() or not np.array_equal(np.sort(one), items):
        raise ValueError("the orderings must list the same distinct items")
    # The position in ``second`` of each item of ``first``, in the order of ``first``:
    # a pair is discordant where a later item of ``first`` comes earlier in ``second``.
    sorter = np.argsort(other, kind="stable")
    places = sorter[np.searchsorted(other, one, sorter=sorter)]
    total = 0
    for start in range(0, places.size, _BLOCK):
        rows = places[start : start + _BLOCK, None]
        total += int(np.count_nonzero(np.triu(rows > places[None, start:], k=1)))
    return total


def occurrence_permutation(sequence: JobSequence, slot_job: JobSequence) -> NDArray[np.intp]:
    """Return the slot that each position of a job sequence stands for.

    The ``k``-th occurrence of job ``j`` in ``sequence`` stands for the ``k``-th
    slot of ``j`` in ``slot_job``, the slot of job ``j``'s ``k``-th operation. The
    result lists these slots in sequence order, so it is a permutation of the
    slot indices, and ``slot_job[result]`` is ``sequence``.

    Args:
        sequence: A job sequence: the jobs of ``slot_job``, each as often, in any
            order.
        slot_job: The job of each slot, such as
            :attr:`~dispatchatlas.solvers.shop.instance.JobShop.slot_job`.

    Returns:
        A new array of slot indices, one per position.

    Raises:
        ValueError: If ``sequence`` is not a permutation of ``slot_job``.
    """
    jobs = np.asarray(sequence)
    slots = np.asarray(slot_job)
    if jobs.ndim != 1 or jobs.shape != slots.shape:
        raise ValueError(f"expected {slots.size} job indices in one row, got shape {jobs.shape}")
    if jobs.size and (jobs.dtype.kind not in "iu" or slots.dtype.kind not in "iu"):
        raise ValueError(f"a job sequence holds integers, got {jobs.dtype}")
    # Positions and slots, each grouped job by job in their own order: the k-th
    # position of a job stands for its k-th slot.
    positions = np.argsort(jobs, kind="stable")
    by_job = np.argsort(slots, kind="stable")
    if not np.array_equal(jobs[positions], slots[by_job]):
        raise ValueError("the sequence does not hold each job as often as the slots do")
    permutation = np.empty(jobs.size, dtype=np.intp)
    permutation[positions] = by_job
    return permutation


def move_distances(
    representation: Representation[Any], *, samples: int, seed: int
) -> NDArray[np.int64]:
    """Return the distance by which one move takes each of ``samples`` random solutions.

    Each sample draws a random solution with
    :meth:`~dispatchatlas.solvers.shop.representation.Representation.initial`
    and one move of it, and measures the :func:`kendall_tau` distance between
    the :func:`occurrence_permutation` of their job sequences. The solutions come
    from the stream :data:`SOLUTION_STREAM` of ``seed`` and the moves from the
    stream :data:`MOVE_STREAM`, so every representation of a job shop is measured
    on the same random job sequences, and a key move draws the same slots and
    normal deviates whatever its step, but for the rare proposal drawn again
    because it left its key unchanged.

    Args:
        representation: The representation whose move is measured.
        samples: The number of solutions, at least 1.
        seed: The non-negative seed of the two streams.

    Returns:
        One distance per sample, in sample order.

    Raises:
        ValueError: If ``samples`` is below 1 or ``seed`` is negative.
    """
    _check_samples(samples)
    solutions = np.random.default_rng(derive_seed(seed, SOLUTION_STREAM))
    moves = np.random.default_rng(derive_seed(seed, MOVE_STREAM))
    slot_job = representation.shop.slot_job
    distances = np.empty(samples, dtype=np.int64)
    for index in range(samples):
        genotype = representation.initial(solutions)
        moved = representation.move(genotype, moves)
        distances[index] = kendall_tau(
            occurrence_permutation(representation.sequence(genotype), slot_job),
            occurrence_permutation(representation.sequence(moved), slot_job),
        )
    return distances


def calibrate_step(
    make: Callable[[float], Representation[Any]],
    target_median: float,
    low: float,
    high: float,
    *,
    samples: int,
    seed: int,
    tolerance: float,
) -> float:
    """Return a step whose move has a median distance within ``tolerance`` of a target.

    ``make(step)`` builds the representation to calibrate, such as a
    :class:`~dispatchatlas.solvers.shop.representation.KeyRepresentation` whose
    ``key_step`` is ``step``. The median of :func:`move_distances` over
    ``samples`` random solutions is measured at ``low`` and ``high``, which must
    bracket the target, and then at the geometric midpoint of the bracket, which
    shrinks towards the target, until a median lies within ``tolerance`` of it.
    Every measurement uses the same ``seed``, so each step is measured on the same
    solutions with the same draws, and the result is the same on every call.

    A perturbed key changes the distance by one each time it passes a key of
    another job, so the median of an odd number of distances moves in whole
    steps and a tolerance of one half finds a step whenever the bracket holds the
    target. The median need not grow with the step everywhere, since reflection
    at the bounds folds long steps back, and how far it reaches depends on how
    many keys a move changes: one changed key moves one job past others, as an
    insert does, while a swap moves two jobs, so the median of a swap may lie
    beyond every step of a move that changes one key.

    Args:
        make: Builds the representation of a step.
        target_median: The median distance to reach, such as the median of
            :func:`move_distances` of the job sequences' move.
        low: The smallest step tried, positive.
        high: The largest step tried, above ``low``.
        samples: The number of random solutions per measurement, at least 1.
        seed: The non-negative seed of every measurement.
        tolerance: How far from the target a median may lie, at least 0.

    Returns:
        The step.

    Raises:
        ValueError: If an argument is out of range, the medians at ``low`` and
            ``high`` do not bracket the target, or no step within the bracket
            gives a median within ``tolerance`` of it.
    """
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError(f"the steps must satisfy 0 < low < high, got {low!r} and {high!r}")
    if not (math.isfinite(target_median) and target_median >= 0.0):
        raise ValueError(f"target_median must be finite and non-negative, got {target_median!r}")
    if not (math.isfinite(tolerance) and tolerance >= 0.0):
        raise ValueError(f"tolerance must be finite and non-negative, got {tolerance!r}")
    _check_samples(samples)

    def median(step: float) -> float:
        return float(np.median(move_distances(make(step), samples=samples, seed=seed)))

    lower, upper = median(low), median(high)
    for step, value in ((low, lower), (high, upper)):
        if abs(value - target_median) <= tolerance:
            return step
    if not lower < target_median < upper:
        raise ValueError(
            f"the target median {target_median} does not lie between the medians "
            f"{lower} at step {low} and {upper} at step {high}"
        )
    closest = (math.inf, low)
    for _ in range(_BISECTIONS):
        middle = math.sqrt(low * high)
        if not low < middle < high:
            break
        value = median(middle)
        gap = abs(value - target_median)
        if gap <= tolerance:
            return middle
        closest = min(closest, (gap, middle))
        if value < target_median:
            low = middle
        else:
            high = middle
    raise ValueError(
        f"no step gives a median within {tolerance} of {target_median}; the closest, "
        f"step {closest[1]!r}, gives one {closest[0]} away"
    )


def _check_samples(samples: int) -> None:
    """Raise :class:`ValueError` unless ``samples`` is an integer of at least 1."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(f"samples must be an integer of at least 1, got {samples!r}")
