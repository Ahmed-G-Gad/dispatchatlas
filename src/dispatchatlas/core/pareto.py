"""Pareto dominance on minimisation vectors.

Every function here assumes that smaller is better in every coordinate. Use
:meth:`dispatchatlas.core.objectives.Objective.minimized` to orient values of
maximised objectives first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "crowding_distance",
    "dominates",
    "fast_non_dominated_sort",
    "hypervolume",
    "non_dominated",
]

type Vector = Sequence[float]


def dominates(a: Vector, b: Vector) -> bool:
    """Return whether ``a`` Pareto-dominates ``b``.

    ``a`` dominates ``b`` when it is no worse in every coordinate and strictly
    better in at least one.
    """
    strictly_better = False
    for x, y in zip(a, b, strict=True):
        if x > y:
            return False
        if x < y:
            strictly_better = True
    return strictly_better


def non_dominated(points: Sequence[Vector]) -> list[int]:
    """Return the indices of the points no other point dominates.

    Duplicates of a non-dominated point are all kept.
    """
    return [
        i
        for i, point in enumerate(points)
        if not any(dominates(other, point) for j, other in enumerate(points) if j != i)
    ]


def fast_non_dominated_sort(points: Sequence[Vector]) -> list[list[int]]:
    """Partition point indices into successive non-dominated fronts.

    Implements the sorting procedure of Deb, Pratap, Agarwal and Meyarivan
    (2002), NSGA-II, doi:10.1109/4235.996017.

    Returns:
        Fronts in rank order. The first front holds the non-dominated points.
    """
    count = len(points)
    dominated_by: list[list[int]] = [[] for _ in range(count)]
    domination_count = [0] * count
    for i in range(count):
        for j in range(i + 1, count):
            if dominates(points[i], points[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(points[j], points[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1
    fronts: list[list[int]] = [[i for i in range(count) if domination_count[i] == 0]]
    current = 0
    while fronts[current]:
        following: list[int] = []
        for i in fronts[current]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    following.append(j)
        current += 1
        fronts.append(sorted(following))
    return fronts[:-1]


def crowding_distance(points: Sequence[Vector]) -> list[float]:
    """Return the NSGA-II crowding distance of every point in one front.

    Boundary points in any coordinate get infinite distance. Coordinates with
    zero spread contribute nothing.
    """
    count = len(points)
    if count == 0:
        return []
    distance = [0.0] * count
    for axis in range(len(points[0])):
        order = sorted(range(count), key=lambda i: points[i][axis])
        low, high = points[order[0]][axis], points[order[-1]][axis]
        distance[order[0]] = distance[order[-1]] = math.inf
        span = high - low
        if span <= 0.0:
            continue
        for rank in range(1, count - 1):
            i = order[rank]
            if not math.isinf(distance[i]):
                distance[i] += (
                    points[order[rank + 1]][axis] - points[order[rank - 1]][axis]
                ) / span
    return distance


def hypervolume(points: Sequence[Vector], reference: Vector) -> float:
    """Return the volume that ``points`` dominate, bounded by ``reference``.

    Points that are not strictly below the reference in every coordinate add
    nothing. The result is exact in any dimension: two coordinates use a sweep
    over the staircase, more coordinates slice along the last one
    (While, Hingston, Barone and Huband 2006, doi:10.1109/TEVC.2005.851275).
    The cost grows as ``n ** (d - 1)``, which suits the front sizes of
    scheduling experiments.

    Args:
        points: Objective vectors in minimisation form.
        reference: A point that every counted vector must dominate.

    Returns:
        The hypervolume, zero for an empty set.
    """
    bound = tuple(float(value) for value in reference)
    kept = [
        tuple(float(value) for value in point)
        for point in points
        if all(value < limit for value, limit in zip(point, bound, strict=True))
    ]
    if not kept:
        return 0.0
    return _slice_volume(kept, bound)


def _slice_volume(points: list[tuple[float, ...]], reference: tuple[float, ...]) -> float:
    if len(reference) == 1:
        return reference[0] - min(point[0] for point in points)
    if len(reference) == 2:
        volume = 0.0
        lowest = reference[1]
        for x, y in sorted(points):
            if y < lowest:
                volume += (reference[0] - x) * (lowest - y)
                lowest = y
        return volume
    ordered = sorted(points, key=lambda point: point[-1])
    volume = 0.0
    for k, point in enumerate(ordered):
        upper = ordered[k + 1][-1] if k + 1 < len(ordered) else reference[-1]
        depth = upper - point[-1]
        if depth > 0.0:
            section = [item[:-1] for item in ordered[: k + 1]]
            volume += depth * _slice_volume(section, reference[:-1])
    return volume
