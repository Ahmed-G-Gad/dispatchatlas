"""Pareto dominance, non-dominated sorting, crowding distance, and hypervolume."""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from dispatchatlas.core.pareto import (
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    hypervolume,
    non_dominated,
)

type Point = tuple[float, ...]


def _random_points(rng: random.Random, count: int, dimension: int, grid: int = 5) -> list[Point]:
    """Points on a small integer grid, so ties and duplicates are common."""
    return [tuple(float(rng.randrange(grid)) for _ in range(dimension)) for _ in range(count)]


def _union_volume(points: list[Point], reference: Point) -> float:
    """Volume of the union of the boxes ``[point, reference]``, by inclusion and exclusion."""
    boxes = [p for p in points if all(x < r for x, r in zip(p, reference, strict=True))]
    total = 0.0
    for size in range(1, len(boxes) + 1):
        for subset in itertools.combinations(boxes, size):
            corner = [max(values) for values in zip(*subset, strict=True)]
            volume = math.prod(r - c for r, c in zip(reference, corner, strict=True))
            total += (-1) ** (size + 1) * volume
    return total


# ---------------------------------------------------------------- dominance


def test_dominance_needs_no_worse_everywhere_and_better_somewhere() -> None:
    assert dominates((1, 2), (2, 3))
    assert dominates((1, 2), (1, 3))
    assert not dominates((1, 2), (1, 2))
    assert not dominates((1, 3), (2, 2))
    assert not dominates((2, 3), (1, 2))
    assert dominates((0, 0, 0, -1), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="zip"):
        dominates((1, 2), (1, 2, 3))


def test_dominance_is_a_strict_partial_order() -> None:
    rng = random.Random(1)
    points = _random_points(rng, 30, 3, grid=3)
    for a, b, c in itertools.product(points[:12], repeat=3):
        assert not dominates(a, a)
        assert not (dominates(a, b) and dominates(b, a))
        if dominates(a, b) and dominates(b, c):
            assert dominates(a, c)


def test_non_dominated_keeps_duplicates() -> None:
    assert non_dominated([(1, 1), (1, 1), (2, 0), (2, 2)]) == [0, 1, 2]
    assert non_dominated([]) == []
    assert non_dominated([(3, 3)]) == [0]


# ------------------------------------------------------------------ sorting


def test_fast_non_dominated_sort_by_hand() -> None:
    points = [(1, 5), (2, 4), (3, 3), (2, 5), (4, 4), (5, 5)]
    assert fast_non_dominated_sort(points) == [[0, 1, 2], [3, 4], [5]]
    assert fast_non_dominated_sort([]) == []
    assert fast_non_dominated_sort([(1, 1), (1, 1)]) == [[0, 1]]


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_fronts_peel_off_successive_non_dominated_sets(dimension: int) -> None:
    rng = random.Random(dimension)
    for _ in range(30):
        points = _random_points(rng, rng.randint(1, 25), dimension)
        fronts = fast_non_dominated_sort(points)
        assert sorted(i for front in fronts for i in front) == list(range(len(points)))
        remaining = list(range(len(points)))
        for rank, front in enumerate(fronts):
            assert front == sorted(front)
            expected = [remaining[i] for i in non_dominated([points[j] for j in remaining])]
            assert front == expected
            if rank:
                for i in front:
                    assert any(dominates(points[j], points[i]) for j in fronts[rank - 1])
            remaining = [i for i in remaining if i not in front]


# ----------------------------------------------------------------- crowding


def test_crowding_distance_by_hand() -> None:
    assert crowding_distance([(0, 4), (1, 3), (2, 2), (4, 0)]) == [math.inf, 1.0, 1.5, math.inf]
    assert crowding_distance([(0, 0, 2), (1, 1, 1), (2, 2, 0)]) == [math.inf, 3.0, math.inf]
    assert crowding_distance([]) == []
    assert crowding_distance([(1, 2)]) == [math.inf]
    assert crowding_distance([(1, 2), (2, 1)]) == [math.inf, math.inf]


def test_coordinates_without_spread_add_nothing_to_interior_points() -> None:
    assert crowding_distance([(1, 0), (1, 1), (1, 3)]) == [math.inf, 1.0, math.inf]
    distance = crowding_distance([(5, 5)] * 4)
    assert distance[1:3] == [0.0, 0.0]


def test_crowding_distance_is_finite_and_non_negative_inside() -> None:
    rng = random.Random(3)
    for _ in range(20):
        front = [(float(x), float(10 - x) + rng.random()) for x in rng.sample(range(10), 6)]
        distance = crowding_distance(front)
        order = sorted(range(len(front)), key=lambda i: front[i][0])
        assert math.isinf(distance[order[0]])
        assert math.isinf(distance[order[-1]])
        assert all(0.0 <= distance[i] < math.inf for i in order[1:-1])


# -------------------------------------------------------------- hypervolume


def test_hypervolume_by_hand() -> None:
    assert hypervolume([(3,), (5,)], (10,)) == 7.0
    # The staircase of three points: 4 + 6 + 1.
    assert hypervolume([(1, 4), (2, 2), (4, 1)], (5, 5)) == 11.0
    assert hypervolume([(1, 1, 1)], (2, 3, 4)) == 6.0
    # Two boxes of 4 and 2 that share a unit cube.
    assert hypervolume([(0, 0, 1), (1, 1, 0)], (2, 2, 2)) == 5.0
    # Boxes of 8 and 2 sharing 1; the third point is dominated by the first.
    four = [(0, 0, 0, 1), (1, 1, 1, 0), (0, 1, 1, 1)]
    assert hypervolume(four, (2, 2, 2, 2)) == 9.0


def test_hypervolume_counts_only_points_strictly_inside_the_reference() -> None:
    assert hypervolume([], (1, 1)) == 0.0
    assert hypervolume([(5, 1), (6, 0), (1, 5)], (5, 5)) == 0.0
    assert hypervolume([(5, 1), (4, 4)], (5, 5)) == 1.0
    with pytest.raises(ValueError, match="zip"):
        hypervolume([(1, 1, 1)], (2, 2))


def test_hypervolume_ignores_order_duplicates_and_dominated_points() -> None:
    points = [(1.0, 4.0, 2.0), (2.0, 2.0, 3.0), (4.0, 1.0, 1.0), (3.0, 3.0, 3.5)]
    reference = (5.0, 5.0, 5.0)
    value = hypervolume(points, reference)
    for permutation in itertools.permutations(points):
        assert hypervolume(list(permutation), reference) == pytest.approx(value, abs=1e-12)
    assert hypervolume([*points, points[0], (4.5, 4.5, 4.5)], reference) == pytest.approx(value)


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_hypervolume_matches_inclusion_and_exclusion(dimension: int) -> None:
    rng = random.Random(dimension)
    for _ in range(25):
        points = [
            tuple(rng.uniform(0, 4) for _ in range(dimension)) for _ in range(rng.randint(1, 7))
        ]
        reference = tuple(rng.uniform(2, 5) for _ in range(dimension))
        expected = _union_volume(points, reference)
        assert hypervolume(points, reference) == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("dimension", [3, 4])
def test_hypervolume_agrees_with_monte_carlo_sampling(dimension: int) -> None:
    generator = np.random.default_rng(dimension)
    # A concave front on the unit sphere plus a few dominated points.
    directions = np.abs(generator.normal(size=(12, dimension)))
    front = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    points = np.vstack([front, front[:3] + 0.2])
    reference = np.full(dimension, 1.1)
    exact = hypervolume([tuple(map(float, p)) for p in points], tuple(map(float, reference)))
    lower = points.min(axis=0)
    box = float(np.prod(reference - lower))
    samples = generator.uniform(lower, reference, size=(200_000, dimension))
    covered = np.zeros(len(samples), dtype=bool)
    for point in points:
        covered |= np.all(samples >= point, axis=1)
    share = covered.mean()
    estimate = box * share
    error = box * math.sqrt(share * (1 - share) / len(samples))
    assert abs(exact - estimate) < 4 * error
    assert 0.0 < exact < box
