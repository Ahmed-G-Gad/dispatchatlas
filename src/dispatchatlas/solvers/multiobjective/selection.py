"""Environmental and mating selection of the multi-objective evolutionary algorithms.

Every function here is pure and works on objective vectors in minimisation
form, as :meth:`~dispatchatlas.solvers.search.Search.vector` returns them.
Functions that break ties at random take the run's generator. The solvers in
this package call them on the feasible members of a population; infeasible
members are ranked by deadline overrun before any of these rules applies.

Contents, by method:

- shared: ideal and nadir points, range normalisation, and the Das and Dennis
  (1998) points of the unit simplex;
- NSGA-II: survival by non-dominated fronts and crowding distance;
- NSGA-III: hyperplane normalisation and niching around reference points;
- SPEA2: strength fitness with nearest-neighbour density, and archive
  truncation;
- IBEA: additive epsilon indicator fitness and its update on removal;
- SMS-EMOA: exact hypervolume contributions and the choice of the member to
  discard;
- MOEA/D: the Tchebycheff scalarisation and weight neighbourhoods;
- RVEA: angle-penalised distance and reference-vector adaptation;
- AGE-MOEA-II: estimation of the front's curvature, geodesic distances, and
  survival scores.
"""

from __future__ import annotations

import math
import operator
import random
from bisect import insort
from collections.abc import Sequence
from dataclasses import dataclass

from dispatchatlas.core.pareto import (
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    hypervolume,
)

__all__ = [
    "ASF_WEIGHT",
    "IBEA_KAPPA",
    "WEIGHT_FLOOR",
    "Survivors",
    "adapt_vectors",
    "additive_epsilon",
    "age_scores",
    "age_survival",
    "corner_points",
    "crowding_survival",
    "das_dennis",
    "extreme_points",
    "front_curvature",
    "geodesic_distance",
    "hyperplane_normalize",
    "hypervolume_contributions",
    "hypervolume_discard",
    "ibea_fitness",
    "ibea_survival",
    "ideal_point",
    "intercepts",
    "last_front",
    "minkowski_norm",
    "nadir_point",
    "newton_curvature",
    "niching_survival",
    "normalize",
    "perpendicular_distance",
    "project_to_front",
    "rvea_survival",
    "simplex_divisions",
    "smallest_angles",
    "spea2_fitness",
    "spea2_survival",
    "spea2_truncation",
    "tchebycheff",
    "unit_vectors",
    "weight_neighbourhoods",
]

type Vector = Sequence[float]
type Point = tuple[float, ...]

ASF_WEIGHT = 1e-6
"""Weight of the other objectives in the achievement function that finds extreme points."""

WEIGHT_FLOOR = 1e-6
"""Smallest weight the Tchebycheff scalarisation gives an objective."""

IBEA_KAPPA = 0.05
"""Fitness scaling factor of IBEA used by Zitzler and Künzli (2004)."""

NEWTON_PRECISION = 1e-3
"""Step size at which the curvature estimate of AGE-MOEA-II stops."""

NEWTON_STEPS = 100
"""Largest number of Newton steps of the curvature estimate."""

HYPERVOLUME_MARGIN = 0.1
"""Distance, as a fraction of the front's range, from its worst values to the reference point."""


@dataclass(frozen=True, slots=True)
class Survivors:
    """The members an environmental selection keeps, with their mating keys.

    Attributes:
        indices: Positions of the kept members in the pool, in selection order.
        keys: One key per kept member, aligned with ``indices``. A binary
            tournament prefers the member with the smaller key.
    """

    indices: tuple[int, ...]
    keys: tuple[tuple[float, ...], ...]


# ----------------------------------------------------------------- shared


def ideal_point(points: Sequence[Vector]) -> Point:
    """Return the smallest value of every objective over ``points``."""
    return tuple(min(values) for values in zip(*points, strict=True))


def nadir_point(points: Sequence[Vector]) -> Point:
    """Return the largest value of every objective over ``points``."""
    return tuple(max(values) for values in zip(*points, strict=True))


def normalize(
    points: Sequence[Vector], lower: Vector | None = None, upper: Vector | None = None
) -> list[Point]:
    """Map every objective linearly so that ``lower`` becomes 0 and ``upper`` becomes 1.

    Args:
        points: Objective vectors.
        lower: Values mapped to 0. Defaults to the ideal point of ``points``.
        upper: Values mapped to 1. Defaults to the nadir point of ``points``.

    Returns:
        The mapped vectors. An objective whose bounds coincide is only
        translated, so its values stay finite.
    """
    if not points:
        return []
    low = ideal_point(points) if lower is None else tuple(lower)
    high = nadir_point(points) if upper is None else tuple(upper)
    spans = [top - bottom if top > bottom else 1.0 for bottom, top in zip(low, high, strict=True)]
    return [
        tuple(
            (value - bottom) / span for value, bottom, span in zip(point, low, spans, strict=True)
        )
        for point in points
    ]


def das_dennis(divisions: int, objectives: int) -> list[Point]:
    """Return the Das and Dennis (1998) lattice of the unit simplex.

    Every point has non-negative coordinates that are multiples of
    ``1 / divisions`` and sum to one. There are
    ``comb(divisions + objectives - 1, objectives - 1)`` of them, listed in
    increasing lexicographic order of their leading coordinates, the last
    coordinate taking what remains: ``(0, 1), (0.5, 0.5), (1, 0)`` for two
    objectives and two divisions. With one objective the lattice is the single
    point ``(1,)``.

    Args:
        divisions: Steps along each objective, at least 1.
        objectives: Number of objectives, at least 1.

    Raises:
        ValueError: If either count is smaller than 1.
    """
    if divisions < 1 or objectives < 1:
        raise ValueError("divisions and objectives must be at least 1")
    points: list[Point] = []

    def extend(prefix: tuple[int, ...], remaining: int, axes: int) -> None:
        if axes == 1:
            points.append(tuple(step / divisions for step in (*prefix, remaining)))
            return
        for step in range(remaining + 1):
            extend((*prefix, step), remaining - step, axes - 1)

    extend((), divisions, objectives)
    return points


def simplex_divisions(objectives: int, size: int) -> int:
    """Return the most divisions whose Das and Dennis lattice has at most ``size`` points.

    This sizes the reference sets of NSGA-III, MOEA/D, and RVEA from a
    population size. The result is at least 1, so with more objectives than
    ``size`` the lattice has one point per objective.
    """
    if objectives <= 1:
        return 1
    divisions = 1
    while math.comb(divisions + objectives, objectives - 1) <= size:
        divisions += 1
    return divisions


def perpendicular_distance(point: Vector, direction: Vector) -> float:
    """Return the distance from ``point`` to the line through the origin along ``direction``."""
    norm = sum(value * value for value in direction)
    if norm <= 0.0:
        return math.sqrt(sum(value * value for value in point))
    scale = sum(a * b for a, b in zip(point, direction, strict=True)) / norm
    return math.sqrt(sum((a - scale * b) ** 2 for a, b in zip(point, direction, strict=True)))


def extreme_points(points: Sequence[Vector]) -> list[int]:
    """Return, for every objective, the point that best approximates its axis.

    The extreme point of objective ``j`` minimises the achievement function
    ``max_i f_i / w_i`` with ``w_j = 1`` and every other weight
    :data:`ASF_WEIGHT`, as in Deb and Jain (2014). Points should be translated
    so that the ideal point is the origin. Ties go to the lowest index, and a
    point may be extreme for several objectives.
    """
    dimension = len(points[0])
    chosen: list[int] = []
    for axis in range(dimension):
        weights = [1.0 if i == axis else ASF_WEIGHT for i in range(dimension)]
        chosen.append(
            min(
                range(len(points)),
                key=lambda row: max(
                    value / weight for value, weight in zip(points[row], weights, strict=True)
                ),
            )
        )
    return chosen


def intercepts(points: Sequence[Vector], extremes: Sequence[int]) -> list[float]:
    """Return where the hyperplane through the extreme points meets each objective axis.

    Points should be translated so that the ideal point is the origin. When
    the extreme points do not span a hyperplane with positive intercepts,
    because two of them coincide or they are linearly dependent, the largest
    value of each objective is used instead, and an objective whose largest
    value is zero keeps the scale 1.
    """
    dimension = len(points[0])
    plane = _solve([list(points[index]) for index in extremes], [1.0] * dimension)
    if plane is not None and all(coefficient > 0.0 for coefficient in plane):
        values = [1.0 / coefficient for coefficient in plane]
        if all(math.isfinite(value) for value in values):
            return values
    return [value if value > 0.0 else 1.0 for value in nadir_point(points)]


def hyperplane_normalize(points: Sequence[Vector]) -> tuple[list[Point], list[int]]:
    """Normalise points by the ideal point and the intercepts of their extreme points.

    The adaptive normalisation of Deb and Jain (2014): translate so that the
    ideal point is the origin, find one extreme point per objective with
    :func:`extreme_points`, and divide each objective by the intercept of the
    hyperplane through them (:func:`intercepts`). AGE-MOEA-II uses the same
    procedure.

    Returns:
        The normalised points and the indices of the extreme points.
    """
    ideal = ideal_point(points)
    translated = [
        tuple(value - low for value, low in zip(point, ideal, strict=True)) for point in points
    ]
    extremes = extreme_points(translated)
    scale = intercepts(translated, extremes)
    normalized = [
        tuple(value / factor for value, factor in zip(point, scale, strict=True))
        for point in translated
    ]
    return normalized, extremes


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Solve ``matrix @ x = rhs`` by Gaussian elimination, or return ``None`` if singular."""
    size = len(matrix)
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    if scale == 0.0:
        return None
    rows = [[*row, value] for row, value in zip(matrix, rhs, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) <= 1e-12 * scale:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        lead = rows[column]
        for row in range(size):
            if row != column and rows[row][column] != 0.0:
                factor = rows[row][column] / lead[column]
                rows[row] = [a - factor * b for a, b in zip(rows[row], lead, strict=True)]
    return [rows[row][size] / rows[row][row] for row in range(size)]


def _apply_order(front: Sequence[int], order: Sequence[int]) -> list[int]:
    return [front[position] for position in order]


# ----------------------------------------------------------------- NSGA-II


def crowding_survival(points: Sequence[Vector], size: int) -> Survivors:
    """Select ``size`` points by non-dominated fronts, then crowding distance.

    The environmental selection of NSGA-II (Deb, Pratap, Agarwal and
    Meyarivan 2002): whole fronts are kept in rank order, and the front that
    does not fit is cut by decreasing crowding distance, boundary points
    first, ties in front order. Each survivor's key is ``(rank, -crowding)``,
    the crowded-comparison order of the binary tournament.
    """
    chosen: list[int] = []
    keys: list[tuple[float, ...]] = []
    for rank, front in enumerate(fast_non_dominated_sort(points)):
        room = size - len(chosen)
        if room <= 0:
            break
        distances = crowding_distance([points[index] for index in front])
        order = list(range(len(front)))
        if len(front) > room:
            order = sorted(order, key=lambda position: -distances[position])[:room]
        for position in order:
            chosen.append(front[position])
            keys.append((float(rank), -distances[position]))
    return Survivors(tuple(chosen), tuple(keys))


# ---------------------------------------------------------------- NSGA-III


def niching_survival(
    points: Sequence[Vector],
    size: int,
    references: Sequence[Vector],
    rng: random.Random,
) -> list[int]:
    """Select ``size`` points by non-dominated fronts, then niching around reference points.

    The environmental selection of NSGA-III (Deb and Jain 2014). Whole fronts
    are kept while they fit. The kept points and the front that does not fit
    are normalised together (:func:`hyperplane_normalize`), and each is
    associated with the reference line, through the origin and a reference
    point, at the smallest perpendicular distance. The niche count of a
    reference point is the number of kept points associated with it. Points
    of the last front are then added one at a time: a reference point with
    the smallest niche count is drawn at random; if no remaining point of the
    last front is associated with it, it leaves the competition; otherwise it
    receives the associated point nearest its line when its count is zero,
    and a random associated point otherwise.

    Returns:
        Indices of the selected points: the kept fronts in rank order, then
        the niched points in the order they were chosen.
    """
    chosen: list[int] = []
    last: list[int] = []
    for front in fast_non_dominated_sort(points):
        if len(chosen) + len(front) <= size:
            chosen.extend(front)
            if len(chosen) == size:
                return chosen
        else:
            last = front
            break
    if not last:
        return chosen
    members = [*chosen, *last]
    normalized, _ = hyperplane_normalize([points[index] for index in members])
    niche: list[int] = []
    distance: list[float] = []
    for point in normalized:
        best, gap = 0, math.inf
        for reference, direction in enumerate(references):
            candidate = perpendicular_distance(point, direction)
            if candidate < gap:
                best, gap = reference, candidate
        niche.append(best)
        distance.append(gap)
    counts = [0] * len(references)
    for position in range(len(chosen)):
        counts[niche[position]] += 1
    waiting: dict[int, list[int]] = {}
    for position in range(len(chosen), len(members)):
        waiting.setdefault(niche[position], []).append(position)
    active = list(range(len(references)))
    picked: list[int] = []
    while len(picked) < size - len(chosen):
        least = min(counts[reference] for reference in active)
        tied = [reference for reference in active if counts[reference] == least]
        reference = tied[rng.randrange(len(tied))] if len(tied) > 1 else tied[0]
        candidates = waiting.get(reference)
        if not candidates:
            active.remove(reference)
            continue
        if counts[reference] == 0:
            position = min(candidates, key=lambda item: distance[item])
        else:
            position = candidates[rng.randrange(len(candidates))]
        candidates.remove(position)
        picked.append(members[position])
        counts[reference] += 1
    return chosen + picked


# ------------------------------------------------------------------- SPEA2


def spea2_fitness(points: Sequence[Vector], k: int | None = None) -> list[float]:
    """Return the SPEA2 fitness of every point: raw fitness plus density.

    Zitzler, Laumanns and Thiele (2001). The strength of a point is the number
    of points it dominates; its raw fitness is the sum of the strengths of the
    points that dominate it, zero for a non-dominated point; its density is
    ``1 / (sigma + 2)``, where ``sigma`` is the Euclidean distance to its k-th
    nearest neighbour. Smaller is better, and exactly the non-dominated points
    score below 1.

    Args:
        points: Objective vectors.
        k: Neighbour rank of the density. Defaults to the integer square root
            of the number of points, the square root of the sample size of
            the paper.
    """
    count = len(points)
    dominated_by: list[list[int]] = [[] for _ in range(count)]
    strength = [0] * count
    for i in range(count):
        for j in range(count):
            if i != j and dominates(points[i], points[j]):
                strength[i] += 1
                dominated_by[j].append(i)
    rank = max(1, math.isqrt(count)) if k is None else k
    fitness: list[float] = []
    for i in range(count):
        raw = sum(strength[j] for j in dominated_by[i])
        gaps = sorted(math.dist(points[i], points[j]) for j in range(count) if j != i)
        sigma = gaps[min(rank, len(gaps)) - 1] if gaps else 0.0
        fitness.append(raw + 1.0 / (sigma + 2.0))
    return fitness


def spea2_truncation(points: Sequence[Vector], size: int) -> list[int]:
    """Return the indices left after SPEA2's archive truncation.

    Points are removed one at a time until ``size`` remain. The removed point
    has the smallest distance to its nearest remaining neighbour; ties are
    decided by the second-nearest distance, then the third, and so on, which
    is the lexicographic order of the sorted distance lists
    (Zitzler, Laumanns and Thiele 2001). Complete ties go to the lowest index.

    Returns:
        The remaining indices, in increasing order.
    """
    count = len(points)
    matrix = [[math.dist(a, b) for b in points] for a in points]
    alive = list(range(count))
    neighbours = {i: sorted(matrix[i][j] for j in alive if j != i) for i in alive}
    while len(alive) > max(size, 0):
        victim = min(alive, key=lambda i: neighbours[i])
        alive.remove(victim)
        del neighbours[victim]
        for i in alive:
            gaps = neighbours[i]
            gaps.remove(matrix[i][victim])
    return alive


def spea2_survival(points: Sequence[Vector], size: int) -> Survivors:
    """Select ``size`` points by the SPEA2 environmental selection.

    Every non-dominated point is kept. A shortfall is filled with the
    dominated points of smallest fitness; a surplus is removed by
    :func:`spea2_truncation`. Objectives are first scaled to the range of the
    pool, so that distances, and with them the density and the truncation, do
    not depend on the objectives' units. Each survivor's key is its fitness.
    """
    scaled = normalize(points)
    fitness = spea2_fitness(scaled)
    front = [index for index in range(len(points)) if fitness[index] < 1.0]
    if len(front) > size:
        kept = _apply_order(front, spea2_truncation([scaled[index] for index in front], size))
    else:
        rest = sorted(
            (index for index in range(len(points)) if fitness[index] >= 1.0),
            key=lambda index: fitness[index],
        )
        kept = front + rest[: size - len(front)]
    return Survivors(tuple(kept), tuple((fitness[index],) for index in kept))


# -------------------------------------------------------------------- IBEA


def additive_epsilon(first: Vector, second: Vector) -> float:
    """Return the additive epsilon indicator ``max_i (first_i - second_i)``.

    It is the smallest amount by which ``first`` must be translated in every
    objective to weakly dominate ``second``, negative when ``first``
    dominates ``second``.
    """
    return max(a - b for a, b in zip(first, second, strict=True))


def _ibea_terms(points: Sequence[Vector], kappa: float) -> tuple[list[list[float]], float]:
    """Return the pairwise indicator values on scaled objectives and the fitness scale."""
    scaled = normalize(points)
    count = len(scaled)
    indicator = [
        [additive_epsilon(scaled[i], scaled[j]) for j in range(count)] for i in range(count)
    ]
    largest = max((abs(value) for row in indicator for value in row), default=0.0)
    return indicator, (largest if largest > 0.0 else 1.0) * kappa


def ibea_fitness(points: Sequence[Vector], kappa: float = IBEA_KAPPA) -> list[float]:
    """Return the IBEA fitness of every point, larger being better.

    The adaptive fitness of Zitzler and Künzli (2004): objectives are scaled to
    the range of the points, ``I`` is the additive epsilon indicator
    (:func:`additive_epsilon`), ``c`` the largest absolute indicator value,
    and ``F(x) = sum over y != x of -exp(-I(y, x) / (c * kappa))``.
    """
    indicator, scale = _ibea_terms(points, kappa)
    count = len(points)
    return [
        sum(-math.exp(-indicator[j][i] / scale) for j in range(count) if j != i)
        for i in range(count)
    ]


def ibea_survival(points: Sequence[Vector], size: int, kappa: float = IBEA_KAPPA) -> Survivors:
    """Select ``size`` points by IBEA's environmental selection.

    The point of smallest fitness (:func:`ibea_fitness`) is removed, and every
    remaining point ``x`` gains ``exp(-I(removed, x) / (c * kappa))``, which
    is exactly its fitness without the removed point; this repeats until
    ``size`` points remain (Zitzler and Künzli 2004). Ties go to the lowest
    index. Each survivor's key is its negated final fitness.

    Returns:
        The survivors in increasing index order.
    """
    indicator, scale = _ibea_terms(points, kappa)
    count = len(points)
    fitness = [
        sum(-math.exp(-indicator[j][i] / scale) for j in range(count) if j != i)
        for i in range(count)
    ]
    alive = list(range(count))
    while len(alive) > max(size, 0):
        worst = min(alive, key=lambda i: fitness[i])
        alive.remove(worst)
        for i in alive:
            fitness[i] += math.exp(-indicator[worst][i] / scale)
    return Survivors(tuple(alive), tuple((-fitness[i],) for i in alive))


# ---------------------------------------------------------------- SMS-EMOA


def hypervolume_contributions(points: Sequence[Vector], reference: Vector) -> list[float]:
    """Return the exclusive hypervolume of every point.

    The contribution of a point is the volume that it alone dominates within
    the box bounded by ``reference``: the hypervolume of the set minus the
    hypervolume of the set without it. Points outside the box, dominated
    points, and repeated points contribute nothing. Two and three objectives
    use exact sweeps; more objectives take differences of
    :func:`dispatchatlas.core.pareto.hypervolume`.

    Args:
        points: Objective vectors in minimisation form.
        reference: The corner of the box, worse than the points in every
            objective.

    Returns:
        One contribution per point.
    """
    bound = tuple(float(value) for value in reference)
    inside = [
        index
        for index, point in enumerate(points)
        if all(value < limit for value, limit in zip(point, bound, strict=True))
    ]
    result = [0.0] * len(points)
    if not inside:
        return result
    dimension = len(bound)
    if dimension == 1:
        values = sorted(points[index][0] for index in inside)
        best = min(inside, key=lambda index: points[index][0])
        runner_up = values[1] if len(values) > 1 else bound[0]
        result[best] = runner_up - values[0]
        return result
    if dimension == 2:
        active = sorted((points[i][0], points[i][1], i) for i in inside)
        for index, area in _exclusive_areas(active, bound[0], bound[1]):
            result[index] = area
        return result
    if dimension == 3:
        return _contributions_3d(points, inside, bound, result)
    kept = [points[index] for index in inside]
    total = hypervolume(kept, bound)
    for position, index in enumerate(inside):
        result[index] = total - hypervolume(kept[:position] + kept[position + 1 :], bound)
    return result


def _exclusive_areas(
    active: Sequence[tuple[float, float, int]], limit_x: float, limit_y: float
) -> list[tuple[int, float]]:
    """Return ``(index, area)`` of the points of a 2-D set that dominate area alone.

    ``active`` holds ``(x, y, index)`` sorted in increasing order. The
    staircase points (not weakly dominated by an earlier point) each own the
    rectangle up to their neighbours on the staircase, less the region that
    the points they dominate cover inside it. Every other point owns nothing.
    """
    stair: list[int] = []
    best = math.inf
    for position, (_, y, _) in enumerate(active):
        if y < best:
            stair.append(position)
            best = y
    found: list[tuple[int, float]] = []
    for number, position in enumerate(stair):
        x, y, index = active[position]
        following = stair[number + 1] if number + 1 < len(stair) else len(active)
        right = active[following][0] if following < len(active) else limit_x
        top = active[stair[number - 1]][1] if number > 0 else limit_y
        area = (right - x) * (top - y)
        covered = [
            (other_x, other_y)
            for other_x, other_y, _ in active[position + 1 : following]
            if other_y < top
        ]
        if covered:
            area -= _area_2d(covered, right, top)
        if area > 0.0:
            found.append((index, area))
    return found


def _area_2d(points: Sequence[tuple[float, float]], limit_x: float, limit_y: float) -> float:
    """Return the area that 2-D points dominate below ``(limit_x, limit_y)``."""
    area = 0.0
    lowest = limit_y
    for x, y in sorted(points):
        if y < lowest:
            area += (limit_x - x) * (lowest - y)
            lowest = y
    return area


def _contributions_3d(
    points: Sequence[Vector],
    inside: Sequence[int],
    bound: tuple[float, ...],
    result: list[float],
) -> list[float]:
    """Sweep the third objective upwards and integrate each point's exclusive 2-D area."""
    order = sorted(inside, key=lambda index: points[index][2])
    active: list[tuple[float, float, int]] = []
    position = 0
    while position < len(order):
        level = points[order[position]][2]
        while position < len(order) and points[order[position]][2] == level:
            index = order[position]
            insort(active, (points[index][0], points[index][1], index))
            position += 1
        upper = points[order[position]][2] if position < len(order) else bound[2]
        depth = upper - level
        if depth > 0.0:
            for index, area in _exclusive_areas(active, bound[0], bound[1]):
                result[index] += depth * area
    return result


def last_front(points: Sequence[Vector]) -> list[int]:
    """Return, in increasing order, the indices of the last non-dominated front.

    It equals the last front of
    :func:`~dispatchatlas.core.pareto.fast_non_dominated_sort` at a lower
    cost: a point's front is one more than the highest front among the
    points that dominate it, and every dominating point precedes the point
    in lexicographic order, so one pass in that order ranks every point.
    """
    vectors = [tuple(point) for point in points]
    order = sorted(range(len(vectors)), key=lambda index: (vectors[index], index))
    rank = [0] * len(vectors)
    for position, index in enumerate(order):
        point = vectors[index]
        highest = -1
        for other in order[:position]:
            if (
                rank[other] > highest
                and vectors[other] != point
                and all(map(operator.le, vectors[other], point))
            ):
                highest = rank[other]
        rank[index] = highest + 1
    top = max(rank)
    return [index for index in range(len(vectors)) if rank[index] == top]


def hypervolume_discard(points: Sequence[Vector], rng: random.Random) -> int:
    """Return the index of the point SMS-EMOA removes from a pool.

    The point comes from the last non-dominated front of the pool
    (Beume, Naujoks and Emmerich 2007). A last front of one point loses it.
    Otherwise the front is scaled to its own range, the best point in each
    objective is kept, as Emmerich, Beume and Naujoks (2005) allow, and the
    point of smallest exact hypervolume contribution
    (:func:`hypervolume_contributions`) is removed, ties at random. The
    reference point lies beyond the front's worst values by
    :data:`HYPERVOLUME_MARGIN` of its range, so the choice does not depend on
    the objectives' units.
    """
    worst = last_front(points)
    if len(worst) == 1:
        return worst[0]
    front = [points[index] for index in worst]
    dimension = len(front[0])
    contributions = hypervolume_contributions(
        normalize(front), [1.0 + HYPERVOLUME_MARGIN] * dimension
    )
    for axis in range(dimension):
        best = min(range(len(front)), key=lambda k: (front[k][axis], tuple(front[k]), k))
        contributions[best] = math.inf
    least = min(contributions)
    tied = [k for k, value in enumerate(contributions) if value == least]
    return worst[tied[rng.randrange(len(tied))] if len(tied) > 1 else tied[0]]


# ------------------------------------------------------------------ MOEA/D


def tchebycheff(vector: Vector, weight: Vector, ideal: Vector, span: Vector | None = None) -> float:
    """Return the weighted Tchebycheff distance of ``vector`` from ``ideal``.

    ``max_i w_i * |f_i - z_i| / s_i`` (Zhang and Li 2007), where each weight
    is at least :data:`WEIGHT_FLOOR`, so that the subproblems on the edge of
    the simplex still prefer Pareto-optimal points, and ``s`` is the
    normalising span of each objective, 1 by default.
    """
    spans = [1.0] * len(vector) if span is None else span
    return max(
        max(w, WEIGHT_FLOOR) * abs(f - z) / s
        for f, w, z, s in zip(vector, weight, ideal, spans, strict=True)
    )


def weight_neighbourhoods(weights: Sequence[Vector], size: int) -> list[list[int]]:
    """Return, for every weight vector, the ``size`` nearest weight vectors.

    Distances are Euclidean, so a vector is its own nearest neighbour and
    comes first. Ties go to the lower index.
    """
    return [
        sorted(range(len(weights)), key=lambda j: (math.dist(weight, weights[j]), j))[:size]
        for weight in weights
    ]


# -------------------------------------------------------------------- RVEA


def unit_vectors(points: Sequence[Vector]) -> list[Point]:
    """Return each point scaled to unit Euclidean length. A zero point is kept as it is."""
    scaled: list[Point] = []
    for point in points:
        norm = math.sqrt(sum(value * value for value in point))
        scaled.append(tuple(value / norm for value in point) if norm > 0.0 else tuple(point))
    return scaled


def smallest_angles(vectors: Sequence[Vector]) -> list[float]:
    """Return the angle between each unit vector and its nearest other vector.

    This is the ``gamma`` that normalises the angle penalty of RVEA. A vector
    without a distinct neighbour, as with one objective, gets 1.
    """
    angles: list[float] = []
    for i, first in enumerate(vectors):
        smallest = math.inf
        for j, second in enumerate(vectors):
            if i != j:
                cosine = sum(a * b for a, b in zip(first, second, strict=True))
                smallest = min(smallest, math.acos(max(-1.0, min(1.0, cosine))))
        angles.append(smallest if 0.0 < smallest < math.inf else 1.0)
    return angles


def adapt_vectors(initial: Sequence[Vector], ideal: Vector, nadir: Vector) -> list[Point]:
    """Rescale reference vectors to the ranges of the objectives.

    RVEA's adaptation (Cheng, Jin, Olhofer and Sendhoff 2016): each initial
    vector is multiplied objective by objective by ``nadir - ideal`` and
    scaled back to unit length. An objective whose range is zero keeps the
    factor 1.
    """
    ranges = [high - low if high > low else 1.0 for low, high in zip(ideal, nadir, strict=True)]
    return unit_vectors(
        [tuple(a * b for a, b in zip(vector, ranges, strict=True)) for vector in initial]
    )


def rvea_survival(
    points: Sequence[Vector],
    vectors: Sequence[Vector],
    gammas: Sequence[float],
    progress: float,
    alpha: float = 2.0,
    violations: Sequence[float] | None = None,
) -> list[int]:
    """Keep at most one point per reference vector by angle-penalised distance.

    RVEA's elitism selection (Cheng, Jin, Olhofer and Sendhoff 2016). Points
    are translated by their ideal point and each joins the unit reference
    vector at the smallest angle ``theta``. Within each group the point of
    smallest angle-penalised distance
    ``(1 + M * progress ** alpha * theta / gamma) * |f'|`` survives, where
    ``M`` is the number of objectives and ``gamma`` the vector's
    :func:`smallest_angles` entry. Early in a run the penalty is small and the
    point closest to the ideal point wins, which favours convergence; late in
    the run the angle dominates, which spreads the points along the vectors.
    With ``violations``, a group keeps its feasible point of smallest distance
    or, if it has none, its point of smallest violation.

    Args:
        points: Objective vectors.
        vectors: Unit reference vectors.
        gammas: Smallest angle between each vector and the others.
        progress: Fraction of the budget used, in ``[0, 1]``.
        alpha: Rate at which the penalty grows with ``progress``.
        violations: Constraint violation of each point, zero when feasible.

    Returns:
        One index per reference vector that attracted a point, in vector order.
    """
    dimension = len(points[0])
    ideal = ideal_point(points)
    penalty = dimension * math.pow(progress, alpha)
    best: dict[int, tuple[tuple[int, float], int]] = {}
    for index, point in enumerate(points):
        shifted = [value - low for value, low in zip(point, ideal, strict=True)]
        norm = math.sqrt(sum(value * value for value in shifted))
        group, angle = 0, 0.0
        if norm > 0.0:
            cosines = [
                sum(a * b for a, b in zip(shifted, vector, strict=True)) / norm
                for vector in vectors
            ]
            group = max(range(len(vectors)), key=lambda j: cosines[j])
            angle = math.acos(max(-1.0, min(1.0, cosines[group])))
        distance = (1.0 + penalty * angle / gammas[group]) * norm
        violation = 0.0 if violations is None else violations[index]
        key = (0, distance) if violation <= 0.0 else (1, violation)
        if group not in best or key < best[group][0]:
            best[group] = (key, index)
    return [best[group][1] for group in sorted(best)]


# ------------------------------------------------------------- AGE-MOEA-II


def minkowski_norm(vector: Vector, p: float) -> float:
    """Return ``(sum |v_i| ** p) ** (1 / p)``, computed without overflow."""
    largest = max((abs(value) for value in vector), default=0.0)
    if largest == 0.0:
        return 0.0
    total = math.fsum(math.pow(abs(value) / largest, p) for value in vector)
    return largest * math.pow(total, 1.0 / p)


def newton_curvature(
    point: Vector, precision: float = NEWTON_PRECISION, steps: int = NEWTON_STEPS
) -> float:
    """Solve ``sum point_i ** p = 1`` for ``p`` by Newton's method.

    The root finder of AGE-MOEA-II (Panichella 2022), as in the author's
    implementation: starting from ``p = 1``, it iterates on
    ``f(p) = log(sum point_i ** p)``, whose derivative is
    ``sum point_i ** p * log(point_i) / sum point_i ** p`` over the positive
    coordinates, until a step moves ``p`` by at most ``precision`` or
    ``steps`` steps are made.

    Returns:
        The estimate, or ``nan`` when the iteration breaks down: no positive
        coordinate, a zero derivative, or overflow.
    """
    positive = [value for value in point if value > 0.0]
    if not positive:
        return math.nan
    logs = [math.log(value) for value in positive]
    x = 1.0
    previous = x
    for _ in range(steps):
        try:
            powers = [math.pow(value, x) for value in positive]
        except OverflowError:
            return math.nan
        total = sum(powers)
        if not 0.0 < total < math.inf:
            return math.nan
        derivative = sum(power * log for power, log in zip(powers, logs, strict=True)) / total
        if derivative == 0.0:
            return math.nan
        x -= math.log(total) / derivative
        if not math.isfinite(x):
            return math.nan
        if abs(x - previous) <= precision:
            break
        previous = x
    return x


def corner_points(front: Sequence[Vector]) -> list[int]:
    """Return, for each objective axis, the point nearest to it.

    A front with no more points than objectives is all corners. Otherwise the
    point at the smallest perpendicular distance from each axis is a corner,
    ties to the lowest index, as in the reference implementation of
    AGE-MOEA-II.
    """
    count, dimension = len(front), len(front[0])
    if count <= dimension:
        return list(range(count))
    corners: list[int] = []
    for axis in range(dimension):
        direction = [1.0 if i == axis else 0.0 for i in range(dimension)]
        corners.append(min(range(count), key=lambda k: perpendicular_distance(front[k], direction)))
    return corners


def front_curvature(front: Sequence[Vector]) -> float:
    """Estimate the exponent ``p`` of the surface ``sum f_i ** p = 1`` through a normalised front.

    AGE-MOEA-II takes the point nearest the origin among those that are not
    corners (:func:`corner_points`) and solves for the ``p`` whose surface
    passes through it (:func:`newton_curvature`): 1 for a linear front, 2 for
    a spherical one, below 1 for a convex one. A failed estimate, a
    non-positive one, or a front without such a point gives 1.
    """
    corners = set(corner_points(front))
    candidates = [k for k in range(len(front)) if k not in corners]
    if not candidates:
        return 1.0
    central = min(candidates, key=lambda k: math.hypot(*front[k]))
    estimate = newton_curvature(front[central])
    return estimate if estimate > 0.0 else 1.0


def project_to_front(point: Vector, p: float) -> Point:
    """Scale ``point`` onto the surface ``sum f_i ** p = 1``. A zero point stays as it is."""
    norm = minkowski_norm([value for value in point if value > 0.0], p)
    if norm <= 0.0:
        return tuple(point)
    return tuple(value / norm for value in point)


def geodesic_distance(first: Vector, second: Vector, p: float) -> float:
    """Return the AGE-MOEA-II distance between two points of the surface ``sum f_i ** p = 1``.

    The midpoint of the two points is projected onto the surface
    (:func:`project_to_front`), and the distance is the sum of the Euclidean
    distances from each point to that projection (Panichella 2022), a
    two-segment approximation of the geodesic that grows with the curvature.
    """
    middle = project_to_front([0.5 * a + 0.5 * b for a, b in zip(first, second, strict=True)], p)
    return math.dist(first, middle) + math.dist(second, middle)


def age_scores(front: Sequence[Vector]) -> tuple[list[float], float]:
    """Return the AGE-MOEA-II survival score of every point of a first front, and ``p``.

    As in the author's implementation: the front is normalised
    (:func:`hyperplane_normalize`), its curvature ``p`` is estimated
    (:func:`front_curvature`), and every point is projected onto the surface
    of that curvature. The extreme points score infinity. The other points
    are ranked greedily: the next one is the point whose two nearest already
    ranked points are farthest in geodesic distance divided by the point's own
    ``p``-norm, and it scores that sum. Every non-extreme point is scored.

    Returns:
        The scores, larger being better, and the curvature ``p``.
    """
    count = len(front)
    normalized, extremes = hyperplane_normalize(front)
    p = front_curvature(normalized)
    scores = [0.0] * count
    ranked = [False] * count
    for index in extremes:
        scores[index] = math.inf
        ranked[index] = True
    projected = [project_to_front(point, p) for point in normalized]
    norms = [minkowski_norm(point, p) for point in normalized]
    distances = [[0.0] * count for _ in range(count)]
    for i in range(count - 1):
        for j in range(i + 1, count):
            distances[i][j] = distances[j][i] = geodesic_distance(projected[i], projected[j], p)
    for i in range(count):
        if norms[i] > 0.0:
            distances[i] = [value / norms[i] for value in distances[i]]
    remaining = [k for k in range(count) if not ranked[k]]
    nearest = {k: sorted(distances[k][j] for j in range(count) if ranked[j])[:2] for k in remaining}
    while remaining:
        chosen = max(remaining, key=lambda k: sum(nearest[k]))
        scores[chosen] = sum(nearest[chosen])
        remaining.remove(chosen)
        for k in remaining:
            nearest[k] = sorted([*nearest[k], distances[k][chosen]])[:2]
    return scores, p


def age_survival(points: Sequence[Vector], size: int) -> Survivors:
    """Select ``size`` points by the environmental selection of AGE-MOEA-II.

    Whole non-dominated fronts are kept in rank order. The first front is
    scored by :func:`age_scores`; every later front by its proximity
    ``1 / |f - z|_p``, the inverse ``p``-norm distance to the ideal point of
    the pool in the objectives' own units, as in the author's implementation.
    The front that does not fit is cut by decreasing score, ties in front
    order (Panichella 2022). Each survivor's key is ``(rank, -score)``.
    """
    fronts = fast_non_dominated_sort(points)
    first_scores, p = age_scores([points[index] for index in fronts[0]])
    ideal = ideal_point(points)
    chosen: list[int] = []
    keys: list[tuple[float, ...]] = []
    for rank, front in enumerate(fronts):
        room = size - len(chosen)
        if room <= 0:
            break
        if rank == 0:
            scores = first_scores
        else:
            scores = []
            for index in front:
                distance = minkowski_norm(
                    [a - b for a, b in zip(points[index], ideal, strict=True)], p
                )
                scores.append(1.0 / distance if distance > 0.0 else math.inf)
        order = list(range(len(front)))
        if len(front) > room:
            order = sorted(order, key=lambda position: -scores[position])[:room]
        for position in order:
            chosen.append(front[position])
            keys.append((float(rank), -scores[position]))
    return Survivors(tuple(chosen), tuple(keys))
