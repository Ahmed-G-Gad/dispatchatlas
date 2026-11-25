"""Reference values for the rank tests.

The expected statistics and p-values were computed once with SciPy 1.18.1
and pasted here; neither the library nor these tests import SciPy:
``scipy.stats.wilcoxon`` (``zero_method="wilcox"``, exact, or asymptotic with
``correction=True``, two-sided or with ``alternative="greater"`` or ``"less"``),
``mannwhitneyu`` (``use_continuity=True``),
``friedmanchisquare``, ``f.sf`` for the Iman-Davenport p-value of SciPy's
Friedman statistic, and ``studentized_range.ppf(1 - alpha, k, inf) / sqrt(2)``
for the Nemenyi table.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import fields, replace

import numpy as np
import pytest
from numpy.typing import NDArray

from dispatchatlas.analysis.stats.distributions import normal_sf
from dispatchatlas.analysis.stats.nonparametric import (
    WILCOXON_EXACT_LIMIT,
    Alternative,
    WilcoxonResult,
    ZeroMethod,
    friedman_statistic,
    friedman_test,
    iman_davenport,
    mann_whitney_u,
    nemenyi_critical_difference,
    nemenyi_q,
    sign_test,
    tie_sum,
    wilcoxon_signed_rank,
)
from dispatchatlas.analysis.stats.ranks import SENSES, midranks, oriented_ranks
from dispatchatlas.core.errors import ConfigError
from tests.analysis.paired_differences import congruential

WILCOXON_EXACT = [
    ([1.0, -2.0, 3.0, 4.0], 2.0, 0.375),
    ([-1.5, 2.25, 3.0, -4.75, 5.5, 6.0, 7.25, 8.5], 5.0, 0.078125),
    (
        [-0.393, 0.641, -1.496, 1.796, 1.038, 0.108, 0.088, 0.704, 0.132, 0.174, 1.12, 0.915],
        16.0,
        0.0771484375,
    ),
    (
        [
            0.236,
            0.215,
            0.461,
            -0.314,
            -0.104,
            0.848,
            0.17,
            -1.074,
            -0.177,
            0.957,
            0.068,
            0.151,
            0.942,
            2.125,
            -0.413,
            1.648,
            -0.93,
            0.475,
            -0.87,
            1.651,
        ],
        68.0,
        0.17685317993164062,
    ),
    (
        [
            1.334,
            1.638,
            -0.386,
            1.185,
            -0.019,
            0.043,
            1.007,
            1.377,
            0.704,
            -0.128,
            -0.326,
            1.944,
            1.094,
            1.22,
            2.683,
            -0.316,
            3.059,
            3.651,
            2.118,
            1.327,
            -0.164,
            1.494,
            0.057,
            0.478,
            0.21,
        ],
        34.0,
        0.00021690130233764648,
    ),
]
WILCOXON_NORMAL = [
    ([1.0, 1.0, 2.0, -2.0, 3.0, 3.0, 3.0, -4.0, 5.0, 6.0], 11.5, 0.11271708474094375),
    (
        [
            0.6,
            1.6,
            -0.3,
            -0.7,
            -0.7,
            -0.7,
            -1.1,
            -0.6,
            1.6,
            -0.3,
            0.6,
            -0.9,
            0.5,
            -1.4,
            -0.4,
            2.6,
            -0.3,
            1.4,
            0.8,
            0.1,
            -0.4,
            1.6,
            0.1,
            1.8,
            -0.4,
            0.4,
            0.8,
            0.7,
            -0.9,
            -0.4,
        ],
        197.0,
        0.47105160603566376,
    ),
    (
        [
            0.0,
            -0.7,
            0.9,
            0.8,
            -1.8,
            -1.6,
            -1.1,
            0.3,
            2.2,
            -0.2,
            0.5,
            -0.9,
            -0.8,
            -1.8,
            0.2,
            1.1,
            -0.2,
            1.6,
            0.4,
            0.2,
            0.7,
            0.7,
            1.3,
            -0.6,
            -2.1,
            0.3,
            -0.4,
            0.2,
            1.9,
            -0.9,
            1.0,
            1.1,
            0.7,
            1.3,
            -0.3,
            -0.5,
            1.2,
            0.9,
            1.6,
            0.4,
        ],
        299.5,
        0.20880143164992093,
    ),
    (
        [
            2.0,
            0.0,
            -1.0,
            -2.0,
            -1.0,
            1.0,
            -2.0,
            -1.0,
            2.0,
            1.0,
            2.0,
            -1.0,
            -3.0,
            1.0,
            -2.0,
            0.0,
            0.0,
            -1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            -1.0,
            1.0,
            -2.0,
            -1.0,
            -2.0,
            0.0,
            1.0,
            1.0,
            3.0,
            1.0,
            -1.0,
            3.0,
            -1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            0.0,
            0.0,
            -1.0,
            1.0,
            -1.0,
            -2.0,
            1.0,
            -1.0,
            0.0,
            2.0,
            1.0,
            1.0,
            1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
        ],
        560.0,
        0.766717184666274,
    ),
]
# Tied samples above the exact limit: concatenations of the three larger samples above, read by
# scipy.stats.wilcoxon(method="asymptotic") of SciPy 1.18.1, with 69, 78, and 87 non-zero pairs.
WILCOXON_NORMAL_LARGE = [
    (WILCOXON_NORMAL[1][0] + WILCOXON_NORMAL[2][0], 967.0, 0.15099689822933146),
    (WILCOXON_NORMAL[1][0] + WILCOXON_NORMAL[3][0], 1361.0, 0.3671104477062823),
    (WILCOXON_NORMAL[2][0] + WILCOXON_NORMAL[3][0], 1673.0, 0.3043594195374256),
]
WILCOXON_TAIL = (0.0, 2.7403472934092292e-67)
MANN_WHITNEY_EXACT = [
    ([1.1, 2.3, 3.7], [4.2, 5.9, 6.1], 0.0, 0.1),
    ([0.5, 2.5, 1.5, 4.5, 3.5], [3.0, 5.0, 6.0, 7.0], 2.0, 0.06349206349206349),
    (
        [-0.204, -3.747, 0.557, 0.115, -0.37, 0.778, 0.266, 0.204, -1.484, -1.459, 0.371, 0.076],
        [
            0.839,
            0.866,
            1.111,
            0.736,
            0.858,
            0.956,
            1.629,
            0.185,
            1.486,
            1.336,
            0.805,
            0.225,
            1.069,
            1.564,
            -0.375,
        ],
        19.0,
        0.00022986839516655105,
    ),
    (
        [
            0.173,
            -1.163,
            0.833,
            2.19,
            0.814,
            -1.448,
            0.095,
            -1.766,
            -1.829,
            0.659,
            0.078,
            -0.687,
            -2.087,
            -1.269,
            0.162,
            0.032,
            0.69,
            1.523,
            -0.339,
            0.469,
        ],
        [
            0.262,
            -0.097,
            0.155,
            1.361,
            -0.628,
            1.541,
            1.877,
            0.436,
            0.655,
            -1.012,
            2.761,
            1.379,
            0.572,
            0.457,
            0.851,
            -0.251,
            0.683,
            -0.524,
            0.897,
            -0.278,
        ],
        133.0,
        0.07177675347139001,
    ),
]
MANN_WHITNEY_NORMAL = [
    ([1.0, 2.0, 2.0, 3.0], [2.0, 4.0, 5.0], 2.0, 0.19908985214820463),
    (
        [
            0.9,
            -0.1,
            0.3,
            3.0,
            0.8,
            0.3,
            0.0,
            -1.0,
            1.1,
            1.6,
            0.1,
            1.7,
            -0.1,
            -1.8,
            -0.7,
            -1.1,
            -0.3,
            0.6,
        ],
        [
            -0.4,
            0.4,
            0.7,
            -0.2,
            -0.4,
            0.6,
            1.4,
            -0.1,
            -0.3,
            0.4,
            0.8,
            0.0,
            0.2,
            0.3,
            -0.6,
            0.3,
            -0.1,
            2.1,
            0.0,
            1.0,
            0.9,
            -0.6,
        ],
        197.0,
        0.9891387414259561,
    ),
    (
        [
            -1.249,
            -0.445,
            0.663,
            -0.573,
            -0.656,
            -2.9,
            -0.397,
            0.312,
            0.445,
            0.429,
            -0.328,
            -0.227,
            0.832,
            -0.136,
            -2.303,
            1.02,
            -1.078,
            1.375,
            1.694,
            0.762,
            0.904,
            1.858,
            -1.102,
            0.254,
            0.239,
        ],
        [
            -0.667,
            2.312,
            -0.281,
            -0.431,
            0.568,
            1.331,
            -0.301,
            -0.3,
            0.804,
            -1.126,
            -0.473,
            -0.538,
            -0.134,
            0.158,
            2.144,
            1.207,
            0.322,
            0.075,
            1.174,
            0.728,
            1.191,
            1.681,
            -0.708,
            -1.953,
            0.278,
            0.379,
            0.587,
            -0.299,
            0.067,
            0.859,
        ],
        329.0,
        0.44183966352473647,
    ),
]
FRIEDMAN = [
    (
        [
            [-1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 1.0],
        ],
        0.10526315789473087,
        0.94872948001644,
        0.044247787610616936,
        0.9569030982425807,
    ),
    (
        [
            [-1.2, 1.0, 3.6, 2.4],
            [0.7, -0.6, 1.0, 2.1],
            [-0.1, 0.4, -1.1, 0.0],
            [-1.8, 2.2, 1.1, 1.6],
            [-0.2, -1.2, 0.8, 0.3],
            [-1.0, 0.4, 0.7, 0.7],
            [-0.8, -0.7, 0.6, 2.1],
            [-1.6, -0.4, 0.4, 0.7],
            [-0.6, 2.2, 1.0, -0.1],
            [-1.9, 0.1, 2.1, 1.1],
        ],
        14.030303030303017,
        0.0028641939877154378,
        7.907020872865261,
        0.0006080062366324196,
    ),
    (
        [
            [-2.3, -1.1, 0.7, 2.0, 1.3, 1.6, 2.9, 3.4],
            [-0.7, 0.7, -0.4, -0.2, 1.8, 0.1, 2.0, 2.4],
            [0.9, -0.9, 1.0, 0.8, 2.7, 0.8, 0.7, 2.6],
            [-0.9, -1.0, 1.5, 1.3, -1.9, 0.6, 0.7, 2.7],
            [1.5, 2.4, 1.3, -0.2, 0.1, 0.5, 2.3, 3.4],
            [-1.9, 0.9, 1.5, 1.2, 4.4, 2.0, 1.8, 2.7],
            [-0.5, -0.8, 1.0, 2.1, 1.0, 2.4, 1.4, 3.7],
            [0.2, 0.8, 1.4, -0.5, 1.0, 3.3, 1.8, 3.5],
            [0.9, -1.5, 1.3, -0.2, 1.5, 1.1, 1.8, 2.0],
            [0.2, -0.1, -0.8, -0.1, 2.1, 3.2, 2.0, 2.9],
            [1.0, -1.1, -1.3, 0.8, -0.7, 1.2, 0.3, 1.5],
            [-0.8, 0.8, 0.0, -0.2, 1.2, 4.5, 2.3, 0.2],
            [0.1, 2.9, 0.8, 2.2, 2.3, 2.7, 2.9, 2.4],
            [-0.2, 0.0, 0.7, 1.3, -0.1, 1.2, 1.8, 2.2],
            [-0.3, 1.1, 0.5, 0.5, 0.0, 0.7, 2.0, 2.2],
            [-2.0, 1.2, -0.1, 0.0, 2.3, 2.7, 1.1, 1.7],
            [-1.2, 1.1, 1.5, 0.5, 2.6, 2.4, 1.7, 2.8],
            [-1.7, 0.0, 0.9, 0.9, 3.2, 1.8, 2.2, 0.8],
            [-0.2, -0.5, 2.9, 0.5, 1.3, 2.7, 2.5, 0.7],
            [0.6, 1.0, 0.4, 1.4, 1.3, 0.1, 1.2, 2.1],
            [-0.6, 0.8, 1.6, 1.2, 0.8, 2.9, 1.9, 2.5],
            [-0.8, 0.1, 1.6, 1.6, 1.0, 1.5, 2.0, 1.2],
            [-0.2, -0.2, -0.8, 1.1, 1.1, 0.0, 1.0, 2.0],
            [-2.9, 0.5, 0.9, 1.5, 0.6, 0.1, 2.4, 0.7],
            [0.5, -0.4, -0.4, 1.2, 2.3, 1.3, 1.6, 0.4],
            [0.8, 0.4, 0.3, 2.2, -0.4, 1.0, 1.8, 1.8],
            [-2.4, 0.6, 0.6, 1.9, 3.2, 1.3, 2.2, 2.4],
            [0.2, 0.2, 1.0, 2.0, 0.4, 3.4, 2.3, 2.4],
            [-2.2, 0.3, -0.4, 1.5, 2.2, 2.6, 1.6, 2.0],
            [0.4, -1.8, 0.2, 0.4, 1.7, 1.0, 2.5, 1.9],
        ],
        84.94171656686629,
        1.3473917791032995e-15,
        19.69729403615401,
        4.985201500513475e-20,
    ),
]
STUDENTIZED_RANGE_Q = {
    0.05: {
        2: 1.9599639845400534,
        3: 2.343700586378409,
        4: 2.569031772546482,
        5: 2.7277743708703763,
        6: 2.8497054196100016,
        7: 2.9483200175296744,
        8: 3.030878449614413,
        9: 3.1017303413033805,
        10: 3.163683577053373,
        11: 3.2186536073291525,
        12: 3.268003924466142,
    },
    0.1: {
        2: 1.6448536269514722,
        3: 2.0522927304967755,
        4: 2.2913414968880566,
        5: 2.4595157642714183,
        6: 2.588520601922348,
        7: 2.6927321009677594,
        8: 2.779883608152978,
        9: 2.8546064311980253,
        10: 2.9198888400615384,
        11: 2.9777682512648735,
        12: 3.0296941831785844,
    },
}


@pytest.mark.parametrize(("differences", "statistic", "p_value"), WILCOXON_EXACT)
def test_wilcoxon_exact_null_matches_scipy(differences, statistic, p_value):
    result = wilcoxon_signed_rank(differences)
    assert result.method == "exact"
    assert result.statistic == statistic
    assert result.p_value == pytest.approx(p_value, rel=1e-12)


@pytest.mark.parametrize(("differences", "statistic", "p_value"), WILCOXON_NORMAL_LARGE)
def test_wilcoxon_tied_normal_approximation_matches_scipy(differences, statistic, p_value):
    result = wilcoxon_signed_rank(differences)
    assert result.pairs > WILCOXON_EXACT_LIMIT
    assert result.method == "normal"
    assert result.statistic == statistic
    assert result.p_value == pytest.approx(p_value, rel=1e-12)


@pytest.mark.parametrize(("differences", "statistic", "normal"), WILCOXON_NORMAL[1:])
def test_tied_samples_of_30_to_48_pairs_take_the_exact_conditional_null(
    differences, statistic, normal
):
    result = wilcoxon_signed_rank(differences)
    assert 25 < result.pairs <= WILCOXON_EXACT_LIMIT
    assert (result.statistic, result.method) == (statistic, "exact-conditional")
    assert result.p_value == pytest.approx(_distribution_p_value(differences)[1], rel=1e-12)
    # SciPy's asymptotic p-value reads the same statistic, and lies near the exact one.
    assert result.p_value == pytest.approx(normal, abs=0.01)


def _enumerated(differences: list[float]) -> tuple[float, float]:
    """Return ``min(W+, W-)`` and its two-sided p-value by listing every sign assignment.

    The magnitudes keep their midranks while each difference takes either
    sign: the p-value is the share of the assignments whose smaller rank sum
    is at most the observed one.
    """
    nonzero = [value for value in differences if abs(value) > 1e-12]
    ranks = midranks([abs(value) for value in nonzero])
    total = math.fsum(ranks)
    observed = math.fsum(rank for rank, value in zip(ranks, nonzero, strict=True) if value > 0.0)
    statistic = min(observed, total - observed)
    extreme = 0
    for signs in itertools.product((False, True), repeat=len(ranks)):
        positive = math.fsum(rank for rank, sign in zip(ranks, signs, strict=True) if sign)
        extreme += min(positive, total - positive) <= statistic + 1e-9
    return statistic, extreme / 2 ** len(ranks)


def _sign_assignments(ranks: Sequence[float]) -> NDArray[np.int64]:
    """Return the number of sign assignments of the ranks at each doubled positive rank sum.

    Each rank ``r`` contributes the polynomial ``1 + x^(2r)``, and midranks are multiples of one
    half, so the exponents are integers. The product by convolution is exact in 64-bit integers
    up to ``2^62`` assignments, a different computation from the library's in-place recursion.
    """
    counts = np.ones(1, dtype=np.int64)
    for rank in ranks:
        doubled = round(2.0 * rank)
        factor = np.zeros(doubled + 1, dtype=np.int64)
        factor[0] = factor[doubled] = 1
        counts = np.convolve(counts, factor)
    return counts


def _distribution_p_value(differences: Sequence[float]) -> tuple[float, float]:
    """Return ``min(W+, W-)`` and its two-sided p-value from the polynomial product."""
    nonzero = [value for value in differences if abs(value) > 1e-12]
    ranks = midranks([abs(value) for value in nonzero])
    total = math.fsum(ranks)
    positive = math.fsum(rank for rank, value in zip(ranks, nonzero, strict=True) if value > 0.0)
    statistic = min(positive, total - positive)
    lower = int(_sign_assignments(ranks)[: round(2.0 * statistic) + 1].sum())
    return statistic, min(1.0, 2 * lower / 2 ** len(ranks))


def _with_negatives_summing_to(size: int, target: int) -> list[float]:
    """Return the differences ``1..size`` whose negative ones sum to ``target``, chosen greedily.

    Taking the largest magnitude that still fits reaches every target from 0 to
    ``size (size + 1) / 2``.
    """
    differences: list[float] = []
    remaining = target
    for magnitude in range(size, 0, -1):
        if magnitude <= remaining:
            differences.append(-float(magnitude))
            remaining -= magnitude
        else:
            differences.append(float(magnitude))
    assert remaining == 0
    return differences


def test_the_polynomial_product_agrees_with_the_enumeration_it_stands_in_for():
    generator = random.Random(25)
    for _ in range(60):
        size = generator.randint(1, 12)
        differences = [float(generator.randint(-5, 5) or 1) for _ in range(size)]
        statistic, p_value = _enumerated(differences)
        assert _distribution_p_value(differences)[0] == statistic
        assert _distribution_p_value(differences)[1] == pytest.approx(p_value, rel=1e-12)
    assert list(_sign_assignments([1.0, 2.0, 3.0])) == [1, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1]


@pytest.mark.parametrize(
    ("differences", "p_value"),
    [
        ([2.0] * 5, 0.0625),
        ([2.0] * 6, 0.03125),
        ([3.0, 3.0], 0.5),
        ([5.0, 5.0, 5.0, 5.0, -1.0], 0.125),
    ],
)
def test_small_tied_samples_use_the_exact_conditional_null(differences, p_value):
    result = wilcoxon_signed_rank(differences)
    assert result.method == "exact-conditional"
    assert result.p_value == p_value
    assert _enumerated(differences)[1] == p_value
    # The smallest p-value n pairs can attain is 2 / 2^n, with or without ties.
    assert result.p_value >= 2 / 2**result.pairs


def test_the_exact_conditional_null_matches_an_enumeration_of_the_sign_assignments():
    generator = random.Random(20)
    tied = 0
    for _ in range(200):
        size = generator.randint(1, 9)
        # Integer differences in a narrow range tie often, as integer makespans do.
        differences = [float(generator.randint(-4, 4)) for _ in range(size)]
        if generator.random() < 0.2:
            differences.append(generator.choice((math.inf, -math.inf)))
            differences.append(generator.choice((math.inf, -math.inf)))
        result = wilcoxon_signed_rank(differences)
        if result.method == "degenerate":
            continue
        statistic, p_value = _enumerated(differences)
        assert result.statistic == statistic
        assert result.p_value == pytest.approx(p_value, rel=1e-12), differences
        tied += result.method == "exact-conditional"
    assert tied > 100


def test_untied_samples_keep_the_exact_null_of_the_ranks_one_to_n():
    generator = random.Random(21)
    for size in range(1, 12):
        differences = [
            generator.choice((-1.0, 1.0)) * (index + generator.random())
            for index in range(1, size + 1)
        ]
        result = wilcoxon_signed_rank(differences)
        assert result.method == "exact"
        assert result.p_value == pytest.approx(_enumerated(differences)[1], rel=1e-12)
    # The count of sign assignments by rank sum over 1..n, as Wilcoxon tabulates it: with
    # n = 5 and W = 2 the sums 0, 1, 2 have 1, 1, 1 assignments of 32.
    assert wilcoxon_signed_rank([1.0, -2.0, 3.0, 4.0, 5.0]).p_value == 2 * 3 / 32


@pytest.mark.parametrize("size", range(26, 51))
def test_untied_samples_of_26_to_50_pairs_take_the_exact_null(size):
    generator = random.Random(size)
    # Magnitudes in [index, index + 1) never tie; a third of the signs are negative.
    differences = [
        (-1.0 if generator.random() < 0.35 else 1.0) * (index + generator.random())
        for index in range(1, size + 1)
    ]
    result = wilcoxon_signed_rank(differences)
    statistic, p_value = _distribution_p_value(differences)
    assert (result.method, result.pairs, result.zeros) == ("exact", size, 0)
    assert result.statistic == statistic
    assert result.p_value == pytest.approx(p_value, rel=1e-12)


@pytest.mark.parametrize("size", [26, 32, 50])
def test_the_exact_null_of_26_to_50_pairs_has_its_extremes_and_its_centre(size):
    total = size * (size + 1) // 2
    one_sign = wilcoxon_signed_rank([float(index) for index in range(1, size + 1)])
    assert (one_sign.statistic, one_sign.method) == (0.0, "exact")
    assert one_sign.p_value == 2 / 2**size
    # The smallest p-value over a larger sample is smaller: the exact floor falls with n.
    smaller = wilcoxon_signed_rank([float(index) for index in range(1, size)])
    assert one_sign.p_value < smaller.p_value
    # The p-value never falls as the smaller rank sum grows, and reaches 1 at the centre.
    previous = 0.0
    for statistic in range(0, total // 2 + 1, max(1, total // 40)):
        result = wilcoxon_signed_rank(_with_negatives_summing_to(size, statistic))
        assert (result.statistic, result.method) == (float(statistic), "exact")
        assert result.p_value >= previous
        previous = result.p_value
    centre = wilcoxon_signed_rank(_with_negatives_summing_to(size, total // 2))
    assert centre.p_value == 1.0


def test_32_untied_pairs_with_a_negative_rank_sum_of_159_are_significant_exactly():
    # Negative magnitudes 32, 31, 30, 29, 28, and 9 sum to 159 of the total 528.
    differences = _with_negatives_summing_to(32, 159)
    negatives = sorted(-value for value in differences if value < 0.0)
    assert negatives == [9.0, 28.0, 29.0, 30.0, 31.0, 32.0]
    result = wilcoxon_signed_rank(differences)
    assert (result.statistic, result.pairs, result.method) == (159.0, 32, "exact")
    assert result.p_value == pytest.approx(0.04977, abs=5e-6)
    assert result.p_value < 0.05
    # The continuity-corrected normal approximation of the same statistic misses 0.05.
    variance = 32 * 33 * 65 / 24.0
    z = (abs(159.0 - 32 * 33 / 4.0) - 0.5) / math.sqrt(variance)
    assert 2.0 * normal_sf(z) > 0.05


def test_tied_samples_of_32_pairs_take_the_exact_conditional_null():
    generator = random.Random(32)
    # Integer differences of 1 to 12 tie often, as integer makespans do.
    differences = [generator.choice((-1.0, 1.0)) * generator.randint(1, 12) for _ in range(32)]
    assert len(set(map(abs, differences))) < 32
    result = wilcoxon_signed_rank(differences)
    statistic, p_value = _distribution_p_value(differences)
    assert (result.method, result.pairs, result.zeros) == ("exact-conditional", 32, 0)
    assert result.statistic == statistic
    assert result.p_value == pytest.approx(p_value, rel=1e-12)
    assert result.p_value >= 2 / 2**32
    # Ties in magnitude do not move the exact floor: all differences positive gives 2 / 2^n.
    assert wilcoxon_signed_rank([float(1 + index // 2) for index in range(32)]).p_value == 2 / 2**32


def test_a_tied_sample_at_the_exact_limit_is_exact_and_one_above_it_is_normal():
    at_limit = [float(1 + index // 2) for index in range(WILCOXON_EXACT_LIMIT)]
    at_limit[0] = -at_limit[0]
    assert wilcoxon_signed_rank(at_limit).method == "exact-conditional"
    assert wilcoxon_signed_rank([*at_limit, 40.0]).method == "normal"


def test_ten_tied_pairs_no_longer_take_the_normal_approximation():
    differences, statistic, normal = WILCOXON_NORMAL[0]
    result = wilcoxon_signed_rank(differences)
    assert (result.statistic, result.method) == (statistic, "exact-conditional")
    assert result.p_value == pytest.approx(_enumerated(differences)[1], rel=1e-12)
    assert result.p_value > normal


def test_wilcoxon_normal_tail_keeps_tiny_p_values():
    statistic, p_value = WILCOXON_TAIL
    result = wilcoxon_signed_rank([index + 0.5 for index in range(400)])
    assert result.statistic == statistic
    assert result.p_value == pytest.approx(p_value, rel=1e-12)
    assert result.p_value > 0.0


def test_wilcoxon_hand_enumerated_values():
    # Signs over ranks 1..4 give 16 equally likely sums; W- = 2 has 3 with W <= 2.
    assert wilcoxon_signed_rank([1.0, -2.0, 3.0, 4.0]).p_value == pytest.approx(6 / 16)
    assert wilcoxon_signed_rank([-1, -2, -3, -4, -5]).p_value == pytest.approx(2 / 32)


def test_wilcoxon_drops_zeros_and_float_residual_ties():
    result = wilcoxon_signed_rank([0.0, 1e-13, -2.0, 3.0, 4.0])
    assert result.zeros == 2
    assert result.pairs == 3
    degenerate = wilcoxon_signed_rank([0.0, 0.0])
    assert (degenerate.method, degenerate.p_value, degenerate.statistic) == ("degenerate", 1.0, 0.0)


def test_wilcoxon_ranks_infinite_differences_above_finite_ones():
    result = wilcoxon_signed_rank([math.inf, math.inf, 1.0, 2.0, 3.0, 4.0, 5.0, -6.0])
    assert result.method == "exact-conditional"
    assert result.statistic == 6.0


def test_wilcoxon_rejects_empty_and_nan_input():
    with pytest.raises(ConfigError):
        wilcoxon_signed_rank([])
    with pytest.raises(ConfigError):
        wilcoxon_signed_rank([1.0, math.nan])


def test_sign_test_is_the_exact_binomial_tail():
    assert sign_test(5, 0) == pytest.approx(2 / 32)
    assert sign_test(0, 5) == sign_test(5, 0)
    assert sign_test(7, 3) == pytest.approx(2 * (1 + 10 + 45 + 120) / 1024)
    assert sign_test(0, 0) == 1.0
    assert sign_test(3, 3) == 1.0
    assert sign_test(1000, 0) == pytest.approx(2.0**-999, rel=1e-12)
    with pytest.raises(ConfigError):
        sign_test(-1, 2)


@pytest.mark.parametrize(("first", "second", "statistic", "p_value"), MANN_WHITNEY_EXACT)
def test_mann_whitney_exact_null_matches_scipy(first, second, statistic, p_value):
    result = mann_whitney_u(first, second)
    assert result.method == "exact"
    assert result.statistic == statistic
    assert result.p_value == pytest.approx(p_value, rel=1e-12)
    assert mann_whitney_u(second, first).p_value == pytest.approx(result.p_value, rel=1e-12)


@pytest.mark.parametrize(("first", "second", "statistic", "p_value"), MANN_WHITNEY_NORMAL)
def test_mann_whitney_tied_or_large_samples_match_scipy(first, second, statistic, p_value):
    result = mann_whitney_u(first, second)
    assert result.method == "normal"
    assert result.statistic == statistic
    assert result.p_value == pytest.approx(p_value, rel=1e-12)


def test_mann_whitney_rejects_an_empty_sample():
    with pytest.raises(ConfigError):
        mann_whitney_u([], [1.0])


@pytest.mark.parametrize(("rows", "q", "p_value", "f", "f_p_value"), FRIEDMAN)
def test_friedman_and_iman_davenport_match_scipy_with_ties(rows, q, p_value, f, f_p_value):
    blocks = [{f"s{column}": value for column, value in enumerate(row)} for row in rows]
    result = friedman_test(blocks)
    assert result.statistic == pytest.approx(q, rel=1e-12)
    assert result.p_value == pytest.approx(p_value, rel=1e-12)
    corrected = iman_davenport(result)
    assert corrected.statistic == pytest.approx(f, rel=1e-12)
    assert corrected.p_value == pytest.approx(f_p_value, rel=1e-12)
    assert corrected.df_numerator == result.groups - 1
    assert corrected.df_denominator == (result.groups - 1) * (result.blocks - 1)


def test_friedman_consistent_ordering_and_its_iman_davenport_limit():
    blocks = [{"a": 1.0, "b": 2.0, "c": 3.0}] * 6
    result = friedman_test(blocks)
    assert result.statistic == pytest.approx(12.0)
    assert result.average_ranks == (("a", 1.0), ("b", 2.0), ("c", 3.0))
    assert result.kendalls_w == pytest.approx(1.0)
    maximised = friedman_test(blocks, "max")
    assert maximised.average_ranks == (("a", 3.0), ("b", 2.0), ("c", 1.0))
    corrected = iman_davenport(result)
    assert corrected.statistic == math.inf
    assert corrected.p_value == 0.0


def test_friedman_ranks_failures_last():
    blocks = [{"a": math.inf, "b": 2.0, "c": 3.0}, {"a": math.inf, "b": 5.0, "c": math.inf}]
    ranks = dict(friedman_test(blocks).average_ranks)
    assert ranks == {"a": 2.75, "b": 1.0, "c": 2.25}


def test_friedman_refuses_incomplete_or_small_designs():
    with pytest.raises(ConfigError, match="same groups"):
        friedman_test([{"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0}])
    with pytest.raises(ConfigError, match="two blocks"):
        friedman_test([{"a": 1.0, "b": 2.0}])
    with pytest.raises(ConfigError, match="two groups"):
        friedman_test([{"a": 1.0}, {"a": 2.0}])


def test_friedman_statistic_from_rank_sums_and_tie_sum():
    assert tie_sum([1.0, 1.0, 2.0, 3.0, 3.0, 3.0]) == 6.0 + 24.0
    assert friedman_statistic([6.0, 12.0, 18.0], 6) == pytest.approx(12.0)


def test_tie_sum_groups_values_equal_within_the_tolerance_as_one_group():
    nearly_equal = [0.5 + index * 1e-15 for index in range(6)]
    assert len(set(nearly_equal)) == 6
    assert tie_sum(nearly_equal) == 6**3 - 6
    assert tie_sum(nearly_equal, "max") == 6**3 - 6
    # Groups of 2, 1, and 2: the values 1 and 1 + 1e-13 tie, 2 stands alone, 3 and 3 tie.
    assert tie_sum([3.0, 1.0, 2.0, 1.0 + 1e-13, 3.0]) == 6.0 + 6.0
    # A relative tolerance applies to large values, an absolute one near zero.
    assert tie_sum([1e9, 1e9 + 1e-4, -1e9]) == 6.0
    assert tie_sum([0.0, 1e-13, -1e-13, 1.0]) == 24.0
    assert tie_sum([math.inf, math.inf, 1.0, -math.inf]) == 6.0
    assert tie_sum([]) == 0.0
    assert tie_sum([1.0, 2.0, 3.0]) == 0.0


def test_tie_sum_forms_the_groups_of_the_midranks_in_either_orientation():
    # The tolerance does not chain: a group takes the values within it of its first value, which
    # is the smallest when ranking ascending and the largest when ranking descending.
    chain = [0.0, 0.3e-12, 1.05e-12, 1.1e-12]
    assert tie_sum(chain, "min") == 2 * (2**3 - 2)
    assert tie_sum(chain, "max") == 3**3 - 3
    generator = random.Random(23)
    for _ in range(50):
        values = [generator.choice((0.0, 0.3, 0.6, 1.1)) * 1e-12 for _ in range(8)]
        for sense in SENSES:
            ranks = oriented_ranks(values, sense)
            groups = [ranks.count(rank) for rank in set(ranks)]
            assert tie_sum(values, sense) == sum(size**3 - size for size in groups)
    with pytest.raises(ConfigError, match="sense"):
        tie_sum([1.0, 2.0], "median")  # type: ignore[arg-type]


def _noisy(values: Sequence[float], seed: int) -> list[float]:
    """Return ``values`` with float-level noise that ties by the rank tolerance, not by equality."""
    generator = random.Random(seed)
    return [value * (1.0 + 1e-15 * generator.random()) for value in values]


def test_the_wilcoxon_variance_ties_values_by_the_rank_tolerance():
    for differences, _, _ in WILCOXON_NORMAL_LARGE:
        exact = wilcoxon_signed_rank(differences)
        noisy = wilcoxon_signed_rank(_noisy(differences, 3))
        assert noisy.method == "normal"
        assert noisy.statistic == exact.statistic
        assert noisy.p_value == pytest.approx(exact.p_value, rel=1e-12)


def test_the_mann_whitney_variance_ties_values_by_the_rank_tolerance():
    generator = random.Random(24)
    first = [float(generator.randint(1, 9)) for _ in range(25)]
    second = [float(generator.randint(2, 11)) for _ in range(30)]
    exact = mann_whitney_u(first, second)
    noisy = mann_whitney_u(_noisy(first, 1), _noisy(second, 2))
    assert (exact.method, noisy.method) == ("normal", "normal")
    assert noisy.statistic == exact.statistic
    assert noisy.p_value == pytest.approx(exact.p_value, rel=1e-12)


@pytest.mark.parametrize("sense", ["min", "max"])
def test_the_friedman_statistic_ties_values_by_the_rank_tolerance(sense):
    generator = random.Random(26)
    rows = [[float(generator.randint(1, 4)) for _ in range(5)] for _ in range(12)]
    exact = friedman_test(
        [{f"s{column}": v for column, v in enumerate(row)} for row in rows], sense
    )
    noisy_rows = [_noisy(row, index) for index, row in enumerate(rows)]
    noisy = friedman_test(
        [{f"s{column}": v for column, v in enumerate(row)} for row in noisy_rows], sense
    )
    assert noisy.average_ranks == exact.average_ranks
    assert noisy.statistic == pytest.approx(exact.statistic, rel=1e-12)
    assert noisy.p_value == pytest.approx(exact.p_value, rel=1e-12)


def test_nemenyi_table_matches_the_studentized_range():
    for alpha, table in STUDENTIZED_RANGE_Q.items():
        for groups, q in table.items():
            # The table keeps Demšar's three printed decimals.
            assert nemenyi_q(groups, alpha) == pytest.approx(q, abs=7.1e-4)


def test_nemenyi_critical_difference_value_and_alpha_lookup():
    expected = 2.569 * math.sqrt(4 * 5 / (6.0 * 10))
    assert nemenyi_critical_difference(4, 10) == pytest.approx(expected)
    assert nemenyi_critical_difference(4, 10, alpha=1 - 0.95) == pytest.approx(expected)
    with pytest.raises(ConfigError, match="2 to 12"):
        nemenyi_critical_difference(13, 10)
    with pytest.raises(ConfigError, match="alpha"):
        nemenyi_critical_difference(4, 10, alpha=0.01)
    with pytest.raises(ConfigError, match="block"):
        nemenyi_critical_difference(4, 0)


# ------------------------------------------------ one-sided signed-rank tests

ONE_SIDED: tuple[Alternative, ...] = ("greater", "less")

# The two-sided test on `_battery`, as the implementation computed it before it took an
# alternative: (name, statistic, p-value, pairs, zeros, method), to the last bit.
WILCOXON_TWO_SIDED_BEFORE = [
    ("one positive pair", 0.0, 1.0, 1, 0, "exact"),
    ("one negative pair", 0.0, 1.0, 1, 0, "exact"),
    ("four untied pairs", 2.0, 0.375, 4, 0, "exact"),
    ("eight untied pairs", 5.0, 0.078125, 8, 0, "exact"),
    ("ties and zeros", 11.5, 0.40625, 8, 1, "exact-conditional"),
    ("infinities", 6.0, 0.109375, 8, 0, "exact-conditional"),
    ("only zeros", 0.0, 1.0, 0, 2, "degenerate"),
    ("fifty untied pairs", 408.0, 0.02616696817119646, 50, 0, "exact"),
    ("fifty tied pairs", 325.5, 0.00212439779457263, 50, 0, "exact-conditional"),
    ("sixty tied pairs", 855.5, 1.0, 58, 2, "normal"),
    ("eighty untied pairs", 616.0, 1.4862607617949749e-06, 80, 0, "normal"),
]
# (differences, W+, p-value of "greater", p-value of "less"): scipy.stats.wilcoxon with
# alternative="greater" and "less" and method="exact", whose statistic for a one-sided
# alternative is W+. The last differences are congruential(...), see paired_differences.py.
WILCOXON_ONE_SIDED_EXACT = [
    ([0.434, 1.76, 1.625, -0.11, 0.102], 13.0, 0.09375, 0.9375),
    ([0.293, 1.346, 1.042, 1.024, 1.919, -0.906, -0.327, -1.021], 27.0, 0.125, 0.90234375),
    (
        [-0.547, 0.419, 0.404, -0.794, -0.668, -2.107, 1.379, -0.524, 1.037, 0.117, 1.644, 1.237],
        44.0,
        0.36669921875,
        0.661376953125,
    ),
    (
        [
            1.046,
            -0.629,
            -1.223,
            -2.575,
            -0.003,
            1.598,
            0.383,
            0.862,
            1.373,
            -0.532,
            3.002,
            -0.527,
            0.724,
            3.09,
            0.236,
            0.464,
            -0.162,
            0.685,
            -1.777,
            -0.297,
        ],
        130.0,
        0.18413829803466797,
        0.8255949020385742,
    ),
    (
        [
            -1.231,
            -0.737,
            0.594,
            -0.324,
            0.726,
            1.007,
            -1.244,
            1.217,
            -0.396,
            2.294,
            0.929,
            -0.339,
            0.401,
            0.415,
            0.53,
            1.593,
            0.733,
            1.861,
            0.834,
            0.022,
            -1.324,
            -1.361,
            0.405,
            0.891,
            1.643,
        ],
        226.0,
        0.045158058404922485,
        0.9574304819107056,
    ),
    (congruential(50, 37, 101, 40, 0.05), 841.0, 0.024723203400300875, 0.9758452309200818),
]
# The same with method="approx", correction=True, and zero_method="wilcox": 60 and 80 pairs
# without ties, and 70 integer pairs with ties and zeros.
WILCOXON_ONE_SIDED_NORMAL = [
    (congruential(60, 43, 97, 38, 0.04), 1177.0, 0.02711097717474736, 0.9733462048559777),
    (congruential(80, 29, 103, 55, 0.03), 1341.0, 0.9099686929756485, 0.09081288741035598),
    (
        [float((index * 37 + 17) % 15 - 6) for index in range(70)],
        1331.5,
        0.04523513240554112,
        0.9553847898779431,
    ),
]


def _battery() -> dict[str, list[float]]:
    """Return the differences that WILCOXON_TWO_SIDED_BEFORE was computed on."""
    return {
        "one positive pair": [3.0],
        "one negative pair": [-3.0],
        "four untied pairs": [1.0, -2.0, 3.0, 4.0],
        "eight untied pairs": [-1.5, 2.25, 3.0, -4.75, 5.5, 6.0, 7.25, 8.5],
        "ties and zeros": [0.0, 1.0, 1.0, -2.0, 2.0, 3.0, 3.0, 3.0, -4.0],
        "infinities": [math.inf, math.inf, 1.0, 2.0, 3.0, 4.0, 5.0, -6.0],
        "only zeros": [0.0, 0.0],
        "fifty untied pairs": [
            (-1.0 if index % 3 == 0 else 1.0) * (index + 0.5) for index in range(1, 51)
        ],
        "fifty tied pairs": [
            (-1.0 if index % 4 == 0 else 1.0) * (1 + index // 3) for index in range(50)
        ],
        "sixty tied pairs": [((7 * index) % 23 - 11) * 0.5 for index in range(60)],
        "eighty untied pairs": [
            (-1.0 if index % 5 == 0 else 1.0) * (index + 1 + 0.25 * (index % 4))
            for index in range(80)
        ],
    }


def _one_sided_enumeration(differences: Sequence[float]) -> tuple[float, float, float, float]:
    """Return ``W+`` and ``P(W+ >= w+)``, ``P(W+ <= w+)``, ``P(W+ = w+)`` over the sign assignments.

    The probabilities come from the polynomial product of `_sign_assignments`, a computation
    apart from the library's in-place recursion.
    """
    nonzero = [value for value in differences if abs(value) > 1e-12]
    ranks = midranks([abs(value) for value in nonzero])
    positive = math.fsum(rank for rank, value in zip(ranks, nonzero, strict=True) if value > 0.0)
    counts = _sign_assignments(ranks)
    observed = round(2.0 * positive)
    total = 2 ** len(ranks)
    return (
        positive,
        int(counts[observed:].sum()) / total,
        int(counts[: observed + 1].sum()) / total,
        int(counts[observed]) / total,
    )


@pytest.mark.parametrize(
    ("name", "statistic", "p_value", "pairs", "zeros", "method"), WILCOXON_TWO_SIDED_BEFORE
)
def test_two_sided_results_are_unchanged_by_the_new_arguments(
    name, statistic, p_value, pairs, zeros, method
):
    differences = _battery()[name]
    for result in (
        wilcoxon_signed_rank(differences),
        wilcoxon_signed_rank(differences, alternative="two-sided"),
        wilcoxon_signed_rank(differences, zero_method="wilcox"),
        wilcoxon_signed_rank(differences, alternative="two-sided", zero_method="wilcox"),
    ):
        assert (result.alternative, result.zero_method) == ("two-sided", "wilcox")
        found = (result.statistic, result.p_value, result.pairs, result.zeros, result.method)
        # The normal approximation goes through libm's erfc, which differs in the last digits
        # between platforms; every other method is exact arithmetic and keeps its bits.
        wanted = pytest.approx(p_value, rel=1e-12) if method == "normal" else p_value
        assert found == (statistic, wanted, pairs, zeros, method)


def test_the_result_keeps_its_five_fields_in_order_and_adds_two_defaulted_ones_last():
    names = [field.name for field in fields(WilcoxonResult)]
    assert names == [
        "statistic",
        "p_value",
        "pairs",
        "zeros",
        "method",
        "alternative",
        "zero_method",
    ]
    legacy = WilcoxonResult(2.0, 0.375, 4, 0, "exact")
    assert (legacy.alternative, legacy.zero_method) == ("two-sided", "wilcox")
    assert legacy == wilcoxon_signed_rank([1.0, -2.0, 3.0, 4.0])


@pytest.mark.parametrize(("differences", "statistic", "greater", "less"), WILCOXON_ONE_SIDED_EXACT)
@pytest.mark.parametrize("zero_method", ["wilcox", "pratt"])
def test_one_sided_exact_null_matches_scipy(differences, statistic, greater, less, zero_method):
    # Without zeros, SciPy's zero_method="pratt" gives these same constants.
    for alternative, expected in zip(ONE_SIDED, (greater, less), strict=True):
        result = wilcoxon_signed_rank(differences, alternative=alternative, zero_method=zero_method)
        assert (result.method, result.alternative, result.zeros) == ("exact", alternative, 0)
        assert result.zero_method == zero_method
        assert result.statistic == statistic
        assert result.p_value == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(("differences", "statistic", "greater", "less"), WILCOXON_ONE_SIDED_NORMAL)
def test_one_sided_normal_approximation_matches_scipy(differences, statistic, greater, less):
    for alternative, expected in zip(ONE_SIDED, (greater, less), strict=True):
        result = wilcoxon_signed_rank(differences, alternative=alternative)
        assert result.pairs > WILCOXON_EXACT_LIMIT
        assert (result.method, result.alternative) == ("normal", alternative)
        assert result.statistic == statistic
        assert result.p_value == pytest.approx(expected, rel=1e-11, abs=0.0)


def test_one_sided_exact_values_match_the_enumeration_with_ties_zeros_and_infinities():
    generator = random.Random(33)
    tied = 0
    for _ in range(200):
        size = generator.randint(1, 9)
        # Integer differences in a narrow range tie often, as integer makespans do.
        differences = [float(generator.randint(-4, 4)) for _ in range(size)]
        if generator.random() < 0.2:
            differences.append(generator.choice((math.inf, -math.inf)))
            differences.append(generator.choice((math.inf, -math.inf)))
        up = wilcoxon_signed_rank(differences, alternative="greater")
        down = wilcoxon_signed_rank(differences, alternative="less")
        if up.method == "degenerate":
            assert (up.p_value, down.p_value, up.statistic) == (1.0, 1.0, 0.0)
            continue
        positive, p_greater, p_less, point = _one_sided_enumeration(differences)
        assert up.statistic == down.statistic == positive
        assert up.p_value == pytest.approx(p_greater, rel=1e-12, abs=0.0), differences
        assert down.p_value == pytest.approx(p_less, rel=1e-12, abs=0.0), differences
        # The two tails share the observed value, so they sum to one plus its probability.
        assert up.p_value + down.p_value == pytest.approx(1.0 + point, rel=1e-12)
        tied += up.method == "exact-conditional"
    assert tied > 100


def test_one_sided_hand_enumerated_values():
    # Ranks 1 to 4 with signs + - + +: W+ = 8 of 10. Of the 16 equally likely sign assignments,
    # the positive rank sums of at least 8 are {1,3,4} = 8, {2,3,4} = 9, and {1,2,3,4} = 10,
    # and those of at most 8 are all but the last two.
    up = wilcoxon_signed_rank([1.0, -2.0, 3.0, 4.0], alternative="greater")
    down = wilcoxon_signed_rank([1.0, -2.0, 3.0, 4.0], alternative="less")
    assert (up.statistic, up.p_value, up.method) == (8.0, 3 / 16, "exact")
    assert (down.statistic, down.p_value, down.method) == (8.0, 14 / 16, "exact")
    # The two-sided test, on min(W+, W-) = 2, doubles the smaller tail.
    assert wilcoxon_signed_rank([1.0, -2.0, 3.0, 4.0]).p_value == 6 / 16
    # One pair: a positive difference gives W+ = 1 of 1, and the negative one W+ = 0.
    assert wilcoxon_signed_rank([3.0], alternative="greater").p_value == 0.5
    assert wilcoxon_signed_rank([3.0], alternative="less").p_value == 1.0
    assert wilcoxon_signed_rank([-3.0], alternative="greater").p_value == 1.0
    assert wilcoxon_signed_rank([-3.0], alternative="less").p_value == 0.5


@pytest.mark.parametrize("zero_method", ["wilcox", "pratt"])
@pytest.mark.parametrize("name", sorted(_battery()))
def test_a_less_test_is_the_greater_test_of_the_negated_differences(name, zero_method):
    differences = _battery()[name]
    negated = [-value for value in differences]
    down = wilcoxon_signed_rank(differences, alternative="less", zero_method=zero_method)
    mirror = wilcoxon_signed_rank(negated, alternative="greater", zero_method=zero_method)
    assert down.method == mirror.method
    assert down.p_value == pytest.approx(mirror.p_value, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("zero_method", ["wilcox", "pratt"])
@pytest.mark.parametrize("name", sorted(_battery()))
def test_the_two_sided_p_value_is_twice_the_smaller_one_sided_tail(name, zero_method):
    differences = _battery()[name]
    both = wilcoxon_signed_rank(differences, zero_method=zero_method).p_value
    tails = [
        wilcoxon_signed_rank(differences, alternative=a, zero_method=zero_method).p_value
        for a in ("greater", "less")
    ]
    assert both == pytest.approx(min(1.0, 2.0 * min(tails)), rel=1e-12, abs=0.0)


def test_the_smallest_one_sided_p_value_is_one_over_two_to_the_pairs():
    for size in (1, 5, 12, 50):
        ranks = [float(index) for index in range(1, size + 1)]
        up = wilcoxon_signed_rank(ranks, alternative="greater")
        down = wilcoxon_signed_rank([-value for value in ranks], alternative="less")
        assert (up.p_value, down.p_value) == (1 / 2**size, 1 / 2**size)
        assert wilcoxon_signed_rank(ranks, alternative="less").p_value == 1.0
    # Tied magnitudes do not move the floor.
    tied = wilcoxon_signed_rank([2.0] * 5, alternative="greater")
    assert (tied.method, tied.p_value) == ("exact-conditional", 1 / 32)


def test_greater_reads_the_upper_tail_and_less_the_lower_one():
    skewed = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, -0.25]
    assert wilcoxon_signed_rank(skewed, alternative="greater").p_value < 0.01
    assert wilcoxon_signed_rank(skewed, alternative="less").p_value > 0.99
    flipped = [-value for value in skewed]
    assert wilcoxon_signed_rank(flipped, alternative="greater").p_value > 0.99
    assert wilcoxon_signed_rank(flipped, alternative="less").p_value < 0.01


def test_one_sided_tests_drop_zeros_as_the_two_sided_test_does():
    with_zeros = [0.0, 0.0, 1.0, -2.0, 3.0, 4.0, 1e-13]
    without = [1.0, -2.0, 3.0, 4.0]
    for alternative in ("greater", "less"):
        found = wilcoxon_signed_rank(with_zeros, alternative=alternative)
        expected = wilcoxon_signed_rank(without, alternative=alternative)
        assert (found.zeros, found.pairs) == (3, 4)
        assert (found.statistic, found.p_value, found.method) == (
            expected.statistic,
            expected.p_value,
            expected.method,
        )


@pytest.mark.parametrize("zero_method", ["wilcox", "pratt"])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_a_degenerate_result_keeps_its_arguments(alternative, zero_method):
    result = wilcoxon_signed_rank([0.0, 1e-14], alternative=alternative, zero_method=zero_method)
    found = (result.statistic, result.p_value, result.pairs, result.zeros, result.method)
    assert found == (0.0, 1.0, 0, 2, "degenerate")
    assert (result.alternative, result.zero_method) == (alternative, zero_method)


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_one_sided_tests_switch_to_the_normal_approximation_above_the_exact_limit(alternative):
    at_limit = [float(1 + index // 2) for index in range(WILCOXON_EXACT_LIMIT)]
    at_limit[0] = -at_limit[0]
    assert wilcoxon_signed_rank(at_limit, alternative=alternative).method == "exact-conditional"
    assert wilcoxon_signed_rank([*at_limit, 40.0], alternative=alternative).method == "normal"


def test_the_one_sided_normal_approximation_lies_near_the_exact_value_at_the_limit():
    differences = congruential(WILCOXON_EXACT_LIMIT, 37, 101, 40, 0.05)
    exact = wilcoxon_signed_rank(differences, alternative="greater")
    assert (exact.method, exact.statistic) == ("exact", 841.0)
    size = WILCOXON_EXACT_LIMIT
    mean = size * (size + 1) / 4.0
    sigma = math.sqrt(size * (size + 1) * (2 * size + 1) / 24.0)
    assert exact.p_value == pytest.approx(normal_sf((841.0 - mean - 0.5) / sigma), abs=1e-3)


def test_one_sided_normal_tails_keep_tiny_p_values():
    differences = [index + 0.5 for index in range(400)]
    up = wilcoxon_signed_rank(differences, alternative="greater")
    # The two-sided p-value doubles this tail, so the one-sided one is half of it.
    assert up.p_value == pytest.approx(WILCOXON_TAIL[1] / 2.0, rel=1e-12, abs=0.0)
    assert up.p_value > 0.0
    assert wilcoxon_signed_rank(differences, alternative="less").p_value == 1.0


@pytest.mark.parametrize(
    "alternative", ["two_sided", "Greater", "greater ", "", None, 1, ("less",)]
)
def test_an_unknown_alternative_is_refused(alternative):
    with pytest.raises(ConfigError, match="alternative"):
        wilcoxon_signed_rank([1.0, 2.0, 3.0], alternative=alternative)


def test_the_alternative_is_keyword_only_and_input_errors_still_apply():
    with pytest.raises(TypeError):
        wilcoxon_signed_rank([1.0, 2.0], "greater")  # type: ignore[call-arg]
    for alternative in ("two-sided", "greater", "less"):
        with pytest.raises(ConfigError):
            wilcoxon_signed_rank([], alternative=alternative)
        with pytest.raises(ConfigError):
            wilcoxon_signed_rank([1.0, math.nan], alternative=alternative)


# ------------------------------------------------------ Pratt's treatment of zeros

# (differences, two-sided statistic, two-sided p-value, W+, p-value of "greater", p-value of
# "less"): scipy.stats.wilcoxon with zero_method="pratt", method="approx", and correction=True,
# on 100 and 120 integer pairs with ties and 4 and 13 zeros, and on 80 pairs of which 9 are
# zeros and no non-zero magnitudes tie. SciPy's exact method with Pratt's zeros puts all n
# ranks, the zeros' too, into its null instead of those of the non-zero differences, so it is
# no reference where there are zeros; the exact tables above serve Pratt's treatment as they
# stand, since SciPy's "pratt" and "wilcox" agree where there are none.
WILCOXON_PRATT_NORMAL = [
    (
        [float((index * 37 + 17) % 23 - 6) for index in range(100)],
        805.5,
        3.7011586865422335e-09,
        4234.5,
        1.8505793432711167e-09,
        0.9999999981876029,
    ),
    (
        [float((index * 11 + 3) % 9 - 4) for index in range(120)],
        3544.0,
        0.9160518709016874,
        3625.0,
        0.4580259354508437,
        0.5430193891062951,
    ),
    (
        [
            0.0 if index % 9 == 0 else value
            for index, value in enumerate(congruential(80, 29, 103, 55, 0.03))
        ],
        1327.0,
        0.19495476667412714,
        1327.0,
        0.9033468718822332,
        0.09747738333706357,
    ),
]


def _pratt_enumeration(
    differences: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    """Return Pratt's test by listing every sign assignment of the non-zero differences.

    The result is the two-sided statistic and p-value, ``W+``, the p-values of "greater" and
    "less", and the probability of ``W+`` itself. The ranks come from midranking every magnitude,
    the zeros' included, and the zeros' ranks are then set aside: a construction apart from the
    library's lift of the ranks of the non-zero differences by the number of zeros. Only the signs
    of the non-zero differences are random.
    """
    ranks = midranks([abs(value) for value in differences])
    pairs = list(zip(ranks, differences, strict=True))
    kept = [rank for rank, value in pairs if value != 0.0]
    positive = math.fsum(rank for rank, value in pairs if value > 0.0)
    negative = math.fsum(rank for rank, value in pairs if value < 0.0)
    total = math.fsum(kept)
    sums = [
        math.fsum(rank for rank, sign in zip(kept, signs, strict=True) if sign)
        for signs in itertools.product((False, True), repeat=len(kept))
    ]
    size = len(sums)
    statistic = min(positive, negative)
    return (
        statistic,
        sum(min(value, total - value) <= statistic + 1e-9 for value in sums) / size,
        positive,
        sum(value >= positive - 1e-9 for value in sums) / size,
        sum(value <= positive + 1e-9 for value in sums) / size,
        sum(abs(value - positive) <= 1e-9 for value in sums) / size,
    )


def test_pratt_and_wilcox_differ_on_a_hand_computed_case():
    # Differences 0, 1, -2, 3, 4, -5.
    # Wilcoxon drops the zero and ranks |1| to |5| from 1: the signs + - + + - give W+ = 8 and
    # W- = 7. Of the 32 sign assignments, 16 have a positive sum of at most 7, so the two-sided
    # p-value is 1; 16 have a sum of at least 8, and 19 a sum of at most 8.
    # Pratt ranks the zero first, so the others hold the ranks 2 to 6: W+ = 2 + 4 + 5 = 11 and
    # W- = 3 + 6 = 9. 15 assignments have a positive sum of at most 9, so the two-sided p-value is
    # 30 / 32; 15 have a sum of at least 11, and 20 a sum of at most 11.
    differences = [0.0, 1.0, -2.0, 3.0, 4.0, -5.0]
    expected: dict[Alternative, dict[ZeroMethod, tuple[float, float]]] = {
        "two-sided": {"wilcox": (7.0, 32 / 32), "pratt": (9.0, 30 / 32)},
        "greater": {"wilcox": (8.0, 16 / 32), "pratt": (11.0, 15 / 32)},
        "less": {"wilcox": (8.0, 19 / 32), "pratt": (11.0, 20 / 32)},
    }
    for alternative, by_method in expected.items():
        for zero_method, (statistic, p_value) in by_method.items():
            result = wilcoxon_signed_rank(
                differences, alternative=alternative, zero_method=zero_method
            )
            assert (result.statistic, result.p_value) == (statistic, p_value)
            assert (result.pairs, result.zeros, result.zero_method) == (5, 1, zero_method)
    # The default is Wilcoxon's treatment.
    assert wilcoxon_signed_rank(differences) == wilcoxon_signed_rank(
        differences, zero_method="wilcox"
    )
    assert wilcoxon_signed_rank(differences).statistic == 7.0


def test_pratt_lifts_the_ranks_of_the_non_zero_differences_by_the_zeros():
    differences = [0.0, 0.0, 0.0, 5.0, -6.0]
    wilcox = wilcoxon_signed_rank(differences, alternative="greater")
    pratt = wilcoxon_signed_rank(differences, alternative="greater", zero_method="pratt")
    # The zeros hold ranks 1 to 3, so |5| and |6| rank 4 and 5, where Wilcoxon ranks them 1 and 2.
    assert (wilcox.statistic, pratt.statistic) == (1.0, 4.0)
    assert (pratt.pairs, pratt.zeros) == (2, 3)
    # Only the two signs are random: the positive sums 0, 4, 5, and 9 are equally likely.
    assert pratt.p_value == 3 / 4
    assert (
        wilcoxon_signed_rank(differences, alternative="less", zero_method="pratt").p_value == 2 / 4
    )
    two_sided = wilcoxon_signed_rank(differences, zero_method="pratt")
    assert (two_sided.statistic, two_sided.p_value) == (4.0, 1.0)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("name", [name for name, values in _battery().items() if 0.0 not in values])
def test_pratt_without_zeros_equals_wilcox(name, alternative):
    differences = _battery()[name]
    wilcox = wilcoxon_signed_rank(differences, alternative=alternative)
    pratt = wilcoxon_signed_rank(differences, alternative=alternative, zero_method="pratt")
    assert pratt.zero_method == "pratt"
    assert replace(pratt, zero_method="wilcox") == wilcox


def test_pratt_exact_values_match_the_enumeration_with_zeros_ties_and_infinities():
    generator = random.Random(34)
    with_zeros = with_ties = 0
    for _ in range(250):
        size = generator.randint(1, 8)
        # Integer differences in a narrow range tie often, and give zeros.
        differences = [float(generator.randint(-3, 3)) for _ in range(size)]
        if generator.random() < 0.2:
            differences.append(generator.choice((math.inf, -math.inf)))
            differences.append(generator.choice((math.inf, -math.inf)))
        two = wilcoxon_signed_rank(differences, zero_method="pratt")
        up = wilcoxon_signed_rank(differences, alternative="greater", zero_method="pratt")
        down = wilcoxon_signed_rank(differences, alternative="less", zero_method="pratt")
        if two.method == "degenerate":
            continue
        statistic, p_two, positive, p_greater, p_less, point = _pratt_enumeration(differences)
        assert (two.statistic, up.statistic, down.statistic) == (statistic, positive, positive)
        assert two.p_value == pytest.approx(p_two, rel=1e-12, abs=0.0), differences
        assert up.p_value == pytest.approx(p_greater, rel=1e-12, abs=0.0), differences
        assert down.p_value == pytest.approx(p_less, rel=1e-12, abs=0.0), differences
        # The two tails share the observed value, so they sum to one plus its probability.
        assert up.p_value + down.p_value == pytest.approx(1.0 + point, rel=1e-12)
        zeros = differences.count(0.0)
        magnitudes = [abs(value) for value in differences if value != 0.0]
        tied = len(set(magnitudes)) < len(magnitudes)
        assert (two.zeros, two.pairs) == (zeros, len(magnitudes))
        # Zeros that Pratt ranks make the distribution conditional, as ties do.
        assert two.method == ("exact-conditional" if zeros or tied else "exact")
        with_zeros += zeros > 0
        with_ties += tied
    assert with_zeros > 80
    assert with_ties > 100


@pytest.mark.parametrize(
    ("differences", "statistic", "p_value", "positive", "greater", "less"), WILCOXON_PRATT_NORMAL
)
def test_pratt_normal_approximation_matches_scipy(
    differences, statistic, p_value, positive, greater, less
):
    two = wilcoxon_signed_rank(differences, zero_method="pratt")
    assert (two.method, two.zero_method) == ("normal", "pratt")
    assert two.pairs > WILCOXON_EXACT_LIMIT
    assert two.zeros > 0
    assert two.statistic == statistic
    assert two.p_value == pytest.approx(p_value, rel=1e-11, abs=0.0)
    for alternative, expected in zip(ONE_SIDED, (greater, less), strict=True):
        result = wilcoxon_signed_rank(differences, alternative=alternative, zero_method="pratt")
        assert (result.method, result.alternative, result.statistic) == (
            "normal",
            alternative,
            positive,
        )
        assert result.p_value == pytest.approx(expected, rel=1e-11, abs=0.0)
    # Wilcoxon's treatment of the same zeros gives other ranks, so another test.
    assert wilcoxon_signed_rank(differences).p_value != two.p_value


def test_pratt_zeros_make_the_exact_distribution_conditional_and_the_limit_counts_non_zero_pairs():
    untied = [0.0, 0.0, 1.0, -2.0, 3.0]
    assert wilcoxon_signed_rank(untied).method == "exact"
    assert wilcoxon_signed_rank(untied, zero_method="pratt").method == "exact-conditional"
    assert wilcoxon_signed_rank([1.0, -2.0, 3.0], zero_method="pratt").method == "exact"
    at_limit = [float(index) for index in range(1, WILCOXON_EXACT_LIMIT + 1)]
    for alternative in ("two-sided", "greater", "less"):
        exact = wilcoxon_signed_rank(
            [0.0] * 5 + at_limit, alternative=alternative, zero_method="pratt"
        )
        above = wilcoxon_signed_rank(
            [0.0] * 5 + [*at_limit, 51.0], alternative=alternative, zero_method="pratt"
        )
        assert (exact.method, exact.pairs, exact.zeros) == ("exact-conditional", 50, 5)
        assert (above.method, above.pairs, above.zeros) == ("normal", 51, 5)


def test_many_zeros_leave_the_exact_distribution_of_the_rest_in_reach():
    # 200 zeros lift the ranks of 40 positive differences to 201 to 240, and the exact null
    # still has its floor of 2 / 2^40 for two-sided and 1 / 2^40 for one-sided tests.
    differences = [0.0] * 200 + [float(index) for index in range(1, 41)]
    two = wilcoxon_signed_rank(differences, zero_method="pratt")
    assert (two.method, two.pairs, two.zeros) == ("exact-conditional", 40, 200)
    assert two.statistic == 0.0
    assert two.p_value == 2 / 2**40
    up = wilcoxon_signed_rank(differences, alternative="greater", zero_method="pratt")
    assert (up.statistic, up.p_value) == (math.fsum(range(201, 241)), 1 / 2**40)


@pytest.mark.parametrize("zero_method", ["Pratt", "wilcoxon", "zero", "", None, 1, ("pratt",)])
def test_an_unknown_zero_method_is_refused(zero_method):
    with pytest.raises(ConfigError, match="zero_method"):
        wilcoxon_signed_rank([0.0, 1.0, 2.0], zero_method=zero_method)
