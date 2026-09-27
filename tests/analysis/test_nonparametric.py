"""Reference values for the rank tests.

The expected statistics and p-values were computed once with SciPy 1.18.1
and pasted here; neither the library nor these tests import SciPy:
``scipy.stats.wilcoxon`` (``zero_method="wilcox"``, exact, or asymptotic with
``correction=True``), ``mannwhitneyu`` (``use_continuity=True``),
``friedmanchisquare``, ``f.sf`` for the Iman-Davenport p-value of SciPy's
Friedman statistic, and ``studentized_range.ppf(1 - alpha, k, inf) / sqrt(2)``
for the Nemenyi table.
"""

from __future__ import annotations

import math

import pytest

from dispatchatlas.analysis.stats.nonparametric import (
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
from dispatchatlas.core.errors import ConfigError

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


@pytest.mark.parametrize(("differences", "statistic", "p_value"), WILCOXON_NORMAL)
def test_wilcoxon_tied_normal_approximation_matches_scipy(differences, statistic, p_value):
    result = wilcoxon_signed_rank(differences)
    assert result.method == "normal"
    assert result.statistic == statistic
    assert result.p_value == pytest.approx(p_value, rel=1e-12)


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
    assert result.method == "normal"
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
