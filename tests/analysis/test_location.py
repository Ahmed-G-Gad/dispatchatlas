"""The Hodges-Lehmann shift and its exact signed-rank interval.

The expected values are derived here and not read from the library. The Walsh averages come from
their definition. The exact null distribution of the signed-rank sum comes from a subset-sum
recurrence, and for a few sizes from a count by hand, written out in the comments. The inversion of
the tests comes from a scan of every Walsh average and every gap, or of a fine grid of shifts, with
``wilcoxon_signed_rank`` itself, so the bisection of the library is compared with a search that
assumes nothing about the order of the p-values. Neither the library nor these tests import SciPy.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, fields

import pytest

from dispatchatlas import references
from dispatchatlas.analysis import stats
from dispatchatlas.analysis.methods import method
from dispatchatlas.analysis.stats.equivalence import tost_signed_rank
from dispatchatlas.analysis.stats.location import (
    SignedRankInterval,
    hodges_lehmann_shift,
    signed_rank_interval,
)
from dispatchatlas.analysis.stats.nonparametric import (
    WILCOXON_EXACT_LIMIT,
    Alternative,
    ZeroMethod,
    wilcoxon_signed_rank,
)
from dispatchatlas.core.errors import ConfigError
from tests.analysis.paired_differences import congruential

# (size, confidence, k, coverage) for the differences 3^0, ..., 3^(size - 1). Every two sums of
# two of these differ, so their Walsh averages are distinct, none is a difference, and no two
# differences tie: the null distribution of the signed-rank sum W+ is the one of the ranks 1 to
# size between every two Walsh averages. With alpha = (1 - confidence) / 2 and r the largest
# integer with P(W+ <= r) < alpha, the interval is the (r + 1)-th smallest to the (r + 1)-th
# largest Walsh average, so k = r + 1, with the coverage 1 - 2 P(W+ <= r). A k of 0 means that no
# sum has a tail below alpha, so that no shift is rejected and the interval is unbounded.
#
# The number of subsets of {1, ..., size} with the sum s is 1, 1, 1, 2, 2, 3 for s = 0 to 5 ({},
# {1}, {2}, {3} and {1, 2}, {4} and {1, 3}, {5} and {1, 4} and {2, 3}) whatever the size from 5 on,
# and for s = 6 it is 3 for a size of 5 ({1, 5}, {2, 4}, {1, 2, 3}) and 4 from a size of 6 on
# (adding {6}). The tails follow from the cumulative counts:
#   size 5, 0.9: P(W+ <= 0) = 1/32 < 0.05 <= P(W+ <= 1) = 2/32, so r = 0, coverage 1 - 2/32.
#   size 6, 0.9: P(W+ <= 2) = 3/64 < 0.05 <= P(W+ <= 3) = 5/64, so r = 2, coverage 1 - 6/64.
#   size 7, 0.9: P(W+ <= 3) = 5/128 < 0.05 <= P(W+ <= 4) = 7/128, so r = 3, coverage 1 - 10/128.
#   size 8, 0.9: P(W+ <= 5) = 10/256 < 0.05 <= P(W+ <= 6) = 14/256, so r = 5, coverage 1 - 20/256.
#   size 7, 0.99: P(W+ <= 0) = 1/128 > 0.005, so no shift is rejected; size 8: 1/256 < 0.005.
# A p-value equal to alpha does not reject: at confidence 0.875 the level is exactly 1/16, which
# is P(W+ <= 0) for a size of 4, so the interval is unbounded, and P(W+ <= 1) for a size of 5,
# so r stays 0 as at 0.9.
EXACT = [
    (4, 0.875, 0, 1.0),
    (5, 0.8, 3, 0.8125),
    (5, 0.875, 1, 0.9375),
    (5, 0.9, 1, 0.9375),
    (5, 0.95, 0, 1.0),
    (6, 0.8, 4, 0.84375),
    (6, 0.9, 3, 0.90625),
    (6, 0.95, 1, 0.96875),
    (6, 0.99, 0, 1.0),
    (7, 0.8, 6, 0.84375),
    (7, 0.9, 4, 0.921875),
    (7, 0.95, 3, 0.953125),
    (7, 0.99, 0, 1.0),
    (8, 0.8, 9, 0.8046875),
    (8, 0.9, 6, 0.921875),
    (8, 0.95, 4, 0.9609375),
    (8, 0.99, 1, 0.9921875),
]


def walsh_averages(values: Sequence[float]) -> list[float]:
    """Every ``(x_i + x_j) / 2`` with ``i <= j``, ascending, straight from the definition."""
    return sorted((a + b) / 2 for a, b in itertools.combinations_with_replacement(values, 2))


def null_counts(size: int) -> list[int]:
    """How many of the 2^size sign assignments of the ranks 1..size give each positive-rank sum."""
    counts = [1]
    for rank in range(1, size + 1):
        grown = [0] * (len(counts) + rank)
        for total, count in enumerate(counts):
            grown[total] += count
            grown[total + rank] += count
        counts = grown
    return counts


def exact_k_and_coverage(size: int, confidence: float) -> tuple[int, float]:
    """The k and the coverage that the exact null of untied ranks gives, by the rule above."""
    alpha = (1.0 - confidence) / 2.0
    counts = null_counts(size)
    below, r = 0, -1
    for total, count in enumerate(counts):
        below += count
        if below / 2**size >= alpha:
            break
        r = total
    return r + 1, (1.0 - 2.0 * sum(counts[: r + 1]) / 2**size if r >= 0 else 1.0)


def p_value(
    values: Sequence[float],
    delta: float,
    alternative: Alternative,
    zero_method: ZeroMethod = "pratt",
) -> float:
    """The p-value of the one-sided signed-rank test of the differences shifted by ``delta``."""
    shifted = [value - delta for value in values]
    return wilcoxon_signed_rank(shifted, alternative=alternative, zero_method=zero_method).p_value


def shifts_of(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """The Walsh averages, and every shift at which a p-value can differ.

    Those are the shifts below all the Walsh averages, at each, in each gap between two, and above.
    """
    walsh = sorted(set(walsh_averages(values)))
    shifts = [walsh[0] - 1.0]
    for index, average in enumerate(walsh):
        following = walsh[index + 1] if index + 1 < len(walsh) else average + 2.0
        shifts += [average, (average + following) / 2.0]
    return walsh, shifts


def scan_bounds(
    values: Sequence[float], confidence: float, zero_method: ZeroMethod
) -> tuple[float, float]:
    """The bounds by brute force: both tests at every Walsh average and every gap, one by one."""
    alpha = (1.0 - confidence) / 2.0
    walsh, shifts = shifts_of(values)
    greater = [p_value(values, shift, "greater", zero_method) >= alpha for shift in shifts]
    less = [p_value(values, shift, "less", zero_method) >= alpha for shift in shifts]
    first = greater.index(True)
    last = len(shifts) - 1 - less[::-1].index(True)
    low = -math.inf if first == 0 else walsh[(first - 1) // 2]
    high = math.inf if last // 2 >= len(walsh) else walsh[last // 2]
    return low, high


def grid_bounds(
    values: Sequence[float], confidence: float, zero_method: ZeroMethod
) -> tuple[float, float]:
    """The bounds from a grid of quarter steps, for integer differences.

    Every Walsh average of integers is a multiple of one half, and the gap between two of them
    holds a multiple of a quarter, so the grid meets every shift at which a p-value can differ.
    The lower bound is the Walsh average at or below the first grid shift that the greater test
    retains, and the upper bound the one at or above the last that the less test retains.
    """
    alpha = (1.0 - confidence) / 2.0
    walsh = sorted(set(walsh_averages(values)))
    start = min(values) - 1.0
    grid = [start + 0.25 * step for step in range(round(4 * (max(values) + 1.0 - start)) + 1)]
    greater = [p_value(values, shift, "greater", zero_method) >= alpha for shift in grid]
    less = [p_value(values, shift, "less", zero_method) >= alpha for shift in grid]
    first = greater.index(True)
    last = len(grid) - 1 - less[::-1].index(True)
    low = -math.inf if first == 0 else max(w for w in walsh if w <= grid[first])
    high = math.inf if last == len(grid) - 1 else min(w for w in walsh if w >= grid[last])
    return low, high


def gaps_beside(values: Sequence[float], bound: float) -> tuple[float, float]:
    """The shifts in the gaps just below and just above a Walsh average."""
    walsh = sorted(set(walsh_averages(values)))
    index = walsh.index(bound)
    below = (walsh[index - 1] + bound) / 2.0 if index > 0 else bound - 1.0
    above = (bound + walsh[index + 1]) / 2.0 if index + 1 < len(walsh) else bound + 1.0
    return below, above


def inside(interval: SignedRankInterval, margin: float) -> bool:
    """Whether the interval lies inside ``(-margin, margin)``, as the two tests read it."""
    return -margin < interval.low and interval.high < margin


# ------------------------------------------------------------------------------ the shift


@pytest.mark.parametrize(
    ("differences", "shift"),
    [
        ([4.0], 4.0),
        # Walsh averages 0, 5, 10: the median is the one in the middle.
        ([0.0, 10.0], 5.0),
        # 1, 1.5, 2, 2.5, 3, 4: six averages, so the mean of the third and the fourth, which is
        # neither a difference nor their median, 2.
        ([1.0, 2.0, 4.0], 2.25),
        # 1, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 4: ten averages, the middle two are 2.5 and 2.5.
        ([1.0, 2.0, 3.0, 4.0], 2.5),
        # 1, 1.5, 2, 3.5, 4, 6: the mean of 2 and 3.5, where the median of the differences is 2.
        ([1.0, 2.0, 6.0], 2.75),
        # -3, -2, -1, -1, 0, 1, 1, 2, 3, 3, 4, 5, 5, 7, 9: fifteen averages, the eighth is 2, where
        # the median of the differences is 1.
        ([-3.0, -1.0, 1.0, 5.0, 9.0], 2.0),
    ],
)
def test_the_shift_is_the_median_of_the_walsh_averages(differences, shift):
    assert hodges_lehmann_shift(differences) == shift
    assert hodges_lehmann_shift(differences) == statistics.median(walsh_averages(differences))


def test_the_shift_of_many_differences_is_the_median_of_their_walsh_averages():
    generator = random.Random(81)
    for size in range(1, 31):
        decimals = generator.choice((1, 3))
        differences = [round(generator.gauss(0.2, 1.0), decimals) for _ in range(size)]
        expected = statistics.median(walsh_averages(differences))
        assert hodges_lehmann_shift(differences) == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("size", range(2, 10))
def test_the_signed_rank_sum_at_the_shift_is_half_its_range(size):
    # Without ties, and with no Walsh average that is a difference, the shift is the middle Walsh
    # average or the mean of the two middle ones, and the positive-rank sum is half of 1 + ... + n.
    differences = [3.0**index for index in range(size)]
    shift = hodges_lehmann_shift(differences)
    result = wilcoxon_signed_rank(
        [value - shift for value in differences], alternative="greater", zero_method="pratt"
    )
    assert result.statistic == size * (size + 1) / 4


def test_the_shift_does_not_overflow_where_a_sum_of_two_differences_would():
    big = 1.5e308
    assert hodges_lehmann_shift([big, big]) == big
    assert hodges_lehmann_shift([-big, big]) == 0.0
    assert hodges_lehmann_shift([big, big, big, big]) == big


def test_the_shift_follows_the_differences_when_shifted_negated_scaled_or_permuted():
    generator = random.Random(82)
    for size in (1, 2, 5, 9, 14):
        # Multiples of one sixteenth are exact, so shifting and scaling keep every tie.
        differences = [generator.randint(-16, 16) / 16 for _ in range(size)]
        base = hodges_lehmann_shift(differences)
        permuted = differences[:]
        generator.shuffle(permuted)
        assert hodges_lehmann_shift(permuted) == base
        assert hodges_lehmann_shift([value + 1.25 for value in differences]) == base + 1.25
        assert hodges_lehmann_shift([-value for value in differences]) == -base
        assert hodges_lehmann_shift([4 * value for value in differences]) == 4 * base


# ------------------------------------------------------------------- the exact, untied interval


@pytest.mark.parametrize(("size", "confidence", "k", "coverage"), EXACT)
def test_untied_differences_give_the_kth_smallest_and_largest_walsh_averages(
    size, confidence, k, coverage
):
    differences = [3.0**index for index in range(size)]
    result = signed_rank_interval(differences, confidence)
    walsh = walsh_averages(differences)
    assert len(set(walsh)) == size * (size + 1) // 2
    if k == 0:
        assert (result.low, result.high) == (-math.inf, math.inf)
    else:
        assert (result.low, result.high) == (walsh[k - 1], walsh[-k])
    assert result.coverage == coverage
    assert result.coverage >= confidence
    assert (result.confidence, result.zero_method, result.pairs) == (confidence, "pratt", size)
    assert result.shift == statistics.median(walsh)


def test_the_hand_computed_rows_are_the_exact_null_of_the_signed_rank_sum():
    for size, confidence, k, coverage in EXACT:
        assert exact_k_and_coverage(size, confidence) == (k, coverage)


@pytest.mark.parametrize("size", range(1, 15))
def test_the_interval_of_distinct_differences_is_set_by_the_exact_null_of_untied_ranks(size):
    # Distinct differences have the same null between every two Walsh averages, whatever else
    # they do, so the rule holds for any signs of 3^0, ..., 3^(size - 1) and any confidence.
    generator = random.Random(83 + size)
    for confidence in (0.5, 0.8, 0.9, 0.95, 0.99):
        k, coverage = exact_k_and_coverage(size, confidence)
        differences = [3.0**index * generator.choice((1, -1)) for index in range(size)]
        walsh = walsh_averages(differences)
        result = signed_rank_interval(differences, confidence)
        if k == 0:
            assert (result.low, result.high) == (-math.inf, math.inf)
        else:
            assert (result.low, result.high) == (walsh[k - 1], walsh[-k])
        assert result.coverage == pytest.approx(coverage, rel=1e-14, abs=0.0)
        assert result.coverage >= confidence


def test_a_sample_too_small_to_reject_gives_an_unbounded_interval_that_no_margin_contains():
    for size in range(1, 5):
        differences = [3.0**index for index in range(size)]
        result = signed_rank_interval(differences, 0.9)
        assert (result.low, result.high, result.coverage) == (-math.inf, math.inf, 1.0)
        # The most extreme sample has a one-sided p-value of 2^-size, at least 0.05.
        for margin in (0.1, 10.0, 1e6):
            assert tost_signed_rank(differences, margin).p_value >= 2.0**-size
    assert signed_rank_interval([3.0**index for index in range(5)], 0.9).low == 1.0


def test_a_p_value_equal_to_the_level_is_retained_as_the_equivalence_tests_read_it():
    # Four differences at confidence 0.875: the level is exactly 1/16, the smallest p-value that
    # four pairs can give. A test rejects only below the level, so no shift is rejected, and the
    # two one-sided tests, which find equivalence only below it, find none at any margin.
    differences = [3.0**index for index in range(4)]
    result = signed_rank_interval(differences, 0.875)
    assert (result.low, result.high) == (-math.inf, math.inf)
    for margin in (0.5, 1e6):
        assert tost_signed_rank(differences, margin).p_value >= 1 / 16
    assert tost_signed_rank(differences, 1e6).p_value == 1 / 16
    # With five differences the p-value 2/32 at the sum 1 equals the level too, and is retained,
    # so the interval is the one of the sum 0 alone: the whole range.
    five = signed_rank_interval([3.0**index for index in range(5)], 0.875)
    assert (five.low, five.high, five.coverage) == (1.0, 81.0, 0.9375)


def test_the_interval_of_five_differences_spans_the_whole_range_at_ninety_percent():
    result = signed_rank_interval([1.0, 2.0, 4.0, 8.0, 16.0], 0.9)
    assert (result.low, result.high, result.coverage) == (1.0, 16.0, 0.9375)


def test_the_interval_of_six_differences_has_hand_computed_endpoints():
    # The Walsh averages of 1, 2, 4, 8, 16, 32 in order: 1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 6, 8, 8.5,
    # 9, 10, 12, 16, 16.5, 17, 18, 20, 24, 32. At confidence 0.9, k = 3: the third smallest and
    # the third largest, with the coverage 1 - 2 * 3 / 64. The shift is the eleventh, 8.5.
    result = signed_rank_interval([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], 0.9)
    assert (result.low, result.high, result.coverage) == (2.0, 20.0, 0.90625)
    assert result.shift == 8.5


def test_the_interval_of_fifty_differences_reads_the_exact_null_and_the_library_test():
    generator = random.Random(91)
    differences = [generator.gauss(0.1, 0.6) for _ in range(50)]
    k, coverage = exact_k_and_coverage(50, 0.9)
    walsh = walsh_averages(differences)
    assert len(set(walsh)) == len(walsh)
    result = signed_rank_interval(differences, 0.9)
    assert (result.low, result.high) == (walsh[k - 1], walsh[-k])
    assert result.coverage == pytest.approx(coverage, rel=1e-12)
    assert (result.pairs, result.confidence, result.zero_method) == (50, 0.9, "pratt")
    alpha = (1.0 - 0.9) / 2.0
    below, above = gaps_beside(differences, result.low)
    assert p_value(differences, below, "greater") < alpha <= p_value(differences, above, "greater")
    below, above = gaps_beside(differences, result.high)
    assert p_value(differences, above, "less") < alpha <= p_value(differences, below, "less")
    # The two one-sided tests find equivalence for a margin just outside the interval, and not
    # for one just inside it.
    reach = max(-result.low, result.high)
    assert tost_signed_rank(differences, reach + 1e-6).p_value < alpha
    assert tost_signed_rank(differences, reach - 1e-6).p_value >= alpha


def test_equal_walsh_averages_leave_the_endpoints_and_raise_the_coverage():
    # Fifty distinct differences whose Walsh averages repeat, as rounded differences' do: the
    # endpoints are still the k-th smallest and largest, and the p-values just outside them are
    # lower than the exact null of the continuous case gives, so the coverage is higher.
    differences = congruential(50, 41, 83, 41, 0.006)
    k, coverage = exact_k_and_coverage(50, 0.9)
    walsh = walsh_averages(differences)
    assert len(set(walsh)) < len(walsh)
    result = signed_rank_interval(differences, 0.9)
    assert (result.low, result.high) == (walsh[k - 1], walsh[-k])
    assert result.coverage > coverage
    assert result.coverage < 1.0


# ------------------------------------------------------------------ ties, zeros and the search


def test_the_positive_rank_sum_counts_the_walsh_averages_above_the_shift():
    # Under Pratt's treatment, at every shift: the averages above it, and half of those equal to
    # it except an average of two differences that both equal it (two zeros).
    generator = random.Random(92)
    for _ in range(60):
        size = generator.randint(1, 9)
        differences = [float(generator.randint(-3, 3)) for _ in range(size)]
        for shift in shifts_of(differences)[1]:
            counted = 0.0
            for first, second in itertools.combinations_with_replacement(range(size), 2):
                average = (differences[first] + differences[second]) / 2
                if average > shift:
                    counted += 1.0
                elif average == shift and not (differences[first] == differences[second] == shift):
                    counted += 0.5
            shifted = [value - shift for value in differences]
            result = wilcoxon_signed_rank(shifted, alternative="greater", zero_method="pratt")
            assert result.statistic == counted, (differences, shift)


def test_ties_and_zeros_have_a_hand_computed_interval():
    # Differences 0, 0, 1. The Walsh averages are 0 (three times), 1/2 (twice) and 1, and their
    # median is 1/4. At confidence 0.6 the one-sided level is 0.2. The p-values of the two tests
    # under Pratt's treatment, at a shift in each region, by hand:
    #   below 0: all three positive, W+ = 6 of 6: P(W >= 6) = 1/8 for "greater", 1 for "less".
    #   at 0: two zeros take the ranks 1 and 2, the 1 ranks 3 and is positive: W+ = 3 of {0, 3}:
    #     1/2 and 1.
    #   in (0, 1/2): the magnitudes d, d, 1 - d rank 1.5, 1.5, 3 and W+ = 3; W takes 0, 1.5, 3,
    #     4.5, 6 with the weights 1, 2, 2, 2, 1 of 8: P(W >= 3) = P(W <= 3) = 5/8.
    #   at 1/2: the three magnitudes tie at the rank 2 and W+ = 2 of 2 Bin(3): P(W >= 2) = 7/8
    #     and P(W <= 2) = 4/8.
    #   in (1/2, 1): the positive one ranks 1 and the others tie at 2.5; W+ = 1; W takes 0, 1,
    #     2.5, 3.5, 5, 6 with the weights 1, 1, 2, 2, 1, 1 of 8: P(W >= 1) = 7/8, P(W <= 1) = 2/8.
    #   at 1: the zero takes the rank 1 and the others tie at 2.5: W+ = 0 of {0, 2.5, 2.5, 5}: 1
    #     and 1/4.
    #   above 1: all negative, W+ = 0: 1 and 1/8.
    differences = [0.0, 0.0, 1.0]
    table = [
        (-1.0, 1 / 8, 1.0),
        (0.0, 1 / 2, 1.0),
        (0.25, 5 / 8, 5 / 8),
        (0.5, 7 / 8, 4 / 8),
        (0.75, 7 / 8, 2 / 8),
        (1.0, 1.0, 1 / 4),
        (2.0, 1.0, 1 / 8),
    ]
    for shift, greater, less in table:
        assert p_value(differences, shift, "greater") == greater
        assert p_value(differences, shift, "less") == less
    # The greater test retains (p >= 0.2) from the shift 0 on, the less test up to the shift 1.
    result = signed_rank_interval(differences, 0.6)
    assert (result.low, result.high) == (0.0, 1.0)
    assert result.coverage == 1 - 1 / 8 - 1 / 8
    assert result.shift == 0.25
    assert result.pairs == 3
    # Wilcoxon's treatment drops each zero before ranking, and has the same p-values here.
    assert signed_rank_interval(differences, 0.6, zero_method="wilcox") == SignedRankInterval(
        0.25, 0.0, 1.0, 0.6, 0.75, "wilcox", 3
    )


def test_each_endpoint_separates_a_rejected_gap_from_a_retained_one():
    generator = random.Random(84)
    for _ in range(60):
        size = generator.randint(8, 16)
        decimals = generator.choice((1, 2))
        differences = [round(generator.gauss(0.1, 0.5), decimals) for _ in range(size)]
        confidence = generator.choice((0.8, 0.9, 0.95))
        alpha = (1.0 - confidence) / 2.0
        result = signed_rank_interval(differences, confidence)
        if math.isfinite(result.low):
            below, above = gaps_beside(differences, result.low)
            assert p_value(differences, below, "greater") < alpha
            assert p_value(differences, above, "greater") >= alpha
        if math.isfinite(result.high):
            below, above = gaps_beside(differences, result.high)
            assert p_value(differences, above, "less") < alpha
            assert p_value(differences, below, "less") >= alpha


@pytest.mark.parametrize("zero_method", ["pratt", "wilcox"])
def test_the_bisection_finds_what_a_grid_of_every_shift_finds_for_integer_differences(zero_method):
    generator = random.Random(85)
    for _ in range(200):
        size = generator.randint(1, 9)
        span = generator.choice((2, 3, 5))
        differences = [float(generator.randint(-span, span)) for _ in range(size)]
        confidence = generator.choice((0.5, 0.8, 0.9, 0.95, 0.99))
        found = signed_rank_interval(differences, confidence, zero_method=zero_method)
        assert (found.low, found.high) == grid_bounds(differences, confidence, zero_method), (
            differences,
            confidence,
        )


@pytest.mark.parametrize("zero_method", ["pratt", "wilcox"])
def test_the_bisection_finds_what_a_scan_of_every_walsh_average_and_gap_finds(zero_method):
    generator = random.Random(86)
    for index in range(60):
        size = generator.randint(1, 9)
        decimals = (1, 2, 6)[index % 3]
        differences = [round(generator.gauss(0.1, 0.5), decimals) for _ in range(size)]
        confidence = generator.choice((0.8, 0.9, 0.95))
        found = signed_rank_interval(differences, confidence, zero_method=zero_method)
        assert (found.low, found.high) == scan_bounds(differences, confidence, zero_method)


def test_the_p_values_are_monotone_in_the_shift_under_pratts_treatment():
    # The bisection relies on this order. Scan every Walsh average and gap of tied and untied data.
    generator = random.Random(87)
    for index in range(60):
        size = generator.randint(1, 9)
        differences = [round(generator.gauss(0.0, 1.0), (0, 1, 3)[index % 3]) for _ in range(size)]
        shifts = shifts_of(differences)[1]
        greater = [p_value(differences, shift, "greater") for shift in shifts]
        less = [p_value(differences, shift, "less") for shift in shifts]
        assert greater == sorted(greater)
        assert less == sorted(less, reverse=True)


def test_pratts_and_wilcoxons_treatments_of_a_zero_give_different_intervals():
    # At the shift 1 the difference 1 is a zero. Pratt's treatment ranks it below the other five
    # non-zero ones, and Wilcoxon's drops it before ranking, so the p-values of the less test
    # differ there, and the upper bound at confidence 0.6 is 1/2 for one and 1 for the other.
    differences = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0]
    pratt = signed_rank_interval(differences, 0.6)
    wilcox = signed_rank_interval(differences, 0.6, zero_method="wilcox")
    assert (pratt.low, pratt.high, pratt.coverage) == (0.0, 0.5, 0.8046875)
    assert (wilcox.low, wilcox.high, wilcox.coverage) == (0.0, 1.0, 0.890625)
    assert (pratt.zero_method, wilcox.zero_method) == ("pratt", "wilcox")
    assert pratt.shift == wilcox.shift == 0.0
    assert (
        p_value(differences, 1.0, "less", "pratt")
        < 0.2
        <= p_value(differences, 1.0, "less", "wilcox")
    )
    for found in (pratt, wilcox):
        assert (found.low, found.high) == grid_bounds(differences, 0.6, found.zero_method)


def test_tied_differences_can_leave_no_shift_that_both_tests_retain():
    # Six zeros and thirty ones: the greater test retains the shift 1 and above, the less test up
    # to the gap below 1, so no shift is retained by both. At the shift 1 the less test has the
    # p-value 2^-6 < 0.05: thirty differences are zero, and the six others are all negative.
    differences = [0.0] * 6 + [1.0] * 30
    result = signed_rank_interval(differences, 0.9)
    assert (result.low, result.high) == (1.0, 1.0)
    assert result.coverage == 0.0
    assert result.shift == 1.0
    assert p_value(differences, 1.0, "greater") == 1.0
    assert p_value(differences, 1.0, "less") == 2.0**-6
    # The statement about the margin holds all the same.
    assert tost_signed_rank(differences, 0.5).p_value >= 0.05
    assert tost_signed_rank(differences, 1.5).p_value < 0.05
    assert not inside(result, 0.5)
    assert inside(result, 1.5)


# ------------------------------------------------------- the equivalence tests read the interval


def test_two_signed_rank_tests_find_equivalence_exactly_when_the_interval_is_inside_the_margin():
    generator = random.Random(88)
    counts = {"inside": 0, "outside": 0, "boundary": 0, "ties": 0, "zeros": 0}
    for _ in range(400):
        size = generator.randint(1, 16)
        decimals = generator.choice((1, 2, 3))
        centre = generator.uniform(-0.3, 0.3)
        spread = generator.choice((0.1, 0.2, 0.3))
        differences = [round(generator.gauss(centre, spread), decimals) for _ in range(size)]
        confidence = generator.choice((0.8, 0.9, 0.95))
        alpha = (1.0 - confidence) / 2.0
        margin = generator.choice((0.1, 0.2, 0.25, 0.3, 0.5))
        interval = signed_rank_interval(differences, confidence)
        equivalent = tost_signed_rank(differences, margin).p_value < alpha
        counts["ties"] += len({abs(value) for value in differences}) < size
        counts["zeros"] += any(abs(abs(value) - margin) < 1e-12 for value in differences)
        if abs(interval.low + margin) < 1e-9 or abs(interval.high - margin) < 1e-9:
            counts["boundary"] += 1
            continue
        within = inside(interval, margin)
        counts["inside" if within else "outside"] += 1
        assert within == equivalent, (differences, confidence, margin)
    assert counts["inside"] > 80
    assert counts["outside"] > 80
    assert counts["ties"] > 100
    assert counts["zeros"] > 40


def test_a_margin_on_an_endpoint_is_decided_by_the_test_at_that_walsh_average():
    # Closed: differences 0, 0, 1 at confidence 0.6 have the interval [0, 1], and the margin 1 is
    # the upper endpoint. The less test at the shift 1 has the p-value 1/4 >= 0.2, so it retains
    # that shift, and the tests do not find equivalence, as for an interval that is not inside.
    closed = signed_rank_interval([0.0, 0.0, 1.0], 0.6)
    assert (closed.low, closed.high) == (0.0, 1.0)
    assert tost_signed_rank([0.0, 0.0, 1.0], 1.0).p_upper == 1 / 4
    assert tost_signed_rank([0.0, 0.0, 1.0], 1.0).p_value >= 0.2
    # Open: differences 0, 0, 1, 1, 1, 2 at confidence 0.8 have the interval [1/2, 3/2], and the
    # margin 3/2 is the upper endpoint. At the shift 3/2 the shifted differences are -3/2, -3/2,
    # and four of magnitude 1/2, one of them positive, which rank 2.5. The less test has the
    # p-value P(2.5 Bin(4) + 5.5 Bin(2) <= 2.5) = 5/64 < 0.1, so it rejects that shift although it
    # retains the shifts just below it, and the tests find equivalence.
    differences = [0.0, 0.0, 1.0, 1.0, 1.0, 2.0]
    open_end = signed_rank_interval(differences, 0.8)
    assert (open_end.low, open_end.high) == (0.5, 1.5)
    result = tost_signed_rank(differences, 1.5)
    assert result.p_upper == 5 / 64
    assert result.p_value < 0.1
    assert not inside(open_end, 1.5)
    assert p_value(differences, 1.25, "less") >= 0.1


def test_wilcoxons_treatment_can_break_the_statement_at_a_shift_that_equals_a_difference():
    # Differences 0, 0, 0, 2, 3, 3, 3, margin 2, confidence 0.6. At the shift 2, Wilcoxon's
    # treatment drops the difference 2 and ranks the six others: the less test has the p-value
    # 11/64 < 0.2 there, while the shifts around it are retained. The interval reaches 2.5, yet
    # the tests find equivalence within 2. Pratt's treatment ranks the difference, and has the
    # p-value 5/16 >= 0.2.
    differences = [0.0, 0.0, 0.0, 2.0, 3.0, 3.0, 3.0]
    wilcox = signed_rank_interval(differences, 0.6, zero_method="wilcox")
    assert (wilcox.low, wilcox.high) == (1.0, 2.5)
    broken = tost_signed_rank(differences, 2.0, zero_method="wilcox")
    assert broken.p_upper == 11 / 64
    assert broken.p_value < 0.2
    pratt = signed_rank_interval(differences, 0.6)
    kept = tost_signed_rank(differences, 2.0)
    assert kept.p_upper == 5 / 16
    assert inside(pratt, 2.0) == (kept.p_value < 0.2)


# ------------------------------------------------------- above the exact limit, and with ties


def test_above_the_exact_limit_the_interval_inverts_the_normal_approximation():
    differences = congruential(WILCOXON_EXACT_LIMIT + 4, 43, 101, 54, 0.007)
    alpha = (1.0 - 0.9) / 2.0
    result = signed_rank_interval(differences, 0.9)
    assert wilcoxon_signed_rank(differences).method == "normal"
    low_below, low_above = gaps_beside(differences, result.low)
    high_below, high_above = gaps_beside(differences, result.high)
    assert wilcoxon_signed_rank([value - low_below for value in differences]).method == "normal"
    assert p_value(differences, low_below, "greater") < alpha
    assert p_value(differences, low_above, "greater") >= alpha
    assert p_value(differences, high_above, "less") < alpha
    assert p_value(differences, high_below, "less") >= alpha
    attained = p_value(differences, low_below, "greater") + p_value(differences, high_above, "less")
    assert result.coverage == pytest.approx(1.0 - attained, rel=1e-12)
    assert result.coverage > 0.9
    assert result.low < result.shift < result.high
    reach = max(-result.low, result.high)
    assert tost_signed_rank(differences, reach + 1e-6).p_value < alpha
    assert tost_signed_rank(differences, reach - 1e-6).p_value >= alpha


def test_the_interval_with_ties_at_the_study_size_agrees_with_the_library_tests():
    generator = random.Random(89)
    differences = [round(generator.gauss(0.15, 0.6), 1) for _ in range(50)]
    assert len({abs(value) for value in differences}) < 40
    result = signed_rank_interval(differences, 0.9)
    alpha = (1.0 - 0.9) / 2.0
    low_below, low_above = gaps_beside(differences, result.low)
    high_below, high_above = gaps_beside(differences, result.high)
    assert p_value(differences, low_below, "greater") < alpha
    assert p_value(differences, low_above, "greater") >= alpha
    assert p_value(differences, high_above, "less") < alpha
    assert p_value(differences, high_below, "less") >= alpha
    assert result.low <= result.shift <= result.high
    assert 0.9 < result.coverage < 1.0
    reach = max(-result.low, result.high)
    assert tost_signed_rank(differences, reach + 1e-6).p_value < alpha
    assert tost_signed_rank(differences, reach - 1e-6).p_value >= alpha


# --------------------------------------------------------------------- symmetries of the interval


@pytest.mark.parametrize("zero_method", ["pratt", "wilcox"])
def test_the_interval_follows_the_differences_when_shifted_negated_scaled_or_permuted(zero_method):
    generator = random.Random(90)
    for size in (1, 2, 5, 9, 14):
        # Multiples of one sixteenth are exact, so shifting and scaling keep every tie.
        differences = [generator.randint(-16, 16) / 16 for _ in range(size)]
        base = signed_rank_interval(differences, 0.9, zero_method=zero_method)
        permuted = differences[:]
        generator.shuffle(permuted)
        assert signed_rank_interval(permuted, 0.9, zero_method=zero_method) == base
        moved = signed_rank_interval([v + 1.25 for v in differences], 0.9, zero_method=zero_method)
        assert (moved.shift, moved.low, moved.high) == (
            base.shift + 1.25,
            base.low + 1.25,
            base.high + 1.25,
        )
        assert (moved.coverage, moved.pairs) == (base.coverage, base.pairs)
        flipped = signed_rank_interval([-v for v in differences], 0.9, zero_method=zero_method)
        assert (flipped.shift, flipped.low, flipped.high) == (-base.shift, -base.high, -base.low)
        assert flipped.coverage == base.coverage
        scaled = signed_rank_interval([4 * v for v in differences], 0.9, zero_method=zero_method)
        assert (scaled.shift, scaled.low, scaled.high) == (
            4 * base.shift,
            4 * base.low,
            4 * base.high,
        )
        assert scaled.coverage == base.coverage


# ----------------------------------------------------------------------------------- the types


def test_the_result_is_a_frozen_slotted_record_with_the_documented_fields():
    names = [field.name for field in fields(SignedRankInterval)]
    assert names == ["shift", "low", "high", "confidence", "coverage", "zero_method", "pairs"]
    result = signed_rank_interval([0.1, 0.4, 0.2, 0.9, 0.3, 0.7], 0.9)
    assert type(result) is SignedRankInterval
    with pytest.raises(FrozenInstanceError):
        result.low = 0.0  # type: ignore[misc]
    assert not hasattr(result, "__dict__")


def test_differences_may_be_any_iterable_of_numbers_and_zeros_count_as_pairs():
    assert hodges_lehmann_shift(value for value in (1, 2, 4)) == 2.25
    result = signed_rank_interval((value for value in (0.0, 0.0, 0.5, 1, 2, 3, 4, 6)), 0.9)
    assert result.pairs == 8
    assert result.zero_method == "pratt"
    assert signed_rank_interval([1, 2, 4, 8, 16], confidence=0.9).low == 1.0


def test_the_confidence_defaults_to_ninety_percent_and_the_zero_method_to_pratt():
    differences = [0.3, -0.1, 0.8, 0.5, 0.2, 0.9, -0.2, 0.6]
    assert signed_rank_interval(differences) == signed_rank_interval(
        differences, 0.9, zero_method="pratt"
    )
    assert signed_rank_interval(differences).confidence == 0.9


def test_the_zero_method_is_keyword_only():
    with pytest.raises(TypeError):
        signed_rank_interval([0.1, 0.2, -0.1, 0.3], 0.9, "wilcox")  # type: ignore[call-arg]


def test_the_functions_are_exported_with_the_stats_package():
    assert stats.hodges_lehmann_shift is hodges_lehmann_shift
    assert stats.signed_rank_interval is signed_rank_interval
    assert stats.SignedRankInterval is SignedRankInterval
    for name in ("hodges_lehmann_shift", "signed_rank_interval", "SignedRankInterval"):
        assert name in stats.__all__


def test_the_methods_are_catalogued_with_their_sources():
    shift = method("hodges-lehmann-shift")
    interval = method("signed-rank-interval")
    assert shift.function.endswith(".location.hodges_lehmann_shift")
    assert interval.function.endswith(".location.signed_rank_interval")
    assert (shift.family, interval.family) == ("effect-size", "interval")
    for entry in (shift, interval):
        assert "hodges1963estimates" in entry.references
    assert "pratt1959remarks" in interval.references
    source = references.reference("hodges1963estimates")
    assert source.doi == "10.1214/aoms/1177704172"
    assert (source.year, source.authors) == (1963, "J. L. Hodges and E. L. Lehmann")
    assert source.title == "Estimates of location based on rank tests"
    assert source.venue == "The Annals of Mathematical Statistics 34(2):598-611"


# -------------------------------------------------------------------------------- the refusals


@pytest.mark.parametrize("function", [hodges_lehmann_shift, signed_rank_interval])
def test_no_difference_is_refused(function):
    with pytest.raises(ConfigError, match="at least one"):
        function([])


@pytest.mark.parametrize("function", [hodges_lehmann_shift, signed_rank_interval])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "x", None, True])
def test_non_finite_or_non_numeric_differences_are_refused(function, bad):
    with pytest.raises(ConfigError, match=r"NaN|finite|numbers"):
        function([0.1, bad, -0.2, 0.3])


@pytest.mark.parametrize(
    "confidence", [0.0, 1.0, -0.1, 1.5, 90, math.nan, math.inf, -math.inf, True, "0.9", None]
)
def test_a_confidence_outside_zero_to_one_is_refused(confidence):
    with pytest.raises(ConfigError, match="confidence"):
        signed_rank_interval([0.1, 0.2, -0.1, 0.3], confidence)


@pytest.mark.parametrize("zero_method", ["Pratt", "wilcoxon", "", None, 1])
def test_an_unknown_zero_method_is_refused(zero_method):
    with pytest.raises(ConfigError, match="zero_method"):
        signed_rank_interval([0.1, 0.2, -0.1, 0.3], 0.9, zero_method=zero_method)


@pytest.mark.parametrize("confidence", [1e-9, 0.01, 0.999999])
def test_extreme_confidence_levels_are_accepted(confidence):
    result = signed_rank_interval([0.3, -0.1, 0.8, 0.5, 0.2, 0.9, -0.2, 0.6], confidence)
    assert result.confidence == confidence
    assert result.coverage >= 0.0
