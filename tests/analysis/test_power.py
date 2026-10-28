"""Power of the Wilcoxon signed-rank test: floors, an analytic case, and monotonicity."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from dispatchatlas.analysis.stats.nonparametric import (
    WILCOXON_EXACT_LIMIT,
    wilcoxon_signed_rank,
)
from dispatchatlas.analysis.stats.power import (
    _rejection_threshold,
    attainable_floor,
    detectable_effect_size,
    required_paired_count,
    wilcoxon_power,
)
from dispatchatlas.core.errors import ConfigError

REPLICATES = 4000


def test_attainable_floor_is_the_smallest_count_that_can_reject():
    assert attainable_floor(0.05) == 6
    assert attainable_floor(0.10) == 5
    assert attainable_floor(0.01) == 8
    assert 2.0 / 2**6 <= 0.05 < 2.0 / 2**5


@pytest.mark.parametrize("pairs", [1, 2, 5])
def test_counts_below_the_floor_are_refused(pairs):
    with pytest.raises(ConfigError, match="no effect is detectable"):
        detectable_effect_size(pairs, replicates=REPLICATES)
    assert wilcoxon_power(pairs, 5.0, replicates=REPLICATES) == 0.0


def test_six_pairs_match_the_analytic_power():
    # With six pairs the test rejects only when every difference has one sign.
    normal = NormalDist()
    for effect in (1.0, 1.8, 2.5):
        exact = normal.cdf(effect) ** 6 + normal.cdf(-effect) ** 6
        assert wilcoxon_power(6, effect, replicates=20000) == pytest.approx(exact, abs=0.01)


def test_detectable_effect_shrinks_as_problems_accumulate():
    effects = [
        detectable_effect_size(pairs, replicates=REPLICATES).effect for pairs in (6, 8, 12, 30)
    ]
    assert effects == sorted(effects, reverse=True)
    # The analytic value at six pairs solves Phi(d)^6 + Phi(-d)^6 = 0.8.
    assert effects[0] == pytest.approx(1.7907, abs=0.05)


def test_detectable_effect_and_required_count_are_consistent():
    assessment = detectable_effect_size(12, replicates=REPLICATES, seed=4)
    assert (assessment.pairs, assessment.alpha, assessment.power) == (12, 0.05, 0.8)
    assert wilcoxon_power(12, assessment.effect, replicates=REPLICATES, seed=4) >= 0.8
    needed = required_paired_count(assessment.effect, replicates=REPLICATES, seed=4)
    assert needed <= 12
    assert required_paired_count(10.0, replicates=REPLICATES) == 6


def test_the_simulation_is_seeded():
    first = wilcoxon_power(10, 0.9, replicates=REPLICATES, seed=1)
    assert wilcoxon_power(10, 0.9, replicates=REPLICATES, seed=1) == first
    assert wilcoxon_power(40, 0.0, replicates=REPLICATES) == pytest.approx(0.05, abs=0.015)


def _attained_size(pairs: int, alpha: float = 0.05) -> float:
    """Return the rejection probability under the null of the exact two-sided test.

    The count of sign assignments by rank sum is the coefficient list of the product of
    ``1 + x^r`` over the ranks, formed by convolution, and the test rejects when the smaller
    rank sum has a two-sided tail of at most ``alpha``.
    """
    counts = np.ones(1, dtype=np.int64)
    for rank in range(1, pairs + 1):
        factor = np.zeros(rank + 1, dtype=np.int64)
        factor[0] = factor[rank] = 1
        counts = np.convolve(counts, factor)
    cumulative = [int(total) for total in np.cumsum(counts)]
    rejecting = [
        value for value, tail in enumerate(cumulative) if 2.0 * tail / (1 << pairs) <= alpha
    ]
    return 2.0 * cumulative[max(rejecting)] / (1 << pairs)


def _greedy_negatives(pairs: int, target: int) -> list[float]:
    """Return the differences ``1..pairs`` whose negative magnitudes sum to ``target``."""
    differences: list[float] = []
    remaining = target
    for magnitude in range(pairs, 0, -1):
        if magnitude <= remaining:
            remaining -= magnitude
            differences.append(-float(magnitude))
        else:
            differences.append(float(magnitude))
    assert remaining == 0
    return differences


@pytest.mark.parametrize("pairs", [25, 26, 32, 50])
def test_the_rejection_threshold_is_the_largest_statistic_the_exact_test_rejects(pairs):
    threshold = _rejection_threshold(pairs, 0.05)
    assert threshold is not None
    assert threshold == int(threshold)
    rejected = wilcoxon_signed_rank(_greedy_negatives(pairs, int(threshold)))
    assert (rejected.method, rejected.statistic) == ("exact", threshold)
    assert rejected.p_value <= 0.05
    kept = wilcoxon_signed_rank(_greedy_negatives(pairs, int(threshold) + 1))
    assert (kept.method, kept.statistic) == ("exact", threshold + 1)
    assert kept.p_value > 0.05


def test_the_threshold_reads_the_exact_null_up_to_the_limit_and_the_normal_one_above():
    # 32 untied pairs reject at a smaller rank sum of 159 (p = 0.04977) and not at 160. The
    # continuity-corrected normal reading puts the threshold at 158.68, so it rejects only to 158.
    assert _rejection_threshold(32, 0.05) == 159.0
    assert _rejection_threshold(WILCOXON_EXACT_LIMIT, 0.05) == 434.0
    above = _rejection_threshold(WILCOXON_EXACT_LIMIT + 1, 0.05)
    mean = 51 * 52 / 4.0
    deviation = math.sqrt(51 * 52 * 103 / 24.0)
    assert above is not None
    assert above == pytest.approx(mean - 0.5 - NormalDist().inv_cdf(0.975) * deviation, abs=1e-4)
    assert above != int(above)


@pytest.mark.parametrize("pairs", [26, 32, 50])
def test_the_simulated_size_at_26_to_50_pairs_is_the_attained_size_of_the_exact_test(pairs):
    attained = _attained_size(pairs)
    assert 0.04 < attained <= 0.05
    size = wilcoxon_power(pairs, 0.0, replicates=40000, seed=7)
    assert size == pytest.approx(attained, abs=0.006)


def test_power_is_continuous_across_the_exact_limit():
    at_limit = wilcoxon_power(WILCOXON_EXACT_LIMIT, 0.4, replicates=REPLICATES, seed=3)
    above = wilcoxon_power(WILCOXON_EXACT_LIMIT + 1, 0.4, replicates=REPLICATES, seed=3)
    assert 0.5 < at_limit < 1.0
    assert above == pytest.approx(at_limit, abs=0.05)


def test_the_detectable_effect_at_32_pairs_is_searched_on_the_exact_test():
    assessment = detectable_effect_size(32, replicates=REPLICATES, seed=5)
    assert wilcoxon_power(32, assessment.effect, replicates=REPLICATES, seed=5) >= 0.8
    # The paired t-test needs a d_z of about 0.51 for 80% power at 32 pairs; the signed-rank
    # test on normal differences is nearly as efficient.
    assert assessment.effect == pytest.approx(0.51, abs=0.06)
    assert required_paired_count(assessment.effect, replicates=REPLICATES, seed=5) <= 32


@pytest.mark.parametrize(
    ("call", "arguments"),
    [
        (detectable_effect_size, {"pairs": 10, "alpha": 1.0}),
        (detectable_effect_size, {"pairs": 10, "power": 0.0}),
        (detectable_effect_size, {"pairs": 0}),
        (wilcoxon_power, {"pairs": 10, "effect": math.nan}),
        (wilcoxon_power, {"pairs": 10, "effect": 0.5, "replicates": 0}),
        (required_paired_count, {"effect": 0.0}),
        (required_paired_count, {"effect": 0.5, "alpha": 0.0}),
        (attainable_floor, {"alpha": 1.5}),
    ],
)
def test_invalid_arguments_are_refused(call, arguments):
    with pytest.raises(ConfigError):
        call(**arguments)
