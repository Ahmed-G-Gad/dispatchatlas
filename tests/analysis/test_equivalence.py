"""Equivalence tests: SciPy reference values, the interval rule, the four outcomes, and refusals.

The reference values were computed once with SciPy 1.18.1 and pasted here; neither the library nor
these tests import SciPy: ``scipy.stats.ttest_1samp`` against the bounds with the alternatives
``"greater"`` and ``"less"``, ``scipy.stats.t.interval`` for the interval, and
``scipy.stats.wilcoxon`` with ``method="exact"`` on the differences shifted by each bound. The
t-tail p-values are compared to a relative 1e-10 with ``abs=0.0``, since ``pytest.approx`` keeps an
absolute floor of 1e-12 when only a relative tolerance is given.
"""

from __future__ import annotations

import math
import random
from dataclasses import FrozenInstanceError, fields

import pytest

from dispatchatlas.analysis.stats.equivalence import (
    TostPairedTResult,
    TostSignedRankResult,
    equivalence_report,
    tost_paired_t,
    tost_signed_rank,
)
from dispatchatlas.analysis.stats.nonparametric import wilcoxon_signed_rank
from dispatchatlas.core.errors import ConfigError
from tests.analysis.paired_differences import congruential

# (differences, margin, alpha, mean, p_lower, p_upper, interval low, interval high):
# scipy.stats.ttest_1samp against -margin ("greater") and +margin ("less"), and
# scipy.stats.t.interval(1 - 2 alpha, n - 1, loc=mean, scale=se). The last two differences are
# congruential(...), see paired_differences.py.
TOST_T = [
    (
        [-0.108, 0.156, 0.293, -0.339, 0.357, 0.152, 0.363, 0.159, 0.515, -0.325, 0.76, 0.531],
        0.5,
        0.05,
        0.20950000000000002,
        7.748674442786576e-06,
        0.00617636410211083,
        0.034870995123450654,
        0.38412900487654933,
    ),
    (
        [
            0.695,
            0.601,
            0.135,
            0.542,
            -0.241,
            -0.948,
            1.157,
            0.436,
            -1.09,
            -0.485,
            -0.222,
            0.355,
            0.946,
            0.335,
            -0.11,
            0.43,
            0.017,
            0.26,
            -0.25,
            -0.574,
        ],
        0.4,
        0.05,
        0.09945,
        0.0006634118230282659,
        0.01779412642023273,
        -0.13026379377757602,
        0.3291637937775758,
    ),
    (
        [-0.424, 1.294, 1.635, 1.262, 1.903, 0.603, -0.681, 0.548],
        0.5,
        0.1,
        0.7675000000000001,
        0.003259768894758513,
        0.776827002634747,
        0.2983480051798171,
        1.236651994820183,
    ),
    (
        congruential(49, 31, 89, 46, 0.012),
        0.25,
        0.05,
        -0.003151836734693879,
        6.002822378294253e-07,
        3.660884206501529e-07,
        -0.07771594092291405,
        0.07141226745352626,
    ),
    (
        congruential(50, 41, 83, 41, 0.006),
        0.15,
        0.01,
        -0.021059999999999995,
        9.441820653335118e-08,
        8.292378587226177e-11,
        -0.07222539399535122,
        0.03010539399535122,
    ),
]
# (differences, margin, p_lower, p_upper): scipy.stats.wilcoxon(d + margin, alternative="greater")
# and scipy.stats.wilcoxon(d - margin, alternative="less"), both with method="exact".
TOST_RANK = [
    (
        [-0.237, 0.524, 0.408, -0.168, -0.169, 0.822, 0.411, -0.271, -0.01, -0.122],
        0.5,
        0.0009765625,
        0.009765625,
    ),
    (
        [
            0.352,
            -0.32,
            0.575,
            0.67,
            -0.776,
            -0.451,
            0.264,
            0.042,
            0.192,
            -0.227,
            0.64,
            0.589,
            0.233,
            0.764,
            0.434,
            -0.23,
        ],
        0.4,
        0.000213623046875,
        0.07952880859375,
    ),
    (
        [
            0.022,
            0.239,
            -0.393,
            -0.554,
            -1.096,
            0.386,
            -0.092,
            0.003,
            -0.492,
            0.513,
            0.372,
            -0.161,
            -0.378,
            -0.278,
            -0.499,
            -0.087,
            -0.408,
            0.193,
            -0.28,
            -0.277,
            0.669,
            -0.463,
            -0.884,
            0.083,
            -1.139,
            0.032,
            0.391,
            -0.065,
            -0.095,
            -1.125,
        ],
        0.3,
        0.07594345975667238,
        1.1743977665901184e-06,
    ),
    (congruential(50, 37, 101, 50, 0.004), 0.15, 2.4802382370125997e-11, 7.724487716131989e-12),
]

SAMPLES = {
    # Differences of two solvers, over thirty problems, as the four outcomes need them.
    "equivalent": [((index * 7) % 11 - 5) * 0.01 + 0.002 for index in range(30)],
    "different": [1.0 + ((index * 7) % 11 - 5) * 0.05 for index in range(30)],
    "equivalent-and-different": [0.05 + ((index * 7) % 11 - 5) * 0.004 for index in range(30)],
    "inconclusive": [0.9, -1.1, 0.4, -0.2, 1.3, -0.8],
}
MARGINS = {
    "equivalent": 0.5,
    "different": 0.1,
    "equivalent-and-different": 0.5,
    "inconclusive": 0.3,
}


@pytest.mark.parametrize(
    ("differences", "margin", "alpha", "mean", "p_lower", "p_upper", "low", "high"), TOST_T
)
def test_tost_paired_t_matches_scipy(differences, margin, alpha, mean, p_lower, p_upper, low, high):
    result = tost_paired_t(differences, margin, alpha)
    assert isinstance(result, TostPairedTResult)
    assert (result.pairs, result.margin, result.alpha) == (len(differences), margin, alpha)
    assert result.mean_difference == pytest.approx(mean, rel=1e-12, abs=0.0)
    assert result.p_lower == pytest.approx(p_lower, rel=1e-10, abs=0.0)
    assert result.p_upper == pytest.approx(p_upper, rel=1e-10, abs=0.0)
    assert result.p_value == max(result.p_lower, result.p_upper)
    assert result.interval == pytest.approx((low, high), rel=1e-10)


@pytest.mark.parametrize("zero_method", [None, "pratt", "wilcox"])
@pytest.mark.parametrize(("differences", "margin", "p_lower", "p_upper"), TOST_RANK)
def test_tost_signed_rank_matches_scipy(differences, margin, p_lower, p_upper, zero_method):
    # These differences hit no bound, so the shifted ones have no zeros and the two treatments of
    # zeros agree; None leaves the default.
    options = {} if zero_method is None else {"zero_method": zero_method}
    result = tost_signed_rank(differences, margin, **options)
    assert type(result) is TostSignedRankResult
    assert (result.pairs, result.margin) == (len(differences), margin)
    assert result.p_lower == pytest.approx(p_lower, rel=1e-12, abs=0.0)
    assert result.p_upper == pytest.approx(p_upper, rel=1e-12, abs=0.0)
    assert result.p_value == max(result.p_lower, result.p_upper)


def test_tost_paired_t_has_a_hand_computed_value():
    # Two differences, 0.1 and 0.3, have mean 0.2 and standard error 0.1 on one degree of freedom,
    # the Cauchy distribution, whose upper tail is atan(1 / t) / pi. The statistics are
    # (0.2 + 0.5) / 0.1 = 7 and (0.2 - 0.5) / 0.1 = -3, and the 50% interval of alpha = 0.25 adds
    # the quantile tan(pi / 4) = 1 standard errors to each side of the mean.
    result = tost_paired_t([0.1, 0.3], 0.5, alpha=0.25)
    assert result.mean_difference == pytest.approx(0.2, rel=1e-14)
    assert result.p_lower == pytest.approx(math.atan(1.0 / 7.0) / math.pi, rel=1e-12)
    assert result.p_upper == pytest.approx(math.atan(1.0 / 3.0) / math.pi, rel=1e-12)
    assert result.p_value == result.p_upper
    assert result.interval == pytest.approx((0.1, 0.3), rel=1e-12)


def test_the_tost_p_value_is_the_larger_one_sided_p_value_and_never_below_either():
    generator = random.Random(61)
    for _ in range(200):
        size = generator.randint(3, 70)
        centre = generator.uniform(-0.5, 0.5)
        differences = [round(generator.gauss(centre, 0.6), 2) for _ in range(size)]
        margin = generator.choice((0.1, 0.25, 0.5, 1.0))
        for result in (tost_paired_t(differences, margin), tost_signed_rank(differences, margin)):
            assert 0.0 <= result.p_lower <= 1.0
            assert 0.0 <= result.p_upper <= 1.0
            assert result.p_value >= result.p_lower
            assert result.p_value >= result.p_upper
            assert result.p_value == max(result.p_lower, result.p_upper)


def test_the_paired_t_procedure_rejects_exactly_when_the_interval_lies_inside_the_margin():
    generator = random.Random(62)
    inside = outside = 0
    for _ in range(300):
        size = generator.randint(3, 40)
        differences = [
            round(generator.gauss(generator.uniform(-0.4, 0.4), 0.5), 3) for _ in range(size)
        ]
        margin = generator.choice((0.1, 0.25, 0.5, 1.0))
        alpha = generator.choice((0.01, 0.05, 0.1))
        result = tost_paired_t(differences, margin, alpha)
        low, high = result.interval
        within = -margin < low and high < margin
        assert (result.p_value < alpha) == within
        inside += within
        outside += not within
    assert inside > 30
    assert outside > 30


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.25])
def test_the_p_value_equals_alpha_when_the_margin_sits_on_the_interval(alpha):
    differences = [0.4, -0.1, 0.9, 0.3, 0.2, -0.3, 0.6, 0.1, 0.5, 0.0, 0.25]
    low, high = tost_paired_t(differences, 10.0, alpha).interval
    margin = max(-low, high)
    found = tost_paired_t(differences, margin, alpha).p_value
    assert found == pytest.approx(alpha, rel=1e-9, abs=0.0)


def test_alpha_sets_the_interval_and_leaves_the_p_values():
    differences = [0.4, -0.1, 0.9, 0.3, 0.2, -0.3, 0.6, 0.1, 0.5, 0.0, 0.25]
    narrow = tost_paired_t(differences, 1.0, 0.10)
    wide = tost_paired_t(differences, 1.0, 0.01)
    assert (narrow.p_lower, narrow.p_upper) == (wide.p_lower, wide.p_upper)
    assert wide.interval[0] < narrow.interval[0] < narrow.interval[1] < wide.interval[1]
    # Each interval is centred on the mean difference.
    for result in (narrow, wide):
        low, high = result.interval
        assert result.mean_difference - low == pytest.approx(
            high - result.mean_difference, rel=1e-12
        )


def test_differences_without_spread_read_each_statistic_as_a_limit():
    # Exactly representable values keep the mean and the spread exact.
    inside = tost_paired_t([0.25] * 5, 0.5)
    assert (inside.p_lower, inside.p_upper, inside.p_value) == (0.0, 0.0, 0.0)
    assert inside.interval == (0.25, 0.25)
    on_the_bound = tost_paired_t([0.5] * 5, 0.5)
    assert (on_the_bound.p_lower, on_the_bound.p_upper, on_the_bound.p_value) == (0.0, 0.5, 0.5)
    above = tost_paired_t([0.75] * 5, 0.5)
    assert (above.p_lower, above.p_upper, above.p_value) == (0.0, 1.0, 1.0)
    below = tost_paired_t([-0.75] * 5, 0.5)
    assert (below.p_lower, below.p_upper, below.p_value) == (1.0, 0.0, 1.0)
    # Two identical solvers: every difference is zero, which is inside any margin.
    identical = tost_paired_t([0.0] * 8, 0.125)
    assert (identical.p_value, identical.mean_difference, identical.interval) == (
        0.0,
        0.0,
        (0.0, 0.0),
    )


def test_tost_signed_rank_is_the_pair_of_one_sided_wilcoxon_tests_of_the_shifted_differences():
    generator = random.Random(63)
    differing = 0
    for _ in range(70):
        size = generator.randint(1, 58)
        # Two decimals make ties and, once shifted by a bound, zeros.
        differences = [round(generator.gauss(0.0, 0.5), 2) for _ in range(size)]
        margin = generator.choice((0.1, 0.25, 0.5))
        found = {"default": tost_signed_rank(differences, margin)}
        for zero_method in ("pratt", "wilcox"):
            found[zero_method] = tost_signed_rank(differences, margin, zero_method=zero_method)
            lower = wilcoxon_signed_rank(
                [value + margin for value in differences],
                alternative="greater",
                zero_method=zero_method,
            )
            upper = wilcoxon_signed_rank(
                [value - margin for value in differences],
                alternative="less",
                zero_method=zero_method,
            )
            result = found[zero_method]
            assert (result.p_lower, result.p_upper) == (lower.p_value, upper.p_value)
            assert result.p_value == max(lower.p_value, upper.p_value)
            assert result.pairs == size
        # Pratt's treatment is the default.
        assert found["default"] == found["pratt"]
        differing += found["pratt"] != found["wilcox"]
    assert differing > 8


def test_a_difference_on_a_bound_is_a_zero_of_that_one_sided_test():
    # With margin 0.5, the first difference shifted by +0.5 and the second shifted by -0.5 are
    # zeros of the lower and the upper test. Wilcoxon's treatment drops each before ranking;
    # Pratt's ranks it first, and drops its rank from the sums.
    differences = [-0.5, 0.5, 0.1, 0.2, -0.1, 0.3, 0.05, 0.4]
    without_lower = [value + 0.5 for value in differences[1:]]
    without_upper = [value - 0.5 for value in [differences[0], *differences[2:]]]
    wilcox = tost_signed_rank(differences, 0.5, zero_method="wilcox")
    assert wilcox.p_lower == wilcoxon_signed_rank(without_lower, alternative="greater").p_value
    assert wilcox.p_upper == wilcoxon_signed_rank(without_upper, alternative="less").p_value
    pratt = tost_signed_rank(differences, 0.5)
    shifted_up = [value + 0.5 for value in differences]
    shifted_down = [value - 0.5 for value in differences]
    lower = wilcoxon_signed_rank(shifted_up, alternative="greater", zero_method="pratt")
    upper = wilcoxon_signed_rank(shifted_down, alternative="less", zero_method="pratt")
    assert (pratt.p_lower, pratt.p_upper) == (lower.p_value, upper.p_value)
    # Both treatments count all eight pairs, the zeros included.
    assert pratt.pairs == wilcox.pairs == 8


def test_the_two_treatments_of_a_zero_at_a_bound_have_a_hand_computed_lower_test():
    # Margin 0.5. The lower test reads the differences shifted by +0.5: 0, -1.7, 0.75, 1.5, 1.0,
    # 0.8, whose first is a zero. Wilcoxon drops it and ranks the magnitudes 0.75, 0.8, 1.0, 1.5,
    # 1.7 from 1, so W+ = 1 + 2 + 3 + 4 = 10 of 15, and the one-sided p-value is the share of the
    # 32 sign assignments with W- of at most 5: 10 of 32. Pratt ranks the zero 1 and these 2 to
    # 6, so W+ = 14 of 20, and the share with W- of at most 6 is 8 of 32.
    differences = [-0.5, -2.2, 0.25, 1.0, 0.5, 0.3]
    wilcox = tost_signed_rank(differences, 0.5, zero_method="wilcox")
    pratt = tost_signed_rank(differences, 0.5)
    assert wilcox.p_lower == 10 / 32
    assert pratt.p_lower == 8 / 32


@pytest.mark.parametrize("function", [tost_paired_t, tost_signed_rank])
def test_negating_the_differences_swaps_the_two_one_sided_tests(function):
    generator = random.Random(64)
    for size in (2, 5, 12, 30, 60):
        differences = [round(generator.gauss(0.1, 0.5), 3) for _ in range(size)]
        forward = function(differences, 0.4)
        backward = function([-value for value in differences], 0.4)
        assert backward.p_lower == pytest.approx(forward.p_upper, rel=1e-12, abs=0.0)
        assert backward.p_upper == pytest.approx(forward.p_lower, rel=1e-12, abs=0.0)
        assert backward.p_value == pytest.approx(forward.p_value, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("tost_p", "difference_p", "alpha", "outcome"),
    [
        (0.001, 0.40, 0.05, "equivalent"),
        (0.40, 0.001, 0.05, "different"),
        (0.001, 0.001, 0.05, "equivalent-and-different"),
        (0.40, 0.40, 0.05, "inconclusive"),
        # A test rejects only when its p-value is below alpha.
        (0.05, 0.05, 0.05, "inconclusive"),
        (0.0499, 0.05, 0.05, "equivalent"),
        (0.05, 0.0499, 0.05, "different"),
        (0.0, 1.0, 0.05, "equivalent"),
        (1.0, 0.0, 0.05, "different"),
        # The level changes the reading.
        (0.008, 0.02, 0.01, "equivalent"),
        (0.008, 0.02, 0.05, "equivalent-and-different"),
    ],
)
def test_the_report_names_the_four_outcomes(tost_p, difference_p, alpha, outcome):
    tost = TostSignedRankResult(tost_p, 0.0, tost_p, 0.5, 10)
    assert equivalence_report(tost, difference_p, alpha) == outcome


def test_the_report_defaults_to_the_five_percent_level():
    tost = TostSignedRankResult(0.04, 0.01, 0.04, 0.5, 10)
    assert equivalence_report(tost, 0.06) == "equivalent"
    assert equivalence_report(tost, 0.04) == "equivalent-and-different"


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_each_outcome_arises_from_data_with_both_procedures(name):
    differences, margin = SAMPLES[name], MARGINS[name]
    difference_p = wilcoxon_signed_rank(differences).p_value
    for result in (tost_paired_t(differences, margin), tost_signed_rank(differences, margin)):
        assert equivalence_report(result, difference_p) == name


def test_the_samples_of_the_outcomes_are_far_from_the_level():
    # The outcomes above must not hinge on a p-value that sits near 0.05.
    for name, differences in SAMPLES.items():
        margin = MARGINS[name]
        difference_p = wilcoxon_signed_rank(differences).p_value
        assert not 0.02 < difference_p < 0.12, name
        for result in (tost_paired_t(differences, margin), tost_signed_rank(differences, margin)):
            assert not 0.02 < result.p_value < 0.12, name


def test_a_non_significant_difference_alone_is_inconclusive_not_equivalent():
    # Six noisy differences: the difference test does not reject, and neither does the
    # equivalence test, so nothing may be claimed.
    differences = SAMPLES["inconclusive"]
    assert wilcoxon_signed_rank(differences).p_value > 0.05
    assert tost_paired_t(differences, 0.3).p_value > 0.05
    assert tost_signed_rank(differences, 0.3).p_value > 0.05
    assert equivalence_report(tost_paired_t(differences, 0.3), 0.9) == "inconclusive"


@pytest.mark.parametrize("function", [tost_paired_t, tost_signed_rank])
@pytest.mark.parametrize(
    "margin", [0.0, -0.1, -1e-9, math.inf, -math.inf, math.nan, True, "0.5", None]
)
def test_a_margin_that_is_not_positive_and_finite_is_refused(function, margin):
    with pytest.raises(ConfigError, match="margin"):
        function([0.1, 0.2, -0.1, 0.3], margin)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.75, 1.0, -0.05, math.nan, True, "0.05", None])
def test_an_alpha_outside_zero_to_one_half_is_refused_for_the_interval(alpha):
    with pytest.raises(ConfigError, match="alpha"):
        tost_paired_t([0.1, 0.2, -0.1, 0.3], 0.5, alpha)


@pytest.mark.parametrize("function", [tost_paired_t, tost_signed_rank])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "x", None])
def test_non_finite_differences_are_refused(function, bad):
    with pytest.raises(ConfigError, match=r"NaN|finite|numbers"):
        function([0.1, bad, -0.2, 0.3], 0.5)


@pytest.mark.parametrize("zero_method", ["Pratt", "wilcoxon", "", None, 1])
def test_an_unknown_zero_method_is_refused(zero_method):
    with pytest.raises(ConfigError, match="zero_method"):
        tost_signed_rank([0.1, 0.2, -0.1, 0.3], 0.5, zero_method=zero_method)


def test_the_zero_method_is_keyword_only():
    with pytest.raises(TypeError):
        tost_signed_rank([0.1, 0.2, -0.1, 0.3], 0.5, "wilcox")  # type: ignore[call-arg]


def test_too_few_differences_are_refused():
    for differences in ([], [0.1]):
        with pytest.raises(ConfigError, match="at least two"):
            tost_paired_t(differences, 0.5)
    with pytest.raises(ConfigError, match="at least one"):
        tost_signed_rank([], 0.5)
    # One pair is enough for the rank procedure: each test ranks one shifted difference.
    one = tost_signed_rank([0.1], 0.5)
    assert (one.p_lower, one.p_upper, one.p_value, one.pairs) == (0.5, 0.5, 0.5, 1)


@pytest.mark.parametrize("difference_p", [-0.01, 1.01, math.nan, math.inf, True, "0.5", None])
def test_the_report_refuses_something_that_is_not_a_p_value(difference_p):
    tost = TostSignedRankResult(0.01, 0.01, 0.01, 0.5, 10)
    with pytest.raises(ConfigError, match="p-value"):
        equivalence_report(tost, difference_p)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1, math.nan, True, "0.05", None])
def test_the_report_refuses_an_alpha_outside_zero_to_one(alpha):
    tost = TostSignedRankResult(0.01, 0.01, 0.01, 0.5, 10)
    with pytest.raises(ConfigError, match="alpha"):
        equivalence_report(tost, 0.5, alpha)


def test_the_report_accepts_the_result_of_either_procedure():
    differences = SAMPLES["equivalent"]
    assert equivalence_report(tost_paired_t(differences, 0.5), 0.6) == "equivalent"
    assert equivalence_report(tost_signed_rank(differences, 0.5), 0.6) == "equivalent"


def test_the_results_are_frozen_slotted_records_with_the_documented_fields():
    rank = [field.name for field in fields(TostSignedRankResult)]
    assert rank == ["p_lower", "p_upper", "p_value", "margin", "pairs"]
    paired = [field.name for field in fields(TostPairedTResult)]
    assert paired == [*rank, "mean_difference", "alpha", "interval"]
    for result in (tost_paired_t([0.1, 0.2, 0.3], 0.5), tost_signed_rank([0.1, 0.2, 0.3], 0.5)):
        with pytest.raises(FrozenInstanceError):
            result.p_value = 0.0  # type: ignore[misc]
        assert not hasattr(result, "__dict__")


def test_differences_may_be_any_iterable_of_numbers():
    generator = (value for value in (0.1, 0.2, -0.1, 0.3))
    assert tost_paired_t(generator, 0.5).pairs == 4
    assert tost_signed_rank((0.1, 0.2, -0.1, 0.3), 0.5).pairs == 4
    assert tost_signed_rank([1, 2, -1, 3], 5).margin == 5.0
    sample = congruential(12, 31, 89, 46, 0.012)
    assert tost_paired_t(sample, 0.25).pairs == 12
