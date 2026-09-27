"""Reference values for the distribution functions.

The literals below were computed once with SciPy 1.18.1
(``scipy.stats.norm``, ``chi2``, and ``f``) and pasted here; neither the
library nor these tests import SciPy. Acklam's normal quantile is checked
against its documented relative error of 1.15e-9, every other function to a
relative error of 1e-12.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from dispatchatlas.analysis.stats import distributions
from dispatchatlas.analysis.stats.distributions import (
    chi2_sf,
    f_sf,
    normal_cdf,
    normal_ppf,
    normal_sf,
)
from dispatchatlas.core.errors import ConfigError

NORMAL_CDF = [
    (-30.0, 4.906713927147908e-198),
    (-10.0, 7.61985302416047e-24),
    (-5.0, 2.866515718791933e-07),
    (-1.96, 0.024997895148220435),
    (-1.0, 0.15865525393145707),
    (-0.25, 0.4012936743170763),
    (0.0, 0.5),
    (0.5, 0.6914624612740131),
    (1.0, 0.8413447460685429),
    (1.96, 0.9750021048517795),
    (3.0, 0.9986501019683699),
    (8.0, 0.9999999999999993),
]
NORMAL_SF = [
    (-3.0, 0.9986501019683699),
    (0.0, 0.5),
    (1.0, 0.15865525393145707),
    (2.5, 0.006209665325776132),
    (5.0, 2.866515718791933e-07),
    (10.0, 7.61985302416047e-24),
    (20.0, 2.7536241186061556e-89),
    (37.0, 5.7255712225239266e-300),
]
NORMAL_PPF = [
    (1e-12, -7.034483825301131),
    (1e-06, -4.753424308822899),
    (0.001, -3.090232306167813),
    (0.02, -2.053748910631823),
    (0.02425, -1.972961051311885),
    (0.025, -1.9599639845400545),
    (0.1, -1.2815515655446004),
    (0.3, -0.5244005127080409),
    (0.7, 0.5244005127080407),
    (0.975, 1.959963984540054),
    (0.97575, 1.972961051311885),
    (0.99, 2.3263478740408408),
    (0.999999, 4.753424308817087),
]
CHI2_SF = [
    (0.1, 1, 0.7518296340458492),
    (0.5, 1, 0.47950012218695337),
    (2.9, 1, 0.08857955257977622),
    (3.1, 1, 0.07829229414640966),
    (13, 1, 0.0003114909767673841),
    (0.1, 2, 0.951229424500714),
    (1.0, 2, 0.6065306597126334),
    (3.9, 2, 0.14227407158651356),
    (4.1, 2, 0.12873490358780423),
    (16, 2, 0.0003354626279025119),
    (0.1, 3, 0.9918374237318764),
    (1.5, 3, 0.6822703303362125),
    (4.9, 3, 0.17926780368434939),
    (5.1, 3, 0.16461940408195308),
    (19, 3, 0.0002733988874908009),
    (0.1, 5, 0.9998376833880774),
    (2.5, 5, 0.7764950711233227),
    (6.9, 5, 0.22818430805111548),
    (7.1, 5, 0.21330878992839813),
    (25, 5, 0.0001393337911856263),
    (0.1, 10, 0.9999999975020487),
    (5.0, 10, 0.8911780189141513),
    (11.9, 10, 0.2918048992122272),
    (12.1, 10, 0.2784196375227773),
    (40, 10, 1.694474393006737e-05),
    (0.1, 30, 1.0),
    (15.0, 30, 0.9897395720876574),
    (31.9, 30, 0.37219263906631933),
    (32.1, 30, 0.3628911468709768),
    (100, 30, 1.8568023365102314e-09),
    (0.1, 100, 1.0),
    (50.0, 100, 0.9999930466947524),
    (101.9, 100, 0.42834171717764924),
    (102.1, 100, 0.4228739243260632),
    (310, 100, 2.4529530895502585e-23),
    (0.1, 1000, 1.0),
    (500.0, 1000, 1.0),
    (1001.9, 1000, 0.47712766259631517),
    (1002.1, 1000, 0.47534917481192535),
    (3010, 1000, 5.818415604564598e-200),
]
CHI2_SF_LARGE = [
    (10000.0, 10000, 0.49811936596618267),
    (50000.0, 50000, 0.4991589556391113),
    (1000000.0, 1000000, 0.4998119368033945),
]
F_SF = [
    (0.2, 1, 2, 0.6984886554222364),
    (1.0, 1, 2, 0.42264973081037427),
    (3.0, 1, 2, 0.22540333075851665),
    (20.0, 1, 2, 0.04653741075440769),
    (0.2, 1, 10, 0.6642514717311352),
    (1.0, 1, 10, 0.34089313230206),
    (3.0, 1, 10, 0.1139374121519203),
    (20.0, 1, 10, 0.0011934668300203242),
    (0.2, 1, 50, 0.6566522291891375),
    (1.0, 1, 50, 0.32212564510024433),
    (3.0, 1, 50, 0.08943024142233263),
    (20.0, 1, 50, 4.4691919874169643e-05),
    (0.2, 1, 1000, 0.654817692227434),
    (1.0, 1, 1000, 0.31755241808467255),
    (3.0, 1, 1000, 0.08357287713077692),
    (20.0, 1, 1000, 8.630967031997829e-06),
    (0.2, 2, 2, 0.8333333333333333),
    (1.0, 2, 2, 0.5),
    (3.0, 2, 2, 0.25),
    (20.0, 2, 2, 0.047619047619047616),
    (0.2, 2, 10, 0.8219271067593518),
    (1.0, 2, 10, 0.4018775720164609),
    (3.0, 2, 10, 0.09536743164062497),
    (20.0, 2, 10, 0.0003200000000000001),
    (0.2, 2, 50, 0.8193825245475238),
    (1.0, 2, 50, 0.3751168022539642),
    (3.0, 2, 50, 0.05882330655225374),
    (20.0, 2, 50, 4.151331094201027e-07),
    (0.2, 2, 1000, 0.8187634942322402),
    (1.0, 2, 1000, 0.36824701435263507),
    (3.0, 2, 1000, 0.05023537401574282),
    (20.0, 2, 1000, 3.0431989864223175e-09),
    (0.2, 4, 2, 0.9183673469387755),
    (1.0, 4, 2, 0.5555555555555556),
    (3.0, 4, 2, 0.2653061224489796),
    (20.0, 4, 2, 0.04818560380725759),
    (0.2, 4, 10, 0.9326510477869951),
    (1.0, 4, 10, 0.451555049341686),
    (3.0, 4, 10, 0.07232322228814021),
    (20.0, 4, 10, 9.220214473478709e-05),
    (0.2, 4, 50, 0.9371882752856744),
    (1.0, 4, 50, 0.41642143252942054),
    (3.0, 4, 50, 0.026963406052932723),
    (20.0, 4, 50, 6.919943552910409e-10),
    (0.2, 4, 1000, 0.9383837828799109),
    (1.0, 4, 1000, 0.40654611103932314),
    (3.0, 4, 1000, 0.017799390987706498),
    (20.0, 4, 1000, 7.384633327469185e-16),
    (0.2, 9, 2, 0.9653502513810186),
    (1.0, 9, 2, 0.5946555702940973),
    (3.0, 9, 2, 0.27498674237812226),
    (20.0, 9, 2, 0.048508250034138695),
    (0.2, 9, 10, 0.9881679196908592),
    (1.0, 9, 10, 0.49539021990783394),
    (3.0, 9, 10, 0.051002477672771954),
    (20.0, 9, 10, 2.9509374381178428e-05),
    (0.2, 9, 50, 0.9930947156003058),
    (1.0, 9, 50, 0.45244561214953916),
    (3.0, 9, 50, 0.0061511486673617366),
    (20.0, 9, 50, 1.076441450962294e-13),
    (0.2, 9, 1000, 0.9941933147675037),
    (1.0, 9, 1000, 0.43810091278889274),
    (3.0, 9, 1000, 0.001547163961210472),
    (20.0, 9, 1000, 4.0311469565470495e-31),
]


@pytest.mark.parametrize(("z", "expected"), NORMAL_CDF)
def test_normal_cdf_matches_scipy(z, expected):
    assert normal_cdf(z) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(("z", "expected"), NORMAL_SF)
def test_normal_sf_keeps_precision_deep_in_the_tail(z, expected):
    assert normal_sf(z) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(("p", "expected"), NORMAL_PPF)
def test_normal_ppf_is_within_acklams_error_bound(p, expected):
    assert normal_ppf(p) == pytest.approx(expected, rel=1.15e-9)


def test_normal_ppf_is_zero_at_one_half_and_refuses_the_ends():
    assert normal_ppf(0.5) == 0.0
    for probability in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ConfigError, match="probability"):
            normal_ppf(probability)


@pytest.mark.parametrize(("x", "df", "expected"), CHI2_SF)
def test_chi2_sf_matches_scipy_on_both_sides_of_the_series_transition(x, df, expected):
    assert chi2_sf(x, df) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(("x", "df", "expected"), CHI2_SF_LARGE)
def test_chi2_sf_converges_for_very_large_degrees_of_freedom(x, df, expected):
    assert chi2_sf(x, df) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(("f", "d1", "d2", "expected"), F_SF)
def test_f_sf_matches_scipy(f, d1, d2, expected):
    assert f_sf(f, d1, d2) == pytest.approx(expected, rel=1e-12)


def test_tails_are_one_below_zero_and_zero_at_infinity():
    assert chi2_sf(0.0, 3) == 1.0
    assert chi2_sf(-2.0, 3) == 1.0
    assert chi2_sf(math.inf, 3) == 0.0
    assert f_sf(0.0, 2, 5) == 1.0
    assert f_sf(math.inf, 2, 5) == 0.0


def test_invalid_arguments_are_refused():
    with pytest.raises(ConfigError):
        chi2_sf(1.0, 0)
    with pytest.raises(ConfigError):
        chi2_sf(math.nan, 2)
    with pytest.raises(ConfigError):
        f_sf(1.0, 2, -1)
    with pytest.raises(ConfigError):
        f_sf(math.nan, 2, 3)


def test_tails_decrease_monotonically():
    grid = [0.25 * step for step in range(1, 120)]
    chi = [chi2_sf(x, 4) for x in grid]
    f = [f_sf(x / 4.0, 3, 12) for x in grid]
    assert all(a >= b for a, b in pairwise(chi))
    assert all(a >= b for a, b in pairwise(f))


def test_non_converging_expansions_raise_instead_of_returning_a_partial_sum(monkeypatch):
    monkeypatch.setattr(distributions, "_cap", lambda *shapes: 2)
    with pytest.raises(ArithmeticError, match="did not converge"):
        chi2_sf(3.0, 4)
    with pytest.raises(ArithmeticError, match="did not converge"):
        chi2_sf(40.0, 4)
    with pytest.raises(ArithmeticError, match="did not converge"):
        f_sf(1.5, 5, 30)
