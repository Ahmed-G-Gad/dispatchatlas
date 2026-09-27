"""Effect sizes: hand-computed values and a SciPy reference for A12.

``A12_SAMPLES`` holds two tied samples and ``U1 / (n1 n2)`` from
``scipy.stats.mannwhitneyu`` (SciPy 1.18.1, computed once and pasted), which is
A12 when larger values are better. The library does not import SciPy.
"""

from __future__ import annotations

import math

import pytest

from dispatchatlas.analysis.stats.effect_sizes import (
    a12_magnitude,
    cliffs_delta,
    cliffs_delta_magnitude,
    cohens_d,
    cohens_d_magnitude,
    cohens_dz,
    paired_outcomes,
    paired_win_share,
    vargha_delaney_a12,
)
from dispatchatlas.core.errors import ConfigError

A12_SAMPLES = (
    [0.6, 1.3, -2.2, -1.6, 1.5, 0.6, 0.8, -0.3, -0.5],
    [0.8, -0.3, -0.9, -0.5, 0.7, 1.2, 2.3, 1.0, 1.1, 2.3, -0.2],
    0.35858585858585856,
)


def test_a12_and_cliffs_delta_match_the_mann_whitney_statistic():
    left, right, a12_max = A12_SAMPLES
    assert vargha_delaney_a12(left, right, "max") == pytest.approx(a12_max, rel=1e-12)
    assert vargha_delaney_a12(left, right, "min") == pytest.approx(1.0 - a12_max, rel=1e-12)
    assert cliffs_delta(left, right, "min") == pytest.approx(1.0 - 2.0 * a12_max, rel=1e-12)
    assert cliffs_delta(left, right, "max") == pytest.approx(2.0 * a12_max - 1.0, rel=1e-12)


def test_a12_orients_by_sense_and_counts_ties_as_half():
    assert vargha_delaney_a12([1.0, 2.0], [3.0, 4.0]) == 1.0
    assert vargha_delaney_a12([1.0, 2.0], [3.0, 4.0], "max") == 0.0
    assert vargha_delaney_a12([1.0], [1.0 + 1e-13]) == 0.5
    assert cliffs_delta([1.0], [1.0 + 1e-13]) == 0.0


def test_cohens_d_hand_value_and_orientation():
    left, right = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    # Means differ by 3; the pooled standard deviation is 1.
    assert cohens_d(left, right) == pytest.approx(3.0)
    assert cohens_d(left, right, "max") == pytest.approx(-3.0)
    assert cohens_d([2.0, 2.0], [2.0, 2.0]) == 0.0
    with pytest.raises(ConfigError, match="two values"):
        cohens_d([1.0], [2.0, 3.0])


def test_cohens_dz_is_the_standardized_mean_difference():
    assert cohens_dz([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert cohens_dz([-1.0, -2.0, -3.0]) == pytest.approx(-2.0)
    assert cohens_dz([1.0]) is None
    assert cohens_dz([2.0, 2.0, 2.0]) is None
    assert cohens_dz([1.0, math.inf]) is None


def test_paired_outcomes_and_win_share():
    left, right = [1.0, 5.0, 3.0, 4.0], [2.0, 4.0, 3.0 + 1e-13, 9.0]
    assert paired_outcomes(left, right) == (2, 1, 1)
    assert paired_win_share(left, right) == pytest.approx(2.5 / 4)
    assert paired_win_share(left, right, "max") == pytest.approx(1.5 / 4)
    assert paired_outcomes([math.inf], [math.inf]) == (0, 0, 1)
    assert paired_outcomes([5.0], [math.inf]) == (1, 0, 0)
    with pytest.raises(ConfigError, match="same length"):
        paired_outcomes([1.0], [1.0, 2.0])
    with pytest.raises(ConfigError, match="one pair"):
        paired_win_share([], [])


def test_published_magnitude_labels():
    assert [a12_magnitude(value) for value in (0.5, 0.56, 0.64, 0.71, 0.29)] == [
        "negligible",
        "small",
        "medium",
        "large",
        "large",
    ]
    assert [cliffs_delta_magnitude(value) for value in (0.1, 0.147, 0.33, -0.474)] == [
        "negligible",
        "small",
        "medium",
        "large",
    ]
    assert [cohens_d_magnitude(value) for value in (0.1, 0.2, 0.5, -0.8)] == [
        "negligible",
        "small",
        "medium",
        "large",
    ]


def test_empty_or_nan_samples_are_refused():
    with pytest.raises(ConfigError):
        vargha_delaney_a12([], [1.0])
    with pytest.raises(ConfigError):
        cliffs_delta([math.nan], [1.0])
