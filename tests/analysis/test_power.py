"""Power of the Wilcoxon signed-rank test: floors, an analytic case, and monotonicity."""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

from dispatchatlas.analysis.stats.power import (
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
