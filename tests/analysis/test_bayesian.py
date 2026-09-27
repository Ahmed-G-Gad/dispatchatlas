"""Bayesian sign and signed-rank tests against independent references.

``SIGN_TEST_PROBABILITIES`` are the posterior probabilities that each region
holds the largest mass under ``Dirichlet(2, 3, 6)``, integrated numerically
once with ``scipy.integrate.dblquad`` over ``scipy.stats.dirichlet``
(SciPy 1.18.1). ``SIGNED_RANK_BAYCOMP`` comes from the reference
implementation of the method's authors, baycomp 1.0.3
(``SignedRankTest.sample(..., rope=0.1, prior=0.5, nsamples=300000)``):
the probabilities and the mean region masses. Both were pasted here; nothing
here imports SciPy or baycomp.
"""

from __future__ import annotations

import math

import pytest

from dispatchatlas.analysis.stats.bayesian import bayesian_sign_test, bayesian_signed_rank_test
from dispatchatlas.core.errors import ConfigError

DIFFERENCES = [0.8, 1.3, -0.4, 0.05, 2.1, 0.9, -0.02, 1.7, 0.6, -1.1]
SIGN_TEST_PROBABILITIES = (0.047915714067977525, 0.1341193463267797, 0.8179649396052444)
SIGNED_RANK_BAYCOMP = (
    (0.042126666666666666, 0.0033366666666666666, 0.9545366666666667),
    (0.18228072455928182, 0.10546741114079591, 0.7122518642999097),
)


def test_sign_test_posterior_means_are_exact():
    result = bayesian_sign_test(DIFFERENCES, rope=0.1)
    # Two differences below the rope, two inside plus the prior, six above.
    assert (result.mean_below, result.mean_rope, result.mean_above) == pytest.approx(
        (2 / 11, 3 / 11, 6 / 11)
    )


def test_sign_test_probabilities_match_the_integrated_posterior():
    result = bayesian_sign_test(DIFFERENCES, rope=0.1, samples=20000)
    found = (result.probability_below, result.probability_rope, result.probability_above)
    assert found == pytest.approx(SIGN_TEST_PROBABILITIES, abs=0.01)
    golden = bayesian_sign_test(DIFFERENCES, rope=0.1)
    assert (golden.probability_below, golden.probability_rope, golden.probability_above) == (
        0.0465,
        0.136,
        0.8175,
    )


def test_signed_rank_matches_baycomp():
    result = bayesian_signed_rank_test(DIFFERENCES, rope=0.1, samples=20000)
    probabilities, means = SIGNED_RANK_BAYCOMP
    found = (result.probability_below, result.probability_rope, result.probability_above)
    assert found == pytest.approx(probabilities, abs=0.01)
    assert (result.mean_below, result.mean_rope, result.mean_above) == pytest.approx(
        means, abs=0.005
    )


def test_signed_rank_single_pair_has_an_analytic_posterior():
    # Atoms 0 and 1 with Dirichlet(1, 1) weights: the rope holds w0^2, so its
    # expected mass is 1/3 and it is the largest region when w0 > sqrt(1/2).
    result = bayesian_signed_rank_test([1.0], prior=1.0, samples=20000)
    assert result.mean_below == 0.0
    assert result.mean_rope == pytest.approx(1 / 3, abs=0.01)
    assert result.probability_above == pytest.approx(math.sqrt(0.5), abs=0.01)


def test_signed_rank_saturates_inside_a_wide_rope_and_is_seeded():
    inside = bayesian_signed_rank_test([0.01, -0.02, 0.03] * 4, rope=1.0)
    assert inside.probability_rope == 1.0
    first = bayesian_signed_rank_test(DIFFERENCES, rope=0.1, seed=3)
    assert bayesian_signed_rank_test(DIFFERENCES, rope=0.1, seed=3) == first
    assert bayesian_signed_rank_test(DIFFERENCES, rope=0.1, seed=4) != first


@pytest.mark.parametrize(
    "options",
    [{"rope": -0.1}, {"prior": 0.0}, {"samples": 0}, {"seed": -1}],
)
def test_invalid_parameters_are_refused(options):
    with pytest.raises(ConfigError):
        bayesian_sign_test(DIFFERENCES, **options)
    with pytest.raises(ConfigError):
        bayesian_signed_rank_test(DIFFERENCES, **options)


def test_empty_or_non_finite_differences_are_refused():
    with pytest.raises(ConfigError):
        bayesian_sign_test([])
    with pytest.raises(ConfigError):
        bayesian_signed_rank_test([1.0, math.inf])
    assert bayesian_sign_test([math.inf, 1.0]).mean_above == pytest.approx(2 / 3)
