"""Bootstrap intervals: a SciPy reference and golden values.

``SCIPY_BCA`` and ``SCIPY_PERCENTILE`` were computed once with
``scipy.stats.bootstrap(..., n_resamples=1000, rng=numpy.random.default_rng(0))``
(SciPy 1.18.1) on ``BOOTSTRAP_SAMPLE`` and pasted here. The library draws the
same resampling indices, so the percentile interval agrees to rounding, and
the BCa interval to about 1e-9 relative: the bias correction goes through
Acklam's normal quantile. No replicate ties the estimate, where SciPy counts
half and the library, by design, counts none. Neither side imports SciPy.
"""

from __future__ import annotations

import pytest

from dispatchatlas.analysis.stats.bootstrap import bootstrap_interval, quantile
from dispatchatlas.core.errors import ConfigError

BOOTSTRAP_SAMPLE = [
    1.001231,
    1.348167,
    0.760227,
    0.410413,
    0.634657,
    0.370965,
    1.061989,
    3.819866,
    0.611276,
    0.537689,
    1.632058,
    1.428874,
    1.111171,
    0.394369,
    0.971172,
]
SCIPY_BCA = (0.8131097619943408, 1.7891747077870837)
SCIPY_PERCENTILE = (0.7408780800000002, 1.5460049583333335)

GOLDEN_BCA = (1.0729416, 0.8131097618851327, 1.7891747095584933)
GOLDEN_MEDIAN_BCA = (0.971172, 0.537689, 1.061989)


def test_bca_matches_scipy_and_the_golden_value():
    interval = bootstrap_interval(BOOTSTRAP_SAMPLE)
    assert interval.method == "bca"
    assert (interval.low, interval.high) == pytest.approx(SCIPY_BCA, rel=2e-9)
    assert (interval.estimate, interval.low, interval.high) == pytest.approx(GOLDEN_BCA, rel=1e-12)


def test_percentile_interval_matches_scipy():
    interval = bootstrap_interval(BOOTSTRAP_SAMPLE, method="percentile")
    assert interval.method == "percentile"
    assert (interval.low, interval.high) == pytest.approx(SCIPY_PERCENTILE, rel=1e-12)


def test_median_bca_golden_value():
    interval = bootstrap_interval(BOOTSTRAP_SAMPLE, statistic="median")
    assert (interval.estimate, interval.low, interval.high) == pytest.approx(
        GOLDEN_MEDIAN_BCA, rel=1e-12
    )


def test_intervals_are_deterministic_and_depend_on_the_seed():
    first = bootstrap_interval(BOOTSTRAP_SAMPLE, seed=5)
    assert bootstrap_interval(BOOTSTRAP_SAMPLE, seed=5) == first
    assert bootstrap_interval(BOOTSTRAP_SAMPLE, seed=6) != first
    assert first.low < first.estimate < first.high


def test_degenerate_and_fallback_intervals():
    single = bootstrap_interval([3.0])
    assert (single.low, single.high, single.method) == (3.0, 3.0, "degenerate")
    none = bootstrap_interval([1.0, 2.0], resamples=0)
    assert none.method == "degenerate"
    constant = bootstrap_interval([2.0, 2.0, 2.0])
    assert (constant.low, constant.high, constant.method) == (2.0, 2.0, "percentile")


def test_a_statistic_of_rows_resamples_whole_rows():
    rows = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    interval = bootstrap_interval(rows, statistic=lambda sample: float(sample[:, 1].sum()))
    assert interval.estimate == 100.0
    assert interval.low <= interval.estimate <= interval.high
    with pytest.raises(ConfigError, match="function"):
        bootstrap_interval(rows)


def test_quantile_is_type_seven():
    ordered = [1.0, 2.0, 4.0, 8.0]
    assert quantile(ordered, 0.0) == 1.0
    assert quantile(ordered, 1.0) == 8.0
    assert quantile(ordered, 0.5) == 3.0
    assert quantile(ordered, 0.25) == pytest.approx(1.75)
    with pytest.raises(ConfigError):
        quantile([], 0.5)


@pytest.mark.parametrize(
    "arguments",
    [
        {"sample": []},
        {"level": 1.0},
        {"level": 0.0},
        {"resamples": -1},
        {"seed": -2},
        {"method": "basic"},
        {"sample": [1.0, float("inf")]},
    ],
)
def test_invalid_arguments_are_refused(arguments):
    options = {"sample": [1.0, 2.0, 3.0]} | arguments
    sample = options.pop("sample")
    with pytest.raises(ConfigError):
        bootstrap_interval(sample, **options)
