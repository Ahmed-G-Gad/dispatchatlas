"""Bootstrap confidence intervals: bias-corrected and accelerated, and percentile.

:func:`bootstrap_interval` resamples the observations with replacement,
computes the statistic on every resample, and reads the interval from the
sorted replicates with the linear-interpolation quantile :func:`quantile`
(type 7 of Hyndman and Fan 1996).

The **percentile** interval takes the replicate quantiles at ``alpha/2`` and
``1 - alpha/2``. The **bias-corrected and accelerated** (BCa) interval (Efron
1987) shifts those levels by the bias correction ``z0 = Phi^-1(#{replicates <
estimate} / B)``, with a strict inequality, and by the acceleration
``a = sum(d^3) / (6 (sum(d^2))^(3/2))`` from the leave-one-out jackknife
deviations ``d``, and reads the quantiles at
``Phi(z0 + (z0 + z) / (1 - a (z0 + z)))`` for ``z = Phi^-1(alpha/2)`` and
``Phi^-1(1 - alpha/2)``, clamped to ``[1e-9, 1 - 1e-9]``. When every replicate
lies on one side of the estimate, ``z0`` is infinite and the interval falls
back to the percentile one; the same happens when ``1 - a (z0 + z)`` is not
positive, where the correction is undefined. A sample of one value, or zero
resamples, gives the degenerate interval at the estimate.

Resampling draws indices from ``numpy.random.default_rng(seed)``, one row of
``n`` indices per resample, so equal inputs and seed give equal intervals for
a given NumPy version.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.analysis.stats.distributions import normal_cdf, normal_ppf
from dispatchatlas.core.errors import ConfigError

__all__ = [
    "BootstrapInterval",
    "IntervalMethod",
    "Statistic",
    "bootstrap_interval",
    "quantile",
]

type IntervalMethod = Literal["bca", "percentile"]
"""The bootstrap interval to compute."""

type Statistic = Literal["mean", "median"] | Callable[[NDArray[np.float64]], float]
"""A statistic: the mean, the median, or a function of the resampled rows."""

_CLAMP = 1e-9
_JACKKNIFE_CHUNK = 256
"""Leave-one-out samples evaluated at once, which bounds the jackknife's memory."""


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A bootstrap confidence interval.

    Attributes:
        estimate: The statistic of the observed sample.
        low: Lower bound.
        high: Upper bound.
        level: Nominal coverage, for example 0.95.
        resamples: Number of resamples drawn.
        method: The interval computed: ``"bca"``, ``"percentile"`` (asked
            for, or the fallback of BCa), or ``"degenerate"``.
    """

    estimate: float
    low: float
    high: float
    level: float
    resamples: int
    method: Literal["bca", "percentile", "degenerate"]


def quantile(ordered: Sequence[float], q: float) -> float:
    """Return the ``q`` quantile of sorted values by linear interpolation.

    The position ``(n - 1) q`` interpolates between the neighbouring order
    statistics, type 7 of Hyndman and Fan (1996) and NumPy's default.

    Args:
        ordered: Values sorted in ascending order.
        q: A probability in ``[0, 1]``.

    Raises:
        ConfigError: If ``ordered`` is empty or ``q`` is outside ``[0, 1]``.
    """
    if not ordered:
        raise ConfigError("a quantile needs at least one value")
    if not 0.0 <= q <= 1.0:
        raise ConfigError(f"a quantile level must be in [0, 1], got {q!r}")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def bootstrap_interval(
    sample: Sequence[float] | Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    statistic: Statistic = "mean",
    level: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
    method: IntervalMethod = "bca",
) -> BootstrapInterval:
    """Return a bootstrap confidence interval of a statistic.

    Args:
        sample: The observations: numbers, or rows of numbers when the
            statistic is a function of several columns. Resampling draws
            whole rows.
        statistic: ``"mean"``, ``"median"``, or a function that maps an array
            of rows (one resample) to a number.
        level: Nominal coverage, strictly between 0 and 1.
        resamples: Number of resamples, at least 0.
        seed: Seed of the resampling generator, at least 0.
        method: ``"bca"`` or ``"percentile"``.

    Returns:
        The interval and the estimate on the observed sample.

    Raises:
        ConfigError: If the sample is empty or not finite, or a parameter is
            out of range.
    """
    data = np.asarray(sample, dtype=np.float64)
    if data.ndim == 0 or data.shape[0] == 0:
        raise ConfigError("a bootstrap interval needs at least one observation")
    if not np.all(np.isfinite(data)):
        raise ConfigError("a bootstrap interval needs finite observations")
    if not 0.0 < level < 1.0:
        raise ConfigError(f"the confidence level must be in (0, 1), got {level!r}")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 0:
        raise ConfigError(f"resamples must be a non-negative integer, got {resamples!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"the seed must be a non-negative integer, got {seed!r}")
    if method not in ("bca", "percentile"):
        raise ConfigError(f"the interval method must be 'bca' or 'percentile', got {method!r}")
    evaluate = _evaluator(statistic, data.ndim)
    estimate = float(evaluate(data[np.newaxis, ...])[0])
    count = data.shape[0]
    if count == 1 or resamples == 0:
        return BootstrapInterval(estimate, estimate, estimate, level, resamples, "degenerate")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, count, size=(resamples, count))
    replicates = np.sort(evaluate(data[indices]))
    ordered = [float(value) for value in replicates]
    alpha = 1.0 - level
    if method == "percentile":
        return _percentile(ordered, estimate, level, resamples)
    below = int(np.count_nonzero(replicates < estimate))
    if below in (0, resamples):
        return _percentile(ordered, estimate, level, resamples)
    bias = normal_ppf(below / resamples)
    acceleration = _acceleration(_leave_one_out(data, statistic, evaluate))
    low = _bca_level(bias, acceleration, normal_ppf(alpha / 2.0))
    high = _bca_level(bias, acceleration, normal_ppf(1.0 - alpha / 2.0))
    if low is None or high is None:
        return _percentile(ordered, estimate, level, resamples)
    return BootstrapInterval(
        estimate, quantile(ordered, low), quantile(ordered, high), level, resamples, "bca"
    )


def _evaluator(
    statistic: Statistic, dimensions: int
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Return a function mapping a stack of resamples to one statistic per resample."""
    if statistic == "mean" and dimensions == 1:
        return lambda stack: np.asarray(np.mean(stack, axis=1), dtype=np.float64)
    if statistic == "median" and dimensions == 1:
        return lambda stack: np.asarray(np.median(stack, axis=1), dtype=np.float64)
    if callable(statistic):
        function = statistic
        return lambda stack: np.array([float(function(rows)) for rows in stack])
    raise ConfigError(
        f"statistic {statistic!r} needs one number per observation; pass a function for rows"
    )


def _leave_one_out(
    data: NDArray[np.float64],
    statistic: Statistic,
    evaluate: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Return the statistic of the sample without each observation in turn."""
    count = data.shape[0]
    if statistic == "mean" and data.ndim == 1:
        return np.asarray((np.sum(data) - data) / (count - 1), dtype=np.float64)
    parts = []
    for start in range(0, count, _JACKKNIFE_CHUNK):
        stop = min(count, start + _JACKKNIFE_CHUNK)
        rows = np.stack([np.delete(data, index, axis=0) for index in range(start, stop)])
        parts.append(evaluate(rows))
    return np.concatenate(parts)


def _acceleration(leave_one_out: NDArray[np.float64]) -> float:
    """Return the BCa acceleration from the leave-one-out jackknife estimates."""
    deviations = [float(value) for value in (np.mean(leave_one_out) - leave_one_out)]
    squares = math.fsum(value * value for value in deviations)
    denominator = 6.0 * squares * math.sqrt(squares)
    if denominator == 0.0:
        return 0.0
    return math.fsum(value * value * value for value in deviations) / denominator


def _bca_level(bias: float, acceleration: float, z: float) -> float | None:
    """Return the replicate quantile level of one BCa bound, or ``None`` if undefined."""
    shifted = bias + z
    denominator = 1.0 - acceleration * shifted
    if denominator <= 0.0:
        return None
    return min(1.0 - _CLAMP, max(_CLAMP, normal_cdf(bias + shifted / denominator)))


def _percentile(
    ordered: Sequence[float], estimate: float, level: float, resamples: int
) -> BootstrapInterval:
    alpha = 1.0 - level
    return BootstrapInterval(
        estimate,
        quantile(ordered, alpha / 2.0),
        quantile(ordered, 1.0 - alpha / 2.0),
        level,
        resamples,
        "percentile",
    )
