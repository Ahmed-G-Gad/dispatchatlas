"""Power of the paired Wilcoxon signed-rank test, and the sample size it needs.

The pairwise comparisons test one paired difference per problem, so their
sample size is the number of problems two solvers share, not the number of
runs. Two questions follow, one for reading a campaign and one for planning
it (Campelo and Takahashi 2019): which standardized effect can ``n`` pairs
detect, and how many pairs does an effect need?

Both are answered for the test this package applies, not for an asymptotic
stand-in: :func:`wilcoxon_power` simulates paired differences drawn from a
normal distribution with mean ``effect`` and standard deviation 1, so the
effect is the paired Cohen's d_z, and counts how often the two-sided test
rejects at ``alpha``. It uses the exact null distribution up to
:data:`~dispatchatlas.analysis.stats.nonparametric.WILCOXON_EXACT_LIMIT`
pairs and the continuity-corrected normal approximation above, as
:func:`~dispatchatlas.analysis.stats.nonparametric.wilcoxon_signed_rank` does.
The draws come from ``numpy.random.default_rng(seed)``, so results are
reproducible, and every effect is evaluated on the same draws, which keeps
the simulated power monotone enough to search.

With ``n`` pairs, the smallest two-sided exact p-value is ``2 / 2^n``, so
below :func:`attainable_floor` no data can reach significance at ``alpha``,
whatever the effect. :func:`detectable_effect_size` refuses such counts
instead of reporting a detectable effect that does not exist.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.analysis.stats.distributions import normal_ppf
from dispatchatlas.analysis.stats.nonparametric import WILCOXON_EXACT_LIMIT
from dispatchatlas.core.errors import ConfigError

__all__ = [
    "PowerAssessment",
    "attainable_floor",
    "detectable_effect_size",
    "required_paired_count",
    "wilcoxon_power",
]

_BLOCK_ELEMENTS = 2_000_000
"""Simulated differences held in memory at once."""

_BISECTIONS = 20
"""Halvings of the effect interval when searching the detectable effect."""


@dataclass(frozen=True, slots=True)
class PowerAssessment:
    """The smallest standardized effect a paired design detects.

    Attributes:
        pairs: Number of paired observations, such as shared problems.
        alpha: Two-sided significance level.
        power: Target probability of rejecting the null hypothesis.
        effect: Smallest paired Cohen's d_z detected with that power.
        replicates: Simulated samples per power estimate.
        seed: Seed of the simulation.
    """

    pairs: int
    alpha: float
    power: float
    effect: float
    replicates: int
    seed: int


def attainable_floor(alpha: float = 0.05) -> int:
    """Return the fewest pairs with which a two-sided Wilcoxon test can reject at ``alpha``.

    With ``n`` pairs the smallest attainable p-value is ``2 / 2^n``, reached
    when every difference has the same sign. At ``alpha = 0.05`` the floor
    is 6 pairs (``2 / 64 = 0.031``); 5 pairs cannot go below 0.0625.

    Raises:
        ConfigError: If ``alpha`` is not strictly between 0 and 1.
    """
    _rate(alpha, "alpha")
    pairs = 1
    while 2.0 / 2.0**pairs > alpha:
        pairs += 1
    return pairs


def wilcoxon_power(
    pairs: int,
    effect: float,
    *,
    alpha: float = 0.05,
    replicates: int = 10000,
    seed: int = 0,
) -> float:
    """Return the simulated power of the two-sided Wilcoxon signed-rank test.

    Args:
        pairs: Number of paired differences, at least 1.
        effect: Mean of the normal differences, in standard deviations.
        alpha: Two-sided significance level.
        replicates: Simulated samples, at least 1.
        seed: Seed of the simulation, at least 0.

    Returns:
        The share of simulated samples the test rejects.

    Raises:
        ConfigError: If an argument is out of range.
    """
    _counts(pairs, replicates, seed)
    _rate(alpha, "alpha")
    if not math.isfinite(effect):
        raise ConfigError(f"the effect must be finite, got {effect!r}")
    threshold = _rejection_threshold(pairs, alpha)
    if threshold is None:
        return 0.0
    rejections = 0
    for block in _draws(pairs, replicates, seed):
        rejections += _rejections(block + effect, threshold)
    return rejections / replicates


def detectable_effect_size(
    pairs: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    replicates: int = 10000,
    seed: int = 0,
) -> PowerAssessment:
    """Return the smallest paired d_z that ``pairs`` pairs detect with the given power.

    The effect is found by bisection on :func:`wilcoxon_power`, evaluated on
    one fixed set of simulated samples.

    Args:
        pairs: Number of paired observations.
        alpha: Two-sided significance level.
        power: Target power, strictly between 0 and 1.
        replicates: Simulated samples per power estimate.
        seed: Seed of the simulation.

    Raises:
        ConfigError: If ``pairs`` is below :func:`attainable_floor`, where no
            effect is detectable, or an argument is out of range.
    """
    _counts(pairs, replicates, seed)
    _rate(alpha, "alpha")
    _rate(power, "power")
    if _rejection_threshold(pairs, alpha) is None:
        raise ConfigError(
            f"with {pairs} pairs the smallest two-sided Wilcoxon p-value is "
            f"{2.0 / 2.0**pairs:.4g}, above alpha = {alpha:g}: no effect is detectable; "
            f"at least {attainable_floor(alpha)} pairs are needed"
        )

    def simulated(effect: float) -> float:
        return wilcoxon_power(pairs, effect, alpha=alpha, replicates=replicates, seed=seed)

    low, high = 0.0, 1.0
    while simulated(high) < power:
        low, high = high, 2.0 * high
    for _ in range(_BISECTIONS):
        middle = (low + high) / 2.0
        if simulated(middle) >= power:
            high = middle
        else:
            low = middle
    return PowerAssessment(pairs, float(alpha), float(power), high, replicates, seed)


def required_paired_count(
    effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    replicates: int = 10000,
    seed: int = 0,
) -> int:
    """Return the number of pairs that detects a paired d_z with the given power.

    Starting from :func:`attainable_floor`, the count doubles until the
    simulated power reaches the target, and bisection then finds the
    smallest count that reaches it.

    Args:
        effect: The paired Cohen's d_z to detect, positive.
        alpha: Two-sided significance level.
        power: Target power, strictly between 0 and 1.
        replicates: Simulated samples per power estimate.
        seed: Seed of the simulation.

    Raises:
        ConfigError: If an argument is out of range.
    """
    if not (math.isfinite(effect) and effect > 0.0):
        raise ConfigError(f"the effect must be positive and finite, got {effect!r}")
    _rate(alpha, "alpha")
    _rate(power, "power")
    _counts(1, replicates, seed)

    def reaches(pairs: int) -> bool:
        return wilcoxon_power(pairs, effect, alpha=alpha, replicates=replicates, seed=seed) >= power

    low = attainable_floor(alpha)
    if reaches(low):
        return low
    high = 2 * low
    while not reaches(high):
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        if reaches(middle):
            high = middle
        else:
            low = middle
    return high


def _rejection_threshold(pairs: int, alpha: float) -> float | None:
    """Return the largest ``min(W+, W-)`` the two-sided test rejects, or ``None`` if none.

    Up to the exact limit it is the largest integer ``c`` whose exact
    two-sided p-value ``2 P(W <= c)`` is at most ``alpha``. Above, it is the
    statistic at which the continuity-corrected normal p-value equals
    ``alpha``, when that statistic is attainable.
    """
    if pairs <= WILCOXON_EXACT_LIMIT:
        counts = [0] * (pairs * (pairs + 1) // 2 + 1)
        counts[0] = 1
        for rank in range(1, pairs + 1):
            for total in range(len(counts) - 1, rank - 1, -1):
                counts[total] += counts[total - rank]
        threshold = None
        cumulative = 0
        for value, count in enumerate(counts):
            cumulative += count
            if 2.0 * cumulative / (1 << pairs) > alpha:
                break
            threshold = float(value)
        return threshold
    mean = pairs * (pairs + 1) / 4.0
    deviation = math.sqrt(pairs * (pairs + 1) * (2 * pairs + 1) / 24.0)
    threshold = mean - 0.5 - normal_ppf(1.0 - alpha / 2.0) * deviation
    return threshold if threshold >= 0.0 else None


def _draws(pairs: int, replicates: int, seed: int) -> Iterator[NDArray[np.float64]]:
    """Yield the standard normal samples in blocks of rows that bound the memory."""
    generator = np.random.default_rng(seed)
    rows = max(1, _BLOCK_ELEMENTS // pairs)
    for start in range(0, replicates, rows):
        yield generator.standard_normal((min(rows, replicates - start), pairs))


def _rejections(differences: NDArray[np.float64], threshold: float) -> int:
    """Count the rows whose ``min(W+, W-)`` is at most ``threshold``."""
    count = differences.shape[1]
    ranks = np.argsort(np.argsort(np.abs(differences), axis=1), axis=1) + 1
    positive = np.sum(np.where(differences > 0.0, ranks, 0), axis=1)
    statistic = np.minimum(positive, count * (count + 1) // 2 - positive)
    return int(np.count_nonzero(statistic <= threshold + 1e-9))


def _rate(value: float, what: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value < 1.0:
        raise ConfigError(f"{what} must lie strictly between 0 and 1, got {value!r}")


def _counts(pairs: int, replicates: int, seed: int) -> None:
    for what, value, minimum in (("pairs", pairs, 1), ("replicates", replicates, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ConfigError(f"{what} must be an integer of at least {minimum}, got {value!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"the seed must be a non-negative integer, got {seed!r}")
