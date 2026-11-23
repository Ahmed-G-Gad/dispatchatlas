"""Equivalence tests: two one-sided tests of paired differences against a margin.

A test that finds no difference does not show that there is none: a small or
noisy sample lacks the power to find one. An equivalence test turns the
question round (Schuirmann 1987; Lakens 2017). The analyst fixes a margin,
the largest difference too small to matter, before looking at the data, and
the null hypothesis is that the true difference lies outside
``[-margin, margin]``. That null is the union of two one-sided nulls, a
difference of at most ``-margin`` and one of at least ``+margin``. Each is
tested by a one-sided test at the level of the whole procedure, and
equivalence is shown only when both reject, so the p-value of the procedure
is the larger of the two. This is the procedure of the two one-sided tests
(TOST).

- :func:`tost_paired_t` runs the two one-sided paired t tests and reports the
  mean difference with its ``(1 - 2 alpha)`` confidence interval.
- :func:`tost_signed_rank` runs the two one-sided Wilcoxon signed-rank tests
  of :func:`~dispatchatlas.analysis.stats.nonparametric.wilcoxon_signed_rank`
  on the differences shifted by each bound.
- :func:`equivalence_report` combines the equivalence test with a test of a
  difference into one of four outcomes.

Differences are oriented as the first sample minus the second, or the other
way round, as the caller chooses: the margin is symmetric, so equivalence does
not depend on the orientation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from dispatchatlas.analysis.stats.distributions import t_ppf, t_sf
from dispatchatlas.analysis.stats.nonparametric import ZeroMethod, wilcoxon_signed_rank
from dispatchatlas.analysis.stats.ranks import finite_or_infinite
from dispatchatlas.core.errors import ConfigError

__all__ = [
    "EquivalenceOutcome",
    "TostPairedTResult",
    "TostResult",
    "TostSignedRankResult",
    "equivalence_report",
    "tost_paired_t",
    "tost_signed_rank",
]

type EquivalenceOutcome = Literal[
    "equivalent", "different", "equivalent-and-different", "inconclusive"
]
"""What an equivalence test and a test of a difference show together.

See :func:`equivalence_report` for the rule of each outcome.
"""


@dataclass(frozen=True, slots=True)
class TostPairedTResult:
    """The two one-sided paired t tests of equivalence within a margin.

    Attributes:
        p_lower: p-value of the test against the lower bound, whose null
            hypothesis is that the true mean difference is at most
            ``-margin``.
        p_upper: p-value of the test against the upper bound, whose null
            hypothesis is that it is at least ``margin``.
        p_value: The larger of the two, the p-value of the procedure:
            equivalence is shown at level ``alpha`` when it is below
            ``alpha``.
        margin: The equivalence margin, in the units of the differences.
        pairs: Number of paired differences.
        mean_difference: The mean of the paired differences.
        alpha: The level of the procedure, which sets the interval.
        interval: The ``(1 - 2 alpha)`` confidence interval of the mean
            difference. Equivalence is shown at level ``alpha`` exactly when
            the interval lies inside ``(-margin, margin)``.
    """

    p_lower: float
    p_upper: float
    p_value: float
    margin: float
    pairs: int
    mean_difference: float
    alpha: float
    interval: tuple[float, float]


@dataclass(frozen=True, slots=True)
class TostSignedRankResult:
    """The two one-sided signed-rank tests of equivalence within a margin.

    Attributes:
        p_lower: p-value of the test against the lower bound, whose null
            hypothesis is that the centre of the differences is at most
            ``-margin``.
        p_upper: p-value of the test against the upper bound, whose null
            hypothesis is that it is at least ``margin``.
        p_value: The larger of the two, the p-value of the procedure:
            equivalence is shown at level ``alpha`` when it is below
            ``alpha``.
        margin: The equivalence margin, in the units of the differences.
        pairs: Number of paired differences, including those that either
            test drops as zero.
    """

    p_lower: float
    p_upper: float
    p_value: float
    margin: float
    pairs: int


type TostResult = TostPairedTResult | TostSignedRankResult
"""The result of either equivalence procedure."""


def tost_paired_t(
    differences: Iterable[float], margin: float, alpha: float = 0.05
) -> TostPairedTResult:
    """Return the two one-sided paired t tests of equivalence within a margin.

    With ``n`` differences of mean ``m`` and standard deviation ``s``
    (denominator ``n - 1``), the standard error is ``s / sqrt(n)`` and the
    statistics are ``(m + margin) / se`` and ``(m - margin) / se``, read
    against a t distribution with ``n - 1`` degrees of freedom. The lower
    test rejects that the true mean is at most ``-margin`` for a large first
    statistic, its p-value the upper tail; the upper test rejects that it is
    at least ``margin`` for a small second one, its p-value the lower tail
    (Schuirmann 1987; Lakens 2017). The procedure rejects at level ``alpha``
    exactly when the ``(1 - 2 alpha)`` confidence interval of the mean lies
    inside ``(-margin, margin)`` (Schuirmann 1987), which is why the result
    carries the interval.

    Differences without spread, which a pair of identical solvers gives, make
    the standard error zero. The test then reads each statistic as the limit
    of a vanishing standard error: infinite with the sign of its numerator, so
    its p-value is 0 or 1, or zero when the mean sits exactly on a bound, so
    its p-value is one half. The interval is then the mean itself.

    Args:
        differences: One finite difference per matched pair, at least two.
        margin: The equivalence margin, positive and finite, in the units of
            the differences.
        alpha: The level of the procedure, strictly between 0 and 0.5. It
            sets the confidence level ``1 - 2 alpha`` of the interval and does
            not change the p-values.

    Returns:
        Both one-sided p-values, their maximum, the mean difference, and the
        interval.

    Raises:
        ConfigError: If a difference is not finite, there are fewer than two
            differences, the margin is not positive and finite, or ``alpha``
            is not strictly between 0 and 0.5.
    """
    bound = _check_margin(margin)
    level = _check_level(alpha, 0.5)
    values = _finite(differences)
    count = len(values)
    if count < 2:
        raise ConfigError("the paired t equivalence test needs at least two differences")
    mean = math.fsum(values) / count
    spread = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (count - 1))
    error = spread / math.sqrt(count)
    df = count - 1
    p_lower = _upper_tail(mean + bound, error, df)
    p_upper = _upper_tail(bound - mean, error, df)
    half = t_ppf(1.0 - level, df) * error
    return TostPairedTResult(
        p_lower=p_lower,
        p_upper=p_upper,
        p_value=max(p_lower, p_upper),
        margin=bound,
        pairs=count,
        mean_difference=mean,
        alpha=level,
        interval=(mean - half, mean + half),
    )


def tost_signed_rank(
    differences: Iterable[float], margin: float, *, zero_method: ZeroMethod = "pratt"
) -> TostSignedRankResult:
    """Return the two one-sided signed-rank tests of equivalence within a margin.

    The lower test is the one-sided Wilcoxon signed-rank test, with the
    alternative ``"greater"``, of the differences shifted by ``+margin``: it
    rejects that their centre is at most ``-margin``. The upper test is the
    one with the alternative ``"less"`` of the differences shifted by
    ``-margin``: it rejects that their centre is at least ``margin``
    (Schuirmann 1987; Lakens 2017). Both use
    :func:`~dispatchatlas.analysis.stats.nonparametric.wilcoxon_signed_rank`
    as it is, so its rules for ties and for the exact and the normal null
    distribution apply to the shifted differences, and tied magnitudes take
    averaged ranks.

    A zero of a test is a shifted difference equal to 0, that is a difference
    equal to its bound, ``-margin`` for the lower test and ``+margin`` for the
    upper one. With Pratt's treatment (Pratt 1959), the default, such a
    difference is ranked with the others and its rank then leaves the sums;
    with Wilcoxon's treatment (Wilcoxon 1945) it is dropped before ranking.

    Args:
        differences: One finite difference per matched pair, at least one.
        margin: The equivalence margin, positive and finite, in the units of
            the differences.
        zero_method: ``"pratt"`` or ``"wilcox"``, as in
            :func:`~dispatchatlas.analysis.stats.nonparametric.wilcoxon_signed_rank`.

    Returns:
        Both one-sided p-values and their maximum.

    Raises:
        ConfigError: If a difference is not finite, there is none, the margin
            is not positive and finite, or the zero method is unknown.
    """
    bound = _check_margin(margin)
    values = _finite(differences)
    if not values:
        raise ConfigError("the signed-rank equivalence test needs at least one difference")
    lower = wilcoxon_signed_rank(
        [value + bound for value in values], alternative="greater", zero_method=zero_method
    )
    upper = wilcoxon_signed_rank(
        [value - bound for value in values], alternative="less", zero_method=zero_method
    )
    return TostSignedRankResult(
        p_lower=lower.p_value,
        p_upper=upper.p_value,
        p_value=max(lower.p_value, upper.p_value),
        margin=bound,
        pairs=len(values),
    )


def equivalence_report(
    equivalence: TostResult, difference_p_value: float, alpha: float = 0.05
) -> EquivalenceOutcome:
    """Return what an equivalence test and a test of a difference show together.

    A test rejects at ``alpha`` when its p-value is below ``alpha``. The two
    tests ask different questions, so each combination of their results has
    its own reading (Lakens 2017):

    - ``"equivalent"``: the equivalence test rejects and the difference test
      does not. The difference is shown to lie within the margin, and none is
      detected.
    - ``"different"``: the difference test rejects and the equivalence test
      does not. A difference is detected and is not shown to lie within the
      margin: it may be too large to ignore, or the sample too small to say.
    - ``"equivalent-and-different"``: both reject. A difference is detected
      and also shown to lie within the margin: it is real but too small to
      matter.
    - ``"inconclusive"``: neither rejects. The sample is too small or too
      noisy to show a difference or to rule out one beyond the margin, and
      the missing difference must not be read as equivalence.

    Args:
        equivalence: The result of :func:`tost_paired_t` or
            :func:`tost_signed_rank`.
        difference_p_value: The p-value of a test of the null hypothesis that
            there is no difference, such as the ``p_value`` of
            :func:`~dispatchatlas.analysis.stats.nonparametric.wilcoxon_signed_rank`,
            two-sided, or one-sided in the direction of interest, or a p-value
            adjusted for several comparisons.
        alpha: The level of both tests, strictly between 0 and 1.

    Raises:
        ConfigError: If the p-value is not a number in ``[0, 1]`` or ``alpha``
            is not strictly between 0 and 1.
    """
    level = _check_level(alpha, 1.0)
    p_value = _check_p_value(difference_p_value)
    equivalent = equivalence.p_value < level
    different = p_value < level
    if equivalent and different:
        return "equivalent-and-different"
    if equivalent:
        return "equivalent"
    if different:
        return "different"
    return "inconclusive"


def _upper_tail(shift: float, error: float, df: int) -> float:
    """Return ``P(T > shift / error)``, reading a zero ``error`` as the limit of a small one."""
    if error > 0.0:
        return t_sf(shift / error, df)
    if shift > 0.0:
        return 0.0
    if shift < 0.0:
        return 1.0
    return 0.5


def _finite(differences: Iterable[float]) -> list[float]:
    """Return the differences as floats, refusing NaN and infinite ones."""
    values = finite_or_infinite(differences, "equivalence differences")
    if not all(math.isfinite(value) for value in values):
        raise ConfigError("equivalence differences must be finite")
    return values


def _check_margin(margin: float) -> float:
    """Return the margin as a float, refusing anything but a positive finite number."""
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ConfigError(f"the equivalence margin must be a number, got {margin!r}")
    value = float(margin)
    if not (math.isfinite(value) and value > 0.0):
        raise ConfigError(f"the equivalence margin must be positive and finite, got {margin!r}")
    return value


def _check_level(alpha: float, upper: float) -> float:
    """Return ``alpha`` as a float, refusing anything outside ``(0, upper)``."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ConfigError(f"alpha must be a number, got {alpha!r}")
    value = float(alpha)
    if not 0.0 < value < upper:
        raise ConfigError(f"alpha must be strictly between 0 and {upper:g}, got {alpha!r}")
    return value


def _check_p_value(p_value: float) -> float:
    """Return a p-value as a float, refusing anything but a number in ``[0, 1]``."""
    if isinstance(p_value, bool) or not isinstance(p_value, (int, float)):
        raise ConfigError(f"a p-value must be a number, got {p_value!r}")
    value = float(p_value)
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"a p-value must lie in [0, 1], got {p_value!r}")
    return value
