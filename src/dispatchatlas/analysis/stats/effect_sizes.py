"""Effect sizes for paired and independent comparisons.

A p-value says whether a difference is detectable; an effect size says how
large it is. Every function here is oriented so that a positive direction
favours the left (first) sample under the objective sense: an A12 above 0.5,
a positive Cliff's delta or Cohen's d, a win share above 0.5.

Two designs need different estimands:

- **Independent samples**, such as the runs of two solvers on one problem:
  Vargha and Delaney's A12, Cliff's delta, and Cohen's d_s compare every
  left value with every right value.
- **Paired samples**, such as one value per problem for each of two solvers:
  the paired win share (the share of pairs the left sample wins, ties
  counting one half) and Cohen's d_z (the mean paired difference over its
  standard deviation). The independent forms do not apply here: their cross
  pairs compare different problems, so they measure how spread out the
  problems are rather than how the samples differ.

The independent-sample effect sizes come with the magnitude labels their
sources publish. The paired ones carry no label: the published cutoffs were
calibrated for other estimands.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Literal

from dispatchatlas.analysis.stats.ranks import (
    Sense,
    check_sense,
    finite_or_infinite,
    strictly_better,
    ties,
)
from dispatchatlas.core.errors import ConfigError

__all__ = [
    "Magnitude",
    "a12_magnitude",
    "cliffs_delta",
    "cliffs_delta_magnitude",
    "cohens_d",
    "cohens_d_magnitude",
    "cohens_dz",
    "paired_outcomes",
    "paired_win_share",
    "vargha_delaney_a12",
]

type Magnitude = Literal["negligible", "small", "medium", "large"]
"""A conventional label for the size of an effect."""

# Vargha and Delaney (2000): A12 of 0.56, 0.64, and 0.71 (distances 0.06, 0.14,
# and 0.21 from 0.5) separate negligible, small, medium, and large effects.
_A12_CUTOFFS = (0.56, 0.64, 0.71)
# Romano et al. (2006): |delta| of 0.147, 0.33, and 0.474.
_CLIFF_CUTOFFS = (0.147, 0.33, 0.474)
# Cohen (1988): |d| of 0.2, 0.5, and 0.8.
_COHEN_CUTOFFS = (0.2, 0.5, 0.8)


def vargha_delaney_a12(
    left: Iterable[float], right: Iterable[float], sense: Sense = "min"
) -> float:
    """Return Vargha and Delaney's A12 for two independent samples.

    A12 is the probability that a random left value beats a random right
    value, ties counting one half (Vargha and Delaney 2000).

    Args:
        left: The first sample.
        right: The second sample.
        sense: Whether smaller or larger values are better.

    Returns:
        A12 in ``[0, 1]``, above 0.5 when the left sample tends to be better.

    Raises:
        ConfigError: If a sample is empty or holds NaN.
    """
    first, second = _samples(left, right, "Vargha-Delaney A12")
    orientation = check_sense(sense)
    score = 0.0
    for x in first:
        for y in second:
            if ties(x, y):
                score += 0.5
            elif strictly_better(x, y, orientation):
                score += 1.0
    return score / (len(first) * len(second))


def cliffs_delta(left: Iterable[float], right: Iterable[float], sense: Sense = "min") -> float:
    """Return Cliff's delta for two independent samples.

    Delta is the share of cross pairs the left sample wins minus the share it
    loses (Cliff 1993); it equals ``2 A12 - 1``.

    Returns:
        Delta in ``[-1, 1]``, positive when the left sample tends to be better.

    Raises:
        ConfigError: If a sample is empty or holds NaN.
    """
    first, second = _samples(left, right, "Cliff's delta")
    orientation = check_sense(sense)
    score = 0
    for x in first:
        for y in second:
            if strictly_better(x, y, orientation):
                score += 1
            elif strictly_better(y, x, orientation):
                score -= 1
    return score / (len(first) * len(second))


def cohens_d(left: Iterable[float], right: Iterable[float], sense: Sense = "min") -> float:
    """Return Cohen's d_s for two independent samples.

    The difference of the means divided by the pooled standard deviation
    ``sqrt((SS_left + SS_right) / (n_left + n_right - 2))`` (Cohen 1988),
    without the small-sample correction of Hedges' g.

    Returns:
        d_s, positive when the left mean is better; 0 when both samples are
        constant.

    Raises:
        ConfigError: If a sample has fewer than two values or a value is not
            finite.
    """
    first, second = _samples(left, right, "Cohen's d")
    if len(first) < 2 or len(second) < 2:
        raise ConfigError("Cohen's d needs at least two values in each sample")
    if not all(math.isfinite(value) for value in (*first, *second)):
        raise ConfigError("Cohen's d needs finite values")
    orientation = check_sense(sense)
    mean_left = math.fsum(first) / len(first)
    mean_right = math.fsum(second) / len(second)
    squares = math.fsum((value - mean_left) ** 2 for value in first) + math.fsum(
        (value - mean_right) ** 2 for value in second
    )
    pooled = math.sqrt(squares / (len(first) + len(second) - 2))
    if pooled == 0.0:
        return 0.0
    difference = (mean_left - mean_right) / pooled
    return -difference if orientation == "min" else difference


def cohens_dz(differences: Iterable[float]) -> float | None:
    """Return Cohen's d_z, the mean of paired differences over their standard deviation.

    The standard deviation uses the ``n - 1`` denominator. Orient the
    differences before calling, for example as the advantage of the left
    sample, so that a positive d_z favours it.

    Returns:
        d_z, or ``None`` when it is undefined: fewer than two differences, a
        difference that is not finite, or differences without spread.

    Raises:
        ConfigError: If a difference is NaN.
    """
    values = finite_or_infinite(differences, "paired differences")
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        return None
    mean = math.fsum(values) / len(values)
    spread = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
    if spread == 0.0:
        return None
    return mean / spread


def paired_outcomes(
    left: Sequence[float], right: Sequence[float], sense: Sense = "min"
) -> tuple[int, int, int]:
    """Count the pairs the left sample wins, loses, and ties.

    Two values tie when they are equal within
    :data:`~dispatchatlas.analysis.stats.ranks.TIE_TOLERANCE`.

    Returns:
        ``(wins, losses, ties)`` of the left sample.

    Raises:
        ConfigError: If the samples differ in length or hold NaN.
    """
    first = finite_or_infinite(left, "paired values")
    second = finite_or_infinite(right, "paired values")
    if len(first) != len(second):
        raise ConfigError("paired samples must have the same length")
    orientation = check_sense(sense)
    wins = losses = tied = 0
    for x, y in zip(first, second, strict=True):
        if ties(x, y):
            tied += 1
        elif strictly_better(x, y, orientation):
            wins += 1
        else:
            losses += 1
    return wins, losses, tied


def paired_win_share(left: Sequence[float], right: Sequence[float], sense: Sense = "min") -> float:
    """Return the share of pairs the left sample wins, ties counting one half.

    This is the sign-test counterpart of A12 for a paired design: 1 when the
    left sample wins every pair, 0.5 when wins and losses balance.

    Raises:
        ConfigError: If the samples are empty, differ in length, or hold NaN.
    """
    wins, losses, tied = paired_outcomes(left, right, sense)
    total = wins + losses + tied
    if total == 0:
        raise ConfigError("the paired win share needs at least one pair")
    return (wins + 0.5 * tied) / total


def a12_magnitude(a12: float) -> Magnitude:
    """Return the Vargha-Delaney label of an A12: cutoffs 0.56, 0.64, and 0.71.

    An A12 below 0.5 is labelled by its mirror image ``1 - A12``, which keeps
    the published cutoffs exact rather than subject to the rounding of
    ``A12 - 0.5``.
    """
    return _label(max(a12, 1.0 - a12), _A12_CUTOFFS)


def cliffs_delta_magnitude(delta: float) -> Magnitude:
    """Return the label of a Cliff's delta: cutoffs 0.147, 0.33, and 0.474 (Romano et al. 2006)."""
    return _label(abs(delta), _CLIFF_CUTOFFS)


def cohens_d_magnitude(d: float) -> Magnitude:
    """Return Cohen's label of an independent-sample d_s: cutoffs 0.2, 0.5, and 0.8."""
    return _label(abs(d), _COHEN_CUTOFFS)


def _label(distance: float, cutoffs: tuple[float, float, float]) -> Magnitude:
    if distance < cutoffs[0]:
        return "negligible"
    if distance < cutoffs[1]:
        return "small"
    if distance < cutoffs[2]:
        return "medium"
    return "large"


def _samples(
    left: Iterable[float], right: Iterable[float], what: str
) -> tuple[list[float], list[float]]:
    first = finite_or_infinite(left, f"{what} samples")
    second = finite_or_infinite(right, f"{what} samples")
    if not first or not second:
        raise ConfigError(f"{what} needs two non-empty samples")
    return first, second
