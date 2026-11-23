"""The Hodges-Lehmann shift of paired differences and its exact signed-rank interval.

The median of paired differences is not the centre that the signed-rank test tests. The
Hodges-Lehmann shift is: the shift ``delta`` at which the signed-rank statistic of the shifted
differences ``x_i - delta`` sits in the middle of its range (Hodges and Lehmann 1963). That
statistic, the sum ``W+`` of the ranks of the positive shifted differences, counts the Walsh
averages ``(x_i + x_j) / 2`` with ``i <= j`` that lie above ``delta``, and counts half of those
equal to it, leaving out under Pratt's treatment of zeros those of two differences that both equal
``delta``. The shift is therefore the median of the ``n (n + 1) / 2`` Walsh averages, and the
endpoints of the interval that inverts the tests are Walsh averages as well.

- :func:`hodges_lehmann_shift` returns the median of the Walsh averages.
- :func:`signed_rank_interval` returns it with the interval of shifts that neither one-sided
  signed-rank test rejects. This is the interval that two one-sided tests of equivalence within a
  margin read: :func:`~dispatchatlas.analysis.stats.equivalence.tost_signed_rank` finds
  equivalence when the interval of the same level lies inside the margin.

Both functions hold every Walsh average in memory, so they take memory and time in proportion to
``n^2``.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from dispatchatlas.analysis.stats.nonparametric import (
    Alternative,
    ZeroMethod,
    wilcoxon_signed_rank,
)
from dispatchatlas.analysis.stats.ranks import TIE_TOLERANCE, finite_or_infinite
from dispatchatlas.core.errors import ConfigError

__all__ = ["SignedRankInterval", "hodges_lehmann_shift", "signed_rank_interval"]


@dataclass(frozen=True, slots=True)
class SignedRankInterval:
    """The Hodges-Lehmann shift and the interval of shifts that two signed-rank tests retain.

    Attributes:
        shift: The Hodges-Lehmann shift, the median of the Walsh averages.
        low: The least shift that the one-sided test with the alternative ``"greater"`` retains,
            a Walsh average; ``-inf`` when that test cannot reject at the level
            ``(1 - confidence) / 2`` whatever the shift, as with a handful of pairs.
        high: The greatest shift that the one-sided test with the alternative ``"less"``
            retains, a Walsh average; ``inf`` when that test cannot reject. Tied differences can
            leave no shift that both tests retain, and ``low`` is then at least ``high``.
        confidence: The confidence level asked for, strictly between 0 and 1.
        coverage: The coverage the interval attains: 1 minus the p-values of the two tests at
            the nearest shifts that they reject, those just outside ``low`` and ``high``. For
            differences that do not tie, and have distinct Walsh averages, it is the exact
            coverage and at least ``confidence``; it is 0 when no shift is retained by both
            tests. See :func:`signed_rank_interval`.
        zero_method: How the tests treat a shifted difference of zero, ``"pratt"`` or
            ``"wilcox"``.
        pairs: Number of paired differences, including those that the tests drop as zero.
    """

    shift: float
    low: float
    high: float
    confidence: float
    coverage: float
    zero_method: ZeroMethod
    pairs: int


def hodges_lehmann_shift(differences: Iterable[float]) -> float:
    """Return the Hodges-Lehmann shift of paired differences, the median of their Walsh averages.

    The Walsh averages of ``x_1, ..., x_n`` are the ``n (n + 1) / 2`` means ``(x_i + x_j) / 2``
    of two differences, a difference with itself included (``i <= j``). Their median, the mean
    of the two middle ones when there is an even number of them, is the Hodges-Lehmann estimate
    of the centre (Hodges and Lehmann 1963). It is the shift at which the signed-rank sum of the
    shifted differences equals half its range, so it estimates the centre that the signed-rank
    test tests. Shifting every difference by a constant shifts the estimate by it, and negating
    them negates it.

    Args:
        differences: One finite difference per matched pair, at least one.

    Returns:
        The median of the Walsh averages.

    Raises:
        ConfigError: If a difference is not a finite number, or there is none.
    """
    values = _finite(differences)
    if not values:
        raise ConfigError("the Hodges-Lehmann shift needs at least one difference")
    return _median(_walsh_averages(values))


def signed_rank_interval(
    differences: Iterable[float],
    confidence: float = 0.90,
    *,
    zero_method: ZeroMethod = "pratt",
) -> SignedRankInterval:
    """Return the Hodges-Lehmann shift and the interval that inverts two signed-rank tests.

    A one-sided test retains a shift ``delta`` when its p-value for the differences shifted by
    ``delta``, ``x_i - delta``, is at least ``alpha = (1 - confidence) / 2``, and rejects it when
    the p-value is below ``alpha``. The test with the alternative ``"greater"`` rejects that the
    centre is at most ``delta``, and the one with ``"less"`` that it is at least ``delta``. Both
    are :func:`~dispatchatlas.analysis.stats.nonparametric.wilcoxon_signed_rank` as it is, with
    its exact and its normal null distribution, its tie rule, and ``zero_method``, so the interval
    inverts the very tests that
    :func:`~dispatchatlas.analysis.stats.equivalence.tost_signed_rank` runs. ``low`` is the least
    shift that the first test retains and ``high`` the greatest that the second retains: the
    bounds of the two one-sided confidence sets (Hodges and Lehmann 1963).

    **Endpoints.** The signed-rank sum ``W+`` of the shifted differences counts the Walsh
    averages ``(x_i + x_j) / 2``, ``i <= j``, above ``delta`` and half of those equal to it, not
    counting under Pratt's treatment those of two differences that both equal ``delta``. So it
    changes only where a shifted difference crosses zero (``i = j``) or two shifted magnitudes
    tie (``i < j``): at a Walsh average. Between two consecutive Walsh averages the ranks, and
    so both p-values, are constant. The endpoints are therefore Walsh averages. At a shift equal
    to one with ``i < j``, two shifted differences have equal magnitude and opposite sign and
    take the midrank, so the test reads the null distribution given the ranks that the tie
    leaves. At a shift equal to a difference, a shifted difference is a zero: Pratt's treatment
    (Pratt 1959) ranks it with the others, below them, and then drops its rank from the sums,
    and Wilcoxon's treatment drops it before ranking, which also removes a pair from the null
    distribution. A shift at a Walsh average is retained or rejected by the p-value of its own
    ranks, so a retained set can be open or closed at an endpoint, and ``low`` and ``high`` are
    its infimum and supremum.

    **Search.** The Walsh averages and the gaps between them are the ``O(n^2)`` shifts at which
    the p-values can differ. Under Pratt's treatment the p-value of the test with the alternative
    ``"greater"`` does not fall, and that of the test with ``"less"`` does not rise, from one of
    them to the next higher one, so a bisection over them finds where each test begins or stops
    to retain, in about ``2 log2(n (n + 1))`` evaluations of the test. Under Wilcoxon's treatment
    a shift equal to a difference breaks that order: the bisection runs over the other shifts,
    each shift equal to a difference is evaluated one by one, and the bounds are the extreme
    shifts that the tests retain, whatever lies between them.

    **Equivalence.** Under Pratt's treatment, with or without ties and zeros, the two one-sided
    tests of ``tost_signed_rank`` at level ``alpha`` find equivalence within ``margin`` exactly
    when ``-margin < low`` and ``high < margin``, that is when this interval of confidence
    ``1 - 2 alpha`` lies inside ``(-margin, margin)``. The one exception is a margin equal to an
    endpoint, which the p-value of the test at that Walsh average decides and the interval does
    not record. The statement holds as well when the interval is unbounded, or when no shift is
    retained by both tests. Wilcoxon's treatment can break it at a shift that equals a
    difference.

    **Coverage.** The coverage is 1 minus the p-values of the two tests in the gaps just below
    ``low`` and just above ``high``, the nearest shifts that they reject. When no two differences
    tie, the null distribution of ``W+`` is the same at every shift between Walsh averages. With
    ``r`` the largest integer for which ``P(W+ <= r) < alpha``, ``low`` and ``high`` are then the
    ``(r + 1)``-th smallest and largest Walsh averages. If no two Walsh averages are equal, the
    two p-values are ``P(W+ <= r)`` and the coverage is ``1 - 2 P(W+ <= r)``, at least
    ``confidence`` because the distribution is discrete. Equal Walsh averages, which rounded
    differences produce, put several of them on one side of a gap and lower the p-values, so they
    raise the coverage. With tied differences, or with zeros that Pratt's treatment ranks, the
    null given the ranks depends on the shift, and the coverage is conditional on the ranks in
    those two gaps; it is 0 when tied differences leave no shift that both tests retain. Above
    :data:`~dispatchatlas.analysis.stats.nonparametric.WILCOXON_EXACT_LIMIT` non-zero pairs the
    p-values are the normal approximation's. When even the most extreme sample cannot give a
    p-value below ``alpha``, the test never rejects, that side of the interval is unbounded
    (``-inf`` or ``inf``), and nothing is lost there.

    Args:
        differences: One finite difference per matched pair, at least one.
        confidence: The confidence level, strictly between 0 and 1. The ``(1 - 2 alpha)``
            interval of two one-sided tests at level ``alpha`` has confidence ``1 - 2 alpha``.
        zero_method: ``"pratt"`` or ``"wilcox"``, as in
            :func:`~dispatchatlas.analysis.stats.nonparametric.wilcoxon_signed_rank`.

    Returns:
        The Hodges-Lehmann shift, the two endpoints, the coverage they attain, and the arguments.

    Raises:
        ConfigError: If a difference is not a finite number, there is none, ``confidence`` is not
            strictly between 0 and 1, or the zero method is unknown.
    """
    method = _check_zero_method(zero_method)
    level = _check_confidence(confidence)
    values = _finite(differences)
    if not values:
        raise ConfigError("the signed-rank interval needs at least one difference")
    walsh = _walsh_averages(values)
    inversion = _Inversion(values, list(dict.fromkeys(walsh)), (1.0 - level) / 2.0, method)
    first = inversion.first_retained()
    last = inversion.last_retained()
    low, attained_low = inversion.lower_bound(first)
    high, attained_high = inversion.upper_bound(last)
    return SignedRankInterval(
        shift=_median(walsh),
        low=low,
        high=high,
        confidence=level,
        # No shift is retained by both tests when the first one that the lower test retains
        # lies above the last one that the upper test retains.
        coverage=1.0 - attained_low - attained_high if first <= last else 0.0,
        zero_method=method,
        pairs=len(values),
    )


class _Inversion:
    """The one-sided signed-rank tests of the differences over the shifts that can differ.

    Those are the distinct Walsh averages ``w_0 < ... < w_(m-1)`` and the gaps between them,
    numbered as states in the order of the shift: state ``2 k + 1`` is ``w_k``, state ``2 k`` a
    shift in the gap below ``w_k``, and state ``2 m`` one above ``w_(m-1)``. A p-value is cached
    once it is computed, since the bisection and the coverage read some twice.
    """

    __slots__ = (
        "alpha",
        "cache",
        "candidates",
        "margin",
        "ordered",
        "regular",
        "values",
        "zero_method",
        "zero_states",
    )

    def __init__(
        self, values: list[float], candidates: list[float], alpha: float, zero_method: ZeroMethod
    ) -> None:
        self.values = values
        self.ordered = sorted(values)
        self.candidates = candidates
        self.alpha = alpha
        self.zero_method = zero_method
        self.cache: dict[tuple[int, Alternative], float] = {}
        # Far enough beyond the extreme differences for each shifted one to keep its sign.
        self.margin = 1.0 + candidates[-1] - candidates[0]
        states = range(2 * len(candidates) + 1)
        # Wilcoxon's treatment drops a difference that the shift equals, so the p-value at such
        # a state is out of line with its neighbours: those states are evaluated one by one.
        self.zero_states: list[int] = []
        if zero_method == "wilcox":
            self.zero_states = [state for state in states if self._on_difference(state)]
        skipped = set(self.zero_states)
        self.regular: Sequence[int] = (
            [state for state in states if state not in skipped] if skipped else states
        )

    def lower_bound(self, first: int) -> tuple[float, float]:
        """Return the Walsh average below the first retained state, and the p-value just under it.

        The state is the first that the test ``"greater"`` retains, from :meth:`first_retained`.
        The Walsh average is the endpoint, and the p-value is that of the same test in the gap
        below it, the nearest shift that it surely rejects.
        """
        if first == 0:
            return -math.inf, 0.0
        index = (first - 1) // 2
        return self.candidates[index], self._p_value(2 * index, "greater")

    def upper_bound(self, last: int) -> tuple[float, float]:
        """Return the Walsh average above the last retained state, and the p-value just over it.

        The state is the last that the test ``"less"`` retains, from :meth:`last_retained`.
        """
        index = last // 2
        if index >= len(self.candidates):
            return math.inf, 0.0
        return self.candidates[index], self._p_value(2 * index + 2, "less")

    def first_retained(self) -> int:
        """Return the first state that the test with the alternative ``"greater"`` retains."""
        regular = self.regular
        low, high = 0, len(regular) - 1  # the state above every shift is always retained
        while low < high:
            middle = (low + high) // 2
            if self._retained(regular[middle], "greater"):
                high = middle
            else:
                low = middle + 1
        first = regular[low]
        for state in self.zero_states:
            if state >= first:
                break
            if self._retained(state, "greater"):
                return state
        return first

    def last_retained(self) -> int:
        """Return the last state that the test with the alternative ``"less"`` retains."""
        regular = self.regular
        low, high = 0, len(regular) - 1  # the state below every shift is always retained
        while low < high:
            middle = (low + high + 1) // 2
            if self._retained(regular[middle], "less"):
                low = middle
            else:
                high = middle - 1
        last = regular[low]
        for state in reversed(self.zero_states):
            if state <= last:
                break
            if self._retained(state, "less"):
                return state
        return last

    def _shift(self, state: int) -> float:
        """Return the shift of a state."""
        index, walsh = divmod(state, 2)
        if walsh:
            return self.candidates[index]
        if index == 0:
            return self.candidates[0] - self.margin
        if index == len(self.candidates):
            return self.candidates[-1] + self.margin
        return 0.5 * self.candidates[index - 1] + 0.5 * self.candidates[index]

    def _p_value(self, state: int, alternative: Alternative) -> float:
        """Return the p-value of the one-sided test of the differences shifted to a state."""
        key = (state, alternative)
        if key not in self.cache:
            delta = self._shift(state)
            shifted = [value - delta for value in self.values]
            result = wilcoxon_signed_rank(
                shifted, alternative=alternative, zero_method=self.zero_method
            )
            self.cache[key] = result.p_value
        return self.cache[key]

    def _retained(self, state: int, alternative: Alternative) -> bool:
        """Return whether the one-sided test does not reject the shift of a state."""
        return self._p_value(state, alternative) >= self.alpha

    def _on_difference(self, state: int) -> bool:
        """Return whether a state shifts a difference to a zero, as the test reads zeros."""
        if state % 2 == 0:
            return False
        delta = self._shift(state)
        position = bisect_left(self.ordered, delta)
        return any(
            0 <= index < len(self.ordered) and abs(self.ordered[index] - delta) <= TIE_TOLERANCE
            for index in (position - 1, position)
        )


def _walsh_averages(values: Sequence[float]) -> list[float]:
    """Return the ascending Walsh averages ``(x_i + x_j) / 2`` for ``i <= j``.

    Each value is halved before the sum, which gives the float that halving the sum gives, cannot
    overflow, and makes the average of a value with itself that value exactly.
    """
    halves = [0.5 * value for value in values]
    return sorted(
        halves[first] + halves[second]
        for first in range(len(halves))
        for second in range(first, len(halves))
    )


def _median(ordered: Sequence[float]) -> float:
    """Return the median of ascending values: the middle one, or the mean of the middle two."""
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * ordered[middle - 1] + 0.5 * ordered[middle]


def _finite(differences: Iterable[float]) -> list[float]:
    """Return the differences as floats, refusing NaN and infinite ones."""
    values = finite_or_infinite(differences, "Hodges-Lehmann differences")
    if not all(math.isfinite(value) for value in values):
        raise ConfigError("Hodges-Lehmann differences must be finite")
    return values


def _check_confidence(confidence: float) -> float:
    """Return the confidence level as a float, refusing anything outside ``(0, 1)``."""
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ConfigError(f"the confidence must be a number, got {confidence!r}")
    value = float(confidence)
    if not 0.0 < value < 1.0:
        raise ConfigError(f"the confidence must be strictly between 0 and 1, got {confidence!r}")
    return value


def _check_zero_method(zero_method: object) -> ZeroMethod:
    """Return ``zero_method`` if it is ``"wilcox"`` or ``"pratt"``."""
    if zero_method == "pratt":
        return "pratt"
    if zero_method == "wilcox":
        return "wilcox"
    raise ConfigError(f"zero_method must be 'wilcox' or 'pratt', got {zero_method!r}")
