"""Rank tests: Wilcoxon, sign, Mann-Whitney, Friedman, Iman-Davenport, and Nemenyi.

The tests follow the non-parametric protocol for comparing algorithms over
several problems (Demšar 2006; Derrac et al. 2011): a paired test for two
algorithms, and for several algorithms an omnibus test over complete blocks
followed by a post-hoc comparison of average ranks.

All tests are two-sided, except that the signed-rank test also takes a
one-sided alternative, whose p-value reads one tail of the same null
distribution as the two-sided one. Small samples use the exact null
distribution: the signed-rank test with or without tied magnitudes, where ties,
and zeros that Pratt's treatment ranks, make it the distribution conditional on
the ranks, and the Mann-Whitney test without ties. Larger samples, and tied
Mann-Whitney samples, use the normal approximation with the tie-corrected
variance and a continuity correction of 0.5, whose tail comes from
:func:`~dispatchatlas.analysis.stats.distributions.normal_sf` so that very
small p-values keep their precision. Rank ties follow
:data:`~dispatchatlas.analysis.stats.ranks.TIE_TOLERANCE`, and so do the
variance corrections: :func:`tie_sum` counts the groups that the midranks
share.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from dispatchatlas.analysis.stats.distributions import chi2_sf, f_sf, normal_sf
from dispatchatlas.analysis.stats.ranks import (
    TIE_TOLERANCE,
    Sense,
    check_sense,
    finite_or_infinite,
    midranks,
    oriented_ranks,
)
from dispatchatlas.core.errors import ConfigError

__all__ = [
    "MANN_WHITNEY_EXACT_LIMIT",
    "WILCOXON_EXACT_LIMIT",
    "Alternative",
    "FriedmanResult",
    "ImanDavenportResult",
    "MannWhitneyResult",
    "TestMethod",
    "WilcoxonResult",
    "ZeroMethod",
    "friedman_statistic",
    "friedman_test",
    "iman_davenport",
    "mann_whitney_u",
    "nemenyi_critical_difference",
    "nemenyi_q",
    "sign_test",
    "tie_sum",
    "wilcoxon_signed_rank",
]

type TestMethod = Literal["exact", "exact-conditional", "normal", "degenerate"]
"""How a p-value was obtained.

``"exact"`` is the exact null distribution of untied ranks,
``"exact-conditional"`` the exact null distribution of the signed-rank
statistic given the ranks of the non-zero magnitudes when ties, or zeros that
Pratt's treatment ranks, make them differ from 1 to ``n``, ``"normal"`` the
normal approximation, and ``"degenerate"`` a test without data.
"""

type Alternative = Literal["two-sided", "greater", "less"]
"""The alternative hypothesis of a test of paired differences.

``"two-sided"`` is that the differences tend to differ from zero in either
direction, ``"greater"`` that they tend to be positive, and ``"less"`` that
they tend to be negative.
"""

type ZeroMethod = Literal["wilcox", "pratt"]
"""How the signed-rank test treats differences of zero.

``"wilcox"`` drops them before ranking, so the other magnitudes are ranked from
1 (Wilcoxon 1945). ``"pratt"`` ranks the magnitudes with the zeros included,
which gives the zeros the lowest ranks and moves every other magnitude up by
their number, and then drops the zeros' ranks from the sums (Pratt 1959).
"""

WILCOXON_EXACT_LIMIT = 50
"""Largest number of non-zero pairs for which the exact Wilcoxon null is used.

The subset-sum recursion costs about ``n * n (n + 1)`` additions, 130 thousand
at the limit, and the counts of sign assignments stay below ``2^53``, so their
conversion to floats is exact.
"""

MANN_WHITNEY_EXACT_LIMIT = 20
"""Largest size of either sample for which the exact Mann-Whitney null is used."""

_CONTINUITY = 0.5

# Nemenyi critical values q_alpha: the infinite-degrees-of-freedom studentized
# range quantile divided by sqrt(2), from Demšar (2006), Table 5(a), extended to
# 11 and 12 algorithms by the same construction.
_NEMENYI_Q: Mapping[float, Mapping[int, float]] = {
    0.05: {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
    },
    0.10: {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.459,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
        11: 2.978,
        12: 3.030,
    },
}


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    """A Wilcoxon signed-rank test.

    Attributes:
        statistic: For the two-sided alternative, ``min(W+, W-)``, the smaller
            signed-rank sum; for a one-sided alternative, ``W+``, the sum of
            the ranks of the positive differences.
        p_value: p-value for ``alternative``.
        pairs: Number of non-zero differences the statistic sums.
        zeros: Number of differences of zero, which ``zero_method`` drops
            before ranking or from the sums.
        method: How the p-value was obtained.
        alternative: The alternative hypothesis the p-value tests.
        zero_method: How the zero differences were treated.
    """

    statistic: float
    p_value: float
    pairs: int
    zeros: int
    method: TestMethod
    alternative: Alternative = "two-sided"
    zero_method: ZeroMethod = "wilcox"


@dataclass(frozen=True, slots=True)
class MannWhitneyResult:
    """A two-sided Mann-Whitney U test.

    Attributes:
        statistic: ``min(U1, U2)``.
        p_value: Two-sided p-value.
        first_size: Size of the first sample.
        second_size: Size of the second sample.
        method: How the p-value was obtained.
    """

    statistic: float
    p_value: float
    first_size: int
    second_size: int
    method: TestMethod


@dataclass(frozen=True, slots=True)
class FriedmanResult:
    """A Friedman test over complete blocks.

    Attributes:
        statistic: The tie-corrected chi-square statistic ``Q``.
        p_value: Upper-tail chi-square probability of ``Q``.
        df: Degrees of freedom, the number of groups minus one.
        blocks: Number of blocks ``N``, such as problems.
        groups: Number of groups ``k``, such as solvers.
        average_ranks: Each group's mean rank over the blocks, rank 1 best, in
            the order of the first block's keys.
    """

    statistic: float
    p_value: float
    df: int
    blocks: int
    groups: int
    average_ranks: tuple[tuple[str, float], ...]

    @property
    def kendalls_w(self) -> float:
        """Kendall's coefficient of concordance ``W = Q / (N (k - 1))``, in ``[0, 1]``.

        It is the effect size of the Friedman test: 0 when the rankings
        disagree completely, 1 when every block ranks the groups alike
        (Kendall and Babington Smith 1939).
        """
        return self.statistic / (self.blocks * (self.groups - 1))


@dataclass(frozen=True, slots=True)
class ImanDavenportResult:
    """The Iman-Davenport F statistic derived from a Friedman result.

    Attributes:
        statistic: ``F = (N - 1) Q / (N (k - 1) - Q)``, infinite when every
            block ranks the groups alike.
        p_value: Upper-tail F probability of the statistic.
        df_numerator: ``k - 1``.
        df_denominator: ``(k - 1)(N - 1)``.
    """

    statistic: float
    p_value: float
    df_numerator: int
    df_denominator: int


def wilcoxon_signed_rank(
    differences: Iterable[float],
    *,
    alternative: Alternative = "two-sided",
    zero_method: ZeroMethod = "wilcox",
) -> WilcoxonResult:
    """Return the Wilcoxon signed-rank test of paired differences.

    Differences within :data:`TIE_TOLERANCE` of zero are zeros, which
    ``zero_method`` treats as Wilcoxon (1945) or Pratt (1959) does. Both
    midrank the magnitudes of the non-zero differences. Wilcoxon's treatment
    drops the zeros first, so the ranks run from 1. Pratt's ranks the
    magnitudes with the zeros included, so the zeros hold the lowest ranks and
    the non-zero differences rank above them, and only then drops the zeros'
    ranks from the sums. The two-sided test uses the smaller of the rank sums
    ``W+`` and ``W-`` of the positive and the negative differences. A
    one-sided test uses ``W+``: the alternative ``"greater"``, that the
    differences tend to be positive, rejects for a large ``W+``, and
    ``"less"``, that they tend to be negative, for a small one.

    Up to :data:`WILCOXON_EXACT_LIMIT` non-zero pairs, the p-value comes from
    the exact null distribution of ``W+`` given those ranks: under the null
    hypothesis each difference is positive or negative with equal
    probability, so every assignment of signs to the ranks is equally likely,
    and a subset-sum recursion counts the assignments by their rank sum. A
    one-sided p-value is the probability of a sum at least as extreme as the
    observed one in the direction of the alternative (Conover 1999), and the
    two-sided p-value is twice the smaller of the two tails, capped at 1.
    Without ties, and without zeros that Pratt's treatment ranks, the ranks
    are 1 to ``n`` and the distribution does not depend on the data (method
    ``"exact"``). With tied magnitudes the ranks are midranks, and with Pratt's
    treatment of zeros they start above the zeros; the distribution is then the
    one conditional on them (method ``"exact-conditional"``). Its smallest
    two-sided p-value over ``n`` non-zero pairs is ``2 / 2^n`` as without ties,
    and the smallest one-sided one ``1 / 2^n``; a normal approximation can fall
    below that on a small tied sample. The recursion counts sums up to twice
    the total of the ranks, so Pratt's treatment, which lifts every rank by the
    number of zeros, lengthens it in proportion.

    Above the limit the normal approximation uses the mean ``sum(r) / 2`` and
    the variance ``sum(r^2) / 4`` of the signed-rank sum under that null, for
    the ranks ``r`` of the non-zero differences, and a continuity correction
    of 0.5 towards the null (Wilcoxon 1945). Without zeros that Pratt's
    treatment ranks, they are ``n(n+1)/4`` and
    ``n(n+1)(2n+1)/24 - sum(t^3 - t)/48`` over the tie groups. Infinite
    differences are allowed and rank above every finite one.

    Args:
        differences: One difference per matched pair. Orient them so that a
            positive difference supports the alternative ``"greater"``, for
            example the value of the first sample minus that of the second.
        alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
        zero_method: ``"wilcox"`` drops the zeros before ranking, and
            ``"pratt"`` ranks them and drops their ranks from the sums.

    Returns:
        The test, whose ``method`` names how the p-value was obtained and
        whose ``alternative`` and ``zero_method`` repeat the arguments; with
        no non-zero difference, a degenerate result with p-value 1.

    Raises:
        ConfigError: If there are no differences, one is NaN, or the
            alternative or the zero method is not one of the two documented.
    """
    alternative = _check_alternative(alternative)
    zero_method = _check_zero_method(zero_method)
    values = finite_or_infinite(differences, "Wilcoxon differences")
    if not values:
        raise ConfigError("the Wilcoxon signed-rank test needs at least one difference")
    nonzero = [value for value in values if abs(value) > TIE_TOLERANCE]
    zeros = len(values) - len(nonzero)
    if not nonzero:
        return WilcoxonResult(0.0, 1.0, 0, zeros, "degenerate", alternative, zero_method)
    magnitudes = [abs(value) for value in nonzero]
    ranks = midranks(magnitudes)
    # Pratt's zeros hold the ranks 1 to ``zeros``, so the others start above them.
    lift = zeros if zero_method == "pratt" else 0
    if lift:
        ranks = tuple(rank + lift for rank in ranks)
    positive = math.fsum(rank for rank, value in zip(ranks, nonzero, strict=True) if value > 0.0)
    negative = math.fsum(rank for rank, value in zip(ranks, nonzero, strict=True) if value < 0.0)
    two_sided = alternative == "two-sided"
    statistic = min(positive, negative) if two_sided else positive
    count = len(nonzero)
    if count <= WILCOXON_EXACT_LIMIT:
        untied = lift == 0 and len(set(ranks)) == count
        method: TestMethod = "exact" if untied else "exact-conditional"
        p_value = (
            _wilcoxon_exact(statistic, ranks)
            if two_sided
            else _wilcoxon_exact_tail(positive, ranks, alternative)
        )
        return WilcoxonResult(statistic, p_value, count, zeros, method, alternative, zero_method)
    if lift:
        mean = math.fsum(ranks) / 2.0
        variance = math.fsum(rank * rank for rank in ranks) / 4.0
    else:
        mean = count * (count + 1) / 4.0
        variance = count * (count + 1) * (2 * count + 1) / 24.0 - _rank_tie_sum(ranks) / 48.0
    p_value = (
        _normal_two_sided(statistic, mean, variance)
        if two_sided
        else _normal_tail(positive, mean, variance, alternative)
    )
    return WilcoxonResult(statistic, p_value, count, zeros, "normal", alternative, zero_method)


def _check_alternative(alternative: object) -> Alternative:
    """Return ``alternative`` if it is ``"two-sided"``, ``"greater"``, or ``"less"``.

    Raises:
        ConfigError: For any other value.
    """
    if alternative == "two-sided":
        return "two-sided"
    if alternative == "greater":
        return "greater"
    if alternative == "less":
        return "less"
    raise ConfigError(f"alternative must be 'two-sided', 'greater', or 'less', got {alternative!r}")


def _check_zero_method(zero_method: object) -> ZeroMethod:
    """Return ``zero_method`` if it is ``"wilcox"`` or ``"pratt"``.

    Raises:
        ConfigError: For any other value.
    """
    if zero_method == "wilcox":
        return "wilcox"
    if zero_method == "pratt":
        return "pratt"
    raise ConfigError(f"zero_method must be 'wilcox' or 'pratt', got {zero_method!r}")


def _sign_assignment_counts(ranks: Sequence[float]) -> list[int]:
    """Return how many sign assignments of the ranks give each doubled positive rank sum.

    Every assignment of signs to the ranks is equally likely under the null
    hypothesis. Midranks are multiples of one half, so the recursion counts the
    assignments by twice their positive rank sum, an integer.
    """
    doubled = [round(2.0 * rank) for rank in ranks]
    counts = [0] * (sum(doubled) + 1)
    counts[0] = 1
    for rank in doubled:
        for total in range(len(counts) - 1, rank - 1, -1):
            counts[total] += counts[total - rank]
    return counts


def _wilcoxon_exact(statistic: float, ranks: Sequence[float]) -> float:
    """Return the exact two-sided p-value of ``min(W+, W-)`` given the ranks of the pairs.

    The distribution of the positive rank sum is symmetric about half the
    total, so twice its lower tail at the statistic is the probability that
    the smaller of the two sums is at most the statistic.
    """
    counts = _sign_assignment_counts(ranks)
    lower = sum(counts[: math.floor(2.0 * statistic + 1e-9) + 1])
    return min(1.0, 2.0 * lower / (1 << len(ranks)))


def _wilcoxon_exact_tail(
    positive: float, ranks: Sequence[float], alternative: Alternative
) -> float:
    """Return the exact one-sided p-value of ``W+`` given the ranks of the pairs.

    ``"greater"`` takes the upper tail of the distribution of the positive
    rank sum from the observed value, ``"less"`` the lower tail up to it, the
    observed value itself counting in both.
    """
    counts = _sign_assignment_counts(ranks)
    observed = round(2.0 * positive)
    tail = sum(counts[observed:]) if alternative == "greater" else sum(counts[: observed + 1])
    return tail / (1 << len(ranks))


def sign_test(wins: int, losses: int) -> float:
    """Return the exact two-sided sign test p-value.

    Under the null hypothesis each untied pair is a win with probability
    1/2, so the p-value is twice the binomial tail of the smaller count,
    capped at 1 (Dixon and Mood 1946). Ties carry no direction and are left
    out by the caller.

    Args:
        wins: Pairs the first sample won.
        losses: Pairs the first sample lost.

    Raises:
        ConfigError: If a count is negative or not an integer.
    """
    for name, value in (("wins", wins), ("losses", losses)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"sign test {name} must be a non-negative integer, got {value!r}")
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / (1 << trials))


def mann_whitney_u(first: Iterable[float], second: Iterable[float]) -> MannWhitneyResult:
    """Return the two-sided Mann-Whitney U test of two independent samples.

    The pooled sample is midranked and ``U1`` is the first sample's rank sum
    minus ``n1(n1+1)/2``; the statistic is ``min(U1, n1 n2 - U1)``. When
    neither sample exceeds :data:`MANN_WHITNEY_EXACT_LIMIT` and no values tie,
    the p-value comes from the exact null, which the recursion of Mann and
    Whitney (1947) counts. Otherwise the normal approximation uses the mean
    ``n1 n2 / 2``, the variance
    ``n1 n2 / 12 * ((N + 1) - sum(t^3 - t) / (N (N - 1)))``, and a continuity
    correction of 0.5.

    Args:
        first: The first sample.
        second: The second sample.

    Raises:
        ConfigError: If a sample is empty or holds NaN.
    """
    left = finite_or_infinite(first, "Mann-Whitney samples")
    right = finite_or_infinite(second, "Mann-Whitney samples")
    if not left or not right:
        raise ConfigError("the Mann-Whitney U test needs two non-empty samples")
    n1, n2 = len(left), len(right)
    pooled = left + right
    ranks = midranks(pooled)
    u1 = sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0
    statistic = min(u1, n1 * n2 - u1)
    tied = len(set(ranks)) != len(ranks)
    if n1 <= MANN_WHITNEY_EXACT_LIMIT and n2 <= MANN_WHITNEY_EXACT_LIMIT and not tied:
        return MannWhitneyResult(statistic, _mann_whitney_exact(statistic, n1, n2), n1, n2, "exact")
    total = n1 + n2
    variance = (n1 * n2 / 12.0) * ((total + 1) - _rank_tie_sum(ranks) / (total * (total - 1)))
    p_value = _normal_two_sided(statistic, n1 * n2 / 2.0, variance)
    return MannWhitneyResult(statistic, p_value, n1, n2, "normal")


def _mann_whitney_exact(statistic: float, n1: int, n2: int) -> float:
    """Return the exact two-sided p-value of ``min(U1, U2)`` from the counting recursion.

    ``w(i, j, u) = w(i - 1, j, u - j) + w(i, j - 1, u)`` counts the
    arrangements of ``i`` and ``j`` observations with statistic ``u``.
    """
    previous: list[list[int]] = [[1] for _ in range(n2 + 1)]
    for i in range(1, n1 + 1):
        current: list[list[int]] = [[1]]
        for j in range(1, n2 + 1):
            left = previous[j]
            down = current[j - 1]
            row = [0] * (i * j + 1)
            for u in range(len(row)):
                total = down[u] if u < len(down) else 0
                if 0 <= u - j < len(left):
                    total += left[u - j]
                row[u] = total
            current.append(row)
        previous = current
    distribution = previous[n2]
    lower = sum(distribution[: math.floor(statistic + 1e-9) + 1])
    return min(1.0, 2.0 * lower / sum(distribution))


def friedman_test(blocks: Sequence[Mapping[str, float]], sense: Sense = "min") -> FriedmanResult:
    """Return the Friedman test of several groups over complete blocks.

    Each block, such as one problem, maps every group, such as a solver, to
    its value. Values are midranked within each block with rank 1 for the
    best, and the statistic
    ``Q = [12 / (N k (k+1)) sum_j R_j^2 - 3 N (k+1)] / [1 - sum(t^3 - t) / (N k (k^2 - 1))]``
    divides the rank-sum dispersion by the tie correction; it is read against
    a chi-square distribution with ``k - 1`` degrees of freedom (Friedman
    1937). Infinite values are allowed, for example for failed runs.

    Args:
        blocks: One mapping per block, all with the same keys.
        sense: Whether smaller or larger values are better.

    Raises:
        ConfigError: With fewer than two blocks or two groups, blocks that
            disagree on their keys, or a NaN value.
    """
    orientation = check_sense(sense)
    if len(blocks) < 2:
        raise ConfigError("the Friedman test needs at least two blocks")
    names = tuple(blocks[0])
    if len(names) < 2:
        raise ConfigError("the Friedman test needs at least two groups")
    totals = dict.fromkeys(names, 0.0)
    ties_total = 0.0
    for block in blocks:
        if set(block) != set(names):
            raise ConfigError("every Friedman block must hold the same groups")
        values = finite_or_infinite([block[name] for name in names], "Friedman values")
        ranks = oriented_ranks(values, orientation)
        for name, rank in zip(names, ranks, strict=True):
            totals[name] += rank
        ties_total += _rank_tie_sum(ranks)
    count, groups = len(blocks), len(names)
    statistic = friedman_statistic(list(totals.values()), count, ties_total)
    return FriedmanResult(
        statistic=statistic,
        p_value=chi2_sf(statistic, groups - 1),
        df=groups - 1,
        blocks=count,
        groups=groups,
        average_ranks=tuple((name, totals[name] / count) for name in names),
    )


def friedman_statistic(rank_sums: Sequence[float], blocks: int, ties: float = 0.0) -> float:
    """Return the tie-corrected Friedman statistic from the rank sums of each group.

    Args:
        rank_sums: The sum over blocks of each group's rank.
        blocks: The number of blocks ``N``.
        ties: ``sum(t^3 - t)`` over the tie groups of every block, see
            :func:`tie_sum`.

    Returns:
        ``Q``. When the tie correction is not positive, which happens only
        when every block ties every group, the uncorrected statistic.
    """
    groups = len(rank_sums)
    uncorrected = 12.0 / (blocks * groups * (groups + 1)) * sum(
        total**2 for total in rank_sums
    ) - 3.0 * blocks * (groups + 1)
    correction = 1.0 - ties / (blocks * groups * (groups**2 - 1))
    return uncorrected / correction if correction > 0.0 else uncorrected


def iman_davenport(friedman: FriedmanResult) -> ImanDavenportResult:
    """Return the Iman-Davenport F statistic of a Friedman result.

    ``F = (N - 1) Q / (N (k - 1) - Q)`` with ``k - 1`` and ``(k - 1)(N - 1)``
    degrees of freedom is less conservative than the chi-square reading of
    ``Q`` (Iman and Davenport 1980; Demšar 2006). When every block ranks the
    groups alike the denominator vanishes, and the statistic is infinite with
    p-value 0.
    """
    blocks, groups, q = friedman.blocks, friedman.groups, friedman.statistic
    df_numerator = groups - 1
    df_denominator = (groups - 1) * (blocks - 1)
    denominator = blocks * (groups - 1) - q
    if denominator <= 0.0:
        return ImanDavenportResult(math.inf, 0.0, df_numerator, df_denominator)
    statistic = (blocks - 1) * q / denominator
    return ImanDavenportResult(
        statistic, f_sf(statistic, df_numerator, df_denominator), df_numerator, df_denominator
    )


def nemenyi_q(groups: int, alpha: float = 0.05) -> float:
    """Return the Nemenyi critical value ``q_alpha`` for ``groups`` groups.

    Tabulated for 2 to 12 groups at ``alpha`` 0.05 and 0.10 (Demšar 2006,
    Table 5(a)).

    Raises:
        ConfigError: For another number of groups or another ``alpha``.
    """
    for level, table in _NEMENYI_Q.items():
        if math.isclose(alpha, level, rel_tol=1e-9, abs_tol=1e-12):
            if groups not in table:
                raise ConfigError(
                    f"Nemenyi critical values are tabulated for 2 to 12 groups, got {groups}"
                )
            return table[groups]
    raise ConfigError(f"Nemenyi critical values are tabulated for alpha 0.05 and 0.10, got {alpha}")


def nemenyi_critical_difference(groups: int, blocks: int, alpha: float = 0.05) -> float:
    """Return the Nemenyi critical difference of average ranks.

    Two groups differ significantly when their average ranks over the
    ``blocks`` differ by more than ``q_alpha sqrt(k (k + 1) / (6 N))``
    (Nemenyi 1963; Demšar 2006).

    Raises:
        ConfigError: With fewer than one block, or when :func:`nemenyi_q`
            has no critical value.
    """
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
        raise ConfigError(f"the Nemenyi test needs at least one block, got {blocks!r}")
    return nemenyi_q(groups, alpha) * math.sqrt(groups * (groups + 1) / (6.0 * blocks))


def tie_sum(values: Sequence[float], sense: Sense = "min") -> float:
    """Return ``sum(t^3 - t)`` over the groups ``t`` of tied values.

    This is the tie term of the Wilcoxon, Mann-Whitney, and Friedman
    variance corrections. Values tie when they are equal within
    :data:`~dispatchatlas.analysis.stats.ranks.TIE_TOLERANCE`: the groups are
    the ones that share a midrank in
    :func:`~dispatchatlas.analysis.stats.ranks.oriented_ranks`, so the term
    always agrees with the ranks the tests use.

    Args:
        values: The values of one sample or block.
        sense: Whether smaller or larger values are better, the orientation in
            which the ranks are formed.

    Raises:
        ConfigError: If ``sense`` is neither ``"min"`` nor ``"max"``.
    """
    return _rank_tie_sum(oriented_ranks(values, check_sense(sense)))


def _rank_tie_sum(ranks: Iterable[float]) -> float:
    """Return ``sum(t^3 - t)`` over the groups of equal midranks.

    Disjoint groups of consecutive positions have different mean positions, so
    values share a midrank exactly when they form one tie group.
    """
    return float(sum(size**3 - size for size in Counter(ranks).values()))


def _normal_two_sided(statistic: float, mean: float, variance: float) -> float:
    """Return the continuity-corrected two-sided normal p-value, 1 without variance."""
    if variance <= 0.0:
        return 1.0
    z = max((abs(statistic - mean) - _CONTINUITY) / math.sqrt(variance), 0.0)
    return min(1.0, 2.0 * normal_sf(z))


def _normal_tail(positive: float, mean: float, variance: float, alternative: Alternative) -> float:
    """Return the continuity-corrected one-sided normal p-value of ``W+``, 1 without variance.

    The correction moves ``W+`` half a unit towards the mean, whichever tail
    the alternative reads: the upper one for ``"greater"``, the lower for
    ``"less"``.
    """
    if variance <= 0.0:
        return 1.0
    shift = positive - mean if alternative == "greater" else mean - positive
    return normal_sf((shift - _CONTINUITY) / math.sqrt(variance))
