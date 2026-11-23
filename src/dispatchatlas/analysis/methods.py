"""The catalog of the statistical methods DispatchAtlas implements.

Each :class:`Method` names the function that implements it, the assumptions
its result rests on, and the sources it follows, as keys of the bibliography
in :mod:`dispatchatlas.references`. The analysis bundle lists the methods it
used with their references, and the documentation renders the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dispatchatlas.core.errors import ConfigError

__all__ = ["METHOD_FAMILIES", "Method", "MethodFamily", "implemented_methods", "method"]

type MethodFamily = Literal[
    "test",
    "post-hoc",
    "correction",
    "effect-size",
    "interval",
    "bayesian",
    "power",
    "deviation",
    "indicator",
    "reporting",
]
"""The role a method plays in an analysis."""

METHOD_FAMILIES: tuple[MethodFamily, ...] = (
    "test",
    "post-hoc",
    "correction",
    "effect-size",
    "interval",
    "bayesian",
    "power",
    "deviation",
    "indicator",
    "reporting",
)


@dataclass(frozen=True, slots=True)
class Method:
    """One implemented statistical method.

    Attributes:
        id: Catalog key in kebab case.
        name: Human-readable name.
        family: The role the method plays.
        summary: One sentence on what the method computes.
        assumptions: What the result rests on.
        references: Bibliography keys of the sources.
        function: Dotted path of the function that implements it.
    """

    id: str
    name: str
    family: MethodFamily
    summary: str
    assumptions: tuple[str, ...]
    references: tuple[str, ...]
    function: str


_STATS = "dispatchatlas.analysis.stats"
_PAIRED = "one paired observation per problem, the problems independent of each other"

_METHODS: tuple[Method, ...] = (
    Method(
        "wilcoxon-signed-rank",
        "Wilcoxon signed-rank test",
        "test",
        "Ranks the magnitudes of the paired differences and compares the rank sums of the "
        "positive and negative ones, exactly up to 50 pairs, with or without ties.",
        (_PAIRED, "the differences are symmetric about their median"),
        ("wilcoxon1945individual", "derrac2011practical"),
        f"{_STATS}.nonparametric.wilcoxon_signed_rank",
    ),
    Method(
        "wilcoxon-pratt-zeros",
        "Pratt's treatment of zero differences",
        "test",
        "Ranks the magnitudes of the paired differences with the zeros included, which gives "
        "the zeros the lowest ranks, and drops the zeros' ranks from the signed-rank sums; the "
        "p-value comes from the signs of the non-zero differences given their ranks.",
        (_PAIRED, "the differences are symmetric about their median"),
        ("pratt1959remarks", "wilcoxon1945individual"),
        f"{_STATS}.nonparametric.wilcoxon_signed_rank",
    ),
    Method(
        "sign-test",
        "Sign test",
        "test",
        "Compares the counts of pairs each side wins with a fair binomial distribution.",
        (_PAIRED,),
        ("dixon1946statistical",),
        f"{_STATS}.nonparametric.sign_test",
    ),
    Method(
        "mann-whitney-u",
        "Mann-Whitney U test",
        "test",
        "Compares the rank sums of two independent samples in their pooled ranking.",
        ("two independent samples, such as the runs of two solvers on one problem",),
        ("mann1947test",),
        f"{_STATS}.nonparametric.mann_whitney_u",
    ),
    Method(
        "friedman",
        "Friedman test",
        "test",
        "Ranks the solvers within each problem and tests whether their rank sums differ, "
        "with a chi-square distribution.",
        ("complete blocks: every solver on every problem", "the problems are independent"),
        ("friedman1937use", "demsar2006statistical"),
        f"{_STATS}.nonparametric.friedman_test",
    ),
    Method(
        "iman-davenport",
        "Iman-Davenport test",
        "test",
        "Reads the Friedman statistic through an F distribution, which is less "
        "conservative than the chi-square reading.",
        ("the assumptions of the Friedman test",),
        ("iman1980approximations", "demsar2006statistical"),
        f"{_STATS}.nonparametric.iman_davenport",
    ),
    Method(
        "tost-paired-t",
        "Two one-sided paired t tests",
        "test",
        "Shows that the mean paired difference lies inside an equivalence margin by rejecting, "
        "with one-sided paired t tests, that it is at most the lower bound and that it is at "
        "least the upper bound, and reports the larger of the two p-values.",
        (
            _PAIRED,
            "approximately normal differences, or enough pairs for their mean to be normal",
            "a margin fixed before the data are seen",
        ),
        ("schuirmann1987comparison", "lakens2017equivalence"),
        f"{_STATS}.equivalence.tost_paired_t",
    ),
    Method(
        "tost-signed-rank",
        "Two one-sided signed-rank tests",
        "test",
        "Shows that the paired differences are centred inside an equivalence margin by "
        "rejecting, with one-sided Wilcoxon signed-rank tests of the differences shifted by "
        "each bound, that the centre is at most the lower bound and that it is at least the "
        "upper bound, and reports the larger of the two p-values. A difference equal to a "
        "bound is a zero, which Pratt's treatment ranks and then drops from the sums.",
        (
            _PAIRED,
            "the differences are symmetric about their median",
            "a margin fixed before the data are seen",
        ),
        (
            "schuirmann1987comparison",
            "lakens2017equivalence",
            "wilcoxon1945individual",
            "pratt1959remarks",
            "conover1999practical",
        ),
        f"{_STATS}.equivalence.tost_signed_rank",
    ),
    Method(
        "nemenyi",
        "Nemenyi test",
        "post-hoc",
        "Declares two solvers different when their average ranks differ by more than a "
        "critical difference from the studentized range distribution.",
        ("an omnibus test rejected", "at most 12 solvers, alpha 0.05 or 0.10"),
        ("demsar2006statistical",),
        f"{_STATS}.nonparametric.nemenyi_critical_difference",
    ),
    Method(
        "bonferroni",
        "Bonferroni correction",
        "correction",
        "Multiplies every p-value by the number of comparisons.",
        ("any dependence between the tests",),
        ("dunn1961multiple",),
        f"{_STATS}.corrections.adjust_p_values",
    ),
    Method(
        "holm",
        "Holm correction",
        "correction",
        "Multiplies the i-th smallest of m p-values by m - i + 1, keeping the adjusted "
        "values monotone.",
        ("any dependence between the tests",),
        ("holm1979simple", "garcia2008extension"),
        f"{_STATS}.corrections.adjust_p_values",
    ),
    Method(
        "hochberg",
        "Hochberg correction",
        "correction",
        "Steps up from the largest p-value with Holm's multipliers.",
        ("independent or positively dependent tests",),
        ("hochberg1988sharper", "garcia2008extension"),
        f"{_STATS}.corrections.adjust_p_values",
    ),
    Method(
        "shaffer",
        "Shaffer correction",
        "correction",
        "Replaces Holm's multipliers by the largest number of pairwise hypotheses that can "
        "be true together.",
        ("the family holds every pairwise comparison of the solvers",),
        ("shaffer1986modified", "garcia2008extension"),
        f"{_STATS}.corrections.adjust_p_values",
    ),
    Method(
        "finner",
        "Finner correction",
        "correction",
        "Adjusts the i-th smallest of m p-values to 1 - (1 - p)^(m / i), keeping the "
        "adjusted values monotone.",
        ("independent tests",),
        ("finner1993monotonicity", "garcia2008extension"),
        f"{_STATS}.corrections.adjust_p_values",
    ),
    Method(
        "paired-win-share",
        "Paired win share",
        "effect-size",
        "The share of problems on which one solver beats the other, ties counting one half.",
        (_PAIRED,),
        ("demsar2006statistical",),
        f"{_STATS}.effect_sizes.paired_win_share",
    ),
    Method(
        "cohens-dz",
        "Cohen's d_z",
        "effect-size",
        "The mean paired difference divided by the standard deviation of the differences.",
        (_PAIRED, "finite differences"),
        ("cohen1988statistical",),
        f"{_STATS}.effect_sizes.cohens_dz",
    ),
    Method(
        "vargha-delaney-a12",
        "Vargha-Delaney A12",
        "effect-size",
        "The probability that a random run of one sample beats a random run of the other, "
        "ties counting one half.",
        ("two independent samples",),
        ("vargha2000critique",),
        f"{_STATS}.effect_sizes.vargha_delaney_a12",
    ),
    Method(
        "cliffs-delta",
        "Cliff's delta",
        "effect-size",
        "The share of cross pairs one sample wins minus the share it loses.",
        ("two independent samples",),
        ("cliff1993dominance", "romano2006appropriate"),
        f"{_STATS}.effect_sizes.cliffs_delta",
    ),
    Method(
        "cohens-d",
        "Cohen's d_s",
        "effect-size",
        "The difference of two sample means divided by their pooled standard deviation.",
        ("two independent samples", "comparable variances"),
        ("cohen1988statistical",),
        f"{_STATS}.effect_sizes.cohens_d",
    ),
    Method(
        "kendalls-w",
        "Kendall's W",
        "effect-size",
        "Kendall's coefficient of concordance of the solver rankings over the problems, "
        "the effect size of the Friedman test.",
        ("the assumptions of the Friedman test",),
        ("kendall1939problem",),
        f"{_STATS}.nonparametric.FriedmanResult",
    ),
    Method(
        "hodges-lehmann-shift",
        "Hodges-Lehmann shift",
        "effect-size",
        "The median of the Walsh averages of the paired differences, the estimate of their "
        "centre that the signed-rank test implies.",
        (_PAIRED, "the differences are symmetric about their centre"),
        ("hodges1963estimates",),
        f"{_STATS}.location.hodges_lehmann_shift",
    ),
    Method(
        "bootstrap-bca",
        "Bias-corrected and accelerated bootstrap interval",
        "interval",
        "Reads the interval from resampled statistics at levels corrected for bias and skewness.",
        ("independent observations", "enough observations for resampling to vary"),
        ("efron1987better", "hyndman1996sample"),
        f"{_STATS}.bootstrap.bootstrap_interval",
    ),
    Method(
        "bootstrap-percentile",
        "Percentile bootstrap interval",
        "interval",
        "Reads the interval from the quantiles of resampled statistics.",
        ("independent observations",),
        ("efron1987better", "hyndman1996sample"),
        f"{_STATS}.bootstrap.bootstrap_interval",
    ),
    Method(
        "signed-rank-interval",
        "Signed-rank interval of the shift",
        "interval",
        "The shifts that neither one-sided Wilcoxon signed-rank test of the shifted differences "
        "rejects, found by bisection over the Walsh averages; with Pratt's treatment of zeros it "
        "lies inside an equivalence margin exactly when the two one-sided signed-rank tests of "
        "equivalence reject.",
        (_PAIRED, "the differences are symmetric about their median"),
        (
            "hodges1963estimates",
            "wilcoxon1945individual",
            "pratt1959remarks",
            "schuirmann1987comparison",
        ),
        f"{_STATS}.location.signed_rank_interval",
    ),
    Method(
        "bayesian-sign-test",
        "Bayesian sign test",
        "bayesian",
        "Puts a Dirichlet posterior on the shares of differences below, inside, and above a "
        "region of practical equivalence.",
        (_PAIRED,),
        ("benavoli2017time",),
        f"{_STATS}.bayesian.bayesian_sign_test",
    ),
    Method(
        "bayesian-signed-rank-test",
        "Bayesian signed-rank test",
        "bayesian",
        "Puts a Dirichlet process posterior on the differences and weighs the Walsh averages "
        "that fall below, inside, and above a region of practical equivalence.",
        (_PAIRED,),
        ("benavoli2014bayesian", "benavoli2017time"),
        f"{_STATS}.bayesian.bayesian_signed_rank_test",
    ),
    Method(
        "wilcoxon-power",
        "Power of the Wilcoxon signed-rank test",
        "power",
        "Simulates normal paired differences to find the effect a number of problems "
        "detects, or the number of problems an effect needs.",
        ("normally distributed paired differences", "the untied test as this package runs it"),
        ("campelo2019sample",),
        f"{_STATS}.power.detectable_effect_size",
    ),
    Method(
        "relative-percentage-deviation",
        "Average relative percentage deviation",
        "deviation",
        "Averages the percentage by which run values exceed per-problem reference values.",
        ("a positive reference value for every problem",),
        ("vallada2015new",),
        "dispatchatlas.analysis.deviation.arpd",
    ),
    Method(
        "hypervolume",
        "Hypervolume",
        "indicator",
        "The volume of objective space an approximation set dominates up to a reference point.",
        ("one reference point shared by every compared set",),
        ("zitzler1999multiobjective", "while2006faster", "ishibuchi2018specify"),
        "dispatchatlas.core.pareto.hypervolume",
    ),
    Method(
        "igd",
        "Inverted generational distance",
        "indicator",
        "The mean distance from each reference-set point to the nearest point of an "
        "approximation set.",
        ("one reference set shared by every compared set",),
        ("coello2004study",),
        "dispatchatlas.analysis.indicators.igd",
    ),
    Method(
        "igd-plus",
        "IGD+",
        "indicator",
        "The inverted generational distance counting only the objectives in which the "
        "approximation set is worse.",
        ("one reference set shared by every compared set",),
        ("ishibuchi2015modified",),
        "dispatchatlas.analysis.indicators.igd_plus",
    ),
    Method(
        "effect-sizes-beside-p-values",
        "Effect sizes and intervals beside every p-value",
        "reporting",
        "Every table that reports a p-value also reports an effect size and a confidence interval.",
        (),
        ("carrasco2020recent",),
        "dispatchatlas.analysis.tables.check_p_values",
    ),
    Method(
        "equivalence-report",
        "Equivalence and difference outcomes",
        "reporting",
        "Reads an equivalence test and a test of a difference at one level together as one of "
        "four outcomes: equivalent, different, equivalent and different, or inconclusive.",
        ("both tests ask about the same paired differences, at the same level",),
        ("lakens2017equivalence",),
        f"{_STATS}.equivalence.equivalence_report",
    ),
)


def implemented_methods(family: MethodFamily | None = None) -> tuple[Method, ...]:
    """Return the implemented methods, all of them or those of one family."""
    if family is None:
        return _METHODS
    if family not in METHOD_FAMILIES:
        raise ConfigError(
            f"unknown method family {family!r}; choose one of {list(METHOD_FAMILIES)}"
        )
    return tuple(item for item in _METHODS if item.family == family)


def method(method_id: str) -> Method:
    """Return the method with the given id.

    Raises:
        ConfigError: If no method has that id.
    """
    for item in _METHODS:
        if item.id == method_id:
            return item
    raise ConfigError(f"unknown method {method_id!r}")
