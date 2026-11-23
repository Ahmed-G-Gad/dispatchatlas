"""Statistical kernels: plain numbers in, small frozen results out.

The kernels know nothing about campaigns. The rank tests, effect sizes, and
corrections are pure Python and verified against reference values computed
with SciPy; the bootstrap, the Bayesian tests, and the power simulation draw
from ``numpy.random.default_rng(seed)``.

- :mod:`~dispatchatlas.analysis.stats.distributions`: normal, chi-square, F, and Student t tails.
- :mod:`~dispatchatlas.analysis.stats.ranks`: ties, midranks, and orientation.
- :mod:`~dispatchatlas.analysis.stats.nonparametric`: Wilcoxon, sign,
  Mann-Whitney, Friedman, Iman-Davenport, and Nemenyi.
- :mod:`~dispatchatlas.analysis.stats.equivalence`: two one-sided tests of equivalence
  within a margin, and the four outcomes they give with a test of a difference.
- :mod:`~dispatchatlas.analysis.stats.location`: the Hodges-Lehmann shift of paired differences
  and its exact signed-rank interval.
- :mod:`~dispatchatlas.analysis.stats.effect_sizes`: A12, Cliff's delta,
  Cohen's d and d_z, and the paired win share.
- :mod:`~dispatchatlas.analysis.stats.bootstrap`: BCa and percentile intervals.
- :mod:`~dispatchatlas.analysis.stats.bayesian`: Bayesian sign and signed-rank tests.
- :mod:`~dispatchatlas.analysis.stats.corrections`: multiple-comparison corrections.
- :mod:`~dispatchatlas.analysis.stats.power`: power and sample size of the paired test.
"""

from dispatchatlas.analysis.stats.bayesian import (
    BayesianResult,
    bayesian_sign_test,
    bayesian_signed_rank_test,
)
from dispatchatlas.analysis.stats.bootstrap import BootstrapInterval, bootstrap_interval, quantile
from dispatchatlas.analysis.stats.corrections import (
    CORRECTIONS,
    Correction,
    adjust_p_values,
    shaffer_hypothesis_counts,
)
from dispatchatlas.analysis.stats.distributions import (
    chi2_sf,
    f_sf,
    normal_cdf,
    normal_ppf,
    normal_sf,
    t_ppf,
    t_sf,
)
from dispatchatlas.analysis.stats.effect_sizes import (
    Magnitude,
    a12_magnitude,
    cliffs_delta,
    cliffs_delta_magnitude,
    cohens_d,
    cohens_d_magnitude,
    cohens_dz,
    paired_outcomes,
    paired_win_share,
    vargha_delaney_a12,
)
from dispatchatlas.analysis.stats.equivalence import (
    EquivalenceOutcome,
    TostPairedTResult,
    TostResult,
    TostSignedRankResult,
    equivalence_report,
    tost_paired_t,
    tost_signed_rank,
)
from dispatchatlas.analysis.stats.location import (
    SignedRankInterval,
    hodges_lehmann_shift,
    signed_rank_interval,
)
from dispatchatlas.analysis.stats.nonparametric import (
    Alternative,
    FriedmanResult,
    ImanDavenportResult,
    MannWhitneyResult,
    WilcoxonResult,
    ZeroMethod,
    friedman_test,
    iman_davenport,
    mann_whitney_u,
    nemenyi_critical_difference,
    nemenyi_q,
    sign_test,
    wilcoxon_signed_rank,
)
from dispatchatlas.analysis.stats.power import (
    PowerAssessment,
    attainable_floor,
    detectable_effect_size,
    required_paired_count,
    wilcoxon_power,
)
from dispatchatlas.analysis.stats.ranks import (
    TIE_TOLERANCE,
    Sense,
    midranks,
    oriented_ranks,
    rank_block,
    strictly_better,
    ties,
)

__all__ = [
    "CORRECTIONS",
    "TIE_TOLERANCE",
    "Alternative",
    "BayesianResult",
    "BootstrapInterval",
    "Correction",
    "EquivalenceOutcome",
    "FriedmanResult",
    "ImanDavenportResult",
    "Magnitude",
    "MannWhitneyResult",
    "PowerAssessment",
    "Sense",
    "SignedRankInterval",
    "TostPairedTResult",
    "TostResult",
    "TostSignedRankResult",
    "WilcoxonResult",
    "ZeroMethod",
    "a12_magnitude",
    "adjust_p_values",
    "attainable_floor",
    "bayesian_sign_test",
    "bayesian_signed_rank_test",
    "bootstrap_interval",
    "chi2_sf",
    "cliffs_delta",
    "cliffs_delta_magnitude",
    "cohens_d",
    "cohens_d_magnitude",
    "cohens_dz",
    "detectable_effect_size",
    "equivalence_report",
    "f_sf",
    "friedman_test",
    "hodges_lehmann_shift",
    "iman_davenport",
    "mann_whitney_u",
    "midranks",
    "nemenyi_critical_difference",
    "nemenyi_q",
    "normal_cdf",
    "normal_ppf",
    "normal_sf",
    "oriented_ranks",
    "paired_outcomes",
    "paired_win_share",
    "quantile",
    "rank_block",
    "required_paired_count",
    "shaffer_hypothesis_counts",
    "sign_test",
    "signed_rank_interval",
    "strictly_better",
    "t_ppf",
    "t_sf",
    "ties",
    "tost_paired_t",
    "tost_signed_rank",
    "vargha_delaney_a12",
    "wilcoxon_power",
    "wilcoxon_signed_rank",
]
