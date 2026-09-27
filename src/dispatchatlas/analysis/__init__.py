"""Statistical analysis of campaign results.

Read a campaign directory, compare its solvers, and write every table,
figure, and caveat to one reproducible bundle::

    from dispatchatlas.analysis import AnalysisConfig, load_dataset, summarize, write_bundle

    dataset = load_dataset("results/my-campaign")
    summary = summarize(dataset, AnalysisConfig(correction="holm"))
    write_bundle("results/my-campaign", "analysis/my-campaign")

The modules, from data to reports:

- :mod:`~dispatchatlas.analysis.dataset`: the planned runs of a campaign and
  what became of each; a run without a feasible result counts as a failure.
- :mod:`~dispatchatlas.analysis.config`: the settings of an analysis.
- :mod:`~dispatchatlas.analysis.summary`, :mod:`~dispatchatlas.analysis.comparison`:
  ranks, omnibus and post-hoc tests, and paired comparisons with effect
  sizes and intervals; one solver against baselines.
- :mod:`~dispatchatlas.analysis.deviation`, :mod:`~dispatchatlas.analysis.indicators`:
  relative percentage deviation and multi-objective quality indicators.
- :mod:`~dispatchatlas.analysis.stats`: the statistical kernels.
- :mod:`~dispatchatlas.analysis.methods`: the catalog of implemented methods.
- :mod:`~dispatchatlas.analysis.tables`, :mod:`~dispatchatlas.analysis.manifest`,
  :mod:`~dispatchatlas.analysis.bundle`: tables, the reproducibility manifest,
  and the bundle.
"""

from dispatchatlas.analysis.bundle import write_bundle
from dispatchatlas.analysis.comparison import compare_to_baselines
from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.dataset import Dataset, Run, load_dataset
from dispatchatlas.analysis.deviation import DeviationRow, arpd, relative_deviation
from dispatchatlas.analysis.indicators import (
    IndicatorReport,
    IndicatorRow,
    ReferencePoint,
    ensure_shared_reference_points,
    igd,
    igd_plus,
    indicator_report,
    reference_point,
)
from dispatchatlas.analysis.methods import Method, implemented_methods, method
from dispatchatlas.analysis.results import (
    BaselineComparison,
    CellSummary,
    CriticalDifference,
    Limitation,
    OmnibusResult,
    PairwiseComparison,
    SolverSummary,
    Summary,
)
from dispatchatlas.analysis.stats import (
    BayesianResult,
    BootstrapInterval,
    FriedmanResult,
    ImanDavenportResult,
    MannWhitneyResult,
    PowerAssessment,
    WilcoxonResult,
    adjust_p_values,
    bayesian_sign_test,
    bayesian_signed_rank_test,
    bootstrap_interval,
    chi2_sf,
    cliffs_delta,
    cohens_d,
    cohens_dz,
    detectable_effect_size,
    f_sf,
    friedman_test,
    iman_davenport,
    mann_whitney_u,
    nemenyi_critical_difference,
    normal_cdf,
    normal_ppf,
    paired_win_share,
    required_paired_count,
    sign_test,
    vargha_delaney_a12,
    wilcoxon_signed_rank,
)
from dispatchatlas.analysis.summary import summarize
from dispatchatlas.analysis.tables import Table, to_csv, to_latex, to_markdown

__all__ = [
    "AnalysisConfig",
    "BaselineComparison",
    "BayesianResult",
    "BootstrapInterval",
    "CellSummary",
    "CriticalDifference",
    "Dataset",
    "DeviationRow",
    "FriedmanResult",
    "ImanDavenportResult",
    "IndicatorReport",
    "IndicatorRow",
    "Limitation",
    "MannWhitneyResult",
    "Method",
    "OmnibusResult",
    "PairwiseComparison",
    "PowerAssessment",
    "ReferencePoint",
    "Run",
    "SolverSummary",
    "Summary",
    "Table",
    "WilcoxonResult",
    "adjust_p_values",
    "arpd",
    "bayesian_sign_test",
    "bayesian_signed_rank_test",
    "bootstrap_interval",
    "chi2_sf",
    "cliffs_delta",
    "cohens_d",
    "cohens_dz",
    "compare_to_baselines",
    "detectable_effect_size",
    "ensure_shared_reference_points",
    "f_sf",
    "friedman_test",
    "igd",
    "igd_plus",
    "iman_davenport",
    "implemented_methods",
    "indicator_report",
    "load_dataset",
    "mann_whitney_u",
    "method",
    "nemenyi_critical_difference",
    "normal_cdf",
    "normal_ppf",
    "paired_win_share",
    "reference_point",
    "relative_deviation",
    "required_paired_count",
    "sign_test",
    "summarize",
    "to_csv",
    "to_latex",
    "to_markdown",
    "vargha_delaney_a12",
    "wilcoxon_signed_rank",
    "write_bundle",
]
