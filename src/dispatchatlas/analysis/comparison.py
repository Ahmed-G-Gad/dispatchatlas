"""One solver against a set of baselines.

:func:`compare_to_baselines` answers the question a new method raises: is it
better than each of the established ones? Every baseline is compared with
the focal solver by the paired Wilcoxon signed-rank test of
:func:`~dispatchatlas.analysis.summary.summarize`, and the p-values are
adjusted over these ``k`` comparisons only, the many-to-one family, rather
than over every pair of solvers (Demšar 2006, section 3.2.2). Every row is
oriented towards the focal solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.dataset import Dataset
from dispatchatlas.analysis.results import BaselineComparison
from dispatchatlas.analysis.stats.corrections import adjust_p_values
from dispatchatlas.analysis.summary import summarize
from dispatchatlas.core.errors import ConfigError

__all__ = ["compare_to_baselines"]


def compare_to_baselines(
    dataset: Dataset,
    focal: str,
    baselines: Sequence[str] | None = None,
    config: AnalysisConfig | None = None,
) -> BaselineComparison:
    """Compare one solver with each baseline.

    Args:
        dataset: The campaign's runs.
        focal: Name of the solver under study.
        baselines: Names of the baselines. ``None`` takes every other solver
            the configuration selects.
        config: The analysis settings. Its ``solvers`` field, if set, must
            include the focal solver and the baselines.

    Returns:
        One row per baseline, with the focal solver on the left, adjusted
        with the configured correction over the baselines.

    Raises:
        ConfigError: If the focal solver or a baseline is unknown, a
            baseline is the focal solver or is named twice, a baseline shares
            no problem with the focal solver, or Shaffer's correction is
            asked for, which applies to all pairwise comparisons only.
    """
    settings = config if config is not None else AnalysisConfig()
    if settings.correction == "shaffer":
        raise ConfigError(
            "Shaffer's correction applies to all pairwise comparisons, not to comparisons "
            "with a focal solver; choose Holm's, Hochberg's, or Finner's correction"
        )
    available = [
        entry.name
        for entry in dataset.solvers
        if settings.solvers is None or entry.name in settings.solvers
    ]
    if focal not in available:
        raise ConfigError(_missing("focal solver", focal, available))
    chosen = [name for name in available if name != focal] if baselines is None else list(baselines)
    if not chosen:
        raise ConfigError("the comparison needs at least one baseline")
    for name in chosen:
        if name == focal:
            raise ConfigError(f"solver {focal!r} cannot be its own baseline")
        if name not in available:
            raise ConfigError(_missing("baseline", name, available))
    if len(set(chosen)) != len(chosen):
        raise ConfigError(f"a baseline is named twice: {chosen}")
    ordered = tuple(name for name in available if name in chosen)
    summary = summarize(dataset, replace(settings, solvers=(focal, *ordered), correction="none"))
    rows = []
    for baseline in ordered:
        try:
            rows.append(summary.comparison(focal, baseline))
        except ConfigError:
            raise ConfigError(
                f"baseline {baseline!r} shares no problem with {focal!r}, so they cannot be "
                "compared"
            ) from None
    adjusted = adjust_p_values(
        [row.p_value for row in rows],
        settings.correction,
        keys=[(row.left, row.right) for row in rows],
    )
    rows = [replace(row, adjusted_p_value=value) for row, value in zip(rows, adjusted, strict=True)]
    better = all(row.adjusted_p_value < settings.alpha and row.win_share > 0.5 for row in rows)
    return BaselineComparison(
        focal=focal,
        baselines=ordered,
        protocol=summary.protocol,
        objective=summary.objective,
        sense=summary.sense,
        correction=settings.correction,
        alpha=settings.alpha,
        rows=tuple(rows),
        better_than_all=better,
        limitations=summary.limitations,
    )


def _missing(what: str, name: str, known: Sequence[str]) -> str:
    return f"unknown {what} {name!r}; the analysed solvers are: {', '.join(known)}"
