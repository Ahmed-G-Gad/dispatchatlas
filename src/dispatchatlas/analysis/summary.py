"""The statistical comparison of the solvers of a campaign, under one protocol.

:func:`summarize` follows the non-parametric protocol for comparing
algorithms over several problems (Demšar 2006; Derrac et al. 2011):

1. **Cells.** The replicates of each solver on each problem become one
   *score*, the configured statistic (mean or median) of the planned runs,
   where a run that did not succeed counts as the worst value (infinity for
   a minimised objective). Descriptive statistics and a bootstrap interval of
   the mean describe the succeeded runs.
2. **Reliability.** Each solver's succeeded, infeasible, failed, and missing
   runs, over its planned runs.
3. **Ranks and omnibus test.** On the *complete blocks*, the problems every
   analysed solver is planned on, the scores are ranked per problem (rank 1
   best, ties sharing their mean rank). With at least three solvers and two
   blocks, the Friedman test and its Iman-Davenport form test whether the
   solvers differ, and Kendall's W, with a bootstrap interval over blocks,
   measures by how much.
4. **Post-hoc test.** When the configured omnibus test rejects, the Nemenyi
   critical difference says which average ranks differ.
5. **Pairwise tests.** Every pair of solvers is compared over the problems
   both are planned on with the Wilcoxon signed-rank test, and the p-values
   are adjusted over the family. Each comparison reports, beside its p-value,
   the paired win share with a bootstrap interval, the mean advantage with a
   bootstrap interval when no failure makes it infinite, and Cohen's d_z, as
   Carrasco et al. (2020) recommend.
6. **Limitations.** What qualifies the results: failed or missing runs,
   problems left out of the ranks, too few shared problems for the paired
   test to reach significance, cells below a replicate floor, and tests that
   could not run.

Two scores tie when they are equal within
:data:`~dispatchatlas.analysis.stats.ranks.TIE_TOLERANCE`. A tied pair enters
the Wilcoxon test as a zero difference, so the test and the win counts
always agree on what is a tie. Every bootstrap uses the configured seed.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.analysis._scope import Scope, aggregate, resolve_scope, run_scores
from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.dataset import Dataset
from dispatchatlas.analysis.results import (
    CellSummary,
    CriticalDifference,
    Limitation,
    OmnibusResult,
    PairwiseComparison,
    SolverSummary,
    Summary,
)
from dispatchatlas.analysis.stats.bootstrap import BootstrapInterval, Statistic, bootstrap_interval
from dispatchatlas.analysis.stats.corrections import adjust_p_values
from dispatchatlas.analysis.stats.effect_sizes import cohens_dz
from dispatchatlas.analysis.stats.nonparametric import (
    friedman_statistic,
    friedman_test,
    iman_davenport,
    nemenyi_critical_difference,
    tie_sum,
    wilcoxon_signed_rank,
)
from dispatchatlas.analysis.stats.power import attainable_floor
from dispatchatlas.analysis.stats.ranks import Sense, oriented_ranks, rank_block, ties
from dispatchatlas.core.errors import ConfigError

__all__ = ["summarize"]


def summarize(dataset: Dataset, config: AnalysisConfig | None = None) -> Summary:
    """Compare the solvers of a dataset under one protocol.

    Args:
        dataset: The campaign's runs, from
            :func:`~dispatchatlas.analysis.dataset.load_dataset`.
        config: The analysis settings; the defaults analyse the only
            protocol on the primary objective.

    Returns:
        Cells, solver summaries, omnibus and post-hoc tests, pairwise
        comparisons, and limitations.

    Raises:
        ConfigError: If the configuration names something the dataset lacks,
            leaves the protocol or objective ambiguous, or asks for Shaffer's
            correction when some pair of solvers shares no problem.
    """
    settings = config if config is not None else AnalysisConfig()
    scope = resolve_scope(dataset, settings)
    limitations: list[Limitation] = []
    cells = _cells(scope, settings)
    scores = {(cell.solver, cell.problem): cell.score for cell in cells}
    solvers = tuple(entry.name for entry in scope.dataset.solvers)
    problems = tuple(problem.id for problem in scope.dataset.problems)
    blocks = tuple(
        problem for problem in problems if all((solver, problem) in scores for solver in solvers)
    )
    ranks = _average_ranks(scores, solvers, blocks, scope.sense)
    omnibus = _omnibus(scores, solvers, blocks, scope.sense, settings, limitations)
    critical = _critical_difference(omnibus, ranks, settings, limitations)
    pairwise = _pairwise(scores, solvers, problems, scope.sense, settings, limitations)
    limitations[:0] = _design_limitations(scope, settings, problems, blocks, pairwise)
    return Summary(
        campaign=dataset.campaign,
        protocol=scope.protocol,
        objective=scope.objective,
        sense=scope.sense,
        config=settings,
        solvers=solvers,
        problems=problems,
        blocks=blocks,
        cells=cells,
        solver_summaries=_solver_summaries(scope.dataset, solvers, scores, ranks),
        pairwise=pairwise,
        omnibus=omnibus,
        critical_difference=critical,
        limitations=tuple(limitations),
    )


def _cells(scope: Scope, config: AnalysisConfig) -> tuple[CellSummary, ...]:
    dataset = scope.dataset
    cells: list[CellSummary] = []
    for problem in dataset.problems:
        for solver in dataset.solvers:
            runs = dataset.cell(solver.name, problem.id, scope.protocol)
            if not runs:
                continue
            scores = run_scores(runs, scope.objective, scope.sense)
            values = [score for score, run in zip(scores, runs, strict=True) if run.succeeded]
            interval = _interval(values, config) if values else None
            cells.append(
                CellSummary(
                    solver=solver.name,
                    problem=problem.id,
                    planned=len(runs),
                    succeeded=len(values),
                    score=aggregate(scores, config.statistic),
                    mean=interval.estimate if interval else None,
                    median=float(statistics.median(values)) if values else None,
                    std=statistics.stdev(values) if len(values) > 1 else None,
                    best=(min(values) if scope.sense == "min" else max(values)) if values else None,
                    mean_low=interval.low if interval else None,
                    mean_high=interval.high if interval else None,
                )
            )
    return tuple(cells)


def _interval(
    sample: Sequence[float] | Sequence[Sequence[float]],
    config: AnalysisConfig,
    statistic: Statistic = "mean",
) -> BootstrapInterval:
    return bootstrap_interval(
        sample,
        statistic=statistic,
        level=config.confidence,
        resamples=config.resamples,
        seed=config.seed,
        method=config.interval,
    )


def _average_ranks(
    scores: Mapping[tuple[str, str], float],
    solvers: Sequence[str],
    blocks: Sequence[str],
    sense: Sense,
) -> dict[str, float]:
    if not blocks:
        return {}
    totals = dict.fromkeys(solvers, 0.0)
    for problem in blocks:
        ranked = rank_block({solver: scores[(solver, problem)] for solver in solvers}, sense)
        for solver, rank in ranked.items():
            totals[solver] += rank
    return {solver: total / len(blocks) for solver, total in totals.items()}


def _omnibus(
    scores: Mapping[tuple[str, str], float],
    solvers: Sequence[str],
    blocks: Sequence[str],
    sense: Sense,
    config: AnalysisConfig,
    limitations: list[Limitation],
) -> OmnibusResult | None:
    if len(solvers) < 3:
        return None
    if len(blocks) < 2:
        limitations.append(
            Limitation(
                "omnibus",
                f"The Friedman test needs at least two problems that every solver is planned "
                f"on; there are {len(blocks)}, so no omnibus or post-hoc test is run.",
            )
        )
        return None
    matrix = [[scores[(solver, problem)] for solver in solvers] for problem in blocks]
    friedman = friedman_test([dict(zip(solvers, row, strict=True)) for row in matrix], sense)
    corrected = iman_davenport(friedman)
    rows = [[*oriented_ranks(row, sense), tie_sum(row)] for row in matrix]
    concordance = _interval(rows, config, _kendalls_w)
    p_value = friedman.p_value if config.omnibus == "friedman" else corrected.p_value
    return OmnibusResult(
        blocks=friedman.blocks,
        groups=friedman.groups,
        friedman_statistic=friedman.statistic,
        friedman_p_value=friedman.p_value,
        iman_davenport_statistic=corrected.statistic,
        iman_davenport_p_value=corrected.p_value,
        kendalls_w=friedman.kendalls_w,
        kendalls_w_low=concordance.low,
        kendalls_w_high=concordance.high,
        confidence=config.confidence,
        test=config.omnibus,
        alpha=config.alpha,
        significant=p_value < config.alpha,
    )


def _kendalls_w(rows: NDArray[np.float64]) -> float:
    """Return Kendall's W of rows holding one block's ranks and, last, its tie sum."""
    blocks = int(rows.shape[0])
    groups = int(rows.shape[1]) - 1
    rank_sums = [float(value) for value in rows[:, :groups].sum(axis=0)]
    statistic = friedman_statistic(rank_sums, blocks, float(rows[:, groups].sum()))
    return statistic / (blocks * (groups - 1))


def _critical_difference(
    omnibus: OmnibusResult | None,
    ranks: Mapping[str, float],
    config: AnalysisConfig,
    limitations: list[Limitation],
) -> CriticalDifference | None:
    if omnibus is None:
        return None
    if not omnibus.significant:
        limitations.append(
            Limitation(
                "post-hoc",
                f"The {_OMNIBUS_NAMES[omnibus.test]} test does not reject at alpha = "
                f"{config.alpha:g}, so the Nemenyi post-hoc test is not run.",
            )
        )
        return None
    try:
        value = nemenyi_critical_difference(omnibus.groups, omnibus.blocks, config.alpha)
    except ConfigError as error:
        limitations.append(Limitation("post-hoc", f"The Nemenyi test is not run: {error}."))
        return None
    ordered = sorted(ranks.items(), key=lambda item: (item[1], item[0]))
    pairs = tuple(
        (first, second)
        for (first, low), (second, high) in combinations(ordered, 2)
        if high - low > value
    )
    return CriticalDifference(
        value=value,
        alpha=config.alpha,
        groups=omnibus.groups,
        blocks=omnibus.blocks,
        average_ranks=tuple(ordered),
        significant_pairs=pairs,
        cliques=_cliques(ordered, value),
    )


def _cliques(ordered: Sequence[tuple[str, float]], critical: float) -> tuple[tuple[str, ...], ...]:
    """Return the maximal runs of rank-ordered solvers within ``critical`` of each other."""
    spans: list[tuple[int, int]] = []
    for start, (_, low) in enumerate(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][1] - low <= critical:
            end += 1
        if not spans or end > spans[-1][1]:
            spans.append((start, end))
    return tuple(tuple(name for name, _ in ordered[start : end + 1]) for start, end in spans)


def _pairwise(
    scores: Mapping[tuple[str, str], float],
    solvers: Sequence[str],
    problems: Sequence[str],
    sense: Sense,
    config: AnalysisConfig,
    limitations: list[Limitation],
) -> tuple[PairwiseComparison, ...]:
    rows: list[PairwiseComparison] = []
    unpaired: list[str] = []
    for left, right in combinations(solvers, 2):
        shared = [
            problem
            for problem in problems
            if (left, problem) in scores and (right, problem) in scores
        ]
        if not shared:
            unpaired.append(f"{left} and {right}")
            continue
        rows.append(
            _compare(
                left,
                right,
                [scores[(left, problem)] for problem in shared],
                [scores[(right, problem)] for problem in shared],
                sense,
                config,
            )
        )
    if unpaired:
        limitations.append(
            Limitation(
                "pairwise",
                "These solvers share no problem and are not compared: " + "; ".join(unpaired) + ".",
            )
        )
        if config.correction == "shaffer":
            raise ConfigError(
                "Shaffer's correction needs every pair of solvers compared, but some share no "
                "problem; choose Holm's or Finner's correction"
            )
    adjusted = adjust_p_values(
        [row.p_value for row in rows],
        config.correction,
        keys=[(row.left, row.right) for row in rows],
        groups=len(solvers) if config.correction == "shaffer" else None,
    )
    return tuple(
        replace(row, adjusted_p_value=value) for row, value in zip(rows, adjusted, strict=True)
    )


def _compare(
    left: str,
    right: str,
    left_scores: Sequence[float],
    right_scores: Sequence[float],
    sense: Sense,
    config: AnalysisConfig,
) -> PairwiseComparison:
    """Compare two solvers' scores over their shared problems, before any correction.

    Args:
        left: Name of the first solver.
        right: Name of the second solver.
        left_scores: The first solver's score on each shared problem.
        right_scores: The second solver's score on the same problems.
        sense: Whether smaller or larger scores are better.
        config: Confidence level, bootstrap, and seed.

    Returns:
        The comparison, with the adjusted p-value equal to the raw one.
    """
    wins = losses = tied = 0
    advantages: list[float] = []
    outcomes: list[float] = []
    for mine, theirs in zip(left_scores, right_scores, strict=True):
        if ties(mine, theirs):
            tied += 1
            advantages.append(0.0)
            outcomes.append(0.5)
            continue
        advantage = theirs - mine if sense == "min" else mine - theirs
        advantages.append(advantage)
        if advantage > 0.0:
            wins += 1
            outcomes.append(1.0)
        else:
            losses += 1
            outcomes.append(0.0)
    test = wilcoxon_signed_rank(advantages)
    share = _interval(outcomes, config)
    finite = all(math.isfinite(value) for value in advantages)
    mean = _interval(advantages, config) if finite else None
    return PairwiseComparison(
        left=left,
        right=right,
        pairs=len(advantages),
        wins=wins,
        losses=losses,
        ties=tied,
        win_share=share.estimate,
        win_share_low=share.low,
        win_share_high=share.high,
        median_advantage=_median(advantages),
        mean_advantage=mean.estimate if mean else None,
        advantage_low=mean.low if mean else None,
        advantage_high=mean.high if mean else None,
        cohens_dz=cohens_dz(advantages) if finite else None,
        statistic=test.statistic,
        p_value=test.p_value,
        adjusted_p_value=test.p_value,
        method=test.method,
        confidence=config.confidence,
    )


def _median(values: Sequence[float]) -> float | None:
    """Return the median, or ``None`` when opposite infinities make it undefined."""
    middle = float(statistics.median(values))
    return None if math.isnan(middle) else middle


def _solver_summaries(
    dataset: Dataset,
    solvers: Sequence[str],
    scores: Mapping[tuple[str, str], float],
    ranks: Mapping[str, float],
) -> tuple[SolverSummary, ...]:
    summaries = []
    for solver in solvers:
        runs = [run for run in dataset.runs if run.solver == solver]
        status = [run.status for run in runs]
        summaries.append(
            SolverSummary(
                solver=solver,
                planned=len(runs),
                succeeded=status.count("succeeded"),
                infeasible=status.count("infeasible"),
                failed=status.count("failed"),
                missing=status.count("missing"),
                problems=sum(1 for key in scores if key[0] == solver),
                average_rank=ranks.get(solver),
            )
        )
    return tuple(summaries)


def _design_limitations(
    scope: Scope,
    config: AnalysisConfig,
    problems: Sequence[str],
    blocks: Sequence[str],
    pairwise: Sequence[PairwiseComparison],
) -> list[Limitation]:
    found: list[Limitation] = []
    counts = scope.dataset.counts()
    planned = len(scope.dataset.runs)
    if counts["missing"]:
        found.append(
            Limitation(
                "campaign",
                f"{counts['missing']} of {planned} planned runs have no record yet; they count "
                "as failures.",
            )
        )
    if counts["failed"] or counts["infeasible"]:
        found.append(
            Limitation(
                "failures",
                f"{counts['failed']} runs failed and {counts['infeasible']} completed without a "
                "feasible schedule; each counts as the worst value.",
            )
        )
    left_out = [problem for problem in problems if problem not in blocks]
    if left_out and len(scope.dataset.solvers) > 1:
        found.append(
            Limitation(
                "ranking",
                f"{len(left_out)} of {len(problems)} problems are left out of the ranks and the "
                "omnibus test because the plan excludes some solvers from them: "
                + ", ".join(left_out)
                + ".",
            )
        )
    floor = attainable_floor(config.alpha)
    short = [row for row in pairwise if row.pairs < floor]
    if short:
        fewest = min(row.pairs for row in short)
        found.append(
            Limitation(
                "pairwise",
                f"{len(short)} of {len(pairwise)} pairwise comparisons rest on fewer than {floor} "
                f"shared problems (the fewest is {fewest}); with so few pairs no two-sided "
                f"Wilcoxon test can reach p <= {config.alpha:g}, whatever the difference.",
            )
        )
    if config.replicate_floor is not None:
        below = []
        for entry in scope.dataset.solvers:
            if not entry.stochastic:
                continue
            sizes = {
                len(runs)
                for problem in problems
                if (runs := scope.dataset.cell(entry.name, problem, scope.protocol))
            }
            if sizes and min(sizes) < config.replicate_floor:
                below.append(f"{entry.name} ({min(sizes)})")
        if below:
            found.append(
                Limitation(
                    "replicates",
                    f"Stochastic solvers with fewer than {config.replicate_floor} replicates per "
                    "problem: " + ", ".join(below) + ".",
                )
            )
    return found


_OMNIBUS_NAMES: Mapping[str, str] = {"friedman": "Friedman", "iman-davenport": "Iman-Davenport"}
