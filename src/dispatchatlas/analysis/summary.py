"""The statistical comparison of the solvers of a campaign, under one protocol.

:func:`summarize` follows the non-parametric protocol for comparing
algorithms over several problems (Demšar 2006; Derrac et al. 2011):

1. **Cells.** The replicates of each solver on each problem become one
   *score*, the configured statistic (mean or median) of the planned runs,
   where a run that did not succeed counts as the worst value (infinity for
   a minimised objective). With ``timeout_score`` ``"incumbent"``, a run
   killed at its hard limit counts instead by the feasible incumbent it held
   when its time limit passed. Descriptive statistics and a bootstrap
   interval of the mean describe the succeeded runs.
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
   with the missing ones by cell of solver and problem and
   the failed ones by solver and kind of error,
   runs whose worker did not begin the solve in time,
   runs killed at their hard limit and how they are scored,
   what the plan notes on the protocol, such as solvers declared
   deterministic that run once under a time limit,
   stop reasons that differ among the runs of the iterative solvers,
   completed runs whose wall time passed the time limit of the protocol,
   the processor share and the concurrency the runs of a protocol with a
   time limit executed under,
   problems left out of the ranks, too few shared problems for the paired
   test to reach significance, cells below a replicate floor, tests that
   could not run, and what the check of the campaign directory found beyond
   these (:func:`~dispatchatlas.campaigns.checking.check_campaign`), such as
   an altered campaign file, cells with records for only some of their
   runs, traces that cannot be read against time, or sessions on other
   code.

Two scores tie when they are equal within
:data:`~dispatchatlas.analysis.stats.ranks.TIE_TOLERANCE`. A tied pair enters
the Wilcoxon test as a zero difference, so the test and the win counts
always agree on what is a tie. Every bootstrap uses the configured seed.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
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
from dispatchatlas.campaigns import fairness
from dispatchatlas.campaigns.fairness import ProtocolOutcomes
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
            scores = run_scores(runs, scope.objective, scope.sense, config.timeout_score)
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
        cells = Counter(
            (run.solver, run.problem) for run in scope.dataset.runs if run.status == "missing"
        )
        named = _shown(
            [f"{solver} on {problem} ({count})" for (solver, problem), count in cells.items()]
        )
        found.append(
            Limitation(
                "campaign",
                f"{counts['missing']} of {planned} planned runs have no record yet; they count "
                "as failures. Their cells of solver and problem, each with its number of such "
                f"runs: {named}.",
            )
        )
    killed = [run for run in scope.dataset.runs if run.killed]
    holding = sum(
        1 for run in killed if run.incumbent and scope.objective in run.incumbent.objectives
    )
    rescored = holding if config.timeout_score == "incumbent" else 0
    if counts["failed"] or counts["infeasible"]:
        exception = f", except {rescored} killed runs scored by their incumbent" if rescored else ""
        kinds, meanings = _failure_kinds(scope.dataset)
        breakdown = (
            f" The failed runs by solver and by the kind of error of their last attempt: {kinds}. "
            f"{meanings}"
            if kinds
            else ""
        )
        found.append(
            Limitation(
                "failures",
                f"{counts['failed']} runs failed and {counts['infeasible']} completed without a "
                f"feasible schedule; each counts as the worst value{exception}.{breakdown}",
            )
        )
    unstarted = sum(1 for run in scope.dataset.runs if run.unstarted)
    if unstarted:
        found.append(
            Limitation(
                "startup",
                f"{unstarted} of the failed runs ended at the start-up limit: their worker did "
                "not begin the solve in time, which says nothing about their solver. They "
                "count as the worst value until another attempt completes; running the "
                "campaign again with retry_failed (--retry-failed on the command line) gives "
                "them one.",
            )
        )
    if killed:
        scored = (
            "a run holding one is scored by it, a value its solver reported and no schedule "
            "confirms, and the others count as the worst value"
            if rescored
            else "each counts as the worst value"
        )
        late = sum(1 for run in killed if run.late_incumbent)
        found.append(
            Limitation(
                "timeouts",
                f"{len(killed)} runs were killed at their hard limit and never completed, "
                f"{holding} of them holding a feasible incumbent when their time limit passed; "
                f"with timeout_score '{config.timeout_score}' {scored}. {late} of the killed "
                "runs reported an incumbent after their time limit, which no score uses.",
            )
        )
    records = [run.record for run in scope.dataset.runs if run.record is not None]
    outcomes = fairness.assess_outcomes(
        scope.dataset.plan, records, overrun_tolerance=config.overrun_tolerance
    ).protocol(scope.protocol)
    found.extend(_fairness_limitations(scope, outcomes))
    found.extend(_time_limitations(scope, config, outcomes))
    found.extend(_check_limitations(scope))
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


_ITEMS_SHOWN = 12
"""Cells or items a limitation names before it counts the rest."""

_OWN_FINDINGS = frozenset(
    {
        "runs-unattempted",
        "runs-failed",
        "stop-reasons-diverge",
        "budget-overrun",
        "replicates-below-floor",
    }
)
"""The findings of the campaign check that the analysis states in limitations of its own.

They are, in the same order, the topics ``campaign``, ``failures``,
``stop-reasons``, ``budget``, and ``replicates``, which count over the
analysed solvers and problems and with the settings of the analysis.
"""


def _shown(items: Sequence[str]) -> str:
    """Join the first items of a list, and count the others."""
    more = len(items) - _ITEMS_SHOWN
    return "; ".join(items[:_ITEMS_SHOWN]) + (f"; and {more} more" if more > 0 else "")


def _check_limitations(scope: Scope) -> list[Limitation]:
    """Carry what the check of the campaign directory found and the analysis does not state.

    The dataset holds the report of
    :func:`~dispatchatlas.campaigns.checking.check_campaign` when it was
    loaded from a directory. Each of its blocking findings and limitations
    becomes a limitation whose topic is the code of the finding, unless it
    concerns another protocol than the analysed one or is one the analysis
    states itself (``_OWN_FINDINGS``). Notes are left out. A finding
    covers the whole campaign directory, whatever solvers and problems the
    analysis selects.

    Two findings are carried without the items that the analysis names in
    a limitation of its own, and are left out when no item remains: of the
    cells with a record for only some of their runs, the cells with a run
    without a record, which the limitation ``campaign`` names; and of the
    solvers without a record, those whose every run failed, which the
    limitation ``failures`` counts by solver.
    """
    report = scope.dataset.check
    if report is None:
        return []
    runs = scope.dataset.runs
    statuses: dict[str, set[str]] = {}
    for run in runs:
        statuses.setdefault(run.solver, set()).add(run.status)
    named = {
        "cells-partial": tuple(
            {f"{run.solver} on {run.problem}: " for run in runs if run.status == "missing"}
        ),
        "solver-without-records": tuple(
            f"{solver}: " for solver, held in statuses.items() if held == {"failed"}
        ),
    }
    found: list[Limitation] = []
    for finding in report.findings:
        if finding.severity == "note" or finding.code in _OWN_FINDINGS:
            continue
        if finding.protocol not in (None, scope.protocol):
            continue
        items = finding.items
        if finding.code in named:
            # Each item of these findings begins with its cell or its solver and a colon.
            items = tuple(item for item in items if not item.startswith(named[finding.code]))
            if not items:
                continue
        which = (
            "concerned" if len(items) == len(finding.items) else "that no other limitation names"
        )
        listed = f" The items {which}: {_shown(items)}." if items else ""
        weight = (
            " The check of the campaign directory counts this as blocking: the results are "
            "not ready to be used."
            if finding.severity == "blocking"
            else ""
        )
        found.append(Limitation(finding.code, f"{finding.message}{listed}{weight}"))
    return found


_FAILURE_KINDS: Mapping[str, str] = {
    "exception": "the solver, its warm-up, or the checks of its result raised an exception",
    "timeout": "the engine killed the worker at the hard limit of the run",
    "startup": "the engine killed a worker that had not begun the solve at the start-up limit",
    "worker-exit": (
        "the worker process ended during the run, as after a crash in native code or when "
        "memory runs out, and the engine does not know the cause"
    ),
    "transfer": (
        "the run could not be sent to a worker or rebuilt in it, as when its solver cannot be "
        "pickled"
    ),
}
"""What each kind of error says about who ended a failed attempt."""


def _failure_kinds(dataset: Dataset) -> tuple[str, str]:
    """Count the failed runs of each solver by the error kind of their last attempt.

    Returns:
        The counts in words, and one sentence that states what each kind
        among them means. Both are empty when no failed run has a failure
        record.
    """
    counts: dict[str, Counter[str]] = {}
    for run in dataset.runs:
        if run.status == "failed" and run.failures:
            counts.setdefault(run.solver, Counter())[run.failures[-1].error.kind.value] += 1
    if not counts:
        return "", ""
    breakdown = "; ".join(
        f"{entry.name} "
        + ", ".join(f"{number} {kind}" for kind, number in sorted(counts[entry.name].items()))
        for entry in dataset.solvers
        if entry.name in counts
    )
    present = {kind for counter in counts.values() for kind in counter}
    meanings = "; ".join(
        f"{kind}: {meaning}" for kind, meaning in _FAILURE_KINDS.items() if kind in present
    )
    return breakdown, f"What each kind means: {meanings}."


def _fairness_limitations(scope: Scope, outcomes: ProtocolOutcomes) -> list[Limitation]:
    """Report what the plan notes on the protocol and how its runs ended.

    Each ``plan`` limitation is a note of the plan's fairness assessment on
    the analysed protocol and solvers, such as iterative solvers declared
    deterministic that run once under a time limit. A conditional note of the
    plan becomes one only for the solvers whose condition the outcomes meet:
    solvers declared deterministic that are not iterative and run once are
    reported when the time limit ended one of their completed runs. The
    ``stop-reasons`` limitation counts, for each iterative solver, the
    completed runs by the reason they ended for, when those reasons differ
    within the protocol.
    """
    found: list[Limitation] = []
    selected = [entry.name for entry in scope.dataset.solvers]
    protocol = scope.dataset.protocol(scope.protocol)
    found.extend(
        Limitation("plan", note[0].upper() + note[1:] + ".")
        for note in fairness.notes(scope.dataset.plan, protocol, selected)
    )
    cut = "; ".join(
        f"{name} " + ", ".join(f"{count} {reason}" for reason, count in reasons.items())
        for name, reasons in outcomes.unreplicated.items()
        if name in selected
    )
    if cut:
        found.append(
            Limitation(
                "plan",
                f"Under protocol {scope.protocol}, the time limit ended completed runs of "
                f"solvers declared deterministic that run once per problem: {cut}. A run that "
                "the time limit ends depends on the speed and load of the machine, and one run "
                "does not show by how much; replicate_timed = true in the campaign file gives "
                "such a solver every replicate.",
            )
        )
    if outcomes.divergent:
        counted = "; ".join(
            f"{name} " + ", ".join(f"{count} {reason}" for reason, count in reasons.items())
            for name, reasons in outcomes.stop_reasons.items()
        )
        found.append(
            Limitation(
                "stop-reasons",
                f"Under protocol {scope.protocol}, the completed runs of the iterative solvers "
                f"ended for different reasons: {counted}. Runs that ended for different "
                "reasons did not spend the same budget, in one solver or across several, so "
                "the protocol did not bound them alike; their results are compared as they "
                "are.",
            )
        )
    return found


def _time_limitations(
    scope: Scope, config: AnalysisConfig, outcomes: ProtocolOutcomes
) -> list[Limitation]:
    """Compare the completed runs of a protocol with a time limit against that limit.

    The ``budget`` limitation names the solvers whose completed runs took
    more wall time than the limit allows with ``overrun_tolerance``, as the
    assessment of the outcomes counts them, by the rule of the engine's note
    on a run (:func:`~dispatchatlas.campaigns.execution.overran`). The
    ``clock`` limitation counts the completed runs whose records state that
    the solver's clock started at the start of the attempt, when only some
    of them do. The
    ``timing`` limitation states under which load the completed runs that
    lasted at least half of the time limit executed: the spread of their
    processor share and of the number of runs executing at once. Shorter
    runs are left out of it: the clock did not bound them, and the processor
    time of a run of a few hundredths of a second is coarser than the run.
    """
    limit = scope.dataset.protocol(scope.protocol).stop.time_limit
    completed = [run for run in scope.dataset.runs if run.record is not None]
    if limit is None or not completed:
        return []
    found: list[Limitation] = []
    late = [
        f"{name} in {item.over} of {item.runs} (up to {item.worst:.3g} times the limit)"
        for name, item in outcomes.overruns.items()
    ]
    if late:
        found.append(
            Limitation(
                "budget",
                f"Under protocol {scope.protocol}, whose time limit is {limit:g} s, completed "
                f"runs took more wall time than the limit with overrun_tolerance "
                f"{config.overrun_tolerance:g} allows: " + "; ".join(late) + ". These runs "
                "searched for longer than the protocol states, and their results are compared "
                "as they are.",
            )
        )
    shared = sum(
        1 for run in completed if run.record is not None and run.record.provenance.shared_clock
    )
    if 0 < shared < len(completed):
        found.append(
            Limitation(
                "clock",
                f"Under protocol {scope.protocol}, {shared} of the {len(completed)} completed "
                "runs state that the clock of their solver started with the engine's, at the "
                f"start of the attempt, and {len(completed) - shared} do not state where it "
                "started. A solver whose clock starts later, after its set-up, has the whole "
                "time limit for its search and reports an elapsed time without that set-up, so "
                "the time limit and the elapsed times of the two groups may not measure the "
                "same thing.",
            )
        )
    # A run that ended long before the limit was not bounded by the clock, and
    # its processor time is too short to measure against its wall time.
    timed = [
        run
        for run in completed
        if run.record is not None and run.record.provenance.wall_seconds >= limit / 2.0
    ]
    shares = {
        entry.name: [
            share
            for run in timed
            if run.solver == entry.name and (share := run.cpu_share) is not None
        ]
        for entry in scope.dataset.solvers
    }
    pooled = [share for values in shares.values() for share in values]
    concurrency = [
        run.record.provenance.concurrent
        for run in timed
        if run.record is not None and run.record.provenance.concurrent is not None
    ]
    parts: list[str] = []
    if pooled:
        part = (
            "the processor share of a run, the processor seconds of its own process over its "
            f"wall seconds, lies between {min(pooled):.2f} and {max(pooled):.2f} with median "
            f"{statistics.median(pooled):.2f}"
        )
        medians = sorted(
            (statistics.median(values), name) for name, values in shares.items() if values
        )
        if len(medians) > 1:
            (low, lowest), (high, highest) = medians[0], medians[-1]
            part += (
                f", and its median per solver between {low:.2f} ({lowest}) and "
                f"{high:.2f} ({highest})"
            )
        parts.append(part)
    if concurrency:
        parts.append(
            f"between {min(concurrency)} and {max(concurrency)} runs executed at once, with "
            f"median {statistics.median(concurrency):g}"
        )
    if parts:
        stop = scope.dataset.protocol(scope.protocol).stop
        if stop.max_evaluations is None and stop.max_iterations is None:
            opening = (
                f"Protocol {scope.protocol} gives every run the same wall-clock time, so a run "
                "searches as far as the processor time it received in that time allows."
            )
        else:
            opening = (
                f"Protocol {scope.protocol} bounds a run by a time limit of {limit:g} s beside "
                "its other limits, so a run that the time limit stops searched as far as the "
                "processor time it received allows."
            )
        found.append(
            Limitation(
                "timing",
                f"{opening} Over the {len(timed)} completed runs that lasted at least half of "
                "the time limit, " + "; ".join(parts) + ".",
            )
        )
    return found


_OMNIBUS_NAMES: Mapping[str, str] = {"friedman": "Friedman", "iman-davenport": "Iman-Davenport"}
