"""The results of an analysis: frozen records that tables and figures render.

Directional quantities favour the left solver when positive, or above 0.5 for
the win share, whatever the objective's sense: an *advantage* is how much
better the left solver's value is, in objective units.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.stats.corrections import Correction
from dispatchatlas.analysis.stats.nonparametric import TestMethod
from dispatchatlas.analysis.stats.ranks import Sense
from dispatchatlas.core.errors import ConfigError

__all__ = [
    "BaselineComparison",
    "CellSummary",
    "CriticalDifference",
    "Limitation",
    "OmnibusResult",
    "PairwiseComparison",
    "SolverSummary",
    "Summary",
]


@dataclass(frozen=True, slots=True)
class Limitation:
    """A caveat that qualifies the results.

    Attributes:
        topic: What it concerns, for example ``"pairwise"`` or ``"replicates"``.
        message: One or two sentences stating the caveat.
    """

    topic: str
    message: str


@dataclass(frozen=True, slots=True)
class CellSummary:
    """The runs of one solver on one problem.

    Attributes:
        solver: Solver name.
        problem: Problem id.
        planned: Runs the plan holds for the cell.
        succeeded: Runs that completed with a feasible schedule.
        score: The value the comparisons use: the configured statistic of
            the planned runs, a failed run counting as the worst value, so it
            is infinite when failures decide it.
        mean: Mean value of the succeeded runs, or ``None`` without one.
        median: Median value of the succeeded runs.
        std: Sample standard deviation of the succeeded runs, with at least two.
        best: Best value of the succeeded runs.
        mean_low: Lower confidence bound of the mean.
        mean_high: Upper confidence bound of the mean.
    """

    solver: str
    problem: str
    planned: int
    succeeded: int
    score: float
    mean: float | None
    median: float | None
    std: float | None
    best: float | None
    mean_low: float | None
    mean_high: float | None


@dataclass(frozen=True, slots=True)
class SolverSummary:
    """The reliability and average rank of one solver.

    Attributes:
        solver: Solver name.
        planned: Planned runs over the analysed problems.
        succeeded: Runs that completed with a feasible schedule.
        infeasible: Completed runs without a feasible schedule.
        failed: Runs whose attempts all failed.
        missing: Runs without a record yet.
        problems: Problems the plan pairs the solver with.
        average_rank: Mean rank over the complete blocks, rank 1 best, or
            ``None`` when no complete block exists.
    """

    solver: str
    planned: int
    succeeded: int
    infeasible: int
    failed: int
    missing: int
    problems: int
    average_rank: float | None

    @property
    def success_rate(self) -> float:
        """Share of the planned runs that succeeded; 0 without planned runs."""
        return self.succeeded / self.planned if self.planned else 0.0


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    """A paired comparison of two solvers over the problems they share.

    Each problem contributes one pair: the two solvers' cell scores. The
    Wilcoxon signed-rank test compares them, and the effect sizes and
    intervals beside the p-value say how large the difference is.

    Attributes:
        left: The solver the directional fields favour when positive.
        right: The other solver.
        pairs: Shared problems.
        wins: Problems the left solver wins.
        losses: Problems the left solver loses.
        ties: Problems where the scores tie.
        win_share: ``(wins + ties / 2) / pairs``.
        win_share_low: Lower bootstrap bound of the win share.
        win_share_high: Upper bootstrap bound of the win share.
        median_advantage: Median over problems of how much better the left
            score is, in objective units, or ``None`` when undefined.
        mean_advantage: Mean advantage, or ``None`` when a failure makes an
            advantage infinite.
        advantage_low: Lower bootstrap bound of the mean advantage.
        advantage_high: Upper bootstrap bound of the mean advantage.
        cohens_dz: Mean advantage over its standard deviation, or ``None``.
        statistic: Wilcoxon statistic ``min(W+, W-)``.
        p_value: Two-sided Wilcoxon p-value.
        adjusted_p_value: The p-value adjusted over the family of comparisons.
        method: How the p-value was obtained.
        confidence: Coverage of the intervals.
    """

    left: str
    right: str
    pairs: int
    wins: int
    losses: int
    ties: int
    win_share: float
    win_share_low: float
    win_share_high: float
    median_advantage: float | None
    mean_advantage: float | None
    advantage_low: float | None
    advantage_high: float | None
    cohens_dz: float | None
    statistic: float
    p_value: float
    adjusted_p_value: float
    method: TestMethod
    confidence: float

    def swapped(self) -> PairwiseComparison:
        """Return the same comparison seen from the right solver."""
        return replace(
            self,
            left=self.right,
            right=self.left,
            wins=self.losses,
            losses=self.wins,
            win_share=1.0 - self.win_share,
            win_share_low=1.0 - self.win_share_high,
            win_share_high=1.0 - self.win_share_low,
            median_advantage=_negated(self.median_advantage),
            mean_advantage=_negated(self.mean_advantage),
            advantage_low=_negated(self.advantage_high),
            advantage_high=_negated(self.advantage_low),
            cohens_dz=_negated(self.cohens_dz),
        )


@dataclass(frozen=True, slots=True)
class OmnibusResult:
    """The Friedman and Iman-Davenport tests over the complete blocks.

    Attributes:
        blocks: Complete blocks: problems every analysed solver is planned on.
        groups: Solvers.
        friedman_statistic: Friedman's ``Q``.
        friedman_p_value: Chi-square p-value of ``Q``.
        iman_davenport_statistic: Iman and Davenport's ``F``.
        iman_davenport_p_value: F p-value.
        kendalls_w: Kendall's coefficient of concordance, the effect size.
        kendalls_w_low: Lower bootstrap bound of ``W``, resampling blocks.
        kendalls_w_high: Upper bootstrap bound of ``W``.
        confidence: Coverage of the interval.
        test: The test whose p-value decides significance.
        alpha: Significance level.
        significant: Whether that p-value is below ``alpha``.
    """

    blocks: int
    groups: int
    friedman_statistic: float
    friedman_p_value: float
    iman_davenport_statistic: float
    iman_davenport_p_value: float
    kendalls_w: float
    kendalls_w_low: float
    kendalls_w_high: float
    confidence: float
    test: str
    alpha: float
    significant: bool

    @property
    def p_value(self) -> float:
        """The p-value of the deciding test."""
        return self.friedman_p_value if self.test == "friedman" else self.iman_davenport_p_value


@dataclass(frozen=True, slots=True)
class CriticalDifference:
    """The Nemenyi post-hoc comparison of average ranks.

    Attributes:
        value: The critical difference.
        alpha: Significance level.
        groups: Solvers.
        blocks: Complete blocks.
        average_ranks: Each solver's average rank, best first.
        significant_pairs: Pairs, best first, whose ranks differ by more than
            the critical difference.
        cliques: Maximal runs of solvers, in rank order, whose ranks lie
            within the critical difference of each other, the bars of a
            critical-difference diagram. Singletons are included.
    """

    value: float
    alpha: float
    groups: int
    blocks: int
    average_ranks: tuple[tuple[str, float], ...]
    significant_pairs: tuple[tuple[str, str], ...]
    cliques: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Summary:
    """Every result of :func:`~dispatchatlas.analysis.summary.summarize` for one protocol.

    Attributes:
        campaign: Campaign id.
        protocol: The analysed protocol.
        objective: The analysed objective.
        sense: Its sense.
        config: The configuration used.
        solvers: Analysed solvers, in campaign order.
        problems: Analysed problems, in plan order.
        blocks: The problems every analysed solver is planned on.
        cells: One summary per planned solver and problem.
        solver_summaries: One summary per solver, in campaign order.
        pairwise: One comparison per pair of solvers sharing a problem.
        omnibus: The omnibus tests, or ``None`` when they do not apply.
        critical_difference: The post-hoc test, or ``None``.
        limitations: The caveats that qualify these results.
    """

    campaign: str
    protocol: str
    objective: str
    sense: Sense
    config: AnalysisConfig
    solvers: tuple[str, ...]
    problems: tuple[str, ...]
    blocks: tuple[str, ...]
    cells: tuple[CellSummary, ...]
    solver_summaries: tuple[SolverSummary, ...]
    pairwise: tuple[PairwiseComparison, ...]
    omnibus: OmnibusResult | None
    critical_difference: CriticalDifference | None
    limitations: tuple[Limitation, ...]

    def comparison(self, left: str, right: str) -> PairwiseComparison:
        """Return the comparison of two solvers, oriented towards ``left``.

        Raises:
            ConfigError: If the two solvers were not compared.
        """
        for row in self.pairwise:
            if (row.left, row.right) == (left, right):
                return row
            if (row.left, row.right) == (right, left):
                return row.swapped()
        raise ConfigError(f"solvers {left!r} and {right!r} were not compared")


@dataclass(frozen=True)
class BaselineComparison:
    """One solver compared with each of several baselines.

    Attributes:
        focal: The solver under study; every row has it on the left.
        baselines: The baselines, in campaign order.
        protocol: The analysed protocol.
        objective: The analysed objective.
        sense: Its sense.
        correction: The correction applied over the rows.
        alpha: Significance level.
        rows: One comparison per baseline, adjusted over this family only.
        better_than_all: Whether every row is significant at ``alpha`` after
            correction and has a win share above 0.5.
        limitations: The caveats of the underlying summary.
    """

    focal: str
    baselines: tuple[str, ...]
    protocol: str
    objective: str
    sense: Sense
    correction: Correction
    alpha: float
    rows: tuple[PairwiseComparison, ...]
    better_than_all: bool
    limitations: tuple[Limitation, ...]


def _negated(value: float | None) -> float | None:
    return None if value is None else -value
