"""Performance profiles, data profiles, and runtime distributions.

Three views of how often each solver reaches good results over a common
problem set, each a step function from 0 to 1:

- the **performance profile** of Dolan and Moré (2002): the share of problems
  on which a solver's value is within a factor ``tau`` of the best solver's;
- the **data profile** of Moré and Wild (2009): the share of problems a
  solver solves to accuracy ``tau`` within a budget of evaluations;
- the **runtime distribution** of Hansen et al. (2022): the share of
  (problem, quality target) pairs a solver's runs reach within a budget.

Every profile keeps failures in its denominator (see
:mod:`~dispatchatlas.analysis.figures.frames`): a failed run never reaches a
ratio, an accuracy, or a target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate

from dispatchatlas.analysis.figures.axes import Axis, log_axis, numeric_axis
from dispatchatlas.analysis.figures.chart import Series, label_for, placeholder, xy_chart
from dispatchatlas.analysis.figures.drawing import Figure
from dispatchatlas.analysis.figures.frames import (
    Aggregation,
    Budget,
    check_aggregation,
    check_budget,
    lower_median,
    observe,
    quality_ranges,
)
from dispatchatlas.analysis.figures.theme import DEFAULT_THEME, Theme
from dispatchatlas.analysis.stats.ranks import ties
from dispatchatlas.core.errors import ConfigError

__all__ = [
    "GOULD_SCOTT_CAVEAT",
    "PROFILE_THRESHOLDS",
    "QUALITY_TARGETS",
    "DataCurve",
    "DataProfile",
    "PerformanceProfile",
    "ProfileCurve",
    "RuntimeCurve",
    "RuntimeDistribution",
    "data_profile",
    "performance_profile",
    "plot_data_profile",
    "plot_performance_profile",
    "plot_runtime_distribution",
    "runtime_distribution",
]

PROFILE_THRESHOLDS: tuple[float, ...] = (1.0, 1.05, 1.1, 1.25, 1.5, 2.0)
"""Ratio thresholds at which :meth:`PerformanceProfile.at` reads the profiles by default."""

GOULD_SCOTT_CAVEAT = (
    "Caveat: profiles rank each solver only against the per-problem best; "
    "the relative order of non-best solvers is not implied (Gould & Scott 2016)."
)
"""The caution of Gould and Scott (2016), shown on profiles of more than two solvers."""

QUALITY_TARGETS: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0)
"""Quality targets of the runtime distribution, as shares of a problem's span of values."""

_LOG_SCALE_ABOVE = 4.0

_AGGREGATED = {"median": "median run", "best": "best run"}


# ------------------------------------------------------------ performance


@dataclass(frozen=True)
class ProfileCurve:
    """The performance ratios of one solver.

    Attributes:
        solver: Solver name.
        ratios: One ratio per problem of the profile, ascending; infinite
            where the solver failed.
    """

    solver: str
    ratios: tuple[float, ...]

    def share(self, tau: float) -> float:
        """Return the profile at ``tau``: the share of problems with a ratio of at most ``tau``."""
        if not self.ratios:
            return 0.0
        hits = sum(1 for ratio in self.ratios if math.isfinite(ratio) and ratio <= tau)
        return hits / len(self.ratios)

    @property
    def solved(self) -> int:
        """Problems with a finite ratio."""
        return sum(1 for ratio in self.ratios if math.isfinite(ratio))


@dataclass(frozen=True)
class PerformanceProfile:
    """Dolan-Moré performance profiles of every solver over a common problem set.

    Attributes:
        curves: One curve per solver, in campaign order.
        problems: The problems of the profile.
        excluded: Problems left out because their best value is not
            positive, where ratios are undefined.
        aggregation: How the runs of a cell were aggregated.
        objective: Name of the objective compared, empty when unknown.
    """

    curves: tuple[ProfileCurve, ...]
    problems: tuple[str, ...]
    excluded: tuple[str, ...]
    aggregation: Aggregation
    objective: str = ""

    def at(self, thresholds: Sequence[float] = PROFILE_THRESHOLDS) -> dict[str, tuple[float, ...]]:
        """Return each solver's profile read at the given ratio thresholds."""
        return {
            curve.solver: tuple(curve.share(tau) for tau in thresholds) for curve in self.curves
        }


def performance_profile(source: object, *, aggregation: str = "median") -> PerformanceProfile:
    """Return the performance profiles of Dolan and Moré (2002).

    Each solver's runs on a problem become one value, the median or the best
    run, a failed run counting as infinitely bad. The ratio of a solver on a
    problem is its value over the best solver's value (the best value over
    the solver's value when the objective is maximised), 1 for the best
    solver and infinite for a failure. Values within the analysis tie
    tolerance of the best count as ratio 1. Ratios need positive values, so
    a problem whose best value is zero or negative is left out for every
    solver and reported.

    Args:
        source: Runs of one protocol, anything :func:`observe` accepts.
        aggregation: ``"median"`` or ``"best"``, see :data:`Aggregation`.

    Returns:
        The profiles.

    Raises:
        ConfigError: If the runs span several protocols or the aggregation
            is unknown.
    """
    chosen = check_aggregation(aggregation)
    observations = observe(source).comparable()
    ratios: dict[str, list[float]] = {solver: [] for solver in observations.solvers}
    kept: list[str] = []
    excluded: list[str] = []
    for problem in observations.problems:
        sense = observations.sense(problem)
        scores = {
            solver: observations.cell_score(solver, problem, chosen)
            for solver in observations.solvers
        }
        finite = [score for score in scores.values() if math.isfinite(score)]
        best = min(finite, default=math.inf)
        natural_best = best if sense == "min" else -best
        if finite and natural_best <= 0.0:
            excluded.append(problem)
            continue
        kept.append(problem)
        for solver, score in scores.items():
            ratios[solver].append(_ratio(score, best, sense))
    curves = tuple(ProfileCurve(solver, tuple(sorted(values))) for solver, values in ratios.items())
    label = observations.objective_label()
    return PerformanceProfile(curves, tuple(kept), tuple(excluded), chosen, label)


def _ratio(score: float, best: float, sense: str) -> float:
    """Return the performance ratio of a score against the best, both in minimisation form."""
    if not (math.isfinite(score) and math.isfinite(best)):
        return math.inf
    if ties(score, best):
        return 1.0
    if sense == "min":
        return score / best
    value = -score
    return -best / value if value > 0.0 else math.inf


def plot_performance_profile(
    profile: PerformanceProfile,
    *,
    labels: Mapping[str, str] | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Draw performance profiles as step functions of the ratio ``tau``.

    The ratio axis runs from 1 to the largest finite ratio, on a logarithmic
    scale when that exceeds 4. With more than two solvers the figure carries
    the caveat of Gould and Scott (2016) as a visible note.

    Args:
        profile: The profiles.
        labels: Display names of solvers; a solver without one shows its name.
        theme: The appearance.

    Returns:
        The figure, or a placeholder stating why there is nothing to draw.
    """
    active = theme or DEFAULT_THEME
    name, title = "performance-profile", "Performance profile"
    if not profile.curves or not profile.problems:
        reason = "No problem with a positive best value to compare solvers on."
        return placeholder(name, title, reason, active)
    finite = [ratio for curve in profile.curves for ratio in curve.ratios if math.isfinite(ratio)]
    top = max([1.05, *finite])
    x = (
        log_axis("Performance ratio \u03c4", [1.0, top], low=1.0)
        if top > _LOG_SCALE_ABOVE
        else numeric_axis("Performance ratio \u03c4", [1.0, top * 1.02], low=1.0)
    )
    y = numeric_axis("Share of problems with ratio \u2264 \u03c4", [], low=0.0, high=1.0)
    series = []
    for index, curve in enumerate(profile.curves):
        knots = sorted({ratio for ratio in curve.ratios if math.isfinite(ratio) and ratio > 1.0})
        points = [(1.0, curve.share(1.0))] + [(tau, curve.share(tau)) for tau in knots]
        series.append(
            Series(
                label_for(curve.solver, labels),
                tuple(points),
                active.series(index),
                kind="step",
                until=x.high,
            )
        )
    measure = profile.objective or "value"
    notes = [
        f"Each solver's {measure} on a problem is that of its "
        f"{_AGGREGATED[profile.aggregation]}; a failed run counts as infinitely bad, so it never "
        "comes within any ratio (Dolan and Mor\u00e9 2002).",
    ]
    if profile.excluded:
        notes.append(
            f"{len(profile.excluded)} problem(s) left out because their best value is not "
            "positive, where ratios are undefined: " + ", ".join(profile.excluded) + "."
        )
    if len(profile.curves) > 2:
        notes.append(GOULD_SCOTT_CAVEAT)
    readings = "; ".join(
        f"{label_for(solver, labels)} " + ", ".join(f"{share:.2f}" for share in shares)
        for solver, shares in profile.at().items()
    )
    description = (
        f"Performance profiles of {len(profile.curves)} solvers over "
        f"{len(profile.problems)} problems. Share of problems within ratio "
        + ", ".join(f"{tau:g}" for tau in PROFILE_THRESHOLDS)
        + f" of the best: {readings}. "
        + " ".join(notes)
    )
    return xy_chart(name, title, description, x=x, y=y, series=series, notes=notes, theme=active)


# ------------------------------------------------------------------ data


@dataclass(frozen=True)
class DataCurve:
    """The budgets one solver needs to solve each problem.

    Attributes:
        solver: Solver name.
        budgets: One budget per problem of the profile, ascending, in the
            profile's unit; infinite where the solver never solves it.
    """

    solver: str
    budgets: tuple[float, ...]

    def share(self, kappa: float) -> float:
        """Return the share of problems solved within budget ``kappa``; failures never count."""
        if not self.budgets:
            return 0.0
        hits = sum(1 for budget in self.budgets if math.isfinite(budget) and budget <= kappa)
        return hits / len(self.budgets)

    @property
    def solved(self) -> int:
        """Problems solved within the budget limit."""
        return sum(1 for budget in self.budgets if math.isfinite(budget))

    @property
    def cells(self) -> int:
        """Problems of the profile, solved or not."""
        return len(self.budgets)


@dataclass(frozen=True)
class DataProfile:
    """Moré-Wild data profiles of every solver.

    Attributes:
        curves: One curve per solver, in campaign order.
        problems: The problems of the profile.
        accuracy: The accuracy ``tau`` of the convergence test.
        unit: What a unit of budget is: ``"evaluations"``, or evaluations
            divided by the number of tasks plus one.
        knots: The finite budgets where some curve steps, ascending.
        aggregation: How the runs of a cell were aggregated.
    """

    curves: tuple[DataCurve, ...]
    problems: tuple[str, ...]
    accuracy: float
    unit: str
    knots: tuple[float, ...]
    aggregation: Aggregation = "median"

    def points(self, solver: str) -> tuple[tuple[float, float], ...]:
        """Return a solver's ``(budget, share)`` pairs at every knot.

        Raises:
            ConfigError: If the profile has no such solver.
        """
        for curve in self.curves:
            if curve.solver == solver:
                return tuple((knot, curve.share(knot)) for knot in self.knots)
        raise ConfigError(f"the data profile has no solver {solver!r}")


def data_profile(
    source: object,
    *,
    accuracy: float = 1e-3,
    sizes: Mapping[str, int] | None = None,
    budget_limit: float | None = None,
    aggregation: str = "median",
) -> DataProfile:
    """Return the data profiles of Moré and Wild (2009).

    On each problem, ``f_L`` is the best value any succeeded run reached,
    over its trace and its final value. A run solves the problem at the
    first feasible trace point whose best value so far passes the inclusive
    convergence test ``f <= f_L + tau (f_0 - f_L)``, where ``f_0`` is the
    run's first feasible value (mirrored for a maximised objective); its
    budget is the number of evaluations recorded at that point. A run that
    never passes the test, or did not succeed, never solves the problem. A
    solver's budget on a problem is that of its median run, the lower of
    the two middle runs for an even count, or of its best run. Budgets are
    divided by ``n_p + 1``, where ``n_p`` is the problem's number of tasks,
    as Moré and Wild divide by the dimension plus one; a problem of unknown
    size keeps undivided budgets. A solver the plan excludes from a problem
    never solves it.

    Args:
        source: Runs of one protocol, anything :func:`observe` accepts.
        accuracy: The accuracy ``tau``, strictly between 0 and 1.
        sizes: Number of tasks of each problem. Defaults to the sizes the
            source knows, which an analysis dataset provides.
        budget_limit: Largest budget, in the profile's unit, within which a
            problem can count as solved.
        aggregation: ``"median"`` or ``"best"``, see :data:`Aggregation`.

    Returns:
        The profiles.

    Raises:
        ConfigError: If ``accuracy`` is not strictly between 0 and 1, the
            budget limit is not positive, a size is negative, or the runs
            span several protocols.
    """
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        raise ConfigError(f"the accuracy must be a number, got {accuracy!r}")
    if not 0.0 < accuracy < 1.0:
        raise ConfigError(f"data profiles need an accuracy strictly within (0, 1), got {accuracy}")
    if budget_limit is not None and not (budget_limit > 0.0 and math.isfinite(budget_limit)):
        raise ConfigError(f"the budget limit must be positive and finite, got {budget_limit}")
    chosen = check_aggregation(aggregation)
    observations = observe(source).comparable()
    known = dict(observations.sizes if sizes is None else sizes)
    for problem, size in known.items():
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ConfigError(f"the size of problem {problem!r} must be a count, got {size!r}")
    ranges = quality_ranges(observations, "evaluations")
    budgets: dict[str, list[float]] = {solver: [] for solver in observations.solvers}
    for problem in observations.problems:
        group = known[problem] + 1.0 if problem in known else 1.0
        span = ranges.get(problem)
        for solver in observations.solvers:
            times = []
            for run in observations.cell(solver, problem):
                progress = observations.progress(run, "evaluations")
                if span is None or not progress:
                    times.append(math.inf)
                    continue
                threshold = span.best + accuracy * (progress[0][1] - span.best)
                times.append(
                    next(
                        (spent for spent, value in progress if _passes(value, threshold)),
                        math.inf,
                    )
                )
            if not times:
                kappa = math.inf
            else:
                kappa = (min(times) if chosen == "best" else lower_median(times)) / group
            if budget_limit is not None and kappa > budget_limit:
                kappa = math.inf
            budgets[solver].append(kappa)
    curves = tuple(DataCurve(solver, tuple(sorted(values))) for solver, values in budgets.items())
    knots = tuple(sorted({b for curve in curves for b in curve.budgets if math.isfinite(b)}))
    divided = [problem in known for problem in observations.problems]
    if divided and all(divided):
        unit = "evaluations / (tasks + 1)"
    elif any(divided):
        unit = "evaluations / (tasks + 1), or evaluations where the size is unknown"
    else:
        unit = "evaluations"
    return DataProfile(curves, observations.problems, float(accuracy), unit, knots, chosen)


def _passes(value: float, threshold: float) -> bool:
    return value <= threshold or ties(value, threshold)


def plot_data_profile(
    profile: DataProfile,
    *,
    labels: Mapping[str, str] | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Draw data profiles as step functions of the budget.

    Args:
        profile: The profiles.
        labels: Display names of solvers; a solver without one shows its name.
        theme: The appearance.

    Returns:
        The figure, or a placeholder when no solver solved any problem.
    """
    active = theme or DEFAULT_THEME
    name, title = "data-profile", "Data profile"
    if not profile.curves or not profile.knots:
        return placeholder(name, title, "No solver solved a problem to the accuracy.", active)
    positive = [knot for knot in profile.knots if knot > 0.0]
    label = f"Budget \u03ba ({profile.unit})"
    if positive and max(positive) / min(positive) > 100.0:
        x: Axis = log_axis(label, positive)
    else:
        x = numeric_axis(label, [*profile.knots, 0.0], low=0.0)
    y = numeric_axis("Share of problems solved", [], low=0.0, high=1.0)
    series = []
    for index, curve in enumerate(profile.curves):
        start = x.low
        points = [(start, curve.share(start))]
        points += [(knot, curve.share(knot)) for knot in profile.knots if knot > start]
        series.append(
            Series(
                label_for(curve.solver, labels),
                tuple(points),
                active.series(index),
                kind="step",
                until=x.high,
            )
        )
    notes = [
        f"A problem is solved at the first evaluation whose best value comes within "
        f"\u03c4 = {profile.accuracy:g} of the gap between the run's first value and the "
        "best value any run found (Mor\u00e9 and Wild 2009); failed runs never solve it.",
        f"Each solver's budget on a problem is that of its {_AGGREGATED[profile.aggregation]}.",
    ]
    description = (
        f"Data profiles of {len(profile.curves)} solvers over {len(profile.problems)} "
        "problems. Problems solved: "
        + "; ".join(
            f"{label_for(curve.solver, labels)} {curve.solved} of {curve.cells}"
            for curve in profile.curves
        )
        + ". "
        + " ".join(notes)
    )
    return xy_chart(name, title, description, x=x, y=y, series=series, notes=notes, theme=active)


# --------------------------------------------------------------- runtime


@dataclass(frozen=True)
class RuntimeCurve:
    """The share of (problem, target) pairs one solver's runs reach, by budget.

    Attributes:
        solver: Solver name.
        knots: ``(budget, share reached within it)`` pairs, ascending in both.
    """

    solver: str
    knots: tuple[tuple[float, float], ...]

    def share(self, budget: float) -> float:
        """Return the share of pairs reached within ``budget``."""
        reached = 0.0
        for spent, share in self.knots:
            if spent > budget:
                break
            reached = share
        return reached

    @property
    def final(self) -> float:
        """The share of pairs reached by the end of the runs."""
        return self.knots[-1][1] if self.knots else 0.0


@dataclass(frozen=True)
class RuntimeDistribution:
    """Runtime distributions of every solver over quality targets.

    Attributes:
        curves: One curve per solver, in campaign order.
        problems: The problems of the distribution.
        targets: The quality targets, as shares of each problem's span.
        budget: What measures the runtime.
        horizon: The largest budget any run spent.
    """

    curves: tuple[RuntimeCurve, ...]
    problems: tuple[str, ...]
    targets: tuple[float, ...]
    budget: Budget
    horizon: float


def runtime_distribution(
    source: object,
    *,
    targets: Sequence[float] = QUALITY_TARGETS,
    budget: str = "time",
) -> RuntimeDistribution:
    """Return the empirical distribution of the runtime to reach quality targets.

    Following the anytime assessment of Hansen et al. (2022), each problem
    gets quality targets ``f_L + t (f_U - f_L)`` for every ``t`` in
    ``targets``, where ``f_L`` is the best value any run reached and ``f_U``
    the worst first feasible value of a run. The runtime of a run to a
    target is the budget at which its best value first reaches the target.
    A solver's curve at budget ``b`` is the share of (problem, target) pairs
    reached within ``b``, averaged over its runs on each problem, with every
    problem weighing equally. Failed runs never reach a target, and a
    solver excluded from a problem reaches none of its targets.

    Args:
        source: Runs of one protocol, anything :func:`observe` accepts.
        targets: Target levels between 0 (the best value found) and 1.
        budget: ``"time"`` for elapsed seconds or ``"evaluations"``.

    Returns:
        The distributions.

    Raises:
        ConfigError: If a target is outside ``[0, 1]`` or repeated, there
            is none, the budget is unknown, or the runs span several
            protocols.
    """
    measure = check_budget(budget)
    levels = tuple(sorted({float(level) for level in targets}, reverse=True))
    if not levels or len(levels) != len(targets) or not all(0.0 <= t <= 1.0 for t in levels):
        raise ConfigError(f"targets must be distinct numbers in [0, 1], got {list(targets)}")
    observations = observe(source).comparable()
    ranges = quality_ranges(observations, measure)
    problems = observations.problems
    curves = []
    for solver in observations.solvers:
        hits: list[tuple[float, float]] = []
        for problem in problems:
            span = ranges.get(problem)
            runs = observations.cell(solver, problem)
            if span is None or not runs:
                continue
            weight = 1.0 / (len(problems) * len(levels) * len(runs))
            for run in runs:
                progress = observations.progress(run, measure)
                for level in levels:
                    goal = span.best + level * (span.start - span.best)
                    spent = next((spent for spent, value in progress if _passes(value, goal)), None)
                    if spent is not None:
                        hits.append((spent, weight))
        curves.append(RuntimeCurve(solver, _cumulative(hits)))
    horizon = max((run.spent(measure) for run in observations.runs), default=0.0)
    return RuntimeDistribution(tuple(curves), problems, levels, measure, horizon)


def _cumulative(hits: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Return ``(budget, cumulative weight)`` knots, one per distinct budget."""
    hits.sort()
    shares = list(accumulate(weight for _, weight in hits))
    knots: list[tuple[float, float]] = []
    for (spent, _), share in zip(hits, shares, strict=True):
        if knots and knots[-1][0] == spent:
            knots[-1] = (spent, min(1.0, share))
        else:
            knots.append((spent, min(1.0, share)))
    return tuple(knots)


def plot_runtime_distribution(
    distribution: RuntimeDistribution,
    *,
    labels: Mapping[str, str] | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Draw runtime distributions as step functions on a logarithmic budget axis.

    Args:
        distribution: The distributions.
        labels: Display names of solvers; a solver without one shows its name.
        theme: The appearance.

    Returns:
        The figure, or a placeholder when no run reached any target.
    """
    active = theme or DEFAULT_THEME
    name = "runtime-ecdf"
    title = "Runtime to reach quality targets"
    reached = [spent for curve in distribution.curves for spent, _ in curve.knots]
    if not distribution.curves or not reached:
        return placeholder(name, title, "No run reached a quality target.", active)
    positive = [spent for spent in [*reached, distribution.horizon] if spent > 0.0]
    unit = "Elapsed seconds" if distribution.budget == "time" else "Evaluations"
    x = log_axis(unit, positive or [1.0])
    y = numeric_axis("Share of targets reached", [], low=0.0, high=1.0)
    series = []
    for index, curve in enumerate(distribution.curves):
        points = [(x.low, curve.share(x.low))]
        points += [(spent, share) for spent, share in curve.knots if spent > x.low]
        series.append(
            Series(
                label_for(curve.solver, labels),
                tuple(points),
                active.series(index),
                kind="step",
                until=max(x.low, min(distribution.horizon, x.high)),
            )
        )
    percents = ", ".join(f"{100.0 * level:g}%" for level in distribution.targets)
    notes = [
        f"Targets on each problem: the best value any run found plus {percents} of the gap to "
        "the worst first value of a run; every problem weighs equally, and failed runs reach "
        "no target (Hansen et al. 2022).",
    ]
    description = (
        f"Runtime distributions of {len(distribution.curves)} solvers over "
        f"{len(distribution.problems)} problems and {len(distribution.targets)} targets. "
        "Share of pairs reached by the end: "
        + "; ".join(
            f"{label_for(curve.solver, labels)} {curve.final:.2f}" for curve in distribution.curves
        )
        + ". "
        + " ".join(notes)
    )
    return xy_chart(name, title, description, x=x, y=y, series=series, notes=notes, theme=active)
