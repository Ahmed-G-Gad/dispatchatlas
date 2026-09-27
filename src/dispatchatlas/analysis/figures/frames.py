"""What the figures read: every planned run, judged by one outcome policy.

:func:`observe` turns the results of a campaign into :class:`Observations`,
one :class:`Observation` per planned run. It accepts:

- an analysis :class:`~dispatchatlas.analysis.dataset.Dataset`, whose runs
  already carry their status and which also knows the runs that never
  completed or never started;
- a sequence of :class:`~dispatchatlas.campaigns.records.RunRecord`;
- any object with a ``runs`` attribute holding run records and, optionally,
  a ``failures`` attribute holding
  :class:`~dispatchatlas.campaigns.records.FailureRecord` objects.

**Outcome policy.** It is the analysis package's: a run *succeeds* when it
completed with a feasible schedule and a value. An *infeasible* run (its
schedule breaks a constraint), a *failed* run (every attempt raised), and a
*missing* run (no record yet) are failures. Failures are never dropped: they
count in every denominator, as the worst value in every aggregate, and as
never reaching any target. A solver that the plan excludes from a problem
counts as failing it in the figures that compare solvers over a common
problem set, the performance and data profiles.

**Analysed objective.** Figures read each problem's primary objective, or
the objective and sense an
:class:`~dispatchatlas.analysis.config.AnalysisConfig` names
(:meth:`Observations.configured`), as the tables do. Traces record the
primary objective only, so the anytime figures, which read traces, have no
data when another objective is analysed.

**Aggregation over replicates** is explicit (:data:`Aggregation`):
``"median"`` takes the median run, the lower of the two middle runs for an
even count, so that the value is one a run achieved and a cell where more
than half the runs fail is a failure, while an even split keeps the finite
middle run; ``"best"`` takes the best run.

**Progress.** The anytime figures read how the best value found improved
during a run (:meth:`Observation.progress`): the feasible points of the
recorded trace, and the final value at the final budget. The budget is the
number of evaluations or the elapsed seconds (:data:`Budget`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Literal

from dispatchatlas.analysis.config import AnalysisConfig
from dispatchatlas.analysis.dataset import Dataset, RunStatus
from dispatchatlas.analysis.stats.ranks import Sense
from dispatchatlas.campaigns.records import FailureRecord, RunRecord
from dispatchatlas.core.errors import ConfigError, ModelError
from dispatchatlas.core.logs import get_logger
from dispatchatlas.core.objectives import get_objective

__all__ = [
    "AGGREGATIONS",
    "BUDGETS",
    "Aggregation",
    "Budget",
    "Observation",
    "Observations",
    "QualityRange",
    "aggregate",
    "check_aggregation",
    "check_budget",
    "lower_median",
    "objective_sense",
    "observe",
    "quality_ranges",
]

type Aggregation = Literal["median", "best"]
"""How the replicates of a solver on a problem become one value."""

type Budget = Literal["evaluations", "time"]
"""What measures the effort of a run: schedules evaluated, or elapsed seconds."""

AGGREGATIONS: tuple[Aggregation, ...] = ("median", "best")
BUDGETS: tuple[Budget, ...] = ("evaluations", "time")

_LOG = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Observation:
    """One planned run.

    Attributes:
        run_id: Run id.
        solver: Solver name: the campaign label, or the solver id.
        problem: Problem id.
        protocol: Protocol name.
        replicate: Replicate number, from 0.
        status: ``"succeeded"``, ``"infeasible"``, ``"failed"``, or ``"missing"``.
        record: The run record of a completed run.
    """

    run_id: str
    solver: str
    problem: str
    protocol: str
    replicate: int
    status: RunStatus
    record: RunRecord | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the run completed with a feasible schedule and a value."""
        return self.status == "succeeded" and self.record is not None

    def progress(self, sense: Sense, budget: Budget) -> tuple[tuple[float, float], ...]:
        """Return how the best feasible value of the primary objective improved during the run.

        Args:
            sense: Whether the primary objective is minimised or maximised.
            budget: The effort measure of the first coordinate.

        Returns:
            ``(budget, best value so far)`` pairs, the value in minimisation
            form and strictly improving: one per feasible trace point that
            improved on the previous ones, then the final value at the final
            budget if the trace ends above it. Empty unless the run succeeded.
        """
        if not self.succeeded or self.record is None:
            return ()
        outcome = self.record.outcome
        sign = 1.0 if sense == "min" else -1.0
        points: list[tuple[float, float]] = []
        spent = 0.0
        for point in outcome.trace:
            spent = max(
                spent, float(point.evaluations if budget == "evaluations" else point.elapsed)
            )
            if not point.feasible:
                continue
            value = sign * point.value
            if not points or value < points[-1][1]:
                points.append((spent, value))
        final = sign * outcome.value if outcome.value is not None else math.inf
        end = max(spent, float(outcome.evaluations if budget == "evaluations" else outcome.elapsed))
        if math.isfinite(final) and (not points or final < points[-1][1]):
            points.append((end, final))
        return tuple(points)

    def spent(self, budget: Budget) -> float:
        """Return the effort the run recorded, zero for a run without a record."""
        if self.record is None:
            return 0.0
        outcome = self.record.outcome
        trace = outcome.trace
        if budget == "evaluations":
            return float(max([outcome.evaluations, *(point.evaluations for point in trace)]))
        return max([outcome.elapsed, *(point.elapsed for point in trace)])


@dataclass(frozen=True)
class Observations:
    """The planned runs of a campaign, as the figures read them.

    Build one with :func:`observe`, and focus it with :meth:`configured`,
    :meth:`select`, :meth:`restrict`, and :meth:`analysing`.

    Attributes:
        runs: One entry per planned run, ordered by problem, replicate,
            solver, and protocol.
        solvers: Solver names, in campaign order.
        problems: Problem ids, in plan order.
        protocols: Protocol names, in campaign order.
        objectives: Each problem's objectives, primary first, where known.
        sizes: Each problem's number of tasks, where known.
        target: The analysed objective, or ``None`` for each problem's
            primary objective.
        target_sense: The analysed objective's sense, or ``None`` for the
            sense the objective registry declares.
    """

    runs: tuple[Observation, ...]
    solvers: tuple[str, ...]
    problems: tuple[str, ...]
    protocols: tuple[str, ...]
    objectives: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    sizes: Mapping[str, int] = field(default_factory=dict)
    target: str | None = None
    target_sense: Sense | None = None

    @cached_property
    def _cells(self) -> Mapping[tuple[str, str], tuple[Observation, ...]]:
        cells: dict[tuple[str, str], list[Observation]] = {}
        for run in self.runs:
            cells.setdefault((run.solver, run.problem), []).append(run)
        return {key: tuple(items) for key, items in cells.items()}

    @cached_property
    def _progress(self) -> dict[tuple[str, Budget], tuple[tuple[float, float], ...]]:
        return {}

    def cell(self, solver: str, problem: str) -> tuple[Observation, ...]:
        """Return the runs of a solver on a problem; empty when the plan excludes the pair."""
        return self._cells.get((solver, problem), ())

    def primary(self, problem: str) -> str | None:
        """Return the primary objective a problem declares, or ``None`` when unknown."""
        names = self.objectives.get(problem, ())
        return names[0] if names else None

    def objective(self, problem: str) -> str | None:
        """Return the analysed objective of a problem, or ``None`` when unknown."""
        return self.target if self.target is not None else self.primary(problem)

    def sense(self, problem: str) -> Sense:
        """Return the sense of a problem's analysed objective, minimisation when unknown."""
        if self.target_sense is not None:
            return self.target_sense
        name = self.objective(problem)
        return "min" if name is None else objective_sense(name)

    def objective_label(self) -> str:
        """Return the name of the analysed objective, or a phrase when problems differ."""
        names = {self.objective(problem) for problem in self.problems}
        if len(names) == 1 and None not in names:
            return str(names.pop())
        return "primary objective"

    def value(self, run: Observation) -> float | None:
        """Return the analysed objective's value of a succeeded run, else ``None``."""
        if not run.succeeded or run.record is None:
            return None
        outcome = run.record.outcome
        name = self.objective(run.problem)
        if name is None or name == outcome.objective:
            return outcome.value
        return outcome.objectives.get(name)

    def score(self, run: Observation) -> float:
        """Return a run's value in minimisation form, infinite for a run that did not succeed."""
        value = self.value(run)
        if value is None:
            return math.inf
        return value if self.sense(run.problem) == "min" else -value

    def cell_score(self, solver: str, problem: str, aggregation: Aggregation) -> float:
        """Return the aggregated score of a solver's runs on a problem, see :func:`aggregate`."""
        return aggregate([self.score(run) for run in self.cell(solver, problem)], aggregation)

    def traced(self, problem: str) -> bool:
        """Return whether traces record the analysed objective of a problem, its primary one."""
        return self.target is None or self.target == self.primary(problem)

    def progress(self, run: Observation, budget: Budget) -> tuple[tuple[float, float], ...]:
        """Return a run's progress, see :meth:`Observation.progress`, computed once.

        Empty when the analysed objective is not the primary one, which is
        the only objective traces record.
        """
        if not self.traced(run.problem):
            return ()
        key = (run.run_id, budget)
        found = self._progress.get(key)
        if found is None:
            found = self._progress[key] = run.progress(self.sense(run.problem), budget)
        return found

    def select(self, protocol: str) -> Observations:
        """Return the runs of one protocol.

        Raises:
            ConfigError: If no run has that protocol.
        """
        if protocol not in self.protocols:
            raise ConfigError(f"unknown protocol {protocol!r}; the runs have {self.protocols}")
        runs = tuple(run for run in self.runs if run.protocol == protocol)
        return replace(self, runs=runs, protocols=(protocol,))

    def restrict(
        self, solvers: Iterable[str] | None = None, problems: Iterable[str] | None = None
    ) -> Observations:
        """Return the runs of some solvers on some problems, in their original order.

        Args:
            solvers: Solver names to keep, or ``None`` for all.
            problems: Problem ids to keep, or ``None`` for all.

        Raises:
            ConfigError: If a name is not among the runs'.
        """
        kept_solvers = _kept("solver", self.solvers, solvers)
        kept_problems = _kept("problem", self.problems, problems)
        runs = tuple(
            run for run in self.runs if run.solver in kept_solvers and run.problem in kept_problems
        )
        return replace(self, runs=runs, solvers=kept_solvers, problems=kept_problems)

    def analysing(self, objective: str | None, sense: Sense | None = None) -> Observations:
        """Return these runs read for another objective, or for each problem's primary one.

        Args:
            objective: The objective to read, or ``None`` for each problem's
                primary objective.
            sense: Its sense, or ``None`` for the sense the registry declares.

        Raises:
            ConfigError: If a problem whose objectives are known does not
                declare ``objective``.
        """
        if objective is not None:
            for problem in self.problems:
                names = self.objectives.get(problem)
                if names is not None and objective not in names:
                    raise ConfigError(
                        f"problem {problem!r} does not declare objective {objective!r}"
                    )
        return replace(self, target=objective, target_sense=sense)

    def configured(self, config: AnalysisConfig) -> Observations:
        """Return the runs an analysis with these settings reads.

        That is its protocol, when it names one, its solvers and problems,
        and its objective and sense.

        Raises:
            ConfigError: If the settings name a protocol, solver, problem, or
                objective the runs lack.
        """
        focused = self if config.protocol is None else self.select(config.protocol)
        restricted = focused.restrict(config.solvers, config.problems)
        return restricted.analysing(config.objective, config.sense)

    def comparable(self) -> Observations:
        """Return these runs, checking that they come from one protocol.

        Runs under different stop criteria cannot be compared, so every
        figure reads one protocol at a time.

        Raises:
            ConfigError: If the runs span several protocols.
        """
        if len(self.protocols) > 1:
            raise ConfigError(
                f"the runs span protocols {', '.join(self.protocols)}; select one with "
                "Observations.select"
            )
        return self


def observe(source: object) -> Observations:
    """Return the planned runs of a campaign's results, with their status.

    Args:
        source: An analysis dataset, a sequence of run records, an object
            with ``runs`` (run records) and optionally ``failures`` (failure
            records), or observations, which are returned as they are.

    Returns:
        The observations. Records and failures give no missing runs, since
        only a dataset knows the plan; a run whose attempts failed and that
        never completed is a failed run.

    Raises:
        TypeError: If ``source`` is none of these, or holds other objects.
    """
    if isinstance(source, Observations):
        return source
    if isinstance(source, Dataset):
        return _from_dataset(source)
    runs = getattr(source, "runs", None)
    if runs is not None:
        failures = getattr(source, "failures", None) or ()
        return _from_records(_typed(runs, RunRecord), _typed(failures, FailureRecord))
    if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        return _from_records(_typed(source, RunRecord), [])
    raise TypeError(
        "figures read an analysis Dataset, a sequence of RunRecord, or an object with "
        f"'runs' and 'failures'; got {type(source).__name__}"
    )


def lower_median(values: Sequence[float]) -> float:
    """Return the lower median: the middle value, or the lower of the two middle values.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("the median of no values is undefined")
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def aggregate(scores: Sequence[float], aggregation: Aggregation) -> float:
    """Return one value for the scores of a cell's runs, in minimisation form.

    Args:
        scores: The runs' scores, infinite for a failed run.
        aggregation: ``"median"`` for the median run, ``"best"`` for the best run.

    Returns:
        The aggregated score, infinite for a cell without runs.
    """
    if not scores:
        return math.inf
    return min(scores) if aggregation == "best" else lower_median(scores)


def check_aggregation(value: object) -> Aggregation:
    """Return ``value`` if it names an aggregation.

    Raises:
        ConfigError: Otherwise.
    """
    for option in AGGREGATIONS:
        if value == option:
            return option
    raise ConfigError(f"aggregation must be one of {list(AGGREGATIONS)}, got {value!r}")


def check_budget(value: object) -> Budget:
    """Return ``value`` if it names a budget measure.

    Raises:
        ConfigError: Otherwise.
    """
    for option in BUDGETS:
        if value == option:
            return option
    raise ConfigError(f"budget must be one of {list(BUDGETS)}, got {value!r}")


@dataclass(frozen=True, slots=True)
class QualityRange:
    """The span of values found on one problem, in minimisation form.

    Attributes:
        best: The best value any succeeded run reached, the ``f_L`` of
            Moré and Wild (2009).
        start: The worst first feasible value of a succeeded run.
    """

    best: float
    start: float

    def gap(self, value: float) -> float:
        """Return how far ``value`` is from the best, as a share of the span, from 0 to 1.

        Infinite values, which stand for no feasible solution, give 1. A
        problem where every run started at the best value gives 0.
        """
        if not math.isfinite(value):
            return 1.0
        if self.start <= self.best:
            return 0.0
        return min(1.0, max(0.0, (value - self.best) / (self.start - self.best)))


def quality_ranges(observations: Observations, budget: Budget) -> dict[str, QualityRange]:
    """Return the span of values found on each problem where some run succeeded."""
    ranges: dict[str, QualityRange] = {}
    for problem in observations.problems:
        best, start = math.inf, -math.inf
        for solver in observations.solvers:
            for run in observations.cell(solver, problem):
                progress = observations.progress(run, budget)
                if progress:
                    best = min(best, progress[-1][1])
                    start = max(start, progress[0][1])
        if math.isfinite(best):
            ranges[problem] = QualityRange(best, start)
    return ranges


# ------------------------------------------------------------------- helpers

_UNREGISTERED: set[str] = set()


def objective_sense(objective: str) -> Sense:
    """Return the sense the objective registry declares, minimisation for an unknown name.

    An unknown objective is logged once, since the figures of a run of a
    plugin's objective can still be drawn when the plugin is not loaded.
    """
    try:
        return get_objective(objective).sense
    except ModelError:
        if objective not in _UNREGISTERED:
            _UNREGISTERED.add(objective)
            _LOG.warning(
                "objective %r is not registered; figures assume it is minimised", objective
            )
        return "min"


def _kept(what: str, names: tuple[str, ...], wanted: Iterable[str] | None) -> tuple[str, ...]:
    if wanted is None:
        return names
    chosen = set(wanted)
    for name in sorted(chosen):
        if name not in names:
            listed = ", ".join(names) if names else "none"
            raise ConfigError(f"unknown {what} {name!r}; the runs have: {listed}")
    return tuple(name for name in names if name in chosen)


def _typed[T](items: object, kind: type[T]) -> list[T]:
    if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
        raise TypeError(f"expected a sequence of {kind.__name__}, got {type(items).__name__}")
    found = list(items)
    for item in found:
        if not isinstance(item, kind):
            raise TypeError(f"expected {kind.__name__} objects, got {type(item).__name__}")
    return found


def _from_dataset(dataset: Dataset) -> Observations:
    runs = tuple(
        Observation(
            run.run_id,
            run.solver,
            run.problem,
            run.protocol,
            run.replicate,
            run.status,
            run.record,
        )
        for run in dataset.runs
    )
    return Observations(
        runs=runs,
        solvers=tuple(entry.name for entry in dataset.solvers),
        problems=tuple(problem.id for problem in dataset.problems),
        protocols=tuple(protocol.name for protocol in dataset.protocols),
        objectives={problem.id: tuple(problem.objectives) for problem in dataset.problems},
        sizes={problem.id: problem.tasks for problem in dataset.problems},
    )


def _from_records(records: list[RunRecord], failures: list[FailureRecord]) -> Observations:
    runs: dict[str, Observation] = {}
    for record in records:
        outcome = record.outcome
        status: RunStatus = (
            "succeeded" if outcome.feasible and outcome.value is not None else "infeasible"
        )
        runs[record.run_id] = Observation(
            record.run_id,
            record.solver.name,
            record.problem.id,
            record.protocol.name,
            record.replicate,
            status,
            record,
        )
    for failure in failures:
        if failure.run_id not in runs:
            runs[failure.run_id] = Observation(
                failure.run_id,
                failure.solver.name,
                failure.problem.id,
                failure.protocol.name,
                failure.replicate,
                "failed",
            )
    ordered = list(runs.values())
    solvers = _first_seen(run.solver for run in ordered)
    problems = _first_seen(run.problem for run in ordered)
    protocols = _first_seen(run.protocol for run in ordered)
    position = {
        "solver": {name: index for index, name in enumerate(solvers)},
        "problem": {name: index for index, name in enumerate(problems)},
        "protocol": {name: index for index, name in enumerate(protocols)},
    }
    ordered.sort(
        key=lambda run: (
            position["problem"][run.problem],
            run.replicate,
            position["solver"][run.solver],
            position["protocol"][run.protocol],
            run.run_id,
        )
    )
    objectives: dict[str, tuple[str, ...]] = {}
    for record in records:
        outcome = record.outcome
        if record.problem.id not in objectives:
            others = sorted(name for name in outcome.objectives if name != outcome.objective)
            objectives[record.problem.id] = (outcome.objective, *others)
    return Observations(tuple(ordered), solvers, problems, protocols, objectives)


def _first_seen(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))
