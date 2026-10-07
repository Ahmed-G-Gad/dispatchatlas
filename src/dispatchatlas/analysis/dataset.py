"""Campaign results as analysis data.

:func:`load_dataset` reads a campaign directory written by
:func:`dispatchatlas.campaigns.run` through the readers of
:mod:`dispatchatlas.campaigns.store`, and returns a :class:`Dataset` that holds
one :class:`Run` for every run of the plan, in plan order, whether or not the
run completed.

Every planned run has one of four statuses:

- ``"succeeded"``: the run completed with a feasible schedule;
- ``"infeasible"``: the run completed, but its schedule breaks a constraint,
  such as a hard deadline, or the solver returned no schedule;
- ``"failed"``: every attempt of the run failed;
- ``"missing"``: the run has no record yet, because the campaign is still
  running or was interrupted.

The analysis counts the last three as failures and never drops them: success
rates divide by the planned runs, and value statistics score a failure as the
worst possible value. Denominators are therefore fixed by the plan, and a
solver cannot look better by failing on hard problems.

A failed run whose last attempt was killed at its hard limit is a *killed*
run, and may hold the incumbent its search had reported when its time limit
passed. It stays a failed run in every count; an analysis that asks for it
scores the run by that incumbent instead of the worst value. What its search
found after the time limit is never a score.

:func:`load_dataset` also checks the directory
(:func:`~dispatchatlas.campaigns.checking.check_campaign`) and keeps the
report in :attr:`Dataset.check`, from which the analysis takes the findings
it does not report itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

from dispatchatlas.campaigns.checking import CheckReport, check_campaign
from dispatchatlas.campaigns.config import Protocol
from dispatchatlas.campaigns.planning import Plan, PlannedProblem
from dispatchatlas.campaigns.records import (
    ErrorKind,
    FailureRecord,
    Incumbent,
    Outcome,
    Provenance,
    RunRecord,
    SessionRecord,
    SolverEntry,
)
from dispatchatlas.campaigns.store import read_failures, read_plan, read_runs, read_sessions
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.logs import get_logger
from dispatchatlas.core.solver import TracePoint

__all__ = ["RUN_STATUSES", "Dataset", "Run", "RunStatus", "load_dataset"]

_LOG = get_logger(__name__)

type RunStatus = Literal["succeeded", "infeasible", "failed", "missing"]
"""The outcome of a planned run, see the module documentation."""

RUN_STATUSES: tuple[RunStatus, ...] = ("succeeded", "infeasible", "failed", "missing")


@dataclass(frozen=True)
class Run:
    """One planned run and what became of it.

    Attributes:
        run_id: Run id from the plan.
        solver: Solver name: the campaign label, or the solver id.
        problem: Problem id.
        protocol: Protocol name.
        replicate: Replicate number, from 0.
        seed: Seed the solver received.
        status: What became of the run.
        record: The run record, for a completed run.
        failures: The failed attempts of the run, in attempt order. A run can
            fail and later complete, so a succeeded run may have failures too.
    """

    run_id: str
    solver: str
    problem: str
    protocol: str
    replicate: int
    seed: int
    status: RunStatus
    record: RunRecord | None = None
    failures: tuple[FailureRecord, ...] = ()

    @property
    def succeeded(self) -> bool:
        """Whether the run completed with a feasible schedule."""
        return self.status == "succeeded"

    @property
    def killed(self) -> bool:
        """Whether the run never completed and its last attempt was killed at its hard limit."""
        return (
            self.status == "failed"
            and bool(self.failures)
            and self.failures[-1].error.kind is ErrorKind.TIMEOUT
        )

    @property
    def unstarted(self) -> bool:
        """Whether the run never completed and its last attempt ended at the start-up limit.

        Its worker did not begin the solve in time, so the failure says
        nothing about the solver.
        """
        return (
            self.status == "failed"
            and bool(self.failures)
            and self.failures[-1].error.kind is ErrorKind.STARTUP
        )

    @property
    def incumbent(self) -> Incumbent | None:
        """The feasible incumbent a killed run held when its time limit passed, or ``None``.

        The last candidate the run's search had reported within the time
        limit of its protocol, if the search found it feasible. A candidate
        found after the limit is never returned: a solver that stopped on
        time had no such seconds to search in. The values are the solver's
        own: no schedule exists to check them against.
        """
        if not self.killed:
            return None
        held = self.failures[-1].limit_incumbent
        return held if held is not None and held.feasible else None

    @property
    def late_incumbent(self) -> bool:
        """Whether the search of a killed run reported an incumbent after its time limit.

        No score uses such an incumbent, see :attr:`incumbent`.
        """
        if not self.killed:
            return False
        failure = self.failures[-1]
        last = failure.last_incumbent
        return last is not None and last != failure.limit_incumbent

    @property
    def outcome(self) -> Outcome | None:
        """The recorded outcome, or ``None`` for a run that did not complete."""
        return None if self.record is None else self.record.outcome

    @property
    def provenance(self) -> Provenance | None:
        """How the last attempt of the run executed, or ``None`` for a run never attempted.

        The attempt that completed the run, else its last failed attempt.
        """
        if self.record is not None:
            return self.record.provenance
        return self.failures[-1].provenance if self.failures else None

    @property
    def cpu_share(self) -> float | None:
        """Processor seconds of the completing attempt over its wall seconds, or ``None``.

        The processor seconds are those of the process that executed the
        attempt and, on POSIX systems, of the child processes it waited for.
        A share below 1 means that this process was not computing for part
        of the run: it waited for a processor, slept, or left the work to
        another process whose time the record does not hold, as a solver
        does on Windows when its backend runs as a program of its own. The
        share alone does not say which. A share above 1 means the solver ran
        threads or child processes side by side.

        ``None`` for a run that did not complete, and when either time was
        not measured or the wall time is zero. The processor time of a
        process advances in steps, of about 16 milliseconds on Windows, so
        the share of a run shorter than a few tenths of a second is coarse.
        """
        if self.record is None:
            return None
        executed = self.record.provenance
        if executed.cpu_seconds is None or not executed.wall_seconds > 0.0:
            return None
        return executed.cpu_seconds / executed.wall_seconds

    @property
    def objectives(self) -> Mapping[str, float]:
        """Recorded objective values, empty for a run that did not complete."""
        return {} if self.record is None else self.record.outcome.objectives

    @property
    def trace(self) -> tuple[TracePoint, ...]:
        """Improvements of the best solution during the run, empty if not recorded."""
        return () if self.record is None else self.record.outcome.trace

    def value(self, objective: str) -> float | None:
        """Return the recorded value of an objective, or ``None`` if there is none.

        The value is returned whatever the status, so an infeasible run
        reports the value of its infeasible schedule. The analysis reads it
        only for succeeded runs.
        """
        return self.objectives.get(objective)

    def points(self, objectives: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Return the objective vectors of the solutions the run found.

        A multi-objective solver records a front; any other solver
        contributes the single vector of its best schedule.

        Args:
            objectives: The problem's objectives, in the problem's order,
                which is the order of recorded front vectors.

        Returns:
            The vectors, in the recorded (natural) orientation. Empty unless
            the run succeeded.
        """
        if not self.succeeded or self.record is None:
            return ()
        outcome = self.record.outcome
        if outcome.front:
            return tuple(tuple(point) for point in outcome.front)
        return (tuple(outcome.objectives[name] for name in objectives),)


@dataclass(frozen=True)
class Dataset:
    """The runs of a campaign, with the plan that explains them.

    Build one with :func:`load_dataset`, and narrow it with :meth:`select`.

    Attributes:
        plan: The campaign's plan, complete even after a selection.
        runs: The selected runs, in plan order.
        solvers: The selected solvers, in campaign order.
        problems: The selected problems, in plan order.
        protocols: The selected protocols, in campaign order.
        sessions: The runner sessions that produced the results.
        check: What the check of the campaign directory found when the
            dataset was loaded, for the whole directory whatever the
            selection. ``None`` for a dataset that was not loaded from a
            directory or was loaded without the check.
    """

    plan: Plan
    runs: tuple[Run, ...]
    solvers: tuple[SolverEntry, ...]
    problems: tuple[PlannedProblem, ...]
    protocols: tuple[Protocol, ...]
    sessions: tuple[SessionRecord, ...] = field(default=(), compare=False)
    check: CheckReport | None = field(default=None, compare=False)

    @property
    def campaign(self) -> str:
        """Campaign id."""
        return self.plan.campaign

    @property
    def config_hash(self) -> str:
        """Config hash of the campaign."""
        return self.plan.config_hash

    @property
    def objectives(self) -> tuple[str, ...]:
        """Every objective the selected problems declare, in first-seen order."""
        seen: dict[str, None] = {}
        for problem in self.problems:
            seen.update(dict.fromkeys(problem.objectives))
        return tuple(seen)

    def solver(self, name: str) -> SolverEntry:
        """Return the selected solver with the given name.

        Raises:
            ConfigError: If no selected solver has that name.
        """
        for entry in self.solvers:
            if entry.name == name:
                return entry
        raise ConfigError(_unknown("solver", name, [entry.name for entry in self.solvers]))

    def problem(self, problem_id: str) -> PlannedProblem:
        """Return the selected problem with the given id.

        Raises:
            ConfigError: If no selected problem has that id.
        """
        for item in self.problems:
            if item.id == problem_id:
                return item
        raise ConfigError(_unknown("problem", problem_id, [item.id for item in self.problems]))

    def protocol(self, name: str) -> Protocol:
        """Return the selected protocol with the given name.

        Raises:
            ConfigError: If no selected protocol has that name.
        """
        for item in self.protocols:
            if item.name == name:
                return item
        raise ConfigError(_unknown("protocol", name, [item.name for item in self.protocols]))

    def cell(self, solver: str, problem: str, protocol: str) -> tuple[Run, ...]:
        """Return the runs of one solver on one problem under one protocol, by replicate.

        The tuple is empty when the plan excludes the pair, because the
        solver does not model a feature of the problem.
        """
        return self._cells.get((solver, problem, protocol), ())

    def counts(self) -> dict[RunStatus, int]:
        """Return the number of selected runs of each status."""
        totals: dict[RunStatus, int] = dict.fromkeys(RUN_STATUSES, 0)
        for run in self.runs:
            totals[run.status] += 1
        return totals

    def select(
        self,
        *,
        protocols: Iterable[str] | None = None,
        solvers: Iterable[str] | None = None,
        problems: Iterable[str] | None = None,
    ) -> Dataset:
        """Return the dataset restricted to some protocols, solvers, and problems.

        Args:
            protocols: Protocol names to keep, or ``None`` to keep all.
            solvers: Solver names to keep, or ``None`` to keep all.
            problems: Problem ids to keep, or ``None`` to keep all.

        Returns:
            A dataset with the selected entries in their original order.

        Raises:
            ConfigError: If a name is not in the dataset or is given twice.
        """
        kept_protocols = _pick("protocol", self.protocols, protocols, lambda item: item.name)
        kept_solvers = _pick("solver", self.solvers, solvers, lambda item: item.name)
        kept_problems = _pick("problem", self.problems, problems, lambda item: item.id)
        protocol_names = {item.name for item in kept_protocols}
        solver_names = {item.name for item in kept_solvers}
        problem_ids = {item.id for item in kept_problems}
        runs = tuple(
            run
            for run in self.runs
            if run.protocol in protocol_names
            and run.solver in solver_names
            and run.problem in problem_ids
        )
        return Dataset(
            self.plan,
            runs,
            kept_solvers,
            kept_problems,
            kept_protocols,
            self.sessions,
            self.check,
        )

    @cached_property
    def _cells(self) -> Mapping[tuple[str, str, str], tuple[Run, ...]]:
        cells: dict[tuple[str, str, str], list[Run]] = {}
        for run in self.runs:
            cells.setdefault((run.solver, run.problem, run.protocol), []).append(run)
        return {
            key: tuple(sorted(items, key=lambda run: run.replicate)) for key, items in cells.items()
        }


def load_dataset(directory: str | Path, *, check: bool = True) -> Dataset:
    """Read a campaign directory into a dataset.

    Args:
        directory: A campaign directory, ``<root>/<campaign id>``, as
            :func:`dispatchatlas.campaigns.run` writes it.
        check: Check the directory with
            :func:`~dispatchatlas.campaigns.checking.check_campaign` and keep
            the report in :attr:`Dataset.check`. The check reads every
            record once more. It does not derive the plan again from the
            stored campaign, which would need the campaign's instance files
            and solvers; ``dispatchatlas campaign check`` does. A blocking
            finding is logged as a warning.

    Returns:
        Every planned run with its status, record, and failed attempts, and
        the report of the check.

    Raises:
        SerializationError: If the plan or a record is missing, malformed,
            or no longer matches its result hash.
    """
    folder = Path(directory)
    plan = read_plan(folder)
    records = {record.run_id: record for record in read_runs(folder)}
    failures: dict[str, list[FailureRecord]] = {}
    for failure in read_failures(folder):
        failures.setdefault(failure.run_id, []).append(failure)
    runs = []
    for planned in plan.runs:
        record = records.get(planned.run_id)
        failed = tuple(failures.get(planned.run_id, ()))
        runs.append(
            Run(
                run_id=planned.run_id,
                solver=planned.solver,
                problem=planned.problem,
                protocol=planned.protocol,
                replicate=planned.replicate,
                seed=planned.seed,
                status=_status(record, failed),
                record=record,
                failures=failed,
            )
        )
    report = check_campaign(folder, replan=False) if check else None
    if report is not None and not report.ok:
        blocking = report.of("blocking")
        _LOG.warning(
            "%s has %d blocking finding(s), so its results are not ready to be used: %s",
            folder,
            len(blocking),
            ", ".join(dict.fromkeys(finding.code for finding in blocking)),
        )
    return Dataset(
        plan=plan,
        runs=tuple(runs),
        solvers=plan.solvers,
        problems=plan.problems,
        protocols=plan.protocols,
        sessions=read_sessions(folder),
        check=report,
    )


def _status(record: RunRecord | None, failures: tuple[FailureRecord, ...]) -> RunStatus:
    if record is not None:
        outcome = record.outcome
        return "succeeded" if outcome.feasible and outcome.value is not None else "infeasible"
    return "failed" if failures else "missing"


def _pick[T](
    what: str, items: tuple[T, ...], wanted: Iterable[str] | None, key: Callable[[T], str]
) -> tuple[T, ...]:
    if wanted is None:
        return items
    names = list(wanted)
    known = [key(item) for item in items]
    for name in names:
        if name not in known:
            raise ConfigError(_unknown(what, name, known))
    if len(set(names)) != len(names):
        raise ConfigError(f"a {what} is selected twice: {names}")
    chosen = set(names)
    return tuple(item for item in items if key(item) in chosen)


def _unknown(what: str, name: str, known: Sequence[str]) -> str:
    listed = ", ".join(known) if known else "none"
    return f"unknown {what} {name!r}; the dataset has: {listed}"
