"""Checking a campaign directory before its results are used.

:func:`check_campaign` reads a campaign directory, writes nothing, and returns
a :class:`CheckReport`: a list of :class:`Finding` objects, each with a code,
a severity, a message, and the items it concerns. It answers what no reader
of results asks: whether the stored campaign is the one that was run, whether
the plan is the plan that campaign gives, whether every planned run has a
valid record or failures that settle it, whether the budgets were kept, and
whether the sessions agree on code and settings.

A finding has one of three severities:

- ``blocking``: the directory does not hold what it claims, so its results
  are not ready to be used: a missing or altered campaign file, a plan that
  is not the campaign's, a run without a record that is owed an attempt, a
  record that fails a hash, a seed used twice.
- ``limitation``: the results stand, with a caveat a report of them has to
  state: runs settled by failures, cells with fewer runs than planned,
  budgets that were passed, sessions on other code.
- ``note``: a fact worth knowing that qualifies nothing, such as an
  optional field that the records of the directory do not state.

:data:`CODES` lists every code with its severity and meaning. Findings on the
runs of one protocol name the protocol, so a reader of one protocol's results
can leave out the others.

The check reads what the engine stored and derives the rest again: the
ledger of the store (:meth:`Store.holdings
<dispatchatlas.campaigns.store.Store.holdings>`) says which records are valid,
:func:`~dispatchatlas.campaigns.execution.observations` derives the facts of a
record's time and trace from the record alone, so records without stored
notes are covered,
:func:`~dispatchatlas.campaigns.fairness.assess_outcomes` compares the wall
times and stop reasons with the plan, and
:func:`~dispatchatlas.campaigns.runner.session_changes` compares the sessions
that executed the counted attempts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dispatchatlas.benchmarks.registry import FamilyRegistry
from dispatchatlas.campaigns import fairness, planning
from dispatchatlas.campaigns.config import Campaign
from dispatchatlas.campaigns.execution import OVERRUN_MARGIN, observations
from dispatchatlas.campaigns.io import campaign_from_dict
from dispatchatlas.campaigns.planning import Plan, PlannedRun
from dispatchatlas.campaigns.records import (
    FailureRecord,
    RunRecord,
    SessionRecord,
    SessionStatus,
    settled,
)
from dispatchatlas.campaigns.runner import session_changes
from dispatchatlas.campaigns.store import (
    CAMPAIGN_FILE,
    CAMPAIGN_SCHEMA,
    PLAN_FILE,
    RUNS,
    Ledger,
    Stale,
    Store,
    read_plan,
    read_sessions,
    read_strays,
)
from dispatchatlas.core.errors import ConfigError, DispatchAtlasError, SerializationError
from dispatchatlas.core.io import read_json
from dispatchatlas.solvers.registry import SolverRegistry

__all__ = [
    "CODES",
    "SEVERITIES",
    "CheckReport",
    "Finding",
    "Severity",
    "check_campaign",
]

type Severity = Literal["blocking", "limitation", "note"]
"""How much a finding weighs: it blocks the use of the results, qualifies them, or informs."""

SEVERITIES: tuple[Severity, ...] = ("blocking", "limitation", "note")
"""The severities, from the heaviest."""

CODES: Mapping[str, tuple[Severity, str]] = {
    "recipe-missing": ("blocking", "campaign.json is missing."),
    "recipe-unreadable": ("blocking", "campaign.json cannot be read as a campaign."),
    "recipe-altered": ("blocking", "campaign.json no longer matches its recorded config hash."),
    "plan-unreadable": ("blocking", "plan.json is missing or cannot be read."),
    "plan-foreign": ("blocking", "plan.json states another config hash than campaign.json."),
    "plan-differs": ("blocking", "plan.json is not the plan the stored campaign gives."),
    "plan-digest-differs": (
        "blocking",
        "plan.json holds other runs or seeds than the plan the last session executed.",
    ),
    "runs-unattempted": ("blocking", "Planned runs have no record and no failed attempt."),
    "runs-unsettled": (
        "blocking",
        "Planned runs failed and are owed another attempt, which the campaign's retries allow.",
    ),
    "record-unreadable": ("blocking", "The file at a planned run's path is no run record."),
    "record-foreign": (
        "blocking",
        "The record at a planned run's path belongs to another configuration or to another "
        "version of its problem.",
    ),
    "record-hash-mismatch": (
        "blocking",
        "A record fails its result hash, its schedule digest, or its identity hash.",
    ),
    "record-differs-from-plan": (
        "blocking",
        "A record states another problem, solver, protocol, replicate, or seed than the plan.",
    ),
    "records-set-aside": (
        "blocking",
        "The valid record of a planned run lies under superseded/ and not at its path.",
    ),
    "seed-reused": ("blocking", "One seed serves two replicates of one problem."),
    "plan-not-derived": (
        "limitation",
        "The plan could not be derived again from the stored campaign on this installation.",
    ),
    "plan-changed": (
        "limitation",
        "The plan changed between sessions while the campaign file stayed the same.",
    ),
    "runs-failed": ("limitation", "Runs never completed and are settled by their failures."),
    "cells-partial": (
        "limitation",
        "A solver has records for some of the runs planned on a problem, not for all.",
    ),
    "solver-without-records": ("limitation", "A solver with planned runs has no record."),
    "solver-excluded-everywhere": (
        "limitation",
        "The plan excludes a solver from every problem, so it has no run.",
    ),
    "records-unplanned": (
        "limitation",
        "Records under runs/ belong to runs the plan does not hold; no reader returns them.",
    ),
    "files-unreadable": (
        "limitation",
        "Files under runs/, at no path of a planned run, cannot be read as run records.",
    ),
    "failures-unreadable": (
        "limitation",
        "Files under failures/ cannot be read as records of failed attempts.",
    ),
    "trace-time-axis": (
        "limitation",
        "Traces hold a negative time, times or evaluation counts that decrease, or a time "
        "beyond the elapsed time of their run.",
    ),
    "trace-contradicts-outcome": (
        "limitation",
        "The last value of a trace differs from the value of its outcome.",
    ),
    "record-without-trace": (
        "limitation",
        "Records of an iterative solver hold a schedule and no trace.",
    ),
    "budget-overrun": (
        "limitation",
        "Completed runs took more wall time than their time limit and the tolerance allow.",
    ),
    "hard-limits-differ": (
        "limitation",
        "The attempts under one protocol ran under different hard limits.",
    ),
    "stop-reasons-diverge": (
        "limitation",
        "The runs of the iterative solvers under one protocol ended for different reasons.",
    ),
    "sessions-unreadable": ("limitation", "A session record cannot be read."),
    "sessions-missing": (
        "limitation",
        "Attempts name sessions the directory holds no record of.",
    ),
    "sessions-differ-code": (
        "limitation",
        "The sessions that executed the counted attempts differ in the Python version, the "
        "packages, the library, or a git state.",
    ),
    "sessions-differ-plugins": (
        "limitation",
        "The sessions that executed the counted attempts differ in their plugins.",
    ),
    "sessions-differ-execution": (
        "limitation",
        "The sessions that executed the counted attempts differ in their execution settings.",
    ),
    "replicates-below-floor": (
        "limitation",
        "Cells of replicated solvers hold fewer completed replicates than the floor.",
    ),
    "seeds-unpaired": (
        "limitation",
        "One replicate of a problem ran with different seeds, so its runs are not paired.",
    ),
    "plan-not-compared": (
        "note",
        "The plan was not derived again from the stored campaign, as the caller asked.",
    ),
    "plan-digest-unstated": ("note", "No session record states the digest of its plan."),
    "pairs-excluded": ("note", "The plan excludes pairs of problem and solver."),
    "files-stray": ("note", "Files under runs/ are no JSON documents."),
    "failures-foreign": (
        "note",
        "Files at the paths of failed attempts belong to another plan and count as no attempt.",
    ),
    "records-without-identity-hash": (
        "note",
        "Records state no identity hash; they are compared with the plan field by field.",
    ),
    "records-noted": (
        "note",
        "Records carry notes of the engine that the check does not derive from the record, "
        "such as a claim of the solver that the checks of its result contradict.",
    ),
    "clock-disagreement": (
        "note",
        "A solver's elapsed time differs from the engine's wall time, or a run stopped for "
        "its time limit before the limit.",
    ),
    "sessions-differ-machine": (
        "note",
        "The sessions that executed the counted attempts differ in their machine.",
    ),
    "session-unfinished": ("note", "The last session is not recorded as finished."),
}
"""Every code a finding can have, with its severity and what it means."""

_SHOWN = 3
"""Runs a plan difference names before it counts the rest."""

_VALUE_CHARACTERS = 80
"""Longest value of a session field an item quotes."""

_TRACE_CODES: Mapping[str, str] = {
    "trace-time": "trace-time-axis",
    "trace-evaluations": "trace-time-axis",
    "trace-value": "trace-contradicts-outcome",
    "no-trace": "record-without-trace",
    "elapsed": "clock-disagreement",
    "early-stop": "clock-disagreement",
}
"""The code of the finding for each kind of fact a record's time and trace show.

An overrun is left to the assessment of the outcomes, which applies the
caller's tolerance.
"""

_ENTRY_FIELDS = ("id", "parameters", "family", "stochastic", "iterative", "multi_objective")
"""The fields of a solver entry every plan states."""

_OPTIONAL_ENTRY_FIELDS = ("threads", "replicates")
"""The fields of a solver entry a plan may not state, compared only when it does."""


@dataclass(frozen=True)
class Finding:
    """One thing a check of a campaign directory found.

    Attributes:
        code: What was found, a key of :data:`CODES`.
        severity: How much it weighs.
        message: One or two sentences that state it.
        items: What it concerns, one entry each: cells, files, solvers, or
            fields, in the order of the plan or of the paths.
        protocol: The protocol whose runs it concerns, or ``None`` when it
            concerns the whole directory.
    """

    code: str
    severity: Severity
    message: str
    items: tuple[str, ...] = ()
    protocol: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a JSON-ready dictionary."""
        return {
            "code": self.code,
            "severity": self.severity,
            "protocol": self.protocol,
            "message": self.message,
            "items": list(self.items),
        }


@dataclass(frozen=True)
class CheckReport:
    """What a check of a campaign directory found.

    Attributes:
        directory: The campaign directory.
        campaign: Campaign id of the plan, or ``None`` when the plan cannot
            be read.
        config_hash: Config hash of the plan, or else the one
            ``campaign.json`` records, or ``None``.
        findings: The findings, the blocking ones first, then the
            limitations, then the notes.
        counts: The planned runs by what the directory holds for them:
            ``planned``, ``completed`` (a valid record), ``failed`` (settled
            by failures), ``unsettled`` (failed and owed an attempt),
            ``unattempted``, ``invalid`` (a file at the path that is no
            valid record), and ``set_aside`` (the valid record lies under
            ``superseded/``). Empty when the plan cannot be read.
    """

    directory: Path
    campaign: str | None
    config_hash: str | None
    findings: tuple[Finding, ...]
    counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether no finding is blocking."""
        return not self.of("blocking")

    def of(self, severity: Severity) -> tuple[Finding, ...]:
        """Return the findings of one severity."""
        return tuple(item for item in self.findings if item.severity == severity)

    def find(self, code: str) -> tuple[Finding, ...]:
        """Return the findings with the given code, one per protocol it was found under."""
        return tuple(item for item in self.findings if item.code == code)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-ready dictionary."""
        return {
            "directory": str(self.directory),
            "campaign": self.campaign,
            "config_hash": self.config_hash,
            "ok": self.ok,
            "counts": dict(self.counts),
            "severities": {severity: len(self.of(severity)) for severity in SEVERITIES},
            "findings": [item.to_dict() for item in self.findings],
        }


def check_campaign(
    directory: str | Path,
    *,
    replicate_floor: int | None = None,
    overrun_tolerance: float = OVERRUN_MARGIN,
    replan: bool = True,
    registry: SolverRegistry | None = None,
    families: FamilyRegistry | None = None,
) -> CheckReport:
    """Check a campaign directory and report what qualifies its results.

    Nothing is written, moved, or run. The findings are, by what they
    concern:

    - the campaign file: missing, unreadable, or no longer matching its
      config hash;
    - the plan: of another config hash, different from the plan the stored
      campaign gives when it is derived again, different from the plan the
      last session executed, or changed between sessions;
    - the planned runs: without a record and without a failed attempt, with
      failures that do not settle them, settled by failures (by solver and
      kind of error), cells with records for only some of their runs,
      solvers without any record, and, apart from those, the pairs and the
      solvers the plan excludes;
    - the files: a file at a planned run's path that is no valid record of
      the run (unreadable, of another configuration or problem version,
      failing a hash, or differing from the plan), valid records lying
      under ``superseded/``, records of runs the plan does not hold, and
      files that cannot be read or are no records;
    - the records: a trace with an unusable time axis or a last value that
      contradicts the outcome, an iterative solver's record without a
      trace, and wall times beyond the time limit, by solver with the
      largest ratio;
    - the protocols: hard limits that differ among the attempts and stop
      reasons that diverge among the iterative solvers;
    - the sessions: those that executed the counted attempts and differ in
      code, plugins, execution settings, or machine, and those without a
      record;
    - the replication: cells below ``replicate_floor``, a seed used for two
      replicates of one problem, and a replicate that ran with two seeds.

    Records and session records may lack the fields that are optional in
    them. Where the check cannot make a comparison for that reason, it says
    so in a note: records without an identity hash, and session records
    without the digest of their plan.

    Args:
        directory: A campaign directory, ``<root>/<campaign id>``.
        replicate_floor: The fewest completed replicates a cell of a
            replicated solver may hold without a finding. ``None`` sets no
            floor.
        overrun_tolerance: Share of a time limit by which the wall time of a
            completed run may pass it without counting as over, as in
            :func:`~dispatchatlas.campaigns.fairness.assess_outcomes`.
        replan: Derive the plan again from the stored campaign and compare
            it with ``plan.json``. This loads every problem of the campaign,
            so it needs the campaign's instance files and solvers.
        registry: Solver registry for deriving the plan. Defaults to the
            built-in solvers and plugins.
        families: Benchmark-family registry for deriving the plan. Defaults
            to the built-in families and plugins.

    Returns:
        The report. With an unreadable plan it holds the findings on the
        campaign file and the plan only.

    Raises:
        ConfigError: If ``directory`` is no directory, or ``replicate_floor``
            is not a positive integer, or ``overrun_tolerance`` is not zero
            or a positive number.
    """
    folder = Path(directory)
    if not folder.is_dir():
        raise ConfigError(f"{folder} is not a directory")
    floor = replicate_floor
    if floor is not None and (isinstance(floor, bool) or not isinstance(floor, int) or floor < 1):
        raise ConfigError(f"replicate_floor must be a positive integer, got {floor!r}")
    found = _Findings()
    campaign, recorded = _recipe(folder, found)
    try:
        plan = read_plan(folder)
    except SerializationError as error:
        found.add("plan-unreadable", f"{error}. Nothing else of the directory can be checked.")
        return CheckReport(folder, None, recorded, found.ordered())
    _check_plan(plan, campaign, recorded, found, replan, registry, families)
    sessions = _sessions(folder, found)
    ledger, records = Store(folder).holdings(plan)
    states = _states(folder, plan, ledger, _max_attempts(campaign, sessions))
    _check_runs(plan, ledger, states, found)
    _check_files(folder, plan, ledger, found)
    _check_records(plan, records, overrun_tolerance, found)
    _check_hard_limits(plan, records, ledger, found)
    if sessions is not None:
        _check_sessions(plan, sessions, records, ledger, found)
    if floor is not None:
        _check_replicates(plan, states, floor, found)
    _check_seeds(plan, records, found)
    counts = Counter(states.values())
    return CheckReport(
        directory=folder,
        campaign=plan.campaign,
        config_hash=plan.config_hash,
        findings=found.ordered(),
        counts={"planned": len(plan.runs), **{state: counts[state] for state in _STATES}},
    )


# ------------------------------------------------------------------ helpers

_STATES = ("completed", "failed", "unsettled", "unattempted", "invalid", "set_aside")
"""What a directory can hold for a planned run, as :attr:`CheckReport.counts` names it."""


class _Findings:
    """Collects findings and orders them by severity."""

    def __init__(self) -> None:
        self._found: list[Finding] = []

    def add(
        self, code: str, message: str, items: Iterable[str] = (), protocol: str | None = None
    ) -> None:
        self._found.append(Finding(code, CODES[code][0], message, tuple(items), protocol))

    def ordered(self) -> tuple[Finding, ...]:
        return tuple(sorted(self._found, key=lambda finding: SEVERITIES.index(finding.severity)))


def _short(value: str | None) -> str:
    return "none" if value is None else value[:12]


def _relative(path: Path, folder: Path) -> str:
    try:
        return path.relative_to(folder).as_posix()
    except ValueError:
        return path.as_posix()


def _tally(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{number} {name}" for name, number in sorted(counts.items()))


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _named(run: PlannedRun) -> str:
    return f"{run.solver} on {run.problem} under {run.protocol}, replicate {run.replicate}"


def _first(runs: Sequence[PlannedRun]) -> str:
    more = f", and {len(runs) - _SHOWN} more" if len(runs) > _SHOWN else ""
    return "; ".join(_named(run) for run in runs[:_SHOWN]) + more


# -------------------------------------------------------- campaign and plan


def _recipe(folder: Path, found: _Findings) -> tuple[Campaign | None, str | None]:
    """Read the stored campaign.

    Returns:
        The campaign when it matches the config hash recorded with it, else
        ``None``, and the recorded config hash when the file states one.
    """
    path = folder / CAMPAIGN_FILE
    if not path.is_file():
        found.add(
            "recipe-missing",
            f"{CAMPAIGN_FILE} is missing: the directory does not hold the campaign its runs "
            "were made from.",
        )
        return None, None
    try:
        document = read_json(path)
    except (SerializationError, OSError) as error:
        found.add("recipe-unreadable", f"{CAMPAIGN_FILE} cannot be read: {error}.")
        return None, None
    if not isinstance(document, Mapping) or document.get("schema") != CAMPAIGN_SCHEMA:
        found.add("recipe-unreadable", f"{CAMPAIGN_FILE} is not a {CAMPAIGN_SCHEMA!r} document.")
        return None, None
    stated = document.get("config_hash")
    recorded = stated if isinstance(stated, str) else None
    base = document.get("base")
    try:
        campaign = campaign_from_dict(
            document.get("campaign", {}), base=None if base is None else Path(str(base))
        )
    except ConfigError as error:
        found.add("recipe-unreadable", f"{CAMPAIGN_FILE} holds no campaign: {error}.")
        return None, recorded
    if campaign.config_hash != recorded:
        found.add(
            "recipe-altered",
            f"{CAMPAIGN_FILE} no longer matches its recorded config hash: it records "
            f"{_short(recorded)} and its content gives {_short(campaign.config_hash)}, so the "
            "campaign was changed after it was stored.",
        )
        return None, recorded
    return campaign, recorded


def _check_plan(
    plan: Plan,
    campaign: Campaign | None,
    recorded: str | None,
    found: _Findings,
    replan: bool,
    registry: SolverRegistry | None,
    families: FamilyRegistry | None,
) -> None:
    if recorded is not None and plan.config_hash != recorded:
        found.add(
            "plan-foreign",
            f"{PLAN_FILE} states config hash {_short(plan.config_hash)} and {CAMPAIGN_FILE} "
            f"records {_short(recorded)}: the plan is no plan of the stored campaign.",
        )
        return
    if campaign is None:
        return
    if not replan:
        found.add(
            "plan-not-compared",
            f"{PLAN_FILE} was not compared with the plan the stored campaign gives.",
        )
        return
    try:
        derived = planning.plan(campaign, registry=registry, families=families)
    except (DispatchAtlasError, OSError) as error:
        found.add(
            "plan-not-derived",
            f"The plan could not be derived again from {CAMPAIGN_FILE}, so {PLAN_FILE} is not "
            f"compared with it: {error}.",
        )
        return
    differences = _plan_differences(plan, derived)
    if differences:
        found.add(
            "plan-differs",
            f"{PLAN_FILE} is not the plan that {CAMPAIGN_FILE} gives on this installation: the "
            "plan file was changed, or the problems, the metadata of the solvers, or the "
            "library that plans differ from those of the last session.",
            differences,
        )


def _plan_differences(stored: Plan, derived: Plan) -> list[str]:
    """Return how the plan a directory holds differs from the plan derived again."""
    differences: list[str] = []
    left = [run for run in stored.runs if not derived.has_run(run.run_id)]
    entered = [run for run in derived.runs if not stored.has_run(run.run_id)]
    reseeded = [
        run
        for run in stored.runs
        if derived.has_run(run.run_id) and derived.run(run.run_id).seed != run.seed
    ]
    for runs, what in (
        (left, "runs of the stored plan that are not in the derived plan"),
        (entered, "runs of the derived plan that are not in the stored plan"),
        (reseeded, "runs with another seed in the derived plan"),
    ):
        if runs:
            differences.append(f"{what} ({len(runs)}): {_first(runs)}")
    digests = {problem.id: problem.digest for problem in derived.problems}
    changed = [
        problem.id
        for problem in stored.problems
        if digests.get(problem.id, problem.digest) != problem.digest
    ]
    if changed:
        differences.append("problems with another content digest: " + ", ".join(changed))
    entries = {entry.name: entry.to_dict() for entry in derived.solvers}
    for entry in stored.solvers:
        if entry.name not in entries:
            continue
        held, again = entry.to_dict(), entries[entry.name]
        compared = [
            *_ENTRY_FIELDS,
            *(name for name in _OPTIONAL_ENTRY_FIELDS if held.get(name) is not None),
        ]
        fields = [name for name in compared if held.get(name) != again.get(name)]
        if fields:
            differences.append(f"solver {entry.name} differs in " + ", ".join(fields))
    excluded = {(item.problem, item.solver) for item in stored.exclusions}
    again_excluded = {(item.problem, item.solver) for item in derived.exclusions}
    if excluded != again_excluded:
        differences.append(
            "pairs of problem and solver excluded in one plan and not in the other: "
            f"{len(excluded ^ again_excluded)}"
        )
    return differences


def _sessions(folder: Path, found: _Findings) -> tuple[SessionRecord, ...] | None:
    try:
        return read_sessions(folder)
    except (SerializationError, OSError) as error:
        found.add(
            "sessions-unreadable",
            f"A session record cannot be read, so the sessions are not compared: {error}.",
        )
        return None


def _max_attempts(campaign: Campaign | None, sessions: Sequence[SessionRecord] | None) -> int:
    """Return the attempts a run gets, from the campaign or else from the last session."""
    if campaign is not None:
        return campaign.execution.max_attempts
    return sessions[-1].execution.max_attempts if sessions else 1


# --------------------------------------------------------------------- runs


def _states(folder: Path, plan: Plan, ledger: Ledger, max_attempts: int) -> dict[str, str]:
    """Return what the directory holds for each planned run, by run id and in plan order."""
    invalid = {item.run_id for item in ledger.stale if _at_record_path(item, folder)}
    aside = {item.run_id for item in ledger.kept}
    states: dict[str, str] = {}
    for run in plan.runs:
        failures = ledger.failures.get(run.run_id, ())
        if run.run_id in ledger.completed:
            states[run.run_id] = "completed"
        elif run.run_id in aside:
            states[run.run_id] = "set_aside"
        elif run.run_id in invalid:
            states[run.run_id] = "invalid"
        elif failures:
            states[run.run_id] = "failed" if settled(failures, max_attempts) else "unsettled"
        else:
            states[run.run_id] = "unattempted"
    return states


def _at_record_path(stale: Stale, folder: Path) -> bool:
    return stale.path.is_relative_to(folder / RUNS)


def _cells(
    plan: Plan, states: Mapping[str, str], protocol: str
) -> dict[tuple[str, str], Counter[str]]:
    """Count the runs of each solver and problem under a protocol by state, in plan order."""
    cells: dict[tuple[str, str], Counter[str]] = {}
    for run in plan.runs:
        if run.protocol == protocol:
            cells.setdefault((run.solver, run.problem), Counter())[states[run.run_id]] += 1
    return cells


def _check_runs(plan: Plan, ledger: Ledger, states: Mapping[str, str], found: _Findings) -> None:
    for protocol in plan.protocols:
        name = protocol.name
        cells = _cells(plan, states, name)
        planned = sum(sum(counts.values()) for counts in cells.values())
        for state, code, what in (
            ("unattempted", "runs-unattempted", "have no record and no failed attempt"),
            (
                "unsettled",
                "runs-unsettled",
                "have failed attempts that do not settle them, so they are owed another attempt",
            ),
        ):
            short = {cell: counts for cell, counts in cells.items() if counts[state]}
            if short:
                total = sum(counts[state] for counts in short.values())
                found.add(
                    code,
                    f"Under protocol {name}, {total} of {planned} planned runs {what}, in "
                    f"{_count(len(short), 'cell')} of solver and problem.",
                    (
                        f"{solver} on {problem}: {counts[state]} of {sum(counts.values())} runs"
                        for (solver, problem), counts in short.items()
                    ),
                    name,
                )
        kinds: dict[str, Counter[str]] = {}
        for run in plan.runs:
            if run.protocol == name and states[run.run_id] == "failed":
                last = ledger.failures[run.run_id][-1]
                kinds.setdefault(run.solver, Counter())[last.error.kind.value] += 1
        if kinds:
            total = sum(sum(counts.values()) for counts in kinds.values())
            found.add(
                "runs-failed",
                f"Under protocol {name}, {total} of {planned} planned runs never completed and "
                "are settled by their failures. The items count them by solver and by the kind "
                "of error of their last attempt.",
                (f"{solver}: {_tally(counts)}" for solver, counts in kinds.items()),
                name,
            )
        partial = {
            cell: counts
            for cell, counts in cells.items()
            if 0 < counts["completed"] < sum(counts.values())
        }
        if partial:
            found.add(
                "cells-partial",
                f"Under protocol {name}, {len(partial)} of {len(cells)} cells of solver and "
                "problem have a record for some of their planned runs and not for all.",
                (
                    f"{solver} on {problem}: {counts['completed']} of {sum(counts.values())} "
                    "runs have a record"
                    for (solver, problem), counts in partial.items()
                ),
                name,
            )
        by_solver: dict[str, Counter[str]] = {}
        for (solver, _), counts in cells.items():
            by_solver.setdefault(solver, Counter()).update(counts)
        empty = {solver: counts for solver, counts in by_solver.items() if not counts["completed"]}
        if empty:
            found.add(
                "solver-without-records",
                f"Under protocol {name}, {len(empty)} of {len(by_solver)} solvers with planned "
                "runs have no record.",
                (f"{solver}: {_tally(counts)}" for solver, counts in empty.items()),
                name,
            )
    running = {run.solver for run in plan.runs}
    idle = [entry.name for entry in plan.solvers if entry.name not in running]
    if idle:
        excluded = Counter(item.solver for item in plan.exclusions)
        found.add(
            "solver-excluded-everywhere",
            f"{len(idle)} of {len(plan.solvers)} solvers have no planned run: the plan excludes "
            "them from every problem.",
            (f"{solver}: excluded from {_count(excluded[solver], 'problem')}" for solver in idle),
        )
    if plan.exclusions:
        found.add(
            "pairs-excluded",
            "Pairs of problem and solver that the plan excludes, which have no run and are "
            f"not missing: {len(plan.exclusions)}.",
            (item.reason for item in plan.exclusions),
        )


# -------------------------------------------------------------------- files


def _check_files(folder: Path, plan: Plan, ledger: Ledger, found: _Findings) -> None:
    kinds: dict[str, list[str]] = {}
    for stale in ledger.stale:
        where = _relative(stale.path, folder)
        if not _at_record_path(stale, folder):
            kinds.setdefault("failures-foreign", []).append(where)
            continue
        code = _record_kind(stale, plan)
        reason = stale.reason.removesuffix(f" ({stale.path})").removeprefix(f"{stale.path}: ")
        kinds.setdefault(code, []).append(f"{where}: {reason}")
    messages = {
        "record-unreadable": (
            "Files at the paths of planned runs that cannot be read as run records"
        ),
        "record-foreign": (
            "Records at the paths of planned runs that belong to another configuration or to "
            "another version of their problem, so that their runs have no record"
        ),
        "record-hash-mismatch": (
            "Records that no longer match a hash stored with them, and so were changed after "
            "they were written"
        ),
        "record-differs-from-plan": (
            "Records that state something else than the plan states for their run"
        ),
        "failures-foreign": (
            "Files at the paths of failed attempts that belong to another plan and count as no "
            "attempt"
        ),
    }
    for code, text in messages.items():
        if code in kinds:
            found.add(code, f"{text}: {len(kinds[code])}.", kinds[code])
    if ledger.kept:
        found.add(
            "records-set-aside",
            "Planned runs whose valid record lies under superseded/ and not at its path, "
            f"where no reader finds it: {len(ledger.kept)}. The next session of the campaign "
            "moves such a record back.",
            (
                f"{_relative(item.kept, folder)} belongs at {_relative(item.path, folder)}"
                for item in ledger.kept
            ),
        )
    if ledger.unreadable:
        found.add(
            "failures-unreadable",
            "Files under failures/ that cannot be read as records of failed attempts and "
            f"count as no attempt: {len(ledger.unreadable)}.",
            (_relative(path, folder) for path in ledger.unreadable),
        )
    strays = read_strays(folder)
    for paths, code, text in (
        (
            strays.unplanned,
            "records-unplanned",
            "Records under runs/ of runs the plan does not hold, which no reader returns",
        ),
        (
            strays.unreadable,
            "files-unreadable",
            "Files under runs/, at no path of a planned run, that cannot be read as run records",
        ),
        (strays.other, "files-stray", "Files under runs/ that are no JSON documents"),
    ):
        if paths:
            found.add(code, f"{text}: {len(paths)}.", (_relative(path, folder) for path in paths))


def _record_kind(stale: Stale, plan: Plan) -> str:
    """Return the code for a file at a planned run's path that is no valid record of the run."""
    try:
        record = RunRecord.from_dict(read_json(stale.path))
    except (SerializationError, OSError):
        return "record-unreadable"
    if record.config_hash != plan.config_hash or record.run_id != stale.run_id:
        return "record-foreign"
    return "record-hash-mismatch" if record.verify() else "record-differs-from-plan"


# ------------------------------------------------------------------ records


def _check_records(
    plan: Plan, records: Mapping[str, RunRecord], overrun_tolerance: float, found: _Findings
) -> None:
    held: Counter[tuple[str, str]] = Counter()
    shown: dict[str, Counter[tuple[str, str]]] = {}
    examples: dict[str, dict[str, None]] = {}
    unhashed = 0
    for record in records.values():
        run = plan.run(record.run_id)
        cell = (run.protocol, run.solver)
        held[cell] += 1
        unhashed += record.identity_hash is None
        facts = observations(
            record.outcome,
            record.protocol.stop,
            record.provenance.wall_seconds,
            iterative=record.solver.iterative,
        )
        codes = {_TRACE_CODES[kind] for kind, _ in facts if kind in _TRACE_CODES}
        derived = {text for _, text in facts}
        other = [note for note in record.notes if note not in derived]
        if other:
            codes.add("records-noted")
            examples.setdefault(run.protocol, {}).setdefault(other[0])
        for code in codes:
            shown.setdefault(code, Counter())[cell] += 1
    texts = {
        "trace-time-axis": (
            "records whose trace holds a negative time, times or evaluation counts that "
            "decrease, or a time beyond the elapsed time of the run, so that it cannot be read "
            "against time"
        ),
        "trace-contradicts-outcome": (
            "records whose trace ends at another value than the value of the outcome"
        ),
        "record-without-trace": ("records of iterative solvers that hold a schedule and no trace"),
        "clock-disagreement": (
            "records whose elapsed time differs from the engine's wall time, or whose run "
            "stopped for its time limit before the limit"
        ),
        "records-noted": (
            "records with notes of the engine that the check does not derive from them"
        ),
    }
    for protocol in plan.protocols:
        name = protocol.name
        for code, text in texts.items():
            cells = {
                solver: count
                for (under, solver), count in shown.get(code, Counter()).items()
                if under == name
            }
            if cells:
                items = [
                    f"{solver}: {count} of {held[name, solver]} records"
                    for solver, count in cells.items()
                ]
                if code == "records-noted":
                    items += [f"for example: {note}" for note in list(examples[name])[:_SHOWN]]
                found.add(
                    code,
                    f"Under protocol {name}, {text}: {sum(cells.values())}.",
                    items,
                    name,
                )
    if unhashed:
        found.add(
            "records-without-identity-hash",
            f"{unhashed} of {len(records)} records state no identity hash, so no hash covers "
            "their problem digest, solver, protocol, replicate, and seed; they are compared "
            "with the plan field by field.",
        )
    outcomes = fairness.assess_outcomes(plan, records.values(), overrun_tolerance=overrun_tolerance)
    for item in outcomes.protocols:
        limit = plan.protocol(item.protocol).stop.time_limit
        if item.overruns and limit is not None:
            worst = max(overrun.worst for overrun in item.overruns.values())
            found.add(
                "budget-overrun",
                f"Under protocol {item.protocol}, whose time limit is {limit:g} s, completed "
                f"runs of {len(item.overruns)} solvers took more wall time than the limit and a "
                f"tolerance of {outcomes.overrun_tolerance:g} of it allow; the longest took "
                f"{worst:.3g} times the limit.",
                (
                    f"{solver}: {overrun.over} of {overrun.runs} runs, the longest "
                    f"{overrun.worst:.3g} times the limit"
                    for solver, overrun in item.overruns.items()
                ),
                item.protocol,
            )
        if item.divergent:
            found.add(
                "stop-reasons-diverge",
                f"Under protocol {item.protocol}, the completed runs of the iterative solvers "
                "ended for different reasons, so the protocol did not bound them alike.",
                (f"{solver}: {_tally(reasons)}" for solver, reasons in item.stop_reasons.items()),
                item.protocol,
            )


def _check_hard_limits(
    plan: Plan, records: Mapping[str, RunRecord], ledger: Ledger, found: _Findings
) -> None:
    limits: dict[str, Counter[float | None]] = {}
    attempts: list[RunRecord | FailureRecord] = list(records.values())
    for failures in ledger.failures.values():
        attempts.extend(failures)
    for attempt in attempts:
        under = plan.run(attempt.run_id).protocol
        limits.setdefault(under, Counter())[attempt.provenance.hard_limit] += 1
    for protocol in plan.protocols:
        counted = limits.get(protocol.name, Counter())
        if len(counted) > 1:
            ordered = sorted(counted.items(), key=lambda item: (item[0] is None, item[0] or 0.0))
            found.add(
                "hard-limits-differ",
                f"Under protocol {protocol.name}, the attempts ran under {len(counted)} "
                "different hard limits, so the engine did not end them alike.",
                (
                    ("no hard limit" if limit is None else f"{limit:g} s")
                    + f": {_count(count, 'attempt')}"
                    for limit, count in ordered
                ),
                protocol.name,
            )


# ----------------------------------------------------------------- sessions


def _check_sessions(
    plan: Plan,
    sessions: Sequence[SessionRecord],
    records: Mapping[str, RunRecord],
    ledger: Ledger,
    found: _Findings,
) -> None:
    numbers = {record.provenance.session for record in records.values()}
    for failures in ledger.failures.values():
        numbers.update(failure.provenance.session for failure in failures)
    by_number = {session.session: session for session in sessions}
    absent = sorted(number for number in numbers if number not in by_number)
    if absent:
        found.add(
            "sessions-missing",
            "Sessions that the attempts of the directory name and that it holds no record "
            f"of, so that the code and the settings they ran under are not known: {len(absent)}.",
            (f"session {number}" for number in absent),
        )
    executing = [by_number[number] for number in sorted(numbers) if number in by_number]
    _compare_sessions(executing, found)
    changed = [
        session
        for session in sessions
        if session.plan_left
        or session.plan_entered
        or session.plan_reseeded
        or session.plan_solvers
    ]
    if changed:
        found.add(
            "plan-changed",
            "Sessions whose plan differs from the plan before them while the campaign file "
            "stayed the same, because the problems, the metadata of the solvers, or the library "
            f"that plans changed: {len(changed)}.",
            (
                f"session {session.session}: runs that left {len(session.plan_left)}, "
                f"entered {len(session.plan_entered)}, got another seed "
                f"{len(session.plan_reseeded)}; changed fields of solver entries "
                f"{len(session.plan_solvers)}"
                for session in changed
            ),
        )
    stated = [session for session in sessions if session.plan_digest is not None]
    if not stated:
        found.add(
            "plan-digest-unstated",
            f"No session record states the digest of the plan it executed, so {PLAN_FILE} is "
            "not compared with the plan of the last session.",
        )
    elif stated[-1].plan_digest != plan.runs_digest:
        last = stated[-1]
        found.add(
            "plan-digest-differs",
            f"{PLAN_FILE} holds runs and seeds with digest {_short(plan.runs_digest)}, and "
            f"session {last.session}, the last that states one, executed a plan with digest "
            f"{_short(last.plan_digest)}: the plan file was changed after that session, or "
            "the session ended before it replaced the plan file.",
        )
    if sessions and sessions[-1].status is not SessionStatus.FINISHED:
        last = sessions[-1]
        found.add(
            "session-unfinished",
            f"Session {last.session}, the last one, is recorded as {last.status.value} and not "
            "as finished.",
        )


def _compare_sessions(executing: Sequence[SessionRecord], found: _Findings) -> None:
    """Report how the sessions that executed counted attempts differ from the first of them."""
    later = executing[1:]
    if not later:
        return
    first = executing[0]
    grouped: dict[str, list[str]] = {}
    for session in later:
        changes = session_changes(first, session.execution, session.workers, session.environment)
        for change in changes:
            name = str(change["field"])
            if change["kind"] == "execution":
                code = "sessions-differ-execution"
            elif change["kind"] == "machine":
                code = "sessions-differ-machine"
            elif name.startswith("environment.plugins"):
                code = "sessions-differ-plugins"
            else:
                code = "sessions-differ-code"
            grouped.setdefault(code, []).append(
                f"{name}: {_quoted(change['earlier'])} in session {first.session}, "
                f"{_quoted(change['now'])} in session {session.session}"
            )
    texts = {
        "sessions-differ-code": "the Python version, the packages, the library, or a git state",
        "sessions-differ-plugins": "their plugins",
        "sessions-differ-execution": "their execution settings",
        "sessions-differ-machine": "their machine",
    }
    for code, text in texts.items():
        if code in grouped:
            found.add(
                code,
                f"The {len(executing)} sessions that executed the attempts of the directory "
                f"differ in {text}. Differences from session {first.session}, the first of "
                f"them: {len(grouped[code])}.",
                grouped[code],
            )


def _quoted(value: object) -> str:
    text = repr(value)
    return text if len(text) <= _VALUE_CHARACTERS else text[: _VALUE_CHARACTERS - 3] + "..."


# -------------------------------------------------------------- replication


def _check_replicates(plan: Plan, states: Mapping[str, str], floor: int, found: _Findings) -> None:
    stochastic = {entry.name for entry in plan.solvers if entry.stochastic}
    for protocol in plan.protocols:
        below: dict[str, list[int]] = {}
        replicated = 0
        for (solver, _), counts in _cells(plan, states, protocol.name).items():
            if solver not in stochastic and sum(counts.values()) == 1:
                continue
            replicated += 1
            if counts["completed"] < floor:
                below.setdefault(solver, []).append(counts["completed"])
        if below:
            total = sum(len(cells) for cells in below.values())
            found.add(
                "replicates-below-floor",
                f"Under protocol {protocol.name}, {total} of {replicated} cells of replicated "
                f"solvers hold fewer than {floor} completed replicates.",
                (
                    f"{solver}: {_count(len(cells), 'cell')}, the fewest with "
                    f"{_count(min(cells), 'completed replicate')}"
                    for solver, cells in below.items()
                ),
                protocol.name,
            )


def _check_seeds(plan: Plan, records: Mapping[str, RunRecord], found: _Findings) -> None:
    """Compare the seeds of the runs: those of the records, and the plan's for the other runs."""
    replicates: dict[tuple[str, int], set[int]] = {}
    seeds: dict[tuple[str, int], set[int]] = {}
    for run in plan.runs:
        record = records.get(run.run_id)
        seed = run.seed if record is None else record.seed
        replicates.setdefault((run.problem, seed), set()).add(run.replicate)
        seeds.setdefault((run.problem, run.replicate), set()).add(seed)
    reused = {key: value for key, value in replicates.items() if len(value) > 1}
    if reused:
        found.add(
            "seed-reused",
            "Seeds that serve two or more replicates of one problem, which are then not "
            f"independent: {len(reused)}.",
            (
                f"{problem}: seed {seed} for replicates "
                + ", ".join(str(number) for number in sorted(numbers))
                for (problem, seed), numbers in reused.items()
            ),
        )
    unpaired = {key: value for key, value in seeds.items() if len(value) > 1}
    if unpaired:
        found.add(
            "seeds-unpaired",
            "Replicates of a problem that ran with more than one seed, so that their runs "
            f"do not share a seed across solvers and protocols: {len(unpaired)}.",
            (
                f"{problem}, replicate {replicate}: seeds "
                + ", ".join(str(seed) for seed in sorted(values))
                for (problem, replicate), values in unpaired.items()
            ),
        )
