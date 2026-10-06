"""The result store: where a campaign's files live, and how to read them back.

Layout under an output root::

    <root>/<campaign id>/
        campaign.json      the campaign document, its base directory, its config hash
        plan.json          the plan, with parities and the fairness assessment
        sessions/<n>.json  one session record per invocation of the runner, with the
                           digest of its plan and the runs that left or entered it
        runs/<solver>/<problem>/r<replicate:03d>-<protocol>.json
        failures/<solver>/<problem>/r<replicate:03d>-<protocol>.attempt-<n>.json
        superseded/<solver>/<problem>/r<replicate:03d>-<protocol>.<run id>.json
                           files moved aside before their runs were repeated
        logs/session-<n>.log  what each session logged, with UTC time stamps
        .lock              locked while a runner works on the campaign

``<solver>`` is the solver's name, its label or else its id. Every document
is written atomically, with sorted keys, two-space indentation, ASCII text,
and a final newline, and a value that is not finite is refused.

The readers :func:`read_campaign`, :func:`read_plan`, :func:`read_runs`,
:func:`read_failures`, and :func:`read_sessions` return typed records. They
only return records of the plan in ``plan.json``: a file left by an earlier
plan, whose run id or config hash differs, is skipped with a warning. A record
whose content no longer matches its result hash or its identity hash raises an
error. :func:`read_runs` also warns about every other file under ``runs/``,
which :func:`read_strays` lists: records of runs the plan does not hold, files
that cannot be read, and files that are no records.

:class:`Store` is the writing side, used by the runner: it locks the
directory, refuses to mix configurations, and settles which runs remain. It
never replaces a file that is not the valid record of a planned run: before
the run is repeated, such a file is moved under ``superseded/``, where nothing
reads it and nothing replaces it.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from dispatchatlas.campaigns.config import Campaign, check_name
from dispatchatlas.campaigns.io import campaign_from_dict, campaign_to_dict
from dispatchatlas.campaigns.planning import Plan, PlannedRun
from dispatchatlas.campaigns.records import (
    FailureRecord,
    PlanChange,
    RunRecord,
    SessionRecord,
    settled,
)
from dispatchatlas.core.errors import ConfigError, SerializationError
from dispatchatlas.core.io import atomic_write_json, read_json
from dispatchatlas.core.logs import get_logger

__all__ = [
    "CAMPAIGN_FILE",
    "CAMPAIGN_SCHEMA",
    "FAILURES",
    "LOGS",
    "PLAN_FILE",
    "RUNS",
    "SESSIONS",
    "SUPERSEDED",
    "Ledger",
    "Stale",
    "Store",
    "Strays",
    "campaign_directory",
    "failure_path",
    "read_campaign",
    "read_failures",
    "read_plan",
    "read_runs",
    "read_sessions",
    "read_strays",
    "recorded_config_hash",
    "run_path",
]

CAMPAIGN_SCHEMA = "dispatchatlas.campaign/1"
CAMPAIGN_FILE = "campaign.json"
PLAN_FILE = "plan.json"
SESSIONS = "sessions"
RUNS = "runs"
FAILURES = "failures"
SUPERSEDED = "superseded"
LOGS = "logs"
_LOCK = ".lock"

_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
"""What a run id must match to become part of a file name."""

_SHOWN = 3
"""Runs a warning about a changed plan names before it counts the rest."""

_LOG = get_logger(__name__)


def campaign_directory(root: str | Path, campaign_id: str) -> Path:
    """Return the directory of a campaign under an output root."""
    return Path(root) / check_name(campaign_id, "campaign id")


def run_path(directory: Path, solver: str, problem: str, replicate: int, protocol: str) -> Path:
    """Return the path of the record of one run."""
    return directory / RUNS / solver / problem / f"r{replicate:03d}-{protocol}.json"


def failure_path(
    directory: Path, solver: str, problem: str, replicate: int, protocol: str, attempt: int
) -> Path:
    """Return the path of the record of one failed attempt."""
    name = f"r{replicate:03d}-{protocol}.attempt-{attempt}.json"
    return directory / FAILURES / solver / problem / name


def _planned_path(directory: Path, run: PlannedRun) -> Path:
    return run_path(directory, run.solver, run.problem, run.replicate, run.protocol)


def recorded_config_hash(directory: str | Path) -> str | None:
    """Return the config hash a campaign directory records, or ``None`` if it has none.

    A missing or unreadable ``campaign.json`` records none.
    """
    path = Path(directory) / CAMPAIGN_FILE
    if not path.is_file():
        return None
    try:
        document = read_json(path)
    except SerializationError:
        return None
    found = document.get("config_hash") if isinstance(document, Mapping) else None
    return found if isinstance(found, str) else None


# ------------------------------------------------------------------ readers


def read_campaign(directory: str | Path) -> Campaign:
    """Read the campaign a store was created from.

    Relative paths in its sources refer to the directory recorded with it,
    so the campaign can be planned again from the store alone.

    Raises:
        SerializationError: If ``campaign.json`` is missing or malformed, or
            the campaign no longer matches its recorded config hash.
    """
    path = Path(directory) / CAMPAIGN_FILE
    document = _read(path)
    if not isinstance(document, Mapping) or document.get("schema") != CAMPAIGN_SCHEMA:
        raise SerializationError(f"{path} is not a {CAMPAIGN_SCHEMA!r} document")
    base = document.get("base")
    try:
        campaign = campaign_from_dict(
            document.get("campaign", {}), base=None if base is None else Path(str(base))
        )
    except ConfigError as error:
        raise SerializationError(f"{path}: {error}") from error
    if campaign.config_hash != document.get("config_hash"):
        raise SerializationError(f"{path} does not match its recorded config hash")
    return campaign


def read_plan(directory: str | Path) -> Plan:
    """Read the plan of a campaign.

    Raises:
        SerializationError: If ``plan.json`` is missing or malformed.
    """
    path = Path(directory) / PLAN_FILE
    try:
        return Plan.from_dict(_read(path))
    except SerializationError as error:
        raise SerializationError(f"{path}: {error}") from error


def read_runs(directory: str | Path, *, verify: bool = True) -> tuple[RunRecord, ...]:
    """Read every completed run of a campaign's plan, in plan order.

    Args:
        directory: The campaign directory.
        verify: Check that every record still matches its result hash,
            schedule digest, and identity hash.

    Every other file under ``runs/`` is warned about, by kind and with the
    first names (:func:`read_strays`).

    Raises:
        SerializationError: If the plan or a record is malformed, or, with
            ``verify``, a record was modified after it was written.
    """
    folder = Path(directory)
    plan = read_plan(folder)
    records: list[RunRecord] = []
    for run in plan.runs:
        path = _planned_path(folder, run)
        if not path.is_file():
            continue
        record = _parse(path, RunRecord.from_dict)
        stale = _foreign(record.run_id, record.config_hash, run.run_id, plan.config_hash)
        if stale:
            _LOG.warning("skipping %s: %s", path, stale)
            continue
        if verify and (issues := record.verify()):
            raise SerializationError(f"{path}: {'; '.join(issues)}")
        records.append(record)
    strays = _strays(folder, plan)
    for paths, what in (
        (strays.unplanned, "record(s) of runs the plan does not hold, which are not read"),
        (strays.unreadable, "file(s) that cannot be read as run records"),
        (strays.other, "file(s) that are no records"),
    ):
        if paths:
            shown = ", ".join(path.relative_to(folder).as_posix() for path in paths[:_SHOWN])
            more = f" and {len(paths) - _SHOWN} more" if len(paths) > _SHOWN else ""
            _LOG.warning("%s holds %d %s: %s%s", folder / RUNS, len(paths), what, shown, more)
    return tuple(records)


@dataclass(frozen=True)
class Strays:
    """The files under ``runs/`` that are at no path of a planned run.

    Attributes:
        unplanned: Run records, such as those of runs that left the plan.
        unreadable: JSON files that cannot be read as run records.
        other: Files that are no JSON documents, such as a temporary file
            an interrupted write left behind.
    """

    unplanned: tuple[Path, ...] = ()
    unreadable: tuple[Path, ...] = ()
    other: tuple[Path, ...] = ()

    def __bool__(self) -> bool:
        """Whether there is any such file."""
        return bool(self.unplanned or self.unreadable or self.other)


def read_strays(directory: str | Path) -> Strays:
    """List the files under ``runs/`` that are at no path of a run of the plan.

    No reader returns such a file, and no session replaces it.

    Raises:
        SerializationError: If the plan is missing or malformed.
    """
    folder = Path(directory)
    return _strays(folder, read_plan(folder))


def _strays(folder: Path, plan: Plan) -> Strays:
    planned = {_planned_path(folder, run) for run in plan.runs}
    unplanned: list[Path] = []
    unreadable: list[Path] = []
    other: list[Path] = []
    for path in sorted((folder / RUNS).rglob("*")):
        if path in planned or not path.is_file():
            continue
        if path.suffix != ".json" or path.name.startswith("."):
            other.append(path)
            continue
        try:
            _parse(path, RunRecord.from_dict)
        except SerializationError:
            unreadable.append(path)
        else:
            unplanned.append(path)
    return Strays(tuple(unplanned), tuple(unreadable), tuple(other))


def read_failures(directory: str | Path) -> tuple[FailureRecord, ...]:
    """Read every failed attempt of a campaign's plan, by plan order and attempt.

    Raises:
        SerializationError: If the plan or a record is malformed.
    """
    folder = Path(directory)
    plan = read_plan(folder)
    order = {run.run_id: index for index, run in enumerate(plan.runs)}
    records: list[FailureRecord] = []
    for path in sorted((folder / FAILURES).rglob("*.json")):
        record = _parse(path, FailureRecord.from_dict)
        if record.run_id not in order or record.config_hash != plan.config_hash:
            _LOG.warning("skipping %s: it belongs to another plan", path)
            continue
        records.append(record)
    records.sort(key=lambda item: (order[item.run_id], item.provenance.attempt))
    return tuple(records)


def read_sessions(directory: str | Path) -> tuple[SessionRecord, ...]:
    """Read every session record of a campaign, by session number.

    Raises:
        SerializationError: If a record is malformed.
    """
    folder = Path(directory) / SESSIONS
    records = [_parse(path, SessionRecord.from_dict) for path in folder.glob("*.json")]
    return tuple(sorted(records, key=lambda item: item.session))


# ------------------------------------------------------------------- writer


@dataclass(frozen=True)
class Stale:
    """A file at the path of a planned run that is no valid record of the run.

    Attributes:
        run_id: Id of the planned run.
        problem: Id of the problem of the planned run.
        path: The file.
        found: The run id the file states, or ``run_id`` when it states
            none that can be part of a file name.
        reason: Why the file does not count as the record of the run.
        changed_problem: Whether the file is a record made on another
            version of the problem of the run.
    """

    run_id: str
    problem: str
    path: Path
    found: str
    reason: str
    changed_problem: bool = False


@dataclass(frozen=True)
class Ledger:
    """What a store already holds for a plan.

    Attributes:
        completed: Ids of the runs with a valid record.
        failures: Failed attempts of the plan's runs, by run id, in attempt order.
        stale: The files at the paths of planned runs that do not count as
            their records, in plan order.
    """

    completed: frozenset[str] = frozenset()
    failures: Mapping[str, tuple[FailureRecord, ...]] = field(default_factory=dict)
    stale: tuple[Stale, ...] = ()

    def next_attempt(self, run_id: str) -> int:
        """Return the number of the next attempt of a run: one past its last failure."""
        return 1 + max(
            (failure.provenance.attempt for failure in self.failures.get(run_id, ())), default=0
        )

    def closed(self, run_id: str, max_attempts: int) -> bool:
        """Return whether failures settle a run that never completed.

        See :func:`~dispatchatlas.campaigns.records.settled` for the rule.
        """
        failures = self.failures.get(run_id, ())
        return run_id not in self.completed and settled(failures, max_attempts)

    def pending(
        self, plan: Plan, max_attempts: int, *, retry_failed: bool = False
    ) -> tuple[PlannedRun, ...]:
        """Return the runs of a plan that still need an attempt, in plan order.

        Args:
            plan: The plan.
            max_attempts: Attempts a run gets before a recoverable failure settles it.
            retry_failed: Also return runs that failures settled.
        """
        return tuple(
            run
            for run in plan.runs
            if run.run_id not in self.completed
            and (retry_failed or not self.closed(run.run_id, max_attempts))
        )


class Store:
    """The writing side of a campaign directory.

    Args:
        directory: The campaign directory, created when missing.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the campaign's lock, so two runners never work on it at once.

        The lock is an operating-system file lock, released when the process
        ends in any way. Where the platform offers none, nothing is locked.

        Raises:
            ConfigError: If another process holds the lock.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / _LOCK).open("a+b") as handle:
            if not _try_lock(handle):
                raise ConfigError(f"{self.directory} is in use by another campaign runner")
            try:
                yield
            finally:
                _unlock(handle)

    def open(self, campaign: Campaign, *, fresh: bool = False) -> None:
        """Prepare the directory for a campaign and record the campaign in it.

        Args:
            campaign: The campaign.
            fresh: Discard every result the directory holds first.

        Records are checked one by one when the runner reads them, so a
        ``campaign.json`` that cannot be read is simply replaced.

        Raises:
            ConfigError: If the directory holds results of another
                configuration and ``fresh`` is false.
        """
        if fresh:
            self._clear()
        found = recorded_config_hash(self.directory)
        if found is not None and found != campaign.config_hash:
            raise ConfigError(
                f"{self.directory} holds results of another configuration (config hash "
                f"{found[:12]}, now {campaign.config_hash[:12]}); pass fresh=True to discard "
                "them, or give the campaign another id"
            )
        atomic_write_json(
            self.directory / CAMPAIGN_FILE,
            {
                "schema": CAMPAIGN_SCHEMA,
                "config_hash": campaign.config_hash,
                "base": None if campaign.base is None else str(campaign.base.absolute()),
                "campaign": campaign_to_dict(campaign),
            },
        )

    def plan_change(self, plan: Plan) -> PlanChange:
        """Report how a plan differs from the plan the directory holds, and warn about it.

        A plan can change while the campaign file, and so the config hash,
        stays the same: the plan also follows the metadata the registered
        solvers declare, the content of the problems, and the library that
        plans. When the directory holds a plan of the same config hash, a
        warning names the problems whose content changed, and one names the
        runs that left the plan, the runs that entered it, the runs whose
        seed changed, and the fields of the solver entries that changed. The
        records of runs that left stay in the directory and are no longer
        read. Nothing is written, so the comparison gives the same answer
        until :meth:`write_plan` replaces the plan.

        Returns:
            The differences. Empty when the directory holds no plan, one
            that cannot be read, or one of another config hash.
        """
        if not (self.directory / PLAN_FILE).is_file():
            return PlanChange()
        try:
            previous = read_plan(self.directory)
        except SerializationError:
            return PlanChange()
        if previous.config_hash != plan.config_hash:
            return PlanChange()
        digests = {problem.id: problem.digest for problem in previous.problems}
        for problem in plan.problems:
            if digests.get(problem.id, problem.digest) != problem.digest:
                _LOG.warning(
                    "problem %s changed since the last session; its runs will run again",
                    problem.id,
                )
        left = [run for run in previous.runs if not plan.has_run(run.run_id)]
        entered = [run for run in plan.runs if not previous.has_run(run.run_id)]
        reseeded = [
            run
            for run in plan.runs
            if previous.has_run(run.run_id) and previous.run(run.run_id).seed != run.seed
        ]
        solvers = _entry_changes(previous, plan)
        change = PlanChange(
            tuple(run.to_dict() for run in left),
            tuple(run.to_dict() for run in entered),
            tuple(
                {**run.to_dict(), "earlier_seed": previous.run(run.run_id).seed} for run in reseeded
            ),
            tuple(solvers),
        )
        if not change:
            return change
        parts: list[str] = []
        effects: list[str] = []
        if left or entered:
            parts.append(
                f"{len(left)} run(s) left it and {len(entered)} entered it"
                f"{_named('; left', left)}{_named('; entered', entered)}"
            )
        if left:
            effects.append("records of runs that left stay in the directory and are no longer read")
        if reseeded:
            parts.append(f"the seed of {len(reseeded)} run(s) changed{_named('', reseeded)}")
            effects.append("a record made with the earlier seed still counts as its run's record")
        if solvers:
            parts.append(f"{len(solvers)} field(s) of the solver entries changed{_fields(solvers)}")
            effects.append("records made before keep the entries they were made with")
        _LOG.warning(
            "the plan of campaign %s changed under config hash %s: %s; the campaign file is "
            "the same, so the solvers' metadata, the problems, or the library changed%s",
            plan.campaign,
            plan.config_hash[:12],
            "; ".join(parts),
            "".join(f"; {effect}" for effect in effects),
        )
        return change

    def write_plan(self, plan: Plan) -> None:
        """Write ``plan.json``, replacing the plan the directory holds."""
        atomic_write_json(self.directory / PLAN_FILE, plan.to_dict())

    def ledger(self, plan: Plan) -> Ledger:
        """Read which runs of a plan are complete and which attempts failed.

        A file at the path of a planned run that cannot be read, belongs to
        another plan, or no longer matches its hashes does not count: it is
        listed as stale and its run is attempted again. Nothing is moved or
        written here; :meth:`set_aside` keeps the stale files before new
        records take their places.
        """
        completed: set[str] = set()
        stale: list[Stale] = []
        for run in plan.runs:
            path = _planned_path(self.directory, run)
            if not path.is_file():
                continue
            try:
                record = _parse(path, RunRecord.from_dict)
            except SerializationError as error:
                stale.append(Stale(run.run_id, run.problem, path, run.run_id, str(error)))
                continue
            foreign = _foreign(record.run_id, record.config_hash, run.run_id, plan.config_hash)
            issues = foreign or "; ".join(record.verify())
            if not issues:
                completed.add(run.run_id)
                continue
            found = record.run_id if _FILE_NAME.fullmatch(record.run_id) else run.run_id
            changed = bool(foreign) and record.config_hash == plan.config_hash
            reason = f"{issues} ({path})"
            stale.append(Stale(run.run_id, run.problem, path, found, reason, changed))
        failures: dict[str, list[FailureRecord]] = {}
        for path in sorted((self.directory / FAILURES).rglob("*.json")):
            try:
                failure = _parse(path, FailureRecord.from_dict)
            except SerializationError as error:
                _LOG.warning("ignoring %s: %s", path, error)
                continue
            if failure.config_hash == plan.config_hash and plan.has_run(failure.run_id):
                failures.setdefault(failure.run_id, []).append(failure)
        return Ledger(
            frozenset(completed),
            {
                run_id: tuple(sorted(items, key=lambda item: item.provenance.attempt))
                for run_id, items in failures.items()
            },
            tuple(stale),
        )

    def refuse_changed_problems(self, ledger: Ledger) -> None:
        """Refuse a directory that holds a record made on another version of a problem.

        Raises:
            ConfigError: If a stale file of the ledger is the record of a
                planned run made on another version of its problem.
        """
        changed = [item for item in ledger.stale if item.changed_problem]
        if changed:
            problems = ", ".join(sorted({item.problem for item in changed}))
            raise ConfigError(
                f"{self.directory} holds {len(changed)} record(s) made on another version of "
                f"problem(s) {problems}, and on_changed_problem is 'refuse'; restore the "
                "problem(s), set on_changed_problem to 'rerun' to keep those records aside "
                "and run their runs again, or give the campaign another id"
            )

    def set_aside(self, ledger: Ledger) -> tuple[Path, ...]:
        """Move the stale files of a ledger under ``superseded/``, and warn about each.

        A file keeps its path below ``runs/``, with the run id it states
        before ``.json``, and a number after that when the name is taken, so
        no file is replaced.

        Returns:
            Where the files are kept, in the order of the ledger.
        """
        kept: list[Path] = []
        for item in ledger.stale:
            relative = item.path.relative_to(self.directory / RUNS)
            stem = f"{item.path.stem}.{item.found}"
            target = self.directory / SUPERSEDED / relative.with_name(f"{stem}.json")
            number = 1
            while target.exists():
                number += 1
                target = target.with_name(f"{stem}.{number}.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            item.path.replace(target)
            kept.append(target)
            _LOG.warning(
                "will run %s again: %s; the earlier file is kept as %s",
                item.run_id,
                item.reason,
                target,
            )
        return tuple(kept)

    def last_session(self) -> SessionRecord | None:
        """Return the record of the latest session, or ``None`` when there is none.

        A session record that cannot be read is skipped with a warning, and
        the one before it is returned.
        """
        paths = [path for path in (self.directory / SESSIONS).glob("*.json") if path.stem.isdigit()]
        for path in sorted(paths, key=lambda item: int(item.stem), reverse=True):
            try:
                return _parse(path, SessionRecord.from_dict)
            except SerializationError as error:
                _LOG.warning("cannot compare with session %s: %s", path.stem, error)
        return None

    def log_path(self, session: int) -> Path:
        """Return the path of the log file of a session."""
        return self.directory / LOGS / f"session-{session}.log"

    def next_session(self) -> int:
        """Return the number of the next session: one past the highest recorded."""
        numbers = [
            int(path.stem)
            for path in (self.directory / SESSIONS).glob("*.json")
            if path.stem.isdigit()
        ]
        return 1 + max(numbers, default=0)

    def write_session(self, record: SessionRecord) -> None:
        """Write a session record."""
        atomic_write_json(self.directory / SESSIONS / f"{record.session}.json", record.to_dict())

    def write_run(self, record: RunRecord) -> Path:
        """Write the record of a completed run and return its path."""
        path = run_path(
            self.directory,
            record.solver.name,
            record.problem.id,
            record.replicate,
            record.protocol.name,
        )
        atomic_write_json(path, record.to_dict())
        return path

    def write_failure(self, record: FailureRecord) -> Path:
        """Write the record of a failed attempt and return its path."""
        path = failure_path(
            self.directory,
            record.solver.name,
            record.problem.id,
            record.replicate,
            record.protocol.name,
            record.provenance.attempt,
        )
        atomic_write_json(path, record.to_dict())
        return path

    def _clear(self) -> None:
        """Remove every result, plan, session, log, and campaign file of the directory."""
        runs = len(list((self.directory / RUNS).rglob("*.json")))
        failures = len(list((self.directory / FAILURES).rglob("*.json")))
        if runs or failures:
            _LOG.warning(
                "discarding %d run records and %d failure records in %s",
                runs,
                failures,
                self.directory,
            )
        for name in (RUNS, FAILURES, SESSIONS, SUPERSEDED, LOGS):
            shutil.rmtree(self.directory / name, ignore_errors=True)
        for name in (PLAN_FILE, CAMPAIGN_FILE):
            (self.directory / name).unlink(missing_ok=True)


# ------------------------------------------------------------------- helpers


def _read(path: Path) -> Any:
    try:
        return read_json(path)
    except FileNotFoundError:
        raise SerializationError(f"{path} does not exist") from None


def _parse[T](path: Path, build: Callable[[object], T]) -> T:
    """Read a record file, naming the file in any error."""
    try:
        return build(_read(path))
    except SerializationError as error:
        raise SerializationError(f"{path}: {error}") from error


def _named(heading: str, runs: list[PlannedRun]) -> str:
    """Name the first runs of a list after ``heading``, or return an empty string."""
    if not runs:
        return ""
    shown = ", ".join(
        f"{run.run_id} ({run.solver} on {run.problem} under {run.protocol}, "
        f"replicate {run.replicate})"
        for run in runs[:_SHOWN]
    )
    more = f" and {len(runs) - _SHOWN} more" if len(runs) > _SHOWN else ""
    return f"{heading}: {shown}{more}"


_UNSTATED = ("threads", "replicates")
"""The fields of a solver entry that read as ``None`` when a stored entry lacks them."""


def _entry_changes(previous: Plan, plan: Plan) -> list[dict[str, Any]]:
    """Return the fields by which the solver entries of two plans differ, in plan order.

    A solver only one plan holds is left out: its runs left or entered. A
    field the earlier entry does not state is not compared.
    """
    earlier = {entry.name: entry.to_dict() for entry in previous.solvers}
    changes: list[dict[str, Any]] = []
    for entry in plan.solvers:
        before = earlier.get(entry.name)
        if before is None:
            continue
        for key, now in entry.to_dict().items():
            was = before[key]
            if was != now and not (was is None and key in _UNSTATED):
                changes.append({"solver": entry.name, "field": key, "earlier": was, "now": now})
    return changes


def _fields(changes: list[dict[str, Any]]) -> str:
    """Name the first changed fields of solver entries, each with both values."""
    shown = ", ".join(
        f"{item['solver']}: {item['field']} {item['earlier']} -> {item['now']}"
        for item in changes[:_SHOWN]
    )
    more = f" and {len(changes) - _SHOWN} more" if len(changes) > _SHOWN else ""
    return f": {shown}{more}"


def _foreign(found_id: str, found_hash: str, run_id: str, config_hash: str) -> str:
    """Return why a record at a planned run's path is not that run's, or an empty string.

    With the configuration fixed, the path fixes every input of the run id but
    the problem digest, so a record of another run id solved another version
    of the problem.
    """
    if found_hash != config_hash:
        return "the record belongs to another configuration"
    if found_id != run_id:
        return "the record belongs to another version of its problem"
    return ""


def _try_lock(handle: IO[bytes]) -> bool:
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415 - Windows only

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl  # noqa: PLC0415 - POSIX only

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle: IO[bytes]) -> None:
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415 - Windows only

        with contextlib.suppress(OSError):
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl  # noqa: PLC0415 - POSIX only

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
