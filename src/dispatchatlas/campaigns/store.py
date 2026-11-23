"""The result store: where a campaign's files live, and how to read them back.

Layout under an output root::

    <root>/<campaign id>/
        campaign.json      the campaign document, its base directory, its config hash
        plan.json          the plan, with parities and the fairness assessment
        sessions/<n>.json  one session record per invocation of the runner, with the
                           digest of its plan and the runs that left or entered it
        runs/<solver>/<problem>/r<replicate:03d>-<protocol>.json
        failures/<solver>/<problem>/r<replicate:03d>-<protocol>.attempt-<n>.json
        superseded/runs/<solver>/<problem>/r<replicate:03d>-<protocol>.<run id>.json
        superseded/failures/<solver>/<problem>/r<replicate:03d>-<protocol>.attempt-<n>.<run id>.json
                           files moved aside when new records took their places
        logs/session-<n>.log  what each session logged, with UTC time stamps
        .lock              locked while a runner works on the campaign
        .lock.excl         on a file system without flock, the lock instead: created
                           exclusively by the runner, with its host, process id and
                           time, and removed when the runner leaves

``<solver>`` is the solver's name, its label or else its id. Every document
is written atomically, with sorted keys, two-space indentation, ASCII text,
and a final newline, and a value that is not finite is refused.

The readers :func:`read_campaign`, :func:`read_plan`, :func:`read_runs`,
:func:`read_failures`, and :func:`read_sessions` return typed records. They
only return records of the plan in ``plan.json``: a file left by an earlier
plan, whose run id or config hash differs, is skipped with a warning. A run
record raises an error when its content no longer matches its result hash or
its identity hash, or when it states another problem, solver, protocol,
replicate, or seed than the plan states for its run. :func:`read_runs` also
warns about every other file under ``runs/``, which :func:`read_strays` lists:
records of runs the plan does not hold, files that cannot be read, and files
that are no records.

:class:`Store` is the writing side, used by the runner: it locks the
directory, refuses to mix configurations, and settles which runs remain. It
replaces no record. A file at the path of a planned run that is not the valid
record of that run, such as a record of another version of its problem, stays
where it is until the run is repeated: when the repeated run gets its first
record, of its result or of a failed attempt, the file is moved under
``superseded/``, with the run id it states before ``.json``. The file of a
failed attempt is moved the same way when a new attempt takes its path. So a
session that repeats no run moves nothing, and a record counts again when its
problem is restored.
Nothing replaces a file under ``superseded/``, and the only reader of that
folder is the store itself, which moves a record back to its path when it is
the valid record of a planned run that has none.
"""

from __future__ import annotations

import contextlib
import enum
import errno
import json
import os
import re
import shutil
import socket
import sys
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from dispatchatlas.campaigns.config import Campaign, check_name
from dispatchatlas.campaigns.execution import utc_now
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
    "Kept",
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
_EXCLUSIVE_LOCK = ".lock.excl"

_UNSUPPORTED = frozenset({errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP})
"""The errno values of a lock call that the file system does not implement."""

# Windows: the exit code that a running process reports, the access right that reads it, and
# the error of OpenProcess for a process id that no process has.
_STILL_ACTIVE = 259
_QUERY_LIMITED_INFORMATION = 0x1000
_NO_SUCH_PROCESS = 87

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
            schedule digest, and identity hash, and that it states what
            the plan states for its run: the problem and its digest, the
            solver with its parameters, the protocol, the replicate, and
            the seed. The comparison with the plan needs no hash, so it
            also covers a record that states no identity hash. A seed the
            run had in an earlier plan of the directory is accepted, as
            the session records name it.

    Every other file under ``runs/`` is warned about, by kind and with the
    first names (:func:`read_strays`).

    Raises:
        SerializationError: If the plan or a record is malformed, or, with
            ``verify``, a record was modified after it was written.
    """
    folder = Path(directory)
    plan = read_plan(folder)
    stated = _Stated(plan, _earlier_seeds(folder))
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
        if verify and (issues := stated.issues(record, run)):
            raise SerializationError(f"{path}: {issues}")
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
        unreadable: Files named ``*.json`` that cannot be read as run
            records: one that is not UTF-8 text or not JSON, one that is
            no run record, and one the operating system does not let read.
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
        except (SerializationError, OSError):
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
        path: The file: the one at the path of the run's record, or one at
            the path of a failed attempt of the run.
        found: The run id the file states, or ``run_id`` when it states
            none that can be part of a file name.
        reason: Why the file does not count as a record of the run.
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
class Kept:
    """A file under ``superseded/`` that is the valid record of a planned run without one.

    Attributes:
        run_id: Id of the planned run.
        kept: The file under ``superseded/``.
        path: The path of the run's record, where the file belongs.
    """

    run_id: str
    kept: Path
    path: Path


@dataclass(frozen=True)
class Ledger:
    """What a store already holds for a plan.

    Attributes:
        completed: Ids of the runs with a valid record.
        failures: Failed attempts of the plan's runs, by run id, in attempt order.
        stale: The files at the paths of planned runs that do not count as
            their records: those at the paths of run records in plan order,
            then those at the paths of failed attempts in path order.
        kept: The valid records under ``superseded/`` of planned runs
            without a valid record at their paths, in plan order.
        unreadable: The files under ``failures/`` that cannot be read as
            records of failed attempts, in path order. They count as no
            attempt of any run.
    """

    completed: frozenset[str] = frozenset()
    failures: Mapping[str, tuple[FailureRecord, ...]] = field(default_factory=dict)
    stale: tuple[Stale, ...] = ()
    kept: tuple[Kept, ...] = ()
    unreadable: tuple[Path, ...] = ()

    def restored(self) -> Ledger:
        """Return the ledger as it stands once every kept record is back at its path.

        The runs of the kept records are complete, and the stale files at
        their paths are no longer listed. Nothing is moved.
        """
        paths = {item.path for item in self.kept}
        return Ledger(
            self.completed | {item.run_id for item in self.kept},
            self.failures,
            tuple(item for item in self.stale if item.path not in paths),
            unreadable=self.unreadable,
        )

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

        The lock is an operating-system file lock on ``.lock``, released when the
        process ends in any way. A file system that does not implement it, such as
        Lustre mounted without flock, answers the lock call with ``ENOSYS``,
        ``ENOTSUP`` or ``EOPNOTSUPP``, and the lock is then the file ``.lock.excl``,
        created exclusively (``O_CREAT | O_EXCL``) with the host name, the process
        id and the UTC time of the runner, which removes it when it leaves. Any
        other failure of the call, such as ``EWOULDBLOCK`` or ``EACCES``, means that
        another process holds the lock. The two locks are independent: runners that
        take different ones of one directory do not exclude each other.

        A runner that is killed leaves ``.lock.excl`` behind. A session that finds
        it written on this host by a process that no longer exists refuses to start
        and names the file. The library never removes it: whoever runs the campaign
        does, after checking that no runner works on it. A file written on another
        host, or one that holds no readable owner, counts as held, because this
        host cannot tell.

        Raises:
            ConfigError: If another process holds the lock, or ``.lock.excl`` is
                stale.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / _LOCK).open("a+b") as handle:
            outcome = _try_lock(handle)
            if outcome is _Lock.HELD:
                raise ConfigError(f"{self.directory} is in use by another campaign runner")
            if outcome is _Lock.UNSUPPORTED:
                with _exclusive_file(self.directory):
                    yield
            else:
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
        another plan, no longer matches its hashes, or states another
        problem, solver, protocol, replicate, or seed than the plan states
        for the run does not count: it is listed as stale and its run is
        attempted again. The file of a failed attempt at the path of a
        planned run that belongs to another plan is listed as stale too, and
        is no attempt of the run. A run without a valid record whose valid
        record lies under ``superseded/`` is listed as kept. Nothing is
        moved or written here: :meth:`reconcile` brings the kept records
        back, and :meth:`write_run` and :meth:`write_failure` move a stale
        file aside when a new record takes its place.

        The library that plans derives the seeds, so a plan can give a run
        another seed than the plan the directory holds. A record made with
        the seed its run has in that plan counts, as one made with a seed a
        session recorded as replaced does.
        """
        return self._ledger(plan, None)

    def holdings(self, plan: Plan) -> tuple[Ledger, dict[str, RunRecord]]:
        """Read the ledger of a plan together with the valid records it counts.

        Returns:
            The ledger, as :meth:`ledger` returns it, and the record of every
            run it lists as completed, by run id and in plan order. Each
            record is read and checked once.
        """
        records: dict[str, RunRecord] = {}
        return self._ledger(plan, records), records

    def _ledger(self, plan: Plan, records: dict[str, RunRecord] | None) -> Ledger:
        seeds = _earlier_seeds(self.directory)
        if seeds is not None and (self.directory / PLAN_FILE).is_file():
            try:
                for held in read_plan(self.directory).runs:
                    seeds.setdefault(held.run_id, set()).add(held.seed)
            except SerializationError:
                seeds = None
        stated = _Stated(plan, seeds)
        aside = (self.directory / SUPERSEDED / RUNS).is_dir()
        completed: set[str] = set()
        stale: list[Stale] = []
        kept: list[Kept] = []
        for run in plan.runs:
            path = _planned_path(self.directory, run)
            if path.is_file():
                found = _examined(path, run, stated)
                if not isinstance(found, Stale):
                    completed.add(run.run_id)
                    if records is not None:
                        records[run.run_id] = found
                    continue
                stale.append(found)
            if aside and (back := self._kept(path, run, stated)) is not None:
                kept.append(Kept(run.run_id, back, path))
        attempts = {_attempts_path(self.directory, run): run for run in plan.runs}
        failures: dict[str, list[FailureRecord]] = {}
        unreadable: list[Path] = []
        for path in sorted((self.directory / FAILURES).rglob("*.json")):
            try:
                failure = _parse(path, FailureRecord.from_dict)
            except SerializationError as error:
                _LOG.warning("ignoring %s: %s", path, error)
                unreadable.append(path)
                continue
            if failure.config_hash == plan.config_hash and plan.has_run(failure.run_id):
                failures.setdefault(failure.run_id, []).append(failure)
                continue
            stem, marker, _ = path.name.rpartition(".attempt-")
            planned = attempts.get(path.with_name(stem)) if marker else None
            if planned is not None:
                reason = _foreign(
                    failure.run_id, failure.config_hash, planned.run_id, plan.config_hash
                )
                stale.append(
                    Stale(
                        planned.run_id,
                        planned.problem,
                        path,
                        _stated_id(failure.run_id, planned.run_id),
                        f"{reason} ({path})",
                        failure.config_hash == plan.config_hash,
                    )
                )
        return Ledger(
            frozenset(completed),
            {
                run_id: tuple(sorted(items, key=lambda item: item.provenance.attempt))
                for run_id, items in failures.items()
            },
            tuple(stale),
            tuple(kept),
            tuple(unreadable),
        )

    def _kept(self, path: Path, run: PlannedRun, stated: _Stated) -> Path | None:
        """Return a file under ``superseded/`` that is the valid record of a run, if any.

        ``path`` is the path of the run's record. The files moved aside
        from it that state the run's id are examined, the latest first.
        """
        folder = self.directory / SUPERSEDED / path.parent.relative_to(self.directory)
        if not folder.is_dir():
            return None
        prefix = f"{path.stem}.{run.run_id}."
        for candidate in sorted(folder.iterdir(), reverse=True):
            if not (candidate.name.startswith(prefix) and candidate.suffix == ".json"):
                continue
            try:
                if candidate.is_file() and _stale(candidate, run, stated) is None:
                    return candidate
            except OSError:
                continue
        return None

    def refuse_changed_problems(self, ledger: Ledger) -> None:
        """Refuse a directory that holds a record made on another version of a problem.

        Raises:
            ConfigError: If a stale file of the ledger is the record of a
                planned run, or of a failed attempt of one, made on another
                version of its problem.
        """
        changed = [item for item in ledger.stale if item.changed_problem]
        if changed:
            problems = ", ".join(sorted({item.problem for item in changed}))
            raise ConfigError(
                f"{self.directory} holds {len(changed)} record(s) made on another version of "
                f"problem(s) {problems}, and on_changed_problem is 'refuse'; restore the "
                "problem(s), set on_changed_problem to 'rerun' to run their runs again and "
                "keep those records aside, or give the campaign another id"
            )

    def reconcile(self, ledger: Ledger) -> Ledger:
        """Bring back the kept records of a ledger, and warn about its stale files.

        Each kept record is moved from ``superseded/`` back to the path of
        its run, after the stale file at that path, if there is one, is
        moved aside. A move the operating system refuses is warned about,
        and the run is then attempted again. One warning names the stale
        files that remain: each stays where it is until a session repeats
        its run (:meth:`write_run`, :meth:`write_failure`).

        Returns:
            The ledger after the moves: the runs whose records are back
            are complete, and no record is listed as kept.
        """
        completed = set(ledger.completed)
        remaining = {item.path: item for item in ledger.stale}
        for item in ledger.kept:
            try:
                earlier = remaining.get(item.path)
                if earlier is not None:
                    self._move_aside(item.path, earlier.found)
                    del remaining[item.path]
                item.path.parent.mkdir(parents=True, exist_ok=True)
                item.kept.replace(item.path)
            except OSError as error:
                _LOG.warning(
                    "cannot bring %s back as the record of %s, which will run again: %s",
                    item.kept,
                    item.run_id,
                    error,
                )
                continue
            completed.add(item.run_id)
            _LOG.warning(
                "%s: the record kept as %s is the valid record of the run and is back at %s",
                item.run_id,
                item.kept,
                item.path,
            )
        stale = tuple(remaining.values())
        if stale:
            shown = "; ".join(f"{item.run_id}: {item.reason}" for item in stale[:_SHOWN])
            more = f"; and {len(stale) - _SHOWN} more" if len(stale) > _SHOWN else ""
            _LOG.warning(
                "%d file(s) at the paths of planned runs are not their valid records (%s%s); "
                "none is replaced: a session moves such a file under %s before a record of "
                "its repeated run takes its place",
                len(stale),
                shown,
                more,
                self.directory / SUPERSEDED,
            )
        return Ledger(frozenset(completed), ledger.failures, stale, ())

    def _move_aside(self, path: Path, found: str) -> Path:
        """Move a file of the directory under ``superseded/`` and return where it is kept.

        The file keeps its path below the campaign directory, with the run
        id ``found`` before ``.json``, and a number after that when the name
        is taken, so no file is replaced.

        Raises:
            OSError: If the operating system refuses the move.
        """
        relative = path.relative_to(self.directory)
        stem = f"{path.stem}.{found}"
        target = self.directory / SUPERSEDED / relative.with_name(f"{stem}.json")
        number = 1
        while target.exists():
            number += 1
            target = target.with_name(f"{stem}.{number}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        return target

    def _keep(
        self,
        path: Path,
        record: RunRecord | FailureRecord,
        build: Callable[[object], RunRecord | FailureRecord],
    ) -> None:
        """Move the file at ``path`` under ``superseded/`` when a run gets a new record.

        Args:
            path: The path the new record takes, or the path of the run's
                record when the new record is that of a failed attempt.
            record: The new record.
            build: Reads the kind of record that belongs at ``path``.

        Raises:
            ConfigError: If the operating system refuses the move, for
                example because another program holds the file open.
        """
        if not path.exists():
            return
        found, reason = _standing(path, record, build)
        try:
            target = self._move_aside(path, found)
        except OSError as error:
            raise ConfigError(
                f"{path} is not the record that {record.run_id} now gets, and it cannot be "
                f"moved under {self.directory / SUPERSEDED} ({error}), so the new record is "
                "not written; close any program that holds the file open and run the "
                "campaign again"
            ) from error
        _LOG.warning("%s: %s; the earlier file is kept as %s", record.run_id, reason, target)

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
        """Write the record of a completed run and return its path.

        A file that is at the path is moved under ``superseded/`` first.

        Raises:
            ConfigError: If a file at the path cannot be moved aside. The
                record is then not written.
        """
        path = run_path(
            self.directory,
            record.solver.name,
            record.problem.id,
            record.replicate,
            record.protocol.name,
        )
        self._keep(path, record, RunRecord.from_dict)
        atomic_write_json(path, record.to_dict())
        return path

    def write_failure(self, record: FailureRecord) -> Path:
        """Write the record of a failed attempt and return its path.

        A file that is at the path, such as the record of an attempt on
        another version of the problem, is moved under ``superseded/`` first.
        So is a file at the path of the run's record: a run is attempted
        only while it has no valid record, so what stands there is none,
        and the run it stood for has now been repeated.

        Raises:
            ConfigError: If such a file cannot be moved aside. The record
                is then not written.
        """
        path = failure_path(
            self.directory,
            record.solver.name,
            record.problem.id,
            record.replicate,
            record.protocol.name,
            record.provenance.attempt,
        )
        standing = run_path(
            self.directory,
            record.solver.name,
            record.problem.id,
            record.replicate,
            record.protocol.name,
        )
        self._keep(standing, record, RunRecord.from_dict)
        self._keep(path, record, FailureRecord.from_dict)
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


class _Stated:
    """What a plan states for each of its runs, to compare the records of the runs with.

    Args:
        plan: The plan.
        seeds: The seeds the runs had in earlier plans, by run id. ``None``
            when they are not known: the seed of a record is then not
            compared.
    """

    def __init__(self, plan: Plan, seeds: Mapping[str, Collection[int]] | None) -> None:
        self.config_hash = plan.config_hash
        self._digests = {problem.id: problem.digest for problem in plan.problems}
        self._solvers = {entry.name: (entry.id, dict(entry.parameters)) for entry in plan.solvers}
        self._protocols = {protocol.name: protocol.to_dict() for protocol in plan.protocols}
        self._seeds = seeds

    def issues(self, record: RunRecord, run: PlannedRun) -> str:
        """Return why a record of a run's id is not the intact record of the run, if it is not.

        The record must match its hashes
        (:meth:`~dispatchatlas.campaigns.records.RunRecord.verify`) and state
        what the plan states for the run: the problem and its digest, the
        solver with its id and parameters, the protocol, the replicate, and
        the seed, or a seed the run had in an earlier plan.

        Returns:
            The problems, or an empty string for an intact record.
        """
        found = list(record.verify())
        solver = record.solver
        pairs = (
            ("problem", record.problem.id, run.problem),
            ("problem digest", record.problem.digest, self._digests.get(run.problem)),
            ("solver", solver.name, run.solver),
            (
                "solver id or parameters",
                (solver.id, dict(solver.parameters)),
                self._solvers.get(run.solver),
            ),
            ("protocol", record.protocol.to_dict(), self._protocols.get(run.protocol)),
            ("replicate", record.replicate, run.replicate),
        )
        differing = [name for name, recorded, planned in pairs if recorded != planned]
        if (
            record.seed != run.seed
            and self._seeds is not None
            and record.seed not in self._seeds.get(run.run_id, ())
        ):
            differing.append("seed")
        if differing:
            found.append(f"the record states another {', '.join(differing)} than the plan")
        return "; ".join(found)


def _earlier_seeds(folder: Path) -> dict[str, set[int]] | None:
    """Return the seeds the runs of a campaign directory had in earlier plans, by run id.

    The session records name them: a session whose plan gives a run another
    seed than the plan before it stores the earlier seed. ``None`` when a
    session record cannot be read as far as that, since the seeds it may
    name are then not known.
    """
    seeds: dict[str, set[int]] = {}
    for path in (folder / SESSIONS).glob("*.json"):
        try:
            document = read_json(path)
        except (SerializationError, OSError):
            return None
        reseeded = document.get("plan_reseeded", []) if isinstance(document, Mapping) else None
        if not isinstance(reseeded, list):
            return None
        for item in reseeded:
            run_id = item.get("run_id") if isinstance(item, Mapping) else None
            seed = item.get("earlier_seed") if isinstance(item, Mapping) else None
            if not isinstance(run_id, str) or not isinstance(seed, int):
                return None
            seeds.setdefault(run_id, set()).add(seed)
    return seeds


def _attempts_path(directory: Path, run: PlannedRun) -> Path:
    """Return the path of a failed attempt of a run, up to ``.attempt-<n>.json``."""
    return directory / FAILURES / run.solver / run.problem / f"r{run.replicate:03d}-{run.protocol}"


def _stated_id(found: str, run_id: str) -> str:
    """Return the run id a file states when it can be part of a file name, else ``run_id``."""
    return found if _FILE_NAME.fullmatch(found) else run_id


def _stale(path: Path, run: PlannedRun, stated: _Stated) -> Stale | None:
    """Return why the file at ``path`` is not the valid record of a planned run, if it is not."""
    found = _examined(path, run, stated)
    return found if isinstance(found, Stale) else None


def _examined(path: Path, run: PlannedRun, stated: _Stated) -> RunRecord | Stale:
    """Return the valid record of a planned run at ``path``, or why the file is none."""
    try:
        record = _parse(path, RunRecord.from_dict)
    except SerializationError as error:
        return Stale(run.run_id, run.problem, path, run.run_id, str(error))
    foreign = _foreign(record.run_id, record.config_hash, run.run_id, stated.config_hash)
    issues = foreign or stated.issues(record, run)
    if not issues:
        return record
    changed = bool(foreign) and record.config_hash == stated.config_hash
    found = _stated_id(record.run_id, run.run_id)
    return Stale(run.run_id, run.problem, path, found, f"{issues} ({path})", changed)


def _standing(
    path: Path,
    record: RunRecord | FailureRecord,
    build: Callable[[object], RunRecord | FailureRecord],
) -> tuple[str, str]:
    """Describe the file that a new record of a run moves aside.

    Returns:
        The run id the file states, or that of the new record when it
        states none that can be part of a file name, and why the file is
        not the record the run now gets.
    """
    try:
        earlier = _parse(path, build)
    except (SerializationError, OSError) as error:
        return record.run_id, str(error)
    reason = _foreign(earlier.run_id, earlier.config_hash, record.run_id, record.config_hash)
    if not reason and isinstance(earlier, RunRecord):
        reason = "; ".join(earlier.verify())
    reason = reason or "the file is not the valid record of the run"
    return _stated_id(earlier.run_id, record.run_id), f"{reason} ({path})"


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


class _Lock(enum.Enum):
    """What an attempt to take the operating system's lock on the lock file came to."""

    TAKEN = enum.auto()
    HELD = enum.auto()
    UNSUPPORTED = enum.auto()


def _try_lock(handle: IO[bytes]) -> _Lock:
    """Try to take the operating system's exclusive lock on an open file without waiting.

    The call is ``msvcrt.locking`` on Windows and ``flock`` elsewhere. The errno of a
    failed call tells whether another process holds the lock or the file system cannot
    lock at all. The names below are those of the :mod:`errno` module, the same on every
    platform though their numbers differ:

    - **Another holder**, :attr:`_Lock.HELD`: ``EWOULDBLOCK`` or ``EAGAIN`` (11 on Linux)
      and ``EACCES`` (13). On Linux, ``flock`` fails with ``EWOULDBLOCK`` while another
      open file description holds the lock. On Windows, ``msvcrt.locking`` fails with
      ``EACCES`` while another handle holds the byte.
    - **Unsupported**, :attr:`_Lock.UNSUPPORTED`: ``ENOSYS``, "function not implemented"
      (38 on Linux, 40 on Windows), ``ENOTSUP`` and ``EOPNOTSUPP``, "operation not
      supported" (95 for both on Linux, 129 and 10045 on Windows). A file system that
      lacks the call answers with one of them, as Lustre mounted without flock does. No
      lock can be taken there, so the failure says nothing about other holders.
    - **Any other errno**, such as ``EINVAL`` or ``ENOLCK``, or none, is taken as a
      holder too, as every failure was: a runner never starts without a lock because of
      an error it does not understand.

    Returns:
        :attr:`_Lock.TAKEN` when the lock is held now, :attr:`_Lock.HELD` when the call
        failed for another holder or for an errno that is neither case, and
        :attr:`_Lock.UNSUPPORTED` when the file system cannot lock.
    """
    try:
        if sys.platform == "win32":
            import msvcrt  # noqa: PLC0415 - Windows only

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl  # noqa: PLC0415 - POSIX only

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        return _Lock.UNSUPPORTED if error.errno in _UNSUPPORTED else _Lock.HELD
    return _Lock.TAKEN


def _unlock(handle: IO[bytes]) -> None:
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415 - Windows only

        with contextlib.suppress(OSError):
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl  # noqa: PLC0415 - POSIX only

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _exclusive_file(directory: Path) -> Iterator[None]:
    """Hold the exclusive-create lock file of a campaign directory for the context.

    The file is created with ``O_CREAT | O_EXCL``, which fails when it exists, and holds
    the host name, the process id and the UTC time of this process. The context removes
    it when it ends, unless it no longer holds that text: another process wrote it, and
    it stays.

    Raises:
        ConfigError: If the file exists. The message says who holds it, or that the file
            is stale, and names the file.
    """
    path = directory / _EXCLUSIVE_LOCK
    owner = _owner_text()
    try:
        stream = path.open("xb")
    except FileExistsError:
        raise ConfigError(_refusal(directory, path)) from None
    try:
        with stream:
            stream.write(owner)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    try:
        yield
    finally:
        _release(path, owner)


def _owner_text() -> bytes:
    """Return the text of this process's lock file: its host name, process id and time."""
    owner = {"host": socket.gethostname(), "pid": os.getpid(), "time": utc_now()}
    return (json.dumps(owner, sort_keys=True) + "\n").encode("ascii")


def _owner(path: Path) -> tuple[str, int, str] | None:
    """Return the host name, process id and time that a lock file holds, or ``None``.

    A file that is empty, as one is between its creation and its first write, or does not
    hold all three as a string, a positive integer and a string, holds no owner.
    """
    try:
        document = json.loads(path.read_bytes())
        host, pid, since = document["host"], document["pid"], document["time"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if (
        isinstance(host, str)
        and isinstance(since, str)
        and isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
    ):
        return host, pid, since
    return None


def _refusal(directory: Path, path: Path) -> str:
    """Return why the lock file ``path`` keeps a session out of ``directory``.

    The file is stale when it was written on this host by a process that no longer
    exists. A file written on another host is never called stale, because that host's
    processes cannot be asked, and neither is one that holds no readable owner.
    """
    owner = _owner(path)
    if owner is None:
        return (
            f"{directory} is in use by another campaign runner: its lock file {path} holds no "
            "readable owner. If no runner works on the campaign, remove the file"
        )
    host, pid, since = owner
    who = f"process {pid} on {host}, since {since}"
    if host != socket.gethostname():
        return (
            f"{directory} is in use by another campaign runner ({who}); this host cannot tell "
            f"whether it still runs. If it is gone, remove the lock file {path}"
        )
    if _alive(pid):
        return f"{directory} is in use by another campaign runner ({who})"
    return (
        f"the lock file {path} is stale: it was taken by {who}, and that process no longer "
        "runs. Remove the file and run again"
    )


def _release(path: Path, owner: bytes) -> None:
    """Remove the lock file ``path`` if it still holds ``owner``, the text this process wrote."""
    try:
        if path.read_bytes() != owner:
            _LOG.warning("the lock file %s was replaced while the session ran; it stays", path)
            return
        path.unlink()
    except OSError as error:
        _LOG.warning("the lock file %s could not be removed: %s", path, error)


def _alive(pid: int) -> bool:
    """Return whether a process with this id exists on this host.

    A process counts as gone only when the system says that no process has the id, so a
    lock file is never called stale on a guess: a process of another user, which the
    caller may not signal, still exists.
    """
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415 - Windows only
        from ctypes import wintypes  # noqa: PLC0415 - Windows only

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            try:
                code = wintypes.DWORD()
                read = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                alive = not read or code.value == _STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        else:
            alive = ctypes.get_last_error() != _NO_SUCH_PROCESS
    else:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except OSError:
            alive = True
    return alive
