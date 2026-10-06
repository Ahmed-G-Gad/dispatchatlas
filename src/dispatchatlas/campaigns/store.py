"""The result store: where a campaign's files live, and how to read them back.

Layout under an output root::

    <root>/<campaign id>/
        campaign.json      the campaign document, its base directory, its config hash
        plan.json          the plan, with parities and the fairness assessment
        sessions/<n>.json  one session record per invocation of the runner, with the
                           digest of its plan and the runs that left or entered it
        runs/<solver>/<problem>/r<replicate:03d>-<protocol>.json
        failures/<solver>/<problem>/r<replicate:03d>-<protocol>.attempt-<n>.json
        .lock              locked while a runner works on the campaign

``<solver>`` is the solver's name, its label or else its id. Every document
is written atomically, with sorted keys, two-space indentation, ASCII text,
and a final newline, and a value that is not finite is refused.

The readers :func:`read_campaign`, :func:`read_plan`, :func:`read_runs`,
:func:`read_failures`, and :func:`read_sessions` return typed records. They
only return records of the plan in ``plan.json``: a file left by an earlier
plan, whose run id or config hash differs, is skipped with a warning. A record
whose content no longer matches its result hash raises an error.

:class:`Store` is the writing side, used by the runner: it locks the
directory, refuses to mix configurations, and settles which runs remain.
"""

from __future__ import annotations

import contextlib
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
    "PLAN_FILE",
    "RUNS",
    "SESSIONS",
    "Ledger",
    "Store",
    "campaign_directory",
    "failure_path",
    "read_campaign",
    "read_failures",
    "read_plan",
    "read_runs",
    "read_sessions",
    "recorded_config_hash",
    "run_path",
]

CAMPAIGN_SCHEMA = "dispatchatlas.campaign/1"
CAMPAIGN_FILE = "campaign.json"
PLAN_FILE = "plan.json"
SESSIONS = "sessions"
RUNS = "runs"
FAILURES = "failures"
_LOCK = ".lock"

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
        verify: Check that every record still matches its result hash and
            schedule digest.

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
    return tuple(records)


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
class Ledger:
    """What a store already holds for a plan.

    Attributes:
        completed: Ids of the runs with a valid record.
        failures: Failed attempts of the plan's runs, by run id, in attempt order.
    """

    completed: frozenset[str] = frozenset()
    failures: Mapping[str, tuple[FailureRecord, ...]] = field(default_factory=dict)

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

    def write_plan(self, plan: Plan) -> PlanChange:
        """Write ``plan.json`` and report how it differs from the plan it replaces.

        A plan can change while the campaign file, and so the config hash,
        stays the same: the plan also follows the metadata the registered
        solvers declare, the content of the problems, and the library that
        plans. When the directory holds a plan of the same config hash, a
        warning names the problems whose content changed, and one names the
        runs that left the plan and the runs that entered it. The records of
        runs that left stay in the directory and are no longer read.

        Returns:
            The runs that left and entered. Empty when the directory holds
            no plan, one that cannot be read, or one of another config hash.
        """
        path = self.directory / PLAN_FILE
        change = PlanChange()
        if path.is_file():
            try:
                previous = read_plan(self.directory)
            except SerializationError:
                previous = None
            if previous is not None and previous.config_hash == plan.config_hash:
                digests = {problem.id: problem.digest for problem in previous.problems}
                for problem in plan.problems:
                    if digests.get(problem.id, problem.digest) != problem.digest:
                        _LOG.warning(
                            "problem %s changed since the last session; its runs will run again",
                            problem.id,
                        )
                left = [run for run in previous.runs if not plan.has_run(run.run_id)]
                entered = [run for run in plan.runs if not previous.has_run(run.run_id)]
                change = PlanChange(
                    tuple(run.to_dict() for run in left),
                    tuple(run.to_dict() for run in entered),
                )
                if change:
                    _LOG.warning(
                        "the plan of campaign %s changed under config hash %s: %d run(s) left "
                        "it and %d entered it%s%s; the campaign file is the same, so the "
                        "solvers' metadata, the problems, or the library changed; records of "
                        "runs that left stay in the directory and are no longer read",
                        plan.campaign,
                        plan.config_hash[:12],
                        len(left),
                        len(entered),
                        _named("; left", left),
                        _named("; entered", entered),
                    )
        atomic_write_json(path, plan.to_dict())
        return change

    def ledger(self, plan: Plan) -> Ledger:
        """Read which runs of a plan are complete and which attempts failed.

        A record that cannot be read, belongs to another plan, or no longer
        matches its result hash does not count: its run is attempted again,
        and the new record replaces it.
        """
        completed: set[str] = set()
        for run in plan.runs:
            path = _planned_path(self.directory, run)
            if not path.is_file():
                continue
            try:
                record = _parse(path, RunRecord.from_dict)
            except SerializationError as error:
                _LOG.warning("will run %s again: %s", run.run_id, error)
                continue
            stale = _foreign(record.run_id, record.config_hash, run.run_id, plan.config_hash)
            issues = stale or "; ".join(record.verify())
            if issues:
                _LOG.warning("will run %s again: %s (%s)", run.run_id, issues, path)
                continue
            completed.add(run.run_id)
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
        )

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
        """Remove every result, plan, session, and campaign file of the directory."""
        runs = len(list((self.directory / RUNS).rglob("*.json")))
        failures = len(list((self.directory / FAILURES).rglob("*.json")))
        if runs or failures:
            _LOG.warning(
                "discarding %d run records and %d failure records in %s",
                runs,
                failures,
                self.directory,
            )
        for name in (RUNS, FAILURES, SESSIONS):
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
