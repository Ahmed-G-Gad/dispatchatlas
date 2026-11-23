"""Running a campaign: plan it, execute what remains, and record every attempt.

:func:`run` plans the campaign, opens its directory under the output root, and
executes every run that is neither complete nor settled by failures. Calling
it again after an interruption therefore resumes the campaign: records already
written are kept when they belong to the same configuration and still match
their result hash. A directory that holds results of another configuration is
refused unless ``fresh`` asks to discard them.

Every attempt leaves a record the moment it ends: a run record for a
completion, a failure record for a failure (see
:mod:`~dispatchatlas.campaigns.records`). Attempt numbers continue across
sessions. A failed run gets another attempt, at the end of the queue, until
its failures settle it: its latest failure is terminal, or it has failed
``1 + retries`` times. Each session writes ``sessions/<n>.json`` with the
environment it runs in, when it starts and again when it ends.

A session that resumes a campaign compares its execution settings and its
environment with those of the last session the directory recorded
(:func:`session_changes`). It logs a warning that names each difference and
stores them in its record, and a campaign whose ``on_changed_code`` is
``"refuse"`` refuses a session on other code instead. A file at the path of a
planned run that is not the valid record of that run, such as a record made
on another version of its problem, does not count, and its run is attempted
again unless ``on_changed_problem`` is ``"refuse"``. The file stays where it
is until the new record is written, and is then moved under ``superseded/``;
a record kept there that is the valid record of a planned run again, because
its problem was restored, is moved back when a session starts.

A session may be given a deadline, in wall seconds from its start, with the
argument ``deadline_seconds``: no run starts that cannot end before it, a run
without a time limit starts only while the deadline has not passed, and the runs
that do not start stay pending for the next session.

Each session also writes ``logs/session-<n>.log``, each line with the UTC
time of its record. The log holds what the engine itself logs while
:func:`run` works, from the level ``INFO``, and what the other loggers of the
library let through, in this process and in its worker processes: ``WARNING``
and higher unless the application lowered a level. The execution key
``log_level`` lowers those loggers for the session, at the cost that a
solver which logs in its loop then pays inside its time limit. What the
handlers of the application show does not change either way.

Progress is logged through :mod:`dispatchatlas.core.logs`: a line when the
session starts, one at most every :data:`PROGRESS_INTERVAL` seconds while it
runs, and a summary at the end.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dispatchatlas.benchmarks.registry import FamilyRegistry
from dispatchatlas.campaigns.backends import backend_for
from dispatchatlas.campaigns.config import (
    THREAD_VARIABLES,
    Campaign,
    Execution,
    available_cpus,
)
from dispatchatlas.campaigns.environment import LIBRARY, capture_environment, sample_load
from dispatchatlas.campaigns.execution import Completion, Result, Task, utc_now
from dispatchatlas.campaigns.planning import PlannedRun, Prepared, prepare
from dispatchatlas.campaigns.processors import affinity_cpus, describe_cpus
from dispatchatlas.campaigns.records import (
    FailureRecord,
    ProblemRef,
    Provenance,
    RunRecord,
    SessionRecord,
    SessionStatus,
    settled,
)
from dispatchatlas.campaigns.store import Store, campaign_directory
from dispatchatlas.campaigns.timeout import hard_limit
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.logs import SessionLog, get_logger
from dispatchatlas.solvers.registry import SolverRegistry

__all__ = [
    "PROGRESS_INTERVAL",
    "RunSummary",
    "overridden",
    "run",
    "session_changes",
    "tasks_for",
]

PROGRESS_INTERVAL = 10.0
"""Seconds between progress lines while a session runs."""

_LOG = get_logger(__name__)

_ENGINE = __name__.rpartition(".")[0]
"""Name of the logger above the loggers of the engine's modules."""

_LOG_LEVELS = {"inherit": None, "info": logging.INFO, "debug": logging.DEBUG}
"""The level each value of ``log_level`` lowers the library's loggers to, if any."""

_UNCOMPARED = ("on_changed_code", "on_changed_problem")
"""Execution settings that say what a session does about a difference, and are no difference."""

_CODE = ("python", "packages", "library", "plugins", "git")
"""The parts of an environment that describe code; the others describe the machine."""

_OPEN = (("packages",), ("plugins",), ("git",))
"""The parts of an environment whose entries come and go: an entry one side lacks differs."""

_NAMED = 8
"""Differences a message names before it counts the rest."""

_DIGEST_LENGTH = 32
"""Length from which a hexadecimal value is shortened in a message."""


@dataclass(frozen=True)
class RunSummary:
    """What a session of :func:`run` did, and where the campaign stands.

    Attributes:
        campaign: Campaign id.
        config_hash: Config hash of the campaign.
        directory: The campaign directory.
        session: Number of the session.
        planned: Runs in the plan.
        completed: Runs with a record, from every session.
        failed: Runs settled by failures, from every session.
        pending: Runs that still need an attempt.
        attempted: Attempts this session made.
        succeeded: Attempts this session that completed a run.
        errors: Attempts this session that failed.
        elapsed: Wall-clock seconds of the session.
        held_back: Runs the session's deadline kept from starting, which stay
            pending; 0 without a deadline.
    """

    campaign: str
    config_hash: str
    directory: Path
    session: int
    planned: int
    completed: int
    failed: int
    pending: int
    attempted: int
    succeeded: int
    errors: int
    elapsed: float
    held_back: int = 0

    @property
    def done(self) -> bool:
        """Whether no run needs another attempt."""
        return self.pending == 0


def run(
    campaign: Campaign,
    root: str | Path,
    *,
    registry: SolverRegistry | None = None,
    families: FamilyRegistry | None = None,
    fresh: bool = False,
    retry_failed: bool = False,
    max_runs: int | None = None,
    deadline_seconds: float | None = None,
    backend: str | None = None,
    workers: int | str | None = None,
    affinity: str | Sequence[int] | None = None,
) -> RunSummary:
    """Run a campaign, resuming from the results its directory already holds.

    Args:
        campaign: The campaign.
        root: Output root; the results go to ``root / campaign.id``.
        registry: Solver registry. Defaults to the built-in solvers and plugins.
        families: Benchmark-family registry. Defaults to the built-in families
            and plugins.
        fresh: Discard every result in the campaign directory first.
        retry_failed: Give every run that failures settled one more attempt.
        max_runs: Attempt at most this many pending runs in this session.
        deadline_seconds: Wall seconds from the start of this call after which no run
            starts; ``None``, the default, sets none. A run whose stop criteria set a
            time limit (its protocol's, or the cap of its solver) starts only if the
            time elapsed plus that limit is at most the deadline. A run without one,
            which stops on evaluations or iterations alone, has no known end and starts
            only while the deadline has not passed. The test is made when the run would
            start, for a retry as for a first attempt. A run that does not start has no
            attempt and no record, so it stays pending for the next session, and the
            session log says how many the deadline held back. The bound is the time
            limit alone: it leaves out a worker's start-up, a solver's overshoot, and
            the hard limit that kills a run which ignores its limit, so a caller who
            must end by a wall time takes that margin off the deadline. The deadline
            belongs to the session: no execution setting, configuration hash, or
            record holds it.
        backend: Override of the campaign's execution backend.
        workers: Override of the campaign's worker count, or ``"auto"``.
        affinity: Override of the campaign's ``affinity``: ``"inherit"``,
            ``"pin"``, or processor indices.

    Returns:
        The summary of the session.

    Raises:
        ConfigError: If ``deadline_seconds`` is not a positive finite number, the
            campaign cannot be planned, its directory holds
            results of another configuration, another runner is using it, or
            runs are pending and its workers cannot be pinned as ``affinity``
            asks: the platform cannot pin a process, the workers outnumber
            their processors, or a listed processor is not one this process
            may use. A campaign refused for its pinning loses no record, also
            with ``fresh``, and a new one leaves no directory. Also if the
            campaign's ``on_changed_code`` is ``"refuse"`` and the code
            differs from that of the last recorded session, or its
            ``on_changed_problem`` is ``"refuse"`` and the directory holds a
            record made on another version of a planned run's problem; such
            a refusal moves no file and writes no session record. Also if a
            file that is not the valid record of a run cannot be moved under
            ``superseded/`` when the new record of the run is to be written,
            for example because another program holds it open: the session
            then ends as failed, and the run is attempted again by the next.
        KeyboardInterrupt: When interrupted. Records written so far are kept
            and the session is marked interrupted.
    """
    begun = time.perf_counter()
    deadline = None if deadline_seconds is None else _Deadline(begun, deadline_seconds)
    execution = overridden(campaign.execution, backend, workers, affinity)
    # The engine's loggers record from INFO for the session log. The other loggers
    # of the library keep their levels unless the campaign asks for a lower one.
    level = _LOG_LEVELS[execution.log_level]
    lowered = (_ENGINE,) if level is None else (LIBRARY, _ENGINE)
    with SessionLog(min(level or logging.INFO, logging.INFO), lowered=lowered) as log:
        prepared = prepare(campaign, registry=registry, families=families)
        plan = prepared.plan
        count = execution.worker_count()
        store = Store(campaign_directory(root, campaign.id))
        if fresh or not store.directory.exists():
            # Every run is pending, so a refusal comes before records are discarded
            # or a directory is created.
            execution.worker_cpus(count)
        with store.locked():
            store.open(campaign, fresh=fresh)
            number = store.next_session()
            environment = capture_environment(exclude=[store.directory])
            changes = _compare(store, campaign.id, number, execution, count, environment)
            change = store.plan_change(plan)
            ledger = store.ledger(plan)
            if execution.on_changed_problem == "refuse":
                store.refuse_changed_problems(ledger)
            ledger = store.reconcile(ledger)
            pending = ledger.pending(plan, execution.max_attempts, retry_failed=retry_failed)
            if max_runs is not None:
                pending = pending[: max(0, max_runs)]
            cpus = execution.worker_cpus(count) if pending else None

            def from_worker(worker: str, created: float, level: int, name: str, text: str) -> None:
                log.write(created, level, name, f"[{worker}] {text}")

            chosen = backend_for(
                execution.backend,
                count,
                cpus=cpus,
                threads=execution.threads_per_worker,
                startup_limit=execution.startup_limit,
                source=environment["library"]["source_digest"],
                log=from_worker,
                log_level=level,
            )
            started, clock = utc_now(), time.perf_counter()
            load: dict[str, Any] = {"start": sample_load()} if pending else {}
            record = SessionRecord(
                session=number,
                campaign=campaign.id,
                config_hash=plan.config_hash,
                started=started,
                finished=None,
                status=SessionStatus.RUNNING,
                execution=execution,
                workers=count,
                environment=environment,
                plan_digest=plan.runs_digest,
                plan_left=change.left,
                plan_entered=change.entered,
                plan_reseeded=change.reseeded,
                plan_solvers=change.solvers,
                changes=changes,
                load=load,
            )
            # The session record, which states how the plan changed, is written before
            # the plan it was compared with is replaced, so a session that ends in
            # between leaves the comparison to the next one.
            store.write_session(record)
            log.open(store.log_path(number))
            store.write_plan(plan)
            _warn_about_timing(prepared, execution, count, cpus)
            if cpus is not None:
                _LOG.info(
                    "campaign %s: workers pinned to processor(s) %s",
                    campaign.id,
                    describe_cpus(cpus),
                )
            _LOG.info(
                "campaign %s, session %d: %d of %d runs to attempt, %s backend, %d worker(s)",
                campaign.id,
                record.session,
                len(pending),
                len(plan.runs),
                chosen.name,
                count,
            )
            if deadline is not None:
                _LOG.info(
                    "campaign %s, session %d: deadline of %g s from the start of the session; "
                    "a run with a time limit starts only if the limit ends before it, a run "
                    "without one only while it has not passed",
                    campaign.id,
                    record.session,
                    deadline.seconds,
                )
            books = _Books(prepared, store, execution, record.session, ledger.completed)
            for run_id, failures in ledger.failures.items():
                books.failures[run_id] = list(failures)
            queue = deque(tasks_for(prepared, pending, execution, ledger.next_attempt))
            status = SessionStatus.FAILED
            try:
                executing = (
                    chosen.run(queue) if deadline is None else chosen.run(queue, deadline.admits)
                )
                with contextlib.closing(executing) as results:
                    for result in results:
                        retry = books.record(result)
                        if retry is not None:
                            queue.append(retry)
                        books.progress(len(queue))
                status = SessionStatus.FINISHED
            except KeyboardInterrupt:
                status = SessionStatus.INTERRUPTED
                raise
            finally:
                counts = books.counts()
                if pending and status is SessionStatus.FINISHED:
                    load = {**load, "end": sample_load()}
                store.write_session(
                    replace(record, finished=utc_now(), status=status, counts=counts, load=load)
                )
                if deadline is not None:
                    _LOG.info(
                        "campaign %s, session %d: the deadline held back %d run(s), which "
                        "stay pending",
                        campaign.id,
                        record.session,
                        len(deadline.held),
                    )
                _LOG.info(
                    "campaign %s, session %d %s: %d completed, %d failed, %d pending",
                    campaign.id,
                    record.session,
                    status.value,
                    counts["completed"],
                    counts["failed"],
                    counts["pending"],
                )
    return RunSummary(
        campaign=campaign.id,
        config_hash=plan.config_hash,
        directory=store.directory,
        session=record.session,
        planned=counts["planned"],
        completed=counts["completed"],
        failed=counts["failed"],
        pending=counts["pending"],
        attempted=counts["attempted"],
        succeeded=counts["succeeded"],
        errors=counts["errors"],
        elapsed=time.perf_counter() - clock,
        held_back=0 if deadline is None else len(deadline.held),
    )


def session_changes(
    previous: SessionRecord,
    execution: Execution,
    workers: int,
    environment: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return how a session differs from an earlier session of its campaign directory.

    The execution settings are compared one by one, except those that only
    say what a session does about a difference, and the resolved worker
    count is compared when both sessions state the same ``workers``. The
    environments are compared value by value. A value the earlier record
    does not state is not compared; a package, a plugin, or a git checkout
    that only one session has is a difference. The load of the machine is no
    part of the environment and is not compared.

    Args:
        previous: The record of the earlier session.
        execution: The execution settings of the session.
        workers: Its number of workers, with ``"auto"`` resolved.
        environment: Its environment, as
            :func:`~dispatchatlas.campaigns.environment.capture_environment`
            records it.

    Returns:
        One document per difference, with ``field``, a dotted path under
        ``execution`` or ``environment`` or the name ``workers``; ``kind``,
        which is ``"execution"``, ``"code"`` for the Python version, the
        packages, the library, the plugins, and the git states, or
        ``"machine"`` for the rest of the environment; and the values
        ``earlier`` and ``now``.
    """
    changes: list[dict[str, Any]] = []
    before, now = previous.execution.to_dict(), execution.to_dict()
    for key, value in now.items():
        if key not in _UNCOMPARED and before[key] != value:
            changes.append(
                {
                    "field": f"execution.{key}",
                    "kind": "execution",
                    "earlier": before[key],
                    "now": value,
                }
            )
    if before["workers"] == now["workers"] and previous.workers != workers:
        changes.append(
            {"field": "workers", "kind": "execution", "earlier": previous.workers, "now": workers}
        )
    for path, earlier, current in _differences(previous.environment, environment, ()):
        kind = "code" if path[0] in _CODE else "machine"
        field = ".".join(("environment", *path))
        changes.append({"field": field, "kind": kind, "earlier": earlier, "now": current})
    return tuple(changes)


def tasks_for(
    prepared: Prepared,
    runs: Sequence[PlannedRun],
    execution: Execution,
    attempt: Callable[[str], int],
) -> list[Task]:
    """Build the tasks that execute runs of a prepared plan.

    Args:
        prepared: The prepared plan.
        runs: Runs of its plan.
        execution: Execution settings, for hard limits and schedule recording.
        attempt: The attempt number of a run, given its id.

    Returns:
        One task per run, each with a fresh solver instance. A run under a
        time cap of its solver (:attr:`~dispatchatlas.campaigns.config.SolverSpec.time_caps`)
        has the cap as its time limit, and its hard limit follows from it.
    """
    digests = {problem.id: problem.digest for problem in prepared.plan.problems}
    tasks: list[Task] = []
    for item in runs:
        spec = prepared.campaign.solver(item.solver)
        protocol = prepared.campaign.protocol(item.protocol)
        stop, _ = spec.capped(protocol, item.problem)
        tasks.append(
            Task(
                run_id=item.run_id,
                attempt=attempt(item.run_id),
                problem=prepared.problems[item.problem],
                digest=digests[item.problem],
                solver=prepared.registry.create(spec.id, **spec.arguments()),
                stop=stop,
                seed=item.seed,
                hard_limit=hard_limit(stop, execution),
                record_schedule=execution.record_schedules,
            )
        )
    return tasks


def overridden(
    base: Execution,
    backend: str | None,
    workers: int | str | None,
    affinity: str | Sequence[int] | None = None,
) -> Execution:
    """Return execution settings with a caller's overrides applied.

    Choosing the sequential backend, which has no worker processes, sets what
    only worker processes have to its default: ``affinity`` to ``"inherit"``,
    ``threads_per_worker`` to 1, and, without a worker count, ``workers``
    to 1.

    Args:
        base: The campaign's settings.
        backend: Another backend, or ``None`` to keep the campaign's.
        workers: Another worker count or ``"auto"``, or ``None``.
        affinity: Another ``affinity``: ``"inherit"``, ``"pin"``, or
            processor indices; or ``None``.

    Raises:
        ConfigError: If the resulting settings are invalid.
    """
    changes: dict[str, Any] = {}
    if backend is not None:
        changes["backend"] = backend
        if backend == "sequential":
            changes.update(affinity="inherit", threads_per_worker=1)
            if workers is None:
                changes["workers"] = 1
    if workers is not None:
        changes["workers"] = workers
    if affinity is not None:
        changes["affinity"] = affinity if isinstance(affinity, str) else tuple(affinity)
    return replace(base, **changes) if changes else base


# ------------------------------------------------------------------- helpers


class _Deadline:
    """The deadline of a session: a run starts only if it can end before it.

    The backend asks :meth:`admits` about each task at the instant it would start. A task
    whose stop criteria set a time limit, the protocol's or the cap of its solver, may
    start when the time elapsed plus that limit is at most the deadline. A task without a
    time limit has no known end and may start while the deadline has not passed. A time
    that no longer fits never fits later, so a refused task is dropped from the queue
    for good, and its run id is kept in :attr:`held`.

    Args:
        begun: The reading of ``clock`` at the start of the session.
        seconds: Seconds from ``begun`` to the deadline, positive and finite.
        clock: The clock, in seconds; the monotonic performance counter by default.

    Raises:
        ConfigError: If ``seconds`` is not a positive finite number.
    """

    def __init__(
        self, begun: float, seconds: float, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not math.isfinite(seconds)
            or seconds <= 0.0
        ):
            raise ConfigError(f"deadline_seconds is a positive finite number, not {seconds!r}")
        self.seconds = float(seconds)
        self.held: set[str] = set()
        self._at = begun + self.seconds
        self._clock = clock

    def admits(self, task: Task) -> bool:
        """Return whether ``task`` may start now; a task that may not is counted as held back."""
        now = self._clock()
        limit = task.stop.time_limit
        fits = now < self._at if limit is None else now + limit <= self._at
        if not fits:
            self.held.add(task.run_id)
        return fits


class _Books:
    """The bookkeeping of one session: records, retries, counts, and progress."""

    def __init__(
        self,
        prepared: Prepared,
        store: Store,
        execution: Execution,
        session: int,
        completed: frozenset[str],
    ) -> None:
        self.prepared = prepared
        self.store = store
        self.execution = execution
        self.session = session
        self.completed = set(completed)
        self.failures: dict[str, list[FailureRecord]] = {}
        self.attempted = 0
        self.succeeded = 0
        self.errors = 0
        plan = prepared.plan
        self._problems = {
            item.id: ProblemRef(item.id, item.family, item.digest) for item in plan.problems
        }
        self._solvers = {item.name: item for item in plan.solvers}
        self._protocols = {item.name: item for item in plan.protocols}
        self._started = self._reported = time.perf_counter()

    def record(self, result: Result) -> Task | None:
        """Write the record of an attempt and return the retry it earns, if any."""
        plan = self.prepared.plan
        planned = plan.run(result.run_id)
        measured = result.measurement
        campaign = self.prepared.campaign
        _, cap = campaign.solver(planned.solver).capped(
            campaign.protocol(planned.protocol), planned.problem
        )
        provenance = Provenance(
            attempt=result.attempt,
            session=self.session,
            started=measured.started,
            finished=measured.finished,
            wall_seconds=measured.wall_seconds,
            cpu_seconds=measured.cpu_seconds,
            peak_rss_bytes=measured.peak_rss_bytes,
            worker=measured.worker,
            hard_limit=measured.hard_limit,
            cpu=measured.cpu,
            concurrent=measured.concurrent,
            waited_seconds=measured.waited_seconds,
            shared_clock=measured.shared_clock,
            time_cap=cap,
        )
        identity: dict[str, Any] = {
            "run_id": planned.run_id,
            "campaign": plan.campaign,
            "config_hash": plan.config_hash,
            "problem": self._problems[planned.problem],
            "solver": self._solvers[planned.solver],
            "protocol": self._protocols[planned.protocol],
            "replicate": planned.replicate,
            "seed": planned.seed,
        }
        self.attempted += 1
        where = (
            f"run {planned.run_id} ({planned.solver} on {planned.problem}, "
            f"protocol {planned.protocol}, replicate {planned.replicate})"
        )
        if isinstance(result, Completion):
            completed = RunRecord(
                **identity,
                outcome=result.outcome,
                result_hash=result.result_hash,
                schedule=result.schedule,
                provenance=provenance,
                notes=result.notes,
            )
            self.store.write_run(replace(completed, identity_hash=completed.identity_digest()))
            self.completed.add(planned.run_id)
            self.succeeded += 1
            for note in result.notes:
                _LOG.warning("%s: %s", where, note)
            _LOG.debug("%s: %s = %s", where, result.outcome.objective, result.outcome.value)
            return None
        failure = FailureRecord(
            **identity,
            category=result.category,
            error=result.error,
            provenance=provenance,
            solve_started=result.solve_started,
            last_incumbent=result.last_incumbent,
            limit_incumbent=result.limit_incumbent,
        )
        self.store.write_failure(failure)
        self.errors += 1
        history = self.failures.setdefault(planned.run_id, [])
        history.append(failure)
        again = not settled(history, self.execution.max_attempts)
        _LOG.warning(
            "%s failed on attempt %d (%s%s): %s",
            where,
            result.attempt,
            result.category.value,
            ", will try again" if again else "",
            result.error.message,
        )
        if not again:
            return None
        return tasks_for(self.prepared, [planned], self.execution, lambda _: result.attempt + 1)[0]

    def counts(self) -> dict[str, int]:
        """Return the run counts of the campaign and the attempt counts of the session."""
        runs = self.prepared.plan.runs
        completed = sum(1 for item in runs if item.run_id in self.completed)
        failed = sum(
            1
            for item in runs
            if item.run_id not in self.completed
            and settled(self.failures.get(item.run_id, ()), self.execution.max_attempts)
        )
        return {
            "planned": len(runs),
            "completed": completed,
            "failed": failed,
            "pending": len(runs) - completed - failed,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "errors": self.errors,
        }

    def progress(self, queued: int) -> None:
        """Log progress when :data:`PROGRESS_INTERVAL` has passed since the last line."""
        now = time.perf_counter()
        if now - self._reported < PROGRESS_INTERVAL:
            return
        self._reported = now
        rate = self.attempted / max(now - self._started, 1e-9)
        _LOG.info(
            "campaign %s: %d attempts made (%d failed), %d queued, about %s left",
            self.prepared.plan.campaign,
            self.attempted,
            self.errors,
            queued,
            _duration(queued / rate) if rate > 0 else "unknown",
        )


def _compare(
    store: Store,
    campaign: str,
    session: int,
    execution: Execution,
    workers: int,
    environment: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Compare a session with the last recorded one and warn about every difference.

    Raises:
        ConfigError: If the code differs and ``on_changed_code`` is ``"refuse"``.
    """
    previous = store.last_session()
    if previous is None:
        return ()
    changes = session_changes(previous, execution, workers, environment)
    code = [item for item in changes if item["kind"] == "code"]
    if code and execution.on_changed_code == "refuse":
        raise ConfigError(
            f"{store.directory}: the code differs from that of session {previous.session} "
            f"({_named(code)}), and on_changed_code is 'refuse'; run on the code of that "
            "session, set on_changed_code to 'warn' to resume on this code, or give the "
            "campaign another id"
        )
    if changes:
        _LOG.warning(
            "campaign %s: session %d differs from session %d: %s; every record keeps the "
            "session that made it, whose record states what it ran on",
            campaign,
            session,
            previous.session,
            _named(changes),
        )
    return changes


def _differences(
    earlier: object, now: object, path: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], object, object]]:
    """Yield the paths at which two environments differ, with both values.

    A key the earlier mapping lacks is skipped, except in the parts of
    :data:`_OPEN`, where an entry that only one side has differs.
    """
    if isinstance(earlier, Mapping) and isinstance(now, Mapping):
        for key in sorted({*earlier, *now}, key=str):
            if key in earlier or path in _OPEN:
                yield from _differences(earlier.get(key), now.get(key), (*path, str(key)))
    elif earlier != now:
        yield path, earlier, now


def _named(changes: Sequence[Mapping[str, Any]]) -> str:
    """Name the first differences of a list, each with both values."""
    shown = "; ".join(
        f"{item['field']} {_brief(item['earlier'])} -> {_brief(item['now'])}"
        for item in changes[:_NAMED]
    )
    more = f"; and {len(changes) - _NAMED} more" if len(changes) > _NAMED else ""
    return f"{shown}{more}"


def _brief(value: object) -> str:
    """Return a value as a short text: a long hexadecimal value by its first 12 digits."""
    if isinstance(value, str):
        text = value
        if len(text) >= _DIGEST_LENGTH and all(digit in "0123456789abcdef" for digit in text):
            return text[:12]
    else:
        text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= 2 * _DIGEST_LENGTH else f"{text[: 2 * _DIGEST_LENGTH]}..."


def _warn_about_timing(
    prepared: Prepared, execution: Execution, workers: int, cpus: Sequence[int] | None = None
) -> None:
    """Warn when wall-clock budgets cannot be enforced or do not buy equal processor time.

    Args:
        prepared: The prepared plan.
        execution: The execution settings in force.
        workers: Number of workers.
        cpus: The processor each worker is pinned to, or ``None`` when the
            workers share the processors this process may use.
    """
    timed = [protocol.name for protocol in prepared.campaign.protocols if protocol.stop.time_limit]
    if not timed:
        return
    names = ", ".join(timed)
    if execution.backend == "sequential":
        _LOG.warning(
            "protocols %s set time limits, which the sequential backend cannot enforce; "
            "solvers must honour them",
            names,
        )
        unset = [name for name in THREAD_VARIABLES if not os.environ.get(name)]
        if unset:
            _LOG.warning(
                "protocols %s set time limits and the sequential backend runs in this "
                "process, where %s set(s) no thread limit, so a numerical library may use "
                "every processor; set the variable(s) before Python starts, or use the "
                "process backend",
                names,
                ", ".join(unset),
            )
    threads = _threads_per_worker(prepared, execution)
    if cpus is not None:
        if threads > 1:
            _LOG.warning(
                "%d pinned worker(s) run up to %d thread(s) each on one processor apiece, so "
                "the threads of a run share it within the wall-clock budgets of protocols %s",
                workers,
                threads,
                names,
            )
        return
    processors = available_cpus()
    if affinity_cpus() is None:
        _LOG.warning(
            "the processors this process may use cannot be read, so all %d of the machine "
            "are counted; if the process is confined to fewer, the wall-clock budgets of "
            "protocols %s buy less processor time than the worker count suggests",
            processors,
            names,
        )
    if workers * threads > processors:
        _LOG.warning(
            "%d worker(s) with up to %d thread(s) each share %d processor(s), so the "
            "wall-clock budgets of protocols %s differ between runs",
            workers,
            threads,
            processors,
            names,
        )


def _threads_per_worker(prepared: Prepared, execution: Execution) -> int:
    """Return the most threads one worker may run: its thread limit or a solver's ``threads``."""
    declared = [entry.threads or 1 for entry in prepared.plan.solvers]
    return max([execution.threads_per_worker, *declared])


def _duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{rest:02d}s"
    return f"{rest}s"
