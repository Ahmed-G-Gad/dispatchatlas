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

Progress is logged through :mod:`dispatchatlas.core.logs`: a line when the
session starts, one at most every :data:`PROGRESS_INTERVAL` seconds while it
runs, and a summary at the end.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections import deque
from collections.abc import Callable, Sequence
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
from dispatchatlas.campaigns.environment import capture_environment
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
from dispatchatlas.core.logs import get_logger
from dispatchatlas.solvers.registry import SolverRegistry

__all__ = ["PROGRESS_INTERVAL", "RunSummary", "overridden", "run", "tasks_for"]

PROGRESS_INTERVAL = 10.0
"""Seconds between progress lines while a session runs."""

_LOG = get_logger(__name__)


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
        backend: Override of the campaign's execution backend.
        workers: Override of the campaign's worker count, or ``"auto"``.
        affinity: Override of the campaign's ``affinity``: ``"inherit"``,
            ``"pin"``, or processor indices.

    Returns:
        The summary of the session.

    Raises:
        ConfigError: If the campaign cannot be planned, its directory holds
            results of another configuration, another runner is using it, or
            runs are pending and its workers cannot be pinned as ``affinity``
            asks: the platform cannot pin a process, the workers outnumber
            their processors, or a listed processor is not one this process
            may use. A campaign refused for its pinning loses no record, also
            with ``fresh``, and a new one leaves no directory.
        KeyboardInterrupt: When interrupted. Records written so far are kept
            and the session is marked interrupted.
    """
    execution = overridden(campaign.execution, backend, workers, affinity)
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
        change = store.write_plan(plan)
        ledger = store.ledger(plan)
        pending = ledger.pending(plan, execution.max_attempts, retry_failed=retry_failed)
        if max_runs is not None:
            pending = pending[: max(0, max_runs)]
        cpus = execution.worker_cpus(count) if pending else None
        chosen = backend_for(
            execution.backend,
            count,
            cpus=cpus,
            threads=execution.threads_per_worker,
            startup_limit=execution.startup_limit,
        )
        started, clock = utc_now(), time.perf_counter()
        record = SessionRecord(
            session=store.next_session(),
            campaign=campaign.id,
            config_hash=plan.config_hash,
            started=started,
            finished=None,
            status=SessionStatus.RUNNING,
            execution=execution,
            workers=count,
            environment=capture_environment(),
            plan_digest=plan.runs_digest,
            plan_left=change.left,
            plan_entered=change.entered,
        )
        store.write_session(record)
        _warn_about_timing(prepared, execution, count, cpus)
        if cpus is not None:
            _LOG.info(
                "campaign %s: workers pinned to processor(s) %s", campaign.id, describe_cpus(cpus)
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
        books = _Books(prepared, store, execution, record.session, ledger.completed)
        for run_id, failures in ledger.failures.items():
            books.failures[run_id] = list(failures)
        queue = deque(tasks_for(prepared, pending, execution, ledger.next_attempt))
        status = SessionStatus.FAILED
        try:
            with contextlib.closing(chosen.run(queue)) as results:
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
            store.write_session(replace(record, finished=utc_now(), status=status, counts=counts))
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
    )


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
        One task per run, each with a fresh solver instance.
    """
    digests = {problem.id: problem.digest for problem in prepared.plan.problems}
    tasks: list[Task] = []
    for item in runs:
        spec = prepared.campaign.solver(item.solver)
        protocol = prepared.campaign.protocol(item.protocol)
        tasks.append(
            Task(
                run_id=item.run_id,
                attempt=attempt(item.run_id),
                problem=prepared.problems[item.problem],
                digest=digests[item.problem],
                solver=prepared.registry.create(spec.id, **spec.arguments()),
                stop=protocol.stop,
                seed=item.seed,
                hard_limit=hard_limit(protocol.stop, execution),
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
            self.store.write_run(
                RunRecord(
                    **identity,
                    outcome=result.outcome,
                    result_hash=result.result_hash,
                    schedule=result.schedule,
                    provenance=provenance,
                    notes=result.notes,
                )
            )
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
