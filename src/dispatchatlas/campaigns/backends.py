"""Execution backends: where the runs of a campaign execute.

Both backends take :class:`~dispatchatlas.campaigns.execution.Task` items from
a queue that the runner may extend while results arrive (a retry goes to its
end), and yield each :class:`~dispatchatlas.campaigns.execution.Completion`
or :class:`~dispatchatlas.campaigns.execution.Failure` as soon as it exists.
A caller may pass ``admit``, a function that is asked about each task at the instant
the backend would start it and answers whether it may start. A task it refuses is
dropped from the queue, with no attempt and no record, so its run stays pending.

- :class:`SequentialBackend` runs one attempt after another in the calling
  process. Solvers must honour their own time limits, and a crash in native
  code ends the campaign. Nothing has to be picklable, which suits debugging
  and solvers defined interactively.
- :class:`ProcessBackend` runs attempts in worker processes started with the
  ``spawn`` method, so every platform behaves alike, and supervised by the
  calling process. Each worker runs one attempt at a time and keeps the
  problem of its last attempt, so a problem is sent to a worker only when the
  worker's previous attempt was on another one; runs are planned problem by
  problem, which makes that rare. The supervisor enforces two limits (see
  :mod:`~dispatchatlas.campaigns.timeout`). From the hand-over of a run its
  worker has the start-up limit to report that the solve begins; from the
  instant that report states, the run has its hard limit. The supervisor
  kills a worker that passes either, with every process the worker started,
  records a recoverable start-up failure or a terminal timeout, and hands
  the next run to a fresh worker. A hand-over never waits for a worker: a
  worker that is starting gets its run when it reports that it is ready, so
  the supervisor keeps reading reports and enforcing the limits of the other
  runs meanwhile. While a run with a hard limit executes, its worker
  forwards the incumbents its search reports, at most one every
  :data:`PROGRESS_INTERVAL` seconds, and a timeout failure holds the last
  one and the last one found within the run's time limit. A worker that
  dies, from a crash in native code or the
  out-of-memory killer, costs only the attempt it was running, which is
  recorded as a recoverable failure. Every worker starts with the thread
  pools of numerical libraries limited: the variables of
  :data:`~dispatchatlas.campaigns.config.THREAD_VARIABLES` are in the
  environment the worker process is created with, so no import in the
  worker comes before them. A worker can also be pinned to one processor:
  the supervisor creates it while confined to that processor itself, so the
  worker and every thread it starts run there, and the worker reports its
  processors when it is ready, which the supervisor checks. A backend that is
  given the digest of the library source its session started on has every
  worker report the digest of the source it starts on: a worker on other
  source is warned about, and every run it completes carries a note. A
  backend that is given a sink for log records has every worker send the
  records of :data:`WORKER_LOG_LEVEL` or higher that the library's loggers
  let through in it, which a fresh worker does from ``WARNING``; given a
  log level as well, the backend lowers the library's loggers of every
  worker to it, and what the handlers of a worker show stays as it was.
  The thread that logs never waits for the supervisor: a record goes to a
  bounded queue that another thread of the worker sends in batches, a
  record the queue has no room for is dropped and counted, and the
  supervisor adds the count to the log. The supervisor spends at most
  :data:`LOG_READ_SECONDS` on the log records of one worker before it
  turns to the limits and the other workers, and log records do not
  postpone a kill.

Every result states how many runs were executing when its attempt was handed
over (``concurrent``), the processor its worker is pinned to (``cpu``), and
how long the attempt waited for its solve to begin (``waited_seconds``), from
its hand-over, before a fresh worker exists, to the instant its worker states.
"""

from __future__ import annotations

import contextlib
import logging
import math
import multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from multiprocessing.connection import wait
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Protocol

from dispatchatlas.campaigns.config import STARTUP_SECONDS, THREAD_VARIABLES
from dispatchatlas.campaigns.environment import LIBRARY, source_digest
from dispatchatlas.campaigns.execution import (
    Completion,
    Failure,
    Measurement,
    Result,
    Task,
    attempt,
    engine_failure,
    exception_error,
    utc_now,
    warm_up_task,
)
from dispatchatlas.campaigns.processors import affinity_cpus, confined, describe_cpus
from dispatchatlas.campaigns.records import Category, ErrorKind, Incumbent
from dispatchatlas.campaigns.timeout import contain, detach, end_with_parent, kill, release
from dispatchatlas.core.errors import ConfigError, SerializationError
from dispatchatlas.core.logs import Lowered, get_logger
from dispatchatlas.core.serialization import problem_from_dict, problem_to_dict
from dispatchatlas.solvers.search import Improvement, listening

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.context import SpawnContext
    from multiprocessing.process import BaseProcess
    from types import TracebackType

    from dispatchatlas.core.model import Problem

__all__ = [
    "LOG_READ_SECONDS",
    "PROGRESS_INTERVAL",
    "STARTUP_ATTEMPTS",
    "WORKER_LOG_LEVEL",
    "Backend",
    "LogSink",
    "ProcessBackend",
    "SequentialBackend",
    "backend_for",
]

STARTUP_ATTEMPTS = 3
"""Consecutive worker processes that may exit before they start until the backend gives up."""

PROGRESS_INTERVAL = 0.25
"""Fewest seconds between two incumbents a worker forwards to the supervisor."""

WORKER_LOG_LEVEL = logging.INFO
"""Lowest level of the log records a worker sends, unless the backend is given a lower one."""

LOG_READ_SECONDS = 0.05
"""Longest the supervisor handles the log records of one worker in one reading of its pipe."""

_LOG_QUEUE = 10_000
"""Log records a worker holds for sending; a record beyond them is dropped and counted."""

_LOG_BATCH = 200
"""Most log records a worker sends in one message."""

type LogSink = Callable[[str, float, int, str, str], None]
"""Receives a record a worker logged: the worker's name, the time of the record in
seconds since the epoch, its level, the name of its logger, and its message."""

_LOG = get_logger(__name__)

_SENDING = threading.Lock()
"""Held by the thread of a worker process that writes to the worker's pipe."""


class Backend(Protocol):
    """Where attempts execute."""

    name: str
    """``"sequential"`` or ``"process"``."""
    workers: int
    """Attempts that run at once."""

    def run(
        self, queue: deque[Task], admit: Callable[[Task], bool] | None = None
    ) -> Generator[Result]:
        """Execute tasks from ``queue`` until it is empty, yielding each result.

        ``admit`` is asked about each task as it is about to start, and a task it
        refuses is dropped from the queue without an attempt.
        """
        ...


class SequentialBackend:
    """Runs attempts one after another in the calling process."""

    name = "sequential"
    workers = 1

    def run(
        self, queue: deque[Task], admit: Callable[[Task], bool] | None = None
    ) -> Generator[Result]:
        """Execute tasks from ``queue`` until it is empty, yielding each result.

        A task that ``admit`` refuses when its turn comes is dropped without an attempt.
        """
        while queue:
            task = queue.popleft()
            if admit is None or admit(task):
                yield _stamped(attempt(task, "main"), None, 1)


class ProcessBackend:
    """Runs attempts in supervised worker processes.

    Args:
        workers: Number of worker processes.
        cpus: The processor each worker is pinned to, in worker order, or
            ``None`` to leave the workers on the processors the calling
            process may use.
        threads: Threads the numerical libraries of a worker may use.
        startup_limit: Seconds a worker has, from the hand-over of a run, to
            report that the solve begins.
        source: The digest of the library source the session started on
            (:func:`~dispatchatlas.campaigns.environment.source_digest`),
            which every worker compares the source it starts on with.
            ``None`` compares nothing.
        log: Receives the records the workers log. ``None`` leaves them in
            the workers.
        log_level: The level the library's loggers of every worker are
            lowered to, so that ``log`` receives their records from that
            level. ``None`` leaves the loggers as they are, and ``log``
            receives what they let through of :data:`WORKER_LOG_LEVEL` or
            higher. Without ``log`` nothing is lowered.

    Raises:
        ConfigError: If there is no worker, a worker has no thread, the
            start-up limit is not a positive, finite number, or there are
            fewer processors to pin to than workers.
    """

    name = "process"

    def __init__(
        self,
        workers: int,
        *,
        cpus: Sequence[int] | None = None,
        threads: int = 1,
        startup_limit: float = STARTUP_SECONDS,
        source: str | None = None,
        log: LogSink | None = None,
        log_level: int | None = None,
    ) -> None:
        if workers < 1:
            raise ConfigError("a process backend needs at least one worker")
        if threads < 1:
            raise ConfigError("a worker needs at least one thread")
        if not (math.isfinite(startup_limit) and startup_limit > 0.0):
            raise ConfigError("a worker needs a positive, finite start-up limit")
        if cpus is not None and len(cpus) < workers:
            raise ConfigError(
                f"{workers} workers cannot be pinned to {len(cpus)} processor(s); every "
                "worker needs a processor of its own"
            )
        self.workers = workers
        self.cpus = None if cpus is None else tuple(cpus)
        self.threads = threads
        self.startup_limit = float(startup_limit)
        self.source = source
        self.log = log
        self.log_level = log_level

    def run(
        self, queue: deque[Task], admit: Callable[[Task], bool] | None = None
    ) -> Generator[Result]:
        """Execute tasks from ``queue`` until it is empty, yielding each result.

        ``admit`` is asked about a task at its hand-over to a worker, the instant the
        run would start, and a task it refuses is dropped without an attempt; no
        worker is started for it.

        Closing the iterator early, or an exception in the caller, kills the
        workers.

        Raises:
            ConfigError: If worker processes keep exiting before they start,
                for example because a script starts a campaign without an
                ``if __name__ == "__main__":`` guard, or a worker cannot be
                created on its processor or reports other processors.
        """
        count = min(self.workers, len(queue))
        if count == 0:
            return
        context = multiprocessing.get_context("spawn")
        environment = dict.fromkeys(THREAD_VARIABLES, str(self.threads))
        cpus: Sequence[int | None] = [None] * count if self.cpus is None else self.cpus
        slots = [
            _Slot(
                number,
                context,
                environment,
                cpus[number - 1],
                self.startup_limit,
                source=self.source,
                log=self.log,
                log_level=self.log_level,
            )
            for number in range(1, count + 1)
        ]
        documents = _Documents(self.workers + 2)
        failed_starts = 0
        finished = False
        try:
            if self.cpus is not None:
                _start_helpers()
            while True:
                yield from _hand_over(slots, queue, documents, admit)
                busy = [slot for slot in slots if slot.task is not None]
                if not busy:
                    if queue:
                        continue
                    break
                deadlines = [slot.deadline for slot in busy if slot.deadline is not None]
                timeout = max(0.0, min(deadlines) - time.perf_counter()) if deadlines else None
                ready = set(wait(_waitables(busy), timeout))
                for slot in busy:
                    # A slot past its limit is read first: a message that arrived
                    # after the wait may report the start of the solve or its result.
                    if slot.signalled(ready) or (slot.expired() and slot.readable()):
                        result = slot.collect()
                        if result is _NOT_STARTED:
                            failed_starts += 1
                            if failed_starts >= STARTUP_ATTEMPTS:
                                raise ConfigError(
                                    "worker processes exit before they start "
                                    f"({slot.exit_description()}); a script that runs a "
                                    "campaign with the process backend must do so under "
                                    "'if __name__ == \"__main__\":', or use the sequential "
                                    "backend"
                                )
                            queue.appendleft(slot.release())
                            slot.close(graceful=False)
                            continue
                        if isinstance(result, Completion | Failure):
                            failed_starts = 0
                            yield result
                            continue
                    # Whatever else its pipe still holds, such as log records, a
                    # task that is past its limit after the reading is killed now.
                    if slot.task is not None and slot.expired():
                        yield slot.expire()
            finished = True
        finally:
            for slot in slots:
                slot.close(graceful=finished)


def backend_for(
    name: str,
    workers: int,
    *,
    cpus: Sequence[int] | None = None,
    threads: int = 1,
    startup_limit: float = STARTUP_SECONDS,
    source: str | None = None,
    log: LogSink | None = None,
    log_level: int | None = None,
) -> Backend:
    """Return the backend of the given name.

    Args:
        name: ``"sequential"`` or ``"process"``.
        workers: Attempts that run at once.
        cpus: The processor each worker process is pinned to, if any.
        threads: Threads the numerical libraries of a worker process may use.
        startup_limit: Seconds a worker process has, from the hand-over of a
            run, to report that the solve begins.
        source: The digest of the library source the session started on,
            for the worker processes to compare theirs with.
        log: Receives the records the worker processes log.
        log_level: The level the library's loggers of every worker process
            are lowered to, or ``None`` to leave them. The sequential
            backend has no worker process and uses none of the three.

    Raises:
        ConfigError: If the name is unknown, or the sequential backend gets
            more than one worker or processors to pin to.
    """
    if name == "sequential":
        if workers != 1:
            raise ConfigError("the sequential backend runs one attempt at a time")
        if cpus is not None:
            raise ConfigError("the sequential backend does not pin the calling process")
        return SequentialBackend()
    if name == "process":
        return ProcessBackend(
            workers,
            cpus=cpus,
            threads=threads,
            startup_limit=startup_limit,
            source=source,
            log=log,
            log_level=log_level,
        )
    raise ConfigError(f"unknown backend {name!r}; use 'sequential' or 'process'")


# -------------------------------------------------------------- worker side


@dataclass(frozen=True)
class _Job:
    """The part of a task pickled for a worker; the problem travels as a document."""

    run_id: str
    attempt: int
    digest: str
    solver: Any
    stop: Any
    seed: int
    hard_limit: float | None
    record_schedule: bool


def _worker_main(
    connection: Connection,
    name: str,
    forward: int | None = None,
    lower: bool = False,
    identify: bool = False,
) -> None:
    """Serve attempts sent over ``connection`` until it closes or sends ``None``.

    With a level as ``forward`` the worker sends the records of that level
    or higher that the library's loggers let through in it to the
    supervisor, and with ``lower`` it first lowers those loggers to the
    level, while its own handlers show what they showed
    (:class:`~dispatchatlas.core.logs.Lowered`). The records of an attempt
    are sent before its result. With ``identify`` the worker reports the
    digest of the library source it started on before it reports that it
    is ready.
    """
    detach()
    end_with_parent()
    forwarder: _Forwarder | None = None
    if forward is not None:
        if lower:
            Lowered((LIBRARY,), forward).apply()
        forwarder = _Forwarder(connection, forward)
        get_logger(LIBRARY).addHandler(forwarder)
    _load_installed_plugins()
    held: list[Any] = []  # [digest, problem] of the problem this worker keeps
    try:
        if identify:
            _send(connection, ("source", source_digest()))
        _send(connection, ("ready", affinity_cpus()))
        while (message := connection.recv()) is not None:
            result = _serve(message, name, held, connection)
            if forwarder is not None:
                forwarder.drain()
            try:
                payload = pickle.dumps(("done", result), protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as error:
                # A result that cannot travel back is reported instead of lost.
                failure = Failure(
                    result.run_id,
                    result.attempt,
                    Category.TERMINAL,
                    exception_error(error, ErrorKind.TRANSFER),
                    result.measurement,
                    solve_started=True,
                )
                payload = pickle.dumps(("done", failure), protocol=pickle.HIGHEST_PROTOCOL)
            with _SENDING:
                connection.send_bytes(payload)
    except (EOFError, OSError):
        return


def _send(connection: Connection, message: tuple[str, Any]) -> None:
    """Send a message from a worker, which writes to its pipe from one thread at a time."""
    with _SENDING:
        connection.send(message)


class _Forwarder(logging.Handler):
    """Hands the records a worker logs to the supervisor without holding the thread that logs.

    Each record travels as its time, its level, the name of its logger, and
    its message with any traceback. The thread that logs puts it on a
    queue of :data:`_LOG_QUEUE` records and goes on; a daemon thread sends
    what is queued, at most :data:`_LOG_BATCH` records in a message, and
    only that thread waits when the supervisor is slow to read. A record
    the queue has no room for is dropped and counted, and the count travels
    with the next message. A record that cannot be formatted is dropped
    too: the run goes on.
    """

    def __init__(self, connection: Connection, level: int) -> None:
        super().__init__(level)
        self._connection = connection
        self._queue: Queue[tuple[float, int, str, str]] = Queue(_LOG_QUEUE)
        self._dropped = 0
        self._counting = threading.Lock()
        self._sending = threading.Lock()  # held from taking a batch off the queue to sending it
        threading.Thread(target=self._forward, name="dispatchatlas-log", daemon=True).start()

    def emit(self, record: logging.LogRecord) -> None:
        """Queue one record for the supervisor."""
        try:
            item = (record.created, record.levelno, record.name, self.format(record))
        except Exception:  # a record that cannot be formatted is dropped
            return
        try:
            self._queue.put_nowait(item)
        except Full:
            with self._counting:
                self._dropped += 1

    def drain(self) -> None:
        """Send what is queued now, in the calling thread."""
        self._send([])

    def _forward(self) -> None:
        """Send what is queued, as it arrives, until the pipe closes."""
        while self._send([self._queue.get()]):
            pass

    def _send(self, batch: list[tuple[float, int, str, str]]) -> bool:
        """Send ``batch`` and what is queued behind it; return whether the pipe is open."""
        with self._sending:
            while True:
                with contextlib.suppress(Empty):
                    while len(batch) < _LOG_BATCH:
                        batch.append(self._queue.get_nowait())
                with self._counting:
                    dropped, self._dropped = self._dropped, 0
                if not batch and not dropped:
                    return True
                try:
                    _send(self._connection, ("log", (batch, dropped)))
                except (OSError, ValueError):
                    return False
                batch = []


def _load_installed_plugins() -> None:
    """Run the registration hooks of the installed plugins, as the calling process does.

    A plugin may register objectives as well as solvers or families, and a
    worker must know every objective before it rebuilds a problem that
    declares one.
    """
    from dispatchatlas.benchmarks.registry import default_registry as families  # noqa: PLC0415
    from dispatchatlas.solvers.registry import default_registry as solvers  # noqa: PLC0415

    solvers()
    families()


def _serve(message: tuple[Any, ...], name: str, held: list[Any], connection: Connection) -> Result:
    """Rebuild one task from a message, report that its solve begins, and execute it.

    A task with a hard limit executes inside a :class:`_Reporter`, which
    forwards the incumbents of its search.
    """
    run_id, number, digest, document, blob = message
    started, clock = utc_now(), time.perf_counter()
    try:
        # The solver first: unpickling imports its module, which may register
        # an objective the problem declares.
        job: _Job = pickle.loads(blob)
        if document is not None:
            held[:] = [digest, problem_from_dict(document)]
        if not held or held[0] != digest:
            raise ConfigError(f"the worker does not hold problem {digest[:12]}")
    except Exception as error:
        measurement = Measurement(started, utc_now(), time.perf_counter() - clock, worker=name)
        return Failure(
            run_id,
            number,
            Category.TERMINAL,
            exception_error(error, ErrorKind.TRANSFER),
            measurement,
            solve_started=False,
        )
    problem: Problem = held[1]
    task = Task(
        run_id=job.run_id,
        attempt=job.attempt,
        problem=problem,
        digest=job.digest,
        solver=job.solver,
        stop=job.stop,
        seed=job.seed,
        hard_limit=job.hard_limit,
        record_schedule=job.record_schedule,
    )
    # What the solver does once in a process comes before the start of the
    # solve, so the start-up limit covers it and no run's time limit does.
    failure = warm_up_task(task, name)
    if failure is not None:
        return failure
    # One instant is the start of the solve for the hard limit, the reporter,
    # the wall time of the attempt, and every clock of the solver.
    begun = time.perf_counter()
    with contextlib.ExitStack() as stack:
        if task.hard_limit is not None:
            reporter = stack.enter_context(
                _Reporter(connection, task.run_id, task.attempt, begun, task.stop.time_limit)
            )
            stack.enter_context(listening(reporter.note))
        _send(connection, ("started", (task.run_id, task.attempt, begun, utc_now())))
        return attempt(task, name, origin=begun)


class _Reporter:
    """Forwards the incumbents of one solve to the supervisor, at a bounded rate.

    The search calls :meth:`note` with each new incumbent. When at least
    ``interval`` seconds have passed since the last report, :meth:`note`
    sends the incumbent itself, from the thread of the search, so the report
    has left the worker before the solver goes on, whatever the solver does
    next. Otherwise :meth:`note` only keeps the incumbent, and a thread sends
    the latest one kept once the interval has passed. A solver that improves
    often therefore costs the pipe one message per interval, and its search
    one computation of the objective values per interval.

    The reporter also keeps apart the last incumbent found within
    ``time_limit`` seconds of ``begun``, the start of the solve, and reports
    it as such: a solver that searches past its time limit keeps improving,
    and only that one is what it held when the limit passed.

    An incumbent found less than ``interval`` seconds after a report waits
    for the thread. It never leaves the worker if the worker is killed
    before the interval has passed, or if the solver, right after finding
    it, enters a call that keeps the interpreter lock until the kill.

    Reports are sent only inside the ``with`` block, in which the serving
    thread sends nothing, and never by two threads at once: every thread of
    a worker writes to the pipe under one lock, which a thread that logs
    takes as well. Leaving the block ends the thread, also when a report
    falls due at that instant, and an incumbent kept and not yet sent then
    stays unsent.

    Args:
        connection: The worker's end of the pipe.
        run_id: Run id of the task.
        attempt_number: Attempt number of the task.
        begun: When the solve began, on the clock of ``time.perf_counter``.
        time_limit: The time limit of the task's protocol, if it has one.
        interval: Fewest seconds between two reports.
    """

    def __init__(
        self,
        connection: Connection,
        run_id: str,
        attempt_number: int,
        begun: float,
        time_limit: float | None = None,
        interval: float = PROGRESS_INTERVAL,
    ) -> None:
        self._connection = connection
        self._key = (run_id, attempt_number)
        self._limit = math.inf if time_limit is None else begun + time_limit
        self._interval = interval
        # The newest incumbent and the newest one within the time limit, as
        # the search left them, and the ones of each kind that were sent.
        self._latest: Improvement | None = None
        self._timely: Improvement | None = None
        self._sent_latest: Improvement | None = None
        self._sent_timely: Improvement | None = None
        self._due = 0.0  # when the next report may leave
        self._sending = threading.Lock()
        self._news = threading.Event()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._forward, name="dispatchatlas-progress", daemon=True
        )

    def note(self, improvement: Improvement) -> None:
        """Keep ``improvement``, and send it at once when a report is due."""
        if self._closed.is_set():
            return
        now = time.perf_counter()
        if now <= self._limit:
            self._timely = improvement
        self._latest = improvement
        if now >= self._due and self._sending.acquire(blocking=False):
            try:
                self._flush()
            finally:
                self._sending.release()
        elif not self._news.is_set():
            self._news.set()

    def __enter__(self) -> _Reporter:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with self._sending:
            self._closed.set()
        self._news.set()
        self._thread.join()

    def _forward(self) -> None:
        while True:
            self._news.wait()
            if self._closed.wait(max(0.0, self._due - time.perf_counter())):
                return
            self._news.clear()
            # A wait whose timeout and the closing fall together may report the
            # timeout, and the clearing then takes the wake-up of the closing.
            # The closing sets ``_closed`` before ``_news``, so it shows here.
            if self._closed.is_set():
                return
            with self._sending:
                self._flush()

    def _flush(self) -> None:
        """Send the incumbents kept and not yet sent; the caller holds the sending lock."""
        if self._closed.is_set():
            return
        timely, latest = self._timely, self._latest
        reports: list[tuple[Improvement, bool]] = []
        if timely is not None and timely is not self._sent_timely:
            reports.append((timely, True))
        if latest is not None and latest is not self._sent_latest and latest is not timely:
            reports.append((latest, False))
        if not reports:
            return
        self._sent_timely, self._sent_latest = timely, latest
        self._due = time.perf_counter() + self._interval
        # A report that cannot be built or sent is dropped: the run goes on.
        with contextlib.suppress(Exception):
            documents = [(_report(improvement), within) for improvement, within in reports]
            with _SENDING:
                self._connection.send(("progress", (*self._key, documents)))


def _report(improvement: Improvement) -> dict[str, Any]:
    """Describe an improvement as the document of an incumbent."""
    point = improvement.point
    return Incumbent(
        objective=improvement.objective,
        value=float(point.value),
        objectives={name: float(value) for name, value in improvement.objectives().items()},
        feasible=bool(point.feasible),
        evaluations=int(point.evaluations),
        iteration=int(point.iteration),
        elapsed=float(point.elapsed),
    ).to_dict()


# ---------------------------------------------------------- supervisor side


_NOT_STARTED = object()
"""What :meth:`_Slot.collect` returns when the worker exited before it started."""


def _stamped(
    result: Result, cpu: int | None, concurrent: int | None, waited: float | None = None
) -> Result:
    """Return a result whose measurement states its processor, concurrent runs, and wait."""
    measurement = replace(result.measurement, cpu=cpu, concurrent=concurrent, waited_seconds=waited)
    return replace(result, measurement=measurement)


def _start_helpers() -> None:
    """Start the helper process of ``multiprocessing`` before any worker is placed.

    On POSIX systems the first process a program spawns also starts a
    resource tracker, which would inherit the processor of the first pinned
    worker if it started with it.
    """
    if sys.platform != "win32":
        from multiprocessing import resource_tracker  # noqa: PLC0415 - POSIX only

        resource_tracker.ensure_running()


def _placed(cpu: int | None) -> contextlib.AbstractContextManager[None]:
    """Return the context a worker is created in: confined to ``cpu``, or as the caller is."""
    return contextlib.nullcontext() if cpu is None else confined(cpu)


@contextlib.contextmanager
def _exported(values: Mapping[str, str]) -> Iterator[None]:
    """Hold ``values`` in the environment of this process while a child process starts.

    A process started with ``spawn`` is created with a copy of its parent's
    environment, so the values are in place before the child's interpreter
    starts. The environment of this process is as it was afterwards.
    """
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _hand_over(
    slots: list[_Slot],
    queue: deque[Task],
    documents: _Documents,
    admit: Callable[[Task], bool] | None = None,
) -> list[Failure]:
    """Give every idle slot a task from the queue; return the failures of tasks not sent.

    Every task handed over in one call gets the same ``concurrent``: the
    tasks executing once the call has handed out its tasks. A task that ``admit``
    refuses is dropped from the queue, and the slot takes the next one.
    """
    refused: list[Failure] = []
    handed: list[_Slot] = []
    for slot in slots:
        while slot.task is None and queue:
            task = queue.popleft()
            if admit is not None and not admit(task):
                continue
            failure = slot.hand(task, documents)
            if failure is None:
                handed.append(slot)
            else:
                refused.append(failure)
    executing = sum(1 for slot in slots if slot.task is not None)
    for slot in handed:
        slot.concurrent = executing
    return refused


class _Documents:
    """Problem documents by digest, keeping the few most recently used."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, task: Task) -> dict[str, Any]:
        document = self._items.get(task.digest)
        if document is None:
            document = problem_to_dict(task.problem)
            self._items[task.digest] = document
            if len(self._items) > self._size:
                self._items.popitem(last=False)
        self._items.move_to_end(task.digest)
        return document


class _Slot:
    """One worker process and the task it runs."""

    def __init__(
        self,
        number: int,
        context: SpawnContext,
        environment: Mapping[str, str],
        cpu: int | None,
        startup_limit: float,
        *,
        source: str | None = None,
        log: LogSink | None = None,
        log_level: int | None = None,
    ) -> None:
        self.name = f"worker-{number}"
        self._source = source
        self._log = log
        # The lowest level the worker sends, and whether it lowers its loggers to it.
        self._forward: int | None = None
        self._lower = log is not None and log_level is not None
        if log is not None:
            self._forward = WORKER_LOG_LEVEL if log_level is None else log_level
        self._note: str | None = None  # what every run of this worker is noted with
        self.cpu = cpu
        self.concurrent: int | None = None
        self._context = context
        self._environment = environment
        self._startup_limit = startup_limit
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._job: int | None = None  # the worker's job object, on Windows
        self._held: str | None = None  # digest of the problem the worker keeps
        self._ready = False
        # The message of a task handed to a worker that is not ready yet.
        self._pending: tuple[str, int, str, dict[str, Any] | None, bytes] | None = None
        # Of the task the slot runs, or ran last: when the clock of its current
        # limit started, whether its solve has begun, and what it has reported.
        self._started = ""
        self._clock = 0.0
        self._solving = False
        self._waited: float | None = None
        self._incumbent: Incumbent | None = None
        self._limit_incumbent: Incumbent | None = None
        self.task: Task | None = None
        self.deadline: float | None = None

    @property
    def process(self) -> BaseProcess:
        assert self._process is not None, "the slot has no worker"
        return self._process

    @property
    def connection(self) -> Connection:
        assert self._connection is not None, "the slot has no worker"
        return self._connection

    def start(self) -> None:
        """Start a worker process for this slot, with its thread limits and on its processor.

        The process is created while the supervisor holds the thread limits
        in its environment and, for a pinned slot, is confined to the slot's
        processor, so the worker has both before its interpreter starts.

        Raises:
            ConfigError: If the worker cannot be created on its processor.
        """
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main,
            args=(child, self.name, self._forward, self._lower, self._source is not None),
            name=f"dispatchatlas-{self.name}",
        )
        try:
            with _exported(self._environment), _placed(self.cpu):
                process.start()
        except ConfigError:
            parent.close()
            child.close()
            raise
        child.close()
        # On Windows the pipe's end is typed as a PipeConnection, a sibling of Connection.
        self._connection = parent  # type: ignore[assignment, unused-ignore]
        self._process = process
        self._job = contain(process)
        self._held, self._ready, self._note = None, False, None

    def hand(self, task: Task, documents: _Documents) -> Failure | None:
        """Send a task to the worker, or return the failure of a task that cannot be sent.

        The hand-over never waits for the worker. A worker that has reported
        that it is ready is waiting for a task, and takes this one at once.
        A slot whose worker was killed or died starts a fresh one here, so no
        worker is started for a queue that has run dry, and that worker gets
        the task when it reports that it is ready (:meth:`collect`): a
        message larger than the pipe's buffer would otherwise hold the
        supervisor until the new interpreter reads it. The hand-over arms
        the start-up limit, before the fresh worker exists; the hard limit
        waits for the start of the solve.

        Raises:
            ConfigError: If a fresh worker cannot be created on its processor.
        """
        started, clock = utc_now(), time.perf_counter()
        job = _Job(
            task.run_id,
            task.attempt,
            task.digest,
            task.solver,
            task.stop,
            task.seed,
            task.hard_limit,
            task.record_schedule,
        )
        try:
            blob = pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            waited = time.perf_counter() - clock
            measurement = Measurement(started, utc_now(), waited, worker=self.name)
            error_record = exception_error(error, ErrorKind.TRANSFER)
            return Failure(
                task.run_id,
                task.attempt,
                Category.TERMINAL,
                error_record,
                measurement,
                solve_started=False,
            )
        if self._process is None:
            self.start()
        self.task = task
        self._solving, self._waited = False, None
        self._incumbent, self._limit_incumbent = None, None
        self._started, self._clock = started, clock
        self.deadline = clock + self._startup_limit
        if self._ready:
            document = None if self._held == task.digest else documents.get(task)
            self._pending = (task.run_id, task.attempt, task.digest, document, blob)
            try:
                self._dispatch()
            except OSError:
                # The worker died while idle: a fresh one takes the whole task.
                self.close(graceful=False)
                self.start()
            else:
                return None
        self._pending = (task.run_id, task.attempt, task.digest, documents.get(task), blob)
        return None

    def _dispatch(self) -> None:
        """Send the task that waits for the worker, which is ready to read it."""
        message, self._pending = self._pending, None
        if message is not None:
            self.connection.send(message)
            self._held = message[2]

    def signalled(self, ready: set[Any]) -> bool:
        """Return whether ``wait`` reported this worker's pipe or process."""
        return self.connection in ready or self.process.sentinel in ready

    def readable(self) -> bool:
        """Return whether the worker's pipe holds a message, or has closed."""
        try:
            return self.connection.poll()
        except (EOFError, OSError):
            return True

    def collect(self) -> object:
        """Read what the worker sent.

        A report that the worker is ready sends it the task that waits for
        it. A report that the solve begins arms the hard limit of the running
        task, and a report of an incumbent is kept for its failure record. The
        records the worker logged go to the sink of the backend, and the
        digest of the source the worker started on is compared with the
        session's: every run a worker on other source completes carries a
        note that says so.

        The reading ends once it has handled log records for
        :data:`LOG_READ_SECONDS`, whatever the pipe still holds, so a worker
        that logs without pause keeps the supervisor from neither the
        limits nor the other workers.

        Returns:
            The result of the task; ``None`` when only reports arrived; a
            recoverable failure when the worker died during the task; or
            :data:`_NOT_STARTED` when it died before it started.

        Raises:
            ConfigError: If a pinned worker reports that it may run on other
                processors than the slot's.
        """
        closed = False
        begun = time.perf_counter()
        try:
            while self.connection.poll():
                kind, payload = self.connection.recv()
                if kind == "ready":
                    self._ready = True
                    if self.cpu is not None and payload not in (None, (self.cpu,)):
                        raise ConfigError(
                            f"{self.name} was created on processor {self.cpu} but reports "
                            f"that it may run on processor(s) {describe_cpus(payload)}"
                        )
                    self._dispatch()
                    continue
                if kind == "started":
                    self._begin(payload)
                    continue
                if kind == "progress":
                    self._keep(payload)
                    continue
                if kind == "log":
                    self._logged(payload)
                    if time.perf_counter() - begun >= LOG_READ_SECONDS:
                        return None
                    continue
                if kind == "source":
                    self._identify(payload)
                    continue
                result = _stamped(payload, self.cpu, self.concurrent, self._waited)
                if self._note is not None and isinstance(result, Completion):
                    result = replace(result, notes=(*result.notes, self._note))
                if isinstance(result, Failure) and result.error.kind is ErrorKind.TRANSFER:
                    self._held = None
                self.release()
                return result
        except (EOFError, OSError):
            closed = True
        if not closed and self.process.exitcode is None:
            return None
        self.process.join(5.0)
        if not self._ready:
            return _NOT_STARTED
        task = self.release()
        failure = engine_failure(
            task.run_id,
            task.attempt,
            ErrorKind.WORKER_EXIT,
            f"the worker process died during the run ({self.exit_description()})",
            self._measurement(task),
            solve_started=self._solving,
        )
        self.close(graceful=False)
        return failure

    def _begin(self, report: object) -> None:
        """Start the hard limit of the running task: its worker reports that the solve begins.

        The limit and the wait count from the instant the worker states, not
        from the reading of its report, which the supervisor may come to
        late. The worker reads ``time.perf_counter``, whose clock is one for
        every process of a system on Windows, Linux, and macOS. An instant
        that does not lie between the hand-over and the reading is not from
        that clock, and the reading time takes its place.
        """
        task = self.task
        if task is None or self._solving or not isinstance(report, tuple) or len(report) != 4:
            return
        if report[:2] != (task.run_id, task.attempt):
            return
        now, stamp = time.perf_counter(), utc_now()
        instant, stated = report[2], report[3]
        if isinstance(instant, float) and self._clock <= instant <= now and isinstance(stated, str):
            now, stamp = instant, stated
        self._solving, self._waited = True, now - self._clock
        self._started, self._clock = stamp, now
        self.deadline = None if task.hard_limit is None else now + task.hard_limit

    def _logged(self, report: object) -> None:
        """Hand the records the worker logged to the sink; ignore any other report.

        A report holds the records and the number of records the worker
        dropped since its last report, which the sink receives as a warning.
        """
        if self._log is None or not isinstance(report, tuple) or len(report) != 2:
            return
        records, dropped = report
        for item in records if isinstance(records, list) else ():
            if not isinstance(item, tuple) or len(item) != 4:
                continue
            created, level, name, message = item
            if (
                isinstance(created, float)
                and isinstance(level, int)
                and isinstance(name, str)
                and isinstance(message, str)
            ):
                self._log(self.name, created, level, name, message)
        if isinstance(dropped, int) and dropped > 0:
            self._log(
                self.name,
                time.time(),
                logging.WARNING,
                _LOG.name,
                f"{dropped} log record(s) of the worker were dropped: it logged faster than "
                "they could be handed over",
            )

    def _identify(self, found: object) -> None:
        """Compare the source the worker started on with the session's; warn when they differ."""
        expected = self._source
        if expected is None or not isinstance(found, str) or found == expected:
            return
        self._note = (
            f"the worker started on other library source (digest {found[:12]}) than the "
            f"session (digest {expected[:12]})"
        )
        _LOG.warning(
            "%s started on other library source (digest %s) than the session (digest %s): the "
            "source files changed while the session ran; every run this worker completes "
            "carries a note",
            self.name,
            found[:12],
            expected[:12],
        )

    def _keep(self, report: object) -> None:
        """Keep the incumbents the running task reports; ignore any other report.

        The newest one is the task's last incumbent. One that the worker
        marks as found within the time limit, and whose search had not run
        longer than the limit either, is also its incumbent at the limit.
        """
        task = self.task
        if task is None or not isinstance(report, tuple) or len(report) != 3:
            return
        if report[:2] != (task.run_id, task.attempt) or not isinstance(report[2], list):
            return
        limit = task.stop.time_limit
        for item in report[2]:
            try:
                document, within = item
                incumbent = Incumbent.from_dict(document)
            except (TypeError, ValueError, SerializationError):
                continue
            self._incumbent = incumbent
            if within is True and limit is not None and incumbent.elapsed <= limit:
                self._limit_incumbent = incumbent

    def expired(self) -> bool:
        """Return whether the running task has passed its start-up limit or its hard limit."""
        return self.deadline is not None and time.perf_counter() >= self.deadline

    def expire(self) -> Failure:
        """Kill the worker whose task passed its limit, and describe the failure.

        Past the hard limit the failure is a timeout that holds the last
        incumbent the task reported and the last one it found within its
        time limit. Past the start-up limit, with the solve not begun, it is
        a start-up failure.
        """
        task = self.release()
        kill(self.process, self._job)
        if self._solving:
            limit = task.stop.time_limit
            failure = engine_failure(
                task.run_id,
                task.attempt,
                ErrorKind.TIMEOUT,
                f"the run passed its hard limit of {task.hard_limit:g} s (time limit "
                f"{limit:g} s), counted from the start of its solve, and its worker was killed",
                self._measurement(task),
                solve_started=True,
                last_incumbent=self._incumbent,
                limit_incumbent=self._limit_incumbent,
            )
        else:
            failure = engine_failure(
                task.run_id,
                task.attempt,
                ErrorKind.STARTUP,
                "the worker did not begin the solve within the start-up limit of "
                f"{self._startup_limit:g} s and was killed",
                self._measurement(task),
                solve_started=False,
            )
        self.close(graceful=False)
        return failure

    def release(self) -> Task:
        """Forget the running task and return it."""
        task = self.task
        assert task is not None, "no task is running"
        self.task, self.deadline, self._pending = None, None, None
        return task

    def exit_description(self) -> str:
        """Describe how the worker process ended."""
        code = self.process.exitcode
        if code is not None and code < 0:
            try:
                return f"killed by {signal.Signals(-code).name}"
            except ValueError:
                return f"killed by signal {-code}"
        return f"exit code {code}"

    def close(self, *, graceful: bool) -> None:
        """Stop the worker: ask it to finish when idle and ``graceful``, else kill it."""
        process, connection = self._process, self._connection
        if process is None or connection is None:
            return
        if graceful and self.task is None and process.is_alive():
            try:
                connection.send(None)
                process.join(5.0)
            except OSError:
                pass
        if process.is_alive():
            kill(process, self._job)
        else:
            process.join(1.0)
        release(self._job)
        connection.close()
        self._process, self._connection, self._job = None, None, None

    def _measurement(self, task: Task) -> Measurement:
        return Measurement(
            started=self._started,
            finished=utc_now(),
            wall_seconds=time.perf_counter() - self._clock,
            worker=self.name,
            hard_limit=task.hard_limit,
            cpu=self.cpu,
            concurrent=self.concurrent,
            waited_seconds=self._waited,
        )


def _waitables(slots: list[_Slot]) -> list[Any]:
    items: list[Any] = [slot.connection for slot in slots]
    items.extend(slot.process.sentinel for slot in slots)
    return items
