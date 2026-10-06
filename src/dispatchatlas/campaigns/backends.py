"""Execution backends: where the runs of a campaign execute.

Both backends take :class:`~dispatchatlas.campaigns.execution.Task` items from
a queue that the runner may extend while results arrive (a retry goes to its
end), and yield each :class:`~dispatchatlas.campaigns.execution.Completion`
or :class:`~dispatchatlas.campaigns.execution.Failure` as soon as it exists.

- :class:`SequentialBackend` runs one attempt after another in the calling
  process. Solvers must honour their own time limits, and a crash in native
  code ends the campaign. Nothing has to be picklable, which suits debugging
  and solvers defined interactively.
- :class:`ProcessBackend` runs attempts in worker processes started with the
  ``spawn`` method, so every platform behaves alike, and supervised by the
  calling process. Each worker runs one attempt at a time and keeps the
  problem of its last attempt, so a problem is sent to a worker only when the
  worker's previous attempt was on another one; runs are planned problem by
  problem, which makes that rare. The supervisor enforces hard time limits
  (see :mod:`~dispatchatlas.campaigns.timeout`): it kills a worker whose run
  passes its limit, records a terminal timeout, and hands the next run to a
  fresh worker. A worker that dies, from a crash in native code or the
  out-of-memory killer, costs only the attempt it was running, which is
  recorded as a recoverable failure. Every worker starts with the thread
  pools of numerical libraries limited: the variables of
  :data:`~dispatchatlas.campaigns.config.THREAD_VARIABLES` are in the
  environment the worker process is created with, so no import in the
  worker comes before them. A worker can also be pinned to one processor,
  which the supervisor does as soon as the process exists.

Every result states how many runs were executing when its attempt was handed
over (``concurrent``), and the processor its worker is pinned to (``cpu``).
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import pickle
import signal
import time
from collections import OrderedDict, deque
from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from multiprocessing.connection import wait
from typing import TYPE_CHECKING, Any, Protocol

from dispatchatlas.campaigns.config import THREAD_VARIABLES
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
)
from dispatchatlas.campaigns.processors import pin_process
from dispatchatlas.campaigns.records import Category, ErrorKind
from dispatchatlas.campaigns.timeout import detach, kill
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.serialization import problem_from_dict, problem_to_dict

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.context import SpawnContext
    from multiprocessing.process import BaseProcess

    from dispatchatlas.core.model import Problem

__all__ = [
    "STARTUP_ATTEMPTS",
    "Backend",
    "ProcessBackend",
    "SequentialBackend",
    "backend_for",
]

STARTUP_ATTEMPTS = 3
"""Consecutive worker processes that may exit before they start until the backend gives up."""


class Backend(Protocol):
    """Where attempts execute."""

    name: str
    """``"sequential"`` or ``"process"``."""
    workers: int
    """Attempts that run at once."""

    def run(self, queue: deque[Task]) -> Generator[Result]:
        """Execute tasks from ``queue`` until it is empty, yielding each result."""
        ...


class SequentialBackend:
    """Runs attempts one after another in the calling process."""

    name = "sequential"
    workers = 1

    def run(self, queue: deque[Task]) -> Generator[Result]:
        """Execute tasks from ``queue`` until it is empty, yielding each result."""
        while queue:
            yield _stamped(attempt(queue.popleft(), "main"), None, 1)


class ProcessBackend:
    """Runs attempts in supervised worker processes.

    Args:
        workers: Number of worker processes.
        cpus: The processor each worker is pinned to, in worker order, or
            ``None`` to leave the workers on the processors the calling
            process may use.
        threads: Threads the numerical libraries of a worker may use.
    """

    name = "process"

    def __init__(
        self, workers: int, *, cpus: Sequence[int] | None = None, threads: int = 1
    ) -> None:
        if workers < 1:
            raise ConfigError("a process backend needs at least one worker")
        if threads < 1:
            raise ConfigError("a worker needs at least one thread")
        if cpus is not None and len(cpus) < workers:
            raise ConfigError(
                f"{workers} workers cannot be pinned to {len(cpus)} processor(s); every "
                "worker needs a processor of its own"
            )
        self.workers = workers
        self.cpus = None if cpus is None else tuple(cpus)
        self.threads = threads

    def run(self, queue: deque[Task]) -> Generator[Result]:
        """Execute tasks from ``queue`` until it is empty, yielding each result.

        Closing the iterator early, or an exception in the caller, kills the
        workers.

        Raises:
            ConfigError: If worker processes keep exiting before they start,
                for example because a script starts a campaign without an
                ``if __name__ == "__main__":`` guard, or a worker cannot be
                pinned to its processor.
        """
        count = min(self.workers, len(queue))
        if count == 0:
            return
        context = multiprocessing.get_context("spawn")
        environment = dict.fromkeys(THREAD_VARIABLES, str(self.threads))
        cpus: Sequence[int | None] = [None] * count if self.cpus is None else self.cpus
        slots = [
            _Slot(number, context, environment, cpus[number - 1]) for number in range(1, count + 1)
        ]
        documents = _Documents(self.workers + 2)
        failed_starts = 0
        finished = False
        try:
            for slot in slots:
                slot.start()
            while True:
                yield from _hand_over(slots, queue, documents)
                busy = [slot for slot in slots if slot.task is not None]
                if not busy:
                    if queue:
                        continue
                    break
                deadlines = [slot.deadline for slot in busy if slot.deadline is not None]
                timeout = max(0.0, min(deadlines) - time.perf_counter()) if deadlines else None
                ready = set(wait(_waitables(busy), timeout))
                for slot in busy:
                    if slot.signalled(ready):
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
                        elif isinstance(result, Completion | Failure):
                            failed_starts = 0
                            yield result
                    elif slot.expired():
                        yield slot.expire()
            finished = True
        finally:
            for slot in slots:
                slot.close(graceful=finished)


def backend_for(
    name: str, workers: int, *, cpus: Sequence[int] | None = None, threads: int = 1
) -> Backend:
    """Return the backend of the given name.

    Args:
        name: ``"sequential"`` or ``"process"``.
        workers: Attempts that run at once.
        cpus: The processor each worker process is pinned to, if any.
        threads: Threads the numerical libraries of a worker process may use.

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
        return ProcessBackend(workers, cpus=cpus, threads=threads)
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


def _worker_main(connection: Connection, name: str) -> None:
    """Serve attempts sent over ``connection`` until it closes or sends ``None``."""
    detach()
    _load_installed_plugins()
    held: list[Any] = []  # [digest, problem] of the problem this worker keeps
    try:
        connection.send(("ready", os.getpid()))
        while (message := connection.recv()) is not None:
            result = _serve(message, name, held)
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
                )
                payload = pickle.dumps(("done", failure), protocol=pickle.HIGHEST_PROTOCOL)
            connection.send_bytes(payload)
    except (EOFError, OSError):
        return


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


def _serve(message: tuple[Any, ...], name: str, held: list[Any]) -> Result:
    """Rebuild one task from a message and execute it."""
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
    return attempt(task, name)


# ---------------------------------------------------------- supervisor side


_NOT_STARTED = object()
"""What :meth:`_Slot.collect` returns when the worker exited before it started."""


def _stamped(result: Result, cpu: int | None, concurrent: int | None) -> Result:
    """Return a result whose measurement states its processor and concurrent runs."""
    measurement = replace(result.measurement, cpu=cpu, concurrent=concurrent)
    return replace(result, measurement=measurement)


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


def _hand_over(slots: list[_Slot], queue: deque[Task], documents: _Documents) -> list[Failure]:
    """Give every idle slot a task from the queue; return the failures of tasks not sent.

    Every task handed over in one call gets the same ``concurrent``: the
    tasks executing once the call has handed out its tasks.
    """
    refused: list[Failure] = []
    handed: list[_Slot] = []
    for slot in slots:
        while slot.task is None and queue:
            failure = slot.hand(queue.popleft(), documents)
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
    ) -> None:
        self.name = f"worker-{number}"
        self.cpu = cpu
        self.concurrent: int | None = None
        self._context = context
        self._environment = environment
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._held: str | None = None  # digest of the problem the worker keeps
        self._ready = False
        self._started = ""
        self._clock = 0.0
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

        Raises:
            ConfigError: If the worker cannot be pinned to its processor.
        """
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main,
            args=(child, self.name),
            name=f"dispatchatlas-{self.name}",
        )
        with _exported(self._environment):
            process.start()
        child.close()
        self._process, self._connection = process, parent
        self._held, self._ready = None, False
        if self.cpu is not None:
            assert process.pid is not None, "a started process has an id"
            try:
                pin_process(process.pid, self.cpu)
            except ConfigError:
                self.close(graceful=False)
                raise

    def hand(self, task: Task, documents: _Documents) -> Failure | None:
        """Send a task to the worker, or return the failure of a task that cannot be sent.

        A slot whose worker was killed or died starts a fresh one here, so no
        worker is started for a queue that has run dry.
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
            return Failure(task.run_id, task.attempt, Category.TERMINAL, error_record, measurement)
        if self._process is None:
            self.start()
        document = None if self._held == task.digest else documents.get(task)
        try:
            self.connection.send((task.run_id, task.attempt, task.digest, document, blob))
        except OSError:
            # The worker died while idle: replace it and send the whole task again.
            self.close(graceful=False)
            self.start()
            document = documents.get(task)
            self.connection.send((task.run_id, task.attempt, task.digest, document, blob))
        self._held = task.digest
        self.task = task
        self._started, self._clock = utc_now(), time.perf_counter()
        self.deadline = None if task.hard_limit is None else self._clock + task.hard_limit
        return None

    def signalled(self, ready: set[Any]) -> bool:
        """Return whether ``wait`` reported this worker's pipe or process."""
        return self.connection in ready or self.process.sentinel in ready

    def collect(self) -> object:
        """Read what the worker sent.

        Returns:
            The result of the task; ``None`` when only a start-up message
            arrived; a recoverable failure when the worker died during the
            task; or :data:`_NOT_STARTED` when it died before it started.
        """
        closed = False
        try:
            while self.connection.poll():
                kind, payload = self.connection.recv()
                if kind == "ready":
                    self._ready = True
                    continue
                result = _stamped(payload, self.cpu, self.concurrent)
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
        )
        self.close(graceful=False)
        return failure

    def expired(self) -> bool:
        """Return whether the running task has passed its hard limit."""
        return self.deadline is not None and time.perf_counter() >= self.deadline

    def expire(self) -> Failure:
        """Kill the worker whose task passed its hard limit."""
        task = self.release()
        kill(self.process)
        limit = task.stop.time_limit
        failure = engine_failure(
            task.run_id,
            task.attempt,
            ErrorKind.TIMEOUT,
            f"the run passed its hard limit of {task.hard_limit:g} s (time limit {limit:g} s) "
            "and its worker was killed",
            self._measurement(task),
        )
        self.close(graceful=False)
        return failure

    def release(self) -> Task:
        """Forget the running task and return it."""
        task = self.task
        assert task is not None, "no task is running"
        self.task, self.deadline = None, None
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
            kill(process)
        else:
            process.join(1.0)
        connection.close()
        self._process, self._connection = None, None

    def _measurement(self, task: Task) -> Measurement:
        return Measurement(
            started=self._started,
            finished=utc_now(),
            wall_seconds=time.perf_counter() - self._clock,
            worker=self.name,
            hard_limit=task.hard_limit,
            cpu=self.cpu,
            concurrent=self.concurrent,
        )


def _waitables(slots: list[_Slot]) -> list[Any]:
    items: list[Any] = [slot.connection for slot in slots]
    items.extend(slot.process.sentinel for slot in slots)
    return items
