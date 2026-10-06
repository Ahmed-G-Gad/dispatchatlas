"""One attempt of one run: solve, check, measure, and describe the outcome.

:func:`attempt` is what every backend executes, in the calling process or in
a worker process. It calls the solver, then checks the result without trusting
it: :func:`~dispatchatlas.core.validation.check_schedule` decides whether the
schedule is feasible, and :func:`~dispatchatlas.core.objectives.evaluate`
recomputes every objective from the schedule. It returns plain data, a
:class:`Completion` or a :class:`Failure`, which crosses process boundaries.

Measurements cover the solve alone: wall time with :func:`time.perf_counter`;
CPU time of the executing process with :func:`time.process_time`, which
includes any threads the solver starts, plus, on POSIX systems, the CPU time
of the child processes the solver started and waited for, from
:func:`os.times`; and, on Linux, the peak resident memory of the process
during the solve, read from ``/proc/self/status`` after resetting it through
``/proc/self/clear_refs``. On Windows the CPU time of a child process, such
as a backend that runs as a program of its own, is not in the measurement.

The instant the wall time counts from is also the origin of the solver's
clock: :func:`attempt` calls the solver inside
:func:`~dispatchatlas.core.clock.clock_origin`, so the time limit and the
elapsed time of every solver cover everything the solver does from the start
of the attempt, including what it does before its search starts, such as
building an evaluator or a model. What a solver does once in a process, such
as importing a backend, is done before that instant by :func:`prepare`, so
that it shortens the search of no run.

:func:`observe` adds what the time and the trace of the attempt show: a wall
time beyond the time limit, an elapsed time the engine's clock contradicts,
and a trace that cannot be drawn against time or evaluations or that ends at
another value than the outcome. These notes and those of :func:`describe`
are returned with the completion, logged, and stored in the run record.

:func:`classify` sorts errors into the categories of
:class:`~dispatchatlas.campaigns.records.Category`.
"""

from __future__ import annotations

import importlib
import itertools
import math
import os
import time
import traceback
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from dispatchatlas.campaigns.records import (
    Category,
    ErrorKind,
    Incumbent,
    Outcome,
    RunError,
    schedule_digest,
)
from dispatchatlas.core.clock import clock_origin
from dispatchatlas.core.errors import DispatchAtlasError, SolverError
from dispatchatlas.core.model import Problem
from dispatchatlas.core.objectives import evaluate
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.solver import Solver, SolveResult, Status, StopCriteria, StopReason
from dispatchatlas.core.validation import check_schedule

__all__ = [
    "CLOCK_MARGIN_SECONDS",
    "CLOCK_MARGIN_SHARE",
    "OVERRUN_MARGIN",
    "Completion",
    "Failure",
    "Measurement",
    "Result",
    "Task",
    "attempt",
    "classify",
    "describe",
    "engine_failure",
    "exception_error",
    "observe",
    "prepare",
    "prepare_task",
    "utc_now",
]

OVERRUN_MARGIN = 0.05
"""Share of its time limit by which the wall time of a run may pass the limit unnoted."""

CLOCK_MARGIN_SECONDS = 0.05
"""Seconds two times of a run may differ by unnoted, whatever their size.

The scheduler of the machine and the resolution of its clocks move the end of
a run by a few hundredths of a second. The margin applies to the wall time
against the time limit, and to the solver's elapsed time against the wall
time.
"""

CLOCK_MARGIN_SHARE = 0.05
"""Share of the engine's wall time by which a solver's elapsed time may differ from it unnoted.

Also the share of the time limit by which a run that stops for its time limit
may end before the limit unnoted.
"""

_TRACE_TIME_TOLERANCE = 1e-6
"""Seconds by which a trace time may pass the solver's elapsed time unnoted.

A solver that reads its clock once for a trace point and again for its
elapsed time may be a fraction of a microsecond apart.
"""

_TRACE_VALUE_TOLERANCE = 1e-6
"""Relative and absolute tolerance between the last trace value and the outcome's value."""

_TRACEBACK_CHARACTERS = 10_000
"""Longest traceback a failure record keeps; longer ones keep their end."""

_MALFORMED = frozenset({"unknown-task", "duplicate-task", "missing-task", "invalid-mode"})
"""Violation codes after which objective values cannot be computed from a schedule."""

_CLEAR_REFS = Path("/proc/self/clear_refs")
_STATUS = Path("/proc/self/status")


@dataclass(frozen=True)
class Task:
    """One attempt of one run, ready to execute.

    Attributes:
        run_id: Run id.
        attempt: Attempt number, from 1.
        problem: The problem, as solved.
        digest: Digest of the problem.
        solver: A solver instance built with the campaign's parameters.
        stop: Stop criteria of the run's protocol.
        seed: Seed of the run.
        hard_limit: Seconds after the start of the solve at which the process
            backend kills the run, or ``None``.
        record_schedule: Whether to return the schedule for the record.
    """

    run_id: str
    attempt: int
    problem: Problem
    digest: str
    solver: Solver
    stop: StopCriteria
    seed: int
    hard_limit: float | None = None
    record_schedule: bool = True


@dataclass(frozen=True)
class Measurement:
    """When, where, and at what cost an attempt executed.

    Attributes:
        started: UTC start time in ISO 8601.
        finished: UTC end time in ISO 8601.
        wall_seconds: Wall-clock seconds.
        cpu_seconds: CPU seconds of the executing process, if measured, and
            on POSIX systems also of the child processes it waited for
            during the attempt.
        peak_rss_bytes: Peak resident memory of the executing process, if measured.
        worker: Where the attempt executed: ``"main"`` or a worker slot.
        hard_limit: The hard limit armed for the attempt, if any.
        cpu: The processor the executing worker is pinned to, if any.
        concurrent: Runs executing when the attempt was handed to where it
            executes, itself included. ``None`` for an attempt that was
            never handed over.
        waited_seconds: Seconds from the hand-over of the attempt to its
            worker's slot until the instant the worker states as the start
            of the solve. ``None`` when the solve never began, and for the
            sequential backend.
    """

    started: str
    finished: str
    wall_seconds: float
    cpu_seconds: float | None = None
    peak_rss_bytes: int | None = None
    worker: str = "main"
    hard_limit: float | None = None
    cpu: int | None = None
    concurrent: int | None = None
    waited_seconds: float | None = None


@dataclass(frozen=True)
class Completion:
    """An attempt that returned a result.

    Attributes:
        run_id: Run id.
        attempt: Attempt number.
        outcome: The checked outcome.
        result_hash: :meth:`~dispatchatlas.campaigns.records.Outcome.result_hash`
            of the outcome.
        schedule: The schedule, when the campaign records schedules.
        measurement: How the attempt executed.
        notes: What the engine notices about the attempt: the claims of the
            solver that the checks contradict (:func:`describe`), and what
            its time and trace show (:func:`observe`).
    """

    run_id: str
    attempt: int
    outcome: Outcome
    result_hash: str
    schedule: Schedule | None
    measurement: Measurement
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Failure:
    """An attempt that ended without a result.

    Attributes:
        run_id: Run id.
        attempt: Attempt number.
        category: Whether another attempt may succeed.
        error: What ended the attempt.
        measurement: How the attempt executed.
        solve_started: Whether the solve had begun when the attempt ended.
        last_incumbent: The last candidate the search of a run killed at its
            hard limit had reported, if any.
        limit_incumbent: The last candidate that search had reported within
            the time limit of the run, if any.
    """

    run_id: str
    attempt: int
    category: Category
    error: RunError
    measurement: Measurement
    solve_started: bool | None = None
    last_incumbent: Incumbent | None = None
    limit_incumbent: Incumbent | None = None


type Result = Completion | Failure
"""What an attempt produces."""


def utc_now() -> str:
    """Return the current UTC time in ISO 8601 with microseconds and a ``Z`` suffix."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def classify(error: BaseException) -> Category:
    """Return whether an attempt that raised ``error`` may succeed when repeated.

    Typed DispatchAtlas errors are terminal: they state a fact about the
    configuration, the problem, or the installation that another attempt
    would meet again. Any other exception is recoverable.
    """
    return Category.TERMINAL if isinstance(error, DispatchAtlasError) else Category.RECOVERABLE


def prepare(solver: Solver) -> None:
    """Do what a solver does once in a process before it can solve.

    The first call of a solver in a process may pay a cost that no later
    call pays: the import of a backend, or of a module that loads on first
    use. Paid inside a run, that cost would shorten the search of the first
    run of every process and of no other. This function pays it outside any
    run: it imports :mod:`numpy.random`, which NumPy loads on first use, and
    calls the ``prepare`` method of the solver when the solver defines one.

    Args:
        solver: The solver about to be called.
    """
    importlib.import_module("numpy.random")
    hook = getattr(solver, "prepare", None)
    if callable(hook):
        hook()


def prepare_task(task: Task, worker: str = "main") -> Failure | None:
    """Prepare the solver of an attempt, before the clock of the attempt starts.

    Args:
        task: The attempt.
        worker: Where it executes, for the measurement.

    Returns:
        ``None``, or the failure of the attempt when :func:`prepare` raised.
    """
    started, clock = utc_now(), time.perf_counter()
    try:
        prepare(task.solver)
    except Exception as error:
        measurement = Measurement(
            started,
            utc_now(),
            time.perf_counter() - clock,
            worker=worker,
            hard_limit=task.hard_limit,
        )
        return Failure(
            task.run_id,
            task.attempt,
            classify(error),
            exception_error(error),
            measurement,
            solve_started=False,
        )
    return None


def attempt(task: Task, worker: str = "main", origin: float | None = None) -> Result:
    """Execute one attempt of a run.

    The solver is prepared first (:func:`prepare_task`), outside the time of the
    attempt. It is then called inside
    :func:`~dispatchatlas.core.clock.clock_origin`, whose origin is the
    instant the wall time of the attempt counts from, so the solver's
    elapsed time and time limit count from the same instant. A completion
    carries the notes of :func:`describe` and :func:`observe`.

    Exceptions raised by the solver, by its preparation, or by the checks of
    its result become a :class:`Failure`. ``KeyboardInterrupt`` and
    ``SystemExit`` propagate.

    Args:
        task: The attempt.
        worker: Where it executes, for the measurement.
        origin: The instant the attempt counts from, as
            :func:`time.perf_counter` counts, for a caller that has prepared
            the solver with :func:`prepare_task` and times the attempt from an
            instant of its own. By default the attempt prepares the solver
            and counts from the instant that is done.

    Returns:
        The completion or the failure.
    """
    if origin is None:
        failure = prepare_task(task, worker)
        if failure is not None:
            return failure
    started = utc_now()
    measured = _peak_rss_resettable()
    cpu = _cpu_seconds()
    with clock_origin(origin) as clock:
        try:
            result = task.solver.solve(task.problem, task.stop, task.seed)
        except Exception as error:
            measurement = _measurement(started, clock, cpu, measured, worker, task.hard_limit)
            return Failure(
                task.run_id,
                task.attempt,
                classify(error),
                exception_error(error),
                measurement,
                solve_started=True,
            )
    measurement = _measurement(started, clock, cpu, measured, worker, task.hard_limit)
    try:
        outcome, schedule, notes = describe(task.problem, result)
        result_hash = outcome.result_hash()
        notes += observe(
            outcome, task.stop, measurement.wall_seconds, iterative=task.solver.info.iterative
        )
    except Exception as error:
        return Failure(
            task.run_id,
            task.attempt,
            classify(error),
            exception_error(error),
            measurement,
            solve_started=True,
        )
    return Completion(
        run_id=task.run_id,
        attempt=task.attempt,
        outcome=outcome,
        result_hash=result_hash,
        schedule=schedule if task.record_schedule else None,
        measurement=measurement,
        notes=notes,
    )


def describe(
    problem: Problem, result: SolveResult
) -> tuple[Outcome, Schedule | None, tuple[str, ...]]:
    """Check a solver's result and describe it as an outcome.

    Feasibility and objective values come from the schedule, not from the
    solver: the violation codes of
    :func:`~dispatchatlas.core.validation.check_schedule`, and every problem
    objective recomputed by :func:`~dispatchatlas.core.objectives.evaluate`.
    Only a malformed schedule, one that misses, repeats, or invents tasks or
    uses a mode a task lacks, keeps the solver's values, since none can be
    computed from it.

    Args:
        problem: The problem solved.
        result: The solver's result.

    Returns:
        The outcome, the schedule with normalised numbers (or ``None``), and
        notes on every claim of the solver the checks contradict.

    Raises:
        SolverError: If the elapsed time, or a time or a value of the trace,
            is not a finite number. Such a result can be neither compared
            with the clock of the engine nor stored in a record.
    """
    _require_finite(result)
    notes: list[str] = []
    schedule: Schedule | None = None
    violations: tuple[str, ...] = ()
    objectives = {name: float(value) for name, value in result.objectives.items()}
    if result.schedule is not None:
        assignments = tuple(
            Assignment(item.task, int(item.mode), float(item.start), float(item.end))
            for item in result.schedule.assignments
        )
        # The checks run on a copy: evaluation caches a task index in the
        # schedule it reads, and the returned schedule must stay picklable.
        checked_schedule = Schedule(assignments)
        schedule = Schedule(assignments)
        violations = tuple(
            sorted({violation.code for violation in check_schedule(problem, checked_schedule)})
        )
        if _MALFORMED.isdisjoint(violations):
            computed = evaluate(problem, checked_schedule)
            for name, value in objectives.items():
                if name in computed and not math.isclose(
                    computed[name], value, rel_tol=1e-9, abs_tol=1e-9
                ):
                    notes.append(
                        f"the solver reported {name} = {value!r}, its schedule gives "
                        f"{computed[name]!r}"
                    )
            objectives = computed
        else:
            notes.append(f"the schedule is malformed ({', '.join(violations)})")
    status = Status(result.status)
    feasible = schedule is not None and not violations
    if status in (Status.OPTIMAL, Status.FEASIBLE) and not feasible:
        notes.append(f"the solver calls its schedule {status.value}, the check finds it infeasible")
    checked = replace(result, objectives=objectives)
    outcome = Outcome(
        status=status,
        stop_reason=result.stop_reason,
        objective=result.objective,
        value=checked.value,
        objectives=objectives,
        bound=result.bound,
        gap=checked.gap,
        feasible=feasible,
        violations=violations,
        evaluations=result.evaluations,
        iterations=result.iterations,
        elapsed=result.elapsed,
        trace=result.trace,
        front=tuple(point.values for point in result.front),
        schedule_digest=None if schedule is None else schedule_digest(schedule),
        extras=dict(result.extras),
    )
    return outcome, schedule, tuple(notes)


def observe(
    outcome: Outcome, stop: StopCriteria, wall_seconds: float, *, iterative: bool
) -> tuple[str, ...]:
    """Return what the time and the trace of a completed attempt show.

    One note is returned for each of these facts:

    - the wall time the engine measured passes the time limit by more than
      :data:`OVERRUN_MARGIN` of the limit and by more than
      :data:`CLOCK_MARGIN_SECONDS`;
    - the solver stopped for its time limit while that wall time is below the
      limit by more than :data:`CLOCK_MARGIN_SECONDS` and by more than
      :data:`CLOCK_MARGIN_SHARE` of the limit, as when a solver gives a
      search a share of its limit that counts from the start of the attempt;
    - the elapsed time the solver reports differs from that wall time by more
      than :data:`CLOCK_MARGIN_SECONDS` and by more than
      :data:`CLOCK_MARGIN_SHARE` of it;
    - the trace holds a negative time, times that decrease, or a time beyond
      the elapsed time the solver reports;
    - the evaluation counts of the trace decrease;
    - the last value of the trace differs from the value of the outcome;
    - an iterative solver returned a schedule and no trace.

    Args:
        outcome: The checked outcome of the attempt.
        stop: The stop criteria the solver received.
        wall_seconds: Wall-clock seconds of the solve, on the engine's clock.
        iterative: Whether the solver runs until a stop criterion fires.

    Returns:
        The notes, empty when nothing is found.
    """
    notes: list[str] = []
    limit = stop.time_limit
    if limit is not None:
        if wall_seconds > limit + max(CLOCK_MARGIN_SECONDS, OVERRUN_MARGIN * limit):
            notes.append(
                f"the run took {wall_seconds:.6g} s of wall time under a time limit of "
                f"{limit:g} s, {wall_seconds / limit:.3g} times the limit"
            )
        early = wall_seconds < limit - max(CLOCK_MARGIN_SECONDS, CLOCK_MARGIN_SHARE * limit)
        if early and outcome.stop_reason is StopReason.TIME_LIMIT:
            notes.append(
                f"the solver stopped for its time limit after {wall_seconds:.6g} s of wall "
                f"time, under a time limit of {limit:g} s"
            )
    elapsed = outcome.elapsed
    margin = max(CLOCK_MARGIN_SECONDS, CLOCK_MARGIN_SHARE * wall_seconds)
    if abs(wall_seconds - elapsed) > margin:
        notes.append(
            f"the solver reports {elapsed:.6g} s elapsed, the engine measured "
            f"{wall_seconds:.6g} s of wall time"
        )
    trace = outcome.trace
    if not trace:
        if iterative and outcome.schedule_digest is not None:
            notes.append("the solver is iterative and returned a schedule without a trace")
        return tuple(notes)
    times = [point.elapsed for point in trace]
    if min(times) < 0.0:
        notes.append(f"the trace holds a negative time ({min(times)!r} s)")
    fall = next(((a, b) for a, b in itertools.pairwise(times) if b < a), None)
    if fall is not None:
        notes.append(f"the times of the trace decrease (from {fall[0]!r} s to {fall[1]!r} s)")
    if max(times) > elapsed + _TRACE_TIME_TOLERANCE:
        notes.append(
            f"the trace holds a time ({max(times)!r} s) beyond the elapsed time the solver "
            f"reports ({elapsed!r} s)"
        )
    counts = [point.evaluations for point in trace]
    drop = next(((a, b) for a, b in itertools.pairwise(counts) if b < a), None)
    if drop is not None:
        notes.append(f"the evaluation counts of the trace decrease (from {drop[0]} to {drop[1]})")
    last = trace[-1].value
    if outcome.value is not None and not math.isclose(
        last, outcome.value, rel_tol=_TRACE_VALUE_TOLERANCE, abs_tol=_TRACE_VALUE_TOLERANCE
    ):
        notes.append(
            f"the last value of the trace ({last!r}) differs from the value of the outcome "
            f"({outcome.value!r})"
        )
    return tuple(notes)


def engine_failure(
    run_id: str,
    attempt_number: int,
    kind: ErrorKind,
    message: str,
    measurement: Measurement,
    *,
    solve_started: bool | None = None,
    last_incumbent: Incumbent | None = None,
    limit_incumbent: Incumbent | None = None,
) -> Failure:
    """Return the failure of an attempt that the engine, not the solver, ended.

    Timeouts and failed transfers are terminal; a worker that dies, and one
    that does not begin the solve within the start-up limit, are recoverable.
    """
    recoverable = kind in (ErrorKind.WORKER_EXIT, ErrorKind.STARTUP)
    return Failure(
        run_id,
        attempt_number,
        Category.RECOVERABLE if recoverable else Category.TERMINAL,
        RunError(kind, message),
        measurement,
        solve_started=solve_started,
        last_incumbent=last_incumbent,
        limit_incumbent=limit_incumbent,
    )


def exception_error(error: BaseException, kind: ErrorKind = ErrorKind.EXCEPTION) -> RunError:
    """Describe an exception: its qualified class, message, and traceback.

    Classes of the ``builtins`` module keep their plain name. A traceback
    longer than 10,000 characters keeps its end, where the error is raised.
    """
    cls = type(error)
    module = cls.__module__
    name = cls.__qualname__ if module == "builtins" else f"{module}.{cls.__qualname__}"
    text = "".join(traceback.format_exception(error))
    return RunError(
        kind=kind,
        message=str(error) or name,
        type=name,
        traceback=text[-_TRACEBACK_CHARACTERS:],
    )


# ------------------------------------------------------------------- helpers


def _require_finite(result: SolveResult) -> None:
    """Raise when the elapsed time or a number of the trace of ``result`` is not finite."""
    if not math.isfinite(result.elapsed):
        raise SolverError(f"the solver reports an elapsed time of {result.elapsed!r} s")
    for index, point in enumerate(result.trace):
        if not math.isfinite(point.elapsed):
            raise SolverError(f"point {index} of the trace holds a time of {point.elapsed!r} s")
        if not math.isfinite(point.value):
            raise SolverError(f"point {index} of the trace holds a value of {point.value!r}")


def _measurement(
    started: str, clock: float, cpu: float, measured: bool, worker: str, hard_limit: float | None
) -> Measurement:
    wall = time.perf_counter() - clock
    cpu_seconds = _cpu_seconds() - cpu
    return Measurement(
        started=started,
        finished=utc_now(),
        wall_seconds=wall,
        cpu_seconds=cpu_seconds,
        peak_rss_bytes=_peak_rss() if measured else None,
        worker=worker,
        hard_limit=hard_limit,
    )


def _cpu_seconds() -> float:
    """Return the CPU seconds of this process and of the children it has waited for.

    The children's part is what :func:`os.times` reports, which is zero on
    Windows.
    """
    times = os.times()
    return time.process_time() + times.children_user + times.children_system


def _peak_rss_resettable() -> bool:
    """Reset the peak resident memory of this process; return whether it worked."""
    try:
        _CLEAR_REFS.write_text("5", encoding="ascii")
    except OSError:
        return False
    return True


def _peak_rss() -> int | None:
    try:
        text = _STATUS.read_text(encoding="ascii")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None
