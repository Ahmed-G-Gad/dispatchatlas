"""One attempt of one run: solve, check, measure, and describe the outcome.

:func:`attempt` is what every backend executes, in the calling process or in
a worker process. It calls the solver, then checks the result without trusting
it: :func:`~dispatchatlas.core.validation.check_schedule` decides whether the
schedule is feasible, and :func:`~dispatchatlas.core.objectives.evaluate`
recomputes every objective from the schedule. It returns plain data, a
:class:`Completion` or a :class:`Failure`, which crosses process boundaries.

Measurements cover the solve alone: wall time with :func:`time.perf_counter`;
CPU time of the executing process with :func:`time.process_time`, which
includes any threads the solver starts; and, on Linux, the peak resident
memory of the process during the solve, read from ``/proc/self/status`` after
resetting it through ``/proc/self/clear_refs``.

:func:`classify` sorts errors into the categories of
:class:`~dispatchatlas.campaigns.records.Category`.
"""

from __future__ import annotations

import math
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
from dispatchatlas.core.errors import DispatchAtlasError
from dispatchatlas.core.model import Problem
from dispatchatlas.core.objectives import evaluate
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.solver import Solver, SolveResult, Status, StopCriteria
from dispatchatlas.core.validation import check_schedule

__all__ = [
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
    "utc_now",
]

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
        cpu_seconds: CPU seconds of the executing process, if measured.
        peak_rss_bytes: Peak resident memory of the executing process, if measured.
        worker: Where the attempt executed: ``"main"`` or a worker slot.
        hard_limit: The hard limit armed for the attempt, if any.
        cpu: The processor the executing worker is pinned to, if any.
        concurrent: Runs executing when the attempt was handed to where it
            executes, itself included. ``None`` for an attempt that was
            never handed over.
        waited_seconds: Seconds from the hand-over of the attempt to its
            worker until the supervising process learned that the solve had
            begun. ``None`` when the solve never began, and for the
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
        notes: Discrepancies between the solver's claims and the checks.
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
    """

    run_id: str
    attempt: int
    category: Category
    error: RunError
    measurement: Measurement
    solve_started: bool | None = None
    last_incumbent: Incumbent | None = None


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


def attempt(task: Task, worker: str = "main") -> Result:
    """Execute one attempt of a run.

    Exceptions raised by the solver or by the checks of its result become a
    :class:`Failure`. ``KeyboardInterrupt`` and ``SystemExit`` propagate.

    Args:
        task: The attempt.
        worker: Where it executes, for the measurement.

    Returns:
        The completion or the failure.
    """
    started = utc_now()
    measured = _peak_rss_resettable()
    clock = time.perf_counter()
    cpu = time.process_time()
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
    """
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


def engine_failure(
    run_id: str,
    attempt_number: int,
    kind: ErrorKind,
    message: str,
    measurement: Measurement,
    *,
    solve_started: bool | None = None,
    last_incumbent: Incumbent | None = None,
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


def _measurement(
    started: str, clock: float, cpu: float, measured: bool, worker: str, hard_limit: float | None
) -> Measurement:
    wall = time.perf_counter() - clock
    cpu_seconds = time.process_time() - cpu
    return Measurement(
        started=started,
        finished=utc_now(),
        wall_seconds=wall,
        cpu_seconds=cpu_seconds,
        peak_rss_bytes=_peak_rss() if measured else None,
        worker=worker,
        hard_limit=hard_limit,
    )


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
