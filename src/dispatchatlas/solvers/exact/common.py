"""Machinery the exact solvers share: exact integer time, horizons, and checked results.

The model-based solvers state a problem in integer time. :func:`decimal_scale`
picks the smallest power of ten that turns every time value into an integer
and refuses a problem it cannot represent exactly, so an optimum the backend
proves for the model is an optimum of the problem, never of a rounded
neighbour. :func:`serial_horizon` bounds the finish time of some optimal
schedule, which lets a model restrict time without losing every optimum.
:func:`exact_result` checks a backend's schedule with
:func:`~dispatchatlas.core.validation.check_schedule` before publishing it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Collection, Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version

from dispatchatlas.core.clock import clock_start
from dispatchatlas.core.errors import SolverError, UnsupportedProblemError
from dispatchatlas.core.model import Dependency, Problem, Scalar
from dispatchatlas.core.objectives import evaluate, is_built_in
from dispatchatlas.core.schedule import Schedule
from dispatchatlas.core.solver import (
    SolveResult,
    SolverInfo,
    Status,
    StopCriteria,
    StopReason,
    TracePoint,
)
from dispatchatlas.core.validation import check_schedule

__all__ = [
    "MAX_DECIMALS",
    "VALUE_TOLERANCE",
    "Clock",
    "backend_version",
    "confirm_value",
    "decimal_scale",
    "exact_result",
    "jobs",
    "primary_objective",
    "require_bounded",
    "require_features",
    "scaled",
    "serial_horizon",
    "time_values",
    "transfer_times",
]

MAX_DECIMALS = 6
"""Most digits after the decimal point an exactly scaled value may have."""

_ROUNDING = 1e-9
"""Largest distance, in the value's own unit, from a multiple of one over the scale.

It absorbs the rounding noise of binary floating point and stays three orders
of magnitude below the tolerance of
:func:`~dispatchatlas.core.validation.check_schedule`, so rounding to the
scale never moves a deadline, release, or duration by a detectable amount.
"""

_LARGEST_EXACT = 2**53
"""Largest magnitude below which every integer is exact as a float."""

VALUE_TOLERANCE = 1e-6
"""Relative tolerance when comparing a backend's objective with the schedule's."""


def scaled(value: float, scale: int) -> int:
    """Return ``value * scale`` as the nearest integer."""
    return round(value * scale)


def decimal_scale(values: Iterable[float], what: str) -> int:
    """Return the smallest power of ten, at most ``10**6``, that makes every value integral.

    Args:
        values: The numbers to represent exactly.
        what: Plural description of the numbers, used in the error message.

    Returns:
        The scale factor.

    Raises:
        UnsupportedProblemError: If some value has more than
            :data:`MAX_DECIMALS` decimals, or a scaled value is too large for
            exact integer arithmetic.
    """
    numbers = [float(value) for value in values if value != 0.0]
    for decimals in range(MAX_DECIMALS + 1):
        scale: int = 10**decimals
        if all(_is_integral(number * scale, scale) for number in numbers):
            largest = max((abs(number) * scale for number in numbers), default=0.0)
            if largest > _LARGEST_EXACT:
                raise UnsupportedProblemError(
                    f"{what} reach {largest:.3g} at scale {scale}, too large to represent "
                    "exactly in integer arithmetic"
                )
            return scale
    raise UnsupportedProblemError(
        f"{what} are not decimals with at most {MAX_DECIMALS} digits after the point, "
        "so an integer model cannot represent them exactly"
    )


def _is_integral(value: float, scale: int) -> bool:
    return abs(value - round(value)) <= _ROUNDING * scale


def transfer_times(problem: Problem, dependency: Dependency) -> dict[tuple[int, int], float]:
    """Return the transfer time of a dependency for every pair of modes.

    Args:
        problem: The problem.
        dependency: One of its dependencies.

    Returns:
        ``times[(before mode, after mode)]``, empty when the dependency moves no
        data across a network. Pairs located on one resource transfer in zero
        time.
    """
    if problem.network is None or dependency.data <= 0.0:
        return {}
    before = problem.task(dependency.before)
    after = problem.task(dependency.after)
    return {
        (a, b): problem.transfer_time(dependency, source.location, target.location)
        for a, source in enumerate(before.modes)
        for b, target in enumerate(after.modes)
    }


def time_values(problem: Problem, *, dues: bool) -> list[float]:
    """Return every time value a model of ``problem`` states.

    That is every duration, release, deadline, lag, transfer time, and setup
    time, plus the due dates when ``dues`` is true.
    """
    values: list[float] = []
    for task in problem.tasks:
        values.append(task.release)
        values.extend(mode.duration for mode in task.modes)
        if task.deadline is not None:
            values.append(task.deadline)
        if dues and task.due is not None:
            values.append(task.due)
    for dependency in problem.dependencies:
        values.append(dependency.lag)
        values.extend(transfer_times(problem, dependency).values())
    if problem.setups is not None:
        values.extend(problem.setups.times.values())
    return values


def serial_horizon(problem: Problem, scale: int) -> int:
    """Return a scaled time by which some optimal schedule finishes every task.

    The bound is the latest release plus, for every task, its longest mode and
    its largest incoming setup, plus every dependency's lag and largest
    transfer time.

    It holds for every regular objective, one that never improves when a task
    finishes later: decoding the start order of an optimal schedule with the
    serial schedule-generation scheme starts no task later than that schedule
    (Kolisch 1996), and starts each task no later than the finish of the tasks
    before it plus its own release, lags, transfers, and setup, which sums to
    at most this bound.

    Args:
        problem: The problem.
        scale: Time scale from :func:`decimal_scale`.

    Returns:
        The horizon in scaled time units.
    """
    total = max(scaled(task.release, scale) for task in problem.tasks)
    for task in problem.tasks:
        total += max(scaled(mode.duration, scale) for mode in task.modes)
    for dependency in problem.dependencies:
        transfers = transfer_times(problem, dependency).values()
        total += scaled(dependency.lag, scale) + max(
            (scaled(value, scale) for value in transfers), default=0
        )
    if problem.setups is not None:
        into: dict[str, int] = {}
        for (_, _, following), setup in problem.setups.times.items():
            into[following] = max(into.get(following, 0), scaled(setup, scale))
        total += sum(into.values())
    return total


def jobs(problem: Problem) -> list[tuple[float, float, tuple[int, ...]]]:
    """Return ``(weight, release, task indices)`` of every job.

    A job is a named group of tasks or a task without a group, the unit that
    completion-time objectives sum over. The order matches
    :mod:`dispatchatlas.core.objectives`: groups first, then ungrouped tasks.
    """
    result: list[tuple[float, float, tuple[int, ...]]] = []
    for members in problem.groups.values():
        first = problem.tasks[members[0]]
        release = min(problem.tasks[i].release for i in members)
        result.append((first.weight, release, members))
    for i, task in enumerate(problem.tasks):
        if task.group is None:
            result.append((task.weight, task.release, (i,)))
    return result


def require_features(info: SolverInfo, problem: Problem) -> None:
    """Refuse a problem with features the solver does not declare.

    Raises:
        UnsupportedProblemError: If ``problem`` has an unsupported feature.
    """
    missing = info.unsupported(problem)
    if missing:
        raise UnsupportedProblemError(
            f"solver {info.id!r} does not model features {sorted(missing)} "
            f"of problem {problem.id!r}"
        )


def require_bounded(info: SolverInfo, stop: StopCriteria) -> None:
    """Refuse unbounded stop criteria for an iterative solver.

    Raises:
        SolverError: If the solver is iterative and ``stop`` sets no time,
            evaluation, or iteration limit.
    """
    if info.iterative and not stop.bounded:
        raise SolverError(f"solver {info.id!r} needs a time, evaluation, or iteration limit")


def primary_objective(problem: Problem, supported: Collection[str], solver_id: str) -> str:
    """Return the problem's primary objective if the solver models it exactly.

    Raises:
        UnsupportedProblemError: If the objective is not in ``supported``, or
            its name is registered with a definition other than the library's.
    """
    name = problem.objectives[0]
    if name not in supported:
        raise UnsupportedProblemError(
            f"solver {solver_id!r} cannot optimise objective {name!r}; "
            f"it models {sorted(supported)}"
        )
    if not is_built_in(name):
        raise UnsupportedProblemError(
            f"objective {name!r} is registered with a custom definition, "
            f"which solver {solver_id!r} does not model"
        )
    return name


def backend_version(distribution: str) -> str:
    """Return the installed version of a distribution, or ``"unknown"``."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


class Clock:
    """Wall-clock time of one run against an optional limit.

    The clock starts at :func:`~dispatchatlas.core.clock.clock_start`: the
    instant it is created, or the origin set by
    :func:`~dispatchatlas.core.clock.clock_origin`.

    Args:
        limit: Seconds the run may take, or ``None``.
    """

    def __init__(self, limit: float | None) -> None:
        self.limit = limit
        self._started = clock_start()

    @property
    def elapsed(self) -> float:
        """Seconds since the clock started."""
        return time.perf_counter() - self._started

    def remaining(self) -> float | None:
        """Seconds left before the limit, never negative, or ``None`` without one."""
        if self.limit is None:
            return None
        return max(0.0, self.limit - self.elapsed)

    def expired(self) -> bool:
        """Whether the limit has passed."""
        return self.limit is not None and self.elapsed >= self.limit


def confirm_value(
    solver_id: str, reported: float, schedule_value: float, *, absolute: float = 1e-9
) -> None:
    """Check that a backend's objective value matches its schedule's.

    Args:
        solver_id: Id of the solver, for the message.
        reported: The backend's objective value, in problem units.
        schedule_value: The objective of the schedule read back from it.
        absolute: Absolute tolerance, for backends with floating-point
            feasibility tolerances.

    Raises:
        SolverError: If the two differ, which means the model and the problem
            disagree and no optimality claim may be published.
    """
    if not math.isclose(reported, schedule_value, rel_tol=VALUE_TOLERANCE, abs_tol=absolute):
        raise SolverError(
            f"solver {solver_id!r} reported objective {reported} for a schedule "
            f"whose objective is {schedule_value}"
        )


def exact_result(
    info: SolverInfo,
    problem: Problem,
    objective: str,
    *,
    status: Status,
    schedule: Schedule | None,
    stop_reason: StopReason,
    elapsed: float,
    bound: float | None = None,
    evaluations: int = 0,
    iterations: int = 0,
    trace: Sequence[TracePoint] = (),
    extras: Mapping[str, Scalar] | None = None,
) -> SolveResult:
    """Build the result of an exact solver after checking its schedule.

    Args:
        info: Metadata of the solver.
        problem: The problem solved.
        objective: Name of the primary objective.
        status: What is known about the schedule.
        schedule: The backend's schedule, or ``None`` without one.
        stop_reason: Why the run ended.
        elapsed: Wall-clock seconds of the run.
        bound: Proven lower bound on the primary objective.
        evaluations: Schedules the solver scored.
        iterations: Main-loop iterations, as the solver defines them.
        trace: Improvements of the incumbent over time.
        extras: Backend facts such as its raw status.

    Returns:
        The result.

    Raises:
        SolverError: If the schedule breaks a constraint of the problem, or a
            status that needs a schedule comes without one.
    """
    if schedule is None:
        if status in (Status.OPTIMAL, Status.FEASIBLE):
            raise SolverError(f"solver {info.id!r} claims status {status} without a schedule")
        values: dict[str, float] = {}
    else:
        violations = check_schedule(problem, schedule)
        if violations:
            details = "; ".join(violation.message for violation in violations[:3])
            raise SolverError(f"solver {info.id!r} produced an infeasible schedule: {details}")
        values = evaluate(problem, schedule)
    return SolveResult(
        solver=info.id,
        problem=problem.id,
        objective=objective,
        status=status,
        schedule=schedule,
        objectives=values,
        stop_reason=stop_reason,
        elapsed=elapsed,
        evaluations=evaluations,
        iterations=iterations,
        bound=bound,
        trace=tuple(trace),
        extras=dict(extras or {}),
    )
