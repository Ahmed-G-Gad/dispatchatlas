"""The ``milp`` solver: a time-indexed 0-1 program solved through PuLP.

The formulation follows Pritsker, Watters and Wolfe (1969), with modes as in
Talbot (1982). Time is divided into periods of equal length, and a binary
``x[i, m, t]`` states that task ``i`` starts in mode ``m`` at period ``t``:

- every task starts exactly once: ``sum over m, t of x[i, m, t] = 1``;
- start and finish are linear in ``x``:
  ``S[i] = sum t * x[i, m, t]`` and ``F[i] = sum (t + p[i, m]) * x[i, m, t]``;
- every dependency: ``S[after] >= F[before] + lag``;
- every resource and period: the demands of the tasks running in that period
  sum to at most the capacity;
- on an exclusive resource, one where every demand is the full capacity, a
  zero-duration task started at period ``t`` excludes every task running
  across ``t``, the instant the data model defines;
- releases and hard deadlines remove the start periods that break them.

**Why the periods lose no optimum.** Times are first scaled to integers
exactly (see :func:`~dispatchatlas.solvers.exact.common.decimal_scale`). For a
regular objective some optimal schedule is active, and every start of an
active schedule is a sum of releases, durations, and lags, so it is a
multiple of their greatest common divisor, the period length used here. The
horizon is :func:`~dispatchatlas.solvers.exact.common.serial_horizon`, tightened
by a heuristic schedule when that is valid: its makespan bounds the optimal
makespan, and its total completion time divided by the smallest job weight
bounds every completion time of an optimal schedule.
"""

from __future__ import annotations

import importlib
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, ClassVar

from dispatchatlas.core.errors import MissingDependencyError, SolverError, UnsupportedProblemError
from dispatchatlas.core.model import Problem, Scalar
from dispatchatlas.core.objectives import evaluate
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.solver import (
    SolveResult,
    SolverInfo,
    Status,
    StopCriteria,
    StopReason,
    TracePoint,
)
from dispatchatlas.core.validation import DEFAULT_TOLERANCE, deadline_excess
from dispatchatlas.solvers.exact.common import (
    Clock,
    backend_version,
    confirm_value,
    decimal_scale,
    exact_result,
    jobs,
    primary_objective,
    require_features,
    scaled,
    serial_horizon,
    time_values,
)
from dispatchatlas.solvers.scheduling import Decoder

__all__ = ["BACKENDS", "OBJECTIVES", "TimeIndexedMilp"]

BACKENDS = ("cbc", "highs", "gurobi", "scip")
"""Backends the ``milp`` solver can run through PuLP."""

OBJECTIVES = frozenset({"makespan", "total_completion_time", "total_tardiness"})
"""Objectives the ``milp`` solver optimises exactly. Others are refused."""

_DISTRIBUTIONS = {"cbc": "pulp", "highs": "highspy", "gurobi": "gurobipy", "scip": "PySCIPOpt"}


@dataclass(frozen=True)
class TimeIndexedMilp:
    """Prove optimal schedules with a time-indexed MILP.

    The solver models precedence with lags, modes, multi-resource and
    cumulative demands, zero-duration tasks as the data model defines them,
    releases, due dates, and hard deadlines, and optimises ``makespan``,
    ``total_completion_time``, or ``total_tardiness``; other objectives,
    network transfers, setup times, and the permutation constraint are
    refused. Its size grows with the number of periods, so it suits problems
    with short horizons.

    Statuses come from the backend's own solution status, never from PuLP's
    overall status, which reports a time-limited CBC incumbent as optimal.
    ``OPTIMAL`` means the backend proved optimality, ``FEASIBLE`` that it
    stopped earlier with a solution, ``INFEASIBLE`` that it proved no schedule
    exists, and ``UNKNOWN`` that it stopped without a solution. ``bound``
    holds the backend's best bound when it reports one; CBC's is read from its
    log, where it is printed to six significant digits and so is loosened by
    one unit in the last digit. The published schedule is read from the
    backend's solution and checked with
    :func:`~dispatchatlas.core.validation.check_schedule`.

    The solver is not iterative: it runs until the backend proves optimality
    or infeasibility, or until the time limit. The limit covers building the
    model and handing it over: the backend gets the time left after building,
    minus the building time again, about what PuLP needs to pass the model
    on. Evaluation and iteration limits do not apply, and the result counts
    one evaluation, the schedule read back.

    Attributes:
        backend: ``"cbc"`` (bundled with PuLP), ``"highs"`` (needs
            ``highspy``), ``"gurobi"`` (needs ``gurobipy`` and a licence), or
            ``"scip"`` (needs ``pyscipopt``).
        threads: Threads the backend may use. SCIP runs single-threaded.
        relative_gap: Relative gap at which the backend may stop. The default
            0 asks for a proof. A run that stops at a positive gap reports
            ``FEASIBLE`` unless the reported bound meets the value.
        max_variables: Largest number of start binaries built. Larger models
            are refused before any memory is spent on them.
    """

    backend: str = "cbc"
    threads: int = 1
    relative_gap: float = 0.0
    max_variables: int = 200_000

    info: ClassVar[SolverInfo] = SolverInfo(
        id="milp",
        name="Time-indexed MILP",
        family="exact",
        summary=(
            "Formulates the problem as a 0-1 program with one binary per task, mode, and "
            "start period, capacity rows per resource and period, and precedence rows on "
            "start times, and solves it by branch and bound with CBC, HiGHS, Gurobi, or SCIP."
        ),
        references=(
            "pritsker1969multiproject",
            "talbot1982resource",
            "mitchell2011pulp",
            "forrest2005cbc",
            "huangfu2018parallelizing",
        ),
        stochastic=False,
        iterative=False,
        supports=frozenset(
            {
                "precedence",
                "lags",
                "modes",
                "multi-resource",
                "cumulative",
                "release",
                "due",
                "deadline",
                "groups",
            }
        ),
        requires=("exact",),
    )

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {list(BACKENDS)}, got {self.backend!r}")
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError(f"threads must be a positive integer, got {self.threads!r}")
        if self.backend == "scip" and self.threads > 1:
            raise ValueError("the scip backend runs single-threaded; use threads=1")
        gap = self.relative_gap
        if isinstance(gap, bool) or not isinstance(gap, (int, float)) or math.isnan(gap):
            raise ValueError(f"relative_gap must be a number, got {gap!r}")
        if not 0.0 <= gap <= 1.0:
            raise ValueError(f"relative_gap must lie between 0 and 1, got {gap!r}")
        object.__setattr__(self, "relative_gap", float(gap))
        limit = self.max_variables
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"max_variables must be a positive integer, got {limit!r}")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Build the time-indexed model of ``problem`` and solve it with the backend.

        Args:
            problem: The problem.
            stop: Stop criteria. Only the time limit applies.
            seed: Ignored; the backends run deterministically.

        Returns:
            The result with the backend's schedule, status, and bound.

        Raises:
            UnsupportedProblemError: If a feature or the objective is not
                modelled, a value cannot be scaled exactly, or the model would
                exceed ``max_variables``.
            MissingDependencyError: If PuLP or the chosen backend is missing.
            SolverError: If the backend fails or returns an inconsistent
                solution.
        """
        del seed
        require_features(self.info, problem)
        objective = primary_objective(problem, OBJECTIVES, self.info.id)
        pulp = _import_pulp()
        runner = _runner(self.backend, pulp)
        clock = Clock(stop.time_limit)
        dues = objective == "total_tardiness"
        scale = decimal_scale(time_values(problem, dues=dues), "time values of the problem")
        heuristic = _heuristic(problem)
        horizon = _horizon(problem, objective, scale, heuristic)
        extras: dict[str, Scalar] = {
            "backend": self.backend,
            "backend_version": backend_version(_DISTRIBUTIONS[self.backend]),
            "pulp_version": backend_version("pulp"),
            "threads": self.threads,
            "relative_gap": self.relative_gap,
            "time_scale": scale,
            "horizon": horizon / scale,
        }

        def unsolved(status: Status, reason: StopReason, raw: str) -> SolveResult:
            extras["raw_status"] = raw
            return exact_result(
                self.info,
                problem,
                objective,
                status=status,
                schedule=None,
                stop_reason=reason,
                elapsed=clock.elapsed,
                extras=extras,
            )

        building = clock.elapsed
        try:
            model = _TimeIndexedModel(problem, objective, scale, horizon, pulp, clock)
            model.build(self.max_variables)
        except _OutOfTime:
            return unsolved(Status.UNKNOWN, StopReason.TIME_LIMIT, "NOT_STARTED")
        building = clock.elapsed - building
        extras["period"] = model.period / scale
        extras["variables"] = model.variables
        if model.stranded is not None:
            # Release, durations, and deadline leave the task no start period.
            return unsolved(Status.INFEASIBLE, StopReason.COMPLETED, "NO_START_PERIOD")
        remaining = clock.remaining()
        if remaining is not None:
            # PuLP needs about as long again to hand the model to the backend.
            remaining -= building
            if remaining <= 0.0:
                return unsolved(Status.UNKNOWN, StopReason.TIME_LIMIT, "NOT_STARTED")
        outcome = runner(pulp, model.lp, remaining, self.threads, self.relative_gap)
        extras["raw_status"] = outcome.raw
        if outcome.version is not None:
            extras["backend_version"] = outcome.version
        if outcome.nodes is not None:
            extras["nodes"] = outcome.nodes
        limited = StopReason.TIME_LIMIT if stop.time_limit is not None else StopReason.COMPLETED
        bound = None if outcome.bound is None else model.original(outcome.bound)
        if outcome.infeasible:
            return unsolved(Status.INFEASIBLE, StopReason.COMPLETED, outcome.raw)
        if not outcome.solution:
            return exact_result(
                self.info,
                problem,
                objective,
                status=Status.UNKNOWN,
                schedule=None,
                stop_reason=limited,
                elapsed=clock.elapsed,
                bound=bound,
                extras=extras,
            )
        schedule = model.schedule()
        value = evaluate(problem, schedule, [objective])[objective]
        proven = outcome.proven and (
            self.relative_gap == 0.0 or (bound is not None and bound >= value - 1e-9)
        )
        if proven:
            reported = model.original(outcome.value)
            confirm_value(self.info.id, reported, value, absolute=DEFAULT_TOLERANCE)
            status, reason, bound = Status.OPTIMAL, StopReason.COMPLETED, value
        else:
            status = Status.FEASIBLE
            reason = StopReason.COMPLETED if outcome.proven else limited
            bound = None if bound is None else min(bound, value)
        # One reading of the clock: the time of the trace point is the elapsed time.
        elapsed = clock.elapsed
        return exact_result(
            self.info,
            problem,
            objective,
            status=status,
            schedule=schedule,
            stop_reason=reason,
            elapsed=elapsed,
            bound=bound,
            evaluations=1,
            trace=(TracePoint(elapsed, 1, 0, value),),
            extras=extras,
        )

    def prepare(self) -> None:
        """Import PuLP and the backend, which the first call in a process pays for.

        Raises:
            MissingDependencyError: If PuLP or the chosen backend is missing.
        """
        _runner(self.backend, _import_pulp())
        package = _PACKAGES.get(self.backend)
        if package is not None:
            importlib.import_module(package)


def _import_pulp() -> Any:
    try:
        import pulp  # noqa: PLC0415 - optional dependency
    except ImportError as error:
        raise MissingDependencyError("pulp", "exact", "The milp solver") from error
    return pulp


def _heuristic(problem: Problem) -> Schedule | None:
    """Return the schedule of the input order if it meets every deadline.

    Tasks are decoded by release time, then in their listed order, each in
    the mode that finishes it first.
    """
    order = sorted(range(problem.size), key=lambda i: (problem.tasks[i].release, i))
    schedule = Decoder(problem).schedule(order)
    return schedule if deadline_excess(problem, schedule) <= 0.0 else None


def _horizon(problem: Problem, objective: str, scale: int, heuristic: Schedule | None) -> int:
    """Return a scaled time by which some optimal schedule finishes.

    Starts from the serial horizon and, given a heuristic schedule that meets
    every deadline, tightens it with that schedule's value: the optimal
    makespan is at most the heuristic makespan, and in a schedule whose total
    completion time is at most the heuristic's, no job completes later than
    that total divided by the smallest job weight.
    """
    horizon = serial_horizon(problem, scale)
    if heuristic is None:
        return horizon
    if objective == "makespan":
        limit = heuristic.makespan
    elif objective == "total_completion_time":
        lightest = min(weight for weight, _, _ in jobs(problem))
        if lightest <= 0.0:
            return horizon
        limit = evaluate(problem, heuristic, [objective])[objective] / lightest
    else:
        return horizon
    return min(horizon, math.ceil(limit * scale - 1e-9))


class _OutOfTime(Exception):  # noqa: N818 - a control-flow signal, not an error
    """Raised while building the model when the time limit passes."""


class _TimeIndexedModel:
    """The time-indexed 0-1 program of a problem, built with PuLP.

    Times are in periods of ``period`` scaled units. ``stranded`` names a task
    that has no admissible start period, which proves the problem infeasible.
    """

    def __init__(
        self, problem: Problem, objective: str, scale: int, horizon: int, pulp: Any, clock: Clock
    ) -> None:
        self.problem = problem
        self.objective_name = objective
        self.scale = scale
        self.pulp = pulp
        self.clock = clock
        self.durations = [[scaled(mode.duration, scale) for mode in t.modes] for t in problem.tasks]
        values = [d for row in self.durations for d in row]
        values += [scaled(task.release, scale) for task in problem.tasks]
        values += [scaled(dependency.lag, scale) for dependency in problem.dependencies]
        self.period = math.gcd(*values) or 1
        self.periods = -(-horizon // self.period)
        self.lp = pulp.LpProblem("dispatchatlas_time_indexed", pulp.LpMinimize)
        self.starts: list[list[dict[int, Any]]] = []
        self.variables = 0
        self.stranded: str | None = None

    def original(self, value: float) -> float:
        """Return an objective value or bound of the model in problem units."""
        return value * self.period / self.scale

    def _check_time(self) -> None:
        if self.clock.expired():
            raise _OutOfTime

    def _variable(self, name: str, low: float, high: float | None, category: str = "") -> Any:
        """Create a variable of the model, continuous unless a category is given.

        PuLP 3.3.1 and later create variables through the problem; earlier
        releases through the variable class.
        """
        category = category or self.pulp.LpContinuous
        factory = getattr(self.lp, "add_variable", None)
        if factory is not None:
            return factory(name, low, high, category)
        return self.pulp.LpVariable(name, low, high, category)

    def _windows(self) -> list[list[range]]:
        """Return the admissible start periods of every task and mode."""
        windows: list[list[range]] = []
        for i, task in enumerate(self.problem.tasks):
            first = scaled(task.release, self.scale) // self.period
            row: list[range] = []
            for duration in self.durations[i]:
                last = self.periods - duration // self.period
                if task.deadline is not None:
                    latest_end = scaled(task.deadline, self.scale)
                    last = min(last, (latest_end - duration) // self.period)
                row.append(range(first, last + 1))
            windows.append(row)
        return windows

    def build(self, max_variables: int) -> None:
        """Create the variables, constraints, and objective.

        Raises:
            UnsupportedProblemError: If the model needs more than
                ``max_variables`` start binaries.
            _OutOfTime: If the time limit passes while building.
        """
        pulp = self.pulp
        windows = self._windows()
        self.variables = sum(len(window) for row in windows for window in row)
        if self.variables > max_variables:
            raise UnsupportedProblemError(
                f"the time-indexed model of problem {self.problem.id!r} needs "
                f"{self.variables} start binaries, above max_variables={max_variables}"
            )
        for i, task in enumerate(self.problem.tasks):
            if not any(windows[i]):
                self.stranded = task.id
                return
            row = [
                {t: self._variable(f"x_{i}_{m}_{t}", 0, 1, pulp.LpBinary) for t in window}
                for m, window in enumerate(windows[i])
            ]
            self.starts.append(row)
            once = [(variable, 1) for slots in row for variable in slots.values()]
            self.lp.addConstraint(pulp.LpAffineExpression(once) == 1, f"start_{i}")
            self._check_time()
        self._add_dependencies()
        self._add_capacities()
        self.lp.setObjective(self._objective())

    def _terms(self, task: int, *, finish: bool) -> list[tuple[Any, int]]:
        """Return ``(variable, period)`` pairs whose sum is the start or finish period."""
        terms: list[tuple[Any, int]] = []
        for m, slots in enumerate(self.starts[task]):
            span = self.durations[task][m] // self.period if finish else 0
            terms.extend((variable, t + span) for t, variable in slots.items())
        return terms

    def _add_dependencies(self) -> None:
        pulp = self.pulp
        index = self.problem.task_index
        for n, dependency in enumerate(self.problem.dependencies):
            before, after = index[dependency.before], index[dependency.after]
            terms = self._terms(after, finish=False)
            terms += [(variable, -period) for variable, period in self._terms(before, finish=True)]
            lag = scaled(dependency.lag, self.scale) // self.period
            self.lp.addConstraint(pulp.LpAffineExpression(terms) >= lag, f"after_{n}")
            self._check_time()

    def _add_capacities(self) -> None:
        pulp = self.pulp
        exclusive = self.problem.exclusive_resources
        for r, resource in enumerate(self.problem.resources):
            usage: dict[int, list[tuple[Any, float]]] = {}
            peak: dict[int, dict[int, float]] = {}
            across: dict[int, list[Any]] = {}
            instants: list[tuple[Any, int]] = []
            watch = resource.id in exclusive
            for i, task in enumerate(self.problem.tasks):
                for m, mode in enumerate(task.modes):
                    units = mode.demands.get(resource.id)
                    if units is None:
                        continue
                    span = self.durations[i][m] // self.period
                    for t, variable in self.starts[i][m].items():
                        if span == 0:
                            instants.append((variable, t))
                            continue
                        for period in range(t, t + span):
                            usage.setdefault(period, []).append((variable, units))
                            load = peak.setdefault(period, {})
                            load[i] = max(load.get(i, 0.0), units)
                        if watch:
                            for boundary in range(t + 1, t + span):
                                across.setdefault(boundary, []).append(variable)
            for period in sorted(usage):
                if sum(peak[period].values()) <= resource.capacity:
                    continue
                expression = pulp.LpAffineExpression(usage[period])
                self.lp.addConstraint(expression <= resource.capacity, f"capacity_{r}_{period}")
            if watch:
                self._add_instants(r, instants, across)
            self._check_time()

    def _add_instants(
        self, resource: int, instants: list[tuple[Any, int]], across: dict[int, list[Any]]
    ) -> None:
        """Keep every task on an exclusive resource from running across an instant.

        A zero-duration start at period ``t`` and a start whose task runs from
        before ``t`` to after it exclude each other.
        """
        for n, (variable, t) in enumerate(instants):
            running = across.get(t)
            if running:
                terms = [(variable, 1)] + [(other, 1) for other in running]
                expression = self.pulp.LpAffineExpression(terms)
                self.lp.addConstraint(expression <= 1, f"instant_{resource}_{n}")

    def _objective(self) -> Any:
        pulp = self.pulp
        tasks = self.problem.tasks
        if self.objective_name == "makespan":
            latest = self._variable("makespan", 0, self.periods)
            self._at_least(latest, range(len(tasks)), 0.0, "makespan")
            return pulp.LpAffineExpression([(latest, 1)])
        objective: list[tuple[Any, float]] = []
        if self.objective_name == "total_completion_time":
            for n, (weight, _, members) in enumerate(jobs(self.problem)):
                if len(members) == 1:
                    objective += [(v, weight * p) for v, p in self._terms(members[0], finish=True)]
                    continue
                completion = self._variable(f"job_{n}", 0, self.periods)
                self._at_least(completion, members, 0.0, f"job_{n}")
                objective.append((completion, weight))
            return pulp.LpAffineExpression(objective)
        for i, task in enumerate(tasks):
            if task.due is None:
                continue
            tardiness = self._variable(f"tardiness_{i}", 0, None)
            due = scaled(task.due, self.scale) / self.period
            self._at_least(tardiness, (i,), due, f"tardiness_{i}")
            objective.append((tardiness, task.weight))
        return pulp.LpAffineExpression(objective)

    def _at_least(self, variable: Any, tasks: Iterable[int], offset: float, name: str) -> None:
        """Constrain ``variable >= finish(task) - offset`` for every task given."""
        for i in tasks:
            terms = [(variable, 1)] + [(v, -p) for v, p in self._terms(i, finish=True)]
            self.lp.addConstraint(self.pulp.LpAffineExpression(terms) >= -offset, f"{name}_{i}")

    def schedule(self) -> Schedule:
        """Return the schedule of the backend's solution, in problem time.

        Raises:
            SolverError: If some task does not start exactly once.
        """
        assignments = []
        for i, task in enumerate(self.problem.tasks):
            chosen = [
                (m, t)
                for m, slots in enumerate(self.starts[i])
                for t, variable in slots.items()
                if (variable.varValue or 0.0) > 0.5
            ]
            if len(chosen) != 1:
                raise SolverError(f"the backend started task {task.id!r} {len(chosen)} times")
            m, t = chosen[0]
            start = t * self.period
            end = start + self.durations[i][m]
            assignments.append(Assignment(task.id, m, start / self.scale, end / self.scale))
        return Schedule.of(assignments)


# ------------------------------------------------------------------ backends


@dataclass(frozen=True)
class _Outcome:
    """What a backend reports after a solve.

    Attributes:
        raw: The backend's own status text.
        solution: Whether an integer-feasible solution is available.
        proven: Whether the backend proved it optimal within its gap tolerance.
        infeasible: Whether the backend proved that no solution exists.
        value: Model objective value of the solution, when there is one.
        bound: Best bound on the model objective, when reported.
        nodes: Branch-and-bound nodes explored, when reported.
        version: The backend's own version, when it reports one.
    """

    raw: str
    solution: bool
    proven: bool
    infeasible: bool
    value: float = math.nan
    bound: float | None = None
    nodes: int | None = None
    version: str | None = None


type _Runner = Callable[[Any, Any, float | None, int, float], _Outcome]

_PACKAGES = {"highs": "highspy", "gurobi": "gurobipy", "scip": "pyscipopt"}
"""The Python package of every backend that has one."""


def _runner(backend: str, pulp: Any) -> _Runner:
    """Return the function that solves a PuLP problem with ``backend``.

    Raises:
        MissingDependencyError: If the backend is not installed.
    """
    if backend == "cbc":
        path = _cbc_path(pulp)

        def cbc(pulp: Any, lp: Any, limit: float | None, threads: int, gap: float) -> _Outcome:
            return _solve_cbc(pulp, lp, limit, threads, gap, path)

        return cbc
    if find_spec(_PACKAGES[backend]) is None:
        raise MissingDependencyError(
            _PACKAGES[backend], backend, f"The milp solver's {backend} backend"
        )
    return {"highs": _solve_highs, "gurobi": _solve_gurobi, "scip": _solve_scip}[backend]


def _cbc_path(pulp: Any) -> str:
    """Return the CBC executable: PuLP's bundled binary, else ``cbc`` on the path.

    Raises:
        MissingDependencyError: If neither is executable.
    """
    candidates: list[str | None] = [shutil.which("cbc")]
    try:
        from pulp.apis import coin_api  # noqa: PLC0415 - optional dependency

        candidates.insert(0, pulp.COIN_CMD.executableExtension(coin_api.pulp_cbc_path))
    except (ImportError, AttributeError):
        pass
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise MissingDependencyError("pulp", "exact", "The milp solver's CBC backend")


_CBC_BOUND = re.compile(r"^Lower bound:\s*(\S+)", re.MULTILINE)
_CBC_NODES = re.compile(r"^Enumerated nodes:\s*(\d+)", re.MULTILINE)
_CBC_RESULT = re.compile(r"^Result - (.+)$", re.MULTILINE)
_CBC_VERSION = re.compile(r"^Version:\s*(\S+)", re.MULTILINE)
_CBC_WALL = re.compile(r"^Total time .*\(Wallclock seconds\):\s*(\S+)", re.MULTILINE)
_CBC_VERDICTS = ("Pre-processing says infeasible or unbounded", "Problem is infeasible")


def _solve_cbc(
    pulp: Any, lp: Any, limit: float | None, threads: int, gap: float, path: str
) -> _Outcome:
    """Solve with the CBC executable and read its solution status and log.

    PuLP's overall status calls a time-limited incumbent optimal, so the proof
    is read from ``sol_status``. CBC also declares a model infeasible when
    its time limit interrupts preprocessing, in the same words as a genuine
    proof, so an infeasibility verdict counts only when CBC finished before
    its limit. CBC prints its bound only in its log, to six significant
    digits, so the bound is loosened by one unit in the last printed digit to
    stay valid.

    No starting solution is passed: the CBC 2.10.3 binary bundled with PuLP
    crashes when a time limit expires while it processes one. The thread
    count is passed only above one, because any thread option switches that
    binary to its threaded code, which can stall for seconds; without it CBC
    runs its serial code.
    """
    handle, log = tempfile.mkstemp(prefix="dispatchatlas-cbc-", suffix=".log")
    os.close(handle)
    try:
        solver = pulp.COIN_CMD(
            path=path,
            msg=False,
            timeLimit=limit,
            gapRel=gap,
            threads=threads if threads > 1 else None,
            logPath=log,
        )
        try:
            lp.solve(solver)
        except pulp.PulpSolverError as error:
            raise SolverError(f"CBC failed: {error}") from error
        text = Path(log).read_text(encoding="utf-8", errors="replace")
    finally:
        Path(log).unlink(missing_ok=True)
    solution = lp.sol_status in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible)
    wall = _CBC_WALL.search(text)
    finished = limit is None or (wall is not None and float(wall.group(1)) < limit)
    return _Outcome(
        raw=_cbc_verdict(text) or f"{pulp.LpStatus[lp.status]}; {pulp.LpSolution[lp.sol_status]}",
        solution=solution,
        proven=lp.sol_status == pulp.LpSolutionOptimal,
        infeasible=lp.status == pulp.LpStatusInfeasible and finished,
        value=float(pulp.value(lp.objective) or 0.0) if solution else math.nan,
        bound=_cbc_bound(text),
        nodes=int(nodes.group(1)) if (nodes := _CBC_NODES.search(text)) else None,
        version=release.group(1) if (release := _CBC_VERSION.search(text)) else None,
    )


def _cbc_verdict(log: str) -> str | None:
    """Return CBC's final verdict line from its log, if it printed one."""
    result = _CBC_RESULT.search(log)
    if result is not None:
        return result.group(1).strip()
    return next((verdict for verdict in _CBC_VERDICTS if verdict in log), None)


def _cbc_bound(log: str) -> float | None:
    """Return the lower bound printed in a CBC log, loosened to stay valid."""
    match = _CBC_BOUND.search(log)
    if match is None:
        return None
    try:
        printed = float(match.group(1))
    except ValueError:
        return None
    if not math.isfinite(printed) or abs(printed) >= 1e30:
        return None
    return printed - max(abs(printed) * 1e-5, 1e-9)


def _solve_highs(pulp: Any, lp: Any, limit: float | None, threads: int, gap: float) -> _Outcome:
    """Solve with HiGHS in process and read its model status and information."""
    import highspy  # noqa: PLC0415 - optional dependency

    solver = pulp.HiGHS(msg=False, timeLimit=limit, gapRel=gap, threads=threads)
    lp.solve(solver)
    highs = lp.solverModel
    status = highs.getModelStatus()
    info = highs.getInfo()
    feasible = int(highspy.SolutionStatus.kSolutionStatusFeasible)
    solution = int(info.primal_solution_status) == feasible
    bound = float(info.mip_dual_bound)
    return _Outcome(
        raw=str(highs.modelStatusToString(status)),
        solution=solution,
        proven=status == highspy.HighsModelStatus.kOptimal,
        infeasible=status == highspy.HighsModelStatus.kInfeasible,
        value=float(info.objective_function_value) if solution else math.nan,
        bound=bound if math.isfinite(bound) else None,
        nodes=int(info.mip_node_count),
    )


_GUROBI_STATUS = {
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    7: "ITERATION_LIMIT",
    8: "NODE_LIMIT",
    9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT",
    11: "INTERRUPTED",
    12: "NUMERIC",
    13: "SUBOPTIMAL",
}


def _solve_gurobi(pulp: Any, lp: Any, limit: float | None, threads: int, gap: float) -> _Outcome:
    """Solve with Gurobi in process and read its status, solution count, and bound.

    The gap is passed as Gurobi's ``MIPGap`` parameter directly, because PuLP
    skips a zero gap and Gurobi's default gap is not zero.
    """
    import gurobipy  # noqa: PLC0415 - optional dependency

    options: dict[str, Any] = {"MIPGap": gap, "Threads": threads}
    # A managed environment starts silent and is released by close().
    solver = pulp.GUROBI(msg=False, timeLimit=limit, manageEnv=True, **options)
    try:
        lp.solve(solver)
        model = lp.solverModel
        code = int(model.Status)
        solution = int(model.SolCount) > 0
        try:
            bound = float(model.ObjBound)
        except (AttributeError, gurobipy.GurobiError):
            bound = math.nan
        outcome = _Outcome(
            raw=_GUROBI_STATUS.get(code, f"STATUS_{code}"),
            solution=solution,
            proven=code == 2,
            infeasible=code in (3, 4),
            value=float(model.ObjVal) if solution else math.nan,
            bound=bound if math.isfinite(bound) else None,
            nodes=int(model.NodeCount),
        )
    except gurobipy.GurobiError as error:
        raise SolverError(f"Gurobi failed: {error}") from error
    finally:
        solver.close()
    return outcome


def _solve_scip(pulp: Any, lp: Any, limit: float | None, threads: int, gap: float) -> _Outcome:
    """Solve with SCIP in process and read its status, solutions, and dual bound."""
    del threads  # SCIP runs single-threaded; the solver refuses threads > 1.
    solver = pulp.SCIP_PY(msg=False, timeLimit=limit, gapRel=gap)
    lp.solve(solver)
    model = lp.solverModel
    status = str(model.getStatus())
    solution = int(model.getNSols()) > 0
    bound = float(model.getDualbound())
    return _Outcome(
        raw=status,
        solution=solution,
        proven=status == "optimal",
        infeasible=status == "infeasible",
        value=float(model.getObjVal()) if solution else math.nan,
        bound=bound if math.isfinite(bound) and abs(bound) < 1e19 else None,
        nodes=int(model.getNNodes()),
    )
