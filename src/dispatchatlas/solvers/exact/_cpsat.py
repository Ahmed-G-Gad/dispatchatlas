"""The CP-SAT interval model behind the ``cp-sat`` solver.

This module imports OR-Tools when it loads, so
:class:`~dispatchatlas.solvers.exact.cpsat.CpSat` imports it only when a solve
starts. Every constraint and objective is stated exactly in integer time (see
:func:`~dispatchatlas.solvers.exact.common.decimal_scale`), and every objective
is encoded with equalities, so the model's objective value of any solution
equals the problem's objective value of the schedule read back from it.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ortools.sat.python import cp_model

from dispatchatlas.core.errors import SolverError, UnsupportedProblemError
from dispatchatlas.core.logs import get_logger
from dispatchatlas.core.model import Problem, Resource, Scalar, SetupTimes
from dispatchatlas.core.objectives import evaluate
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.solver import (
    BoundPoint,
    SolveResult,
    SolverInfo,
    Status,
    StopCriteria,
    StopReason,
    TracePoint,
)
from dispatchatlas.solvers.exact.common import (
    Clock,
    backend_version,
    confirm_value,
    decimal_scale,
    exact_result,
    jobs,
    scaled,
    serial_horizon,
    time_values,
    transfer_times,
)

__all__ = ["IntervalModel", "ModelObjective", "solve"]

_LOG = get_logger(__name__)

_DUE_OBJECTIVES = frozenset({"total_tardiness", "max_lateness", "tardy_tasks"})

type _Literals = Sequence[cp_model.BoolVarT]


@dataclass(frozen=True)
class ModelObjective:
    """An integer objective of the model and its conversion to problem units.

    The problem's objective value of a solution is
    ``(model value + offset) / divisor``.

    Attributes:
        expression: The integer expression the model minimises.
        offset: Constant added before dividing.
        divisor: Positive integer that undoes the time and weight scaling.
    """

    expression: cp_model.LinearExprT
    offset: int
    divisor: int

    def original(self, value: float) -> float:
        """Return a model objective value, or bound, in the problem's units."""
        return (value + self.offset) / self.divisor


def _total(
    expressions: Sequence[cp_model.IntVar], weights: Sequence[int] | None = None
) -> cp_model.LinearExpr:
    coefficients = list(weights) if weights is not None else [1] * len(expressions)
    return cp_model.LinearExpr.weighted_sum(list(expressions), coefficients)


class IntervalModel:
    """A problem stated as a CP-SAT model with one optional interval per task mode.

    Each task has a start and an end variable and one Boolean per mode, exactly
    one of them true. Each mode owns an optional interval of fixed size that is
    present when the mode is chosen, so every resource constraint sees the
    chosen mode's demand. The model states:

    - release times and hard deadlines as bounds on starts and ends;
    - a no-overlap constraint on each exclusive resource, one where every
      demand is the full capacity, and a cumulative constraint on every other
      resource;
    - each dependency as ``start(after) >= end(before) + lag``, plus the
      transfer time of each pair of modes, enforced when both are chosen;
    - setup times with one circuit per machine, whose arcs choose the task
      that directly follows another and enforce the setup between them;
    - the permutation constraint with one position per group, all different:
      of two tasks of different groups on one machine, the task of the earlier
      group finishes before the other starts.

    A zero-duration mode keeps its interval of size zero. No-overlap treats it
    as an instant that no other interval may run across, and cumulative
    constraints leave it out, which is exactly how
    :mod:`dispatchatlas.core.model` defines zero-duration tasks on exclusive
    and on other resources.

    Args:
        problem: The problem.
        scale: Time units per problem time unit, from
            :func:`~dispatchatlas.solvers.exact.common.decimal_scale`.
        horizon: Scaled time by which some optimal schedule finishes.

    Raises:
        UnsupportedProblemError: If a demand or capacity has too many decimals.
    """

    def __init__(self, problem: Problem, scale: int, horizon: int) -> None:
        self.problem = problem
        self.scale = scale
        self.horizon = horizon
        self.model = cp_model.CpModel()
        self.starts: list[cp_model.IntVar] = []
        self.ends: list[cp_model.IntVar] = []
        self.chosen: list[list[cp_model.IntVar]] = []
        self.durations: list[list[int]] = []
        self.intervals: list[list[cp_model.IntervalVar]] = []
        self._uses: dict[tuple[int, str], cp_model.IntVar] = {}
        self._makespan: cp_model.IntVar | None = None
        self.transfer_pairs = 0
        self._add_tasks()
        self._add_resources()
        self._add_dependencies()
        if problem.setups is not None and problem.setups.times:
            self._add_setups(problem.setups)
        if problem.permutation:
            self._add_permutation()

    # ------------------------------------------------------------ structure

    def _time(self, value: float) -> int:
        return scaled(value, self.scale)

    def _add_tasks(self) -> None:
        model = self.model
        for i, task in enumerate(self.problem.tasks):
            release = self._time(task.release)
            start = model.new_int_var(release, self.horizon, f"start_{i}")
            end = model.new_int_var(release, self.horizon, f"end_{i}")
            if task.deadline is not None:
                model.add(end <= self._time(task.deadline))
            chosen = [model.new_bool_var(f"mode_{i}_{m}") for m in range(len(task.modes))]
            model.add_exactly_one(chosen)
            durations = [self._time(mode.duration) for mode in task.modes]
            model.add(end == start + _total(chosen, durations))
            intervals = [
                model.new_optional_fixed_size_interval_var(start, size, literal, f"run_{i}_{m}")
                for m, (size, literal) in enumerate(zip(durations, chosen, strict=True))
            ]
            self.starts.append(start)
            self.ends.append(end)
            self.chosen.append(chosen)
            self.durations.append(durations)
            self.intervals.append(intervals)

    def _add_resources(self) -> None:
        exclusive = self.problem.exclusive_resources
        for resource in self.problem.resources:
            entries: list[tuple[cp_model.IntervalVar, float]] = []
            for i, task in enumerate(self.problem.tasks):
                for m, mode in enumerate(task.modes):
                    units = mode.demands.get(resource.id)
                    if units is None:
                        continue
                    if resource.id in exclusive or self.durations[i][m] > 0:
                        entries.append((self.intervals[i][m], units))
            if not entries:
                continue
            intervals = [interval for interval, _ in entries]
            if resource.id in exclusive:
                self.model.add_no_overlap(intervals)
                continue
            units_scale = decimal_scale(
                [units for _, units in entries] + [resource.capacity],
                f"demands and capacity of resource {resource.id!r}",
            )
            self.model.add_cumulative(
                intervals,
                [scaled(units, units_scale) for _, units in entries],
                scaled(resource.capacity, units_scale),
            )

    def _add_dependencies(self) -> None:
        index = self.problem.task_index
        for dependency in self.problem.dependencies:
            before, after = index[dependency.before], index[dependency.after]
            lag = self._time(dependency.lag)
            transfers = {
                pair: self._time(value)
                for pair, value in transfer_times(self.problem, dependency).items()
            }
            least = min(transfers.values(), default=0)
            self.model.add(self.starts[after] >= self.ends[before] + lag + least)
            for (source, target), transfer in sorted(transfers.items()):
                if transfer > least:
                    self.transfer_pairs += 1
                    both: list[cp_model.BoolVarT] = [
                        self.chosen[before][source],
                        self.chosen[after][target],
                    ]
                    self.model.add(
                        self.starts[after] >= self.ends[before] + lag + transfer
                    ).only_enforce_if(both)

    def uses(self, task: int, resource: str) -> cp_model.IntVar:
        """Return a literal that is true when ``task`` runs in a mode demanding ``resource``."""
        literal = self._uses.get((task, resource))
        if literal is None:
            literal = self.model.new_bool_var(f"uses_{task}_{resource}")
            modes = [
                self.chosen[task][m]
                for m, mode in enumerate(self.problem.tasks[task].modes)
                if resource in mode.demands
            ]
            self.model.add(literal == _total(modes))
            self._uses[(task, resource)] = literal
        return literal

    def _add_setups(self, setups: SetupTimes) -> None:
        tasks = self.problem.tasks
        model = self.model
        for resource in sorted(setups.resources):
            users = [
                i for i, task in enumerate(tasks) if any(resource in m.demands for m in task.modes)
            ]
            if not users:
                continue
            present = [self.uses(i, resource) for i in users]
            idle = model.new_bool_var(f"idle_{resource}")
            model.add(_total(present) == 0).only_enforce_if(idle)
            model.add(_total(present) >= 1).only_enforce_if(idle.negated())
            arcs: list[tuple[int, int, cp_model.BoolVarT]] = [(0, 0, idle)]
            for k, i in enumerate(users, start=1):
                arcs.append((k, k, present[k - 1].negated()))
                first = model.new_bool_var(f"first_{resource}_{i}")
                arcs.append((0, k, first))
                initial = self._time(setups.get(resource, None, tasks[i].id))
                if initial > 0:
                    model.add(self.starts[i] >= initial).only_enforce_if(first)
                arcs.append((k, 0, model.new_bool_var(f"last_{resource}_{i}")))
                for n, j in enumerate(users, start=1):
                    if j == i:
                        continue
                    follows = model.new_bool_var(f"next_{resource}_{i}_{j}")
                    arcs.append((k, n, follows))
                    setup = self._time(setups.get(resource, tasks[i].id, tasks[j].id))
                    model.add(self.starts[j] >= self.ends[i] + setup).only_enforce_if(follows)
            model.add_circuit(arcs)

    def _add_permutation(self) -> None:
        tasks = self.problem.tasks
        keys = [task.group if task.group is not None else task.id for task in tasks]
        groups = list(dict.fromkeys(keys))
        if len(groups) < 2:
            return
        number = {key: n for n, key in enumerate(groups)}
        # The data model guarantees that a group uses each machine at most once.
        users = {
            resource.id: [
                i
                for i, task in enumerate(tasks)
                if any(resource.id in mode.demands for mode in task.modes)
            ]
            for resource in self.problem.resources
        }
        model = self.model
        positions = [model.new_int_var(0, len(groups) - 1, f"position_{n}") for n in groups]
        model.add_all_different(positions)
        earlier: dict[tuple[int, int], cp_model.IntVar] = {}
        for resource_id, members in users.items():
            for x, i in enumerate(members):
                for j in members[x + 1 :]:
                    g, h = sorted((number[keys[i]], number[keys[j]]))
                    literal = earlier.get((g, h))
                    if literal is None:
                        literal = model.new_bool_var(f"before_{g}_{h}")
                        model.add(positions[g] < positions[h]).only_enforce_if(literal)
                        model.add(positions[g] > positions[h]).only_enforce_if(literal.negated())
                        earlier[(g, h)] = literal
                    first, second = (i, j) if number[keys[i]] == g else (j, i)
                    on_resource = (self.uses(i, resource_id), self.uses(j, resource_id))
                    self._ordered(first, second, (literal, *on_resource))
                    self._ordered(second, first, (literal.negated(), *on_resource))

    def _ordered(self, first: int, second: int, literals: _Literals) -> None:
        """Make ``first`` finish before ``second`` starts when all literals hold."""
        condition: list[cp_model.BoolVarT] = list(literals)
        self.model.add(self.ends[first] <= self.starts[second]).only_enforce_if(condition)

    # ------------------------------------------------------------ objectives

    def makespan(self) -> cp_model.IntVar:
        """Return a variable equal to the latest end."""
        if self._makespan is None:
            self._makespan = self.model.new_int_var(0, self.horizon, "makespan")
            self.model.add_max_equality(self._makespan, self.ends)
        return self._makespan

    def objective(self, name: str) -> ModelObjective:
        """Return the integer form of a built-in objective.

        Raises:
            UnsupportedProblemError: If the objective is not modelled, or its
                weights or coefficients have too many decimals.
        """
        match name:
            case "makespan":
                return ModelObjective(self.makespan(), 0, self.scale)
            case "total_completion_time":
                return self._completion(flow=False)
            case "total_flow_time":
                return self._completion(flow=True)
            case "mean_response_time":
                return self._response()
            case "total_tardiness":
                return self._tardiness()
            case "max_lateness":
                return self._lateness()
            case "tardy_tasks":
                return self._tardy()
            case "energy":
                return self._energy()
            case "cost":
                return self._cost()
        raise UnsupportedProblemError(f"the interval model does not state objective {name!r}")

    def _job_completions(self) -> list[tuple[float, int, cp_model.IntVar]]:
        """Return ``(weight, scaled release, completion)`` of every job."""
        result: list[tuple[float, int, cp_model.IntVar]] = []
        for weight, release, members in jobs(self.problem):
            if len(members) == 1:
                completion = self.ends[members[0]]
            else:
                completion = self.model.new_int_var(0, self.horizon, f"job_{len(result)}")
                self.model.add_max_equality(completion, [self.ends[i] for i in members])
            result.append((weight, self._time(release), completion))
        return result

    def _completion(self, *, flow: bool) -> ModelObjective:
        completions = self._job_completions()
        weights = [weight for weight, _, _ in completions]
        weight_scale = decimal_scale(weights, "job weights")
        coefficients = [scaled(weight, weight_scale) for weight in weights]
        expression = _total([completion for _, _, completion in completions], coefficients)
        offset = 0
        if flow:
            offset = -sum(
                coefficient * release
                for coefficient, (_, release, _) in zip(coefficients, completions, strict=True)
            )
        return ModelObjective(expression, offset, self.scale * weight_scale)

    def _response(self) -> ModelObjective:
        completions = self._job_completions()
        expression = _total([completion for _, _, completion in completions])
        offset = -sum(release for _, release, _ in completions)
        return ModelObjective(expression, offset, self.scale * len(completions))

    def _tardiness(self) -> ModelObjective:
        terms: list[cp_model.IntVar] = []
        weights: list[float] = []
        for i, task in enumerate(self.problem.tasks):
            if task.due is None:
                continue
            tardiness = self.model.new_int_var(0, self.horizon, f"tardiness_{i}")
            self.model.add_max_equality(tardiness, [self.ends[i] - self._time(task.due), 0])
            terms.append(tardiness)
            weights.append(task.weight)
        if not terms:
            return ModelObjective(0, 0, 1)
        weight_scale = decimal_scale(weights, "task weights")
        coefficients = [scaled(weight, weight_scale) for weight in weights]
        return ModelObjective(_total(terms, coefficients), 0, self.scale * weight_scale)

    def _lateness(self) -> ModelObjective:
        dues = [
            (i, self._time(task.due))
            for i, task in enumerate(self.problem.tasks)
            if task.due is not None
        ]
        if not dues:
            return ModelObjective(0, 0, 1)
        lateness = self.model.new_int_var(-max(due for _, due in dues), self.horizon, "lateness")
        self.model.add_max_equality(lateness, [self.ends[i] - due for i, due in dues])
        return ModelObjective(lateness, 0, self.scale)

    def _tardy(self) -> ModelObjective:
        literals: list[cp_model.IntVar] = []
        for i, task in enumerate(self.problem.tasks):
            if task.due is None:
                continue
            due = self._time(task.due)
            late = self.model.new_bool_var(f"late_{i}")
            self.model.add(self.ends[i] >= due + 1).only_enforce_if(late)
            self.model.add(self.ends[i] <= due).only_enforce_if(late.negated())
            literals.append(late)
        if not literals:
            return ModelObjective(0, 0, 1)
        return ModelObjective(_total(literals), 0, 1)

    def _energy(self) -> ModelObjective:
        idle = sum(resource.idle_power for resource in self.problem.resources)
        busy = self._busy(lambda resource: resource.busy_power - resource.idle_power)
        network = self._network(energy=True)
        values = [idle] + [amount * self.scale for _, _, amount in busy]
        values += [amount * self.scale for *_, amount in network]
        coefficient_scale = decimal_scale(values, "energy coefficients")
        expression = self._mode_costs(busy, network, self.scale * coefficient_scale)
        if idle > 0.0:
            expression += scaled(idle, coefficient_scale) * self.makespan()
        return ModelObjective(expression, 0, self.scale * coefficient_scale)

    def _cost(self) -> ModelObjective:
        busy = self._busy(lambda resource: resource.cost_rate)
        network = self._network(energy=False)
        values = [amount for _, _, amount in busy] + [amount for *_, amount in network]
        coefficient_scale = decimal_scale(values, "cost coefficients")
        expression = self._mode_costs(busy, network, coefficient_scale)
        return ModelObjective(expression, 0, coefficient_scale)

    def _busy(self, rate: Callable[[Resource], float]) -> list[tuple[int, int, float]]:
        """Return ``(task, mode, amount)`` of every mode with a non-zero busy amount.

        The amount is the rate times the utilised fraction of each demanded
        resource times the duration, as the energy and cost objectives define.
        """
        result: list[tuple[int, int, float]] = []
        for i, task in enumerate(self.problem.tasks):
            for m, mode in enumerate(task.modes):
                amount = 0.0
                for resource_id, units in mode.demands.items():
                    resource = self.problem.resource(resource_id)
                    amount += rate(resource) * units / resource.capacity * mode.duration
                if amount != 0.0:
                    result.append((i, m, amount))
        return result

    def _network(self, *, energy: bool) -> list[tuple[int, int, int, int, float]]:
        """Return ``(before, mode, after, mode, amount)`` of every priced transfer."""
        problem = self.problem
        network = problem.network
        result: list[tuple[int, int, int, int, float]] = []
        if network is None:
            return result
        index = problem.task_index
        for dependency in problem.dependencies:
            if dependency.data <= 0.0:
                continue
            a, b = index[dependency.before], index[dependency.after]
            for ma, source in enumerate(problem.tasks[a].modes):
                for mb, target in enumerate(problem.tasks[b].modes):
                    if source.location is None or target.location is None:
                        continue
                    if source.location == target.location:
                        continue
                    link = network.link(
                        problem.resource(source.location).endpoint,
                        problem.resource(target.location).endpoint,
                    )
                    if link is None:
                        continue
                    rate = link.energy_per_unit if energy else link.cost_per_unit
                    if rate > 0.0:
                        result.append((a, ma, b, mb, rate * dependency.data))
        return result

    def _mode_costs(
        self,
        busy: list[tuple[int, int, float]],
        network: list[tuple[int, int, int, int, float]],
        factor: int,
    ) -> cp_model.LinearExpr:
        """Return the sum of mode-dependent amounts, each multiplied by ``factor``.

        A transfer's amount depends on the modes of two tasks, so it is carried
        by a literal equal to the conjunction of the two mode literals.
        """
        literals: list[cp_model.IntVar] = []
        coefficients: list[int] = []
        for i, m, amount in busy:
            literals.append(self.chosen[i][m])
            coefficients.append(scaled(amount, factor))
        for a, ma, b, mb, amount in network:
            source, target = self.chosen[a][ma], self.chosen[b][mb]
            both = self.model.new_bool_var(f"pair_{a}_{ma}_{b}_{mb}")
            self.model.add_bool_and([source, target]).only_enforce_if(both)
            self.model.add_bool_or([source.negated(), target.negated(), both])
            literals.append(both)
            coefficients.append(scaled(amount, factor))
        return _total(literals, coefficients)

    # --------------------------------------------------------------- reading

    def schedule(self, solver: cp_model.CpSolver) -> Schedule:
        """Return the schedule of the solver's best solution, in problem time."""
        assignments = []
        for i, task in enumerate(self.problem.tasks):
            mode = next(m for m, lit in enumerate(self.chosen[i]) if solver.boolean_value(lit))
            start = solver.value(self.starts[i]) / self.scale
            end = solver.value(self.ends[i]) / self.scale
            assignments.append(Assignment(task.id, mode, start, end))
        return Schedule.of(assignments)


class _Monitor(cp_model.CpSolverSolutionCallback):  # type: ignore[misc, unused-ignore]
    """Counts CP-SAT solutions, records the trace and the bounds, and applies the stop criteria.

    Each improving solution CP-SAT reports is one evaluation and one iteration. The monitor
    also records, in the problem's units, each improvement of the best bound CP-SAT has
    proven: the bound it holds with every solution, and, where OR-Tools has a bound
    callback (see :func:`_report_bounds`), every improvement between solutions.
    """

    def __init__(self, objective: ModelObjective, stop: StopCriteria, clock: Clock) -> None:
        super().__init__()
        self.objective = objective
        self.stop = stop
        self.clock = clock
        self.solutions = 0
        self.trace: list[TracePoint] = []
        self.bounds: list[BoundPoint] = []
        self.reason: StopReason | None = None
        # CP-SAT may call the bound callback from another thread than the solution
        # callback, so the comparison with the last bound and the append are one step.
        self._bounds_lock = threading.Lock()

    def on_solution_callback(self) -> None:
        """Record a new solution and its bound, and stop the search if a criterion is met."""
        self.solutions += 1
        value = self.objective.original(self.objective_value)
        self.trace.append(TracePoint(self.clock.elapsed, self.solutions, self.solutions, value))
        self.on_bound(self.best_objective_bound)
        stop = self.stop
        if stop.target is not None and value <= stop.target:
            self.reason = StopReason.TARGET_REACHED
        elif stop.max_evaluations is not None and self.solutions >= stop.max_evaluations:
            self.reason = StopReason.EVALUATION_LIMIT
        elif stop.max_iterations is not None and self.solutions >= stop.max_iterations:
            self.reason = StopReason.ITERATION_LIMIT
        if self.reason is not None:
            self.stop_search()

    def on_bound(self, bound: float) -> None:
        """Record ``bound``, CP-SAT's best bound on the model's objective, if it improves.

        A bound that is not finite is CP-SAT's lack of one and is skipped. A bound that does not
        exceed the last one recorded is skipped too, so the recorded values only rise.
        """
        if not math.isfinite(bound):
            return
        value = self.objective.original(bound)
        with self._bounds_lock:
            if not self.bounds or value > self.bounds[-1].value:
                self.bounds.append(BoundPoint(self.clock.elapsed, value))


def _report_bounds(solver: cp_model.CpSolver, monitor: _Monitor) -> bool:
    """Have ``solver`` pass every improvement of its bound to ``monitor``, if it can.

    OR-Tools builds that offer a bound callback expose it as the attribute
    ``best_bound_callback`` of the solver. The attribute is tested, never the version of the
    package, so a build is used as it is.

    Returns:
        Whether the callback is set. Without it, the monitor still holds the bound of every
        solution.
    """
    if not hasattr(solver, "best_bound_callback"):
        return False
    solver.best_bound_callback = monitor.on_bound
    return True


def _ending_at(
    points: Sequence[BoundPoint], final: float | None, elapsed: float
) -> tuple[BoundPoint, ...]:
    """Return the recorded bounds as a series that ends at ``final``, the bound the run proved.

    CP-SAT may improve its bound after its last callback, so a point at ``elapsed`` closes the
    series when the last recorded bound is below ``final``. A recorded bound above ``final``
    cannot hold and is dropped. Without a final bound there is no series.
    """
    if final is None:
        return ()
    bounds = [point for point in points if point.value <= final]
    if not bounds or bounds[-1].value < final:
        bounds.append(BoundPoint(elapsed, final))
    return tuple(bounds)


def _unlocated_transfers(problem: Problem) -> int:
    """Count the dependencies that carry data across a network but have a mode without a location.

    The data model prices a transfer only between two located modes, so such a pair
    exchanges its data at no cost and no delay. The count lets a caller see that the
    model treats those pairs as co-located.
    """
    if problem.network is None:
        return 0
    index = problem.task_index
    count = 0
    for dependency in problem.dependencies:
        if dependency.data <= 0.0:
            continue
        pair = (problem.tasks[index[dependency.before]], problem.tasks[index[dependency.after]])
        if any(mode.location is None for task in pair for mode in task.modes):
            count += 1
    return count


def solve(
    problem: Problem,
    stop: StopCriteria,
    info: SolverInfo,
    objective_name: str,
    clock: Clock,
    *,
    threads: int,
    relative_gap: float,
    log: bool,
) -> SolveResult:
    """Build the interval model of ``problem``, solve it, and return the checked result.

    Args:
        problem: The problem.
        stop: Stop criteria. Evaluation and iteration limits count solutions.
        info: Metadata of the calling solver.
        objective_name: The primary objective, already checked as supported.
        clock: The run's clock, started before the model is built.
        threads: CP-SAT workers.
        relative_gap: Relative gap at which CP-SAT may stop.
        log: Send CP-SAT's search log to the library logger.

    Returns:
        The result, with the solver's own schedule.

    Raises:
        UnsupportedProblemError: If the problem cannot be stated exactly.
        SolverError: If CP-SAT rejects the model, or its solution contradicts
            the problem.
    """
    dues = objective_name in _DUE_OBJECTIVES
    scale = decimal_scale(time_values(problem, dues=dues), "time values of the problem")
    horizon = serial_horizon(problem, scale)
    model = IntervalModel(problem, scale, horizon)
    objective = model.objective(objective_name)
    model.model.minimize(objective.expression)
    extras: dict[str, Scalar] = {
        "backend": "ortools-cp-sat",
        "backend_version": backend_version("ortools"),
        "threads": threads,
        "relative_gap": relative_gap,
        "time_scale": scale,
        "horizon": horizon / scale,
        "transfer_pairs": model.transfer_pairs,
        "unlocated_transfers": _unlocated_transfers(problem),
        "model_variables": len(model.model.proto.variables),
        "model_constraints": len(model.model.proto.constraints),
    }
    remaining = clock.remaining()
    if remaining is not None and remaining <= 0.0:
        extras["raw_status"] = "NOT_STARTED"
        return exact_result(
            info,
            problem,
            objective_name,
            status=Status.UNKNOWN,
            schedule=None,
            stop_reason=StopReason.TIME_LIMIT,
            elapsed=clock.elapsed,
            extras=extras,
        )
    solver = cp_model.CpSolver()
    parameters = solver.parameters
    parameters.num_workers = threads
    parameters.relative_gap_limit = relative_gap
    if remaining is not None:
        parameters.max_time_in_seconds = remaining
    if log:
        parameters.log_search_progress = True
        parameters.log_to_stdout = False
        solver.log_callback = _LOG.info
    monitor = _Monitor(objective, stop, clock)
    extras["bound_callback"] = _report_bounds(solver, monitor)
    status = solver.solve(model.model, monitor)
    ended = clock.elapsed
    extras["raw_status"] = solver.status_name(status)
    extras["conflicts"] = solver.num_conflicts
    extras["branches"] = solver.num_branches
    extras["backend_wall_time"] = solver.wall_time
    if status == cp_model.MODEL_INVALID:
        raise SolverError(f"CP-SAT rejected the model: {model.model.validate()}")
    solutions = monitor.solutions
    limited = StopReason.TIME_LIMIT if stop.time_limit is not None else StopReason.COMPLETED
    if status == cp_model.INFEASIBLE:
        return exact_result(
            info,
            problem,
            objective_name,
            status=Status.INFEASIBLE,
            schedule=None,
            stop_reason=StopReason.COMPLETED,
            elapsed=clock.elapsed,
            evaluations=solutions,
            iterations=solutions,
            extras=extras,
        )
    bound = solver.best_objective_bound
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        reached = objective.original(bound) if math.isfinite(bound) else None
        return exact_result(
            info,
            problem,
            objective_name,
            status=Status.UNKNOWN,
            schedule=None,
            stop_reason=monitor.reason or limited,
            elapsed=clock.elapsed,
            bound=reached,
            evaluations=solutions,
            iterations=solutions,
            bounds=_ending_at(monitor.bounds, reached, ended),
            extras=extras,
        )
    schedule = model.schedule(solver)
    value = objective.original(solver.objective_value)
    confirm_value(info.id, value, evaluate(problem, schedule, [objective_name])[objective_name])
    if status == cp_model.OPTIMAL and bound >= solver.objective_value - 0.5:
        # The objective is integral, so a bound within half a unit proves it.
        outcome, reason, proven = Status.OPTIMAL, StopReason.COMPLETED, value
    else:
        # Stopped by a limit, or by the relative gap tolerance before a proof.
        finished = StopReason.COMPLETED if status == cp_model.OPTIMAL else limited
        outcome, reason = Status.FEASIBLE, monitor.reason or finished
        proven = objective.original(bound)
    return exact_result(
        info,
        problem,
        objective_name,
        status=outcome,
        schedule=schedule,
        stop_reason=reason,
        elapsed=clock.elapsed,
        bound=proven,
        evaluations=solutions,
        iterations=solutions,
        trace=monitor.trace,
        bounds=_ending_at(monitor.bounds, proven, ended),
        extras=extras,
    )
