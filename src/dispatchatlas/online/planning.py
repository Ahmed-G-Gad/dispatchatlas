"""The planner protocol: what a planner receives at a decision point, and returns.

At every decision point the runner calls :meth:`Planner.plan` with a
:class:`PlanningContext`. The context holds the state of the run, the tasks
to plan, and the budget of the pass, and evaluates plans with the runner's
own schedule builder, counting every placement. A planner returns a
:class:`Plan`: an order of the tasks to plan and a mode (a node) for each,
or ``None`` to plan nothing and leave every task to the dispatcher.

**Charging.** A :class:`PlanningClock` turns a pass into simulated time. In
``counted`` mode a pass takes ``placements / rate`` seconds, where ``rate``
is the placements per second the caller states, so runs are deterministic
and reproduce on any machine. In ``measured`` mode it takes the wall time of
the call to :meth:`Planner.plan`, for runs on the machine whose speed the
results describe. In both modes a dispatcher's answer costs
``placements / rate``.

**Budget.** A :class:`Budget` caps a pass by placements, by charged seconds,
or both. The context refuses an evaluation that would exceed the placement
cap, or, in ``counted`` mode, the seconds, by raising
:class:`BudgetExhausted`; a planner checks :attr:`PlanningContext.remaining`
first. A pass that lets the exception escape, or whose measured time
exceeds the seconds, **overruns**: it loses its window, nothing it computed
is adopted, and the scheduler is charged the whole budget.

**Estimates.** The objective estimate of a plan is the mean, over the
workflows that have arrived, of the predicted response time: the latest
predicted finish of the workflow's tasks minus its arrival. A finished
workflow counts with its realised response. A workflow with a task that no
node that is up can run has no prediction and is left out of the mean, which
leaves the same workflows out of every plan of one pass.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Dependency, Mode, Problem, Scalar, Task
from dispatchatlas.online._engine import (
    Model,
    dispatch_rollout,
    place_in_order,
    predicted_finishes,
)
from dispatchatlas.online.state import SystemState, TaskStatus

if TYPE_CHECKING:
    from dispatchatlas.solvers.scheduling import Builder

__all__ = [
    "Budget",
    "BudgetExhausted",
    "ClockMode",
    "Estimate",
    "PassOutcome",
    "Plan",
    "Planner",
    "PlanningClock",
    "PlanningContext",
    "Residual",
]


class BudgetExhausted(Exception):  # noqa: N818 - it signals an end, not an error
    """Raised by a :class:`PlanningContext` asked to exceed the budget of its pass."""


class PassOutcome(StrEnum):
    """What became of a planning pass.

    - ``adopted``: its plan took effect when its charged time had elapsed.
    - ``overtaken``: an event came first; it adopted nothing.
    - ``overrun``: it exceeded its budget; it adopted nothing.
    - ``declined``: the planner returned no plan.
    """

    ADOPTED = "adopted"
    OVERTAKEN = "overtaken"
    OVERRUN = "overrun"
    DECLINED = "declined"


type ClockMode = Literal["counted", "measured"]


@dataclass(frozen=True)
class PlanningClock:
    """How planning time is charged in simulated time.

    Attributes:
        mode: ``"counted"`` charges a pass ``placements / rate``;
            ``"measured"`` charges its wall time.
        rate: Placements per second. Infinite makes counted planning, and
            every dispatcher's answer, free.
    """

    mode: ClockMode = "counted"
    rate: float = 1.0e5

    def __post_init__(self) -> None:
        if self.mode not in ("counted", "measured"):
            raise ModelError(f"clock mode must be 'counted' or 'measured', got {self.mode!r}")
        if isinstance(self.rate, bool) or not isinstance(self.rate, (int, float)):
            raise ModelError(f"clock rate must be a number, got {self.rate!r}")
        if math.isnan(self.rate) or self.rate <= 0.0:
            raise ModelError(f"clock rate must be positive, got {self.rate!r}")

    def counted(self, placements: int) -> float:
        """Return the charged time of ``placements`` placements."""
        return 0.0 if math.isinf(self.rate) else placements / self.rate

    def to_dict(self) -> dict[str, Scalar]:
        """Return the clock as a record entry; an infinite rate is written as ``"inf"``."""
        return {"mode": self.mode, "rate": "inf" if math.isinf(self.rate) else self.rate}


@dataclass(frozen=True)
class Budget:
    """The budget of one planning pass. ``None`` leaves a dimension unbounded.

    Attributes:
        placements: Most placements the pass may make.
        seconds: Most charged seconds the pass may take.
    """

    placements: int | None = None
    seconds: float | None = None

    def __post_init__(self) -> None:
        if self.placements is not None and (
            isinstance(self.placements, bool)
            or not isinstance(self.placements, int)
            or self.placements < 0
        ):
            raise ModelError(
                f"budget placements must be a non-negative integer, got {self.placements!r}"
            )
        if self.seconds is not None and not (
            isinstance(self.seconds, (int, float))
            and math.isfinite(self.seconds)
            and self.seconds >= 0
        ):
            raise ModelError(f"budget seconds must be a non-negative number, got {self.seconds!r}")

    def to_dict(self) -> dict[str, Scalar | None]:
        """Return the budget as a record entry."""
        return {"placements": self.placements, "seconds": self.seconds}


@dataclass(frozen=True)
class Plan:
    """A decision for every task to plan: an order, and a mode for each.

    Attributes:
        order: Task indices in priority order. The runner places them by the
            serial scheme in this order, each task after its predecessors.
        modes: Mode index of each task, aligned with ``order``. A negative
            entry lets the runner take the allowed mode that finishes the task
            earliest when it is placed.
    """

    order: tuple[int, ...]
    modes: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", tuple(self.order))
        object.__setattr__(self, "modes", tuple(self.modes))
        if len(self.order) != len(self.modes):
            raise ModelError("a plan needs one mode entry per task in its order")

    def mode_of(self) -> dict[int, int]:
        """Return the mode of each task, keyed by task index."""
        return dict(zip(self.order, self.modes, strict=True))


@dataclass(frozen=True)
class Estimate:
    """The runner's prediction of a plan's outcome.

    Attributes:
        value: Mean predicted response time over the predictable arrived
            workflows, ``None`` when none is predictable.
        finishes: Predicted finish of each unfinished arrived workflow, by
            workflow index; ``None`` for one with an unplaceable task.
    """

    value: float | None
    finishes: Mapping[int, float | None] = field(default_factory=dict)

    def better_than(self, other: Estimate | None) -> bool:
        """Whether this estimate is strictly lower; any value beats ``None``."""
        if self.value is None:
            return False
        return other is None or other.value is None or self.value < other.value - 1e-9


@dataclass(frozen=True)
class Residual:
    """The problem that remains at a decision point, for a static solver.

    The problem holds the tasks to plan, with only their allowed modes and
    their release raised to the decision time, and, as **frozen** tasks, every
    running task and every finished predecessor of a task to plan. A running
    task becomes a single-mode task on its node for its remaining time,
    released at the decision time; a finished predecessor becomes a
    zero-duration task that holds nothing, located on its node and released at
    its finish, so its dependencies keep their lags and transfers. A solver
    that models hard deadlines also gets each running task's finish as its
    deadline, so a schedule that moves it is infeasible.

    Attributes:
        problem: The residual problem.
        tasks: Index, in the run's problem, of each residual task.
        modes: Per residual task, the run's mode index of each residual mode.
        planned: Residual indices of the tasks to plan.
    """

    problem: Problem
    tasks: tuple[int, ...]
    modes: tuple[tuple[int, ...], ...]
    planned: frozenset[int]

    def plan_from(self, starts: Mapping[str, tuple[float, int]]) -> Plan:
        """Return the plan that lists the tasks to plan by start, with their modes.

        Args:
            starts: Residual task id to ``(start, residual mode)``, for example
                from a solver's schedule.
        """
        rows = []
        for r in sorted(self.planned):
            task = self.problem.tasks[r]
            start, mode = starts[task.id]
            rows.append((start, r, self.tasks[r], self.modes[r][mode]))
        rows.sort()
        return Plan(tuple(row[2] for row in rows), tuple(row[3] for row in rows))


class Planner(Protocol):
    """Decides the tasks of a run at each decision point."""

    @property
    def name(self) -> str:
        """Short identifier of the planner, for records."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return the planner's name and parameters, for records."""
        ...

    def plan(self, context: PlanningContext) -> Plan | None:
        """Return a plan of ``context.tasks``, or ``None`` to plan nothing."""
        ...


class PlanningContext:
    """What a planner sees and may do during one pass.

    The runner creates one per pass; a planner never builds one.

    Attributes:
        index: Number of the pass in the run, from 0.
        now: Time of the decision point.
        state: The state of the run at ``now``.
        tasks: Indices of the tasks to plan: every uncommitted task of an
            arrived workflow that some mode on nodes that are up can run,
            and whose uncommitted predecessors are all to be planned too.
        budget: The budget of the pass.
        clock: The planning clock of the run.
        previous: The plan in force before the decision point, cut to
            ``tasks``, with a negative mode for a task it did not hold or held
            on a node now down, or ``None`` if no plan was in force.
        rng: Random generator of the pass, seeded from the run's seed and the
            pass index.
        seed: Seed of the pass, for a solver the planner runs.
        anytime: ``(placements, value)`` after each evaluation that improved
            the best estimate of the pass.
    """

    def __init__(
        self,
        model: Model,
        base: Builder,
        state: SystemState,
        tasks: Sequence[int],
        budget: Budget,
        clock: PlanningClock,
        previous: Plan | None,
        index: int,
        seed: int,
    ) -> None:
        self._model = model
        self._base = base
        self.index = index
        self.now = state.now
        self.state = state
        self.tasks = tuple(tasks)
        self.budget = budget
        self.clock = clock
        self.previous = previous
        self.seed = seed
        self.rng = random.Random(seed)
        self.anytime: list[tuple[int, float]] = []
        self._placements = 0
        self._started = time.perf_counter()
        self._task_set = frozenset(self.tasks)
        self._allowed = {task: model.allowed(task, state.down) for task in self.tasks}
        known = [model.workflow_index[name] for name in state.known]
        self._open = [
            w
            for w in known
            if any(state.status[t] is not TaskStatus.FINISHED for t in model.members[w])
        ]
        open_set = set(self._open)
        closed = [w for w in known if w not in open_set]
        self._closed_total = sum(
            max(base.ends[t] for t in model.members[w]) - model.arrivals[w] for w in closed
        )
        self._closed_count = len(closed)
        self._best: float | None = None

    @property
    def problem(self) -> Problem:
        """The run's problem: every task of every workflow of the trace."""
        return self._model.problem

    @property
    def placements(self) -> int:
        """Placements made in this pass so far."""
        return self._placements

    @property
    def remaining(self) -> int | None:
        """Placements the pass may still make, ``None`` when unbounded.

        In ``counted`` mode a seconds budget also caps the placements, at
        ``seconds * rate``.
        """
        caps: list[int] = []
        if self.budget.placements is not None:
            caps.append(self.budget.placements)
        if (
            self.clock.mode == "counted"
            and self.budget.seconds is not None
            and not math.isinf(self.clock.rate)
        ):
            caps.append(math.floor(self.budget.seconds * self.clock.rate + 1e-9))
        return None if not caps else max(0, min(caps) - self._placements)

    def remaining_seconds(self) -> float | None:
        """Charged seconds left in the budget, ``None`` when unbounded."""
        if self.budget.seconds is None:
            return None
        if self.clock.mode == "counted":
            used = self.clock.counted(self._placements)
        else:
            used = time.perf_counter() - self._started
        return max(0.0, self.budget.seconds - used)

    def allowed(self, task: int) -> tuple[int, ...]:
        """Modes of ``task`` whose resources are all up."""
        allowed = self._allowed.get(task)
        if allowed is None:
            raise ModelError(f"task {task} is not a task to plan")
        return allowed

    def charge(self, placements: int) -> None:
        """Count ``placements`` made outside the context, such as by a static solver.

        Raises:
            BudgetExhausted: If they exceed what remains.
        """
        if placements < 0:
            raise ModelError("placements must be non-negative")
        remaining = self.remaining
        if remaining is not None and placements > remaining:
            self._placements += remaining
            raise BudgetExhausted(f"{placements} placements exceed the {remaining} left")
        self._placements += placements

    def cost(self, plan: Plan) -> int:
        """Placements an evaluation of ``plan`` makes: 1 per fixed mode, else 1 per allowed mode."""
        return sum(
            1 if m >= 0 else len(self.allowed(t))
            for t, m in zip(plan.order, plan.modes, strict=True)
        )

    @property
    def seed_cost(self) -> int:
        """Placements :meth:`seed` makes: one per allowed mode of every task to plan."""
        return sum(len(modes) for modes in self._allowed.values())

    def seed_plan(self) -> tuple[Plan, Estimate]:
        """Rank every task to plan by the earliest-finish dispatch rule, and evaluate it.

        This is the greedy plan: tasks in the order they fall due, ties by the
        run's priority, each on the allowed mode that finishes it earliest
        given the tasks before it. Costs :attr:`seed_cost` placements.

        Raises:
            BudgetExhausted: If the budget cannot pay for it.
        """
        self.charge(self.seed_cost)
        builder = self._base.copy()
        placed = dispatch_rollout(
            self._model, builder, self.tasks, self.state.ready_since, self.allowed, self.now
        )
        plan = Plan(tuple(t for t, _ in placed), tuple(m for _, m in placed))
        return plan, self._estimate(builder)

    def evaluate(self, plan: Plan) -> Estimate:
        """Build ``plan`` on the committed schedule and estimate its outcome.

        Raises:
            BudgetExhausted: If the budget cannot pay :meth:`cost`.
            ModelError: If the plan is not an order of :attr:`tasks` with
                allowed modes.
        """
        self.check(plan)
        self.charge(self.cost(plan))
        return self.estimate_free(plan)

    def estimate_free(self, plan: Plan) -> Estimate:
        """Evaluate ``plan`` without charging it. The runner's own bookkeeping uses this."""
        builder = self._base.copy()
        place_in_order(self._model, builder, plan.order, plan.modes, self.allowed, self.now)
        return self._estimate(builder, record=False)

    def check(self, plan: Plan) -> None:
        """Raise :class:`ModelError` unless ``plan`` orders :attr:`tasks` with allowed modes."""
        if len(plan.order) != len(self.tasks) or set(plan.order) != self._task_set:
            raise ModelError(f"pass {self.index}: a plan must order exactly the tasks to plan")
        for task, mode in zip(plan.order, plan.modes, strict=True):
            if mode >= 0 and mode not in self._allowed[task]:
                raise ModelError(f"pass {self.index}: mode {mode} of task {task} is not allowed")

    def _estimate(self, builder: Builder, *, record: bool = True) -> Estimate:
        model = self._model
        finishes = predicted_finishes(model, builder, self._open)
        total, count = self._closed_total, self._closed_count
        for workflow, finish in finishes.items():
            if finish is not None:
                total += finish - model.arrivals[workflow]
                count += 1
        value = total / count if count else None
        if record and value is not None and (self._best is None or value < self._best):
            self._best = value
            self.anytime.append((self._placements, value))
        return Estimate(value, finishes)

    def residual(self, *, deadlines: bool = True) -> Residual:
        """Return the residual problem of this decision point, uncharged.

        Args:
            deadlines: Give each frozen running task its finish as a hard
                deadline. Pass ``False`` for a solver that does not model
                deadlines.
        """
        model = self._model
        problem = model.problem
        state = self.state
        planned = set(self.tasks)
        frozen: set[int] = {
            t for t, a in state.committed.items() if state.status[t] is TaskStatus.RUNNING
        }
        for task in self.tasks:
            for before in model.predecessors[task]:
                if before not in planned:
                    frozen.add(before)
        members = sorted(frozen | planned, key=lambda t: (t not in frozen, t))
        tasks: list[Task] = []
        mode_maps: list[tuple[int, ...]] = []
        for t in members:
            original = problem.tasks[t]
            if t in planned:
                allowed = self._allowed[t]
                tasks.append(
                    Task(
                        original.id,
                        tuple(original.modes[m] for m in allowed),
                        release=max(original.release, self.now),
                        due=original.due,
                        deadline=original.deadline,
                        weight=original.weight,
                        group=original.group,
                    )
                )
                mode_maps.append(allowed)
                continue
            attempt = state.committed[t]
            mode = original.modes[attempt.mode]
            if state.status[t] is TaskStatus.RUNNING:
                frozen_mode = Mode(attempt.end - self.now, mode.demands, mode.location)
                release, deadline = self.now, (attempt.end if deadlines else None)
            else:
                frozen_mode = Mode(0.0, {}, mode.location)
                release, deadline = attempt.end, None
            tasks.append(
                Task(
                    original.id,
                    (frozen_mode,),
                    release=release,
                    deadline=deadline,
                    weight=original.weight,
                    group=original.group,
                )
            )
            mode_maps.append((attempt.mode,))
        included = {problem.tasks[t].id for t in members}
        planned_ids = {problem.tasks[t].id for t in planned}
        dependencies = tuple(
            Dependency(d.before, d.after, d.lag, d.data)
            for d in problem.dependencies
            if d.after in planned_ids and d.before in included
        )
        residual = Problem(
            id=f"{problem.id}-pass-{self.index}",
            tasks=tuple(tasks),
            resources=problem.resources,
            dependencies=dependencies,
            objectives=("mean_response_time",),
            network=problem.network,
        )
        planned_residual = frozenset(n for n, t in enumerate(members) if t in planned)
        return Residual(residual, tuple(members), tuple(mode_maps), planned_residual)
