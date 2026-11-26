"""The online scheduling loop: a discrete-event run of a planner over a trace.

:func:`run_online` plays a :class:`~dispatchatlas.online.events.Trace` on a
platform (any object with ``resources`` and ``network``, such as a
:class:`~dispatchatlas.benchmarks.continuum.Platform` or a
:class:`~dispatchatlas.core.model.Problem`) and returns an
:class:`OnlineResult` that keeps every raw record of the run.

**Commitment.** A task commits when it starts. Its node, start, and finish
are then a reservation that nothing moves; only a failure of its node ends it
early. Everything not started is uncommitted and may be planned again.

**The scheduler.** One thread of computation does two kinds of work, and its
time is charged in simulated time (see
:mod:`~dispatchatlas.online.planning`):

- A **pass** calls the planner at a decision point. Every event, or every
  set of events at one instant, is a decision point, and a pass begins at it.
  Its plan takes effect, is **adopted**, only once its charged time has
  elapsed. Then the runner places the plan's tasks by the serial scheme, in
  the plan's order and modes, on the committed schedule and never before the
  adoption time, and starts each task at its place.
- An **answer** decides one task with the dispatch rule: it ranks the task's
  nodes and reserves the node that finishes it earliest, given the committed
  tasks and the earlier answers; the task starts there unless a plan is
  adopted first. An answer costs one placement per allowed mode, a mode of
  the task whose resources are all up when the answer begins, charged at the
  clock's rate, and ranks those of them still up when it ends.

**Rules of the loop**, each a decision of this runner:

1. *Freshness.* An event withdraws the plan in force: no task starts on a
   plan made by a pass that began before the last event. Answers already
   made stay, except those on a node that is down.
2. *Answering comes first.* While no fresh plan is in force, every ready
   task is answered when it falls due (it became ready, its workflow arrived,
   or its attempt was lost), in the order of due time, then the run's
   priority, then task index. Answers preempt the pass in progress, which
   finishes that much later. Without a planner every task is decided this
   way: the dispatcher alone.
3. *Every uncommitted task is ranked again at an event.* A pass plans every
   uncommitted task that some node that is up can run.
4. *Overtaking.* An event that comes before a pass is adopted, or at the
   same instant, overtakes it: the pass adopts nothing, and a new pass begins
   at the event.
5. *Overrun.* A pass that exceeds its budget (see
   :class:`~dispatchatlas.online.planning.Budget`) loses its window: it
   adopts nothing and its charge is recorded as the budget's seconds; answers
   continue as they do for every pass, so the charge does not delay them, and
   the dispatcher decides until the next event.
6. *Failures.* A failure ends every attempt running on the node, at the
   failure time; the attempt is lost and its task is ready again, to restart
   from scratch. The node takes no task until its repair. Finished work is
   never lost.
7. *Ties at one instant*: tasks finishing then finish first, then the
   events apply (repairs, failures, arrivals), then tasks due then start, and
   then the scheduler acts. A task finishing at a failure's instant has
   finished; one due to start then on the failing node does not start.

**Forecasts.** At each decision point, before and after its events, and at
each adoption, the runner records a forecast, uncharged: the schedule the
run would execute if nothing else happened and no pass were adopted, built
from the committed tasks, the plan or answers in force, and the dispatch rule
for every other task. Recovery after an event is measured on these
forecasts, see :mod:`~dispatchatlas.online.metrics`.

**The greedy reference.** A planner that seeds with the greedy plan and
never searches commits every task where
:class:`~dispatchatlas.online.planners.OnlineEft` does, at the same time,
under any clock. On a trace of arrivals with planning free (an infinite
clock rate), both commit every task where the dispatcher alone would. Under
failures and repairs the two references differ by rule 3: online EFT ranks
every uncommitted task again at each event, while the dispatcher keeps its
answers and decides again only the tasks a failure displaces.
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import dispatchatlas
from dispatchatlas.core.errors import ModelError, SerializationError
from dispatchatlas.core.model import Dependency, Network, Problem, Resource, Task
from dispatchatlas.core.seeds import derive_seed
from dispatchatlas.core.serialization import problem_digest
from dispatchatlas.online._engine import (
    TIME_TOLERANCE,
    Model,
    dispatch_rollout,
    earliest_finish,
    place_in_order,
    predicted_finishes,
)
from dispatchatlas.online.events import Arrival, Event, Failure, Repair, Trace
from dispatchatlas.online.metrics import recovery
from dispatchatlas.online.planning import (
    Budget,
    BudgetExhausted,
    PassOutcome,
    Plan,
    Planner,
    PlanningClock,
    PlanningContext,
)
from dispatchatlas.online.state import Attempt, AttemptOutcome, SystemState, TaskStatus
from dispatchatlas.solvers.priorities import upward_ranks
from dispatchatlas.solvers.scheduling import Builder

__all__ = ["RESULT_FORMAT", "OnlineResult", "PlatformLike", "online_problem", "run_online"]

RESULT_FORMAT = "dispatchatlas.online-run"
"""Value of the ``format`` field of a run record."""

_VERSION = 1
_TOLERANCE = TIME_TOLERANCE


class PlatformLike(Protocol):
    """Anything with the resources and the network of a platform."""

    @property
    def resources(self) -> tuple[Resource, ...]:
        """The nodes."""
        ...

    @property
    def network(self) -> Network | None:
        """The links between them."""
        ...


@dataclass(frozen=True)
class OnlineResult:
    """Every raw record of one online run, in JSON-ready form.

    Attributes:
        config: Planner, clock, budget, tolerance, priority, and library version.
        seed: Root seed of the run; pass ``k`` draws from
            ``derive_seed(seed, "online-pass", k)``.
        trace: SHA-256 digest of the trace.
        problem: SHA-256 digest of the run's problem.
        events: ``[time, kind, subject]`` of every event.
        passes: One mapping per pass: ``index``, ``began``, ``ended``,
            ``outcome``, ``tasks`` (how many it planned), ``placements``,
            ``charged`` and ``wall`` seconds, ``estimate_before`` (the forecast
            when it began), ``estimate_after`` (its plan's estimate at that
            time, uncharged), and ``anytime``, the ``[placements, value]``
            points at which its best estimate improved.
        answers: ``[task, due, began, done, mode, start, placements]`` of every
            answer.
        attempts: ``[task, mode, start, end, outcome]`` of every attempt, in
            commitment order; ``start`` is the commitment.
        workflows: ``[id, arrival, finish, response]``; the last two ``None``
            for a workflow that never finished.
        forecasts: ``[time, kind, value, [[workflow, finish], ...]]``: kind
            ``pre`` or ``post`` an event, or ``adopt``; value the mean
            predicted response over arrived workflows; the finishes of the
            unfinished ones, by workflow index, ``None`` where unpredictable.
        recovery: One mapping per event, see :func:`~dispatchatlas.online.metrics.recovery`.
        summary: Headline numbers of the run. ``end`` is the time of the
            last step of the loop: the latest event, task start or finish,
            answer end, or pass end the loop reached, which is not the horizon
            of the trace. ``charged`` is the sum of the ``charged`` of every
            pass and of every answer; an overtaken pass counts its full
            charge, including the part that never elapsed.
    """

    config: dict[str, Any]
    seed: int
    trace: str
    problem: str
    events: list[list[Any]] = field(default_factory=list)
    passes: list[dict[str, Any]] = field(default_factory=list)
    answers: list[list[Any]] = field(default_factory=list)
    attempts: list[list[Any]] = field(default_factory=list)
    workflows: list[list[Any]] = field(default_factory=list)
    forecasts: list[list[Any]] = field(default_factory=list)
    recovery: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-ready document."""
        return {
            "format": RESULT_FORMAT,
            "version": _VERSION,
            "config": self.config,
            "seed": self.seed,
            "trace": self.trace,
            "problem": self.problem,
            "events": self.events,
            "passes": self.passes,
            "answers": self.answers,
            "attempts": self.attempts,
            "workflows": self.workflows,
            "forecasts": self.forecasts,
            "recovery": self.recovery,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> OnlineResult:
        """Rebuild a record from :meth:`to_dict` output.

        Raises:
            SerializationError: If the document is not a run record.
        """
        if document.get("format") != RESULT_FORMAT or document.get("version") != _VERSION:
            raise SerializationError(f"not a version {_VERSION} {RESULT_FORMAT} document")
        try:
            return cls(
                config=dict(document["config"]),
                seed=int(document["seed"]),
                trace=str(document["trace"]),
                problem=str(document["problem"]),
                events=list(document["events"]),
                passes=list(document["passes"]),
                answers=list(document["answers"]),
                attempts=list(document["attempts"]),
                workflows=list(document["workflows"]),
                forecasts=list(document["forecasts"]),
                recovery=list(document["recovery"]),
                summary=dict(document["summary"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SerializationError(f"malformed run record: {error}") from error


def online_problem(platform: PlatformLike, trace: Trace) -> Problem:
    """Return the problem of a run: every task of every workflow, on the platform.

    Workflows are listed in arrival order. Each task's release, due date, and
    deadline are its workflow's arrival plus the task's offsets, and its group
    is the workflow id, so ``mean_response_time`` measures the workflows'
    response times. The problem is used for bookkeeping and checking only: a
    planner sees a task once its workflow has arrived.

    Raises:
        ModelError: If an event names an unknown node, or the problem is invalid.
    """
    resources = tuple(platform.resources)
    ids = {resource.id for resource in resources}
    unknown = sorted(trace.nodes - ids)
    if unknown:
        raise ModelError(f"the trace names unknown nodes {unknown}")
    tasks: list[Task] = []
    dependencies: list[Dependency] = []
    for arrival in trace.arrivals:
        at = arrival.time
        for task in arrival.workflow.tasks:
            tasks.append(
                Task(
                    task.id,
                    task.modes,
                    release=at + task.release,
                    due=None if task.due is None else at + task.due,
                    deadline=None if task.deadline is None else at + task.deadline,
                    weight=task.weight,
                    group=arrival.workflow.id,
                    metadata=task.metadata,
                )
            )
        dependencies.extend(arrival.workflow.dependencies)
    if not tasks:
        raise ModelError("the trace has no arrival")
    return Problem(
        id=f"online-{trace.digest[:12]}",
        tasks=tuple(tasks),
        resources=resources,
        dependencies=tuple(dependencies),
        objectives=("mean_response_time",),
        network=platform.network,
    )


def run_online(
    platform: PlatformLike,
    trace: Trace,
    planner: Planner | None,
    *,
    clock: PlanningClock | None = None,
    budget: Budget | None = None,
    seed: int = 0,
    tolerance: float = 0.05,
    priority: Callable[[Problem], Sequence[float]] = upward_ranks,
) -> OnlineResult:
    """Run ``planner`` over ``trace`` on ``platform`` and return every record.

    Args:
        platform: The nodes and network.
        trace: The events.
        planner: Called at every decision point; ``None`` leaves every task to
            the dispatcher.
        clock: How planning is charged. Defaults to counted at 100,000
            placements per second.
        budget: The budget of every pass. Defaults to unbounded.
        seed: Root seed of the planner's randomness.
        tolerance: Relative tolerance of the recovery measure.
        priority: Value per task of the run's problem; higher first among
            tasks due at one instant. Defaults to HEFT's upward rank.

    Returns:
        The run's records.
    """
    if not (math.isfinite(tolerance) and tolerance >= 0.0):
        raise ModelError(f"tolerance must be a non-negative number, got {tolerance!r}")
    problem = online_problem(platform, trace)
    simulation = _Simulation(
        problem,
        trace,
        planner,
        clock or PlanningClock(),
        budget or Budget(),
        seed,
        priority,
    )
    result = simulation.run(tolerance)
    result.recovery.extend(recovery(result, tolerance))
    return result


@dataclass(slots=True)
class _Entry:
    mode: int
    start: float
    source: str
    version: int


@dataclass(slots=True)
class _Pass:
    index: int
    began: float
    remaining: float
    plan: Plan | None
    outcome: PassOutcome
    row: dict[str, Any]


class _Simulation:
    def __init__(
        self,
        problem: Problem,
        trace: Trace,
        planner: Planner | None,
        clock: PlanningClock,
        budget: Budget,
        seed: int,
        priority: Callable[[Problem], Sequence[float]],
    ) -> None:
        self.problem = problem
        self.trace = trace
        self.planner = planner
        self.clock = clock
        self.budget = budget
        self.seed = seed
        arrivals = [arrival.time for arrival in trace.arrivals]
        self.model = model = Model(problem, priority(problem), arrivals)
        self.priority_name = getattr(priority, "__name__", "custom")
        size = problem.size
        self.status = [TaskStatus.UNRELEASED] * size
        self.unfinished_predecessors = [len(p) for p in model.predecessors]
        self.current: list[int | None] = [None] * size
        self.attempts: list[Attempt] = []
        self.commit_order: dict[int, int] = {}
        self.ready_since: dict[int, float] = {}
        self.losses: dict[int, int] = {}
        self.down: set[str] = set()
        self.known: list[int] = []
        self.left = [len(members) for members in model.members]
        self.finish: list[float | None] = [None] * len(model.members)
        self.base: Builder = model.decoder.builder()
        self.reserved: Builder = model.decoder.builder()
        self.entries: dict[int, _Entry] = {}
        self.starts: list[tuple[float, int, int]] = []
        self.running: list[tuple[float, int, int]] = []
        self.queue: list[tuple[tuple[float, float, int], int]] = []
        self.answering: tuple[int, float, float, tuple[int, ...]] | None = None
        self.pass_: _Pass | None = None
        self.passes = 0
        self.in_force: Plan | None = None
        self.version = 0
        self.now = 0.0
        self.thread_last = 0.0
        self.events = list(trace.events)
        self.next_event = 0
        self.rows_events: list[list[Any]] = []
        self.rows_passes: list[dict[str, Any]] = []
        self.rows_answers: list[list[Any]] = []
        self.rows_forecasts: list[list[Any]] = []
        self.answer_placements = 0
        self.answer_charged = 0.0

    # ------------------------------------------------------------ main loop

    def run(self, tolerance: float) -> OnlineResult:
        while True:
            now = self._next_time()
            if now is None:
                break
            self._advance_thread(now)
            self.now = max(self.now, now)
            self._finish_until(self.now)
            batch: list[Event] = []
            while (
                self.next_event < len(self.events)
                and self.events[self.next_event].time <= self.now + _TOLERANCE
            ):
                batch.append(self.events[self.next_event])
                self.next_event += 1
            if batch:
                self._apply(batch)
            self._settle()
            self._thread_step()
        return self._result(tolerance)

    def _next_time(self) -> float | None:
        times: list[float] = []
        if self.next_event < len(self.events):
            times.append(self.events[self.next_event].time)
        while self.running and not self._running_valid(self.running[0]):
            heapq.heappop(self.running)
        if self.running:
            times.append(self.running[0][0])
        while self.starts and not self._start_valid(self.starts[0]):
            heapq.heappop(self.starts)
        if self.starts:
            times.append(self.starts[0][0])
        if self.answering is not None:
            times.append(self.answering[2])
        elif self.queue:
            times.append(self.now)
        elif self.pass_ is not None:
            times.append(self.thread_last + self.pass_.remaining)
        return max(min(times), self.now) if times else None

    def _advance_thread(self, now: float) -> None:
        if self.pass_ is not None and self.answering is None:
            self.pass_.remaining = max(0.0, self.pass_.remaining - (now - self.thread_last))
        self.thread_last = max(self.thread_last, now)

    def _running_valid(self, item: tuple[float, int, int]) -> bool:
        _, attempt, task = item
        return self.current[task] == attempt and self.status[task] is TaskStatus.RUNNING

    def _start_valid(self, item: tuple[float, int, int]) -> bool:
        _, version, task = item
        entry = self.entries.get(task)
        return entry is not None and entry.version == version

    # ------------------------------------------------------------ execution

    def _finish_until(self, now: float) -> None:
        while self.running and self.running[0][0] <= now + _TOLERANCE:
            item = heapq.heappop(self.running)
            if self._running_valid(item):
                self._finish(item[2])

    def _settle(self) -> None:
        now = self.now
        while True:
            while self.starts and not self._start_valid(self.starts[0]):
                heapq.heappop(self.starts)
            if not self.starts or self.starts[0][0] > now + _TOLERANCE:
                return
            _, _, task = heapq.heappop(self.starts)
            entry = self.entries.pop(task)
            self._commit(task, entry.mode, entry.start)
            self._finish_until(now)

    def _commit(self, task: int, mode: int, start: float) -> None:
        model = self.model
        if any(self.status[b] is not TaskStatus.FINISHED for b in model.predecessors[task]):
            raise RuntimeError(f"task {task} starts before a predecessor finished")
        end = start + model.durations[task][mode]
        self.current[task] = len(self.attempts)
        self.attempts.append(Attempt(task, mode, start, end, AttemptOutcome.RUNNING))
        self.commit_order[task] = len(self.commit_order)
        self.status[task] = TaskStatus.RUNNING
        self.ready_since.pop(task, None)
        self.base.place(task, mode, start)
        heapq.heappush(self.running, (end, len(self.attempts) - 1, task))

    def _finish(self, task: int) -> None:
        model = self.model
        index = self.current[task]
        assert index is not None
        attempt = self.attempts[index]
        self.attempts[index] = Attempt(
            task, attempt.mode, attempt.start, attempt.end, AttemptOutcome.FINISHED
        )
        self.status[task] = TaskStatus.FINISHED
        workflow = model.workflow_of[task]
        self.left[workflow] -= 1
        if self.left[workflow] == 0:
            self.finish[workflow] = max(
                self.attempts[self.current[t] or 0].end for t in model.members[workflow]
            )
        for after in model.successors[task]:
            self.unfinished_predecessors[after] -= 1
            if self.unfinished_predecessors[after] == 0:
                self._make_ready(after, attempt.end)

    def _make_ready(self, task: int, since: float) -> None:
        self.status[task] = TaskStatus.READY
        self.ready_since[task] = since
        if task not in self.entries:
            heapq.heappush(self.queue, (self.model.key(task, since), task))

    def _lose(self, task: int) -> None:
        index = self.current[task]
        assert index is not None
        attempt = self.attempts[index]
        self.attempts[index] = Attempt(
            task, attempt.mode, attempt.start, self.now, AttemptOutcome.LOST
        )
        self.current[task] = None
        self.commit_order.pop(task, None)
        self.losses[task] = self.losses.get(task, 0) + 1
        self.status[task] = TaskStatus.READY
        self.ready_since[task] = self.now

    # ------------------------------------------------------------ events

    def _apply(self, batch: list[Event]) -> None:
        model = self.model
        self._forecast("pre")
        lost = False
        for event in batch:
            self.rows_events.append([event.time, event.kind.value, event.subject])
            if isinstance(event, Arrival):
                workflow = model.workflow_index[event.workflow.id]
                self.known.append(workflow)
                for task in model.members[workflow]:
                    self.status[task] = TaskStatus.RELEASED
                    if not model.predecessors[task]:
                        self.status[task] = TaskStatus.READY
                        self.ready_since[task] = self.now
            elif isinstance(event, Failure):
                self.down.add(event.node)
                for task, status in enumerate(self.status):
                    if status is TaskStatus.RUNNING:
                        mode = self.attempts[self.current[task] or 0].mode
                        if event.node in model.mode_nodes[task][mode]:
                            self._lose(task)
                            lost = True
            elif isinstance(event, Repair):
                self.down.discard(event.node)
        if lost:
            self._rebuild_base()
        keep = {
            task: entry
            for task, entry in self.entries.items()
            if (self.planner is None or entry.source == "answer")
            and model.mode_nodes[task][entry.mode].isdisjoint(self.down)
        }
        self.entries = keep
        self._rebuild_reserved()
        self.queue = [
            (model.key(task, self.ready_since[task]), task)
            for task, status in enumerate(self.status)
            if status is TaskStatus.READY and task not in self.entries
        ]
        heapq.heapify(self.queue)
        self._forecast("post")
        if self.planner is not None:
            self._begin_pass()

    def _rebuild_base(self) -> None:
        builder = self.model.decoder.builder()
        for task in sorted(self.commit_order, key=self.commit_order.__getitem__):
            index = self.current[task]
            assert index is not None
            attempt = self.attempts[index]
            builder.place(task, attempt.mode, attempt.start)
        self.base = builder

    def _rebuild_reserved(self) -> None:
        builder = self.base.copy()
        self.starts = []
        for task, entry in self.entries.items():
            builder.place(task, entry.mode, entry.start)
            heapq.heappush(self.starts, (entry.start, entry.version, task))
        self.reserved = builder

    def _add_entry(self, task: int, mode: int, start: float, source: str) -> None:
        self.version += 1
        self.entries[task] = _Entry(mode, start, source, self.version)
        heapq.heappush(self.starts, (start, self.version, task))

    # ------------------------------------------------------------ the scheduler

    def _thread_step(self) -> None:
        now = self.now
        if self.answering is not None:
            task, began, done, allowed = self.answering
            if done <= now + _TOLERANCE:
                self.answering = None
                self._answer(task, began, allowed)
            return
        while self.queue:
            _, task = heapq.heappop(self.queue)
            if self.status[task] is not TaskStatus.READY or task in self.entries:
                continue
            allowed = self.model.allowed(task, self.down)
            if not allowed:
                continue
            cost = len(allowed)
            charged = self.clock.counted(cost)
            self.answer_placements += cost
            self.answer_charged += charged
            self.answering = (task, now, now + charged, allowed)
            return
        current = self.pass_
        if current is not None and current.remaining <= _TOLERANCE:
            self.pass_ = None
            current.row["ended"] = now
            if current.outcome is PassOutcome.ADOPTED and current.plan is not None:
                self._adopt(current.plan)
            current.row["outcome"] = current.outcome.value

    def _answer(self, task: int, began: float, ranked: tuple[int, ...]) -> None:
        if self.status[task] is not TaskStatus.READY or task in self.entries:
            return
        # The answer ranks the nodes it was charged for that are still up.
        allowed = tuple(m for m in ranked if self.model.mode_nodes[task][m].isdisjoint(self.down))
        if not allowed:
            heapq.heappush(self.queue, (self.model.key(task, self.ready_since[task]), task))
            return
        start, _, mode = earliest_finish(self.reserved, task, allowed, self.now)
        self.reserved.place(task, mode, start)
        self._add_entry(task, mode, start, "answer")
        self.rows_answers.append(
            [
                self.problem.tasks[task].id,
                self.ready_since[task],
                began,
                self.now,
                mode,
                start,
                len(ranked),
            ]
        )

    def _plannable(self) -> list[int]:
        model = self.model
        plannable: set[int] = set()
        order: list[int] = []
        for task in model.problem.topological_order:
            status = self.status[task]
            if status not in (TaskStatus.RELEASED, TaskStatus.READY):
                continue
            if not model.allowed(task, self.down):
                continue
            if all(self.status[b].committed or b in plannable for b in model.predecessors[task]):
                plannable.add(task)
                order.append(task)
        return order

    def _state(self) -> SystemState:
        committed = {}
        for task, status in enumerate(self.status):
            if status.committed:
                index = self.current[task]
                assert index is not None
                committed[task] = self.attempts[index]
        return SystemState(
            now=self.now,
            status=tuple(self.status),
            committed=committed,
            down=frozenset(self.down),
            ready_since=dict(self.ready_since),
            known=tuple(self.model.workflows[w] for w in self.known),
            losses=dict(self.losses),
        )

    def _previous(self, tasks: Sequence[int]) -> Plan | None:
        if self.in_force is None:
            return None
        wanted = set(tasks)
        modes = self.in_force.mode_of()
        order = [t for t in self.in_force.order if t in wanted]
        held = set(order)
        rest = sorted(
            (t for t in tasks if t not in held),
            key=lambda t: self.model.key(t, self.ready_since.get(t, math.inf)),
        )
        order.extend(rest)
        allowed = {t: self.model.allowed(t, self.down) for t in order}
        return Plan(
            tuple(order),
            tuple(modes[t] if t in modes and modes[t] in allowed[t] else -1 for t in order),
        )

    def _begin_pass(self) -> None:
        planner = self.planner
        assert planner is not None
        if self.pass_ is not None:
            self.pass_.row["ended"] = self.now
            self.pass_.row["outcome"] = PassOutcome.OVERTAKEN.value
            self.pass_ = None
        index = self.passes
        self.passes += 1
        tasks = self._plannable()
        context = PlanningContext(
            self.model,
            self.base,
            self._state(),
            tasks,
            self.budget,
            self.clock,
            self._previous(tasks),
            index,
            derive_seed(self.seed, "online-pass", index),
        )
        before = self.rows_forecasts[-1][2] if self.rows_forecasts else None
        outcome = PassOutcome.ADOPTED
        plan: Plan | None = None
        started = time.perf_counter()
        try:
            plan = planner.plan(context)
        except BudgetExhausted:
            outcome = PassOutcome.OVERRUN
        wall = time.perf_counter() - started
        if plan is None and outcome is PassOutcome.ADOPTED:
            outcome = PassOutcome.DECLINED
        if plan is not None:
            context.check(plan)
        charged = (
            context.clock.counted(context.placements) if self.clock.mode == "counted" else wall
        )
        seconds = self.budget.seconds
        if outcome is PassOutcome.OVERRUN:
            charged = seconds if seconds is not None else charged
        elif seconds is not None and charged > seconds + _TOLERANCE:
            outcome, charged = PassOutcome.OVERRUN, seconds
        after = context.estimate_free(plan).value if plan is not None else None
        row: dict[str, Any] = {
            "index": index,
            "began": self.now,
            "ended": None,
            "outcome": None,
            "tasks": len(tasks),
            "placements": context.placements,
            "charged": charged,
            "wall": wall,
            "estimate_before": before,
            "estimate_after": after,
            "anytime": [[p, v] for p, v in context.anytime],
        }
        self.rows_passes.append(row)
        self.pass_ = _Pass(index, self.now, charged, plan, outcome, row)
        self.thread_last = self.now

    def _adopt(self, plan: Plan) -> None:
        model = self.model
        builder = self.base.copy()
        order: list[int] = []
        modes: list[int] = []
        for task, mode in zip(plan.order, plan.modes, strict=True):
            if not self.status[task].committed:
                order.append(task)
                modes.append(mode)
        placed = place_in_order(
            model,
            builder,
            order,
            modes,
            lambda t: model.allowed(t, self.down),
            self.now,
        )
        self.entries = {}
        self.starts = []
        for task, mode in placed:
            self._add_entry(task, mode, builder.starts[task], "plan")
        self.reserved = builder
        self.in_force = plan
        self.queue = []
        self._forecast("adopt")

    # ------------------------------------------------------------ forecasts

    def _forecast(self, kind: str) -> None:
        model = self.model
        open_workflows = [w for w in self.known if self.finish[w] is None]
        if not self.known:
            return
        builder = self.reserved.copy()
        tasks = [t for t in self._plannable() if t not in self.entries]
        dispatch_rollout(
            model,
            builder,
            tasks,
            self.ready_since,
            lambda t: model.allowed(t, self.down),
            self.now,
        )
        finishes = predicted_finishes(model, builder, open_workflows)
        total, count = 0.0, 0
        for workflow in self.known:
            finish = self.finish[workflow]
            if finish is None:
                finish = finishes[workflow]
            if finish is not None:
                total += finish - model.arrivals[workflow]
                count += 1
        value = total / count if count else None
        self.rows_forecasts.append(
            [self.now, kind, value, [[w, finishes[w]] for w in open_workflows]]
        )

    # ------------------------------------------------------------ result

    def _result(self, tolerance: float) -> OnlineResult:
        model = self.model
        problem = self.problem
        attempts = [
            [problem.tasks[a.task].id, a.mode, a.start, a.end, a.outcome.value]
            for a in self.attempts
        ]
        workflows: list[list[Any]] = []
        responses: list[float] = []
        for w, name in enumerate(model.workflows):
            finish = self.finish[w]
            response = None if finish is None else finish - model.arrivals[w]
            if response is not None:
                responses.append(response)
            workflows.append([name, model.arrivals[w], finish, response])
        lost = [a for a in self.attempts if a.outcome is AttemptOutcome.LOST]
        outcomes = [row["outcome"] for row in self.rows_passes]
        pass_placements = sum(row["placements"] for row in self.rows_passes)
        pass_charged = sum(row["charged"] for row in self.rows_passes)
        summary = {
            "workflows": len(workflows),
            "finished": len(responses),
            "mean_response": sum(responses) / len(responses) if responses else None,
            "max_response": max(responses, default=None),
            "unfinished_tasks": sum(1 for s in self.status if s is not TaskStatus.FINISHED),
            "lost_attempts": len(lost),
            "lost_work": sum(a.end - a.start for a in lost),
            "passes": len(self.rows_passes),
            **{o.value: outcomes.count(o.value) for o in PassOutcome},
            "answers": len(self.rows_answers),
            "pass_placements": pass_placements,
            "answer_placements": self.answer_placements,
            "charged": pass_charged + self.answer_charged,
            "end": self.now,
        }
        planner = self.planner
        config = {
            "planner": planner.describe() if planner is not None else {"planner": "dispatch"},
            "clock": self.clock.to_dict(),
            "budget": self.budget.to_dict(),
            "tolerance": tolerance,
            "priority": self.priority_name,
            "library": dispatchatlas.__version__,
        }
        return OnlineResult(
            config=config,
            seed=self.seed,
            trace=self.trace.digest,
            problem=problem_digest(problem),
            events=self.rows_events,
            passes=self.rows_passes,
            answers=self.rows_answers,
            attempts=attempts,
            workflows=workflows,
            forecasts=self.rows_forecasts,
            recovery=[],
            summary=summary,
        )
