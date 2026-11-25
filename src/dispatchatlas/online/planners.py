"""Built-in planners: the greedy reference and the warm restart of a static solver.

- :class:`OnlineEft`: online earliest finish time. At every decision point it
  ranks every task to plan again, in the order the tasks fall due (ties by
  the run's priority, by default HEFT's upward rank), each on the node that
  finishes it earliest given the tasks before it. It is the greedy reference:
  every planner that seeds with it and never searches commits exactly where
  it does, and on a trace of arrivals with free planning it commits every
  task where the dispatcher alone would, at the same time.
- :class:`WarmRestart`: re-runs a registered static solver at every decision
  point under the budget, on the :class:`~dispatchatlas.online.planning.Residual`
  problem, and keeps the plan in force unless the solver's plan is strictly
  better. The library's solvers take no initial solution, so the warm start
  is the incumbent the solver must beat: the better of the greedy plan and the
  plan in force before the event.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.solver import StopCriteria
from dispatchatlas.online.planning import BudgetExhausted, Estimate, Plan, PlanningContext
from dispatchatlas.solvers.registry import default_registry

__all__ = ["OnlineEft", "WarmRestart"]


@dataclass(frozen=True)
class OnlineEft:
    """Online earliest finish time: the greedy plan, at every decision point."""

    @property
    def name(self) -> str:
        """``"online-eft"``."""
        return "online-eft"

    def describe(self) -> dict[str, Any]:
        """Return ``{"planner": "online-eft"}``."""
        return {"planner": self.name}

    def plan(self, context: PlanningContext) -> Plan | None:
        """Return the greedy plan."""
        plan, _ = context.seed_plan()
        return plan


@dataclass(frozen=True)
class WarmRestart:
    """Re-run a static solver from the plan in force, under the pass's budget.

    A pass computes the greedy plan, then, budget permitting, evaluates the
    plan in force before the event (repaired: tasks it did not hold, or held
    on a node now down, take their earliest-finishing node), and keeps the
    better as the incumbent. It then runs the solver on the residual problem.
    Each of the solver's evaluations is charged as one decode of the residual
    problem in which every task tries every one of its modes, an upper bound
    on the builder's work; the evaluations it may make are what the budget
    leaves after reserving one evaluation of its result. In ``measured`` mode
    the solver's time limit is the seconds the budget leaves. The solver's
    plan, its tasks ordered by their start in its schedule, replaces the
    incumbent only if its estimate is strictly better. With no budget left
    after the greedy plan the pass returns the greedy plan: seeding without
    searching.

    Attributes:
        solver: Registered id of the static solver.
        parameters: Its parameters.
        patience: Iterations without improvement after which the solver stops,
            for the time-limit-and-stall protocol.
        evaluations: Most evaluations the solver may make in one pass, within
            the budget. 0 never runs it: the pass seeds and never searches.
        reuse: Also evaluate the plan in force as a candidate incumbent.
    """

    solver: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    patience: int | None = None
    evaluations: int | None = None
    reuse: bool = True

    def __post_init__(self) -> None:
        if self.patience is not None and self.patience < 1:
            raise ConfigError(f"patience must be at least 1, got {self.patience!r}")
        if self.evaluations is not None and self.evaluations < 0:
            raise ConfigError(f"evaluations must be non-negative, got {self.evaluations!r}")

    @property
    def name(self) -> str:
        """``"warm-restart"``."""
        return "warm-restart"

    def describe(self) -> dict[str, Any]:
        """Return the planner's name, solver, and parameters."""
        return {
            "planner": self.name,
            "solver": self.solver,
            "parameters": dict(self.parameters),
            "patience": self.patience,
            "evaluations": self.evaluations,
            "reuse": self.reuse,
        }

    def plan(self, context: PlanningContext) -> Plan | None:
        """Seed, reuse the plan in force, then re-run the solver; return the best plan."""
        plan, estimate = context.seed_plan()
        if not context.tasks or (self.evaluations == 0 and not self.reuse):
            return plan
        best: tuple[Plan, Estimate] = (plan, estimate)
        previous = context.previous
        if self.reuse and previous is not None and previous != plan:
            remaining = context.remaining
            if remaining is None or context.cost(previous) <= remaining:
                candidate = context.evaluate(previous)
                if candidate.better_than(best[1]):
                    best = (previous, candidate)
        solver = default_registry().create(self.solver, **dict(self.parameters))
        residual = context.residual(deadlines="deadline" in solver.info.supports)
        per_decode = sum(len(task.modes) for task in residual.problem.tasks)
        final = len(context.tasks)
        remaining = context.remaining
        evaluations = self.evaluations
        if remaining is not None:
            affordable = (remaining - final) // max(1, per_decode)
            evaluations = affordable if evaluations is None else min(evaluations, affordable)
        if evaluations is not None and evaluations < 1:
            return best[0]
        seconds = context.remaining_seconds() if context.clock.mode == "measured" else None
        if evaluations is None and seconds is None:
            return best[0]
        stop = StopCriteria(
            time_limit=seconds,
            max_evaluations=evaluations,
            patience=self.patience,
        )
        result = solver.solve(residual.problem, stop, context.seed)
        try:
            context.charge(result.evaluations * per_decode)
        except BudgetExhausted:
            return best[0]
        if result.schedule is None:
            return best[0]
        starts = {item.task: (item.start, item.mode) for item in result.schedule.assignments}
        candidate_plan = residual.plan_from(starts)
        remaining = context.remaining
        if remaining is not None and context.cost(candidate_plan) > remaining:
            return best[0]
        candidate = context.evaluate(candidate_plan)
        if candidate.better_than(best[1]):
            best = (candidate_plan, candidate)
        return best[0]
