"""A solver and a problem for the tests of the process batch, which child processes import.

The solver costs what the noisy target of ``targets.py`` costs: the true cost of its
parameters, scaled by the instance, plus noise that depends on the seed. It runs no search.
The problem of instance ``i`` is a one-task problem named ``synthetic-<i>``. The configuration
``m = "c"`` makes the solver return no schedule, and the environment variable
:data:`FAIL_ABOVE` makes it raise for a configuration whose ``x`` is above its value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar

from dispatchatlas.core import (
    Problem,
    Resource,
    SolveResult,
    SolverInfo,
    Status,
    StopCriteria,
    StopReason,
    Task,
)
from dispatchatlas.tuning import Configuration, Experiment
from tests.tuning.targets import noisy_cost

FAIL_ABOVE = "DISPATCHATLAS_TEST_FAIL_ABOVE"
"""The environment variable that holds the value of ``x`` above which the solver raises."""

SOLVER = "tests.tuning.batch_targets:make_solver"
PROBLEM = "tests.tuning.batch_targets:make_problem"


@dataclass(frozen=True)
class NoisySolver:
    """Returns the noisy cost of its parameters on the instance of the problem it is given."""

    x: float
    n: int
    k: int
    m: str
    info: ClassVar[SolverInfo] = SolverInfo(
        id="test-noisy",
        name="Noisy target",
        family="random-sampling",
        summary="Returns a synthetic cost of its parameters and runs no search.",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        limit = os.environ.get(FAIL_ABOVE)
        if limit is not None and self.x > float(limit):
            raise RuntimeError(f"the target refuses x = {self.x}")
        if self.m == "c":
            return SolveResult(
                solver=self.info.id,
                problem=problem.id,
                objective="makespan",
                status=Status.UNKNOWN,
                schedule=None,
                objectives={},
                stop_reason=StopReason.TIME_LIMIT,
                elapsed=0.5,
            )
        instance = int(problem.id.removeprefix("synthetic-"))
        configuration: Configuration = {"x": self.x, "n": self.n, "k": self.k, "m": self.m}
        cost = noisy_cost(Experiment(configuration, instance, seed))
        return SolveResult(
            solver=self.info.id,
            problem=problem.id,
            objective="makespan",
            status=Status.FEASIBLE,
            schedule=None,
            objectives={"makespan": cost},
            stop_reason=StopReason.EVALUATION_LIMIT,
            elapsed=0.25,
            evaluations=stop.max_evaluations or 1,
            extras={"first_schedule_digest": f"{seed:016x}", "repairs": 0},
        )


def make_solver(configuration: dict[str, Any]) -> NoisySolver:
    """Return the solver of a configuration."""
    return NoisySolver(**configuration)


def make_problem(instance: int) -> Problem:
    """Return the problem of an instance, an integer."""
    return Problem(
        f"synthetic-{instance}", tasks=(Task.on("a", "m1", 1),), resources=(Resource("m1"),)
    )
