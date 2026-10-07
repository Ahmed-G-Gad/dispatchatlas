"""The processor time the engine measures for an attempt."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from dispatchatlas.campaigns.execution import Completion, Task, attempt
from dispatchatlas.core import Solver, StopCriteria, problem_digest
from dispatchatlas.solvers.rules import InputOrder
from tests.campaigns.solvers import Delegating
from tests.helpers import chain_problem


def _attempt(solver: Solver) -> Completion:
    problem = chain_problem()
    stop = StopCriteria(max_evaluations=5)
    result = attempt(Task("run-0", 1, problem, problem_digest(problem), solver, stop, seed=3))
    assert isinstance(result, Completion)
    return result


def test_the_processor_seconds_of_an_attempt_include_the_children_it_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def times() -> SimpleNamespace:
        # No child time before the solve, 2 s of it after: 1.5 s in user mode, 0.5 s in the system.
        calls.append(1)
        first = len(calls) == 1
        return SimpleNamespace(
            children_user=0.0 if first else 1.5, children_system=0.0 if first else 0.5
        )

    monkeypatch.setattr(os, "times", times)
    measurement = _attempt(InputOrder()).measurement
    assert len(calls) == 2
    assert measurement.cpu_seconds is not None
    assert 2.0 <= measurement.cpu_seconds < 2.5
    # The rule itself computes for a few milliseconds.
    assert measurement.wall_seconds < 0.5


def test_a_solver_that_computes_in_a_child_process_is_measured_where_the_system_reports_it() -> (
    None
):
    measurement = _attempt(Delegating(seconds=0.3)).measurement
    # The child times its own 0.3 s; the two clocks may differ by a tick.
    assert measurement.wall_seconds >= 0.25
    assert measurement.cpu_seconds is not None
    if sys.platform == "win32":
        # Windows reports no time of child processes: the solver's own process was idle.
        assert measurement.cpu_seconds < 0.25
    else:
        assert measurement.cpu_seconds >= 0.25
