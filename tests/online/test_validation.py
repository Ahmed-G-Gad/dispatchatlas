"""The independent check of an executed online schedule."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from dispatchatlas.core import Problem
from dispatchatlas.online import (
    OnlineEft,
    OnlineResult,
    PlanningClock,
    Trace,
    check_run,
    online_problem,
    run_online,
)
from tests.online.cases import TwoNodes, fork_trace


def _run() -> tuple[Problem, Trace, OnlineResult]:
    trace = fork_trace()
    problem = online_problem(TwoNodes(), trace)
    result = run_online(TwoNodes(), trace, OnlineEft(), clock=PlanningClock(rate=math.inf))
    return problem, trace, result


def _with(result: OnlineResult, attempts: list[list[Any]]) -> OnlineResult:
    return replace(result, attempts=attempts)


def test_the_check_accepts_the_runs_own_execution():
    problem, trace, result = _run()
    assert check_run(problem, trace, result) == ()


def test_the_check_finds_each_broken_rule():
    problem, trace, result = _run()
    # The run: a n1 [0, 2), b n1 [2, 3) lost, c n2 [3, 8), b n2 [8, 11).
    cases = {
        "before the data": [
            ["a", 0, 0.0, 2.0, "finished"],
            ["b", 0, 2.0, 3.0, "lost"],
            ["c", 1, 1.0, 6.0, "finished"],
            ["b", 1, 8.0, 11.0, "finished"],
        ],
        "while it is down": [
            ["a", 0, 0.0, 2.0, "finished"],
            ["b", 0, 2.0, 3.0, "lost"],
            ["c", 0, 4.0, 5.0, "finished"],
            ["b", 1, 8.0, 11.0, "finished"],
        ],
        "holds 2.0 units": [
            ["a", 0, 0.0, 2.0, "finished"],
            ["b", 0, 2.0, 3.0, "lost"],
            ["c", 1, 3.0, 8.0, "finished"],
            ["b", 1, 7.0, 10.0, "finished"],
        ],
        "none of its resources failed": [
            ["a", 0, 0.0, 2.0, "finished"],
            ["b", 1, 2.0, 3.0, "lost"],
            ["c", 1, 3.0, 8.0, "finished"],
            ["b", 1, 8.0, 11.0, "finished"],
        ],
        "instead of": [
            ["a", 0, 0.0, 2.5, "finished"],
            ["b", 0, 2.5, 3.0, "lost"],
            ["c", 1, 3.0, 8.0, "finished"],
            ["b", 1, 8.0, 11.0, "finished"],
        ],
        "finished twice": [
            ["a", 0, 0.0, 2.0, "finished"],
            ["a", 1, 2.0, 6.0, "finished"],
            ["c", 1, 6.0, 11.0, "finished"],
            ["b", 1, 11.0, 14.0, "finished"],
        ],
        "still running": [
            ["a", 0, 0.0, 2.0, "finished"],
            ["c", 1, 3.0, 8.0, "running"],
        ],
    }
    for expected, attempts in cases.items():
        issues = check_run(problem, trace, _with(result, attempts))
        assert any(expected in issue for issue in issues), (expected, issues)
