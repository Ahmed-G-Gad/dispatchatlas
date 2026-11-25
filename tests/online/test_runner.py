"""The online loop: hand-checked runs, commitment, failures, the greedy property, and the clock."""

from __future__ import annotations

import math
from typing import Any

import pytest

from dispatchatlas.core.errors import ModelError
from dispatchatlas.online import (
    Arrival,
    Budget,
    Failure,
    OnlineEft,
    PassOutcome,
    Plan,
    PlanningClock,
    PlanningContext,
    TaskStatus,
    Trace,
    WarmRestart,
    check_run,
    online_problem,
    run_online,
)
from tests.online.cases import TwoNodes, continuum_case, fork, fork_trace

FREE = PlanningClock(rate=math.inf)
SLOW = PlanningClock(rate=1.0)  # one placement per second: every charge is visible


def _attempts(result: Any) -> list[tuple[str, int, float, float, str]]:
    return [tuple(row) for row in result.attempts]


def test_the_fork_under_free_planning():
    # a goes to n1 (finishes at 2, not 4). At 2, b and c fall due; b first by
    # index: n1 and n2 both finish it at 5, the lower mode wins, n1 [2, 5); c then
    # finishes at 6 on n1, 7 on n2: n1 [5, 6). n1 fails at 3: b is lost after 1
    # second and c's plan on n1 is withdrawn. Re-planned at 3 on n2 alone, in due
    # order: c (due since 2) [3, 8), then b (due since 3) [8, 11). By the repair at
    # 10 both have started. The workflow finishes at 11.
    for planner in (None, OnlineEft()):
        result = run_online(TwoNodes(), fork_trace(), planner, clock=FREE)
        assert _attempts(result) == [
            ("a", 0, 0.0, 2.0, "finished"),
            ("b", 0, 2.0, 3.0, "lost"),
            ("c", 1, 3.0, 8.0, "finished"),
            ("b", 1, 8.0, 11.0, "finished"),
        ]
        assert result.workflows == [["w", 0.0, 11.0, 11.0]]
        assert result.summary["lost_work"] == 1.0
        assert result.summary["lost_attempts"] == 1
        assert check_run(online_problem(TwoNodes(), fork_trace()), fork_trace(), result) == ()


def test_the_fork_with_one_placement_per_second_under_online_eft():
    # Pass 0 begins at 0 and costs 6 placements (2 modes x 3 tasks), 6 seconds.
    # Answering comes first: a is ranked from 0 to 2 (2 placements) and starts
    # on n1 at 2. n1 fails at 3, losing a after 1 second and overtaking pass 0.
    # Pass 1 begins at 3 and costs 3 (n2 only); a is answered first, from 3 to 4,
    # and starts on n2 at 4 [4, 8); pass 1 resumes at 4 and is adopted at 7: its
    # plan, made at 3, puts b then c on n2, placed after a: b [8, 11), c [11, 16).
    # The repair at 10 withdraws c's plan. Pass 2 begins at 10 and costs 2; c is
    # answered from 10 to 12 on n1, which finishes it at 13, not 17 on n2. Pass 2
    # is adopted at 14 with nothing left to place.
    result = run_online(TwoNodes(), fork_trace(), OnlineEft(), clock=SLOW)
    assert _attempts(result) == [
        ("a", 0, 2.0, 3.0, "lost"),
        ("a", 1, 4.0, 8.0, "finished"),
        ("b", 1, 8.0, 11.0, "finished"),
        ("c", 0, 12.0, 13.0, "finished"),
    ]
    summary = [
        (p["index"], p["began"], p["ended"], p["outcome"], p["placements"], p["charged"])
        for p in result.passes
    ]
    assert summary == [
        (0, 0.0, 3.0, "overtaken", 6, 6.0),
        (1, 3.0, 7.0, "adopted", 3, 3.0),
        (2, 10.0, 14.0, "adopted", 2, 2.0),
    ]
    assert [row[0] for row in result.answers] == ["a", "a", "c"]
    assert [(row[2], row[3], row[6]) for row in result.answers] == [
        (0.0, 2.0, 2),
        (3.0, 4.0, 1),
        (10.0, 12.0, 2),
    ]
    assert result.workflows == [["w", 0.0, 13.0, 13.0]]
    assert result.summary["charged"] == 6 + 3 + 2 + 2 + 1 + 2
    assert result.summary["end"] == 14.0


def test_the_fork_with_one_placement_per_second_under_the_dispatcher():
    # Without a planner every task is answered when it falls due, and an answer is
    # never revised unless its node fails. a: ranked 0-2, n1 at 2, lost at 3,
    # ranked again 3-4 on n2 alone, [4, 8). At 8, b and c fall due with n1 down:
    # b ranked 8-9, n2 [9, 12); c ranked 9-10 over n2 only, which it was charged
    # for, so the repair at 10 does not help it: n2 [12, 17).
    result = run_online(TwoNodes(), fork_trace(), None, clock=SLOW)
    assert _attempts(result) == [
        ("a", 0, 2.0, 3.0, "lost"),
        ("a", 1, 4.0, 8.0, "finished"),
        ("b", 1, 9.0, 12.0, "finished"),
        ("c", 1, 12.0, 17.0, "finished"),
    ]
    assert result.passes == []
    assert result.workflows == [["w", 0.0, 17.0, 17.0]]


def test_recovery_of_the_fork_after_its_failure_and_its_repair():
    # Forecasts of the workflow's response under online EFT at one placement per
    # second. Just after the arrival: a n1 [0, 2), b n1 [2, 5), c n1 [5, 6): 6
    # (forecasts use the dispatch rule at once). Just before the failure at 3: a
    # runs [2, 4) on n1, then b n1 [4, 7) and c n1 [7, 8): 8. Just after: a
    # restarts on n2 alone, a [3, 7), b [7, 10), c [10, 15): 15. At the adoption
    # at 7: b [8, 11), c [11, 16): 16; 16 again just before the repair, 11 just
    # after it (c on n1 [10, 11)), and at the adoption at 14 the workflow has
    # finished, at 13. The failure's cohort never returns within 5% of 8: it is
    # censored at the end of the run, 14, after 11 seconds, with a degradation of
    # (15-8)*4 + (16-8)*3 + (11-8)*4 = 64. The repair's cohort is at 11 against 16
    # at once: recovered after 0 seconds.
    result = run_online(TwoNodes(), fork_trace(), OnlineEft(), clock=SLOW, tolerance=0.05)
    assert [(row[0], row[1], row[2]) for row in result.forecasts] == [
        (0.0, "post", 6.0),
        (3.0, "pre", 8.0),
        (3.0, "post", 15.0),
        (7.0, "adopt", 16.0),
        (10.0, "pre", 16.0),
        (10.0, "post", 11.0),
        (14.0, "adopt", 13.0),
    ]
    arrival, failure, repair = result.recovery
    assert arrival["cohort"] == 0
    assert arrival["recovered"] is None
    assert failure["before"] == 8.0
    assert failure["after"] == 15.0
    assert failure["recovered"] is False
    assert failure["recovery_time"] == 11.0
    assert failure["degradation"] == 64.0
    assert repair["recovered"] is True
    assert repair["recovery_time"] == 0.0
    assert repair["degradation"] == 0.0


@pytest.mark.parametrize("seed", range(4))
def test_online_eft_commits_where_the_dispatcher_does_under_free_planning(seed):
    platform, trace = continuum_case(seed, failures=False)
    alone = run_online(platform, trace, None, clock=FREE)
    greedy = run_online(platform, trace, OnlineEft(), clock=FREE)
    assert greedy.attempts == alone.attempts
    assert greedy.workflows == alone.workflows


@pytest.mark.parametrize("clock", [FREE, PlanningClock(rate=2000.0)])
@pytest.mark.parametrize("seed", range(3))
def test_a_planner_that_seeds_and_never_searches_commits_where_greedy_does(seed, clock):
    platform, trace = continuum_case(seed)
    greedy = run_online(platform, trace, OnlineEft(), clock=clock, seed=seed)
    seeded = run_online(
        platform,
        trace,
        WarmRestart("tabu-search", evaluations=0, reuse=False),
        clock=clock,
        seed=seed,
    )
    assert seeded.attempts == greedy.attempts
    assert [p["placements"] for p in seeded.passes] == [p["placements"] for p in greedy.passes]
    assert seeded.forecasts == greedy.forecasts


class _Recording:
    """Online EFT that keeps what each pass saw as committed, and a searching planner."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.seen: list[dict[int, tuple[int, float, float]]] = []
        self.name = "recording"

    def describe(self) -> dict[str, Any]:
        return {"planner": self.name}

    def plan(self, context: PlanningContext) -> Plan | None:
        self.seen.append({t: (a.mode, a.start, a.end) for t, a in context.state.committed.items()})
        for task in context.tasks:
            assert not context.state.status[task].committed
        plan: Plan | None = self.inner.plan(context)
        return plan


@pytest.mark.parametrize("seed", range(3))
def test_no_committed_reservation_moves_and_every_execution_is_feasible(seed):
    platform, trace = continuum_case(seed, applications=8)
    problem = online_problem(platform, trace)
    planner = _Recording(WarmRestart("tabu-search", reuse=True))
    result = run_online(
        platform, trace, planner, clock=PlanningClock(rate=5e4), budget=Budget(placements=3000)
    )
    assert check_run(problem, trace, result) == ()
    assert result.summary["unfinished_tasks"] == 0
    starts = {}
    for task_id, mode, start, end, outcome in result.attempts:
        starts[(problem.task_index[task_id], start)] = (mode, end, outcome)
    for seen in planner.seen:
        for task, (mode, start, end) in seen.items():
            final_mode, final_end, outcome = starts[(task, start)]
            assert final_mode == mode
            assert final_end == end or (outcome == "lost" and final_end <= end)


def test_a_failure_loses_running_work_and_the_node_takes_nothing_until_repaired():
    platform, trace = continuum_case(11, applications=10)
    result = run_online(platform, trace, OnlineEft(), clock=PlanningClock(rate=1e4))
    problem = online_problem(platform, trace)
    assert check_run(problem, trace, result) == ()
    lost = [row for row in result.attempts if row[4] == "lost"]
    assert lost, "the case must lose some work"
    assert result.summary["lost_work"] == pytest.approx(sum(r[3] - r[2] for r in lost))
    for task_id, *_ in lost:
        finished = [r for r in result.attempts if r[0] == task_id and r[4] == "finished"]
        assert len(finished) == 1


def test_the_counted_clock_reproduces_a_run_exactly():
    platform, trace = continuum_case(2)
    planner = WarmRestart("simulated-annealing")
    kwargs: dict[str, Any] = {
        "clock": PlanningClock(rate=3e4),
        "budget": Budget(placements=600),
        "seed": 7,
    }
    first = run_online(platform, trace, planner, **kwargs)
    second = run_online(platform, trace, planner, **kwargs)
    for result in (first, second):
        for row in result.passes:
            row["wall"] = 0.0
    assert first == second
    third = run_online(platform, trace, planner, **{**kwargs, "seed": 8})
    assert third.attempts != first.attempts or third.passes != first.passes


def test_a_pass_that_exceeds_its_budget_loses_its_window():
    result = run_online(
        TwoNodes(), fork_trace(), OnlineEft(), clock=SLOW, budget=Budget(placements=2)
    )
    # Pass 0 needs 6 placements and would hold the scheduler for the 2 it is
    # allowed, from 2 to 4 after answering a, but the failure at 3 overtakes it.
    # Pass 1 needs 3: it holds the scheduler from 4, after answering a again,
    # to 6, and adopts nothing. Pass 2 needs 2 and is adopted.
    outcomes = [(p["outcome"], p["charged"], p["ended"]) for p in result.passes]
    assert outcomes == [("overtaken", 2.0, 3.0), ("overrun", 2.0, 6.0), ("adopted", 2.0, 12.0)]
    measured = run_online(
        TwoNodes(),
        fork_trace(),
        OnlineEft(),
        clock=PlanningClock("measured", rate=1.0),
        budget=Budget(seconds=0.0),
    )
    assert {p["outcome"] for p in measured.passes} <= {"overrun", "adopted"}
    assert check_run(online_problem(TwoNodes(), fork_trace()), fork_trace(), measured) == ()


def test_an_invalid_plan_is_refused():
    class Bad:
        name = "bad"

        def describe(self) -> dict[str, Any]:
            return {"planner": self.name}

        def plan(self, context: PlanningContext) -> Plan | None:
            return Plan(context.tasks[:1], (0,))

    with pytest.raises(ModelError, match="exactly the tasks"):
        run_online(TwoNodes(), fork_trace(), Bad(), clock=FREE)


def test_a_declined_pass_leaves_the_tasks_to_the_dispatcher():
    class Nothing:
        name = "nothing"

        def describe(self) -> dict[str, Any]:
            return {"planner": self.name}

        def plan(self, context: PlanningContext) -> Plan | None:
            return None

    result = run_online(TwoNodes(), fork_trace(), Nothing(), clock=SLOW)
    alone = run_online(TwoNodes(), fork_trace(), None, clock=SLOW)
    assert {p["outcome"] for p in result.passes} == {PassOutcome.DECLINED.value}
    assert result.attempts == alone.attempts


def test_the_state_a_planner_sees_matches_the_rules():
    seen: list[tuple[float, tuple[TaskStatus, ...], frozenset[str], tuple[int, ...]]] = []

    class Look:
        name = "look"

        def describe(self) -> dict[str, Any]:
            return {"planner": self.name}

        def plan(self, context: PlanningContext) -> Plan | None:
            state = context.state
            seen.append((state.now, state.status, state.down, context.tasks))
            return context.seed_plan()[0]

    run_online(TwoNodes(), fork_trace(), Look(), clock=FREE)
    ready, released, running, finished = (
        TaskStatus.READY,
        TaskStatus.RELEASED,
        TaskStatus.RUNNING,
        TaskStatus.FINISHED,
    )
    assert seen == [
        (0.0, (ready, released, released), frozenset(), (0, 1, 2)),
        (3.0, (finished, ready, ready), frozenset({"n1"}), (1, 2)),
        (10.0, (finished, running, finished), frozenset(), ()),
    ]


def test_a_trace_naming_an_unknown_node_is_refused():
    trace = Trace.of([Arrival(0.0, fork()), Failure(1.0, "n9")])
    with pytest.raises(ModelError, match="unknown nodes"):
        run_online(TwoNodes(), trace, None)
