"""Tests of the origin of solver clocks."""

from __future__ import annotations

import time

import pytest

from dispatchatlas.core.clock import clock_origin, clock_start
from dispatchatlas.core.solver import StopCriteria, StopReason
from dispatchatlas.solvers.continuous.core import Box, FunctionRun
from dispatchatlas.solvers.exact.common import Clock
from dispatchatlas.solvers.rules import RandomSearch
from dispatchatlas.solvers.search import Exhausted, Search
from tests.helpers import random_problem


def test_a_clock_starts_when_it_is_created_without_an_origin() -> None:
    before = time.perf_counter()
    start = clock_start()
    assert before <= start <= time.perf_counter()


def test_clocks_created_inside_the_block_start_at_its_origin() -> None:
    with clock_origin() as origin:
        assert origin <= time.perf_counter()
        time.sleep(0.01)
        assert clock_start() == origin
        with clock_origin(origin - 5.0) as inner:
            assert inner == origin - 5.0
            assert clock_start() == inner
        assert clock_start() == origin
    assert clock_start() > origin


def test_the_origin_ends_with_the_block_when_the_block_raises() -> None:
    def fail() -> None:
        with clock_origin(0.0):
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        fail()
    assert clock_start() > 0.0


@pytest.mark.parametrize("ahead", [5.0, 1e9, float("inf"), float("nan"), float("-inf")])
def test_an_origin_that_is_not_a_past_reading_of_the_clock_is_refused(ahead: float) -> None:
    # A reading of another clock, such as time.time(), lies far ahead of perf_counter.
    with (
        pytest.raises(ValueError, match="a clock origin must be a past reading"),
        clock_origin(time.perf_counter() + ahead),
    ):
        pass
    # The refusal leaves no origin behind.
    before = time.perf_counter()
    assert clock_start() >= before
    with clock_origin(time.perf_counter()) as origin:
        assert clock_start() == origin


def test_a_solver_that_budgets_phases_opens_an_origin_for_each() -> None:
    problem = random_problem(3)
    phase = StopCriteria(time_limit=0.2)
    with clock_origin():
        time.sleep(0.3)
        # The limit of a search created now ended 0.1 s ago, on the outer origin.
        spent = RandomSearch().solve(problem, phase, 1)
        assert spent.stop_reason is StopReason.TIME_LIMIT
        assert spent.evaluations == 1
        assert spent.elapsed >= 0.25
        # A block of its own gives the phase its whole limit, counted from its entry.
        with clock_origin():
            own = RandomSearch().solve(problem, phase, 1)
        assert own.stop_reason is StopReason.TIME_LIMIT
        assert own.evaluations > 1
        # Counted from the outer origin, the elapsed time would be 0.5 s or more.
        assert 0.2 <= own.elapsed < 0.45
        assert own.trace[0].elapsed < 0.2


def test_a_search_counts_its_time_and_its_limit_from_the_origin() -> None:
    problem = random_problem(3)
    stop = StopCriteria(time_limit=5.0, max_evaluations=40)
    alone = RandomSearch().solve(problem, stop, 1)
    assert alone.stop_reason is StopReason.EVALUATION_LIMIT
    assert alone.evaluations == 40
    assert alone.elapsed < 5.0
    # Ten seconds of the run have passed when the solver is called: its
    # time limit ended five seconds ago, so it keeps its first schedule.
    with clock_origin(time.perf_counter() - 10.0):
        late = RandomSearch().solve(problem, stop, 1)
    assert late.stop_reason is StopReason.TIME_LIMIT
    assert late.evaluations == 1
    assert late.elapsed >= 10.0
    assert late.trace[-1].elapsed >= 10.0
    assert RandomSearch().solve(problem, stop, 1).evaluations == 40


def test_the_set_up_of_a_search_is_inside_its_elapsed_time() -> None:
    problem = random_problem(3)
    with clock_origin():
        time.sleep(0.3)
        search = Search(problem, StopCriteria(max_evaluations=1), 0, RandomSearch.info)
        assert search.elapsed >= 0.25
        assert search.remaining_time() is None
    # Outside the block a search counts from its own creation, a moment ago.
    assert Search(problem, StopCriteria(max_evaluations=1), 0, RandomSearch.info).elapsed < 0.3


def test_the_clock_of_the_exact_solvers_starts_at_the_origin() -> None:
    alone = Clock(5.0)
    assert not alone.expired()
    assert alone.elapsed < 5.0
    with clock_origin(time.perf_counter() - 10.0):
        late = Clock(5.0)
        partly = Clock(60.0)
    assert late.expired()
    assert late.elapsed >= 10.0
    assert late.remaining() == 0.0
    remaining = partly.remaining()
    assert remaining is not None
    assert remaining <= 50.0


def test_the_clock_of_a_function_run_starts_at_the_origin() -> None:
    box = Box((0.0,), (1.0,))
    stop = StopCriteria(time_limit=5.0)
    alone = FunctionRun(sum, box, stop)
    alone.evaluate((0.5,))
    alone.check()
    assert alone.elapsed < 5.0
    with clock_origin(time.perf_counter() - 10.0):
        late = FunctionRun(sum, box, stop)
    assert late.elapsed >= 10.0
    late.evaluate((0.5,))
    with pytest.raises(Exhausted):
        late.check()
