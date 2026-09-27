"""The search run-time: budgets, stop reasons, probes, the trace, the archive, and results."""

from __future__ import annotations

import random
from itertools import pairwise
from typing import Any

import pytest

from dispatchatlas.core import objectives as registry
from dispatchatlas.core.errors import SolverError, UnsupportedProblemError
from dispatchatlas.core.model import Problem, Resource, Task
from dispatchatlas.core.objectives import Objective, evaluate, register_objective
from dispatchatlas.core.pareto import dominates
from dispatchatlas.core.solver import SolverInfo, Status, StopCriteria, StopReason, TracePoint
from dispatchatlas.core.validation import check_schedule
from dispatchatlas.solvers import solve
from dispatchatlas.solvers.evaluation import Score
from dispatchatlas.solvers.learning.temporal_difference import Agent, Target
from dispatchatlas.solvers.operators import random_modes, random_topological_order
from dispatchatlas.solvers.rules import RandomSearch
from dispatchatlas.solvers.scheduling import Decoded, Decoder
from dispatchatlas.solvers.search import Exhausted, Search
from tests.helpers import chain_problem, random_problem


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a test register objectives without leaking them into other tests."""
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))


def _info(**changes: Any) -> SolverInfo:
    fields: dict[str, Any] = {
        "id": "toy-search",
        "name": "Toy search",
        "family": "random-sampling",
        "summary": "Samples random orders.",
    }
    fields.update(changes)
    return SolverInfo(**fields)


def _sample(search: Search, *, modes: bool = False) -> None:
    """Random search until a stop criterion fires."""
    problem = search.problem
    with search:
        while True:
            search.next_iteration()
            chosen = random_modes(problem, search.rng) if modes else None
            search.evaluate(random_topological_order(problem, search.rng), chosen)


def _deadline_problem(*objectives: str) -> Problem:
    """``a`` meets its deadline only when it runs first, which delays ``b``."""
    return Problem(
        "deadline",
        tasks=(Task.on("a", "m", 5, deadline=5), Task.on("b", "m", 1)),
        resources=(Resource("m"),),
        objectives=objectives or ("makespan",),
    )


# ------------------------------------------------------------ construction


def test_iterative_solvers_need_a_bounded_budget() -> None:
    problem = chain_problem()
    for stop in (StopCriteria(), StopCriteria(target=1.0), StopCriteria(patience=4)):
        with pytest.raises(SolverError, match="needs a time, evaluation, or iteration limit"):
            Search(problem, stop, 0, _info())
        with pytest.raises(SolverError):
            RandomSearch().solve(problem, stop)
    Search(problem, StopCriteria(), 0, _info(iterative=False))
    for stop in (
        StopCriteria(time_limit=1.0),
        StopCriteria(max_evaluations=1),
        StopCriteria(max_iterations=1),
    ):
        Search(problem, stop, 0, _info())


def test_unsupported_features_are_refused() -> None:
    problem = random_problem(1, setups=True, network=True)
    with pytest.raises(UnsupportedProblemError, match="does not model features") as caught:
        Search(problem, StopCriteria(max_evaluations=1), 0, _info(supports={"precedence"}))
    assert "'setups'" in str(caught.value)
    assert "'network'" in str(caught.value)


# --------------------------------------------------------------- incumbent


def test_evaluate_counts_and_keeps_the_first_of_equal_candidates() -> None:
    search = Search(chain_problem(), StopCriteria(max_evaluations=10), 0, _info())
    first = search.assess([0, 1, 2])
    assert search.evaluations == 1
    assert search.best is first
    assert first.order == [0, 1, 2]
    assert first.modes is None
    assert first.score == Score(0.0, 5.0)
    same = search.assess([2, 0, 1], [0, 0, 0])
    assert same.score == first.score
    assert search.best is first
    assert same.modes == [0, 0, 0]
    worse = search.evaluate([1, 0, 2])
    assert search.best is first
    assert worse >= first.score
    assert search.evaluations == 3


def test_a_feasible_schedule_beats_any_violating_one() -> None:
    problem = _deadline_problem("total_completion_time")
    search = Search(problem, StopCriteria(max_evaluations=5), 0, _info())
    assert search.evaluate([1, 0]) == Score(1.0, 7.0)
    assert search.evaluate([0, 1]) == Score(0.0, 11.0)
    result = search.result()
    assert result.status is Status.FEASIBLE
    assert result.value == 11.0
    assert [(p.value, p.feasible) for p in result.trace[:2]] == [(7.0, False), (11.0, True)]


# ------------------------------------------------------------------- probes


def test_probes_count_but_never_become_the_incumbent() -> None:
    problem = chain_problem()
    search = Search(problem, StopCriteria(max_evaluations=10, target=4.0), 0, _info())
    search.evaluate([0, 1, 2])
    best, trace = search.best, list(search._trace)
    # The partial order scores better, and below the target, yet changes nothing.
    score, decoded = search.probe([0])
    assert best is not None
    assert score == Score(0.0, 3.0) < best.score
    assert decoded.sequence == (0,)
    assert search.evaluations == 2
    assert search.best is best
    assert search._trace == trace
    assert search.front() == []
    result = search.result()
    assert result.value == 5.0
    assert result.evaluations == 2
    assert result.stop_reason is StopReason.COMPLETED


def test_probes_obey_the_evaluation_and_time_limits() -> None:
    search = Search(chain_problem(), StopCriteria(max_evaluations=2, time_limit=60), 0, _info())
    search.evaluate([0, 1, 2])
    search.probe([0, 2])
    with pytest.raises(Exhausted) as caught:
        search.probe([0])
    assert caught.value.reason is StopReason.EVALUATION_LIMIT
    timed = Search(chain_problem(), StopCriteria(time_limit=1.0), 0, _info())
    timed.evaluate([0, 1, 2])
    timed._started -= 10.0
    with pytest.raises(Exhausted) as caught:
        timed.probe([0])
    assert caught.value.reason is StopReason.TIME_LIMIT
    assert timed.evaluations == 1


# --------------------------------------------------------------- stop reasons


def test_the_evaluation_limit_is_exact() -> None:
    search = Search(random_problem(2), StopCriteria(max_evaluations=7), 3, _info())
    _sample(search)
    result = search.result()
    assert result.stop_reason is StopReason.EVALUATION_LIMIT
    assert result.evaluations == 7
    assert result.iterations == 8


def test_the_iteration_limit() -> None:
    search = Search(random_problem(2), StopCriteria(max_iterations=4), 3, _info())
    _sample(search)
    result = search.result()
    assert result.stop_reason is StopReason.ITERATION_LIMIT
    assert (result.iterations, result.evaluations) == (4, 4)


def test_patience_stops_a_stagnating_search() -> None:
    search = Search(chain_problem(), StopCriteria(max_evaluations=100, patience=3), 0, _info())
    with search:
        search.evaluate([0, 1, 2])
        while True:
            search.next_iteration()
            search.evaluate([0, 1, 2])
    assert search.reason is StopReason.STAGNATION
    assert (search.iterations, search.evaluations) == (3, 4)


def test_patience_counts_from_the_last_improvement() -> None:
    problem = _deadline_problem()
    search = Search(problem, StopCriteria(max_iterations=50, patience=2), 0, _info())
    with search:
        search.evaluate([1, 0])
        search.next_iteration()
        search.evaluate([1, 0])
        search.next_iteration()
        search.evaluate([0, 1])
        while True:
            search.next_iteration()
            search.evaluate([1, 0])
    assert search.reason is StopReason.STAGNATION
    assert search.iterations == 4


def test_the_target_needs_a_feasible_schedule() -> None:
    problem = _deadline_problem()
    search = Search(problem, StopCriteria(max_evaluations=10, target=6.0), 0, _info())
    with search:
        search.evaluate([1, 0])
        assert search.best is not None
        assert not search.best.score.feasible
        search.evaluate([0, 1])
        pytest.fail("the target should have stopped the search")
    assert search.reason is StopReason.TARGET_REACHED
    result = search.result()
    assert result.stop_reason is StopReason.TARGET_REACHED
    assert (result.value, result.evaluations) == (6.0, 2)


@pytest.mark.usefixtures("scratch_registry")
def test_the_target_of_a_maximised_objective_is_a_lower_bound() -> None:
    register_objective(
        Objective("span", "max", "time", "Makespan, to maximise.", lambda p, s: s.makespan)
    )
    problem = chain_problem().with_objectives("span")
    search = Search(problem, StopCriteria(max_evaluations=10, target=6.0), 0, _info())
    with search:
        search.evaluate([0, 2, 1], [0, 0, 0])
        assert search.best is not None
        assert search.best.score.value == -5.0
    assert search.reason is None
    search = Search(problem, StopCriteria(max_evaluations=10, target=5.0), 0, _info())
    with search:
        search.evaluate([0, 1, 2])
    assert search.reason is StopReason.TARGET_REACHED
    assert search.result().value == 5.0


def test_the_time_limit_is_checked_only_after_a_first_evaluation() -> None:
    search = Search(chain_problem(), StopCriteria(time_limit=1.0), 0, _info())
    search._started -= 10.0
    search.check()
    search.next_iteration()
    search.evaluate([0, 1, 2])
    for step in (search.check, search.next_iteration, lambda: search.evaluate([0, 1, 2])):
        with pytest.raises(Exhausted) as caught:
            step()
        assert caught.value.reason is StopReason.TIME_LIMIT
        assert str(caught.value) == "time-limit"
    assert search.remaining_time() == 0.0
    assert search.evaluations == 1


def test_a_short_time_limit_alone_ends_a_run() -> None:
    search = Search(random_problem(4, tasks=20), StopCriteria(time_limit=0.05), 1, _info())
    _sample(search)
    result = search.result()
    assert result.stop_reason is StopReason.TIME_LIMIT
    assert result.evaluations >= 1
    assert result.elapsed >= 0.05


def test_leaving_the_block_normally_completes_and_other_errors_propagate() -> None:
    search = Search(chain_problem(), StopCriteria(max_evaluations=5), 0, _info())
    with search:
        search.evaluate([0, 1, 2])
    assert search.reason is None
    assert search.result().stop_reason is StopReason.COMPLETED
    with pytest.raises(KeyError), search:
        raise KeyError("boom")
    assert search.reason is None


# -------------------------------------------------------------------- trace


def _check_trace(trace: tuple[TracePoint, ...], sign: float) -> None:
    """Check a trace; ``sign`` is 1 for a minimised objective and -1 for a maximised one."""
    for earlier, later in pairwise(trace):
        assert later.elapsed >= earlier.elapsed
        assert later.evaluations >= earlier.evaluations
        assert later.iteration >= earlier.iteration
        assert later.feasible or not earlier.feasible
        if earlier.feasible and later.feasible:
            assert sign * later.value <= sign * earlier.value
    improvements = [point.evaluations for point in trace[:-1]]
    assert improvements == sorted(set(improvements))


@pytest.mark.parametrize("seed", range(6))
def test_the_trace_is_monotone_and_ends_at_the_result(seed: int) -> None:
    problem = random_problem(seed, tasks=12, deadlines=True, pools=1)
    search = Search(problem, StopCriteria(max_evaluations=80), seed, _info())
    _sample(search, modes=True)
    result = search.result()
    _check_trace(result.trace, sign=1.0)
    final = result.trace[-1]
    assert final.value == result.value
    assert final.feasible is result.feasible
    assert (final.evaluations, final.iteration) == (result.evaluations, result.iterations)
    assert result.trace[0].evaluations == 1


@pytest.mark.usefixtures("scratch_registry")
def test_the_trace_of_a_maximised_objective_never_decreases() -> None:
    register_objective(
        Objective(
            "busy",
            "max",
            "time",
            "Sum of end times.",
            lambda p, s: sum(a.end for a in s.assignments),
        )
    )
    problem = random_problem(3, tasks=10).with_objectives("busy")
    search = Search(problem, StopCriteria(max_evaluations=60), 3, _info())
    _sample(search)
    result = search.result()
    _check_trace(result.trace, sign=-1.0)
    assert len(result.trace) >= 3


# ------------------------------------------------------------------ archive


def test_the_archive_keeps_exactly_the_non_dominated_feasible_candidates() -> None:
    problem = random_problem(5, tasks=10, deadlines=True, pools=1)
    assert len(problem.objectives) == 3
    search = Search(problem, StopCriteria(max_evaluations=150), 5, _info(), archive=True)
    seen: list[tuple[tuple[float, ...], bool]] = []
    with search:
        while True:
            search.next_iteration()
            candidate = search.assess(
                random_topological_order(problem, search.rng), random_modes(problem, search.rng)
            )
            seen.append((search.vector(candidate), candidate.score.feasible))
    assert any(not feasible for _, feasible in seen), "the instance should violate deadlines"
    members = search.front()
    vectors = [search.vector(member) for member in members]
    assert members
    assert all(member.score.feasible for member in members)
    assert len(set(vectors)) == len(vectors)
    for a in vectors:
        assert not any(dominates(b, a) for b in vectors)
    for vector, feasible in seen:
        if feasible:
            assert any(v == vector or dominates(v, vector) for v in vectors)
    result = search.result()
    assert [point.values for point in result.front] == sorted(
        tuple(search.evaluator.values(member.decoded).values()) for member in members
    )
    for point in result.front:
        assert point.schedule is not None
        assert check_schedule(problem, point.schedule) == ()


def test_the_archive_is_off_for_single_objective_solvers() -> None:
    problem = random_problem(5, tasks=8)
    search = Search(problem, StopCriteria(max_evaluations=20), 5, _info())
    _sample(search)
    assert search.front() == []
    assert search.result().front == ()
    multi = Search(problem, StopCriteria(max_evaluations=20), 5, _info(multi_objective=True))
    _sample(multi)
    assert multi.front()
    assert multi.result().front


# ------------------------------------------------------------------- result


def test_results_report_the_incumbent() -> None:
    problem = _deadline_problem("makespan", "total_completion_time")
    search = Search(problem, StopCriteria(max_evaluations=5), 0, _info())
    with pytest.raises(SolverError, match="evaluated no candidate"):
        search.result()
    search.extras.update({"moves": 3, "note": "kept"})
    search.evaluate([1, 0])
    violated = search.result({"moves": 4})
    assert violated.status is Status.VIOLATED
    assert violated.extras == {"moves": 4, "note": "kept"}
    assert violated.objectives == {"makespan": 6.0, "total_completion_time": 7.0}
    assert violated.schedule is not None
    assert [v.code for v in check_schedule(problem, violated.schedule)] == ["deadline"]
    search.evaluate([0, 1])
    result = search.result()
    assert result.status is Status.FEASIBLE
    assert (result.solver, result.problem, result.objective) == (
        "toy-search",
        "deadline",
        "makespan",
    )
    assert result.objectives == {"makespan": 6.0, "total_completion_time": 11.0}
    assert (result.evaluations, result.iterations) == (2, 0)
    assert result.bound is None
    assert result.front == ()


def test_generators_are_seeded() -> None:
    problem = chain_problem()
    first = Search(problem, StopCriteria(max_evaluations=1), 42, _info())
    second = Search(problem, StopCriteria(max_evaluations=1), 42, _info())
    assert first.rng.random() == second.rng.random() == random.Random(42).random()
    assert first.numpy_rng().random() == second.numpy_rng().random()
    assert first.remaining_time() is None


# ----------------------------------------------------------------- progress


def test_progress_is_the_largest_used_fraction_of_a_set_limit() -> None:
    problem = chain_problem()
    assert Search(problem, StopCriteria(), 0, _info(iterative=False)).progress() == 0.0
    search = Search(problem, StopCriteria(max_evaluations=10, max_iterations=4), 0, _info())
    assert search.progress() == 0.0
    for _ in range(4):
        search.evaluate([0, 1, 2])
    assert search.progress() == pytest.approx(0.4)
    for _ in range(2):
        search.next_iteration()
    assert search.progress() == pytest.approx(0.5)
    for _ in range(2):
        search.next_iteration()
    assert search.progress() == 1.0


def test_progress_ignores_the_clock_unless_time_is_the_only_limit() -> None:
    problem = chain_problem()
    search = Search(problem, StopCriteria(time_limit=10.0, max_evaluations=100), 0, _info())
    search.evaluate([0, 1, 2])
    search._started -= 5.0
    assert search.progress() == pytest.approx(0.01)
    assert search.progress(deterministic=False) == pytest.approx(0.5, abs=0.01)
    search._started -= 50.0
    assert search.progress(deterministic=False) == 1.0
    timed = Search(problem, StopCriteria(time_limit=10.0), 0, _info())
    timed._started -= 2.5
    assert timed.progress() == pytest.approx(0.25, abs=0.01)


# -------------------------------------------------------------- submissions


def _built(search: Search, order: list[int]) -> Decoded:
    """Place ``order`` task by task, each in its best mode, with the search's builder."""
    builder = search.evaluator.decoder.builder()
    for task in order:
        start, _, mode = builder.best(task)
        builder.place(task, mode, start)
    return builder.decoded()


def test_a_submitted_schedule_counts_and_competes_like_an_assessed_one() -> None:
    problem = random_problem(3, tasks=10, max_modes=2, pools=1)
    order = list(problem.topological_order)
    assessed = Search(problem, StopCriteria(max_evaluations=5), 0, _info()).assess(order)
    search = Search(problem, StopCriteria(max_evaluations=5), 0, _info())
    decoded = _built(search, order)
    candidate = search.submit(decoded)
    assert search.evaluations == 1
    assert search.best is candidate
    assert candidate.decoded is decoded
    assert (candidate.score, candidate.decoded) == (assessed.score, assessed.decoded)
    assert (candidate.order, candidate.modes) == (order, list(decoded.modes))
    assert [point.evaluations for point in search._trace] == [1]
    modes = [0] * problem.size
    other = search.submit(decoded, list(reversed(order)), modes)
    assert (other.order, other.modes) == (list(reversed(order)), modes)
    assert search.best is candidate
    assert search.evaluations == 2
    result = search.result()
    assert result.schedule == decoded.schedule(problem)
    assert result.evaluations == 2


def test_partial_schedules_are_refused() -> None:
    search = Search(chain_problem(), StopCriteria(max_evaluations=5), 0, _info())
    builder = search.evaluator.decoder.builder()
    builder.place(0, 0)
    with pytest.raises(SolverError, match="leaves 2 of 3 tasks unplaced"):
        search.submit(builder.decoded())
    assert search.evaluations == 0
    assert search.best is None


def test_submissions_obey_the_budget_and_the_target() -> None:
    search = Search(chain_problem(), StopCriteria(max_evaluations=1, target=5.0), 0, _info())
    decoded = _built(search, [0, 1, 2])
    with pytest.raises(Exhausted) as caught:
        search.submit(decoded)
    assert caught.value.reason is StopReason.TARGET_REACHED
    assert search.best is not None
    assert search.best.decoded is decoded
    with pytest.raises(Exhausted) as caught:
        search.submit(decoded)
    assert caught.value.reason is StopReason.EVALUATION_LIMIT
    assert search.evaluations == 1


def test_submissions_enter_the_archive() -> None:
    problem = random_problem(5, tasks=8)
    search = Search(problem, StopCriteria(max_evaluations=20), 0, _info(), archive=True)
    rng = random.Random(0)
    for _ in range(12):
        search.submit(_built(search, random_topological_order(problem, rng)))
    vectors = [search.vector(member) for member in search.front()]
    assert vectors
    for vector in vectors:
        assert not any(dominates(other, vector) for other in vectors)


@pytest.mark.parametrize(
    "solver_id",
    [
        "min-min",
        "max-min",
        "sufferage",
        "enumeration",
        "q-learning-dispatching",
        "sarsa-dispatching",
    ],
)
def test_constructors_report_the_schedule_they_built_without_decoding_it_again(
    solver_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = random_problem(2, tasks=6, max_modes=2, pools=0)

    def refuse(*_: Any, **__: Any) -> Decoded:
        raise AssertionError("the built schedule was decoded again")

    monkeypatch.setattr(Decoder, "decode", refuse)
    result = solve(problem, solver_id, max_evaluations=40, seed=1)
    assert result.schedule is not None
    assert check_schedule(problem, result.schedule) == ()
    assert result.objectives == pytest.approx(evaluate(problem, result.schedule))


@pytest.mark.parametrize("target", ["q-learning", "sarsa"])
def test_learning_agents_report_the_schedule_their_policy_built(target: Target) -> None:
    problem = random_problem(7, tasks=12, max_modes=3, pools=1, network=True)
    agent = Agent(
        problem, Decoder(problem), target=target, learning_rate=0.5, discount=0.9, stage_buckets=4
    )
    built = agent.greedy().schedule
    result = solve(problem, f"{target}-dispatching", max_evaluations=1)
    assert result.schedule == built.schedule(problem)
    assert result.evaluations == 1
