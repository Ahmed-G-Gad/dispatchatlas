"""The Baldwinian local search of the job-shop steady-state genetic algorithm."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dispatchatlas.core import Problem, SolveResult, StopCriteria, StopReason
from dispatchatlas.core.serialization import canonical_json, schedule_to_dict
from dispatchatlas.solvers.search import Exhausted, Search
from dispatchatlas.solvers.shop import tabu
from dispatchatlas.solvers.shop.decoding import BUILDERS, BuilderName
from dispatchatlas.solvers.shop.family import (
    REPRESENTATIONS,
    RepresentationName,
    make_representation,
)
from dispatchatlas.solvers.shop.genetic import ShopSteadyStateGa, _improve, _Tally
from dispatchatlas.solvers.shop.graph import heads_tails, machine_orders
from dispatchatlas.solvers.shop.instance import JobShop
from dispatchatlas.solvers.shop.observers import TrajectoryObserver
from dispatchatlas.solvers.shop.run import Evaluated, ShopRun
from dispatchatlas.solvers.shop.tabu import critical_block_descent
from tests.helpers import small_job_shop
from tests.solvers.shop.contract import check_contract
from tests.solvers.shop.families import one_job, untimed
from tests.solvers.shop.instances import random_job_shop, ta01

# ------------------------------------------------------------------------ helpers


def _ga(representation: RepresentationName, builder: BuilderName = "active") -> ShopSteadyStateGa:
    return ShopSteadyStateGa(representation, builder, population_size=6, local_search=True)


def _run(
    shop: JobShop,
    representation: RepresentationName,
    builder: BuilderName = "active",
    *,
    evaluations: int = 10**6,
    seed: int = 0,
) -> ShopRun[Any]:
    """Return a run of the GA on ``shop`` with a budget that a test does not reach by default."""
    solver = _ga(representation, builder)
    search = Search(shop.problem, StopCriteria(max_evaluations=evaluations), seed, solver.info)
    representation_ = make_representation(
        shop, representation, key_move="redraw", sequence_move="insert"
    )
    return ShopRun(search, shop, builder, representation_, TrajectoryObserver(shop))


def _cases() -> list[JobShop]:
    """Taillard's first shop and small random ones, with release times and returning jobs."""
    shops = [ta01()]
    for index in range(24):
        problem = random_job_shop(
            index,
            jobs=3 + index % 5,
            machines=2 + index % 4,
            releases=index % 2 == 0,
            revisits=index % 3 == 0,
        )
        shops.append(JobShop.from_problem(problem))
    return shops


def _decoded(run: ShopRun[Any]) -> Evaluated[Any]:
    """Evaluate the first random genotype of the run's operator stream."""
    return run.evaluate(run.representation.initial(run.operators))


def _copy(genotype: Any) -> Any:
    return np.array(genotype, copy=True)


def _extra(result: SolveResult, key: str) -> int:
    value = result.extras[key]
    assert isinstance(value, int)
    return value


def _digest(problem: Problem, result: SolveResult) -> str:
    assert result.schedule is not None
    record = {
        "schedule": schedule_to_dict(result.schedule, problem.id),
        "trace": [[p.evaluations, p.iteration, p.value] for p in result.trace],
        "extras": untimed(result.extras),
        "counts": [result.evaluations, result.iterations, str(result.stop_reason)],
    }
    return hashlib.sha256(canonical_json(record).encode()).hexdigest()


# ------------------------------------------------------------- off changes nothing

# What the algorithm returned on ta01 before it had a local search, at seed 11, 300 evaluations,
# a population of 8 and a mutation rate of 0.5: the schedule, the trace, the untimed extras and
# the counts, hashed.
BEFORE: dict[tuple[str, str], str] = {
    ("keys", "semi-active"): "a02806c1e0aa35e73b7e0e65e301c16d18bdfe5e7874bd233f5bb31f643ae482",
    ("sequence", "semi-active"): "409d3d3185fe693e8b8c3d1300adf2a4a2fe83d752d6867066a55e6cef8ff2de",
    ("keys", "active"): "6c600bbb0b7352fb4d60e4528bb1e99bbecde3e108c95e38590ad0d79ac1adc3",
    ("sequence", "active"): "7219b87a2e32a226c9f65469dd013719fb085223ed04880a5088e0d50c0845a0",
    ("keys", "non-delay"): "49f1ed190a9196fdda666c803ad57ddcd02043717acad700981b26bdad3b64fe",
    ("sequence", "non-delay"): "8a42218dc6f94c86e9621cb762ef7e3071aea1e7e6dcda6545d42f8005b0bb8b",
}


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_without_the_local_search_a_run_returns_what_it_returned_before(
    representation: RepresentationName, builder: BuilderName
) -> None:
    problem = ta01().problem
    stop = StopCriteria(max_evaluations=300)
    default = ShopSteadyStateGa(representation, builder, population_size=8, mutation_rate=0.5)
    named = ShopSteadyStateGa(
        representation, builder, population_size=8, mutation_rate=0.5, local_search=False
    )
    assert default == named
    assert not default.local_search
    result = default.solve(problem, stop, seed=11)
    assert _digest(problem, result) == BEFORE[representation, builder]
    assert "descent_evaluations" not in result.extras
    assert "descent_moves" not in result.extras


# ---------------------------------------------------------------- the descent counts


def test_the_counting_descent_applies_the_swaps_of_the_plain_one() -> None:
    applied = 0
    for shop in _cases():
        for representation in REPRESENTATIONS:
            decoded = _decoded(_run(shop, representation))
            orders = machine_orders(shop, decoded.decoded)
            plain = critical_block_descent(shop, orders, _run(shop, representation))
            counting_run = _run(shop, representation)
            counting = critical_block_descent(shop, orders, counting_run, count_neighbours=True)
            assert counting.orders == plain.orders
            assert counting.moves == plain.moves
            assert counting.decoded.starts == plain.decoded.starts
            # Every applied swap is one of the neighbours counted.
            assert counting_run.search.evaluations >= counting.moves
            applied += counting.moves
    assert applied > 30


def test_the_counting_descent_counts_every_neighbour_it_recomputes_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recomputed: list[bool] = []
    position = tabu._position

    def recorded(shop: JobShop, orders: Any) -> Any:
        found = position(shop, orders)
        recomputed.append(found is not None)
        return found

    monkeypatch.setattr(tabu, "_position", recorded)
    counted = 0
    for shop in _cases():
        start = _decoded(_run(shop, "sequence"))
        run = _run(shop, "sequence")
        recomputed.clear()
        descent = critical_block_descent(
            shop, machine_orders(shop, start.decoded), run, parent=start, count_neighbours=True
        )
        # The first recomputation is the position the descent starts from, which the caller
        # counted; each other one that builds a schedule is one evaluation, and the observer
        # sees each as a move.
        assert run.search.evaluations == sum(recomputed[1:])
        assert isinstance(run.observer, TrajectoryObserver)
        assert run.observer.evaluations == run.observer.moves == run.search.evaluations
        assert run.search.evaluations >= descent.moves
        counted += run.search.evaluations
    # The check is not vacuous.
    assert counted > 20


def test_the_counting_descent_stops_where_the_budget_ends() -> None:
    shop = ta01()
    start = _decoded(_run(shop, "keys"))
    orders = machine_orders(shop, start.decoded)
    for budget in (1, 2, 7):
        run = _run(shop, "keys", evaluations=budget)
        with pytest.raises(Exhausted):
            critical_block_descent(shop, orders, run, count_neighbours=True)
        assert run.search.evaluations == budget


# -------------------------------------------------------- the individual it improves


def test_the_descent_never_worsens_an_individual_and_leaves_its_genotype() -> None:
    improved = 0
    for shop in _cases():
        for representation in REPRESENTATIONS:
            for builder in BUILDERS:
                run = _run(shop, representation, builder)
                decoded = _decoded(run)
                genotype, before = decoded.genotype, _copy(decoded.genotype)
                sequence = decoded.sequence.copy()
                tally = _Tally()
                member = _improve(run, decoded, tally)
                # Baldwinian: the genotype and its job sequence stay as they were.
                assert member.genotype is genotype
                assert np.array_equal(genotype, before)
                assert np.array_equal(member.sequence, sequence)
                assert np.array_equal(run.representation.sequence(genotype), sequence)
                # The fitness is the improved schedule's, and never worse than the decode.
                assert member.score <= decoded.score
                assert member.score.value == max(member.decoded.ends)
                found = heads_tails(shop, machine_orders(shop, member.decoded))
                assert found is not None
                assert member.decoded.starts == found[0]
                # One evaluation for the decode and one for each neighbour.
                assert run.search.evaluations == 1 + tally.evaluations
                assert tally.evaluations >= tally.moves
                assert run.search.best is not None
                assert run.search.best.score <= member.score
                improved += member.score < decoded.score
    assert improved > 40


@pytest.mark.parametrize("builder", BUILDERS)
def test_both_arms_apply_the_same_descent_to_their_equal_schedules(builder: BuilderName) -> None:
    for seed in range(4):
        for shop in (ta01(), *_cases()[1:8]):
            runs = [_run(shop, arm, builder, seed=seed) for arm in REPRESENTATIONS]
            members = []
            for run in runs:
                tally = _Tally()
                genotypes = [run.representation.initial(run.operators) for _ in range(3)]
                improved = [_improve(run, run.evaluate(g), tally) for g in genotypes]
                members.append((improved, tally, run.search.evaluations))
            (keys, keys_tally, keys_count), (sequences, sequences_tally, sequences_count) = members
            # Equal schedules decoded, so equal descents: the same schedules, scores and counts.
            assert [m.decoded.starts for m in keys] == [m.decoded.starts for m in sequences]
            assert [m.score for m in keys] == [m.score for m in sequences]
            assert [m.sequence.tolist() for m in keys] == [m.sequence.tolist() for m in sequences]
            assert (keys_tally.evaluations, keys_tally.moves) == (
                sequences_tally.evaluations,
                sequences_tally.moves,
            )
            assert keys_count == sequences_count
            best = [run.search.best for run in runs]
            assert best[0] is not None
            assert best[1] is not None
            assert best[0].decoded.starts == best[1].decoded.starts


@pytest.mark.parametrize("builder", BUILDERS)
def test_both_arms_run_alike_while_the_budget_ends_inside_the_initial_population(
    builder: BuilderName,
) -> None:
    problem = ta01().problem
    stop = StopCriteria(max_evaluations=60)
    arms = [ShopSteadyStateGa(arm, builder, 30, local_search=True) for arm in REPRESENTATIONS]
    keys, sequence = (arm.solve(problem, stop, seed=5) for arm in arms)
    # The decodes and descents of the first members use the whole budget: no child is bred.
    assert keys.iterations == sequence.iterations == 0
    assert _digest(problem, keys) == _digest(problem, sequence)


# ------------------------------------------------------------------------- the runs


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_the_memetic_algorithm_passes_the_contract(
    representation: RepresentationName, builder: BuilderName
) -> None:
    for problem in (small_job_shop(), ta01().problem):
        result = check_contract(
            ShopSteadyStateGa,
            problem,
            evaluations=200,
            representation=representation,
            builder=builder,
            population_size=4,
            local_search=True,
        )
        assert result.extras["observed_evaluations"] == result.evaluations


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_an_evaluation_limit_bounds_the_decodes_and_the_descents_together(
    representation: RepresentationName,
) -> None:
    problem = ta01().problem
    for evaluations in (1, 5, 37, 400, 900):
        result = _ga(representation).solve(problem, StopCriteria(max_evaluations=evaluations), 3)
        assert result.stop_reason is StopReason.EVALUATION_LIMIT
        assert result.evaluations == evaluations
        assert result.extras["observed_evaluations"] == evaluations
        descents = _extra(result, "descent_evaluations")
        assert 0 <= _extra(result, "descent_moves") <= descents <= evaluations - 1
        decodes = evaluations - descents
        if result.iterations:
            # The population of 6 and the children; the last iteration may end at its decode.
            assert decodes in (6 + result.iterations - 1, 6 + result.iterations)
        else:
            assert decodes <= 6
    long = _ga(representation).solve(problem, StopCriteria(max_evaluations=900), 3)
    assert long.iterations > 0
    assert _extra(long, "descent_moves") > 0


def test_the_memetic_algorithm_replays_from_its_seed() -> None:
    problem = ta01().problem
    stop = StopCriteria(max_evaluations=500)
    for representation in REPRESENTATIONS:
        solver = _ga(representation)
        first, second = (solver.solve(problem, stop, seed=4) for _ in range(2))
        assert _digest(problem, first) == _digest(problem, second)


@pytest.mark.parametrize("operations", [1, 2, 3])
def test_a_shop_of_one_job_has_one_schedule_and_the_run_stops_after_it(operations: int) -> None:
    for representation in REPRESENTATIONS:
        result = _ga(representation).solve(
            one_job(operations), StopCriteria(max_evaluations=30), seed=3
        )
        assert result.evaluations == 1
        assert result.stop_reason is StopReason.COMPLETED
        assert (_extra(result, "descent_evaluations"), _extra(result, "descent_moves")) == (0, 0)


def test_the_memetic_algorithm_runs_under_an_iteration_or_a_time_limit_alone() -> None:
    by_iterations = _ga("keys").solve(small_job_shop(), StopCriteria(max_iterations=3), 2)
    assert by_iterations.stop_reason is StopReason.ITERATION_LIMIT
    assert by_iterations.iterations == 3
    assert by_iterations.evaluations == 6 + 3 + _extra(by_iterations, "descent_evaluations")
    by_time = _ga("sequence").solve(ta01().problem, StopCriteria(time_limit=0.05), seed=2)
    assert by_time.stop_reason is StopReason.TIME_LIMIT
    assert by_time.evaluations >= 1


def test_a_run_that_reaches_its_target_inside_a_descent_ends_there() -> None:
    problem = ta01().problem
    free = _ga("keys").solve(problem, StopCriteria(max_evaluations=60), seed=5)
    assert free.value is not None
    target = free.value
    result = _ga("keys").solve(problem, StopCriteria(max_evaluations=60, target=target), seed=5)
    assert result.stop_reason is StopReason.TARGET_REACHED
    assert result.value == target
    assert result.evaluations <= 60
    assert result.extras["observed_evaluations"] == result.evaluations


# ------------------------------------------------------------------ the first solve

_FRESH = """
import json
import sys

from dispatchatlas.campaigns.execution import warm_up
from dispatchatlas.core import StopCriteria
from dispatchatlas.solvers.shop.genetic import ShopSteadyStateGa
from tests.helpers import small_job_shop

late = {}
for arm in ("keys", "sequence"):
    solver = ShopSteadyStateGa(arm, population_size=4, local_search=True)
    warm_up(solver)
    before = set(sys.modules)
    solver.solve(small_job_shop(), StopCriteria(max_evaluations=60), 2)
    late[arm] = sorted(set(sys.modules) - before)
print(json.dumps(late))
"""


def test_a_first_solve_in_a_fresh_interpreter_imports_no_module_after_the_warm_up() -> None:
    # A fresh interpreter: in this one, earlier tests have imported everything.
    done = subprocess.run(
        [sys.executable, "-c", _FRESH],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"keys": [], "sequence": []}
