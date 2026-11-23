"""The job-shop BRKGA: its parameters, write-back, local search, generations and runs."""

from __future__ import annotations

import json
import math
import pickle
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dispatchatlas.core import (
    Dependency,
    ModelError,
    Problem,
    Resource,
    SolveResult,
    StopCriteria,
    StopReason,
    Task,
    UnsupportedProblemError,
    derive_seed,
)
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.serialization import canonical_json, schedule_to_dict
from dispatchatlas.core.solver import SOLVER_FAMILIES
from dispatchatlas.references import bibliography
from dispatchatlas.solvers.population.brkga import Brkga
from dispatchatlas.solvers.registry import SolverRegistry
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.brkga import ShopBrkga, _Tally, adjust_keys, start_sequence
from dispatchatlas.solvers.shop.decoding import (
    BUILDERS,
    BuilderName,
    JobSequenceDecoder,
    giffler_thompson,
    semi_active,
)
from dispatchatlas.solvers.shop.graph import decoded_from_heads, heads_tails, machine_orders
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from dispatchatlas.solvers.shop.representation import KeyRepresentation, Keys
from dispatchatlas.solvers.shop.run import CONTROL_STREAM, OPERATOR_STREAM, Evaluated, ShopRun
from dispatchatlas.solvers.shop.sampling import ShopRandomSampling
from dispatchatlas.solvers.shop.tabu import Descent, critical_block_descent
from tests.helpers import chain_problem, random_problem, small_job_shop
from tests.solvers.shop.contract import check_contract
from tests.solvers.shop.families import Recording, one_job, untimed
from tests.solvers.shop.instances import random_job_shop, ta01

# ------------------------------------------------------------------------ helpers


def _shop(routes: Sequence[Sequence[tuple[int, float]]]) -> JobShop:
    """Return the job shop whose job ``j`` visits the machines and times of ``routes[j]``."""
    tasks: list[Task] = []
    dependencies: list[Dependency] = []
    for job, route in enumerate(routes):
        for step, (machine, duration) in enumerate(route):
            tasks.append(Task.on(f"j{job}-o{step}", f"m{machine}", duration))
            if step:
                dependencies.append(Dependency(f"j{job}-o{step - 1}", f"j{job}-o{step}"))
    machines = 1 + max(machine for route in routes for machine, _ in route)
    problem = Problem(
        "hand-shop",
        tasks=tuple(tasks),
        resources=tuple(Resource(f"m{machine}") for machine in range(machines)),
        dependencies=tuple(dependencies),
    )
    return JobShop.from_problem(problem)


def _ta01() -> Problem:
    return ta01().problem


def _serialised(problem: Problem, result: SolveResult) -> str:
    assert result.schedule is not None
    return canonical_json(schedule_to_dict(result.schedule, problem.id))


def _extra(result: SolveResult, key: str) -> int:
    value = result.extras[key]
    assert isinstance(value, int)
    return value


def _run(shop: JobShop, evaluations: int = 10**6, builder: BuilderName = "active") -> ShopRun[Keys]:
    """Return a run of ``shop`` on random keys with a budget that a test does not reach."""
    search = Search(shop.problem, StopCriteria(max_evaluations=evaluations), 0, ShopBrkga.info)
    return ShopRun(search, shop, builder, KeyRepresentation(shop))


def _keys(shop: JobShop, seed: int) -> Keys:
    return np.random.default_rng(seed).random(shop.size)


def _cases() -> list[JobShop]:
    """Small random shops that differ in size, release times and returns to a machine."""
    shops = []
    for index in range(40):
        problem = random_job_shop(
            index,
            jobs=3 + index % 5,
            machines=2 + index % 4,
            releases=index % 2 == 0,
            revisits=index % 3 == 0,
        )
        shops.append(JobShop.from_problem(problem))
    return shops


# ------------------------------------------------------------------ the parameters

# Gonçalves and Resende (2011), Table 1: an elite of 0.10 to 0.25 of the population, mutants of
# 0.10 to 0.30, and an inheritance above 0.5 and at most 0.8.
TABLE_1: list[dict[str, Any]] = [
    {"elite_fraction": 0.10, "mutant_fraction": 0.10, "inheritance": 0.55},
    {"elite_fraction": 0.25, "mutant_fraction": 0.30, "inheritance": 0.8},
    {"elite_fraction": 0.15, "mutant_fraction": 0.20, "inheritance": 0.7},
]


@pytest.mark.parametrize("parameters", TABLE_1)
@pytest.mark.parametrize("size", [30, 150, 225, 2000])
def test_the_ranges_of_the_2011_recommendations_are_admitted(
    parameters: dict[str, Any], size: int
) -> None:
    solver = ShopBrkga(population_size=size, **parameters)
    elite, mutants = solver.partition()
    generic = Brkga(population_size=size, **parameters)
    assert (elite, mutants) == generic.partition(size)
    assert elite >= 1
    assert size - elite - mutants >= 1


def test_the_partition_of_a_fraction_is_the_one_the_generic_algorithm_rounds_to() -> None:
    corners = [(0.10, 0.10), (0.25, 0.30)]
    solvers = [ShopBrkga(150, elite_fraction=e, mutant_fraction=m) for e, m in corners]
    assert [solver.partition() for solver in solvers] == [(15, 15), (38, 45)]
    for size in (3, 4, 7, 12):
        for elite in (0.05, 0.34, 0.9):
            for mutants in (0.0, 0.05, 0.09):
                solver = ShopBrkga(size, elite_fraction=elite, mutant_fraction=mutants)
                generic = Brkga(population_size=None, elite_fraction=elite, mutant_fraction=mutants)
                assert solver.partition() == generic.partition(size)


@pytest.mark.parametrize("size", [150, 151, 300, 1000])
def test_the_settings_of_the_2014_experiments_are_admitted_as_counts_and_as_fractions(
    size: int,
) -> None:
    # Gonçalves and Resende (2014), Table 2: 10 elite and 10 mutants, a population of at least
    # 150 and an inheritance of 0.85, which lie outside the ranges of the first source.
    by_count = ShopBrkga(size, elite_count=10, mutant_count=10, inheritance=0.85)
    by_fraction = ShopBrkga(size, elite_fraction=10 / size, mutant_fraction=10 / size)
    assert by_count.partition() == by_fraction.partition() == (10, 10)
    # The generic algorithm refuses none of them, so none of its checks is widened here.
    generic = Brkga(
        population_size=size, elite_fraction=10 / size, mutant_fraction=10 / size, inheritance=0.85
    )
    assert generic.partition(size) == (10, 10)


def test_the_defaults_are_the_settings_of_the_2014_experiments() -> None:
    solver = ShopBrkga()
    assert (solver.population_size, solver.partition(), solver.inheritance) == (150, (10, 10), 0.85)
    assert (solver.local_search, solver.write_back, solver.builder) == (True, True, "active")
    assert (solver.elite_count, solver.elite_fraction) == (None, None)


@pytest.mark.parametrize(
    ("size", "elite", "mutants"), [(2, 1, 0), (3, 1, 1), (12, 10, 1), (150, 10, 10), (50, 40, 9)]
)
def test_counts_are_used_as_given(size: int, elite: int, mutants: int) -> None:
    solver = ShopBrkga(size, elite_count=elite, mutant_count=mutants)
    assert solver.partition() == (elite, mutants)


def test_the_bounds_of_the_parameters_are_admitted() -> None:
    ShopBrkga(population_size=2, elite_count=1, mutant_count=0)
    ShopBrkga(elite_fraction=0.01, mutant_fraction=0.0)
    ShopBrkga(inheritance=0.5)
    ShopBrkga(inheritance=0.999)
    ShopBrkga(local_search=False, write_back=False, builder="semi-active")
    assert ShopBrkga(30, elite_count=3, mutant_fraction=0.2).partition() == (3, 6)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"builder": "giffler"}, "unknown builder"),
        ({"population_size": 1}, "population_size must be an integer of at least 2"),
        ({"population_size": 0}, "population_size"),
        ({"population_size": True}, "population_size"),
        ({"population_size": 150.0}, "population_size"),
        ({"elite_count": 0}, "elite_count must be an integer of at least 1"),
        ({"elite_count": True}, "elite_count"),
        ({"elite_count": 2.0}, "elite_count"),
        ({"mutant_count": -1}, "mutant_count must be an integer of at least 0"),
        ({"mutant_count": 1.5}, "mutant_count"),
        ({"elite_fraction": 0.0}, r"elite_fraction must lie in \(0, 1\)"),
        ({"elite_fraction": 1.0}, "elite_fraction"),
        ({"elite_fraction": "0.2"}, "elite_fraction"),
        ({"elite_fraction": math.nan}, "elite_fraction"),
        ({"mutant_fraction": -0.1}, r"mutant_fraction must lie in \[0, 1\)"),
        ({"mutant_fraction": 1.0}, "mutant_fraction"),
        ({"elite_count": 5, "elite_fraction": 0.1}, "elite_count or elite_fraction"),
        ({"mutant_count": 5, "mutant_fraction": 0.1}, "mutant_count or mutant_fraction"),
        ({"inheritance": 0.4}, r"inheritance must lie in \[0.5, 1\)"),
        ({"inheritance": 1.0}, "inheritance"),
        ({"inheritance": True}, "inheritance"),
        ({"inheritance": "0.8"}, "inheritance"),
        ({"inheritance": math.nan}, "inheritance"),
        ({"local_search": 1}, "local_search must be true or false"),
        ({"write_back": "yes"}, "write_back must be true or false"),
        ({"population_size": 20}, "no place for an offspring"),
        ({"population_size": 10, "elite_count": 6, "mutant_count": 4}, "no place for an offspring"),
        ({"elite_fraction": 0.6, "mutant_fraction": 0.4}, "no place for an offspring"),
        (
            {"population_size": 12, "elite_count": 10, "mutant_fraction": 0.2},
            "no place for an offspring",
        ),
    ],
)
def test_the_algorithm_refuses_bad_parameters(parameters: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShopBrkga(**parameters)


def test_the_algorithm_declares_what_it_models_and_cites_the_bibliography() -> None:
    info = ShopBrkga.info
    assert info.id == "shop-brkga"
    assert info.family in SOLVER_FAMILIES
    assert info.shape == "job-shop"
    assert info.supports == JOB_SHOP_FEATURES
    assert info.accepts(small_job_shop())
    assert not info.accepts(random_problem(2, tasks=8, max_modes=2, pools=1))
    assert info.summary.endswith(".")
    assert {"goncalves2011biased", "goncalves2014extended"} <= set(info.references)
    assert set(info.references) <= set(bibliography())


def test_the_algorithm_registers_and_is_created_by_its_id() -> None:
    registry = SolverRegistry()
    registry.register(ShopBrkga)
    solver = registry.create("shop-brkga", population_size=30, elite_count=3, mutant_count=3)
    assert solver == ShopBrkga(30, elite_count=3, mutant_count=3)
    assert set(registry.info("shop-brkga").parameters) == {
        "population_size",
        "elite_fraction",
        "elite_count",
        "mutant_fraction",
        "mutant_count",
        "inheritance",
        "local_search",
        "write_back",
        "builder",
    }
    with pytest.raises(ConfigError, match="inheritance"):
        registry.create("shop-brkga", inheritance=1.0)
    with pytest.raises(UnsupportedProblemError):
        solver.solve(
            random_problem(2, tasks=8, max_modes=2, pools=1), StopCriteria(max_evaluations=5)
        )


def test_the_algorithm_is_a_value_that_compares_hashes_and_pickles() -> None:
    parameters: dict[str, Any] = {
        "population_size": 30,
        "elite_fraction": 0.2,
        "mutant_count": 3,
        "builder": "non-delay",
    }
    solver = ShopBrkga(**parameters)
    assert pickle.loads(pickle.dumps(solver)) == solver
    assert hash(solver) == hash(ShopBrkga(**parameters))
    assert solver != ShopBrkga(**parameters | {"write_back": False})


# ------------------------------------------------------- the job sequence of a schedule


def _trap() -> tuple[JobShop, list[list[int]]]:
    """Return a shop and machine orders whose schedule is semi-active but not active.

    Job 0 runs q (machine 1, 10), o (machine 0, 1) and t (machine 2, 100), job 1 is p
    (machine 0, 20), and job 2 is x (machine 0, 5). Machine 0 runs o, p, x: o starts at 10 when
    q is done, p at 11 and x at 31, and t starts at 11, so the makespan is 111. Operation x
    could have started at 0, so the schedule is not active. The tasks are q, o, t, p, x.
    """
    shop = _shop([[(1, 10.0), (0, 1.0), (2, 100.0)], [(0, 20.0)], [(0, 5.0)]])
    return shop, [[1, 3, 4], [0], [2]]


def _sequences(shop: JobShop, count: int) -> list[Any]:
    """Return ``count`` random job sequences of ``shop``."""
    rng = np.random.default_rng(derive_seed(5, "tests/shop/brkga", shop.size))
    return [rng.permutation(shop.slot_job) for _ in range(count)]


def _heads(shop: JobShop, orders: Sequence[Sequence[int]]) -> tuple[float, ...]:
    result = heads_tails(shop, orders)
    assert result is not None
    return result[0]


def test_the_sequence_lists_the_operations_by_start_time_then_machine_then_job() -> None:
    shop, orders = _trap()
    assert _heads(shop, orders) == (0.0, 10.0, 11.0, 11.0, 31.0)
    # q starts at 0 and o at 10. Then p and t start together at 11, and the machine breaks the
    # tie, 0 before 2, though job 0 comes before job 1. Last comes x at 31.
    assert start_sequence(shop, orders).tolist() == [0, 0, 1, 0, 2]


def test_the_semi_active_builder_decodes_the_sequence_into_the_schedule_of_the_orders() -> None:
    shop, orders = _trap()
    heads = _heads(shop, orders)
    rebuilt = semi_active(shop, start_sequence(shop, orders))
    assert rebuilt.starts == heads == decoded_from_heads(shop, heads).starts
    assert max(rebuilt.ends) == 111.0


def test_the_sequence_is_the_plain_sort_when_every_operation_takes_time() -> None:
    # The sort by start time, then machine, then job, which the sequence follows unless an
    # operation that takes no time makes it put an operation before one that precedes it.
    for shop in _cases():
        for index, sequence in enumerate(_sequences(shop, 3)):
            orders = machine_orders(shop, semi_active(shop, sequence))
            starts = np.asarray(_heads(shop, orders))
            order = np.lexsort((np.asarray(shop.job), np.asarray(shop.machine), starts))
            expected = np.asarray(shop.job)[order]
            assert start_sequence(shop, orders).tolist() == expected.tolist(), index


def test_an_operation_that_takes_no_time_comes_before_the_operation_it_precedes() -> None:
    # A (job 1, no time) comes before B (job 0, 5) on machine 0, and both start at 0. Sorted by
    # start, machine and job alone, B comes first, and the semi-active builder then starts A at 5.
    shop = _shop([[(0, 5.0)], [(0, 0.0)]])
    orders = [[1, 0]]
    assert start_sequence(shop, orders).tolist() == [1, 0]
    assert semi_active(shop, [1, 0]).starts == (0.0, 0.0)
    assert semi_active(shop, [0, 1]).starts == (0.0, 5.0)
    plain = np.lexsort(
        (np.asarray(shop.job), np.asarray(shop.machine), np.asarray(_heads(shop, orders)))
    )
    assert np.asarray(shop.job)[plain].tolist() == [0, 1]


def _zero_shops() -> list[JobShop]:
    """Return random shops in which about a third of the operations take no time."""
    shops = []
    for index in range(30):
        rng = np.random.default_rng(derive_seed(5, "tests/shop/brkga/zero", index))
        jobs, machines = 3 + index % 3, 2 + index % 2
        routes = [
            [
                (int(rng.integers(machines)), float(rng.choice([0, 0, 1, 2])))
                for _ in range(machines)
            ]
            for _ in range(jobs)
        ]
        shops.append(_shop(routes))
    return shops


def test_the_sequence_reproduces_the_schedule_when_operations_take_no_time() -> None:
    checked = 0
    for shop in _zero_shops():
        for sequence in _sequences(shop, 4):
            orders = machine_orders(shop, semi_active(shop, sequence))
            heads = _heads(shop, orders)
            written = start_sequence(shop, orders)
            assert np.array_equal(np.sort(written), shop.slot_job)
            assert semi_active(shop, written).starts == heads
            checked += 1
    assert checked == 120


def test_the_sequence_refuses_orders_that_are_not_a_schedule() -> None:
    # Job 0 runs a on machine 0 and then b on machine 1, job 1 runs c on machine 1 and then d on
    # machine 0, and the orders d, a and b, c close a cycle with the jobs.
    crossed = _shop([[(0, 1.0), (1, 1.0)], [(1, 1.0), (0, 1.0)]])
    with pytest.raises(ModelError, match="cycle"):
        start_sequence(crossed, [[3, 0], [1, 2]])
    shop, _ = _trap()
    with pytest.raises(ModelError, match="list 4 of the 5 operations"):
        start_sequence(shop, [[1, 3], [0], [2]])


# ------------------------------------------------------------- the keys of a sequence


def test_adjusted_keys_sort_into_the_sequence_and_keep_their_values() -> None:
    for shop in (ta01(), *_cases()[:10]):
        representation = KeyRepresentation(shop)
        for index, sequence in enumerate(_sequences(shop, 3)):
            keys = _keys(shop, index)
            before = keys.copy()
            adjusted = adjust_keys(representation, keys, sequence)
            assert np.array_equal(representation.sequence(adjusted), sequence)
            assert np.array_equal(np.sort(adjusted), np.sort(keys))
            assert np.array_equal(keys, before)


def test_keys_that_already_sort_into_the_sequence_stay_where_they_are() -> None:
    shop = ta01()
    representation = KeyRepresentation(shop)
    keys = _keys(shop, 4)
    adjusted = adjust_keys(representation, keys, representation.sequence(keys))
    assert np.array_equal(adjusted, keys)


def test_equal_keys_are_moved_apart_so_that_any_sequence_can_be_reached() -> None:
    shop = _shop([[(0, 1.0), (1, 1.0)], [(0, 1.0)], [(1, 1.0)]])
    representation = KeyRepresentation(shop)
    assert shop.slot_job.tolist() == [0, 0, 1, 2]
    keys = np.full(4, 0.5)
    # Equal keys sort by slot, so job 2 cannot come before job 1 and job 0 until they differ.
    wanted = [2, 1, 0, 0]
    assert representation.sequence(representation.follow(keys, wanted)).tolist() == [0, 0, 1, 2]
    adjusted = adjust_keys(representation, keys, wanted)
    assert representation.sequence(adjusted).tolist() == wanted
    assert len(set(adjusted.tolist())) == 4
    assert np.abs(adjusted - 0.5).max() < 1e-12


def test_keys_that_are_one_unit_apart_in_the_last_place_are_moved_apart_in_turn() -> None:
    shop = _shop([[(0, 1.0), (1, 1.0)], [(0, 1.0)], [(1, 1.0)]])
    representation = KeyRepresentation(shop)
    keys = np.array([0.5, 0.5, np.nextafter(0.5, 1.0), np.nextafter(0.5, 1.0)])
    adjusted = adjust_keys(representation, keys, [2, 0, 1, 0])
    assert representation.sequence(adjusted).tolist() == [2, 0, 1, 0]
    assert np.all(np.diff(np.sort(adjusted)) > 0)


def test_a_key_with_many_equal_values_still_sorts_into_any_sequence() -> None:
    shop = ta01()
    representation = KeyRepresentation(shop)
    keys = np.random.default_rng(3).integers(1, 6, shop.size) / 10.0
    for sequence in _sequences(shop, 3):
        adjusted = adjust_keys(representation, keys, sequence)
        assert np.array_equal(representation.sequence(adjusted), sequence)
        assert np.abs(np.sort(adjusted) - np.sort(keys)).max() < 1e-12


def test_adjusting_refuses_what_is_not_a_chromosome_or_a_sequence_of_the_shop() -> None:
    shop = _shop([[(0, 1.0), (1, 1.0)], [(0, 1.0)]])
    representation = KeyRepresentation(shop)
    with pytest.raises(ModelError, match="not a job sequence"):
        adjust_keys(representation, np.array([0.1, 0.2, 0.3]), [0, 0, 0])
    with pytest.raises(ValueError, match="3 keys"):
        adjust_keys(representation, np.array([0.1, 0.2]), [0, 0, 1])


# ------------------------------------------------------------------ the write-back


def _improved(shop: JobShop, keys: Keys) -> tuple[KeyRepresentation, Descent]:
    """Decode a chromosome with the active builder and improve its schedule by the descent."""
    representation = KeyRepresentation(shop)
    decoded = giffler_thompson(shop, representation.sequence(keys))
    return representation, critical_block_descent(shop, machine_orders(shop, decoded), _run(shop))


def test_the_semi_active_builder_decodes_the_written_back_keys_into_the_improved_schedule() -> None:
    cases = [(shop, index) for shop in _cases() for index in range(3)]
    cases += [(ta01(), index) for index in range(12)]
    moved = longer = 0
    for shop, index in cases:
        keys = _keys(shop, index)
        representation, descent = _improved(shop, keys)
        sequence = start_sequence(shop, descent.orders)
        written = adjust_keys(representation, keys, sequence)
        assert np.array_equal(representation.sequence(written), sequence)
        rebuilt = semi_active(shop, representation.sequence(written))
        assert rebuilt.starts == descent.decoded.starts
        assert rebuilt.ends == descent.decoded.ends
        moved += descent.moves > 0
        # The active builder gives no such promise: it rebuilds a schedule that is sometimes
        # longer than the improved one, which the algorithm therefore never relies on.
        longer += max(giffler_thompson(shop, representation.sequence(written)).ends) > (
            descent.makespan
        )
    assert len(cases) == 132
    assert moved > len(cases) // 2
    assert longer > 0


def test_the_active_builder_can_give_a_longer_schedule_than_the_improved_one() -> None:
    # The write-back promises an exact rebuild to the semi-active builder only. The improved
    # schedule of the trap is not active, and the active builder puts p ahead of x, which then
    # holds machine 0 when o is ready, so the makespan grows from 111 to 121.
    shop, orders = _trap()
    representation = KeyRepresentation(shop)
    sequence = start_sequence(shop, orders)
    written = adjust_keys(representation, _keys(shop, 1), sequence)
    keys_sequence = representation.sequence(written)
    assert max(semi_active(shop, keys_sequence).ends) == 111.0
    assert max(giffler_thompson(shop, keys_sequence).ends) == 121.0


# ----------------------------------------------------------------- the local search

# A population small enough for a few dozen evaluations to breed many children from it.
SMALL: dict[str, Any] = {"population_size": 8, "elite_count": 2, "mutant_count": 2}


def _member(solver: ShopBrkga, run: ShopRun[Keys], keys: Keys) -> tuple[Evaluated[Keys], _Tally]:
    """Decode and improve one chromosome as a generation does, and return what it counted."""
    tally = _Tally()
    return solver._member(run, keys, None, tally), tally


def _cases_with_chromosomes() -> list[tuple[JobShop, int]]:
    cases = [(shop, index) for shop in _cases() for index in range(2)]
    return cases + [(ta01(), index) for index in range(6)]


def test_the_local_search_never_worsens_a_chromosome_and_counts_its_swaps() -> None:
    solver = ShopBrkga(**SMALL)
    improved = 0
    for shop, index in _cases_with_chromosomes():
        keys = _keys(shop, index)
        plain = giffler_thompson(shop, KeyRepresentation(shop).sequence(keys))
        run = _run(shop)
        member, tally = _member(solver, run, keys)
        assert member.score.value <= max(plain.ends)
        assert member.score.value == max(member.decoded.ends)
        # One evaluation for the decode and one for each swap, and the best schedule of the
        # run is the improved one.
        assert run.search.evaluations == 1 + tally.moves
        assert run.search.best is not None
        assert run.search.best.score == member.score
        again = critical_block_descent(shop, machine_orders(shop, plain), _run(shop))
        assert tally.moves == again.moves
        assert member.decoded.starts == again.decoded.starts
        improved += member.score.value < max(plain.ends)
    assert improved > 20


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_sequence_of_a_member_is_the_one_its_keys_sort_into(builder: BuilderName) -> None:
    for shop, index in _cases_with_chromosomes()[::4]:
        keys = _keys(shop, index)
        member, _ = _member(ShopBrkga(builder=builder, **SMALL), _run(shop, builder=builder), keys)
        representation = KeyRepresentation(shop)
        assert member.sequence.tolist() == representation.sequence(member.genotype).tolist()
        if builder == "semi-active":
            # The semi-active builder rebuilds the schedule of the member from its keys.
            rebuilt = JobSequenceDecoder(shop, builder).decode(member.sequence)
            assert (rebuilt.starts, rebuilt.ends) == (member.decoded.starts, member.decoded.ends)


def test_without_the_write_back_the_keys_stay_and_the_score_is_the_same() -> None:
    kept = ShopBrkga(write_back=False, **SMALL)
    adjusted = ShopBrkga(**SMALL)
    moved = 0
    for shop, index in _cases_with_chromosomes():
        keys = _keys(shop, index)
        before = keys.copy()
        member, _ = _member(kept, _run(shop), keys)
        other, _ = _member(adjusted, _run(shop), keys)
        assert member.genotype is keys
        assert np.array_equal(keys, before)
        assert member.sequence.tolist() == KeyRepresentation(shop).sequence(keys).tolist()
        assert (member.score, member.decoded.starts) == (other.score, other.decoded.starts)
        moved += not np.array_equal(other.genotype, keys)
    assert moved > 20


@pytest.mark.parametrize("write_back", [True, False])
def test_without_the_local_search_a_chromosome_is_its_decode(write_back: bool) -> None:
    solver = ShopBrkga(local_search=False, write_back=write_back, **SMALL)
    for shop, index in _cases_with_chromosomes()[::3]:
        keys = _keys(shop, index)
        plain = giffler_thompson(shop, KeyRepresentation(shop).sequence(keys))
        run = _run(shop)
        member, tally = _member(solver, run, keys)
        assert member.genotype is keys
        assert (run.search.evaluations, tally.moves) == (1, 0)
        assert member.decoded.starts == plain.starts
        assert member.score.value == max(plain.ends)


def test_a_descent_that_raises_the_objective_leaves_the_chromosome_its_decoded_schedule() -> None:
    # Job a (5, due 5) comes first on machine 0 and finishes on time. The descent steers by the
    # makespan, and swapping a with b, the first operation of the long job, lowers the makespan
    # from 110 to 105 but finishes a at 10, five late.
    tasks = (
        Task.on("a", "m0", 5.0, due=5.0),
        Task.on("b", "m0", 5.0),
        Task.on("t", "m1", 100.0),
    )
    problem = Problem(
        "tardy",
        tasks=tasks,
        resources=(Resource("m0"), Resource("m1")),
        dependencies=(Dependency("b", "t"),),
    )
    keys = np.array([0.1, 0.5, 0.6])
    for objective, value, kept in (("makespan", 105.0, False), ("total_tardiness", 0.0, True)):
        shop = JobShop.from_problem(problem.with_objectives(objective))
        run = _run(shop)
        member, tally = _member(ShopBrkga(**SMALL), run, keys)
        assert tally.moves == 1
        assert member.score.value == value
        assert (member.genotype is keys) == kept
        assert run.search.best is not None
        assert run.search.best.score == member.score


# ------------------------------------------------------------------- the generations


def _ranked(population: Sequence[Evaluated[Keys]]) -> list[Evaluated[Keys]]:
    """Return the members best first, equal scores in the order of the population."""
    return sorted(population, key=lambda member: (member.score.violation, member.score.value))


def test_the_elite_are_copied_unchanged_and_never_worsen_across_generations() -> None:
    shop = JobShop.from_problem(random_job_shop(3, jobs=6, machines=4))
    solver = ShopBrkga(population_size=12, elite_count=3, mutant_count=2)
    run, tally = _run(shop), _Tally()
    population = solver._initial(run, tally)
    assert len(population) == 12
    best = _ranked(population)[0].score
    for _ in range(8):
        evaluations, moves = run.search.evaluations, tally.moves
        following = solver._generation(run, population, tally)
        assert len(following) == 12
        # The elite are the same members, not decoded again, and come first.
        assert all(a is b for a, b in zip(following[:3], _ranked(population)[:3], strict=True))
        # The other nine places hold new chromosomes, each decoded once and then improved.
        assert run.search.evaluations - evaluations - (tally.moves - moves) == 9
        assert _ranked(following)[0].score <= best
        best = _ranked(following)[0].score
        population = following
    assert run.search.best is not None
    assert run.search.best.score <= best


def test_a_generation_decodes_mutants_alone_and_offspring_beside_an_elite_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = Recording(monkeypatch)
    solver = ShopBrkga(population_size=12, elite_count=3, mutant_count=2, local_search=False)
    run, tally = _run(ta01()), _Tally()
    population = solver._initial(run, tally)
    solver._generation(run, population, tally)
    evaluations = recording.evaluations[run]
    assert len(evaluations) == 12 + 9
    assert all(evaluation.parent is None for evaluation in evaluations[:14])
    elite = {member.digest: member.score.value for member in _ranked(population)[:3]}
    assert len(elite) == 3
    offspring = evaluations[14:]
    assert len(offspring) == 7
    for evaluation in offspring:
        assert evaluation.parent is not None
        assert evaluation.parent_value == elite[evaluation.parent]


def test_each_offspring_crosses_an_elite_parent_with_a_non_elite_one_in_favour_of_the_elite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Keys, Keys, Keys]] = []
    recombine = KeyRepresentation.recombine

    def recorded(
        self: KeyRepresentation, first: Keys, second: Keys, rng: np.random.Generator, bias: float
    ) -> Keys:
        child = recombine(self, first, second, rng, bias)
        calls.append((first, second, child))
        return child

    monkeypatch.setattr(KeyRepresentation, "recombine", recorded)
    solver = ShopBrkga(
        population_size=60, elite_count=6, mutant_count=6, inheritance=0.85, local_search=False
    )
    run, tally = _run(ta01()), _Tally()
    population = solver._initial(run, tally)
    solver._generation(run, population, tally)
    elite = {id(member.genotype) for member in _ranked(population)[:6]}
    assert len(calls) == 48
    assert all(id(first) in elite and id(second) not in elite for first, second, _ in calls)
    assert {id(first) for first, _, _ in calls} == elite
    assert all(((child == first) | (child == second)).all() for first, second, child in calls)
    taken = sum(int((child == first).sum()) for first, _, child in calls)
    assert abs(taken / (48 * ta01().size) - 0.85) < 0.02


def test_the_streams_are_read_by_the_population_the_mutants_the_crossings_and_the_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = Recording(monkeypatch)
    shop = JobShop.from_problem(small_job_shop())
    result = ShopBrkga(**SMALL).solve(shop.problem, StopCriteria(max_iterations=2), seed=4)
    assert result.stop_reason is StopReason.ITERATION_LIMIT
    assert _extra(result, "descent_moves") > 0
    (run,) = recording.finished
    # The whole initial population first, then, in each generation, the draws of the parents
    # from the control stream, and the mutants and the four crossings from the operator stream.
    # The descent and the write-back draw nothing.
    operators = np.random.default_rng(derive_seed(4, OPERATOR_STREAM))
    control = np.random.default_rng(derive_seed(4, CONTROL_STREAM))
    for _ in range(8):
        operators.random(shop.size)
    for _ in range(2):
        control.integers(2, size=4)
        control.integers(6, size=4)
        for _ in range(2 + 4):
            operators.random(shop.size)
    assert run.operators.bit_generator.state == operators.bit_generator.state
    assert run.control.bit_generator.state == control.bit_generator.state


# ----------------------------------------------------------------------- the solver


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_algorithm_passes_the_contract(builder: BuilderName) -> None:
    problems = [small_job_shop(), chain_problem(), _ta01()]
    problems += [random_job_shop(index, revisits=index % 2 == 0) for index in range(8)]
    for problem in problems:
        result = check_contract(ShopBrkga, problem, builder=builder, **SMALL)
        assert result.extras["repairs"] == 0
        assert result.extras["observed_evaluations"] == result.evaluations == 40
        # Every evaluation is a decoded chromosome or a swap of a descent.
        assert result.evaluations == _extra(result, "decodes") + _extra(result, "descent_moves")
        assert (_extra(result, "elite_members"), _extra(result, "mutant_members")) == (2, 2)


@pytest.mark.parametrize(
    ("local_search", "write_back"), [(False, True), (True, False), (False, False)]
)
def test_the_variants_without_the_local_search_or_the_write_back_pass_the_contract(
    local_search: bool, write_back: bool
) -> None:
    for problem in (small_job_shop(), chain_problem(), _ta01()):
        result = check_contract(
            ShopBrkga, problem, local_search=local_search, write_back=write_back, **SMALL
        )
        assert result.extras["observed_evaluations"] == result.evaluations == 40
        if not local_search:
            assert (_extra(result, "decodes"), _extra(result, "descent_moves")) == (40, 0)


def test_the_algorithm_passes_the_contract_when_operations_take_no_time() -> None:
    for shop in _zero_shops()[:12]:
        for builder in BUILDERS:
            result = check_contract(
                ShopBrkga, shop.problem, evaluations=60, builder=builder, **SMALL
            )
            assert result.extras["repairs"] == 0


def test_the_algorithm_replays_from_its_seed() -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=300)
    solver = ShopBrkga(population_size=10, elite_count=2, mutant_count=2)
    first, second, other = (solver.solve(problem, stop, seed=seed) for seed in (5, 5, 6))
    assert first.iterations > 1
    assert _serialised(problem, first) == _serialised(problem, second)
    assert [(p.evaluations, p.iteration, p.value) for p in first.trace] == [
        (p.evaluations, p.iteration, p.value) for p in second.trace
    ]
    assert untimed(first.extras) == untimed(second.extras)
    assert first.extras["first_schedule_digest"] != other.extras["first_schedule_digest"]
    assert _serialised(problem, first) != _serialised(problem, other)


@pytest.mark.parametrize("seed", [7, 8])
def test_the_algorithm_beats_random_sampling_on_ta01_at_800_evaluations(seed: int) -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=800)
    solver = ShopBrkga(population_size=20, elite_count=3, mutant_count=3)
    brkga = solver.solve(problem, stop, seed=seed)
    sampling = ShopRandomSampling("keys", "active").solve(problem, stop, seed=seed)
    assert brkga.evaluations == sampling.evaluations == 800
    assert brkga.stop_reason is sampling.stop_reason is StopReason.EVALUATION_LIMIT
    assert brkga.value is not None
    assert sampling.value is not None
    assert brkga.value < sampling.value
    assert brkga.value <= 1450.0


def test_the_first_evaluation_is_the_chromosome_that_random_sampling_draws_first() -> None:
    stop = StopCriteria(max_evaluations=12)
    for problem in (small_job_shop(), _ta01()):
        brkga = ShopBrkga(**SMALL).solve(problem, stop, seed=3)
        sampling = ShopRandomSampling("keys", "active").solve(problem, stop, seed=3)
        assert brkga.extras["first_schedule_digest"] == sampling.extras["first_schedule_digest"]
        assert brkga.trace[0].value == sampling.trace[0].value


def test_the_observer_sees_every_evaluation_and_the_swaps_and_crossings_as_moves() -> None:
    result = ShopBrkga(**SMALL).solve(_ta01(), StopCriteria(max_evaluations=150), seed=2)
    assert result.extras["observed_evaluations"] == 150
    # Every swap is a move of its schedule, and so is every offspring of its elite parent.
    assert _extra(result, "observed_moves") >= _extra(result, "descent_moves")
    assert 0.0 < float(result.extras["phenotype_distance_mean"]) < 100.0


@pytest.mark.parametrize("operations", [1, 2, 3])
def test_a_shop_of_one_job_has_one_schedule_and_the_run_stops_after_it(operations: int) -> None:
    result = ShopBrkga(**SMALL).solve(one_job(operations), StopCriteria(max_evaluations=30), seed=3)
    assert result.evaluations == 1
    assert result.stop_reason is StopReason.COMPLETED
    assert result.value == sum(step + 2 for step in range(operations))
    assert (_extra(result, "decodes"), _extra(result, "descent_moves")) == (1, 0)


def test_the_algorithm_runs_under_an_iteration_or_a_time_limit_alone() -> None:
    by_iterations = ShopBrkga(**SMALL).solve(small_job_shop(), StopCriteria(max_iterations=3), 2)
    assert by_iterations.stop_reason is StopReason.ITERATION_LIMIT
    assert by_iterations.iterations == 3
    # The initial population of 8, and the 6 new chromosomes of each of three generations.
    assert _extra(by_iterations, "decodes") == 8 + 3 * 6
    by_time = ShopBrkga(**SMALL).solve(_ta01(), StopCriteria(time_limit=0.05), seed=2)
    assert by_time.stop_reason is StopReason.TIME_LIMIT
    assert by_time.evaluations >= 1


def test_the_time_limit_is_checked_before_every_swap_of_the_descent() -> None:
    problem = _ta01()
    solver = ShopBrkga(**SMALL)
    free = solver.solve(problem, StopCriteria(max_evaluations=20), seed=1)
    assert _extra(free, "descent_moves") > 0
    # With no time left, the first swap is refused: only the first decode is counted.
    late = solver.solve(problem, StopCriteria(time_limit=1e-6), seed=1)
    assert late.stop_reason is StopReason.TIME_LIMIT
    assert (late.evaluations, _extra(late, "descent_moves")) == (1, 0)
    by_time = ShopBrkga(population_size=20, elite_count=3, mutant_count=3).solve(
        problem, StopCriteria(time_limit=0.3), seed=1
    )
    assert by_time.stop_reason is StopReason.TIME_LIMIT
    assert 0.3 <= by_time.elapsed < 1.5


def test_a_run_that_reaches_its_target_ends_there_and_the_observer_saw_that_schedule() -> None:
    stop = StopCriteria(max_evaluations=2000, target=1400.0)
    solver = ShopBrkga(population_size=20, elite_count=3, mutant_count=3)
    result = solver.solve(_ta01(), stop, seed=7)
    assert result.stop_reason is StopReason.TARGET_REACHED
    assert result.value is not None
    assert result.value <= 1400.0
    assert 1 < result.evaluations < 2000
    assert result.extras["observed_evaluations"] == result.evaluations
    assert result.evaluations == _extra(result, "decodes") + _extra(result, "descent_moves")


# ------------------------------------------------------------------ the first solve

_FRESH = """
import json
import sys

from dispatchatlas.campaigns.execution import warm_up
from dispatchatlas.core import StopCriteria
from dispatchatlas.solvers.shop.brkga import ShopBrkga
from tests.helpers import chain_problem, small_job_shop

solver = ShopBrkga(population_size=6, elite_count=1, mutant_count=1)
warm_up(solver)
late = {}
for name, problem in (("chain", chain_problem()), ("job-shop", small_job_shop())):
    before = set(sys.modules)
    solver.solve(problem, StopCriteria(max_evaluations=30), 2)
    late[name] = sorted(set(sys.modules) - before)
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
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"chain": [], "job-shop": []}
