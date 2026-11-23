"""The job-shop tabu search: its neighbourhoods, its estimate, its descent, and its walk."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable, Sequence
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
    derive_seed,
)
from dispatchatlas.core.serialization import canonical_json, schedule_to_dict
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.search import Candidate, Exhausted, Search
from dispatchatlas.solvers.shop.decoding import JobSequenceDecoder
from dispatchatlas.solvers.shop.family import make_representation
from dispatchatlas.solvers.shop.graph import critical_blocks, heads_tails, machine_orders
from dispatchatlas.solvers.shop.instance import JobShop
from dispatchatlas.solvers.shop.observers import TrajectoryObserver
from dispatchatlas.solvers.shop.representation import SequenceRepresentation
from dispatchatlas.solvers.shop.run import CONTROL_STREAM, OPERATOR_STREAM, ShopRun
from dispatchatlas.solvers.shop.sampling import ShopRandomSampling
from dispatchatlas.solvers.shop.tabu import (
    NEIGHBOURHOODS,
    NeighbourhoodName,
    ShopTabuSearch,
    _Candidate,
    _TabuList,
    critical_block_descent,
    n5_moves,
    n7_moves,
    swap_estimate,
)
from tests.helpers import chain_problem, small_job_shop
from tests.solvers.shop.contract import check_contract
from tests.solvers.shop.families import Recording, one_job, untimed
from tests.solvers.shop.instances import random_job_shop, random_sequences, ta01

# ----------------------------------------------------------------- a known shop

# Machine A processes x1 (5), x2 (4), x3 (3), machine B y1 (6), y2 (2), and machine C
# z1 (5), z2 (4), z3 (3). Job x3-y1 and job y2-z1 are the only jobs of two operations.
# Under the orders A = x1 x2 x3, B = y1 y2, C = z1 z2 z3 the heads are 0, 5, 9, 12, 18,
# 20, 25, 29, the makespan is 32, and the one critical path has the blocks
# (x1 x2 x3), (y1 y2) and (z1 z2 z3): a first block, an inner block, and a last block.
NAMES = ("x1", "x2", "x3", "y1", "y2", "z1", "z2", "z3")
TIMES = (5.0, 4.0, 3.0, 6.0, 2.0, 5.0, 4.0, 3.0)
MACHINES = ("A", "A", "A", "B", "B", "C", "C", "C")
ORDERS = [[0, 1, 2], [3, 4], [5, 6, 7]]


def _known() -> JobShop:
    tasks = tuple(
        Task.on(name, machine, time)
        for name, machine, time in zip(NAMES, MACHINES, TIMES, strict=True)
    )
    problem = Problem(
        "known",
        tasks=tasks,
        resources=(Resource("A"), Resource("B"), Resource("C")),
        dependencies=(Dependency("x3", "y1"), Dependency("y2", "z1")),
    )
    return JobShop.from_problem(problem)


def _swapped(orders: Sequence[Sequence[int]], machine: int, place: int) -> list[list[int]]:
    changed = [list(order) for order in orders]
    changed[machine][place : place + 2] = [changed[machine][place + 1], changed[machine][place]]
    return changed


def _heads_tails(
    shop: JobShop, orders: Sequence[Sequence[int]]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    found = heads_tails(shop, orders)
    assert found is not None
    return found


def _makespan(shop: JobShop, orders: Sequence[Sequence[int]]) -> float | None:
    result = heads_tails(shop, orders)
    if result is None:
        return None
    return max(head + time for head, time in zip(result[0], shop.duration, strict=True))


def test_the_known_shop_has_the_critical_path_the_tests_rely_on() -> None:
    shop = _known()
    heads, tails = _heads_tails(shop, ORDERS)
    assert heads == (0.0, 5.0, 9.0, 12.0, 18.0, 20.0, 25.0, 29.0)
    assert tails == (27.0, 23.0, 20.0, 14.0, 12.0, 7.0, 3.0, 0.0)
    assert critical_blocks(shop, ORDERS, heads) == ((0, 1, 2), (3, 4), (5, 6, 7))


# --------------------------------------------------------------------------- N5


def test_n5_swaps_the_inner_ends_of_the_blocks_of_a_critical_path() -> None:
    shop = _known()
    heads, _ = _heads_tails(shop, ORDERS)
    blocks = critical_blocks(shop, ORDERS, heads)
    # The first block gives its last pair only, the inner block its one pair, and the
    # last block its first pair only: x2 x3 on A, y1 y2 on B, z1 z2 on C.
    assert n5_moves(shop, ORDERS, blocks) == [(0, 1, 2), (1, 0, 1), (2, 0, 1)]


@pytest.mark.parametrize(
    ("blocks", "moves"),
    [
        # A first block of two operations keeps its swap, as its last pair.
        (((1, 2), (3, 4)), [(0, 1, 2), (1, 0, 1)]),
        # A last block of two keeps its swap, as its first pair.
        (((3, 4), (6, 7)), [(1, 0, 1), (2, 1, 2)]),
        # A path of one block has no move, whatever its size.
        (((0, 1, 2),), []),
        (((6, 7),), []),
        # Blocks of one operation give nothing.
        (((0,), (3,), (5,)), []),
        # An inner block of three gives its first and its last pair.
        (((3,), (0, 1, 2), (7,)), [(0, 0, 1), (0, 1, 2)]),
    ],
)
def test_n5_applies_its_rule_to_blocks_of_every_position_and_size(
    blocks: tuple[tuple[int, ...], ...], moves: list[tuple[int, int, int]]
) -> None:
    shop = _known()
    assert n5_moves(shop, ORDERS, blocks) == moves


def _returning() -> JobShop:
    """A job whose two operations, a and b, run on machine M one after the other."""
    problem = Problem(
        "returning",
        tasks=(
            Task.on("c", "N", 1),
            Task.on("a", "M", 2),
            Task.on("b", "M", 3),
            Task.on("e", "N", 1),
        ),
        resources=(Resource("M"), Resource("N")),
        dependencies=(Dependency("a", "b"),),
    )
    return JobShop.from_problem(problem)


def test_n5_leaves_out_a_swap_that_the_job_also_joins() -> None:
    shop = _returning()
    orders = [[1, 2], [0, 3]]
    assert n5_moves(shop, orders, ((0,), (1, 2), (3,))) == []
    assert swap_estimate(shop, orders[0], 0, [0.0] * 4, [0.0] * 4) == math.inf
    assert _makespan(shop, _swapped(orders, 0, 0)) is None


# ---------------------------------------------------------------------- the estimate


@pytest.mark.parametrize(
    ("machine", "place", "estimate"), [(0, 1, 28.0), (1, 0, 18.0), (2, 0, 28.0)]
)
def test_the_estimate_of_a_swap_of_the_known_shop_is_worked_out_by_hand(
    machine: int, place: int, estimate: float
) -> None:
    shop = _known()
    heads, tails = _heads_tails(shop, ORDERS)
    assert swap_estimate(shop, ORDERS[machine], place, heads, tails) == estimate
    # Here no longest path avoids both operations, so the estimate is the makespan.
    assert _makespan(shop, _swapped(ORDERS, machine, place)) == estimate


def test_the_estimate_falls_below_the_makespan_when_another_path_stays_longest() -> None:
    shop = _known()
    orders = [[0, 2, 1], [4, 3], [5, 6, 7]]
    heads, tails = _heads_tails(shop, orders)
    assert swap_estimate(shop, orders[0], 0, heads, tails) == 12.0
    # The chain of machine C, which the swap does not touch, still ends at 14.
    assert _makespan(shop, _swapped(orders, 0, 0)) == 14.0


@pytest.mark.parametrize("place", [-1, 2, 5])
def test_the_estimate_refuses_a_place_without_an_operation_after_it(place: int) -> None:
    shop = _known()
    heads, tails = _heads_tails(shop, ORDERS)
    with pytest.raises(ValueError, match="no operation after it"):
        swap_estimate(shop, ORDERS[0], place, heads, tails)


# ---------------------------------------------------------------------------- N7


def _one_machine(operations: int) -> JobShop:
    problem = Problem(
        "one-machine",
        tasks=tuple(Task.on(f"t{number}", "M", number + 1.0) for number in range(operations)),
        resources=(Resource("M"),),
    )
    return JobShop.from_problem(problem)


def test_n7_adds_the_insertions_at_the_ends_of_the_blocks_to_the_n5_swaps() -> None:
    shop = _known()
    heads, _ = _heads_tails(shop, ORDERS)
    blocks = critical_blocks(shop, ORDERS, heads)
    moves = n7_moves(shop, ORDERS, blocks)
    assert moves == [(0, 0, 2), (0, 1, 2), (0, 2, 0), (1, 0, 1), (2, 0, 1), (2, 0, 2), (2, 2, 0)]
    assert set(n5_moves(shop, ORDERS, blocks)) <= set(moves)
    assert n7_moves(shop, ORDERS, blocks[:1]) == []
    for machine, source, target in moves:
        changed = [list(order) for order in ORDERS]
        changed[machine].insert(target, changed[machine].pop(source))
        assert _makespan(shop, changed) is not None


def test_n7_moves_a_block_end_to_every_place_and_every_place_to_a_block_end() -> None:
    shop = _one_machine(5)
    orders = [[0, 1, 2, 3, 4]]
    # An inner block of four: its two ends, and the moves that put an operation at an end.
    assert n7_moves(shop, orders, ((4,), (0, 1, 2, 3), (4,))) == [
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (0, 1, 3),
        (0, 2, 0),
        (0, 2, 3),
        (0, 3, 0),
        (0, 3, 1),
    ]
    # A block that starts at the second place of its machine moves by places of the machine.
    assert n7_moves(shop, orders, ((0,), (1, 2), (4,))) == [(0, 1, 2)]


# -------------------------------------------------------------------- the descent


def _run(shop: JobShop, evaluations: int = 100) -> ShopRun[Any]:
    search = Search(shop.problem, StopCriteria(max_evaluations=evaluations), 0, ShopTabuSearch.info)
    return ShopRun(search, shop, "active", make_representation(shop, "sequence"))


def test_the_descent_applies_the_best_exact_swap_until_none_lowers_the_makespan() -> None:
    shop = _known()
    run = _run(shop)
    result = critical_block_descent(shop, ORDERS, run)
    # B swaps first (32 to 18), then the last pair of A (18 to 14). The swap of x1 and x3
    # that follows is estimated at 12, below the bound, so it is recomputed exactly: 14.
    assert result.orders == ((0, 2, 1), (4, 3), (5, 6, 7))
    assert result.makespan == 14.0
    assert max(result.decoded.ends) == 14.0
    assert result.moves == run.search.evaluations == 2
    assert run.search.best is not None
    assert run.search.best.score.value == 14.0


def test_the_descent_counts_nothing_when_no_swap_lowers_the_makespan() -> None:
    shop = _known()
    run = _run(shop)
    orders = ((0, 2, 1), (4, 3), (5, 6, 7))
    result = critical_block_descent(shop, orders, run)
    assert (result.orders, result.makespan, result.moves) == (orders, 14.0, 0)
    assert max(result.decoded.ends) == 14.0
    assert run.search.evaluations == 0


def test_the_descent_stops_where_the_budget_ends() -> None:
    shop = _known()
    run = _run(shop, evaluations=1)
    with pytest.raises(Exhausted):
        critical_block_descent(shop, ORDERS, run)
    assert run.search.evaluations == 1


def test_the_descent_refuses_a_run_of_another_shop_and_orders_with_a_cycle() -> None:
    shop = _known()
    with pytest.raises(ValueError, match="another job shop"):
        critical_block_descent(_returning(), [[1, 2], [0, 3]], _run(shop))
    returning = _returning()
    with pytest.raises(ModelError, match="cycle"):
        critical_block_descent(returning, [[2, 1], [0, 3]], _run(returning))


# ------------------------------------------------------------- at scale, on random shops


def _shops() -> list[JobShop]:
    """Taillard's first shop and forty small ones, with release times and returning jobs."""
    shops = [ta01()]
    for index in range(40):
        problem = random_job_shop(
            index, jobs=5, machines=4, releases=index % 2 == 0, revisits=index % 3 == 0
        )
        shops.append(JobShop.from_problem(problem))
    return shops


def _starts(shop: JobShop, key: str) -> list[tuple[tuple[int, ...], ...]]:
    """Return the machine orders of the schedules of random job sequences, two builders each."""
    starts = []
    for sequence in random_sequences(shop, 4, key):
        for builder in ("active", "semi-active"):
            starts.append(machine_orders(shop, JobSequenceDecoder(shop, builder).decode(sequence)))
    return starts


def test_the_estimate_never_exceeds_the_exact_makespan_of_any_adjacent_swap() -> None:
    checked = tight = cyclic = 0
    for number, shop in enumerate(_shops()):
        for orders in _starts(shop, f"estimate-{number}"):
            heads, tails = _heads_tails(shop, orders)
            for machine, order in enumerate(orders):
                for place in range(len(order) - 1):
                    estimate = swap_estimate(shop, order, place, heads, tails)
                    exact = _makespan(shop, _swapped(orders, machine, place))
                    if exact is None:
                        cyclic += 1
                        continue
                    checked += 1
                    tight += estimate == exact
                    assert estimate <= exact, (shop.problem.id, machine, place)
    # The check is not vacuous: many swaps are priced exactly, and many strictly below.
    assert checked > 5000
    assert tight > 1000
    assert checked - tight > 100
    assert cyclic > 0


def test_swapping_adjacent_critical_operations_leaves_no_cycle_unless_the_job_joins_them() -> None:
    swaps = joined = 0
    for number, shop in enumerate(_shops()):
        for orders in _starts(shop, f"critical-{number}"):
            heads, tails = _heads_tails(shop, orders)
            for block in critical_blocks(shop, orders, heads):
                machine = shop.machine[block[0]]
                start = orders[machine].index(block[0])
                for place in range(start, start + len(block) - 1):
                    exact = _makespan(shop, _swapped(orders, machine, place))
                    first, second = orders[machine][place], orders[machine][place + 1]
                    if shop.successor[first] == second:
                        joined += 1
                        assert exact is None
                        continue
                    swaps += 1
                    assert exact is not None
                    estimate = swap_estimate(shop, orders[machine], place, heads, tails)
                    assert estimate <= exact
    assert swaps > 1000
    assert joined > 0


def test_n5_moves_of_every_critical_path_are_acyclic_and_inside_its_blocks() -> None:
    for number, shop in enumerate(_shops()):
        for orders in _starts(shop, f"n5-{number}"):
            heads, _ = _heads_tails(shop, orders)
            blocks = critical_blocks(shop, orders, heads)
            moves = n5_moves(shop, orders, blocks)
            inside = {
                (shop.machine[block[0]], orders[shop.machine[block[0]]].index(block[0]) + at)
                for block in blocks
                for at in range(len(block) - 1)
            }
            assert len(moves) == len(set(moves)) <= 2 * len(blocks)
            for machine, place, target in moves:
                assert (machine, place) in inside
                assert target == place + 1
                assert _makespan(shop, _swapped(orders, machine, place)) is not None


# --------------------------------------------------------------------- the solver


def _ta01() -> Problem:
    return ta01().problem


def _serialised(problem: Problem, result: SolveResult) -> str:
    assert result.schedule is not None
    return canonical_json(schedule_to_dict(result.schedule, problem.id))


def _extra(result: SolveResult, key: str) -> int:
    value = result.extras[key]
    assert isinstance(value, int)
    return value


@pytest.mark.parametrize("neighbourhood", NEIGHBOURHOODS)
def test_tabu_search_passes_the_contract(neighbourhood: NeighbourhoodName) -> None:
    problems = [small_job_shop(), chain_problem(), _ta01()]
    problems += [random_job_shop(index, revisits=index % 2 == 0) for index in range(8)]
    for problem in problems:
        result = check_contract(ShopTabuSearch, problem, neighbourhood=neighbourhood)
        assert result.extras["repairs"] == 0
        assert result.extras["observed_evaluations"] == result.evaluations == 40
        # The start, each applied move, and each restart are the evaluations.
        assert result.evaluations == 1 + _extra(result, "moves") + _extra(result, "restarts")


@pytest.mark.parametrize("neighbourhood", NEIGHBOURHOODS)
def test_tabu_search_replays_from_its_seed(neighbourhood: NeighbourhoodName) -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=150)
    solver = ShopTabuSearch(neighbourhood, max_stagnation=5)
    first, second, other = (solver.solve(problem, stop, seed=seed) for seed in (5, 5, 6))
    assert _extra(first, "restarts") > 0
    assert _serialised(problem, first) == _serialised(problem, second)
    assert [(p.evaluations, p.iteration, p.value) for p in first.trace] == [
        (p.evaluations, p.iteration, p.value) for p in second.trace
    ]
    assert untimed(first.extras) == untimed(second.extras)
    assert first.extras["first_schedule_digest"] != other.extras["first_schedule_digest"]
    assert _serialised(problem, first) != _serialised(problem, other)


@pytest.mark.parametrize("seed", [7, 8])
def test_tabu_search_beats_random_sampling_on_ta01_at_2000_evaluations(seed: int) -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=2000)
    tabu = ShopTabuSearch().solve(problem, stop, seed=seed)
    sampling = ShopRandomSampling("sequence", "active").solve(problem, stop, seed=seed)
    assert tabu.evaluations == sampling.evaluations == 2000
    assert tabu.stop_reason is sampling.stop_reason is StopReason.EVALUATION_LIMIT
    assert tabu.value is not None
    assert sampling.value is not None
    assert tabu.value < sampling.value
    assert tabu.value <= 1400.0


def test_tabu_search_starts_from_the_job_sequence_random_sampling_draws_first() -> None:
    stop = StopCriteria(max_evaluations=12)
    for problem in (small_job_shop(), _ta01()):
        tabu = ShopTabuSearch().solve(problem, stop, seed=3)
        sampling = ShopRandomSampling("sequence", "active").solve(problem, stop, seed=3)
        assert tabu.extras["first_schedule_digest"] == sampling.extras["first_schedule_digest"]
        assert tabu.trace[0].value == sampling.trace[0].value


def test_the_observer_sees_every_evaluation_and_a_swap_moves_one_pair() -> None:
    # Without a restart every evaluation but the first is one swap of its parent.
    problem = _ta01()
    result = ShopTabuSearch(max_stagnation=10**9).solve(
        problem, StopCriteria(max_evaluations=120), seed=2
    )
    assert _extra(result, "restarts") == 0
    assert result.extras["observed_evaluations"] == 120
    assert result.extras["observed_moves"] == result.extras["measured_moves"] == 119
    assert result.extras["phenotype_distance_mean"] == 1.0
    assert result.extras["phenotype_distance_median"] == 1.0
    assert result.extras["unchanged_move_share"] == 0.0
    assert float(result.extras["revisit_share"]) < 0.1


# ---------------------------------------------------------------- the tabu rules


def _record_schedules(monkeypatch: pytest.MonkeyPatch) -> list[Decoded]:
    """Keep every schedule that any search counts, in order, while a test lasts."""
    schedules: list[Decoded] = []
    assess = Search.assess_built

    def recorded(
        self: Search, build: Callable[[], Decoded], order: Sequence[int] | None = None
    ) -> Candidate:
        candidate = assess(self, build, order)
        schedules.append(candidate.decoded)
        return candidate

    monkeypatch.setattr(Search, "assess_built", recorded)
    return schedules


def _undone_arcs(shop: JobShop, schedules: Sequence[Decoded], tenure: int) -> tuple[int, int]:
    """Return the swaps of a walk that undo an arc live in the tabu list, and the swaps.

    The walk is read from its schedules, each a swap of the one before it. A swap of
    ``u`` then ``v`` creates the arc ``(v, u)``, which stays live for the next
    ``tenure`` swaps, and a swap is tabu while it reverses a live arc.
    """
    created: dict[tuple[int, int], int] = {}
    swaps = undone = 0
    previous = machine_orders(shop, schedules[0])
    for decoded in schedules[1:]:
        orders = machine_orders(shop, decoded)
        moved = [(old, new) for old, new in zip(previous, orders, strict=True) if old != new]
        assert len(moved) == 1, "a walk without restarts moves one machine at a time"
        old, new = moved[0]
        places = [place for place, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]
        assert len(places) == 2
        assert places[1] == places[0] + 1
        first, second = old[places[0]], old[places[1]]
        assert (new[places[0]], new[places[1]]) == (second, first)
        swaps += 1
        undone += (first, second) in created and created[first, second] + tenure >= swaps
        created[second, first] = swaps
        previous = orders
    return undone, swaps


@pytest.mark.parametrize("tenure", [1, 4, 10**6])
@pytest.mark.parametrize("make", [small_job_shop, _ta01])
def test_a_swap_that_undoes_a_live_arc_is_applied_only_by_aspiration_or_by_force(
    make: Callable[[], Problem], tenure: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    shop = JobShop.from_problem(make())
    schedules = _record_schedules(monkeypatch)
    solver = ShopTabuSearch(tenure_low=tenure, tenure_high=tenure, max_stagnation=10**9)
    result = solver.solve(shop.problem, StopCriteria(max_evaluations=300), seed=4)
    assert _extra(result, "restarts") == 0
    undone, swaps = _undone_arcs(shop, schedules, tenure)
    assert swaps == _extra(result, "moves") == 299
    # Every swap that undid a live arc was admitted by aspiration or forced, and every
    # aspiration and forced move undid one: the list is read exactly as documented.
    assert undone == _extra(result, "aspirations") + _extra(result, "forced_moves")
    if tenure == 10**6 and shop.size == 9:
        assert undone > 20


def _move(value: float, *arcs: tuple[int, int]) -> _Candidate:
    return _Candidate((0, 0, 1), value, None, arcs)


def test_the_tabu_list_forbids_the_reversed_arc_for_the_tenure_of_the_move() -> None:
    tabu = _TabuList()
    # The swap of 1 then 2 at step 5 creates the arc (2, 1), forbidden for the next 3 steps.
    tabu.forbid([(1, 2)], step=5, tenure=3)
    assert tabu.oldest([(2, 1)], 6) == 5
    assert tabu.oldest([(2, 1)], 8) == 5
    assert tabu.oldest([(2, 1)], 9) is None
    assert tabu.oldest([(1, 2)], 6) is None
    # A move across several arcs is tabu through the oldest live entry among them.
    tabu.forbid([(3, 4), (5, 6)], step=7, tenure=3)
    assert tabu.oldest([(6, 5), (2, 1)], 8) == 5
    assert tabu.oldest([(6, 5), (4, 3)], 8) == 7
    tabu.expire(9)
    assert tabu.oldest([(2, 1)], 8) is None
    assert tabu.oldest([(6, 5)], 9) == 7
    tabu.clear()
    assert tabu.oldest([(6, 5)], 9) is None


def test_the_move_of_least_value_wins_among_those_that_are_not_tabu() -> None:
    tabu = _TabuList()
    tabu.forbid([(1, 2)], step=1, tenure=5)
    candidates = [_move(10.0, (2, 1)), _move(12.0, (3, 4)), _move(11.0, (5, 6))]
    control = np.random.default_rng(0)
    assert ShopTabuSearch._select(candidates, 9.0, tabu, 2, control) == (2, "free")
    # No draw is made unless values tie.
    assert control.bit_generator.state == np.random.default_rng(0).bit_generator.state


def test_aspiration_admits_a_tabu_move_only_below_the_best_makespan() -> None:
    tabu = _TabuList()
    tabu.forbid([(1, 2)], step=1, tenure=5)
    candidates = [_move(10.0, (2, 1)), _move(12.0, (3, 4)), _move(11.0, (5, 6))]
    control = np.random.default_rng(0)
    assert ShopTabuSearch._select(candidates, 10.5, tabu, 2, control) == (0, "aspired")
    assert ShopTabuSearch._select(candidates, 10.0, tabu, 2, control) == (2, "free")


def test_when_every_move_is_tabu_the_one_with_the_oldest_entry_is_forced() -> None:
    tabu = _TabuList()
    for step, arc in ((3, (1, 2)), (1, (3, 4)), (2, (5, 6))):
        tabu.forbid([arc], step=step, tenure=9)
    candidates = [_move(10.0, (2, 1)), _move(11.0, (4, 3)), _move(12.0, (6, 5))]
    control = np.random.default_rng(0)
    assert ShopTabuSearch._select(candidates, 9.0, tabu, 4, control) == (1, "forced")
    # A move across two entries is as old as its oldest, and ties go to the first move.
    candidates = [_move(10.0, (2, 1), (6, 5)), _move(11.0, (2, 1), (4, 3)), _move(9.5, (4, 3))]
    assert ShopTabuSearch._select(candidates, 9.0, tabu, 4, control) == (1, "forced")
    assert control.bit_generator.state == np.random.default_rng(0).bit_generator.state


def test_equal_values_are_told_apart_by_one_draw_from_the_control_stream() -> None:
    candidates = [_move(10.0, (1, 2)), _move(10.0, (3, 4)), _move(11.0, (5, 6))]
    chosen = ShopTabuSearch._select(candidates, 9.0, _TabuList(), 1, np.random.default_rng(5))
    assert chosen == (int(np.random.default_rng(5).integers(2)), "free")


@pytest.mark.parametrize("restart_swaps", [1, 3])
def test_a_stagnation_restarts_from_the_best_orders_after_at_most_the_swaps_asked(
    restart_swaps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    shop = ta01()
    schedules = _record_schedules(monkeypatch)
    solver = ShopTabuSearch(max_stagnation=1, restart_swaps=restart_swaps)
    result = solver.solve(shop.problem, StopCriteria(max_evaluations=120), seed=1)
    assert _extra(result, "restarts") > 10
    assert result.evaluations == 1 + _extra(result, "moves") + _extra(result, "restarts")
    # With a stagnation of one, a move that does not improve the best is followed by a
    # restart, whose orders lie at most ``restart_swaps`` swaps from the best orders.
    distance = TrajectoryObserver(shop).distance
    best, best_value, restarting = schedules[0], max(schedules[0].ends), False
    checked = 0
    for decoded in schedules[1:]:
        value = max(decoded.ends)
        if restarting:
            assert distance(best, decoded) <= restart_swaps
            checked += 1
        restarting = not restarting and value >= best_value
        if value < best_value:
            best, best_value = decoded, value
    assert checked == _extra(result, "restarts")


# ------------------------------------------------------------------ edge cases


def test_a_shop_whose_critical_path_has_no_swap_restarts_from_random_sequences() -> None:
    # Job a-b and job c share no machine, so the critical path a-b is two blocks of one.
    problem = Problem(
        "apart",
        tasks=(Task.on("a", "m0", 3), Task.on("b", "m1", 4), Task.on("c", "m2", 5)),
        resources=(Resource("m0"), Resource("m1"), Resource("m2")),
        dependencies=(Dependency("a", "b"),),
    )
    result = ShopTabuSearch().solve(problem, StopCriteria(max_evaluations=20), seed=1)
    assert result.evaluations == 20
    assert result.stop_reason is StopReason.EVALUATION_LIMIT
    assert result.value == 7.0
    assert (_extra(result, "moves"), _extra(result, "restarts")) == (0, 19)


@pytest.mark.parametrize("operations", [1, 2, 3])
def test_a_shop_of_one_job_has_one_schedule_and_the_run_stops_after_it(operations: int) -> None:
    result = ShopTabuSearch().solve(one_job(operations), StopCriteria(max_evaluations=30), seed=3)
    assert result.evaluations == 1
    assert result.stop_reason is StopReason.COMPLETED
    assert result.value == sum(step + 2 for step in range(operations))
    assert (_extra(result, "moves"), _extra(result, "restarts")) == (0, 0)


def test_tabu_search_runs_under_an_iteration_or_a_time_limit_alone() -> None:
    by_iterations = ShopTabuSearch().solve(_ta01(), StopCriteria(max_iterations=6), seed=2)
    assert by_iterations.stop_reason is StopReason.ITERATION_LIMIT
    assert by_iterations.iterations == 6
    assert by_iterations.evaluations == 7
    by_time = ShopTabuSearch("n7").solve(small_job_shop(), StopCriteria(time_limit=0.05), seed=2)
    assert by_time.stop_reason is StopReason.TIME_LIMIT
    assert by_time.evaluations >= 1


def test_a_run_that_reaches_its_target_ends_there_and_the_observer_saw_that_schedule() -> None:
    stop = StopCriteria(max_evaluations=2000, target=1500.0)
    result = ShopTabuSearch().solve(_ta01(), stop, seed=7)
    assert result.stop_reason is StopReason.TARGET_REACHED
    assert result.value is not None
    assert result.value <= 1500.0
    assert 1 < result.evaluations < 2000
    assert result.extras["observed_evaluations"] == result.evaluations


def test_the_start_is_the_first_draw_of_the_operator_stream_and_a_walk_draws_no_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shop = JobShop.from_problem(small_job_shop())
    recording = Recording(monkeypatch)
    solver = ShopTabuSearch(max_stagnation=10**9)
    result = solver.solve(shop.problem, StopCriteria(max_evaluations=25), seed=4)
    assert _extra(result, "restarts") == 0
    (run,) = recording.finished
    operators = np.random.default_rng(derive_seed(4, OPERATOR_STREAM))
    SequenceRepresentation(shop).initial(operators)
    assert run.operators.bit_generator.state == operators.bit_generator.state
    # The tenures come from the control stream, one per move at least.
    control = np.random.default_rng(derive_seed(4, CONTROL_STREAM))
    assert run.control.bit_generator.state != control.bit_generator.state


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"neighbourhood": "n6"}, "unknown neighbourhood"),
        ({"tenure_low": 0}, "tenure_low must be an integer of at least 1"),
        ({"tenure_low": 5, "tenure_high": 4}, "tenure_high must be an integer of at least 5"),
        ({"tenure_high": 2.5}, "tenure_high must be an integer"),
        ({"max_stagnation": 0}, "max_stagnation must be an integer of at least 1"),
        ({"restart_swaps": True}, "restart_swaps must be an integer of at least 1"),
    ],
)
def test_tabu_search_refuses_invalid_parameters(parameters: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShopTabuSearch(**parameters)


# ------------------------------------------------------------- the first solve

_FRESH = """
import json
import sys

from dispatchatlas.campaigns.execution import warm_up
from dispatchatlas.core import StopCriteria
from dispatchatlas.solvers.shop.tabu import ShopTabuSearch
from tests.helpers import chain_problem

late = {}
for neighbourhood in ("n5", "n7"):
    solver = ShopTabuSearch(neighbourhood)
    warm_up(solver)
    before = set(sys.modules)
    solver.solve(chain_problem(), StopCriteria(max_evaluations=30), 2)
    late[neighbourhood] = sorted(set(sys.modules) - before)
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
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"n5": [], "n7": []}
