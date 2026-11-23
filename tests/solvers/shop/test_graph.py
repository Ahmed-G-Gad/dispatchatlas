"""The disjunctive graph of job-shop schedules: machine orders, heads, tails, critical blocks."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from dispatchatlas.core import (
    Dependency,
    ModelError,
    Problem,
    Resource,
    Task,
    check_schedule,
)
from dispatchatlas.solvers.evaluation import Evaluator, Score
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.shop.decoding import BUILDERS, BuilderName, JobSequenceDecoder
from dispatchatlas.solvers.shop.graph import (
    critical_blocks,
    decoded_from_heads,
    heads_tails,
    machine_orders,
)
from dispatchatlas.solvers.shop.instance import JobShop
from tests.helpers import small_job_shop
from tests.solvers.shop.instances import random_job_shop, random_sequences, ta01

# The small job shop under job 2 first, then job 1, then job 0; see test_decoding.py.
LAST_JOB_FIRST = [2, 2, 2, 1, 1, 1, 0, 0, 0]


def _small() -> JobShop:
    return JobShop.from_problem(small_job_shop())


def _shops() -> list[JobShop]:
    """Taillard's ta01 and small random job shops with release times, some revisiting machines."""
    return [
        ta01(),
        *(JobShop.from_problem(random_job_shop(index)) for index in range(4)),
        *(
            JobShop.from_problem(random_job_shop(index, jobs=5, revisits=True))
            for index in range(4, 8)
        ),
    ]


def _schedules(shop: JobShop, count: int = 4) -> list[Decoded]:
    """Schedules of ``shop`` from random sequences under every builder."""
    sequences = random_sequences(shop, count, f"tests/shop/graph/{shop.problem.id}")
    return [
        JobSequenceDecoder(shop, builder).decode(sequence)
        for builder in BUILDERS
        for sequence in sequences
    ]


def _heads_tails(
    shop: JobShop, orders: tuple[tuple[int, ...], ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    found = heads_tails(shop, orders)
    assert found is not None
    return found


def _machine_predecessors(shop: JobShop, orders: tuple[tuple[int, ...], ...]) -> list[int]:
    before = [-1] * shop.size
    for order in orders:
        for previous, task in itertools.pairwise(order):
            before[task] = previous
    return before


# ------------------------------------------------------------- machine orders


def test_machine_orders_list_each_machine_by_start_time() -> None:
    shop = _small()
    semi = JobSequenceDecoder(shop, "semi-active").decode(LAST_JOB_FIRST)
    assert machine_orders(shop, semi) == ((7, 3, 0), (6, 5, 1), (8, 4, 2))
    active = JobSequenceDecoder(shop, "active").decode(LAST_JOB_FIRST)
    assert machine_orders(shop, active) == ((3, 7, 0), (6, 5, 1), (4, 8, 2))


@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_every_builder_appends_so_its_machine_orders_follow_its_placement(shop: JobShop) -> None:
    for decoded in _schedules(shop):
        expected: list[list[int]] = [[] for _ in range(shop.machines)]
        for task in decoded.sequence:
            expected[shop.machine[task]].append(task)
        assert machine_orders(shop, decoded) == tuple(tuple(order) for order in expected)


def test_an_operation_that_takes_no_time_comes_before_one_that_starts_with_it() -> None:
    problem = Problem(
        "instants",
        tasks=(Task.on("a", "m0", 0), Task.on("b", "m0", 2), Task.on("c", "m0", 0)),
        resources=(Resource("m0"),),
    )
    shop = JobShop.from_problem(problem)
    # b was placed first, yet the instant a takes at 0 comes before b's run.
    decoded = Decoded((1, 0, 2), (0, 0, 0), (0.0, 0.0, 2.0), (0.0, 2.0, 2.0))
    assert check_schedule(problem, decoded.schedule(problem)) == ()
    assert machine_orders(shop, decoded) == ((0, 1, 2),)
    # Two instants at one time keep the order the schedule placed them in.
    both = Decoded((2, 0, 1), (0, 0, 0), (0.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    assert machine_orders(shop, both) == ((2, 0, 1),)


def test_machine_orders_refuse_a_schedule_that_leaves_an_operation_out() -> None:
    shop = _small()
    decoded = JobSequenceDecoder(shop, "semi-active").decode(LAST_JOB_FIRST)
    partial = Decoded(decoded.sequence[:8], decoded.modes, decoded.starts, decoded.ends)
    with pytest.raises(ModelError, match="places 8 of the 9 operations"):
        machine_orders(shop, partial)


# ------------------------------------------------------------- heads and tails


def test_the_heads_and_tails_of_the_small_job_shop() -> None:
    shop = _small()
    active = JobSequenceDecoder(shop, "active").decode(LAST_JOB_FIRST)
    heads, tails = _heads_tails(shop, machine_orders(shop, active))
    assert heads == (5.0, 8.0, 10.0, 0.0, 2.0, 3.0, 0.0, 3.0, 5.0)
    assert tails == (4.0, 2.0, 0.0, 9.0, 8.0, 4.0, 9.0, 7.0, 2.0)


@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_heads_reproduce_the_starts_of_every_builder(shop: JobShop) -> None:
    for decoded in _schedules(shop):
        heads, tails = _heads_tails(shop, machine_orders(shop, decoded))
        assert heads == decoded.starts
        makespan = decoded.makespan
        assert max(h + d for h, d in zip(heads, shop.duration, strict=True)) == makespan
        assert all(
            h + d + q <= makespan for h, d, q in zip(heads, shop.duration, tails, strict=True)
        )
        assert min(tails) == 0.0


def _mirrored(problem: Problem) -> Problem:
    """The problem with every job run backwards and without release times."""
    return Problem(
        f"{problem.id}-mirrored",
        tasks=tuple(Task(task.id, task.modes) for task in problem.tasks),
        resources=problem.resources,
        dependencies=tuple(Dependency(item.after, item.before) for item in problem.dependencies),
    )


@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_tails_are_the_heads_of_the_mirrored_schedule(shop: JobShop) -> None:
    mirror = JobShop.from_problem(_mirrored(shop.problem))
    for decoded in _schedules(shop, 2):
        orders = machine_orders(shop, decoded)
        heads, tails = _heads_tails(shop, orders)
        backwards = tuple(tuple(reversed(order)) for order in orders)
        mirror_heads, mirror_tails = _heads_tails(mirror, backwards)
        assert mirror_heads == pytest.approx(tails, abs=1e-9)
        if not any(shop.release):
            assert mirror_tails == pytest.approx(heads, abs=1e-9)


def test_orders_that_form_a_cycle_have_no_heads() -> None:
    shop = _small()
    # j0-o0 -> j0-o1 on its job, j0-o1 before j2-o0 on m1, j2-o0 -> j2-o1 on its
    # job, and j2-o1 before j0-o0 on m0 close a cycle.
    assert heads_tails(shop, ((7, 0, 3), (1, 6, 5), (2, 4, 8))) is None
    revisit = JobShop.from_problem(
        Problem(
            "revisit",
            tasks=(Task.on("a0", "m0", 1), Task.on("a1", "m0", 1)),
            resources=(Resource("m0"),),
            dependencies=(Dependency("a0", "a1"),),
        )
    )
    assert heads_tails(revisit, ((1, 0),)) is None
    assert heads_tails(revisit, ((0, 1),)) == ((0.0, 1.0), (1.0, 0.0))


@pytest.mark.parametrize(
    ("orders", "message"),
    [
        (((3, 7, 0), (6, 5, 1)), "expected 3 machine orders, got 2"),
        (((3, 7, 0), (6, 5, 1), (4, 8)), "list 8 of the 9 operations"),
        (((3, 7, 0, 3), (6, 5, 1), (4, 8, 2)), "lists task 3, which is unknown, listed twice"),
        (((3, 7, 1), (6, 5, 0), (4, 8, 2)), "machine order 0 lists task 1"),
        (((3, 7, 0), (6, 5, 1), (4, 8, 2, 9)), "lists task 9, which is unknown"),
        (((3, 7, 0), (6, 5, 1), (4, 8, -1)), "lists task -1, which is unknown"),
    ],
)
def test_orders_must_list_every_operation_once_on_its_machine(
    orders: tuple[tuple[int, ...], ...], message: str
) -> None:
    shop = _small()
    with pytest.raises(ModelError, match=message):
        heads_tails(shop, orders)
    heads = JobSequenceDecoder(shop, "active").decode(LAST_JOB_FIRST).starts
    with pytest.raises(ModelError, match=message):
        critical_blocks(shop, orders, heads)


# ------------------------------------------------------------- critical blocks


@pytest.mark.parametrize(
    ("builder", "blocks"),
    [
        # j2-o0, then j2-o1 and j0-o0 on m0, then j0-o1 and j0-o2: 3 + 5 + 2 + 2 = 12.
        ("active", ((6,), (7, 0), (1,), (2,))),
        # j2-o0, j2-o1, j2-o2 and j1-o1 on m2, j1-o2 and j0-o1 on m1, j0-o2: 17.
        ("semi-active", ((6,), (7,), (8, 4), (5, 1), (2,))),
    ],
)
def test_the_critical_blocks_of_the_small_job_shop(
    builder: BuilderName, blocks: tuple[tuple[int, ...], ...]
) -> None:
    shop = _small()
    decoded = JobSequenceDecoder(shop, builder).decode(LAST_JOB_FIRST)
    orders = machine_orders(shop, decoded)
    assert critical_blocks(shop, orders, decoded.starts) == blocks


def test_orders_may_hold_numpy_integers() -> None:
    shop = _small()
    active = JobSequenceDecoder(shop, "active").decode(LAST_JOB_FIRST)
    orders = machine_orders(shop, active)
    arrays: Any = [np.array(order, dtype=np.int64) for order in orders]
    assert heads_tails(shop, arrays) == heads_tails(shop, orders)
    blocks = critical_blocks(shop, arrays, active.starts)
    assert blocks == ((6,), (7, 0), (1,), (2,))
    assert all(type(task) is int for block in blocks for task in block)
    floats: Any = [[float(task) for task in order] for order in orders]
    with pytest.raises(TypeError):
        heads_tails(shop, floats)


def test_the_path_takes_the_machine_arc_when_both_arcs_are_tight() -> None:
    problem = Problem(
        "both-tight",
        tasks=(Task.on("a0", "m0", 2), Task.on("a1", "m1", 1), Task.on("b0", "m1", 2)),
        resources=(Resource("m0"), Resource("m1")),
        dependencies=(Dependency("a0", "a1"),),
    )
    shop = JobShop.from_problem(problem)
    decoded = JobSequenceDecoder(shop, "semi-active").decode([0, 1, 0])
    # a1 starts at 2, when both a0 on its job and b0 on its machine complete.
    assert decoded.starts == (0.0, 2.0, 0.0)
    assert critical_blocks(shop, machine_orders(shop, decoded), decoded.starts) == ((2, 1),)


@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_critical_blocks_trace_a_longest_path(shop: JobShop) -> None:
    duration, predecessor = shop.duration, shop.predecessor
    for decoded in _schedules(shop):
        orders = machine_orders(shop, decoded)
        heads, tails = _heads_tails(shop, orders)
        before = _machine_predecessors(shop, orders)
        blocks = critical_blocks(shop, orders, heads)
        path = [task for block in blocks for task in block]
        makespan = decoded.makespan
        ends = [h + d for h, d in zip(heads, duration, strict=True)]
        assert path[-1] == ends.index(makespan)
        assert heads[path[0]] == shop.release[path[0]]
        assert shop.release[path[0]] + sum(duration[task] for task in path) == makespan
        for task in path:
            assert heads[task] + duration[task] + tails[task] == makespan
        for block in blocks:
            for previous, task in itertools.pairwise(block):
                assert before[task] == previous
                assert heads[task] == ends[previous]
        for first, second in itertools.pairwise(blocks):
            # Consecutive blocks meet on a job arc that is no machine arc.
            assert predecessor[second[0]] == first[-1]
            assert before[second[0]] != first[-1]
            assert heads[second[0]] == ends[first[-1]]


def test_critical_blocks_refuse_heads_of_another_length() -> None:
    shop = _small()
    decoded = JobSequenceDecoder(shop, "active").decode(LAST_JOB_FIRST)
    with pytest.raises(ModelError, match="expected 9 heads, got 8"):
        critical_blocks(shop, machine_orders(shop, decoded), decoded.starts[:8])


# ------------------------------------------------------------- schedules from heads


@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_the_schedule_of_the_heads_is_the_schedule_they_came_from(shop: JobShop) -> None:
    problem = shop.problem
    evaluator = Evaluator(problem)
    for decoded in _schedules(shop):
        orders = machine_orders(shop, decoded)
        heads, _ = _heads_tails(shop, orders)
        rebuilt = decoded_from_heads(shop, heads)
        assert (rebuilt.starts, rebuilt.ends) == (decoded.starts, decoded.ends)
        assert rebuilt.modes == (0,) * shop.size
        assert sorted(rebuilt.sequence) == list(range(shop.size))
        assert all(type(value) is float for value in rebuilt.starts + rebuilt.ends)
        assert machine_orders(shop, rebuilt) == orders
        assert check_schedule(problem, rebuilt.schedule(problem)) == ()
        assert evaluator.score(rebuilt) == Score(0.0, decoded.makespan)
        # Its sequence places every operation after its job predecessor.
        position = {task: n for n, task in enumerate(rebuilt.sequence)}
        assert all(
            position[shop.predecessor[task]] < position[task]
            for task in range(shop.size)
            if shop.predecessor[task] >= 0
        )


def test_decoded_from_heads_refuses_heads_of_another_length() -> None:
    with pytest.raises(ModelError, match=r"expected 9 heads, got shape \(8,\)"):
        decoded_from_heads(_small(), [0.0] * 8)
