"""Giffler-Thompson from one rank per operation: the same procedure for any distinct ranks."""

from __future__ import annotations

import itertools
import time
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
    derive_seed,
)
from dispatchatlas.solvers import scheduling
from dispatchatlas.solvers.evaluation import Evaluator, Score
from dispatchatlas.solvers.shop.decoding import giffler_thompson, giffler_thompson_ranked
from dispatchatlas.solvers.shop.instance import JobShop
from tests.helpers import small_job_shop
from tests.solvers.shop.instances import random_job_shop, random_sequences, ta01
from tests.solvers.shop.oracles import naive_giffler_thompson

NON_DELAY = [pytest.param(False, id="active"), pytest.param(True, id="non-delay")]


def _four_jobs() -> JobShop:
    """Four one-operation jobs: a0 (2) and c0 (3) on m0, b0 (2) and d0 (3) on m1."""
    return JobShop.from_problem(
        Problem(
            "four-jobs",
            tasks=(
                Task.on("a0", "m0", 2),
                Task.on("b0", "m1", 2),
                Task.on("c0", "m0", 3),
                Task.on("d0", "m1", 3),
            ),
            resources=(Resource("m0"), Resource("m1")),
        )
    )


def _zero_duration() -> JobShop:
    """Three jobs of two operations on two machines, three of them taking no time."""
    return JobShop.from_problem(
        Problem(
            "zero-duration",
            tasks=(
                Task.on("j0-o0", "m0", 0),
                Task.on("j0-o1", "m1", 3),
                Task.on("j1-o0", "m0", 2),
                Task.on("j1-o1", "m1", 0),
                Task.on("j2-o0", "m1", 0),
                Task.on("j2-o1", "m0", 1),
            ),
            resources=(Resource("m0"), Resource("m1")),
            dependencies=(
                Dependency("j0-o0", "j0-o1"),
                Dependency("j1-o0", "j1-o1"),
                Dependency("j2-o0", "j2-o1"),
            ),
        )
    )


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


def _positions(shop: JobShop, sequence: np.ndarray[Any, Any]) -> list[int]:
    """Return the position in a job sequence of the occurrence that stands for each task."""
    taken = [0] * shop.jobs
    rank = [0] * shop.size
    for position, job in enumerate(sequence.tolist()):
        rank[shop.operations[job][taken[job]]] = position
        taken[job] += 1
    return rank


def _random_ranks(shop: JobShop, count: int, namespace: str) -> list[list[int]]:
    """Return ``count`` uniformly random permutations of the tasks as ranks.

    Few of them are the positions of any job sequence, which ranks the operations
    of every job in route order.
    """
    rng = np.random.default_rng(derive_seed(7, namespace))
    return [rng.permutation(shop.size).tolist() for _ in range(count)]


# ------------------------------------------------------------- the sequence entry point


@pytest.mark.parametrize("non_delay", NON_DELAY)
@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_ranks_that_are_a_job_sequences_positions_decode_as_the_sequence_does(
    shop: JobShop, non_delay: bool
) -> None:
    for sequence in random_sequences(shop, 8, f"tests/shop/ranked/{shop.problem.id}/{non_delay}"):
        ranks = _positions(shop, sequence)
        expected = giffler_thompson(shop, sequence, non_delay=non_delay)
        assert giffler_thompson_ranked(shop, ranks, non_delay=non_delay) == expected
        # The ranks may be any integer sequence or integer array.
        assert giffler_thompson_ranked(shop, tuple(ranks), non_delay=non_delay) == expected
        as_array = np.array(ranks, dtype=np.uint16)
        assert giffler_thompson_ranked(shop, as_array, non_delay=non_delay) == expected


# ------------------------------------------------------------- the procedure on any ranks


@pytest.mark.parametrize("non_delay", NON_DELAY)
@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_the_ranked_builder_equals_the_textbook_procedure_on_arbitrary_ranks(
    shop: JobShop, non_delay: bool
) -> None:
    for ranks in _random_ranks(shop, 8, f"tests/shop/ranked-naive/{shop.problem.id}/{non_delay}"):
        decoded = giffler_thompson_ranked(shop, ranks, non_delay=non_delay)
        assert decoded == naive_giffler_thompson(shop, ranks, non_delay=non_delay)


@pytest.mark.parametrize("non_delay", NON_DELAY)
def test_every_ranking_of_a_shop_with_operations_that_take_no_time_matches_the_textbook(
    non_delay: bool,
) -> None:
    # An operation that takes no time completes when it starts, so it need not be
    # among the operations that start before C*: the conflict set keeps it all the same.
    shop = _zero_duration()
    for ranks in itertools.permutations(range(shop.size)):
        decoded = giffler_thompson_ranked(shop, ranks, non_delay=non_delay)
        assert decoded == naive_giffler_thompson(shop, ranks, non_delay=non_delay)


@pytest.mark.parametrize(
    ("ranks", "placed"),
    [
        # a0 comes before b0, so m0 is the machine; c0 comes first there.
        ([2, 3, 0, 1], (2, 3, 0, 1)),
        # b0 comes before a0, so m1 is the machine; d0 comes first there.
        ([3, 2, 0, 1], (3, 2, 1, 0)),
        # Only the order of the ranks counts: these are those of the first case, spread out.
        ([20, 30, -5, 10], (2, 3, 0, 1)),
        ([30, 20, -5, 10], (3, 2, 1, 0)),
    ],
)
def test_the_operation_of_lowest_rank_fixes_the_machine(
    ranks: list[int], placed: tuple[int, ...]
) -> None:
    # a0 on m0 and b0 on m1 both complete first, at 2, and c0 and d0 can start
    # before then on the same machines. Taking the lowest machine, or the task of
    # lowest index, would schedule c0 first under every ranking.
    decoded = giffler_thompson_ranked(_four_jobs(), ranks)
    assert decoded.sequence == placed
    assert decoded.starts == (3.0, 3.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("ranks", "placed"), [([2, 3, 0, 1], (2, 3, 0, 1)), ([3, 2, 0, 1], (2, 3, 1, 0))]
)
def test_the_non_delay_builder_breaks_ties_of_the_earliest_start_by_rank(
    ranks: list[int], placed: tuple[int, ...]
) -> None:
    # Every operation can start at 0; after c0 and d0, a0 and b0 can both start at 3.
    decoded = giffler_thompson_ranked(_four_jobs(), ranks, non_delay=True)
    assert decoded.sequence == placed
    assert decoded.starts == (3.0, 3.0, 0.0, 0.0)


def test_a_rank_may_put_an_operation_ahead_of_its_job_predecessor() -> None:
    shop = JobShop.from_problem(small_job_shop())
    # Every operation ranks ahead of the one before it in its job, which no job sequence does.
    ranks = list(range(shop.size - 1, -1, -1))
    assert all(
        ranks[shop.predecessor[task]] > ranks[task]
        for task in range(shop.size)
        if shop.predecessor[task] >= 0
    )
    decoded = giffler_thompson_ranked(shop, ranks)
    assert decoded == naive_giffler_thompson(shop, ranks)
    assert check_schedule(shop.problem, decoded.schedule(shop.problem)) == ()
    assert decoded.makespan == 12


# ------------------------------------------------------------- what the ranks must be


@pytest.mark.parametrize(
    ("ranks", "issues"),
    [
        (list(range(8)), ("there are 8 ranks for the 9 operations",)),
        (list(range(10)), ("there are 10 ranks for the 9 operations",)),
        ([], ("there are 0 ranks for the 9 operations",)),
        ([0, 1, 2, 3, 4, 5, 6, 7, 7], ("ranks occur more than once: [7]",)),
        ([4] * 9, ("ranks occur more than once: [4]",)),
        (
            [float(rank) for rank in range(9)],
            ("the ranks are integers, got float64",),
        ),
        ([True, False] * 4 + [True], ("the ranks are integers, got bool",)),
        (
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            ("the ranks are one row of integers, one per operation, got shape (3, 3)",),
        ),
    ],
    ids=["short", "long", "empty", "repeated", "all-equal", "floats", "booleans", "rows"],
)
def test_ranks_that_are_not_one_distinct_integer_per_operation_are_refused(
    ranks: Any, issues: tuple[str, ...]
) -> None:
    shop = JobShop.from_problem(small_job_shop())
    for non_delay in (False, True):
        with pytest.raises(
            ModelError,
            match="not one distinct integer per operation of problem 'small-job-shop'",
        ) as caught:
            giffler_thompson_ranked(shop, ranks, non_delay=non_delay)
        assert caught.value.issues == issues


def test_ranks_may_be_negative_and_far_apart() -> None:
    shop = JobShop.from_problem(small_job_shop())
    ranks = [(-1) ** task * (task + 1) * 10**9 for task in range(shop.size)]
    assert len(set(ranks)) == shop.size
    assert giffler_thompson_ranked(shop, ranks) == naive_giffler_thompson(shop, ranks)


# ------------------------------------------------------------- schedules and cost


@pytest.mark.parametrize("non_delay", NON_DELAY)
@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_random_ranks_decode_into_feasible_schedules(shop: JobShop, non_delay: bool) -> None:
    problem = shop.problem
    evaluator = Evaluator(problem)
    for ranks in _random_ranks(shop, 6, f"tests/shop/ranked-feasible/{problem.id}/{non_delay}"):
        decoded = giffler_thompson_ranked(shop, ranks, non_delay=non_delay)
        assert sorted(decoded.sequence) == list(range(shop.size))
        assert decoded.modes == (0,) * shop.size
        assert check_schedule(problem, decoded.schedule(problem)) == ()
        assert evaluator.score(decoded) == Score(0.0, decoded.makespan)


def test_the_ranked_builder_never_repairs_and_decodes_ta01_in_well_under_a_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("ranks went through the precedence repair")

    monkeypatch.setattr(scheduling, "precedence_feasible", refuse)
    monkeypatch.setattr(scheduling.Decoder, "decode", refuse)
    shop = ta01()
    ranks = _random_ranks(shop, 5, "tests/shop/ranked-timing")
    started = time.perf_counter()
    for each in ranks:
        giffler_thompson_ranked(shop, each)
    # A generous bound: a decode of 225 operations takes milliseconds.
    assert (time.perf_counter() - started) / len(ranks) < 0.5
