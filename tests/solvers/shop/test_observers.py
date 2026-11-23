"""The trajectory observer measures revisits, unchanged moves and distances, and changes nothing."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from dispatchatlas.core import StopCriteria
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.decoding import BUILDERS, BuilderName, JobSequenceDecoder
from dispatchatlas.solvers.shop.graph import machine_orders
from dispatchatlas.solvers.shop.instance import JobShop
from dispatchatlas.solvers.shop.observers import TrajectoryObserver
from dispatchatlas.solvers.shop.representation import (
    KeyRepresentation,
    Representation,
    SequenceRepresentation,
)
from dispatchatlas.solvers.shop.run import Evaluated, ShopRun
from tests.helpers import small_job_shop
from tests.solvers.shop.instances import random_job_shop, random_sequences, ta01
from tests.solvers.shop.runs import INFO, Digests, climb


def _small() -> JobShop:
    return JobShop.from_problem(small_job_shop())


def _run(
    shop: JobShop,
    representation: Representation[Any],
    observer: Any = None,
    *,
    builder: BuilderName = "semi-active",
    evaluations: int = 100,
) -> ShopRun[Any]:
    search = Search(shop.problem, StopCriteria(max_evaluations=evaluations), 5, INFO)
    return ShopRun(search, shop, builder, representation, observer)


def _brute_force(shop: JobShop, first: Decoded, second: Decoded) -> int:
    """Count, machine by machine, the pairs of operations two schedules order differently."""
    total = 0
    for one, other in zip(machine_orders(shop, first), machine_orders(shop, second), strict=True):
        place = {task: position for position, task in enumerate(other)}
        total += sum(
            place[one[i]] > place[one[j]] for i, j in itertools.combinations(range(len(one)), 2)
        )
    return total


ARMS: list[Callable[[JobShop], Representation[Any]]] = [
    lambda shop: KeyRepresentation(shop, "perturb", 0.02),
    lambda shop: KeyRepresentation(shop, "redraw"),
    lambda shop: SequenceRepresentation(shop, "swap"),
    lambda shop: SequenceRepresentation(shop, "mixed"),
]


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("arm", ARMS, ids=["perturb", "redraw", "swap", "mixed"])
def test_observers_do_not_change_the_run(
    builder: BuilderName, arm: Callable[[JobShop], Representation[Any]]
) -> None:
    shop = ta01()
    outcomes = []
    for observer in (None, TrajectoryObserver(shop), TrajectoryObserver(shop, 7), Digests()):
        run = _run(shop, arm(shop), observer, builder=builder, evaluations=120)
        values = climb(run)
        run.finish()
        result = run.search.result()
        outcomes.append(
            (
                values,
                result.evaluations,
                result.iterations,
                result.stop_reason,
                [(point.evaluations, point.iteration, point.value) for point in result.trace],
                result.schedule,
                result.extras["first_schedule_digest"],
                run.control.bit_generator.state,
                run.operators.bit_generator.state,
            )
        )
    assert all(outcome == outcomes[0] for outcome in outcomes[1:])


def _shops() -> list[JobShop]:
    return [
        _small(),
        ta01(),
        *(JobShop.from_problem(random_job_shop(i, jobs=5, revisits=True)) for i in range(4)),
    ]


def test_the_phenotype_distance_equals_brute_force_over_machine_orders() -> None:
    for shop in _shops():
        observer = TrajectoryObserver(shop)
        assert observer.pairs == sum(
            len(order) * (len(order) - 1) // 2 for order in machine_orders(shop, _any(shop))
        )
        for builder in BUILDERS:
            decoder = JobSequenceDecoder(shop, builder)
            sequences = random_sequences(shop, 8, f"tests/shop/observer/{shop.problem.id}")
            schedules = [decoder.decode(sequence) for sequence in sequences]
            for first, second in itertools.combinations(schedules, 2):
                distance = observer.distance(first, second)
                assert distance == _brute_force(shop, first, second)
                assert distance == observer.distance(second, first)
                assert 0 <= distance <= observer.pairs
            assert observer.distance(schedules[0], schedules[0]) == 0


def _any(shop: JobShop) -> Decoded:
    return JobSequenceDecoder(shop, "semi-active").decode(shop.slot_job)


def test_the_observer_counts_revisits_unchanged_moves_and_distances() -> None:
    shop = _small()
    run = _run(shop, SequenceRepresentation(shop), TrajectoryObserver(shop))
    last_first = np.array([2, 2, 2, 1, 1, 1, 0, 0, 0])
    first_first = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    with run.search:
        start = run.evaluate(last_first)
        again = run.evaluate(last_first.copy(), parent=start)  # the same schedule
        other = run.evaluate(first_first, parent=start)
        back = run.evaluate(last_first, parent=other)  # a schedule seen before
    distance = _brute_force(shop, start.decoded, other.decoded)
    assert distance > 0
    assert again.digest == start.digest == back.digest != other.digest
    run.finish()
    extras = run.search.extras
    assert extras["observed_evaluations"] == 4
    assert extras["revisit_share"] == 2 / 4
    assert extras["observed_moves"] == 3
    assert extras["unchanged_move_share"] == 1 / 3
    assert extras["measured_moves"] == 3
    assert extras["phenotype_distance_mean"] == pytest.approx(2 * distance / 3)
    assert extras["phenotype_distance_median"] == distance


def test_a_key_move_that_keeps_the_sequence_is_measured_as_unchanged() -> None:
    # Slots 0 and 1 both hold job 0: their keys trade places, the sequence stays.
    shop = _small()
    observer = TrajectoryObserver(shop)
    run = _run(shop, KeyRepresentation(shop), observer)
    keys = np.array([0.10, 0.12, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    moved = keys.copy()
    moved[0] = 0.13
    with run.search:
        start = run.evaluate(keys)
        run.evaluate(moved, parent=start)
    assert (observer.moves, observer.unchanged, observer.distances) == (1, 1, [0])


def test_distance_every_measures_the_first_move_and_one_in_every_k_after_it() -> None:
    shop = ta01()
    observer = TrajectoryObserver(shop, distance_every=3)
    run = _run(shop, SequenceRepresentation(shop), observer, evaluations=11)
    climb(run)
    assert observer.moves == 10
    assert len(observer.distances) == 4  # moves 1, 4, 7 and 10


def test_the_distances_of_moves_from_many_parents_equal_brute_force() -> None:
    # Parents taken in turn, as a population method takes them, so that the
    # observer's kept ranks are found, dropped, and computed again.
    shop = ta01()
    run = _run(shop, SequenceRepresentation(shop, "insert"))
    with run.search:
        population = [run.evaluate(run.representation.initial(run.operators)) for _ in range(4)]
        pairs: list[tuple[Evaluated[Any], Evaluated[Any]]] = []
        for index in (0, 1, 0, 2, 1, 3, 3, 0, 2, 2, 1):
            parent = population[index]
            child = run.evaluate(run.representation.move(parent.genotype, run.operators))
            pairs.append((parent, child))
    observer = TrajectoryObserver(shop)
    for parent, child in pairs:
        observer.observe(child, parent)
    assert observer.distances == [
        _brute_force(shop, parent.decoded, child.decoded) for parent, child in pairs
    ]


def test_shares_and_distances_over_no_evaluation_are_left_out() -> None:
    shop = _small()
    observer = TrajectoryObserver(shop)
    assert observer.measures() == {
        "observed_evaluations": 0,
        "observed_moves": 0,
        "measured_moves": 0,
    }
    run = _run(shop, SequenceRepresentation(shop), observer, evaluations=3)
    with run.search:
        while True:
            run.evaluate(run.representation.initial(run.operators))
    measures = observer.measures()
    assert set(measures) == {
        "observed_evaluations",
        "observed_moves",
        "measured_moves",
        "revisit_share",
    }


@pytest.mark.parametrize("every", [0, -1, 1.5, True])
def test_the_observer_refuses_a_bad_distance_every(every: Any) -> None:
    with pytest.raises(ValueError, match="distance_every"):
        TrajectoryObserver(_small(), every)
