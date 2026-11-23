"""Kendall tau counts discordant pairs as brute force does; the step calibration hits its target."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from dispatchatlas.core import StopCriteria, derive_seed
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.distance import (
    MOVE_STREAM,
    SOLUTION_STREAM,
    calibrate_step,
    kendall_tau,
    move_distances,
    occurrence_permutation,
)
from dispatchatlas.solvers.shop.instance import JobShop
from dispatchatlas.solvers.shop.representation import (
    KeyRepresentation,
    Representation,
    SequenceRepresentation,
)
from dispatchatlas.solvers.shop.run import ShopRun
from tests.helpers import small_job_shop
from tests.solvers.shop.instances import ta01
from tests.solvers.shop.runs import INFO


def _brute_force(first: list[int], second: list[int]) -> int:
    """Count the pairs of items that two orderings list in different orders."""
    place = {item: position for position, item in enumerate(second)}
    return sum(
        place[first[i]] > place[first[j]] for i, j in itertools.combinations(range(len(first)), 2)
    )


# --------------------------------------------------------------- Kendall tau


@pytest.mark.parametrize("size", [0, 1, 2, 3, 8, 50, 255, 256, 257, 600])
def test_kendall_tau_equals_brute_force(size: int) -> None:
    rng = np.random.default_rng(derive_seed(7, "tests/shop/kendall", size))
    for _ in range(3 if size > 100 else 20):
        first, second = rng.permutation(size), rng.permutation(size)
        assert kendall_tau(first, second) == _brute_force(first.tolist(), second.tolist())


def test_kendall_tau_takes_any_distinct_integers_as_items() -> None:
    rng = np.random.default_rng(3)
    items = rng.choice(10_000, 40, replace=False) - 5_000
    for _ in range(20):
        first, second = rng.permutation(items), rng.permutation(items)
        assert kendall_tau(first, second) == _brute_force(first.tolist(), second.tolist())


def test_kendall_tau_is_zero_for_equal_orderings_and_every_pair_for_reversed_ones() -> None:
    ordering = np.random.default_rng(1).permutation(300)
    assert kendall_tau(ordering, ordering) == 0
    assert kendall_tau(ordering, ordering[::-1]) == 300 * 299 // 2
    assert kendall_tau([0, 1, 2], [1, 0, 2]) == 1


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ([0, 1, 2], [0, 1], "one size"),
        ([[0, 1]], [[1, 0]], "one size"),
        ([0, 1, 1], [1, 0, 1], "distinct"),
        ([0, 1, 2], [0, 1, 3], "same distinct items"),
        ([0.0, 1.0], [1.0, 0.0], "integers"),
    ],
)
def test_kendall_tau_refuses_what_is_not_two_orderings_of_one_set(
    first: Any, second: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        kendall_tau(first, second)


# ----------------------------------------------------- occurrence permutation


def test_occurrence_permutation_lists_the_slot_of_each_occurrence() -> None:
    # Jobs 0 and 1 have two operations each, job 2 one: slots 0-1, 2-3, and 4.
    slot_job = [0, 0, 1, 1, 2]
    permutation = occurrence_permutation([1, 0, 2, 0, 1], slot_job)
    assert permutation.tolist() == [2, 0, 4, 1, 3]
    assert np.asarray(slot_job)[permutation].tolist() == [1, 0, 2, 0, 1]


def test_occurrence_permutation_is_a_permutation_that_gives_the_sequence_back() -> None:
    shop = ta01()
    rng = np.random.default_rng(8)
    for _ in range(20):
        sequence = rng.permutation(shop.slot_job)
        permutation = occurrence_permutation(sequence, shop.slot_job)
        assert np.array_equal(np.sort(permutation), np.arange(shop.size))
        assert np.array_equal(shop.slot_job[permutation], sequence)


def test_occurrence_permutation_refuses_a_sequence_of_other_jobs() -> None:
    with pytest.raises(ValueError, match="as often"):
        occurrence_permutation([0, 0, 1], [0, 1, 1])
    with pytest.raises(ValueError, match="job indices"):
        occurrence_permutation([0, 1], [0, 1, 1])
    with pytest.raises(ValueError, match="integers"):
        occurrence_permutation([0.0, 1.0], [0, 1])  # type: ignore[list-item]


def test_the_distance_of_job_sequences_counts_operations_placed_in_other_orders() -> None:
    shop = JobShop.from_problem(small_job_shop())
    slot_job = shop.slot_job
    sequence = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

    def distance(first: NDArray[np.intp], second: NDArray[np.intp]) -> int:
        return kendall_tau(
            occurrence_permutation(first, slot_job), occurrence_permutation(second, slot_job)
        )

    # Swapping two adjacent positions of different jobs moves the sequence by one.
    adjacent = sequence.copy()
    adjacent[[3, 4]] = adjacent[[4, 3]]
    assert distance(sequence, adjacent) == 1
    # Keys that sort into the same sequence are at distance zero.
    keys = KeyRepresentation(shop)
    genotype = keys.from_keys(np.linspace(0.05, 0.85, 9))
    reordered = genotype.copy()
    reordered[[0, 1]] = reordered[[1, 0]]  # two keys of job 0 trade places
    assert distance(keys.sequence(genotype), keys.sequence(reordered)) == 0
    # Moving the first job 0 to the end passes six operations of other jobs.
    moved = np.array([1, 2, 0, 1, 2, 0, 1, 2, 0])
    assert distance(sequence, moved) == 6


# ------------------------------------------------------------- move distances


class _Recording(SequenceRepresentation):
    """A representation that records the job sequences of its random solutions."""

    def __init__(self, shop: JobShop) -> None:
        super().__init__(shop, "insert")
        self.drawn: list[list[int]] = []

    def initial(self, rng: np.random.Generator) -> NDArray[np.intp]:
        sequence = super().initial(rng)
        self.drawn.append(sequence.tolist())
        return sequence


class _RecordingKeys(KeyRepresentation):
    def __init__(self, shop: JobShop) -> None:
        super().__init__(shop, "perturb", 0.2, 2)
        self.drawn: list[list[int]] = []

    def initial(self, rng: np.random.Generator) -> NDArray[np.float64]:
        keys = super().initial(rng)
        self.drawn.append(self.sequence(keys).tolist())
        return keys


def test_move_distances_measure_every_representation_on_the_same_solutions() -> None:
    shop = ta01()
    orders, keys = _Recording(shop), _RecordingKeys(shop)
    first = move_distances(orders, samples=30, seed=5)
    move_distances(keys, samples=30, seed=5)
    assert orders.drawn == keys.drawn
    solutions = np.random.default_rng(derive_seed(5, SOLUTION_STREAM))
    assert orders.drawn[0] == SequenceRepresentation(shop).initial(solutions).tolist()
    # The same seed gives the same distances; another seed other ones.
    assert np.array_equal(first, move_distances(orders, samples=30, seed=5))
    assert not np.array_equal(first, move_distances(orders, samples=30, seed=6))
    assert SOLUTION_STREAM != MOVE_STREAM


def test_move_distances_equal_brute_force_of_each_solution_and_its_move() -> None:
    shop = JobShop.from_problem(small_job_shop())
    representation = KeyRepresentation(shop, "redraw", 0.1, 2)
    distances = move_distances(representation, samples=25, seed=9)
    solutions = np.random.default_rng(derive_seed(9, SOLUTION_STREAM))
    moves = np.random.default_rng(derive_seed(9, MOVE_STREAM))
    for distance in distances:
        genotype = representation.initial(solutions)
        moved = representation.move(genotype, moves)
        first = occurrence_permutation(representation.sequence(genotype), shop.slot_job)
        second = occurrence_permutation(representation.sequence(moved), shop.slot_job)
        assert distance == _brute_force(first.tolist(), second.tolist())


def test_move_distances_refuse_too_few_samples() -> None:
    with pytest.raises(ValueError, match="samples"):
        move_distances(SequenceRepresentation(ta01()), samples=0, seed=1)


# ---------------------------------------------------------------- calibration


def _median(representation: Representation[Any], samples: int, seed: int) -> float:
    return float(np.median(move_distances(representation, samples=samples, seed=seed)))


def test_calibration_hits_the_median_of_the_swap_with_two_keys() -> None:
    shop = ta01()
    target = _median(SequenceRepresentation(shop, "swap"), 101, 3)

    def make(step: float) -> KeyRepresentation:
        return KeyRepresentation(shop, "perturb", step, 2)

    step = calibrate_step(make, target, 0.01, 1.0, samples=101, seed=3, tolerance=0.5)
    assert 0.01 < step < 1.0
    assert abs(_median(make(step), 101, 3) - target) <= 0.5
    # The calibration is deterministic for its seed.
    assert calibrate_step(make, target, 0.01, 1.0, samples=101, seed=3, tolerance=0.5) == step


def test_calibration_recovers_a_step_from_its_median() -> None:
    shop = ta01()

    def make(step: float) -> KeyRepresentation:
        return KeyRepresentation(shop, "perturb", step, 1)

    for known in (0.02, 0.1, 0.3):
        target = _median(make(known), 51, 4)
        step = calibrate_step(make, target, 1e-4, 2.0, samples=51, seed=4, tolerance=0.5)
        assert abs(_median(make(step), 51, 4) - target) <= 0.5


def test_calibration_does_not_draw_from_a_runs_streams() -> None:
    shop = ta01()
    search = Search(shop.problem, StopCriteria(max_evaluations=5), 7, INFO)
    run = ShopRun(search, shop, "active", KeyRepresentation(shop))
    states = (run.control.bit_generator.state, run.operators.bit_generator.state)
    calibrate_step(
        lambda step: KeyRepresentation(shop, "perturb", step),
        20.0,
        1e-3,
        2.0,
        samples=21,
        seed=1,
        tolerance=0.5,
    )
    assert (run.control.bit_generator.state, run.operators.bit_generator.state) == states


def test_calibration_refuses_a_target_its_bracket_cannot_reach() -> None:
    # One changed key moves one job: it never goes as far as a swap, which moves two.
    shop = ta01()
    target = _median(SequenceRepresentation(shop, "swap"), 51, 2)
    with pytest.raises(ValueError, match="does not lie between"):
        calibrate_step(
            lambda step: KeyRepresentation(shop, "perturb", step, 1),
            target,
            0.01,
            10.0,
            samples=51,
            seed=2,
            tolerance=0.5,
        )


def test_calibration_says_when_no_step_meets_the_tolerance() -> None:
    shop = ta01()

    def make(step: float) -> KeyRepresentation:
        return KeyRepresentation(shop, "perturb", step, 1)

    low, high = _median(make(0.01), 51, 6), _median(make(1.0), 51, 6)
    assert low + 1.0 < high
    # Medians of 51 whole distances are whole, so none lies within 0.1 of a half.
    target = np.floor((low + high) / 2.0) + 0.5
    with pytest.raises(ValueError, match=r"no step gives a median within 0\.1"):
        calibrate_step(make, target, 0.01, 1.0, samples=51, seed=6, tolerance=0.1)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"low": 0.0}, "0 < low < high"),
        ({"low": 2.0, "high": 1.0}, "0 < low < high"),
        ({"high": float("inf")}, "0 < low < high"),
        ({"target_median": -1.0}, "target_median"),
        ({"target_median": float("nan")}, "target_median"),
        ({"tolerance": -0.5}, "tolerance"),
        ({"samples": 0}, "samples"),
    ],
)
def test_calibration_refuses_bad_arguments(arguments: dict[str, Any], message: str) -> None:
    shop = JobShop.from_problem(small_job_shop())
    values: dict[str, Any] = {
        "target_median": 2.0,
        "low": 0.01,
        "high": 1.0,
        "samples": 11,
        "seed": 0,
        "tolerance": 0.5,
    } | arguments
    with pytest.raises(ValueError, match=message):
        calibrate_step(
            lambda step: KeyRepresentation(shop, "perturb", step),
            values.pop("target_median"),
            values.pop("low"),
            values.pop("high"),
            **values,
        )
