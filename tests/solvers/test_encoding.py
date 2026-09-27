"""Vector encodings: order rules, mode keys, and round trips of ``VectorEncoding``."""

from __future__ import annotations

import random
from typing import Literal

import pytest

from dispatchatlas.core.model import Mode, Problem, Resource, Task
from dispatchatlas.solvers.encoding import (
    VectorEncoding,
    keys_from_order,
    largest_value_order,
    modes_from_keys,
    random_key_order,
)
from dispatchatlas.solvers.operators import random_modes, random_order
from tests.helpers import chain_problem, random_problem

RULES: tuple[Literal["random-key", "largest-value"], ...] = ("random-key", "largest-value")


def _flexible() -> Problem:
    """Tasks with one, two, and three modes."""
    return Problem(
        "flexible",
        tasks=(
            Task.on("a", "m1", 1),
            Task("b", (Mode(1, {"m1": 1}), Mode(2, {"m2": 1}))),
            Task.on("c", "m2", 3),
            Task("d", (Mode(1, {"m1": 1}), Mode(2, {"m2": 1}), Mode(3, {"m3": 1}))),
        ),
        resources=(Resource("m1"), Resource("m2"), Resource("m3")),
    )


def test_random_keys_sort_ascending_and_break_ties_by_index() -> None:
    assert random_key_order([0.7, 0.1, 0.5, 0.1]) == [1, 3, 2, 0]
    assert random_key_order([]) == []
    assert random_key_order([-3.0, 12.0, 0.0]) == [0, 2, 1]


def test_largest_values_sort_descending_and_break_ties_by_index() -> None:
    assert largest_value_order([0.7, 0.1, 0.5, 0.7]) == [0, 3, 2, 1]
    assert largest_value_order([-3.0, 12.0, 0.0]) == [1, 2, 0]


def test_keys_from_an_order_reproduce_it() -> None:
    assert keys_from_order([2, 0, 1]) == [1 / 3, 2 / 3, 0.0]
    rng = random.Random(0)
    for size in range(1, 30):
        order = rng.sample(range(size), size)
        keys = keys_from_order(order)
        assert all(0.0 <= key < 1.0 for key in keys)
        assert random_key_order(keys) == order


def test_mode_keys_select_floor_of_key_times_count_and_wrap() -> None:
    problem = _flexible()
    assert modes_from_keys(problem, [0.9, 0.49, 0.2, 0.34]) == [0, 0, 0, 1]
    assert modes_from_keys(problem, [0.0, 0.5, 0.99, 0.67]) == [0, 1, 0, 2]
    # Keys outside [0, 1) wrap: 1.5 acts as 0.5 and -0.25 as 0.75.
    assert modes_from_keys(problem, [5.0, 1.5, -7.0, -0.25]) == [0, 1, 0, 2]
    assert modes_from_keys(problem, [0.0, -1e-20, 0.0, -1e-20]) == [0, 1, 0, 2]
    with pytest.raises(ValueError, match="zip"):
        modes_from_keys(problem, [0.1, 0.2])


@pytest.mark.parametrize("rule", RULES)
def test_vectors_hold_a_key_per_task_and_per_flexible_task(
    rule: Literal["random-key", "largest-value"],
) -> None:
    problem = _flexible()
    searched = VectorEncoding(problem, rule=rule)
    assert (searched.rule, searched.modes) == (rule, "keys")
    assert searched.flexible == (1, 3)
    assert searched.dimension == 6
    built = VectorEncoding(problem, rule=rule, modes="builder")
    assert built.flexible == ()
    assert built.dimension == 4
    single = VectorEncoding(chain_problem(), rule=rule)
    assert single.flexible == ()
    assert single.dimension == 3


def test_decoding_by_hand() -> None:
    problem = _flexible()
    vector = [0.4, 0.1, 0.9, 0.3, 0.75, 0.5]
    assert VectorEncoding(problem).decode(vector) == ([1, 3, 0, 2], [0, 1, 0, 1])
    assert VectorEncoding(problem, rule="largest-value").decode(vector) == (
        [2, 0, 3, 1],
        [0, 1, 0, 1],
    )
    assert VectorEncoding(problem, modes="builder").decode(vector[:4]) == ([1, 3, 0, 2], None)
    assert VectorEncoding(chain_problem()).decode([0.3, 0.2, 0.1]) == ([2, 1, 0], None)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("source", ["keys", "builder"])
def test_encoded_vectors_decode_to_the_encoded_solution(
    rule: Literal["random-key", "largest-value"], source: Literal["keys", "builder"]
) -> None:
    for seed in range(15):
        problem = random_problem(seed, tasks=1 + (seed * 7) % 16, max_modes=3)
        encoding = VectorEncoding(problem, rule=rule, modes=source)
        rng = random.Random(seed)
        for _ in range(5):
            order = random_order(problem, rng)
            modes = random_modes(problem, rng)
            vector = encoding.encode(order, modes)
            assert len(vector) == encoding.dimension
            assert all(0.0 <= value <= 1.0 for value in vector)
            decoded_order, decoded_modes = encoding.decode(vector)
            assert decoded_order == order
            if source == "builder" or not encoding.flexible:
                assert decoded_modes is None
            else:
                assert decoded_modes == modes


@pytest.mark.parametrize("rule", RULES)
def test_missing_or_automatic_modes_encode_as_mode_zero(
    rule: Literal["random-key", "largest-value"],
) -> None:
    encoding = VectorEncoding(_flexible(), rule=rule)
    order = [3, 2, 1, 0]
    for modes in (None, [-1, -1, -1, -1], [0, -1, 0, -1]):
        assert encoding.decode(encoding.encode(order, modes)) == (order, [0, 0, 0, 0])


def test_every_real_vector_decodes() -> None:
    problem = random_problem(3, tasks=9, max_modes=3)
    rng = random.Random(3)
    for rule in RULES:
        encoding = VectorEncoding(problem, rule=rule)
        for _ in range(50):
            vector = [rng.uniform(-10, 10) for _ in range(encoding.dimension)]
            order, modes = encoding.decode(vector)
            assert sorted(order) == list(range(problem.size))
            assert modes is None or all(
                0 <= mode < len(task.modes) for mode, task in zip(modes, problem.tasks, strict=True)
            )


def test_invalid_arguments_are_refused() -> None:
    problem = _flexible()
    with pytest.raises(ValueError, match="unknown order rule 'smallest'"):
        VectorEncoding(problem, rule="smallest")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown mode source 'random'"):
        VectorEncoding(problem, modes="random")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected a vector of length 6, got 4"):
        VectorEncoding(problem).decode([0.1, 0.2, 0.3, 0.4])
