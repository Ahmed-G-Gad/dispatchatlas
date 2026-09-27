"""Variation operators: hand-computed moves and crossovers, and permutation properties."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

import pytest

from dispatchatlas.core.model import Dependency, Mode, Problem, Resource, Task
from dispatchatlas.solvers.operators import (
    insert,
    mutate_modes,
    order_crossover,
    partially_mapped_crossover,
    random_insert,
    random_modes,
    random_order,
    random_swap,
    random_topological_order,
    reverse,
    swap,
    uniform_mode_crossover,
)
from tests.helpers import chain_problem, random_problem


class _Scripted(random.Random):
    """A generator whose ``sample`` returns scripted positions."""

    def __init__(self, *draws: list[int]) -> None:
        super().__init__(0)
        self.draws = list(draws)

    def sample(self, population: Sequence[Any], k: int, **_: Any) -> list[Any]:
        drawn = self.draws.pop(0)
        assert len(drawn) == k
        assert all(0 <= i < len(population) for i in drawn)
        return drawn


def _modal() -> Problem:
    return Problem(
        "modal",
        tasks=(
            Task.on("one", "m1", 1),
            Task("two", (Mode(1, {"m1": 1}), Mode(2, {"m2": 1}))),
            Task("three", (Mode(1, {"m1": 1}), Mode(2, {"m2": 1}), Mode(3, {"m1": 1}))),
        ),
        resources=(Resource("m1"), Resource("m2")),
    )


# -------------------------------------------------------------------- moves


def test_moves_by_hand_leave_their_input_untouched() -> None:
    order = (0, 1, 2, 3, 4)
    assert swap(order, 1, 3) == [0, 3, 2, 1, 4]
    assert insert(order, 1, 3) == [0, 2, 3, 1, 4]
    assert insert(order, 3, 1) == [0, 3, 1, 2, 4]
    assert insert(order, 0, 4) == [1, 2, 3, 4, 0]
    assert reverse(order, 3, 1) == [0, 3, 2, 1, 4]
    assert reverse(order, 0, 4) == [4, 3, 2, 1, 0]
    assert reverse(order, 2, 2) == list(order)
    assert order == (0, 1, 2, 3, 4)


def test_random_moves_use_two_distinct_positions() -> None:
    order = [5, 6, 7, 8, 9]
    assert random_swap(order, _Scripted([3, 1])) == [5, 8, 7, 6, 9]
    assert random_insert(order, _Scripted([3, 1])) == [5, 8, 6, 7, 9]
    rng = random.Random(4)
    for _ in range(200):
        swapped = random_swap(order, rng)
        assert sorted(swapped) == order
        assert sum(a != b for a, b in zip(swapped, order, strict=True)) == 2
        inserted = random_insert(order, rng)
        assert sorted(inserted) == order
        assert inserted != order
    assert order == [5, 6, 7, 8, 9]
    for short in ([], [3]):
        assert random_swap(short, rng) == short
        assert random_swap(short, rng) is not short
        assert random_insert(short, rng) == short


def test_random_orders_are_seeded_permutations() -> None:
    problem = random_problem(1, tasks=7)
    first = random_order(problem, random.Random(9))
    assert sorted(first) == list(range(7))
    assert first == random_order(problem, random.Random(9))
    three = Problem("three", tasks=chain_problem().tasks, resources=chain_problem().resources)
    rng = random.Random(0)
    seen = Counter(tuple(random_order(three, rng)) for _ in range(600))
    assert len(seen) == 6


def test_random_topological_orders_reach_every_linear_extension() -> None:
    problem = chain_problem()
    rng = random.Random(2)
    seen = Counter(tuple(random_topological_order(problem, rng)) for _ in range(300))
    assert set(seen) == {(0, 1, 2), (0, 2, 1), (2, 0, 1)}
    diamond = Problem(
        "diamond",
        tasks=tuple(Task.on(name, "m", 1) for name in "abcd"),
        resources=(Resource("m"),),
        dependencies=(
            Dependency("a", "b"),
            Dependency("a", "c"),
            Dependency("b", "d"),
            Dependency("c", "d"),
        ),
    )
    assert {tuple(random_topological_order(diamond, rng)) for _ in range(100)} == {
        (0, 1, 2, 3),
        (0, 2, 1, 3),
    }
    for seed in range(20):
        dag = random_problem(seed, tasks=15, density=0.3)
        order = random_topological_order(dag, random.Random(seed))
        position = {task: n for n, task in enumerate(order)}
        assert sorted(order) == list(range(dag.size))
        for after, befores in enumerate(dag.predecessors):
            assert all(position[b] < position[after] for b in befores)


# --------------------------------------------------------------- crossovers


def test_order_crossover_by_hand() -> None:
    first = [0, 1, 2, 3, 4, 5, 6, 7]
    second = [7, 6, 5, 4, 3, 2, 1, 0]
    # The slice 2..4 comes from the first parent; the rest follows the second
    # parent's order, starting after the slice and wrapping around.
    assert order_crossover(first, second, _Scripted([4, 2])) == [6, 5, 2, 3, 4, 1, 0, 7]
    assert order_crossover(first, second, _Scripted([0, 7])) == first
    assert order_crossover([3], [3], random.Random(0)) == [3]


def test_partially_mapped_crossover_by_hand() -> None:
    # Goldberg and Lingle's example, with genes renumbered from zero.
    first = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    second = [8, 2, 6, 7, 1, 5, 4, 0, 3]
    child = partially_mapped_crossover(first, second, _Scripted([3, 6]))
    assert child == [8, 2, 1, 3, 4, 5, 6, 0, 7]
    assert partially_mapped_crossover(first, second, _Scripted([0, 8])) == first
    assert partially_mapped_crossover([1], [1], random.Random(0)) == [1]


@pytest.mark.parametrize("crossover", [order_crossover, partially_mapped_crossover])
def test_crossovers_return_permutations_and_keep_their_parents(crossover: Any) -> None:
    rng = random.Random(7)
    for size in range(2, 12):
        for _ in range(20):
            first, second = rng.sample(range(size), size), rng.sample(range(size), size)
            kept = (list(first), list(second))
            child = crossover(first, second, rng)
            assert sorted(child) == list(range(size))
            assert (first, second) == kept
            assert crossover(first, first, rng) == first


# -------------------------------------------------------------------- modes


def test_random_modes_cover_every_mode() -> None:
    problem = _modal()
    rng = random.Random(1)
    drawn = [random_modes(problem, rng) for _ in range(300)]
    assert {modes[0] for modes in drawn} == {0}
    assert {modes[1] for modes in drawn} == {0, 1}
    assert {modes[2] for modes in drawn} == {0, 1, 2}


def test_mode_mutation_moves_to_another_mode_at_the_given_rate() -> None:
    problem = _modal()
    rng = random.Random(5)
    start = [0, 1, 1]
    assert mutate_modes(problem, start, rng, 0.0) == start
    reached: set[int] = set()
    for _ in range(200):
        mutated = mutate_modes(problem, start, rng, 1.0)
        assert mutated[0] == 0
        assert mutated[1] == 0
        assert mutated[2] != 1
        assert 0 <= mutated[2] < 3
        reached.add(mutated[2])
    assert reached == {0, 2}
    assert start == [0, 1, 1]
    changed = sum(mutate_modes(problem, start, rng, 0.25)[2] != 1 for _ in range(2000))
    assert 400 < changed < 600


def test_uniform_mode_crossover_takes_each_gene_from_a_parent() -> None:
    rng = random.Random(3)
    first, second = [0] * 40, [1] * 40
    child = uniform_mode_crossover(first, second, rng)
    assert set(child) == {0, 1}
    assert 10 < sum(child) < 30
    mixed = uniform_mode_crossover([0, 1, 2], [2, 1, 0], rng)
    assert all(gene in pair for gene, pair in zip(mixed, [(0, 2), (1,), (2, 0)], strict=True))
    with pytest.raises(ValueError, match="zip"):
        uniform_mode_crossover([0, 1], [0], rng)
