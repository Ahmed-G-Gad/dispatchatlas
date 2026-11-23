"""Both representations draw the same random solutions, never a null move, and recombine soundly."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from dispatchatlas.core import Dependency, ModelError, Problem, Resource, SolverError, Task
from dispatchatlas.solvers.shop.decoding import check_sequence, sequence_from_keys
from dispatchatlas.solvers.shop.instance import JobShop
from dispatchatlas.solvers.shop.representation import (
    KEY_MOVES,
    SEQUENCE_MOVES,
    KeyMove,
    KeyRepresentation,
    SequenceMove,
    SequenceRepresentation,
    job_order_crossover,
    parameterised_uniform_crossover,
    precedence_preservative_crossover,
    reflect,
)
from tests.helpers import small_job_shop
from tests.solvers.shop.instances import random_job_shop, ta01

DRAWS = 2000


def _small() -> JobShop:
    return JobShop.from_problem(small_job_shop())


def _uneven() -> JobShop:
    """Three jobs of one, two and four operations; the last revisits machine m0 at once."""
    routes = {"a": ("m0",), "b": ("m1", "m0"), "c": ("m0", "m0", "m1", "m0")}
    tasks: list[Task] = []
    dependencies: list[Dependency] = []
    for job, machines in routes.items():
        for step, machine in enumerate(machines):
            tasks.append(Task.on(f"{job}{step}", machine, float(step + 2)))
            if step:
                dependencies.append(Dependency(f"{job}{step - 1}", f"{job}{step}"))
    return JobShop.from_problem(
        Problem(
            "uneven-jobs",
            tasks=tuple(tasks),
            resources=(Resource("m0"), Resource("m1")),
            dependencies=tuple(dependencies),
        )
    )


def _one_job() -> JobShop:
    return JobShop.from_problem(
        Problem(
            "one-job",
            tasks=(Task.on("a0", "m0", 2), Task.on("a1", "m1", 3)),
            resources=(Resource("m0"), Resource("m1")),
            dependencies=(Dependency("a0", "a1"),),
        )
    )


def _shops() -> list[JobShop]:
    return [_small(), _uneven(), JobShop.from_problem(random_job_shop(3, revisits=True)), ta01()]


SHOPS = [
    pytest.param(_small, id="3x3"),
    pytest.param(_uneven, id="uneven"),
    pytest.param(ta01, id="ta01"),
]


def _swaps(sequence: NDArray[np.intp]) -> set[tuple[int, ...]]:
    """Every sequence one swap of two positions of different jobs gives."""
    results = set()
    for first, second in itertools.combinations(range(sequence.size), 2):
        if sequence[first] != sequence[second]:
            moved = sequence.copy()
            moved[first], moved[second] = sequence[second], sequence[first]
            results.add(tuple(moved.tolist()))
    return results


def _inserts(sequence: NDArray[np.intp]) -> set[tuple[int, ...]]:
    """Every sequence one insert gives, the sequence itself left out."""
    results = set()
    for source, target in itertools.permutations(range(sequence.size), 2):
        entries = sequence.tolist()
        entries.insert(target, entries.pop(source))
        results.add(tuple(entries))
    results.discard(tuple(sequence.tolist()))
    return results


# ------------------------------------------------------------ random solutions


@pytest.mark.parametrize("seed", [0, 7, 2026])
def test_both_representations_start_from_the_same_job_sequence(seed: int) -> None:
    for shop in _shops():
        keys_rng, orders_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        keys, orders = KeyRepresentation(shop), SequenceRepresentation(shop)
        for _ in range(5):
            genotype = keys.initial(keys_rng)
            sequence = orders.initial(orders_rng)
            assert np.array_equal(keys.sequence(genotype), sequence)
            assert np.array_equal(orders.sequence(sequence), sequence)
            assert check_sequence(shop, sequence) == ()
        # Both drew the same numbers, so their generators are in the same state.
        assert keys_rng.bit_generator.state == orders_rng.bit_generator.state


def test_the_initial_keys_are_one_uniform_draw_per_slot() -> None:
    shop = ta01()
    keys = KeyRepresentation(shop).initial(np.random.default_rng(11))
    draws = np.random.default_rng(11).random(shop.size)
    assert np.array_equal(keys, draws)
    assert np.array_equal(
        SequenceRepresentation(shop).initial(np.random.default_rng(11)),
        sequence_from_keys(draws, shop.slot_job),
    )


def test_every_job_sequence_can_be_drawn_as_a_random_solution() -> None:
    # The 3x3 job shop has 9! / (3! 3! 3!) = 1680 job sequences.
    shop = _small()
    rng = np.random.default_rng(5)
    representation = SequenceRepresentation(shop)
    drawn = {tuple(representation.initial(rng).tolist()) for _ in range(40_000)}
    assert len(drawn) == 1680


def test_from_keys_refuses_keys_that_do_not_fit() -> None:
    shop = _small()
    for representation in (KeyRepresentation(shop), SequenceRepresentation(shop)):
        with pytest.raises(ValueError, match="expected 9 keys"):
            representation.from_keys([0.5] * 8)
        with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
            representation.from_keys([0.5] * 8 + [1.5])
        with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
            representation.from_keys([0.5] * 8 + [float("nan")])


# --------------------------------------------------------------------- moves


@pytest.mark.parametrize("make", SHOPS)
@pytest.mark.parametrize("move", SEQUENCE_MOVES)
def test_sequence_moves_always_change_the_sequence(
    make: Callable[[], JobShop], move: SequenceMove
) -> None:
    shop = make()
    representation = SequenceRepresentation(shop, move)
    rng = np.random.default_rng(3)
    sequence = representation.initial(rng)
    small = shop.size <= 12
    for _ in range(DRAWS):
        before = sequence.copy()
        moved = representation.move(sequence, rng)
        assert np.array_equal(sequence, before), "the move changed its argument"
        assert not np.array_equal(moved, sequence)
        assert np.array_equal(np.sort(moved), shop.slot_job)
        changed = np.flatnonzero(moved != sequence)
        if move == "swap":
            # Two positions of different jobs exchanged their jobs.
            assert changed.size == 2
            first, second = changed
            assert sequence[first] != sequence[second]
            assert (moved[first], moved[second]) == (sequence[second], sequence[first])
        if small:
            # The result is one of the moves that change the sequence, enumerated.
            swaps, inserts = _swaps(sequence), _inserts(sequence)
            allowed = {"swap": swaps, "insert": inserts, "mixed": swaps | inserts}[move]
            assert tuple(moved.tolist()) in allowed
        sequence = moved


@pytest.mark.parametrize("move", ["swap", "insert"])
def test_every_move_that_changes_the_sequence_can_be_drawn(move: SequenceMove) -> None:
    shop = _small()
    representation = SequenceRepresentation(shop, move)
    rng = np.random.default_rng(17)
    sequence = representation.initial(rng)
    expected = _swaps(sequence) if move == "swap" else _inserts(sequence)
    drawn = {tuple(representation.move(sequence, rng).tolist()) for _ in range(6000)}
    assert drawn == expected


def test_mixed_moves_draw_swaps_and_inserts_alike() -> None:
    shop = ta01()
    representation = SequenceRepresentation(shop, "mixed")
    rng = np.random.default_rng(23)
    sequence = representation.initial(rng)
    swaps = 0
    for _ in range(DRAWS):
        changed = np.flatnonzero(representation.move(sequence, rng) != sequence)
        # A swap changes two positions; an insert does so only when it equals a
        # swap, which few inserts of ta01 do.
        swaps += changed.size == 2
    assert 0.45 * DRAWS < swaps < 0.6 * DRAWS


def test_a_sequence_move_is_refused_for_one_job_whose_keys_still_move() -> None:
    shop = _one_job()
    rng = np.random.default_rng(1)
    for move in SEQUENCE_MOVES:
        representation = SequenceRepresentation(shop, move)
        with pytest.raises(SolverError, match="one job"):
            representation.move(representation.initial(rng), rng)
    keys = KeyRepresentation(shop)
    genotype = keys.initial(rng)
    moved = keys.move(genotype, rng)
    assert not np.array_equal(moved, genotype)
    assert np.array_equal(keys.sequence(moved), keys.sequence(genotype))


@pytest.mark.parametrize("make", SHOPS)
@pytest.mark.parametrize("key_move", KEY_MOVES)
@pytest.mark.parametrize("key_count", [None, 1, 2, 3])
def test_key_moves_change_exactly_their_keys(
    make: Callable[[], JobShop], key_move: KeyMove, key_count: int | None
) -> None:
    shop = make()
    representation = KeyRepresentation(shop, key_move, 0.05, key_count)
    expected = key_count or {"redraw": 1, "perturb": 2}[key_move]
    assert representation.key_count == expected
    rng = np.random.default_rng(9)
    keys = representation.initial(rng)
    for _ in range(DRAWS):
        before = keys.copy()
        moved = representation.move(keys, rng)
        assert np.array_equal(keys, before), "the move changed its argument"
        assert np.count_nonzero(moved != keys) == expected
        assert ((moved >= 0.0) & (moved <= 1.0)).all()
        keys = moved


def test_the_key_moves_default_to_the_count_of_their_kind() -> None:
    # One key redrawn matches the insert, two keys perturbed the swap.
    shop = ta01()
    assert KeyRepresentation(shop, "redraw").key_count == 1
    assert KeyRepresentation(shop, "perturb").key_count == 2
    assert KeyRepresentation(shop).key_count == 2
    single = JobShop.from_problem(
        Problem("one-operation", tasks=(Task.on("a0", "m0", 2),), resources=(Resource("m0"),))
    )
    assert KeyRepresentation(single, "perturb").key_count == 1


def test_one_redrawn_key_moves_its_job_as_an_insert_does() -> None:
    # The key at rank i, given a value between the keys of ranks j - 1 and j among
    # the others, moves the job at position i of the sequence to position j.
    shop = ta01()
    representation = KeyRepresentation(shop, "redraw")
    rng = np.random.default_rng(67)
    for _ in range(200):
        keys = representation.initial(rng)
        sequence = representation.sequence(keys).tolist()
        ranked = np.argsort(keys, kind="stable")
        source, target = (int(position) for position in rng.choice(shop.size, 2, replace=False))
        others = np.delete(keys[ranked], source)
        low = others[target - 1] if target > 0 else 0.0
        high = others[target] if target < others.size else 1.0
        moved = keys.copy()
        moved[ranked[source]] = (low + high) / 2.0
        sequence.insert(target, sequence.pop(source))
        assert representation.sequence(moved).tolist() == sequence


def test_perturbed_keys_are_reflected_into_the_unit_interval_not_clipped() -> None:
    shop = _small()
    representation = KeyRepresentation(shop, "perturb", 4.0, shop.size)
    rng = np.random.default_rng(31)
    keys = np.full(shop.size, 0.999)
    values = np.concatenate([representation.move(keys, rng) for _ in range(500)])
    assert ((values >= 0.0) & (values <= 1.0)).all()
    # Clipping would have put about half of these steps on a bound.
    assert np.count_nonzero((values == 0.0) | (values == 1.0)) == 0
    assert np.unique(values).size == values.size
    # A step this long spreads the keys over the whole interval.
    assert np.histogram(values, bins=4, range=(0.0, 1.0))[0].min() > 0.2 * values.size


def test_reflect_folds_values_into_the_unit_interval() -> None:
    values = [0.0, 0.3, 1.0, -0.2, 1.3, 2.5, -1.25, 2.0, 3.0, -3.0, 7.75]
    folded = reflect(values)
    assert folded.tolist() == pytest.approx(
        [0.0, 0.3, 1.0, 0.2, 0.7, 0.5, 0.75, 0.0, 1.0, 1.0, 0.25]
    )
    inside = np.random.default_rng(2).random(1000)
    assert np.array_equal(reflect(inside), inside)


def test_a_step_too_small_to_change_a_key_moves_it_to_the_adjacent_float() -> None:
    shop = _small()
    representation = KeyRepresentation(shop, "perturb", 1e-300, 2)
    rng = np.random.default_rng(4)
    keys = np.linspace(0.1, 0.9, shop.size)
    moved = representation.move(keys, rng)
    changed = np.flatnonzero(moved != keys)
    assert changed.size == 2
    towards = np.where(keys[changed] < 0.5, 1.0, 0.0)
    assert np.array_equal(moved[changed], np.nextafter(keys[changed], towards))


def test_a_key_move_that_passes_keys_of_its_own_job_only_leaves_the_sequence() -> None:
    # Whether a move changes the job sequence is a consequence, not a rule: the
    # keys of slot 0 and of slot 1, both of job 0, trade places in the sort.
    shop = _small()
    representation = KeyRepresentation(shop, "redraw", 0.1, 1)
    keys = np.array([0.10, 0.12, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    moved = keys.copy()
    moved[0] = 0.13
    assert np.array_equal(representation.sequence(moved), representation.sequence(keys))


def test_the_representations_refuse_genotypes_of_the_other() -> None:
    shop = _small()
    keys, orders = KeyRepresentation(shop), SequenceRepresentation(shop)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="keys"):
        keys.sequence(orders.initial(rng))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="keys"):
        keys.move(orders.initial(rng), rng)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="job indices"):
        orders.sequence(keys.initial(rng))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="job indices"):
        orders.move(keys.initial(rng), rng)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"key_move": "swap"}, "unknown key move"),
        ({"key_step": 0.0}, "key_step"),
        ({"key_step": -1.0}, "key_step"),
        ({"key_step": float("inf")}, "key_step"),
        ({"key_step": float("nan")}, "key_step"),
        ({"key_count": 0}, "key_count"),
        ({"key_count": 10}, "key_count"),
        ({"key_count": 1.5}, "key_count"),
        ({"key_count": True}, "key_count"),
    ],
)
def test_key_representations_refuse_bad_parameters(
    parameters: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        KeyRepresentation(_small(), **parameters)  # type: ignore[arg-type]


def test_sequence_representations_refuse_an_unknown_move_or_crossover() -> None:
    with pytest.raises(ValueError, match="unknown sequence move"):
        SequenceRepresentation(_small(), "reverse")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown crossover"):
        SequenceRepresentation(_small(), crossover="ox")  # type: ignore[arg-type]


# ------------------------------------------------------------ recombination


def _job_order_reference(
    first: NDArray[np.intp], second: NDArray[np.intp], kept: NDArray[np.bool_]
) -> list[int]:
    """JOX by its definition: kept jobs where ``first`` has them, the rest in ``second``'s order."""
    others = iter([job for job in second.tolist() if not kept[job]])
    return [job if kept[job] else next(others) for job in first.tolist()]


@pytest.mark.parametrize("make", SHOPS)
@pytest.mark.parametrize("inheritance", [0.3, 0.5, 0.8])
def test_job_order_crossover_follows_its_definition(
    make: Callable[[], JobShop], inheritance: float
) -> None:
    shop = make()
    representation = SequenceRepresentation(shop, crossover="jox")
    rng = np.random.default_rng(13)
    for _ in range(300):
        first, second = representation.initial(rng), representation.initial(rng)
        before = (first.copy(), second.copy())
        replay = copy.deepcopy(rng)
        child = representation.recombine(first, second, rng, inheritance)
        kept = replay.random(shop.jobs) < inheritance
        assert replay.bit_generator.state == rng.bit_generator.state
        assert child.tolist() == _job_order_reference(first, second, kept)
        assert np.array_equal(np.sort(child), shop.slot_job)
        assert np.array_equal(first, before[0])
        assert np.array_equal(second, before[1])


def test_job_order_crossover_with_inheritance_one_or_zero_copies_a_parent() -> None:
    shop = ta01()
    representation = SequenceRepresentation(shop)
    rng = np.random.default_rng(29)
    first, second = representation.initial(rng), representation.initial(rng)
    assert np.array_equal(job_order_crossover(first, second, rng, 1.0), first)
    assert np.array_equal(job_order_crossover(first, second, rng, 0.0), second)


@pytest.mark.parametrize("inheritance", [0.0, 0.3, 0.7, 1.0])
def test_parameterised_uniform_crossover_takes_each_key_from_a_parent(inheritance: float) -> None:
    shop = ta01()
    representation = KeyRepresentation(shop)
    rng = np.random.default_rng(37)
    taken = 0
    for _ in range(100):
        first, second = representation.initial(rng), representation.initial(rng)
        replay = copy.deepcopy(rng)
        child = representation.recombine(first, second, rng, inheritance)
        mask = replay.random(shop.size) < inheritance
        assert np.array_equal(child, np.where(mask, first, second))
        assert ((child == first) | (child == second)).all()
        taken += int(np.count_nonzero(child == first))
    assert taken / (100 * shop.size) == pytest.approx(inheritance, abs=0.02)


def test_the_crossovers_refuse_parents_that_do_not_match() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="shapes"):
        parameterised_uniform_crossover([0.1, 0.2], [0.3], rng, 0.5)
    for crossover in (job_order_crossover, precedence_preservative_crossover):
        with pytest.raises(ValueError, match="same non-negative job indices"):
            crossover([0, 0, 1], [0, 1, 1], rng, 0.5)
        with pytest.raises(ValueError, match="rows of job indices"):
            crossover([0.0, 1.0], [1.0, 0.0], rng, 0.5)  # type: ignore[list-item]
    for inheritance in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="inheritance"):
            job_order_crossover([0, 1], [1, 0], rng, inheritance)
        with pytest.raises(ValueError, match="inheritance"):
            precedence_preservative_crossover([0, 1], [1, 0], rng, inheritance)
        with pytest.raises(ValueError, match="inheritance"):
            parameterised_uniform_crossover([0.1], [0.2], rng, inheritance)


def _operations(sequence: list[int]) -> list[tuple[int, int]]:
    """The operations a job sequence lists: its k-th occurrence of job j is ``(j, k)``."""
    seen: dict[int, int] = {}
    operations = []
    for job in sequence:
        operations.append((job, seen.get(job, 0)))
        seen[job] = seen.get(job, 0) + 1
    return operations


def _precedence_reference(first: list[int], second: list[int], mask: list[bool]) -> list[int]:
    """PPX as its definition reads: take the leftmost gene left in the parent the mask
    names, and delete the leftmost occurrence of its job from both parents."""
    left, right = list(first), list(second)
    child = []
    for from_first in mask:
        job = (left if from_first else right)[0]
        child.append(job)
        left.remove(job)
        right.remove(job)
    return child


@pytest.mark.parametrize("make", SHOPS)
@pytest.mark.parametrize("inheritance", [0.2, 0.5, 0.9])
def test_precedence_preservative_crossover_follows_its_definition(
    make: Callable[[], JobShop], inheritance: float
) -> None:
    shop = make()
    representation = SequenceRepresentation(shop)
    rng = np.random.default_rng(43)
    for _ in range(300):
        first, second = representation.initial(rng), representation.initial(rng)
        before = (first.copy(), second.copy())
        replay = copy.deepcopy(rng)
        child = representation.recombine(first, second, rng, inheritance)
        # One mask, one draw per position, and nothing else drawn.
        mask = (replay.random(shop.size) < inheritance).tolist()
        assert replay.bit_generator.state == rng.bit_generator.state
        assert child.tolist() == _precedence_reference(first.tolist(), second.tolist(), mask)
        assert child.dtype == np.intp
        assert np.array_equal(np.sort(child), shop.slot_job)
        assert np.array_equal(first, before[0])
        assert np.array_equal(second, before[1])


@pytest.mark.parametrize(
    "make", [pytest.param(_small, id="3x3"), pytest.param(_uneven, id="uneven")]
)
def test_every_precedence_of_a_ppx_child_comes_from_a_parent(make: Callable[[], JobShop]) -> None:
    shop = make()
    representation = SequenceRepresentation(shop)
    rng = np.random.default_rng(47)
    for _ in range(400):
        first, second = representation.initial(rng), representation.initial(rng)
        child = precedence_preservative_crossover(first, second, rng, float(rng.random()))
        places = [
            {operation: place for place, operation in enumerate(_operations(parent.tolist()))}
            for parent in (first, second)
        ]
        operations = _operations(child.tolist())
        assert sorted(operations) == sorted(places[0])
        for earlier, later in itertools.combinations(operations, 2):
            assert any(place[earlier] < place[later] for place in places), (earlier, later)


def test_ppx_with_inheritance_one_or_zero_copies_a_parent() -> None:
    shop = ta01()
    representation = SequenceRepresentation(shop)
    rng = np.random.default_rng(53)
    for _ in range(20):
        first, second = representation.initial(rng), representation.initial(rng)
        assert np.array_equal(precedence_preservative_crossover(first, second, rng, 1.0), first)
        assert np.array_equal(precedence_preservative_crossover(first, second, rng, 0.0), second)


def test_ppx_draws_one_mask_whatever_the_parents() -> None:
    shop = _uneven()
    representation = SequenceRepresentation(shop)
    sequence = representation.initial(np.random.default_rng(0))
    pairs = [
        (sequence, sequence),
        (sequence, sequence[::-1].copy()),
        (shop.slot_job, sequence),
        (representation.initial(np.random.default_rng(1)), sequence),
    ]
    for first, second in pairs:
        for inheritance in (0.0, 0.3, 1.0):
            rng = np.random.default_rng(59)
            precedence_preservative_crossover(first, second, rng, inheritance)
            expected = np.random.default_rng(59)
            expected.random(shop.size)
            assert rng.bit_generator.state == expected.bit_generator.state


def test_the_sequence_crossover_is_ppx_unless_jox_is_named() -> None:
    shop = ta01()
    rng = np.random.default_rng(61)
    first, second = (
        SequenceRepresentation(shop).initial(rng),
        SequenceRepresentation(shop).initial(rng),
    )
    for crossover, function in (
        ("ppx", precedence_preservative_crossover),
        ("jox", job_order_crossover),
    ):
        representation = SequenceRepresentation(shop, crossover=crossover)  # type: ignore[arg-type]
        replay = copy.deepcopy(rng)
        child = representation.recombine(first, second, rng, 0.6)
        assert np.array_equal(child, function(first, second, replay, 0.6))
    assert SequenceRepresentation(shop).crossover == "ppx"


# ------------------------------------------------------------------- follow


@pytest.mark.parametrize("make", SHOPS)
def test_follow_rearranges_the_keys_to_sort_into_the_sequence(make: Callable[[], JobShop]) -> None:
    shop = make()
    keys, orders = KeyRepresentation(shop), SequenceRepresentation(shop)
    rng = np.random.default_rng(41)
    for _ in range(200):
        genotype, target = keys.initial(rng), orders.initial(rng)
        followed = keys.follow(genotype, target)
        assert np.array_equal(keys.sequence(followed), target)
        assert np.array_equal(np.sort(followed), np.sort(genotype))
        # Keys that sort into the sequence already stay as they are.
        assert np.array_equal(keys.follow(genotype, keys.sequence(genotype)), genotype)
        copied = orders.follow(target, target.tolist())
        assert np.array_equal(copied, target)
        assert copied is not target


def test_follow_keeps_the_order_of_each_jobs_keys() -> None:
    # Job 0 holds slots 0 to 2, whose keys rank 2, 0, 1 among them.
    shop = _small()
    keys = np.array([0.50, 0.05, 0.20, 0.60, 0.70, 0.80, 0.30, 0.40, 0.90])
    followed = KeyRepresentation(shop).follow(keys, [0, 1, 0, 1, 0, 1, 2, 2, 2])
    assert followed.tolist() == [0.5, 0.05, 0.3, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]


def test_follow_refuses_what_is_not_a_job_sequence() -> None:
    shop = _small()
    keys = KeyRepresentation(shop)
    for representation, genotype in (
        (keys, keys.initial(np.random.default_rng(0))),
        (SequenceRepresentation(shop), shop.slot_job),
    ):
        with pytest.raises(ModelError, match="not a job sequence"):
            representation.follow(genotype, [0, 0, 0, 1, 1, 1, 2, 2])  # type: ignore[arg-type]
        with pytest.raises(ModelError, match="not a job sequence"):
            representation.follow(genotype, [0, 0, 0, 0, 1, 1, 2, 2, 2])  # type: ignore[arg-type]
