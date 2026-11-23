"""The job-sequence builders give the schedules their rules define, and never repair."""

from __future__ import annotations

import itertools
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dispatchatlas.core import (
    ModelError,
    Problem,
    Resource,
    SolverInfo,
    StopCriteria,
    Task,
    canonical_json,
    check_schedule,
    derive_seed,
    evaluate,
    load_schedule,
    save_schedule,
    schedule_from_dict,
    schedule_to_dict,
)
from dispatchatlas.solvers import scheduling
from dispatchatlas.solvers.evaluation import Evaluator, Score
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.decoding import (
    BUILDERS,
    BuilderName,
    JobSequenceDecoder,
    check_sequence,
    giffler_thompson,
    semi_active,
    sequence_from_keys,
)
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop
from tests.helpers import small_job_shop
from tests.solvers.shop.instances import (
    idle_while_ready,
    left_shift,
    random_job_shop,
    random_sequences,
    ta01,
)

# Job 2 first, then job 1, then job 0, on the small job shop, whose tasks are
# j0-o0, j0-o1, j0-o2, j1-o0, ..., j2-o2 at indices 0 to 8.
LAST_JOB_FIRST = [2, 2, 2, 1, 1, 1, 0, 0, 0]
FIRST_JOB_FIRST = [0, 0, 0, 1, 1, 1, 2, 2, 2]

# Worked by hand from each builder's rule: the start of every task, by task
# index, and the order in which the builder placed the tasks.
HAND_COMPUTED = [
    pytest.param(
        "semi-active",
        LAST_JOB_FIRST,
        (7, 13, 15, 5, 8, 9, 0, 3, 5),
        (6, 7, 8, 3, 4, 5, 0, 1, 2),
        17,
        id="semi-active-last-job-first",
    ),
    pytest.param(
        "active",
        LAST_JOB_FIRST,
        (5, 8, 10, 0, 2, 3, 0, 3, 5),
        (3, 6, 4, 7, 5, 8, 0, 1, 2),
        12,
        id="active-last-job-first",
    ),
    pytest.param(
        "non-delay",
        LAST_JOB_FIRST,
        (2, 7, 10, 0, 2, 3, 0, 5, 7),
        (6, 3, 4, 0, 5, 7, 8, 1, 2),
        12,
        id="non-delay-last-job-first",
    ),
    pytest.param(
        "semi-active",
        FIRST_JOB_FIRST,
        (0, 3, 5, 3, 7, 8, 12, 15, 17),
        (0, 1, 2, 3, 4, 5, 6, 7, 8),
        20,
        id="semi-active-first-job-first",
    ),
    pytest.param(
        "active",
        FIRST_JOB_FIRST,
        (0, 3, 5, 3, 7, 8, 0, 5, 8),
        (0, 6, 1, 3, 2, 7, 4, 8, 5),
        12,
        id="active-first-job-first",
    ),
    pytest.param(
        "non-delay",
        FIRST_JOB_FIRST,
        (0, 3, 5, 3, 7, 8, 0, 5, 8),
        (0, 6, 1, 3, 2, 7, 4, 5, 8),
        12,
        id="non-delay-first-job-first",
    ),
]

INFO = SolverInfo(
    id="job-sequence-decoding",
    name="Job-sequence decoding",
    family="random-sampling",
    summary="Decodes the job sequences it is given.",
    supports=JOB_SHOP_FEATURES,
    shape="job-shop",
)


def _small() -> JobShop:
    return JobShop.from_problem(small_job_shop())


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


# ------------------------------------------------------------- hand-computed


@pytest.mark.parametrize(("builder", "sequence", "starts", "placed", "makespan"), HAND_COMPUTED)
def test_each_builder_gives_its_hand_computed_schedule_of_the_small_job_shop(
    builder: BuilderName,
    sequence: list[int],
    starts: tuple[int, ...],
    placed: tuple[int, ...],
    makespan: int,
) -> None:
    shop = _small()
    decoded = JobSequenceDecoder(shop, builder).decode(sequence)
    assert decoded.starts == starts
    assert decoded.ends == tuple(s + d for s, d in zip(starts, shop.duration, strict=True))
    assert decoded.sequence == placed
    assert decoded.modes == (0,) * 9
    assert decoded.makespan == makespan
    assert Evaluator(shop.problem).score(decoded) == Score(0.0, makespan)


def test_the_hand_computed_schedules_differ_in_the_way_their_classes_do() -> None:
    shop = _small()
    semi = semi_active(shop, LAST_JOB_FIRST)
    active = giffler_thompson(shop, LAST_JOB_FIRST)
    # j1-o0 runs on m0 in [5, 7) and is ready at 0, yet m0 idles in [0, 3).
    assert left_shift(shop, semi) == "task 3 fits into the idle interval [0.0, 3.0)"
    assert left_shift(shop, active) is None
    # m0 idles in [2, 3) while j0-o0 is ready, for j2-o1 comes first in the priority.
    assert idle_while_ready(shop, active) == (
        "task 0 is ready at 0.0 while its machine idles in [2.0, 3.0)"
    )
    assert idle_while_ready(shop, giffler_thompson(shop, LAST_JOB_FIRST, non_delay=True)) is None


@pytest.mark.parametrize(
    ("priority", "placed"),
    [
        # a0 comes before b0, so m0 is the machine; c0 comes first there.
        ([2, 3, 0, 1], (2, 3, 0, 1)),
        # b0 comes before a0, so m1 is the machine; d0 comes first there.
        ([2, 3, 1, 0], (3, 2, 1, 0)),
    ],
)
def test_the_operation_first_in_the_priority_fixes_the_machine(
    priority: list[int], placed: tuple[int, ...]
) -> None:
    # a0 on m0 and b0 on m1 both complete first, at 2, and c0 and d0 can start
    # before then on the same machines. Taking the lowest machine, or the job of
    # lowest index, would schedule c0 first under both priorities.
    shop = _four_jobs()
    decoded = giffler_thompson(shop, priority)
    assert decoded.sequence == placed
    # With positive processing times the machine taken first changes only the
    # order of placement, not the schedule.
    assert decoded.starts == (3.0, 3.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("priority", "placed"), [([2, 3, 0, 1], (2, 3, 0, 1)), ([2, 3, 1, 0], (2, 3, 1, 0))]
)
def test_the_non_delay_builder_breaks_ties_of_the_earliest_start_by_priority(
    priority: list[int], placed: tuple[int, ...]
) -> None:
    # Every operation can start at 0; after c0 and d0, a0 and b0 can both start
    # at 3 on different machines.
    decoded = giffler_thompson(_four_jobs(), priority, non_delay=True)
    assert decoded.sequence == placed
    assert decoded.starts == (3.0, 3.0, 0.0, 0.0)


def test_the_active_builder_reaches_the_optimum_and_the_classes_nest() -> None:
    shop = _small()
    semi: set[tuple[float, ...]] = set()
    active: set[tuple[float, ...]] = set()
    non_delay: set[tuple[float, ...]] = set()
    best = {"semi-active": float("inf"), "active": float("inf")}
    for sequence in set(itertools.permutations(FIRST_JOB_FIRST)):
        decoded = semi_active(shop, sequence)
        semi.add(decoded.starts)
        best["semi-active"] = min(best["semi-active"], decoded.makespan)
        decoded = giffler_thompson(shop, sequence)
        active.add(decoded.starts)
        best["active"] = min(best["active"], decoded.makespan)
        non_delay.add(giffler_thompson(shop, sequence, non_delay=True).starts)
    # Every active schedule is semi-active and every non-delay one active, and
    # some active schedule is optimal.
    assert non_delay < active < semi
    assert best == {"semi-active": 11, "active": 11}


# ------------------------------------------------------------- random keys


def test_keys_sort_their_slots_stably() -> None:
    slots = np.array([0, 0, 1, 1])
    assert sequence_from_keys([0.4, 0.9, 0.4, 0.1], slots).tolist() == [1, 0, 1, 0]
    # Equal keys keep the order of their slots, not of their jobs.
    assert sequence_from_keys([0.5, 0.5], [1, 0]).tolist() == [1, 0]
    assert sequence_from_keys(np.full(4, 0.25), slots).tolist() == [0, 0, 1, 1]
    rng = np.random.default_rng(derive_seed(7, "tests/shop/tied-keys"))
    shop = _small()
    for _ in range(20):
        # Keys from a few values tie often.
        keys = 0.25 * rng.integers(0, 3, size=9)
        expected = [
            int(shop.slot_job[slot]) for slot in sorted(range(9), key=lambda s: (keys[s], s))
        ]
        sequence = sequence_from_keys(keys, shop.slot_job)
        assert sequence.tolist() == expected
        assert check_sequence(shop, sequence) == ()


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        ([0.1, 0.2], "expected 4 keys in one row, got shape"),
        ([[0.1, 0.2], [0.3, 0.4]], "expected 4 keys in one row, got shape"),
        ([0.1, float("nan"), 0.3, 0.4], "every key must be finite"),
        ([0.1, 0.2, float("inf"), 0.4], "every key must be finite"),
    ],
)
def test_keys_that_do_not_fit_the_slots_are_refused(keys: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sequence_from_keys(keys, [0, 0, 1, 1])


# ------------------------------------------------------------- invalid sequences


def test_check_sequence_names_every_defect() -> None:
    shop = _small()
    assert check_sequence(shop, FIRST_JOB_FIRST) == ()
    assert check_sequence(shop, np.array(LAST_JOB_FIRST, dtype=np.uint8)) == ()
    assert check_sequence(shop, [0, 0, 0, 1, 1, 1, 2, 2]) == (
        "it has 8 entries for the 9 operations",
        "job 2 occurs 2 times for its 3 operations",
    )
    assert check_sequence(shop, [0, 0, 0, 0, 1, 1, 2, 2, 2]) == (
        "job 0 occurs 4 times for its 3 operations",
        "job 1 occurs 2 times for its 3 operations",
    )
    assert check_sequence(shop, [-1, 0, 0, 1, 1, 1, 2, 2, 3]) == (
        "it names jobs outside 0..2: [-1, 3]",
        "job 0 occurs 2 times for its 3 operations",
        "job 2 occurs 2 times for its 3 operations",
    )
    # Neither of these is a sequence of integers, as the checks report.
    floats: Any = [0.0] * 9
    rows: Any = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    assert check_sequence(shop, floats) == ("a job sequence holds integers, got float64",)
    assert check_sequence(shop, rows) == (
        "a job sequence is one row of job indices, got shape (3, 3)",
    )


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize(
    "sequence",
    [
        [0, 0, 0, 1, 1, 1, 2, 2],
        [0, 0, 0, 0, 1, 1, 2, 2, 2],
        [-1, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 0, 0, 1, 1, 1, 2, 2, 3],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        [False] * 9,
        [],
    ],
    ids=["short", "job-0-too-often", "negative", "unknown-job", "floats", "booleans", "empty"],
)
def test_the_builders_refuse_what_is_not_a_job_sequence(
    builder: BuilderName, sequence: list[float]
) -> None:
    shop = _small()
    with pytest.raises(
        ModelError, match="not a job sequence of problem 'small-job-shop'"
    ) as caught:
        JobSequenceDecoder(shop, builder).decode(sequence)  # type: ignore[arg-type]
    assert caught.value.issues
    assert caught.value.issues == check_sequence(shop, sequence)  # type: ignore[arg-type]


@pytest.mark.parametrize("builder", BUILDERS)
def test_a_sequence_decodes_alike_as_a_list_a_tuple_and_any_integer_array(
    builder: BuilderName,
) -> None:
    decoder = JobSequenceDecoder(_small(), builder)
    expected = decoder.decode(LAST_JOB_FIRST)
    for sequence in (
        tuple(LAST_JOB_FIRST),
        np.array(LAST_JOB_FIRST, dtype=np.int32),
        np.array(LAST_JOB_FIRST, dtype=np.uint16),
        np.array(LAST_JOB_FIRST, dtype=np.intp),
    ):
        assert decoder.decode(sequence) == expected


def test_the_decoder_runs_the_builder_it_names() -> None:
    shop = _small()
    assert JobSequenceDecoder(shop, "semi-active").decode(LAST_JOB_FIRST) == semi_active(
        shop, LAST_JOB_FIRST
    )
    assert JobSequenceDecoder(shop, "active").decode(LAST_JOB_FIRST) == giffler_thompson(
        shop, LAST_JOB_FIRST
    )
    assert JobSequenceDecoder(shop, "non-delay").decode(LAST_JOB_FIRST) == giffler_thompson(
        shop, LAST_JOB_FIRST, non_delay=True
    )
    assert BUILDERS == ("semi-active", "active", "non-delay")
    with pytest.raises(ValueError, match="unknown builder 'greedy'"):
        JobSequenceDecoder(shop, "greedy")  # type: ignore[arg-type]


# ------------------------------------------------------------- random sequences


@pytest.mark.parametrize("shop", _shops(), ids=repr)
@pytest.mark.parametrize("builder", BUILDERS)
def test_random_sequences_decode_into_feasible_schedules(
    shop: JobShop, builder: BuilderName
) -> None:
    problem = shop.problem
    decoder = JobSequenceDecoder(shop, builder)
    evaluator = Evaluator(problem)
    for sequence in random_sequences(shop, 8, f"tests/shop/feasible/{problem.id}/{builder}"):
        decoded = decoder.decode(sequence)
        assert sorted(decoded.sequence) == list(range(shop.size))
        assert decoded.modes == (0,) * shop.size
        schedule = decoded.schedule(problem)
        assert check_schedule(problem, schedule) == ()
        assert evaluator.score(decoded) == Score(0.0, decoded.makespan)
        assert decoded.makespan == evaluate(problem, schedule)["makespan"]


@pytest.mark.parametrize("builder", BUILDERS)
def test_the_score_is_the_makespan_and_the_schedule_serialises(
    builder: BuilderName, tmp_path: Path
) -> None:
    shop = ta01()
    problem = shop.problem
    (sequence,) = random_sequences(shop, 1, f"tests/shop/serialise/{builder}")
    decoded = JobSequenceDecoder(shop, builder).decode(sequence)
    assert all(type(value) is int for value in decoded.sequence + decoded.modes)
    assert all(type(value) is float for value in decoded.starts + decoded.ends)
    # The makespan of a job shop: the latest completion of an operation.
    makespan = max(start + shop.duration[task] for task, start in enumerate(decoded.starts))
    assert Evaluator(problem).score(decoded) == Score(0.0, makespan)
    schedule = decoded.schedule(problem)
    assert schedule.makespan == makespan
    document = schedule_to_dict(schedule, problem.id)
    assert schedule_from_dict(json.loads(json.dumps(document, allow_nan=False))) == schedule
    path = save_schedule(schedule, tmp_path / "schedule.json", problem.id)
    assert load_schedule(path) == schedule
    assert canonical_json(document) == canonical_json(schedule_to_dict(schedule, problem.id))


@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_active_schedules_admit_no_left_shift(shop: JobShop) -> None:
    sequences = random_sequences(shop, 8, f"tests/shop/active/{shop.problem.id}")
    for sequence in sequences:
        assert left_shift(shop, giffler_thompson(shop, sequence)) is None
    # The check finds the left shifts that semi-active schedules leave.
    assert any(left_shift(shop, semi_active(shop, sequence)) for sequence in sequences)


@pytest.mark.parametrize("shop", _shops(), ids=repr)
def test_non_delay_schedules_never_idle_while_an_operation_is_ready(shop: JobShop) -> None:
    for sequence in random_sequences(shop, 8, f"tests/shop/non-delay/{shop.problem.id}"):
        decoded = giffler_thompson(shop, sequence, non_delay=True)
        assert idle_while_ready(shop, decoded) is None
        assert left_shift(shop, decoded) is None


def test_no_builder_repairs_a_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a job sequence went through the precedence repair")

    monkeypatch.setattr(scheduling, "precedence_feasible", refuse)
    monkeypatch.setattr(scheduling.Decoder, "decode", refuse)
    shop = ta01()
    for builder in BUILDERS:
        decoder = JobSequenceDecoder(shop, builder)
        search = Search(shop.problem, StopCriteria(max_evaluations=10), 0, INFO)
        values: list[float] = []
        with search:
            for sequence in random_sequences(shop, 12, f"tests/shop/no-repair/{builder}"):
                values.append(search.assess_built(partial(decoder.decode, sequence)).score.value)
        assert len(values) == search.evaluations == 10
        assert search.evaluator.decoder.repairs == 0
        assert search.best is not None
        assert search.best.score.value == min(values)


def test_decoding_ta01_takes_well_under_a_second() -> None:
    shop = ta01()
    sequences = random_sequences(shop, 5, "tests/shop/timing")
    for builder in BUILDERS:
        decoder = JobSequenceDecoder(shop, builder)
        started = time.perf_counter()
        for sequence in sequences:
            decoder.decode(sequence)
        # A generous bound: a decode of 225 operations takes milliseconds.
        assert (time.perf_counter() - started) / len(sequences) < 0.5, builder
