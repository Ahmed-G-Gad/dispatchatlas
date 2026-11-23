"""The run decodes through the budget check, holds two seed-derived streams, and measures itself."""

from __future__ import annotations

import hashlib
import struct
import time
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from dispatchatlas.core import SolverError, StopCriteria, StopReason, derive_seed
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.decoding import BUILDERS, BuilderName, JobSequenceDecoder
from dispatchatlas.solvers.shop.instance import JobShop
from dispatchatlas.solvers.shop.observers import TrajectoryObserver
from dispatchatlas.solvers.shop.representation import (
    KeyRepresentation,
    Representation,
    SequenceRepresentation,
)
from dispatchatlas.solvers.shop.run import (
    CONTROL_STREAM,
    OPERATOR_STREAM,
    Evaluated,
    ShopRun,
    schedule_digest,
)
from tests.helpers import small_job_shop
from tests.solvers.shop.instances import random_job_shop, ta01
from tests.solvers.shop.runs import INFO, Digests, climb, sample


def _small() -> JobShop:
    return JobShop.from_problem(small_job_shop())


def _run(
    shop: JobShop,
    representation: Representation[Any],
    *,
    evaluations: int = 30,
    seed: int = 7,
    builder: BuilderName = "active",
    observer: Any = None,
    target: float | None = None,
) -> ShopRun[Any]:
    search = Search(
        shop.problem, StopCriteria(max_evaluations=evaluations, target=target), seed, INFO
    )
    return ShopRun(search, shop, builder, representation, observer)


def _arms(shop: JobShop) -> list[Representation[Any]]:
    return [KeyRepresentation(shop, "perturb", 0.05), SequenceRepresentation(shop, "insert")]


# ------------------------------------------------------------------- streams


def test_the_streams_come_from_the_seed_alone() -> None:
    shop = _small()
    expected_control = np.random.default_rng(derive_seed(11, CONTROL_STREAM)).random(4)
    expected_operators = np.random.default_rng(derive_seed(11, OPERATOR_STREAM)).random(4)
    for representation in _arms(shop):
        for builder in BUILDERS:
            run = _run(shop, representation, seed=11, builder=builder)
            assert np.array_equal(run.control.random(4), expected_control)
            assert np.array_equal(run.operators.random(4), expected_operators)
    assert not np.array_equal(expected_control, expected_operators)


def test_the_run_holds_one_generator_per_stream() -> None:
    # Reading a stream twice continues it: it is not reseeded at each reading.
    shop = _small()
    run = _run(shop, SequenceRepresentation(shop), seed=3)
    for stream, namespace in ((run.control, CONTROL_STREAM), (run.operators, OPERATOR_STREAM)):
        fresh = np.random.default_rng(derive_seed(3, namespace))
        assert [stream.random(), stream.random()] == fresh.random(2).tolist()
    assert [run.control.random()] == np.random.default_rng(derive_seed(3, CONTROL_STREAM)).random(
        3
    )[2:].tolist()


@pytest.mark.parametrize("builder", BUILDERS)
def test_both_representations_start_from_the_same_schedule(builder: BuilderName) -> None:
    for shop in (_small(), ta01()):
        runs = [_run(shop, arm, builder=builder, evaluations=25) for arm in _arms(shop)]
        for run in runs:
            climb(run)
            run.finish()
        digests = {run.first_digest for run in runs}
        assert len(digests) == 1
        assert {run.search.extras["first_schedule_digest"] for run in runs} == {
            runs[0].first_digest.hex()  # type: ignore[union-attr]
        }


@pytest.mark.parametrize("builder", BUILDERS)
def test_random_sampling_gives_the_same_schedules_in_both_representations(
    builder: BuilderName,
) -> None:
    shop = ta01()
    observers = [Digests(), Digests()]
    runs = [
        _run(shop, arm, builder=builder, evaluations=60, observer=observer)
        for arm, observer in zip(_arms(shop), observers, strict=True)
    ]
    for run in runs:
        sample(run)
        run.finish()
    assert observers[0].digests == observers[1].digests
    assert len(set(observers[0].digests)) == 60
    keys, orders = (run.search.result() for run in runs)
    assert keys.schedule == orders.schedule
    assert keys.evaluations == orders.evaluations == 60
    assert [(p.evaluations, p.value) for p in keys.trace] == [
        (p.evaluations, p.value) for p in orders.trace
    ]
    assert keys.extras["repairs"] == orders.extras["repairs"] == 0


# ---------------------------------------------------------------- evaluation


class _Counting(SequenceRepresentation):
    """A representation that counts the job sequences the run asks it for."""

    def __init__(self, shop: JobShop) -> None:
        super().__init__(shop)
        self.calls = 0

    def sequence(self, genotype: NDArray[np.intp]) -> NDArray[np.intp]:
        self.calls += 1
        return super().sequence(genotype)


def test_the_budget_is_checked_before_the_sequence_is_computed() -> None:
    shop = _small()
    representation = _Counting(shop)
    run = _run(shop, representation, evaluations=3)
    with run.search:
        for _ in range(4):
            run.evaluate(representation.initial(run.operators))
    assert run.search.reason is StopReason.EVALUATION_LIMIT
    assert representation.calls == run.search.evaluations == 3


@pytest.mark.parametrize("builder", BUILDERS)
def test_each_evaluation_is_the_builders_schedule_counted_once(builder: BuilderName) -> None:
    shop = JobShop.from_problem(random_job_shop(2, jobs=5, machines=4))
    for representation in _arms(shop):
        run = _run(shop, representation, builder=builder, evaluations=40)
        decoder = JobSequenceDecoder(shop, builder)
        seen: list[Evaluated[Any]] = []
        with run.search:
            current = run.evaluate(representation.initial(run.operators))
            seen.append(current)
            while True:
                run.search.next_iteration()
                current = run.evaluate(
                    representation.move(current.genotype, run.operators), parent=current
                )
                seen.append(current)
        assert run.search.evaluations == len(seen) == 40
        for evaluated in seen:
            assert np.array_equal(evaluated.sequence, representation.sequence(evaluated.genotype))
            assert evaluated.decoded == decoder.decode(evaluated.sequence)
            assert evaluated.score == run.search.evaluator.score(evaluated.decoded)
            assert evaluated.digest == schedule_digest(evaluated.decoded)
        assert run.search.best is not None
        assert run.search.best.score == min(evaluated.score for evaluated in seen)
        assert run.search.evaluator.decoder.repairs == 0


def test_the_schedule_digest_hashes_the_start_times_as_little_endian_floats() -> None:
    shop = _small()
    decoder = JobSequenceDecoder(shop, "semi-active")
    decoded = decoder.decode([2, 2, 2, 1, 1, 1, 0, 0, 0])
    packed = struct.pack("<9d", *decoded.starts)
    assert schedule_digest(decoded) == hashlib.blake2b(packed, digest_size=16).digest()
    assert len(schedule_digest(decoded)) == 16
    # Equal schedules, from different sequences, have equal digests; others do not.
    same = decoder.decode([2, 2, 2, 1, 1, 1, 0, 0, 0])
    other = decoder.decode([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert schedule_digest(same) == schedule_digest(decoded) != schedule_digest(other)


def test_a_schedule_that_reaches_the_target_is_recorded_and_observed() -> None:
    shop = _small()
    observer = TrajectoryObserver(shop)
    run = _run(shop, SequenceRepresentation(shop), observer=observer, target=1e9)
    climb(run)
    run.finish()
    assert run.search.reason is StopReason.TARGET_REACHED
    assert run.search.evaluations == 1
    assert run.search.extras["observed_evaluations"] == 1
    assert run.search.best is not None
    assert run.first_digest == schedule_digest(run.search.best.decoded)


# ------------------------------------------------------------------ measures


def test_finish_writes_the_runs_measures() -> None:
    shop = ta01()
    for representation in _arms(shop):
        run = _run(shop, representation, evaluations=20)
        with run.search:
            first = run.evaluate(representation.initial(run.operators))
            while True:
                run.search.next_iteration()
                run.evaluate(representation.move(first.genotype, run.operators), parent=first)
        run.finish()
        extras = run.search.result().extras
        assert extras["first_schedule_digest"] == first.digest.hex()
        assert extras["repairs"] == 0
        assert extras["observer_seconds"] == 0.0
        assert extras["calibration_seconds"] == 0.0
        assert extras["calibration_evaluations"] == 0
        assert isinstance(extras["decode_seconds"], float)
        assert extras["decode_seconds"] > 0.0
        assert isinstance(extras["decode_share"], float)
        assert 0.0 < extras["decode_share"] <= 1.0
        assert "observed_evaluations" not in extras


def test_finish_refuses_a_run_whose_search_repaired_an_order() -> None:
    shop = _small()
    run = _run(shop, SequenceRepresentation(shop))
    with run.search:
        run.evaluate(run.representation.initial(run.operators))
        # A task order through the search's own decoder goes through the repair.
        run.search.evaluate(list(range(shop.size)))
    assert run.search.evaluator.decoder.repairs == 1
    with pytest.raises(SolverError, match="precedence repair"):
        run.finish()


def test_calibration_counts_toward_the_budget_but_is_neither_observed_nor_decoding() -> None:
    shop = _small()
    observer = TrajectoryObserver(shop)
    run = _run(shop, KeyRepresentation(shop), observer=observer, evaluations=12)
    representation = run.representation
    with run.search:
        with run.calibration():
            for _ in range(5):
                solution = run.evaluate(representation.initial(run.operators))
                run.evaluate(representation.move(solution.genotype, run.operators), parent=solution)
        assert run.decode_seconds == 0.0
        assert observer.evaluations == 0
        current = solution
        while True:
            run.search.next_iteration()
            current = run.evaluate(
                representation.move(current.genotype, run.operators), parent=current
            )
    run.finish()
    extras = run.search.extras
    assert run.search.evaluations == 12
    assert extras["calibration_evaluations"] == 10
    assert extras["observed_evaluations"] == extras["observed_moves"] == 2
    assert isinstance(extras["calibration_seconds"], float)
    assert extras["calibration_seconds"] > 0.0
    assert run.first_digest is not None


def test_calibration_blocks_do_not_nest() -> None:
    shop = _small()
    run = _run(shop, SequenceRepresentation(shop))
    with run.calibration(), pytest.raises(SolverError, match="do not nest"), run.calibration():
        pass
    with run.calibration():
        pass


class _Clock:
    """A clock that ticks one second at each reading, and jumps when told."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


class _Slow:
    """An observer that takes a thousand seconds of the clock for each evaluation."""

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock

    def observe(self, evaluated: Evaluated[Any], parent: Evaluated[Any] | None) -> None:
        self.clock.now += 1000.0

    def measures(self) -> dict[str, Any]:
        return {}


def test_observer_time_counts_toward_neither_the_decoding_nor_the_evaluations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(time, "perf_counter", clock)
    shop = _small()
    run = _run(shop, KeyRepresentation(shop), observer=_Slow(clock), evaluations=10)
    climb(run)
    run.finish()
    result = run.search.result()
    monkeypatch.undo()
    assert result.evaluations == 10
    # Each decode reads the clock twice; each observation takes 1,000 seconds and
    # two readings, and the measures two readings more.
    assert run.decode_seconds == 10.0
    assert run.observer_seconds == 10 * 1001.0 + 1.0
    share = result.extras["decode_share"]
    assert isinstance(share, float)
    assert 10.0 / (result.elapsed - run.observer_seconds) <= share <= 1.0


# ----------------------------------------------------------------- refusals


def test_the_run_refuses_a_representation_or_a_search_of_another_shop() -> None:
    shop, other = _small(), ta01()
    search = Search(shop.problem, StopCriteria(max_evaluations=5), 0, INFO)
    with pytest.raises(ValueError, match="another job shop"):
        ShopRun(search, shop, "active", SequenceRepresentation(other))
    with pytest.raises(ValueError, match="not the search's problem"):
        ShopRun(search, other, "active", SequenceRepresentation(other))
    with pytest.raises(ValueError, match="unknown builder"):
        ShopRun(search, shop, "giffler", SequenceRepresentation(shop))  # type: ignore[arg-type]


def test_the_run_refuses_a_negative_seed() -> None:
    shop = _small()
    search = Search(shop.problem, StopCriteria(max_evaluations=5), -1, INFO)
    with pytest.raises(ValueError, match="non-negative"):
        ShopRun(search, shop, "active", SequenceRepresentation(shop))
