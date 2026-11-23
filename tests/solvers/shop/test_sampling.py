"""Random sampling evaluates the same schedules on both representations, byte for byte."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from dispatchatlas.core import Problem, StopCriteria, StopReason, derive_seed
from dispatchatlas.core.serialization import canonical_json, schedule_to_dict
from dispatchatlas.solvers.shop.decoding import BUILDERS, BuilderName, JobSequenceDecoder
from dispatchatlas.solvers.shop.family import REPRESENTATIONS, RepresentationName
from dispatchatlas.solvers.shop.instance import JobShop
from dispatchatlas.solvers.shop.representation import SequenceRepresentation
from dispatchatlas.solvers.shop.run import CONTROL_STREAM, OPERATOR_STREAM, schedule_digest
from dispatchatlas.solvers.shop.sampling import ShopRandomSampling
from tests.helpers import chain_problem, small_job_shop
from tests.solvers.shop.contract import check_contract
from tests.solvers.shop.families import PAIRED_BUILDERS, Recording, one_job, untimed
from tests.solvers.shop.instances import ta01


def _ta01() -> Problem:
    return ta01().problem


PROBLEMS = [pytest.param(_ta01, id="ta01"), pytest.param(small_job_shop, id="3x3")]


def _serialised(problem: Problem, result: Any) -> str:
    assert result.schedule is not None
    return canonical_json(schedule_to_dict(result.schedule, problem.id))


# ------------------------------------------------------------------ identity


@pytest.mark.parametrize("builder", PAIRED_BUILDERS)
@pytest.mark.parametrize("make", PROBLEMS)
def test_random_sampling_arms_are_byte_identical(
    make: Callable[[], Problem], builder: BuilderName, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = make()
    recording = Recording(monkeypatch)
    stop = StopCriteria(max_evaluations=2000)
    keys, sequence = (
        ShopRandomSampling("keys", builder).solve(problem, stop, seed=7),
        ShopRandomSampling("sequence", builder).solve(problem, stop, seed=7),
    )
    assert _serialised(problem, keys) == _serialised(problem, sequence)
    keys_run, sequence_run = recording.runs()
    assert len(keys_run) == len(sequence_run) == 2000
    assert [e.digest for e in keys_run] == [e.digest for e in sequence_run]
    # Every schedule evaluated, not only its digest, is the same.
    assert [(e.decoded.sequence, e.decoded.starts, e.decoded.ends) for e in keys_run] == [
        (e.decoded.sequence, e.decoded.starts, e.decoded.ends) for e in sequence_run
    ]
    assert keys.evaluations == sequence.evaluations == 2000
    assert keys.stop_reason is sequence.stop_reason is StopReason.EVALUATION_LIMIT
    assert keys.extras["repairs"] == sequence.extras["repairs"] == 0
    assert keys.objectives == sequence.objectives
    assert [(p.evaluations, p.iteration, p.value) for p in keys.trace] == [
        (p.evaluations, p.iteration, p.value) for p in sequence.trace
    ]
    assert keys.extras["first_schedule_digest"] == keys_run[0].digest.hex()
    assert untimed(keys.extras) == untimed(sequence.extras)
    assert len(recording.finished) == 2


# --------------------------------------------------------------------- streams


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_random_sampling_draws_one_solution_per_evaluation_and_no_control_number(
    representation: RepresentationName, monkeypatch: pytest.MonkeyPatch
) -> None:
    shop = JobShop.from_problem(small_job_shop())
    recording = Recording(monkeypatch)
    ShopRandomSampling(representation, "active").solve(
        shop.problem, StopCriteria(max_evaluations=25), seed=4
    )
    (run,) = recording.finished
    control = np.random.default_rng(derive_seed(4, CONTROL_STREAM))
    operators = np.random.default_rng(derive_seed(4, OPERATOR_STREAM))
    assert run.control.bit_generator.state == control.bit_generator.state
    # The first solution comes first from the operator stream: it is the sequence
    # its first draws sort into, decoded by the builder.
    first = SequenceRepresentation(shop).initial(operators)
    assert run.first_digest == schedule_digest(JobSequenceDecoder(shop, "active").decode(first))
    # One solution per evaluation, and the one drawn for the evaluation the budget refused.
    operators.random(25 * shop.size)
    assert run.operators.bit_generator.state == operators.bit_generator.state


# -------------------------------------------------------------------- contract


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_random_sampling_passes_the_contract(
    representation: RepresentationName, builder: BuilderName
) -> None:
    for problem in (small_job_shop(), chain_problem(), _ta01()):
        result = check_contract(
            ShopRandomSampling, problem, representation=representation, builder=builder
        )
        assert result.extras["repairs"] == 0
        assert result.extras["observed_evaluations"] == 40


@pytest.mark.parametrize("evaluations", [1, 2, 57])
def test_every_random_sampling_run_ends_at_its_evaluation_limit(evaluations: int) -> None:
    for representation in REPRESENTATIONS:
        for builder in PAIRED_BUILDERS:
            result = ShopRandomSampling(representation, builder).solve(
                small_job_shop(), StopCriteria(max_evaluations=evaluations), seed=9
            )
            assert result.evaluations == evaluations
            assert result.stop_reason is StopReason.EVALUATION_LIMIT


def test_random_sampling_replays_from_its_seed() -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=80)
    for representation in REPRESENTATIONS:
        solver = ShopRandomSampling(representation, "semi-active")
        first, second, other = (solver.solve(problem, stop, seed=s) for s in (5, 5, 6))
        assert _serialised(problem, first) == _serialised(problem, second)
        assert [(p.evaluations, p.iteration, p.value) for p in first.trace] == [
            (p.evaluations, p.iteration, p.value) for p in second.trace
        ]
        assert untimed(first.extras) == untimed(second.extras)
        assert first.extras["first_schedule_digest"] != other.extras["first_schedule_digest"]


def test_random_sampling_runs_under_an_iteration_or_a_time_limit_alone() -> None:
    for representation in REPRESENTATIONS:
        solver = ShopRandomSampling(representation, "active")
        by_iterations = solver.solve(small_job_shop(), StopCriteria(max_iterations=6), seed=2)
        assert by_iterations.stop_reason is StopReason.ITERATION_LIMIT
        assert by_iterations.iterations == by_iterations.evaluations == 6
        by_time = solver.solve(small_job_shop(), StopCriteria(time_limit=0.05), seed=2)
        assert by_time.stop_reason is StopReason.TIME_LIMIT
        assert by_time.evaluations >= 1


# ---------------------------------------------------------------- edge cases


@pytest.mark.parametrize("operations", [1, 2, 3])
def test_random_sampling_evaluates_the_single_sequence_of_one_job_and_stops(
    operations: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = one_job(operations)
    recording = Recording(monkeypatch)
    for representation in REPRESENTATIONS:
        for builder in BUILDERS:
            result = ShopRandomSampling(representation, builder).solve(
                problem, StopCriteria(max_evaluations=30), seed=3
            )
            assert result.evaluations == 1
            assert result.stop_reason is StopReason.COMPLETED
            assert result.value == sum(step + 2 for step in range(operations))
            assert result.extras["repairs"] == 0
    assert len(recording.finished) == 2 * len(BUILDERS)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"representation": "permutation"}, "unknown representation"),
        ({"builder": "giffler"}, "unknown builder"),
    ],
)
def test_random_sampling_refuses_unknown_names(parameters: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShopRandomSampling(**parameters)
