"""Simulated annealing runs the same control logic on both representations, in step."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from dispatchatlas.core import Problem, StopCriteria, StopReason, derive_seed
from dispatchatlas.core.serialization import canonical_json, schedule_to_dict
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.annealing import ShopAnnealing
from dispatchatlas.solvers.shop.decoding import BUILDERS, BuilderName, JobSequenceDecoder
from dispatchatlas.solvers.shop.family import REPRESENTATIONS, RepresentationName
from dispatchatlas.solvers.shop.instance import JobShop
from dispatchatlas.solvers.shop.representation import (
    KeyMove,
    SequenceMove,
    SequenceRepresentation,
)
from dispatchatlas.solvers.shop.run import CONTROL_STREAM, OPERATOR_STREAM, schedule_digest
from dispatchatlas.solvers.shop.sampling import ShopRandomSampling
from dispatchatlas.solvers.trajectory.space import move_scale
from tests.helpers import chain_problem, small_job_shop
from tests.solvers.shop.contract import check_contract
from tests.solvers.shop.families import PAIRED_BUILDERS, Recording, one_job, untimed
from tests.solvers.shop.instances import ta01

# The matched pairs of moves: one key redrawn with the insert, two keys perturbed with the swap.
MATCHED: list[tuple[KeyMove, SequenceMove]] = [("redraw", "insert"), ("perturb", "swap")]

# A step of two perturbed keys within the range that calibrate_step gives on Taillard's shops.
STEP = 0.6


def _ta01() -> Problem:
    return ta01().problem


PROBLEMS = [pytest.param(_ta01, id="ta01"), pytest.param(small_job_shop, id="3x3")]


def _arms(builder: BuilderName = "active", **parameters: Any) -> list[ShopAnnealing]:
    """Return the solver on random keys and on job sequences, alike in every other parameter."""
    return [
        ShopAnnealing(representation, builder, **parameters) for representation in REPRESENTATIONS
    ]


def _serialised(problem: Problem, result: Any) -> str:
    assert result.schedule is not None
    return canonical_json(schedule_to_dict(result.schedule, problem.id))


# ------------------------------------------------------------- common start


@pytest.mark.parametrize(("key_move", "sequence_move"), MATCHED)
@pytest.mark.parametrize("builder", PAIRED_BUILDERS)
@pytest.mark.parametrize("make", PROBLEMS)
def test_both_annealing_arms_start_from_the_same_schedule(
    make: Callable[[], Problem],
    builder: BuilderName,
    key_move: KeyMove,
    sequence_move: SequenceMove,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = make()
    recording = Recording(monkeypatch)
    samples = 8
    keys, sequence = (
        solver.solve(problem, StopCriteria(max_evaluations=60), seed=7)
        for solver in _arms(
            builder,
            calibration_samples=samples,
            key_move=key_move,
            key_step=STEP,
            sequence_move=sequence_move,
        )
    )
    assert keys.extras["first_schedule_digest"] == sequence.extras["first_schedule_digest"]
    keys_run, sequence_run = recording.runs()
    assert keys_run[0].digest == sequence_run[0].digest
    assert keys.extras["first_schedule_digest"] == keys_run[0].digest.hex()
    # The calibration's random solutions are the same in both; only their moves differ.
    calibration = slice(1, 1 + 2 * samples, 2)
    assert [e.digest for e in keys_run[calibration]] == [
        e.digest for e in sequence_run[calibration]
    ]
    assert keys.extras["calibration_evaluations"] == 2 * samples
    assert sequence.extras["calibration_evaluations"] == 2 * samples


def test_the_first_solution_is_the_first_draw_of_the_operator_stream() -> None:
    shop = JobShop.from_problem(small_job_shop())
    operators = np.random.default_rng(derive_seed(12, OPERATOR_STREAM))
    first = SequenceRepresentation(shop).initial(operators)
    for builder in BUILDERS:
        expected = schedule_digest(JobSequenceDecoder(shop, builder).decode(first)).hex()
        for solver in _arms(builder, calibration_samples=3):
            result = solver.solve(shop.problem, StopCriteria(max_evaluations=20), seed=12)
            assert result.extras["first_schedule_digest"] == expected


# ------------------------------------------------------------------ budget


@pytest.mark.parametrize("evaluations", [1, 2, 10, 11, 12, 13, 150])
def test_every_annealing_run_ends_at_its_evaluation_limit(evaluations: int) -> None:
    # Five samples make ten calibration evaluations: the calibration ends at the eleventh.
    samples = 5
    for builder in PAIRED_BUILDERS:
        for key_move, sequence_move in MATCHED:
            results = [
                solver.solve(small_job_shop(), StopCriteria(max_evaluations=evaluations), seed=9)
                for solver in _arms(
                    builder,
                    calibration_samples=samples,
                    key_move=key_move,
                    key_step=STEP,
                    sequence_move=sequence_move,
                )
            ]
            for result in results:
                assert result.evaluations == evaluations
                assert result.stop_reason is StopReason.EVALUATION_LIMIT
                calibrated = min(evaluations - 1, 2 * samples)
                assert result.extras["calibration_evaluations"] == calibrated
                assert ("initial_temperature" in result.extras) is (evaluations > 2 * samples)
                assert ("last_temperature" in result.extras) is (evaluations > 2 * samples + 1)
                assert result.extras["repairs"] == 0


@pytest.mark.parametrize("make", PROBLEMS)
def test_the_last_move_is_decided_at_the_final_temperature(make: Callable[[], Problem]) -> None:
    # Twelve samples give every arm a move that changes the makespan of the 3x3 shop, so
    # that the temperatures are positive; one redrawn key there often leaves it as it was.
    for builder in PAIRED_BUILDERS:
        for solver in _arms(builder, calibration_samples=12):
            result = solver.solve(make(), StopCriteria(max_evaluations=120), seed=21)
            extras = result.extras
            initial, final = extras["initial_temperature"], extras["final_temperature"]
            assert isinstance(initial, float)
            assert isinstance(final, float)
            assert initial > final > 0.0
            assert math.isclose(initial / final, math.log(1 / 0.01) / math.log(1 / 0.5))
            last = extras["last_temperature"]
            assert isinstance(last, float)
            assert math.isclose(last, final, rel_tol=1e-12)
            assert extras["schedule_clock"] == "evaluations"
            assert extras["schedule_progress"] == 1.0


# ------------------------------------------------------------- calibration


@pytest.mark.parametrize(("key_move", "sequence_move"), MATCHED)
def test_the_temperatures_follow_the_calibration_heuristic(
    key_move: KeyMove, sequence_move: SequenceMove, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = Recording(monkeypatch)
    samples, start, end = 12, 0.4, 0.02
    for solver in _arms(
        "semi-active",
        initial_acceptance=start,
        final_acceptance=end,
        calibration_samples=samples,
        key_move=key_move,
        key_step=STEP,
        sequence_move=sequence_move,
    ):
        result = solver.solve(_ta01(), StopCriteria(max_evaluations=60), seed=11)
        calibration = recording.runs()[-1][1 : 1 + 2 * samples]
        solutions, moves = calibration[::2], calibration[1::2]
        # Each calibration move is a move of the random solution before it.
        assert all(s.parent is None for s in solutions)
        assert [m.parent for m in moves] == [s.digest for s in solutions]
        deltas = [m.value - s.value for s, m in zip(solutions, moves, strict=True)]
        scale = move_scale(deltas)
        assert scale > 0.0
        assert result.extras["initial_temperature"] == scale / math.log(1.0 / start)
        assert result.extras["final_temperature"] == scale / math.log(1.0 / end)


def test_both_temperatures_are_zero_when_no_calibration_move_changes_the_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Machine m1 runs both operations of job 0, for 5, while job 1 runs alone on m2 for 4,
    # so every schedule of the chain problem has a makespan of 5.
    recording = Recording(monkeypatch)
    for solver in _arms("semi-active", calibration_samples=4):
        result = solver.solve(chain_problem(), StopCriteria(max_evaluations=40), seed=1)
        assert {e.value for e in recording.runs()[-1]} == {5.0}
        assert result.extras["initial_temperature"] == 0.0
        assert result.extras["final_temperature"] == 0.0
        assert result.extras["last_temperature"] == 0.0
        # Moves that do not worsen are always accepted.
        assert result.extras["acceptance_share"] == 1.0


# ------------------------------------------------------- control stream


@pytest.mark.parametrize(
    "acceptance", [pytest.param(1e-300, id="cold"), 0.5, pytest.param(1 - 1e-12, id="hot")]
)
def test_the_metropolis_number_is_drawn_at_every_iteration(
    acceptance: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = Recording(monkeypatch)
    evaluations, samples = 90, 7
    for builder in PAIRED_BUILDERS:
        for solver in _arms(
            builder,
            initial_acceptance=acceptance,
            final_acceptance=acceptance,
            calibration_samples=samples,
        ):
            solver.solve(_ta01(), StopCriteria(max_evaluations=evaluations), seed=5)
    expected = np.random.default_rng(derive_seed(5, CONTROL_STREAM))
    # One number for every move decided: all but the first and the calibration's evaluations.
    expected.random(evaluations - 1 - 2 * samples)
    assert len(recording.finished) == 2 * len(PAIRED_BUILDERS)
    for run in recording.finished:
        assert run.control.bit_generator.state == expected.bit_generator.state


def test_a_cold_walk_never_worsens_and_a_hot_walk_takes_every_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = Recording(monkeypatch)
    samples = 6
    walk = slice(1 + 2 * samples, None)
    for representation in REPRESENTATIONS:
        cold = ShopAnnealing(
            representation,
            "semi-active",
            initial_acceptance=1e-300,
            final_acceptance=1e-300,
            calibration_samples=samples,
        ).solve(_ta01(), StopCriteria(max_evaluations=150), seed=8)
        moves = recording.runs()[-1][walk]
        values = [m.parent_value for m in moves if m.parent_value is not None]
        assert len(values) == len(moves)
        assert values == sorted(values, reverse=True)
        assert values[-1] < values[0]
        share = cold.extras["acceptance_share"]
        assert isinstance(share, float)
        assert 0.0 < share < 1.0
        hot = ShopAnnealing(
            representation,
            "semi-active",
            initial_acceptance=1 - 1e-12,
            final_acceptance=1 - 1e-12,
            calibration_samples=samples,
        ).solve(_ta01(), StopCriteria(max_evaluations=150), seed=8)
        moves = recording.runs()[-1][walk]
        # Every move is accepted: the next move starts from it.
        assert [m.parent for m in moves[1:]] == [m.digest for m in moves[:-1]]
        assert hot.extras["acceptance_share"] == 1.0


# --------------------------------------------------------------- the clock


class _Clock:
    """A clock that ticks one second at each reading."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def test_a_time_limit_never_changes_the_temperatures(monkeypatch: pytest.MonkeyPatch) -> None:
    # A run capped by time, even under the binding schedule, anneals as the uncapped run
    # does until the cap stops it: its evaluations are a prefix of the uncapped run's.
    recording = Recording(monkeypatch)
    monkeypatch.setattr(time, "perf_counter", _Clock())
    problem = _ta01()
    solver = ShopAnnealing("sequence", "semi-active", calibration_samples=5)
    capped_stop = StopCriteria(max_evaluations=400, time_limit=1500.0, schedule="binding")
    free = solver.solve(problem, StopCriteria(max_evaluations=400), seed=3)
    capped = solver.solve(problem, capped_stop, seed=3)
    free_run, capped_run = recording.runs()
    assert free.evaluations == 400
    assert capped.stop_reason is StopReason.TIME_LIMIT
    assert 100 < capped.evaluations < 300
    assert [e.digest for e in capped_run] == [e.digest for e in free_run[: len(capped_run)]]
    assert capped.extras["schedule_clock"] == "evaluations"
    # The check has power: a schedule on the time clock anneals differently under the cap.
    progress = Search.progress
    monkeypatch.setattr(
        Search, "progress", lambda self, *, deterministic=None: progress(self, deterministic=False)
    )
    solver.solve(problem, capped_stop, seed=3)
    timed_run = recording.runs()[2]
    assert [e.digest for e in timed_run] != [e.digest for e in free_run[: len(timed_run)]]


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_annealing_runs_under_an_iteration_or_a_time_limit_alone(
    representation: RepresentationName,
) -> None:
    samples = 4
    solver = ShopAnnealing(representation, "active", calibration_samples=samples)
    by_iterations = solver.solve(small_job_shop(), StopCriteria(max_iterations=6), seed=2)
    assert by_iterations.stop_reason is StopReason.ITERATION_LIMIT
    assert by_iterations.iterations == 6
    assert by_iterations.evaluations == 1 + 2 * samples + 6
    extras = by_iterations.extras
    last, final = extras["last_temperature"], extras["final_temperature"]
    assert isinstance(last, float)
    assert isinstance(final, float)
    assert math.isclose(last, final, rel_tol=1e-12)
    by_time = solver.solve(small_job_shop(), StopCriteria(time_limit=0.05), seed=2)
    assert by_time.stop_reason is StopReason.TIME_LIMIT
    assert by_time.extras["schedule_clock"] == "time"


# ------------------------------------------------------------------ results


@pytest.mark.parametrize(("key_move", "sequence_move"), [*MATCHED, ("redraw", "mixed")])
@pytest.mark.parametrize("builder", BUILDERS)
def test_annealing_passes_the_contract(
    builder: BuilderName, key_move: KeyMove, sequence_move: SequenceMove
) -> None:
    for representation in REPRESENTATIONS:
        for problem in (small_job_shop(), chain_problem()):
            result = check_contract(
                ShopAnnealing,
                problem,
                representation=representation,
                builder=builder,
                calibration_samples=5,
                key_move=key_move,
                key_step=STEP,
                sequence_move=sequence_move,
            )
            assert result.extras["repairs"] == 0
            # The observer sees the first solution and every move, not the calibration.
            assert result.extras["observed_evaluations"] == 40 - 10
            assert result.extras["observed_moves"] == 40 - 11


def test_annealing_replays_from_its_seed() -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=150)
    for solver in _arms("semi-active", calibration_samples=6):
        first, second = (solver.solve(problem, stop, seed=13) for _ in range(2))
        assert _serialised(problem, first) == _serialised(problem, second)
        assert [(p.evaluations, p.iteration, p.value) for p in first.trace] == [
            (p.evaluations, p.iteration, p.value) for p in second.trace
        ]
        assert untimed(first.extras) == untimed(second.extras)


def test_annealing_beats_random_sampling_at_equal_evaluations_on_ta01() -> None:
    problem = _ta01()
    stop = StopCriteria(max_evaluations=1000)
    for representation in REPRESENTATIONS:
        sampled = ShopRandomSampling(representation, "semi-active").solve(problem, stop, seed=7)
        annealed = ShopAnnealing(representation, "semi-active").solve(problem, stop, seed=7)
        assert annealed.value is not None
        assert sampled.value is not None
        assert annealed.value < sampled.value


@pytest.mark.parametrize("operations", [1, 2, 3])
def test_annealing_evaluates_the_single_sequence_of_one_job_and_stops(
    operations: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = one_job(operations)
    recording = Recording(monkeypatch)
    for builder in BUILDERS:
        for key_move, sequence_move in MATCHED:
            for solver in _arms(builder, key_move=key_move, sequence_move=sequence_move):
                result = solver.solve(problem, StopCriteria(max_evaluations=30), seed=3)
                assert result.evaluations == 1
                assert result.stop_reason is StopReason.COMPLETED
                assert result.value == sum(step + 2 for step in range(operations))
                assert result.extras["calibration_evaluations"] == 0
                assert "initial_temperature" not in result.extras
    assert len(recording.finished) == 2 * len(MATCHED) * len(BUILDERS)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"representation": "permutation"}, "unknown representation"),
        ({"builder": "giffler"}, "unknown builder"),
        ({"initial_acceptance": 0.0}, "initial_acceptance"),
        ({"initial_acceptance": 1.0}, "initial_acceptance"),
        ({"initial_acceptance": math.nan}, "initial_acceptance"),
        ({"final_acceptance": 0.0}, "final_acceptance"),
        ({"final_acceptance": 0.6}, "final_acceptance"),
        ({"final_acceptance": math.nan}, "final_acceptance"),
        ({"calibration_samples": 0}, "calibration_samples"),
        ({"calibration_samples": True}, "calibration_samples"),
        ({"calibration_samples": 2.0}, "calibration_samples"),
        ({"key_move": "shift"}, "unknown key move"),
        ({"key_step": 0.0}, "key_step"),
        ({"key_step": math.inf}, "key_step"),
        ({"key_step": True}, "key_step"),
        ({"key_step": "0.5"}, "key_step"),
        ({"sequence_move": "reverse"}, "unknown sequence move"),
    ],
)
def test_annealing_refuses_bad_parameters(parameters: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShopAnnealing(**parameters)


def test_the_key_parameters_are_checked_on_job_sequences_too() -> None:
    with pytest.raises(ValueError, match="key_step"):
        ShopAnnealing("sequence", key_step=-1.0)
    with pytest.raises(ValueError, match="sequence move"):
        ShopAnnealing("keys", sequence_move="reverse")  # type: ignore[arg-type]
