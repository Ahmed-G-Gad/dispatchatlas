"""The matched families sample the phenotype distance of their moves, and change nothing else."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import mean, median
from typing import Any

import pytest

from dispatchatlas.core import ConfigError, Problem, Solver, SolveResult, StopCriteria
from dispatchatlas.core.model import Scalar
from dispatchatlas.core.serialization import canonical_json, schedule_to_dict
from dispatchatlas.solvers import SolverRegistry
from dispatchatlas.solvers.shop.annealing import ShopAnnealing
from dispatchatlas.solvers.shop.family import REPRESENTATIONS
from dispatchatlas.solvers.shop.genetic import ShopSteadyStateGa
from dispatchatlas.solvers.shop.observers import TrajectoryObserver
from dispatchatlas.solvers.shop.run import ShopRun
from dispatchatlas.solvers.shop.sampling import ShopRandomSampling
from dispatchatlas.solvers.shop.swarm import ShopParticleSwarm
from tests.solvers.shop.families import Evaluation, Recording, untimed
from tests.solvers.shop.instances import ta01

EVALUATIONS = 64
SEED = 7

FAMILIES: list[tuple[str, type[Any], dict[str, Any], int]] = [
    ("sampling", ShopRandomSampling, {}, 0),
    ("annealing", ShopAnnealing, {"calibration_samples": 5}, 64 - 1 - 2 * 5),
    ("genetic", ShopSteadyStateGa, {"population_size": 6}, 64 - 6),
    ("swarm", ShopParticleSwarm, {"swarm_size": 6}, 64 - 6),
]
"""Each family with its name, the options that leave a long walk within 64 evaluations, and
the moves its observer sees there: the evaluations that have a parent."""

IDS = [name for name, *_ in FAMILIES]

SAMPLED = frozenset({"measured_moves", "phenotype_distance_mean", "phenotype_distance_median"})
"""The extras that depend on ``distance_every``."""

# The untimed extras of each family and arm on ta01, at seed 7 and 64 evaluations, with the
# options above and the active builder, when the observer measures the distance of every move.
DEFAULT_EXTRAS: dict[str, dict[str, Scalar]] = {
    "annealing/keys": {
        "acceptance_share": 0.8490566037735849,
        "calibration_evaluations": 10,
        "final_temperature": 9.663052222347352,
        "first_schedule_digest": "83ff177e14e6d8cfca3b26d6de095c59",
        "initial_temperature": 64.19992931955888,
        "last_temperature": 9.663052222347352,
        "measured_moves": 53,
        "observed_evaluations": 54,
        "observed_moves": 53,
        "phenotype_distance_mean": 15.867924528301886,
        "phenotype_distance_median": 1.0,
        "repairs": 0,
        "revisit_share": 0.48148148148148145,
        "schedule_clock": "evaluations",
        "schedule_progress": 1.0,
        "unchanged_move_share": 0.49056603773584906,
    },
    "annealing/sequence": {
        "acceptance_share": 0.7924528301886793,
        "calibration_evaluations": 10,
        "final_temperature": 7.4553886060058225,
        "first_schedule_digest": "83ff177e14e6d8cfca3b26d6de095c59",
        "initial_temperature": 49.53252973718775,
        "last_temperature": 7.4553886060058225,
        "measured_moves": 53,
        "observed_evaluations": 54,
        "observed_moves": 53,
        "phenotype_distance_mean": 15.566037735849056,
        "phenotype_distance_median": 6.0,
        "repairs": 0,
        "revisit_share": 0.35185185185185186,
        "schedule_clock": "evaluations",
        "schedule_progress": 1.0,
        "unchanged_move_share": 0.33962264150943394,
    },
    "genetic/keys": {
        "calibration_evaluations": 0,
        "first_schedule_digest": "83ff177e14e6d8cfca3b26d6de095c59",
        "measured_moves": 58,
        "observed_evaluations": 64,
        "observed_moves": 58,
        "phenotype_distance_mean": 9.689655172413794,
        "phenotype_distance_median": 0.0,
        "repairs": 0,
        "replacement_share": 0.896551724137931,
        "revisit_share": 0.703125,
        "unchanged_move_share": 0.7758620689655172,
    },
    "genetic/sequence": {
        "calibration_evaluations": 0,
        "first_schedule_digest": "83ff177e14e6d8cfca3b26d6de095c59",
        "measured_moves": 58,
        "observed_evaluations": 64,
        "observed_moves": 58,
        "phenotype_distance_mean": 9.517241379310345,
        "phenotype_distance_median": 0.0,
        "repairs": 0,
        "replacement_share": 0.8620689655172413,
        "revisit_share": 0.671875,
        "unchanged_move_share": 0.7241379310344828,
    },
    "sampling/keys": {
        "calibration_evaluations": 0,
        "first_schedule_digest": "83ff177e14e6d8cfca3b26d6de095c59",
        "measured_moves": 0,
        "observed_evaluations": 64,
        "observed_moves": 0,
        "repairs": 0,
        "revisit_share": 0.0,
    },
    "sampling/sequence": {
        "calibration_evaluations": 0,
        "first_schedule_digest": "83ff177e14e6d8cfca3b26d6de095c59",
        "measured_moves": 0,
        "observed_evaluations": 64,
        "observed_moves": 0,
        "repairs": 0,
        "revisit_share": 0.0,
    },
    "swarm/keys": {
        "calibration_evaluations": 0,
        "first_schedule_digest": "83ff177e14e6d8cfca3b26d6de095c59",
        "measured_moves": 58,
        "observed_evaluations": 64,
        "observed_moves": 58,
        "phenotype_distance_mean": 138.86206896551724,
        "phenotype_distance_median": 145.0,
        "reflected_share": 0.09099811676082863,
        "repairs": 0,
        "revisit_share": 0.0625,
        "unchanged_move_share": 0.06896551724137931,
    },
    "swarm/sequence": {
        "calibration_evaluations": 0,
        "first_schedule_digest": "83ff177e14e6d8cfca3b26d6de095c59",
        "measured_moves": 58,
        "observed_evaluations": 64,
        "observed_moves": 58,
        "phenotype_distance_mean": 30.844827586206897,
        "phenotype_distance_median": 8.0,
        "repairs": 0,
        "revisit_share": 0.359375,
        "unchanged_move_share": 0.3620689655172414,
    },
}


def _solve(problem: Problem, solver: Solver) -> SolveResult:
    return solver.solve(problem, StopCriteria(max_evaluations=EVALUATIONS), seed=SEED)


def _serialised(problem: Problem, result: SolveResult) -> str:
    assert result.schedule is not None
    return canonical_json(schedule_to_dict(result.schedule, problem.id))


def _trail(evaluations: list[Evaluation]) -> list[tuple[Any, ...]]:
    """Return what a run evaluated: each schedule, its digest and value, and its parent's."""
    return [
        (e.digest, e.value, e.parent, e.parent_value, e.decoded.sequence, e.decoded.starts)
        for e in evaluations
    ]


def _unsampled(extras: Mapping[str, Scalar]) -> dict[str, Scalar]:
    """Return the untimed extras that do not depend on ``distance_every``."""
    return {key: value for key, value in untimed(extras).items() if key not in SAMPLED}


def _observer(run: ShopRun[Any]) -> TrajectoryObserver:
    assert isinstance(run.observer, TrajectoryObserver)
    return run.observer


# ------------------------------------------------------------------ the default


@pytest.mark.parametrize("arm", REPRESENTATIONS)
@pytest.mark.parametrize(("name", "make", "options", "moves"), FAMILIES, ids=IDS)
def test_the_default_distance_every_reproduces_the_recorded_extras(
    name: str, make: type[Any], options: dict[str, Any], moves: int, arm: str
) -> None:
    problem = ta01().problem
    default = make(arm, "active", **options)
    assert default.distance_every == 1
    extras = untimed(_solve(problem, default).extras)
    recorded = DEFAULT_EXTRAS[f"{name}/{arm}"]
    assert extras.keys() == recorded.keys()
    for key, value in recorded.items():
        expected = pytest.approx(value, rel=1e-12) if isinstance(value, float) else value
        assert extras[key] == expected, key
    assert extras["observed_moves"] == extras["measured_moves"] == moves
    # Naming the default changes nothing.
    named = make(arm, "active", distance_every=1, **options)
    assert named == default
    assert untimed(_solve(problem, named).extras) == extras


# ------------------------------------------------------------------ the sampling


@pytest.mark.parametrize("arm", REPRESENTATIONS)
@pytest.mark.parametrize(("name", "make", "options", "moves"), FAMILIES, ids=IDS)
def test_distance_every_measures_one_move_in_ten_and_changes_nothing_else(
    name: str,
    make: type[Any],
    options: dict[str, Any],
    moves: int,
    arm: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = ta01().problem
    recording = Recording(monkeypatch)
    every = _solve(problem, make(arm, "active", **options))
    tenth = _solve(problem, make(arm, "active", distance_every=10, **options))
    run_every, run_tenth = recording.finished
    seen_every, seen_tenth = recording.runs()

    # Not one schedule, digest, evaluation, decision or draw differs.
    assert len(seen_every) == len(seen_tenth) == EVALUATIONS
    assert _trail(seen_every) == _trail(seen_tenth)
    assert every.evaluations == tenth.evaluations == EVALUATIONS
    assert every.iterations == tenth.iterations
    assert every.stop_reason is tenth.stop_reason
    assert _serialised(problem, every) == _serialised(problem, tenth)
    assert [(p.evaluations, p.iteration, p.value) for p in every.trace] == [
        (p.evaluations, p.iteration, p.value) for p in tenth.trace
    ]
    assert run_every.control.bit_generator.state == run_tenth.control.bit_generator.state
    assert run_every.operators.bit_generator.state == run_tenth.operators.bit_generator.state
    # Every measure that counts evaluations is the same, the digest-based ones included.
    assert _unsampled(every.extras) == _unsampled(tenth.extras)
    assert every.extras["observed_moves"] == tenth.extras["observed_moves"] == moves

    # The distance is measured on the first move and on every tenth one after it.
    observer_every, observer_tenth = _observer(run_every), _observer(run_tenth)
    assert (observer_every.distance_every, observer_tenth.distance_every) == (1, 10)
    assert len(observer_every.distances) == every.extras["measured_moves"] == moves
    assert observer_tenth.distances == observer_every.distances[::10]
    assert tenth.extras["measured_moves"] == len(observer_tenth.distances) == (moves + 9) // 10
    if moves:
        assert tenth.extras["phenotype_distance_mean"] == pytest.approx(
            mean(observer_tenth.distances)
        )
        assert tenth.extras["phenotype_distance_median"] == median(observer_tenth.distances)
    else:
        assert "phenotype_distance_mean" not in tenth.extras
        assert "phenotype_distance_median" not in tenth.extras


# ------------------------------------------------------------------ both arms


@pytest.mark.parametrize(("name", "make", "options", "moves"), FAMILIES, ids=IDS)
def test_both_arms_of_a_family_take_distance_every_alike(
    name: str, make: type[Any], options: dict[str, Any], moves: int
) -> None:
    problem = ta01().problem
    arms = [make(arm, "active", distance_every=10, **options) for arm in REPRESENTATIONS]
    assert [arm.distance_every for arm in arms] == [10, 10]
    keys, sequence = (_solve(problem, arm) for arm in arms)
    # The arms make as many moves, so they measure as many.
    assert keys.extras["observed_moves"] == sequence.extras["observed_moves"] == moves
    assert keys.extras["measured_moves"] == sequence.extras["measured_moves"] == (moves + 9) // 10
    assert keys.evaluations == sequence.evaluations == EVALUATIONS


@pytest.mark.parametrize("value", [0, -1, 1.5, 10.0, True, "10", None])
@pytest.mark.parametrize("arm", REPRESENTATIONS)
@pytest.mark.parametrize(("name", "make", "options", "moves"), FAMILIES, ids=IDS)
def test_every_family_refuses_a_bad_distance_every_when_it_is_created(
    name: str,
    make: type[Any],
    options: dict[str, Any],
    moves: int,
    arm: str,
    value: Any,
) -> None:
    with pytest.raises(ValueError, match="distance_every must be an integer of at least 1"):
        make(arm, "active", distance_every=value, **options)


@pytest.mark.parametrize("value", [1, 2, 10, 10**9])
@pytest.mark.parametrize("arm", REPRESENTATIONS)
@pytest.mark.parametrize(("name", "make", "options", "moves"), FAMILIES, ids=IDS)
def test_every_family_accepts_a_positive_integer_distance_every(
    name: str,
    make: type[Any],
    options: dict[str, Any],
    moves: int,
    arm: str,
    value: int,
) -> None:
    assert make(arm, "active", distance_every=value, **options).distance_every == value


# ------------------------------------------------------------------ the registry


@pytest.mark.parametrize(("name", "make", "options", "moves"), FAMILIES, ids=IDS)
def test_the_registry_lists_distance_every_with_its_default_and_refuses_a_bad_value(
    name: str, make: type[Any], options: dict[str, Any], moves: int
) -> None:
    registry = SolverRegistry()
    registry.register(make)
    solver_id = make.info.id
    assert registry.info(solver_id).parameters["distance_every"] == 1
    for arm in REPRESENTATIONS:
        created = registry.create(solver_id, representation=arm, distance_every=10)
        assert created == make(arm, distance_every=10)
    with pytest.raises(ConfigError, match="distance_every"):
        registry.create(solver_id, distance_every=0)
