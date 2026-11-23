"""Costs: seconds per evaluation of groups of runs, their projection, and the choice of a budget."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.campaigns import (
    Cost,
    CostModel,
    Protocol,
    Provenance,
    RunRecord,
    first_fitting,
    largest_budget,
    read_runs,
    run,
)
from dispatchatlas.campaigns.records import Outcome, ProblemRef, SolverEntry
from dispatchatlas.core import ConfigError, Status, StopCriteria, StopReason
from tests.campaigns.solvers import registry, tiny_campaign

SIZES = {"ta01": "15x15", "ta02": "15x15", "ta03": "15x15", "ta41": "30x20", "ta42": "30x20"}
"""The size class of each instance of the made-up study."""

ARMS = ("ga-keys-active", "ga-orders-active")
"""Two arms of one family that differ in their representation."""


def _key(solver: str, problem: str) -> Hashable:
    """The study's shape of a group: family, representation, builder, and size class."""
    family, representation, builder = solver.split("-")
    return family, representation, builder, SIZES[problem]


def _record(
    solver: str,
    problem: str,
    evaluations: int,
    seconds: float,
    *,
    extras: dict[str, float] | None = None,
) -> RunRecord:
    """The record of a run of ``evaluations`` evaluations that took ``seconds`` seconds."""
    outcome = Outcome(
        status=Status.FEASIBLE,
        stop_reason=StopReason.EVALUATION_LIMIT,
        objective="makespan",
        value=5.0,
        objectives={"makespan": 5.0},
        bound=None,
        gap=None,
        feasible=True,
        violations=(),
        evaluations=evaluations,
        iterations=evaluations,
        elapsed=seconds,
        extras=extras or {},
    )
    return RunRecord(
        run_id=f"run-{solver}-{problem}-{evaluations}-{seconds}",
        campaign="costs",
        config_hash="0" * 64,
        problem=ProblemRef(problem, None, "1" * 64),
        solver=SolverEntry(solver, solver, None, {}, "random-sampling", True, True, False),
        protocol=Protocol("evals", StopCriteria(max_evaluations=100_000)),
        replicate=0,
        seed=1,
        outcome=outcome,
        result_hash=outcome.result_hash(),
        schedule=None,
        provenance=Provenance(1, 1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", seconds),
    )


def _charging(seconds: float) -> Callable[[RunRecord], float]:
    """A function that charges every record ``seconds`` seconds."""
    return lambda record: seconds


# The costs are binary fractions, so the arithmetic below is exact.
RECORDS = (
    _record("ga-keys-active", "ta01", 1000, 125.0),  # 0.125 s per evaluation
    _record("ga-keys-active", "ta02", 500, 125.0),  # 0.25
    _record("ga-orders-active", "ta01", 1000, 250.0),  # 0.25
    _record("ga-keys-active", "ta41", 100, 100.0),  # 1.0
    _record("ga-orders-active", "ta41", 100, 200.0),  # 2.0
)
MODEL = CostModel.fit(RECORDS, _key)


def test_the_cost_of_a_group_is_the_mean_of_its_runs_seconds_per_evaluation() -> None:
    assert MODEL.groups == {
        ("ga", "keys", "active", "15x15"): Cost(0.1875, 2),  # (0.125 + 0.25) / 2
        ("ga", "orders", "active", "15x15"): Cost(0.25, 1),
        ("ga", "keys", "active", "30x20"): Cost(1.0, 1),
        ("ga", "orders", "active", "30x20"): Cost(2.0, 1),
    }
    # The groups come in the order the records first named them.
    assert next(iter(MODEL.groups)) == ("ga", "keys", "active", "15x15")
    # It is the mean of the ratios of the runs, not the ratio of their sums.
    ratio = (125.0 + 125.0) / (1000 + 500)
    assert MODEL.cost("ga-keys-active", "ta01").seconds_per_evaluation != pytest.approx(ratio)


def test_a_projection_is_the_evaluations_at_the_cost_of_the_group() -> None:
    assert MODEL.seconds("ga-keys-active", "ta01", 100_000) == 100_000 * 0.1875
    # Another instance of the same size class belongs to the same group.
    assert MODEL.seconds("ga-keys-active", "ta03", 100_000) == 18_750.0
    assert MODEL.seconds("ga-orders-active", "ta42", 30_000) == 60_000.0
    assert MODEL.seconds("ga-orders-active", "ta42", 0) == 0.0
    assert MODEL.cost("ga-orders-active", "ta42") == Cost(2.0, 1)


def test_a_group_that_was_never_measured_is_refused_not_extrapolated() -> None:
    with pytest.raises(ConfigError, match=r"no run of group \('ga', 'keys', 'passive', '15x15'\)"):
        MODEL.seconds("ga-keys-passive", "ta01", 10)
    with pytest.raises(ConfigError, match="solver 'ga-keys-passive' on problem 'ta01'"):
        MODEL.cost("ga-keys-passive", "ta01")


@pytest.mark.parametrize("evaluations", [-1, 1.5, 1e5, True, "10"])
def test_a_projection_needs_a_whole_number_of_evaluations(evaluations: Any) -> None:
    with pytest.raises(ConfigError, match="evaluations must be an integer of at least 0"):
        MODEL.seconds("ga-keys-active", "ta01", evaluations)


def test_a_function_can_charge_a_run_less_than_its_wall_seconds() -> None:
    records = [
        _record("ga-keys-active", "ta01", 1000, 125.0, extras={"observer_seconds": 25.0}),
        _record("ga-keys-active", "ta02", 500, 125.0, extras={"observer_seconds": 75.0}),
    ]
    model = CostModel.fit(
        records,
        _key,
        seconds=lambda record: (
            record.provenance.wall_seconds - float(record.outcome.extras["observer_seconds"])
        ),
    )
    # (100 / 1000 + 50 / 500) / 2
    (cost,) = model.groups.values()
    assert list(model.groups) == [("ga", "keys", "active", "15x15")]
    assert cost.seconds_per_evaluation == pytest.approx(0.1)
    assert cost.runs == 2


def test_a_model_is_fitted_on_records_and_refuses_the_ones_it_cannot_use() -> None:
    with pytest.raises(ConfigError, match="at least one run record"):
        CostModel.fit([], _key)
    with pytest.raises(ConfigError, match="evaluated no schedule"):
        CostModel.fit([_record("ga-keys-active", "ta01", 0, 1.0)], _key)
    for charged in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ConfigError, match="not a finite number of at least zero"):
            CostModel.fit([RECORDS[0]], _key, seconds=_charging(charged))


def test_a_model_fitted_on_the_records_of_a_campaign_averages_their_own_numbers(
    tmp_path: Path,
) -> None:
    campaign = tiny_campaign()
    records = read_runs(run(campaign, tmp_path, registry=registry()).directory)
    model = CostModel.fit(records, lambda solver, problem: (solver, problem))
    assert len(model.groups) == 6  # three solvers on two problems
    for group, cost in model.groups.items():
        assert isinstance(group, tuple)
        solver, problem = group
        ratios = [
            record.provenance.wall_seconds / record.outcome.evaluations
            for record in records
            if (record.solver.name, record.problem.id) == (solver, problem)
        ]
        assert cost.runs == len(ratios)
        assert cost.seconds_per_evaluation == pytest.approx(math.fsum(ratios) / len(ratios))
    assert sum(cost.runs for cost in model.groups.values()) == len(records)


# ------------------------------------------------------ choosing a configuration


INSTANCES = ("ta01", "ta02", "ta03", "ta41", "ta42")
CANDIDATES = (
    (100_000, 50),
    (100_000, 30),
    (30_000, 50),
    (30_000, 30),
    (10_000, 50),
    (10_000, 30),
)
"""Evaluations and runs per instance, in the order of preference of the made-up study."""


def _planned(candidate: tuple[int, int]) -> float:
    """The seconds of every arm on every instance, ``runs`` times, from the model."""
    evaluations, runs = candidate
    return runs * sum(
        MODEL.seconds(arm, problem, evaluations) for arm in ARMS for problem in INSTANCES
    )


def test_the_cost_of_a_candidate_is_hand_arithmetic_from_the_model() -> None:
    # Per evaluation, over the arms and instances: 3 * (0.1875 + 0.25) + 2 * (1 + 2).
    per_evaluation = 3 * (0.1875 + 0.25) + 2 * (1.0 + 2.0)
    assert per_evaluation == 7.3125
    assert [_planned(candidate) for candidate in CANDIDATES] == [
        50 * 100_000 * 7.3125,
        30 * 100_000 * 7.3125,
        50 * 30_000 * 7.3125,
        30 * 30_000 * 7.3125,
        50 * 10_000 * 7.3125,
        30 * 10_000 * 7.3125,
    ]


def test_the_first_candidate_that_fits_wins_whatever_the_ones_after_it_cost() -> None:
    # With 1.5 for reruns the candidates cost 54,843,750, 32,906,250, 16,453,125, 9,871,875,
    # 5,484,375, and 3,290,625.
    def chosen(limit: float, factor: float = 1.5) -> tuple[int, int] | None:
        return first_fitting(CANDIDATES, _planned, limit, factor)

    assert chosen(100_000_000) == (100_000, 50)
    assert chosen(54_843_750) == (100_000, 50)  # at the limit, which is allowed
    assert chosen(54_843_749) == (100_000, 30)
    assert chosen(20_000_000) == (30_000, 50)
    assert chosen(16_453_125) == (30_000, 50)
    assert chosen(16_453_124) == (30_000, 30)
    assert chosen(5_484_375) == (10_000, 50)
    assert chosen(3_290_625) == (10_000, 30)
    assert chosen(3_290_624) is None
    # Reruns count: the first candidate fits without them and not with them.
    assert chosen(36_562_500, 1.0) == (100_000, 50)
    assert chosen(36_562_500) == (100_000, 30)


def test_a_candidate_is_projected_only_when_the_ones_before_it_do_not_fit() -> None:
    calls: list[str] = []
    costs = {"a": 9.0, "b": 4.0, "c": 1.0}

    def project(name: str) -> float:
        calls.append(name)
        return costs[name]

    assert first_fitting(iter(["a", "b", "c"]), project, 5.0) == "b"
    assert calls == ["a", "b"]
    assert first_fitting([], project, 5.0) is None


@dataclass(frozen=True)
class Scope:
    """A candidate that is no pair: a budget and the families that stay in the study."""

    evaluations: int
    families: tuple[str, ...]


def test_a_candidate_may_be_any_value_such_as_a_declared_fallback() -> None:
    candidates = (
        Scope(10_000, ("ga", "pso")),
        Scope(10_000, ("ga",)),
        Scope(10_000, ()),
    )
    found = first_fitting(candidates, lambda c: c.evaluations * len(c.families), 10_000.0)
    assert found == Scope(10_000, ("ga",))


@pytest.mark.parametrize("limit", [0.0, -1.0, float("nan"), float("inf")])
def test_a_limit_is_a_finite_number_above_zero(limit: float) -> None:
    with pytest.raises(ConfigError, match="limit must be a finite number above 0"):
        first_fitting(CANDIDATES, _planned, limit)


@pytest.mark.parametrize("factor", [0.0, -1.5, float("nan"), float("inf")])
def test_a_factor_is_a_finite_number_above_zero(factor: float) -> None:
    with pytest.raises(ConfigError, match="factor must be a finite number above 0"):
        first_fitting(CANDIDATES, _planned, 1.0, factor)


@pytest.mark.parametrize("cost", [-1.0, float("nan"), float("inf")])
def test_a_projection_that_is_no_cost_is_refused(cost: float) -> None:
    with pytest.raises(ConfigError, match="projected cost of 'a' is"):
        first_fitting(["a"], lambda _: cost, 1.0)


def test_the_largest_budget_is_the_first_fit_from_the_largest_level_down() -> None:
    levels = (30_000, 10_000, 100_000)

    def planned(evaluations: int) -> float:
        return _planned((evaluations, 50))

    assert largest_budget(levels, planned, 20_000_000, 1.5) == 30_000
    assert largest_budget(levels, planned, 54_843_750, 1.5) == 100_000
    assert largest_budget(levels, planned, 5_484_375, 1.5) == 10_000
    assert largest_budget(levels, planned, 5_484_374, 1.5) is None
    # A cost that does not rise with the level still gives the largest level that fits.
    assert largest_budget([1, 2, 3], {1: 1.0, 2: 9.0, 3: 2.0}.__getitem__, 2.0) == 3
    assert largest_budget([0.5, 2.5, 1.5], lambda level: level, 2.0) == 1.5
