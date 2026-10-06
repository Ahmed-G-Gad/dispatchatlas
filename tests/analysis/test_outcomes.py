"""Outcomes against the plan: stop reasons per protocol, overruns, and the plan's notes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from dispatchatlas.analysis import AnalysisConfig, summarize
from dispatchatlas.analysis.dataset import Dataset
from dispatchatlas.campaigns import Protocol
from dispatchatlas.campaigns.execution import (
    CLOCK_MARGIN_SECONDS,
    OVERRUN_MARGIN,
    observe,
    overran,
)
from dispatchatlas.campaigns.fairness import (
    Overrun,
    assess,
    assess_outcomes,
    conditional_notes,
    notes,
)
from dispatchatlas.campaigns.records import RunRecord
from dispatchatlas.core import ConfigError, StopCriteria, StopReason
from tests.analysis.analysis_data import make_dataset

CELLS = {
    "a": {"p1": [5.0, 5.0], "p2": [6.0, 6.0], "p3": [7.0, 7.0]},
    "b": {"p1": [4.0, 4.0], "p2": [7.0, 7.0], "p3": [6.0, 6.0]},
    "fixed": {"p1": [5.0], "p2": [6.0], "p3": [8.0]},
}
TIMED = StopCriteria(time_limit=1.0, patience=5)


def _dataset(
    reasons: dict[str, list[StopReason]] | None = None,
    *,
    walls: dict[str, list[float]] | None = None,
    stop: StopCriteria = TIMED,
    protocols: tuple[str, ...] = ("p",),
) -> Dataset:
    """Return a dataset whose solvers' runs end for the given reasons, in plan order."""
    dataset = make_dataset(CELLS, deterministic=("fixed",), protocols=protocols)
    stops = tuple(Protocol(name, stop) for name in protocols)
    seen: dict[tuple[str, str], int] = {}
    runs = []
    for run in dataset.runs:
        assert run.record is not None
        index = seen.get((run.protocol, run.solver), 0)
        seen[run.protocol, run.solver] = index + 1
        record = replace(run.record, protocol=Protocol(run.protocol, stop))
        if reasons and run.solver in reasons:
            reason = reasons[run.solver][index % len(reasons[run.solver])]
            record = replace(record, outcome=replace(record.outcome, stop_reason=reason))
        if walls and run.solver in walls:
            wall = walls[run.solver][index % len(walls[run.solver])]
            record = replace(record, provenance=replace(record.provenance, wall_seconds=wall))
        runs.append(replace(run, record=record))
    plan = replace(dataset.plan, protocols=stops)
    return replace(dataset, plan=plan, runs=tuple(runs), protocols=stops)


def _records(dataset: Dataset) -> list[RunRecord]:
    return [run.record for run in dataset.runs if run.record is not None]


def _limitations(dataset: Dataset, **settings: Any) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for item in summarize(dataset, AnalysisConfig(**settings)).limitations:
        found.setdefault(item.topic, []).append(item.message)
    return found


# ---------------------------------------------------------------- stop reasons


def test_stop_reasons_that_agree_do_not_diverge() -> None:
    dataset = _dataset()
    outcomes = assess_outcomes(dataset.plan, _records(dataset))
    assert outcomes.records == 15
    assert outcomes.overrun_tolerance == OVERRUN_MARGIN == AnalysisConfig().overrun_tolerance
    protocol = outcomes.protocol("p")
    assert protocol.stop_reasons == {
        "a": {"evaluation-limit": 6},
        "b": {"evaluation-limit": 6},
        "fixed": {"evaluation-limit": 3},
    }
    assert not protocol.divergent
    assert outcomes.divergent == ()
    assert "stop-reasons" not in _limitations(dataset)


def test_divergent_stop_reasons_are_reported() -> None:
    dataset = _dataset(
        {"a": [StopReason.STAGNATION], "b": [StopReason.TIME_LIMIT, StopReason.STAGNATION]}
    )
    outcomes = assess_outcomes(dataset.plan, _records(dataset))
    protocol = outcomes.protocol("p")
    assert protocol.stop_reasons == {
        "a": {"stagnation": 6},
        "b": {"stagnation": 3, "time-limit": 3},
        "fixed": {"evaluation-limit": 3},
    }
    assert protocol.divergent
    assert outcomes.divergent == ("p",)
    assert outcomes.to_dict()["protocols"][0]["divergent"] is True
    (message,) = _limitations(dataset)["stop-reasons"]
    assert message.startswith(
        "Under protocol p, the completed runs of the iterative solvers ended for different "
        "reasons: a 6 stagnation; b 3 stagnation, 3 time-limit; fixed 3 evaluation-limit."
    )
    assert "did not spend the same budget" in message


def test_one_solver_whose_runs_end_differently_diverges() -> None:
    dataset = _dataset(
        {
            "a": [StopReason.TIME_LIMIT, StopReason.TARGET_REACHED],
            "b": [StopReason.TIME_LIMIT],
            "fixed": [StopReason.TIME_LIMIT],
        }
    )
    protocol = assess_outcomes(dataset.plan, _records(dataset)).protocol("p")
    assert protocol.stop_reasons["a"] == {"target-reached": 3, "time-limit": 3}
    assert protocol.divergent


def test_solvers_that_are_not_iterative_are_left_out_of_the_stop_reasons() -> None:
    dataset = _dataset({"fixed": [StopReason.COMPLETED]})
    constructive = tuple(
        replace(entry, iterative=entry.name != "fixed") for entry in dataset.plan.solvers
    )
    plan = replace(dataset.plan, solvers=constructive)
    protocol = assess_outcomes(plan, _records(dataset)).protocol("p")
    assert sorted(protocol.stop_reasons) == ["a", "b"]
    assert not protocol.divergent
    narrowed = replace(dataset, plan=plan, solvers=constructive)
    assert "stop-reasons" not in _limitations(narrowed)


def test_stop_reasons_are_compared_per_protocol() -> None:
    dataset = _dataset(protocols=("first", "second"))
    runs = tuple(
        replace(
            run,
            record=replace(
                run.record, outcome=replace(run.record.outcome, stop_reason=StopReason.STAGNATION)
            ),
        )
        if run.protocol == "second" and run.solver == "a" and run.record is not None
        else run
        for run in dataset.runs
    )
    mixed = replace(dataset, runs=runs)
    outcomes = assess_outcomes(mixed.plan, _records(mixed))
    assert [item.protocol for item in outcomes.protocols] == ["first", "second"]
    # The two protocols together hold two reasons, yet only one of them diverges.
    assert not outcomes.protocol("first").divergent
    assert outcomes.protocol("second").divergent
    assert outcomes.divergent == ("second",)
    assert "stop-reasons" not in _limitations(mixed, protocol="first")
    assert "stop-reasons" in _limitations(mixed, protocol="second")
    with pytest.raises(KeyError):
        outcomes.protocol("third")


def test_records_of_runs_outside_the_plan_do_not_count() -> None:
    dataset = _dataset({"a": [StopReason.STAGNATION]})
    kept = tuple(run for run in dataset.plan.runs if run.solver != "a")
    plan = replace(dataset.plan, runs=kept)
    outcomes = assess_outcomes(plan, _records(dataset))
    assert outcomes.records == 9
    assert "a" not in outcomes.protocol("p").stop_reasons
    assert not outcomes.protocol("p").divergent


# -------------------------------------------------------------------- overruns


def test_overruns_are_counted_against_the_time_limit() -> None:
    dataset = _dataset(walls={"a": [1.0, 1.04, 1.5], "b": [0.9], "fixed": [2.0]})
    outcomes = assess_outcomes(dataset.plan, _records(dataset))
    assert outcomes.protocol("p").overruns == {
        "a": Overrun(runs=6, over=2, worst=1.5),
        "fixed": Overrun(runs=3, over=3, worst=2.0),
    }
    loose = assess_outcomes(dataset.plan, _records(dataset), overrun_tolerance=1.0)
    assert loose.protocol("p").overruns == {}
    (budget,) = _limitations(dataset)["budget"]
    assert "a in 2 of 6 (up to 1.5 times the limit)" in budget
    assert "fixed in 3 of 3 (up to 2 times the limit)" in budget
    assert "b in" not in budget


def test_the_overrun_tolerance_is_a_share_of_the_limit_above_the_clock_margin() -> None:
    long = StopCriteria(time_limit=10.0, patience=5)
    dataset = _dataset(walls={"a": [10.0, 10.2, 15.0]}, stop=long)
    default = assess_outcomes(dataset.plan, _records(dataset))
    assert default.protocol("p").overruns == {"a": Overrun(runs=6, over=2, worst=1.5)}
    strict = assess_outcomes(dataset.plan, _records(dataset), overrun_tolerance=0.0)
    assert strict.protocol("p").overruns == {"a": Overrun(runs=6, over=4, worst=1.5)}
    # With no share allowed, a run within the clock margin of its limit is still not over.
    within = _dataset(walls={"a": [10.0 + CLOCK_MARGIN_SECONDS]}, stop=long)
    exact = assess_outcomes(within.plan, _records(within), overrun_tolerance=0.0)
    assert exact.protocol("p").overruns == {}


def test_the_assessment_counts_an_overrun_by_the_rule_of_the_engine() -> None:
    short = StopCriteria(time_limit=0.15, patience=5)
    dataset = _dataset(walls={"a": [0.19, 0.21]}, stop=short)
    records = _records(dataset)
    noted = [
        record
        for record in records
        for note in observe(record.outcome, short, record.provenance.wall_seconds, iterative=True)
        if note.startswith("the run took")
    ]
    assert [record.provenance.wall_seconds for record in noted] == [0.21] * 3
    outcomes = assess_outcomes(dataset.plan, records)
    assert outcomes.protocol("p").overruns == {"a": Overrun(runs=6, over=3, worst=0.21 / 0.15)}
    margin = max(CLOCK_MARGIN_SECONDS, OVERRUN_MARGIN * 0.15)
    assert overran(0.15 + margin + 0.001, 0.15)
    assert not overran(0.15 + margin, 0.15)
    # Above a limit of one second the share is the larger margin.
    assert overran(10.6, 10.0)
    assert not overran(10.4, 10.0)
    assert overran(10.4, 10.0, 0.0)


def test_a_protocol_without_a_time_limit_has_no_overrun() -> None:
    dataset = _dataset(walls={"a": [50.0]}, stop=StopCriteria(max_evaluations=10))
    outcomes = assess_outcomes(dataset.plan, _records(dataset))
    assert outcomes.protocol("p").overruns == {}
    assert "budget" not in _limitations(dataset)


@pytest.mark.parametrize("value", [-0.1, True, "0.05", float("nan")])
def test_the_overrun_tolerance_must_be_zero_or_positive(value: Any) -> None:
    dataset = _dataset()
    with pytest.raises(ConfigError, match="overrun_tolerance"):
        assess_outcomes(dataset.plan, _records(dataset), overrun_tolerance=value)


# ------------------------------------------------------------ the plan's notes


def test_the_analysis_reports_the_notes_of_the_plan_as_limitations() -> None:
    dataset = _dataset()
    protocol = dataset.plan.protocol("p")
    (note,) = notes(dataset.plan, protocol)
    assert assess(dataset.plan).notes == (note,)
    assert "one run per problem to the iterative solvers declared deterministic (fixed)" in note
    (message,) = _limitations(dataset)["plan"]
    assert message == "Protocol" + note.removeprefix("protocol") + "."
    # A selection without the solver has nothing to report.
    assert notes(dataset.plan, protocol, ["a", "b"]) == ()
    assert "plan" not in _limitations(dataset, solvers=("a", "b"))


def _not_iterative(dataset: Dataset) -> Dataset:
    """Declare the deterministic solver not iterative, as a constructive rule is."""
    solvers = tuple(
        replace(entry, iterative=entry.name != "fixed") for entry in dataset.plan.solvers
    )
    return replace(dataset, plan=replace(dataset.plan, solvers=solvers), solvers=solvers)


def test_a_single_run_that_is_not_iterative_is_reported_only_when_the_time_limit_ends_it() -> None:
    finished = _not_iterative(_dataset({"fixed": [StopReason.COMPLETED]}))
    protocol = finished.plan.protocol("p")
    assert notes(finished.plan, protocol) == ()
    (conditional,) = conditional_notes(finished.plan, protocol)
    assert "declared deterministic that are not iterative (fixed)" in conditional
    outcomes = assess_outcomes(finished.plan, _records(finished))
    assert outcomes.protocol("p").unreplicated == {}
    assert outcomes.to_dict()["protocols"][0]["unreplicated"] == {}
    assert "plan" not in _limitations(finished)

    cut = _not_iterative(_dataset({"fixed": [StopReason.TIME_LIMIT, StopReason.COMPLETED]}))
    outcomes = assess_outcomes(cut.plan, _records(cut))
    assert outcomes.protocol("p").unreplicated == {"fixed": {"completed": 1, "time-limit": 2}}
    (message,) = _limitations(cut)["plan"]
    assert message.startswith(
        "Under protocol p, the time limit ended completed runs of solvers declared "
        "deterministic that run once per problem: fixed 1 completed, 2 time-limit."
    )
    assert "replicate_timed = true" in message
    assert "plan" not in _limitations(cut, solvers=("a", "b"))

    # With every replicate, or without a time limit, there is nothing to report.
    replicated = replace(cut, plan=replace(cut.plan, replicate_timed=True))
    assert assess_outcomes(replicated.plan, _records(replicated)).protocol("p").unreplicated == {}
    assert "plan" not in _limitations(replicated)
    counted = _not_iterative(
        _dataset({"fixed": [StopReason.TIME_LIMIT]}, stop=StopCriteria(max_evaluations=10))
    )
    assert assess_outcomes(counted.plan, _records(counted)).protocol("p").unreplicated == {}


def test_a_plan_that_replicates_under_the_time_limit_has_no_such_limitation() -> None:
    dataset = _dataset()
    replicated = replace(dataset, plan=replace(dataset.plan, replicate_timed=True))
    assert notes(replicated.plan, replicated.plan.protocol("p")) == ()
    assert "plan" not in _limitations(replicated)
    counted = _dataset(stop=StopCriteria(max_evaluations=10))
    assert notes(counted.plan, counted.plan.protocol("p")) == ()
    assert "plan" not in _limitations(counted)
