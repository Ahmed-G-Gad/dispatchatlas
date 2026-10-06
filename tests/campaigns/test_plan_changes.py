"""A plan that changes under one config hash: the warning, the session record, the digest."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import pytest

from dispatchatlas.campaigns import (
    FileSource,
    SessionRecord,
    SolverSpec,
    plan,
    read_plan,
    read_runs,
    read_sessions,
    run,
)
from dispatchatlas.campaigns.records import PlanChange, SessionStatus
from dispatchatlas.campaigns.store import Store
from dispatchatlas.core import Problem, SolveResult, SolverInfo, StopCriteria, save_problem
from dispatchatlas.core.errors import SerializationError
from dispatchatlas.solvers import SolverRegistry, default_registry
from dispatchatlas.solvers.rules import InputOrder
from tests.campaigns.solvers import registry, tiny_campaign
from tests.helpers import chain_problem

REGISTRY = registry()


def _info(*, stochastic: bool, iterative: bool = False) -> SolverInfo:
    return SolverInfo(
        id="test-switch",
        name="Switch",
        family="random-sampling",
        summary="Builds the input order under metadata the tests change.",
        stochastic=stochastic,
        iterative=iterative,
    )


@dataclass(frozen=True)
class SwitchOnce:
    """A solver declared deterministic."""

    info: ClassVar[SolverInfo] = _info(stochastic=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return InputOrder().solve(problem, stop, seed)


@dataclass(frozen=True)
class SwitchMany:
    """The same solver id, declared stochastic."""

    info: ClassVar[SolverInfo] = _info(stochastic=True)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return InputOrder().solve(problem, stop, seed)


@dataclass(frozen=True)
class SwitchIterative:
    """The same solver id, declared deterministic and iterative."""

    info: ClassVar[SolverInfo] = _info(stochastic=False, iterative=True)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return InputOrder().solve(problem, stop, seed)


def _registry(factory: type[SwitchOnce | SwitchMany | SwitchIterative]) -> SolverRegistry:
    solvers = default_registry(plugins=False)
    solvers.register(factory)
    return solvers


SWITCHING = (SolverSpec("test-switch"), SolverSpec("random-search"))


def _plan_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [message for message in caplog.messages if "the plan of campaign" in message]


def test_a_changed_plan_under_one_hash_is_named(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = tiny_campaign(solvers=SWITCHING, replicates=2)
    with caplog.at_level(logging.WARNING):
        first = run(campaign, tmp_path, registry=_registry(SwitchOnce))
    # Two problems: one run of the switch solver and two of random search on each.
    assert first.planned == first.completed == 6
    assert _plan_warnings(caplog) == []
    before = read_plan(first.directory)
    (opening,) = read_sessions(first.directory)
    assert opening.plan_digest == before.runs_digest
    assert opening.plan_left == ()
    assert opening.plan_entered == ()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        second = run(campaign, tmp_path, registry=_registry(SwitchMany))
    after = read_plan(second.directory)
    assert after.config_hash == before.config_hash == campaign.config_hash
    assert (second.planned, second.attempted, second.completed) == (8, 2, 8)
    entered = [item for item in after.runs if not before.has_run(item.run_id)]
    assert [(item.solver, item.replicate) for item in entered] == [("test-switch", 1)] * 2
    latest = read_sessions(second.directory)[-1]
    assert latest.plan_digest == after.runs_digest
    assert latest.plan_digest != opening.plan_digest
    assert latest.plan_left == ()
    assert [dict(item) for item in latest.plan_entered] == [item.to_dict() for item in entered]
    (warning,) = _plan_warnings(caplog)
    assert f"config hash {campaign.config_hash[:12]}" in warning
    assert "0 run(s) left it and 2 entered it" in warning
    for item in entered:
        assert f"{item.run_id} (test-switch on {item.problem} under evals, replicate 1)" in warning

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        third = run(campaign, tmp_path, registry=_registry(SwitchOnce))
    assert (third.planned, third.attempted, third.completed) == (6, 0, 6)
    closing = read_sessions(third.directory)[-1]
    assert closing.plan_digest == opening.plan_digest
    assert [dict(item) for item in closing.plan_left] == [item.to_dict() for item in entered]
    assert closing.plan_entered == ()
    (warning,) = _plan_warnings(caplog)
    assert "2 run(s) left it and 0 entered it" in warning
    assert entered[0].run_id in warning
    # The records of the runs that left stay in the directory and are no longer read.
    assert len(list((third.directory / "runs").rglob("*.json"))) == 8
    assert len(read_runs(third.directory)) == 6


def test_an_unchanged_plan_is_not_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = tiny_campaign()
    with caplog.at_level(logging.WARNING):
        first = run(campaign, tmp_path, registry=REGISTRY, max_runs=3)
        run(campaign.with_execution(retries=2), tmp_path, registry=REGISTRY)
    assert _plan_warnings(caplog) == []
    opening, resumed = read_sessions(first.directory)
    assert opening.plan_digest == resumed.plan_digest == read_plan(first.directory).runs_digest
    assert resumed.plan_left == resumed.plan_entered == ()


def test_a_changed_problem_names_the_runs_it_replaces(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    save_problem(chain_problem(), tmp_path / "chain.json")
    campaign = replace(tiny_campaign(instances=(FileSource(Path("chain.json")),)), base=tmp_path)
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    before = read_plan(first.directory)
    save_problem(
        replace(chain_problem(), objectives=("total_completion_time",)), tmp_path / "chain.json"
    )
    with caplog.at_level(logging.WARNING):
        second = run(campaign, tmp_path / "out", registry=REGISTRY)
    after = read_plan(second.directory)
    latest = read_sessions(second.directory)[-1]
    assert [item["run_id"] for item in latest.plan_left] == [item.run_id for item in before.runs]
    assert [item["run_id"] for item in latest.plan_entered] == [item.run_id for item in after.runs]
    (warning,) = _plan_warnings(caplog)
    assert f"{len(before.runs)} run(s) left it and {len(after.runs)} entered it" in warning
    assert "more" in warning


def test_the_store_returns_the_change_it_warns_about(tmp_path: Path) -> None:
    campaign = tiny_campaign(solvers=SWITCHING, replicates=3)
    once = plan(campaign, registry=_registry(SwitchOnce))
    many = plan(campaign, registry=_registry(SwitchMany))
    store = Store(tmp_path / "tiny")
    store.directory.mkdir()
    assert store.plan_change(once) == PlanChange()
    store.write_plan(once)
    assert not store.plan_change(once)
    change = store.plan_change(many)
    assert change
    assert change.left == ()
    # In plan order: by problem, then by replicate.
    assert [item["replicate"] for item in change.entered] == [1, 2, 1, 2]
    # The comparison writes nothing: the directory still holds the first plan.
    assert read_plan(store.directory) == once
    assert store.plan_change(many) == change
    store.write_plan(many)
    back = store.plan_change(once)
    assert [item["run_id"] for item in back.left] == [item["run_id"] for item in change.entered]
    assert back.entered == ()
    # A plan of another configuration is not compared with the one the directory holds.
    other = plan(replace(campaign, seed=campaign.seed + 1), registry=_registry(SwitchOnce))
    assert store.plan_change(other) == PlanChange()


def test_a_plan_change_is_stored_after_a_session_that_ended_before_its_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tiny_campaign(solvers=SWITCHING, replicates=2)
    first = run(campaign, tmp_path, registry=_registry(SwitchOnce))
    before = read_plan(first.directory)

    def interrupted(self: Store, plan: object) -> None:
        raise KeyboardInterrupt

    # The session sees the changed plan and ends before it writes its record.
    with monkeypatch.context() as patch:
        patch.setattr(Store, "ledger", interrupted)
        with pytest.raises(KeyboardInterrupt):
            run(campaign, tmp_path, registry=_registry(SwitchMany))
    assert len(read_sessions(first.directory)) == 1
    assert read_plan(first.directory) == before
    assert len(_plan_warnings(caplog)) == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        resumed = run(campaign, tmp_path, registry=_registry(SwitchMany))
    after = read_plan(resumed.directory)
    entered = [item for item in after.runs if not before.has_run(item.run_id)]
    assert len(entered) == 2
    latest = read_sessions(resumed.directory)[-1]
    assert [dict(item) for item in latest.plan_entered] == [item.to_dict() for item in entered]
    assert latest.plan_digest == after.runs_digest != before.runs_digest
    (warning,) = _plan_warnings(caplog)
    assert "0 run(s) left it and 2 entered it" in warning


def test_a_changed_seed_under_one_run_id_is_named(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    result = plan(tiny_campaign(), registry=REGISTRY)
    first = result.runs[0]
    reseeded = replace(result, runs=(replace(first, seed=first.seed + 1), *result.runs[1:]))
    assert reseeded.runs_digest != result.runs_digest
    store = Store(tmp_path / "tiny")
    store.directory.mkdir()
    store.write_plan(result)
    with caplog.at_level(logging.WARNING):
        change = store.plan_change(reseeded)
    assert change
    assert change.left == change.entered == change.solvers == ()
    assert [dict(item) for item in change.reseeded] == [
        {**first.to_dict(), "seed": first.seed + 1, "earlier_seed": first.seed}
    ]
    (warning,) = _plan_warnings(caplog)
    assert "the seed of 1 run(s) changed" in warning
    assert f"{first.run_id} ({first.solver} on {first.problem}" in warning
    assert "left it" not in warning


def test_a_changed_solver_entry_under_one_hash_is_named(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    campaign = tiny_campaign(solvers=SWITCHING, replicates=2)
    first = run(campaign, tmp_path, registry=_registry(SwitchOnce))
    before = read_plan(first.directory)
    assert before.solver("test-switch").iterative is False

    with caplog.at_level(logging.WARNING):
        second = run(campaign, tmp_path, registry=_registry(SwitchIterative))
    after = read_plan(second.directory)
    assert after.solver("test-switch").iterative is True
    # The runs are the same, so the digest of the runs is, and the change is still named.
    assert after.runs_digest == before.runs_digest
    assert second.attempted == 0
    latest = read_sessions(second.directory)[-1]
    assert latest.plan_left == latest.plan_entered == latest.plan_reseeded == ()
    assert [dict(item) for item in latest.plan_solvers] == [
        {"solver": "test-switch", "field": "iterative", "earlier": False, "now": True}
    ]
    (warning,) = _plan_warnings(caplog)
    assert "test-switch: iterative False -> True" in warning
    assert "left it" not in warning

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        run(campaign, tmp_path, registry=_registry(SwitchIterative))
    assert _plan_warnings(caplog) == []
    assert read_sessions(second.directory)[-1].plan_solvers == ()


def test_an_entry_that_states_no_threads_is_not_a_change(tmp_path: Path) -> None:
    result = plan(tiny_campaign(), registry=REGISTRY)
    silent = replace(
        result, solvers=tuple(replace(entry, threads=None) for entry in result.solvers)
    )
    stated = replace(result, solvers=tuple(replace(entry, threads=2) for entry in result.solvers))
    store = Store(tmp_path / "tiny")
    store.directory.mkdir()
    store.write_plan(silent)
    assert store.plan_change(stated) == PlanChange()
    store.write_plan(stated)
    change = store.plan_change(silent)
    assert [(item["field"], item["earlier"], item["now"]) for item in change.solvers] == [
        ("threads", 2, None)
    ] * len(result.solvers)


def test_the_runs_digest_identifies_the_runs_and_their_seeds() -> None:
    result = plan(tiny_campaign(), registry=REGISTRY)
    assert len(result.runs_digest) == 64
    assert result.to_dict()["runs_digest"] == result.runs_digest
    reordered = replace(result, runs=tuple(reversed(result.runs)))
    assert reordered.runs_digest == result.runs_digest
    shorter = replace(result, runs=result.runs[:-1])
    assert shorter.runs_digest != result.runs_digest
    last = result.runs[-1]
    reseeded = replace(result, runs=(*result.runs[:-1], replace(last, seed=last.seed + 1)))
    assert reseeded.runs_digest != result.runs_digest
    assert plan(tiny_campaign(), registry=REGISTRY).runs_digest == result.runs_digest


def test_a_session_record_without_the_plan_fields_still_loads(tmp_path: Path) -> None:
    summary = run(tiny_campaign(), tmp_path, registry=REGISTRY)
    (session,) = read_sessions(summary.directory)
    document = session.to_dict()
    assert document["plan_digest"] == read_plan(summary.directory).runs_digest
    assert document["plan_left"] == document["plan_entered"] == []
    assert document["plan_reseeded"] == document["plan_solvers"] == []
    assert SessionRecord.from_dict(document) == session
    for key in ("plan_digest", "plan_left", "plan_entered", "plan_reseeded", "plan_solvers"):
        del document[key]
    stored = SessionRecord.from_dict(document)
    assert stored.plan_digest is None
    assert stored.plan_left == stored.plan_entered == ()
    assert stored.plan_reseeded == stored.plan_solvers == ()
    assert stored == replace(session, plan_digest=None)


def test_a_session_record_reads_back_every_change_of_plan() -> None:
    execution = tiny_campaign().execution
    record = SessionRecord(
        session=2,
        campaign="tiny",
        config_hash="0" * 64,
        started="2000-01-01T00:00:00+00:00",
        finished=None,
        status=SessionStatus.RUNNING,
        execution=execution,
        workers=1,
        environment={},
        plan_digest="1" * 64,
        plan_reseeded=({"run_id": "run-0", "seed": 2, "earlier_seed": 1},),
        plan_solvers=({"solver": "a", "field": "iterative", "earlier": False, "now": True},),
    )
    assert SessionRecord.from_dict(record.to_dict()) == record
    document = record.to_dict()
    document["plan_solvers"] = [{"field": "iterative"}]
    with pytest.raises(SerializationError, match="plan_solvers"):
        SessionRecord.from_dict(document)
