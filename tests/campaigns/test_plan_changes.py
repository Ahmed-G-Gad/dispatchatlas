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
from dispatchatlas.campaigns.records import PlanChange
from dispatchatlas.campaigns.store import Store
from dispatchatlas.core import Problem, SolveResult, SolverInfo, StopCriteria, save_problem
from dispatchatlas.solvers import SolverRegistry, default_registry
from dispatchatlas.solvers.rules import InputOrder
from tests.campaigns.solvers import registry, tiny_campaign
from tests.helpers import chain_problem

REGISTRY = registry()


def _info(*, stochastic: bool) -> SolverInfo:
    return SolverInfo(
        id="test-switch",
        name="Switch",
        family="random-sampling",
        summary="Builds the input order under metadata the tests change.",
        stochastic=stochastic,
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


def _registry(factory: type[SwitchOnce | SwitchMany]) -> SolverRegistry:
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
    assert store.write_plan(once) == PlanChange()
    assert not store.write_plan(once)
    change = store.write_plan(many)
    assert change
    assert change.left == ()
    # In plan order: by problem, then by replicate.
    assert [item["replicate"] for item in change.entered] == [1, 2, 1, 2]
    back = store.write_plan(once)
    assert [item["run_id"] for item in back.left] == [item["run_id"] for item in change.entered]
    assert back.entered == ()
    # A plan of another configuration replaces the earlier one without a comparison.
    other = plan(replace(campaign, seed=campaign.seed + 1), registry=_registry(SwitchOnce))
    assert store.write_plan(other) == PlanChange()


def test_the_runs_digest_identifies_the_set_of_runs() -> None:
    result = plan(tiny_campaign(), registry=REGISTRY)
    assert len(result.runs_digest) == 64
    assert result.to_dict()["runs_digest"] == result.runs_digest
    reordered = replace(result, runs=tuple(reversed(result.runs)))
    assert reordered.runs_digest == result.runs_digest
    shorter = replace(result, runs=result.runs[:-1])
    assert shorter.runs_digest != result.runs_digest
    assert plan(tiny_campaign(), registry=REGISTRY).runs_digest == result.runs_digest


def test_a_session_record_without_the_plan_fields_still_loads(tmp_path: Path) -> None:
    summary = run(tiny_campaign(), tmp_path, registry=REGISTRY)
    (session,) = read_sessions(summary.directory)
    document = session.to_dict()
    assert document["plan_digest"] == read_plan(summary.directory).runs_digest
    assert document["plan_left"] == document["plan_entered"] == []
    assert SessionRecord.from_dict(document) == session
    del document["plan_digest"], document["plan_left"], document["plan_entered"]
    stored = SessionRecord.from_dict(document)
    assert stored.plan_digest is None
    assert stored.plan_left == stored.plan_entered == ()
    assert stored == replace(session, plan_digest=None)
