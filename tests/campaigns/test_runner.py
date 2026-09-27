"""The runner and the store: resumption, retries, settlement, and the readers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.campaigns import (
    Category,
    SolverSpec,
    campaign_directory,
    plan,
    read_campaign,
    read_failures,
    read_plan,
    read_runs,
    read_sessions,
    run,
)
from dispatchatlas.campaigns.records import ErrorKind, SessionStatus
from dispatchatlas.campaigns.store import Store, failure_path, run_path
from dispatchatlas.core import ConfigError, SerializationError, check_schedule
from tests.campaigns.solvers import registry, tiny_campaign
from tests.helpers import chain_problem

REGISTRY = registry()


def _hashes(directory: Path) -> dict[str, str]:
    return {record.run_id: record.result_hash for record in read_runs(directory)}


def test_a_campaign_writes_one_verified_record_per_run(tmp_path: Path) -> None:
    campaign = tiny_campaign()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    assert summary.done
    assert summary.session == 1
    assert (summary.planned, summary.completed, summary.failed) == (10, 10, 0)
    assert summary.directory == campaign_directory(tmp_path, "tiny")

    records = read_runs(summary.directory)
    stored = read_plan(summary.directory)
    assert stored == plan(campaign, registry=REGISTRY)
    assert [record.run_id for record in records] == [item.run_id for item in stored.runs]
    for record, planned in zip(records, stored.runs, strict=True):
        assert record.config_hash == campaign.config_hash
        assert record.seed == planned.seed
        assert record.solver.name == planned.solver
        assert record.provenance.attempt == 1
        assert record.provenance.session == 1
        assert record.provenance.worker == "main"
        assert record.provenance.hard_limit is None
        assert record.provenance.wall_seconds >= 0.0
        assert record.outcome.feasible
        path = run_path(
            summary.directory, planned.solver, planned.problem, planned.replicate, planned.protocol
        )
        assert path.is_file()
        text = path.read_text()
        assert text.endswith("\n")
        assert text.isascii()
        assert json.loads(text)["schema"] == "dispatchatlas.run/1"
    assert read_failures(summary.directory) == ()
    assert read_campaign(summary.directory) == campaign


def test_schedules_are_checked_and_can_be_left_out(tmp_path: Path) -> None:
    from dispatchatlas.benchmarks import generate  # noqa: PLC0415

    with_schedules = read_runs(run(tiny_campaign(), tmp_path / "a", registry=REGISTRY).directory)
    problem = generate("flow-shop", 1, jobs=3, machines=2)
    for record in with_schedules:
        if record.problem.id == problem.id:
            assert record.schedule is not None
            assert check_schedule(problem, record.schedule) == ()
    without = tiny_campaign(record_schedules=False)
    records = read_runs(run(without, tmp_path / "b", registry=REGISTRY).directory)
    assert all(record.schedule is None for record in records)
    assert {r.run_id: r.result_hash for r in records} == {
        r.run_id: r.result_hash for r in with_schedules
    }


def test_resuming_completes_the_campaign_with_the_same_results(tmp_path: Path) -> None:
    campaign = tiny_campaign()
    first = run(campaign, tmp_path / "split", registry=REGISTRY, max_runs=4)
    assert (first.completed, first.pending, first.attempted) == (4, 6, 4)
    second = run(campaign, tmp_path / "split", registry=REGISTRY)
    assert (second.session, second.attempted, second.completed, second.pending) == (2, 6, 10, 0)
    third = run(campaign, tmp_path / "split", registry=REGISTRY)
    assert third.attempted == 0
    assert third.done
    whole = run(campaign, tmp_path / "whole", registry=REGISTRY)
    assert _hashes(second.directory) == _hashes(whole.directory)
    sessions = read_sessions(second.directory)
    assert [session.session for session in sessions] == [1, 2, 3]
    assert all(session.status is SessionStatus.FINISHED for session in sessions)
    assert sessions[0].counts["attempted"] == 4
    assert sessions[1].counts["completed"] == 10


def test_an_interrupted_session_resumes_where_it_stopped(tmp_path: Path) -> None:
    marker = tmp_path / "interrupt-once"
    solvers = (
        SolverSpec("input-order"),
        SolverSpec("test-interrupter", parameters={"marker": str(marker)}),
        SolverSpec("random-search"),
    )
    campaign = tiny_campaign(solvers=solvers)
    with pytest.raises(KeyboardInterrupt):
        run(campaign, tmp_path / "out", registry=REGISTRY)
    directory = campaign_directory(tmp_path / "out", "tiny")
    (session,) = read_sessions(directory)
    assert session.status is SessionStatus.INTERRUPTED
    assert session.finished is not None
    assert session.counts["completed"] == 1
    assert len(read_runs(directory)) == 1
    summary = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert summary.done
    assert summary.completed == len(plan(campaign, registry=REGISTRY).runs)
    assert _hashes(directory) == _hashes(
        run(campaign, tmp_path / "again", registry=REGISTRY).directory
    )


def test_recoverable_failures_are_retried_within_the_budget(tmp_path: Path) -> None:
    marker = tmp_path / "calls"
    marker.mkdir()
    solvers = (SolverSpec("test-flaky", parameters={"marker": str(marker), "failures": 1}),)
    campaign = tiny_campaign(solvers=solvers, replicates=1, retries=1)
    summary = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert (summary.completed, summary.failed, summary.errors, summary.succeeded) == (2, 0, 2, 2)
    failures = read_failures(summary.directory)
    assert [(f.provenance.attempt, f.category) for f in failures] == [
        (1, Category.RECOVERABLE),
        (1, Category.RECOVERABLE),
    ]
    assert failures[0].error.type == "RuntimeError"
    assert failures[0].error.kind is ErrorKind.EXCEPTION
    assert "transient failure 1" in failures[0].error.message
    assert all(record.provenance.attempt == 2 for record in read_runs(summary.directory))
    first = failures[0]
    path = failure_path(
        summary.directory,
        first.solver.name,
        first.problem.id,
        first.replicate,
        first.protocol.name,
        1,
    )
    assert path.is_file()


def test_settled_runs_wait_for_a_larger_retry_budget(tmp_path: Path) -> None:
    marker = tmp_path / "calls"
    marker.mkdir()
    solvers = (SolverSpec("test-flaky", parameters={"marker": str(marker), "failures": 1}),)
    campaign = tiny_campaign(solvers=solvers, replicates=1)
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert (first.completed, first.failed, first.pending) == (0, 2, 0)
    again = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert again.attempted == 0
    assert again.failed == 2
    retried = run(campaign.with_execution(retries=1), tmp_path / "out", registry=REGISTRY)
    assert (retried.attempted, retried.completed, retried.failed) == (2, 2, 0)
    assert all(record.provenance.attempt == 2 for record in read_runs(retried.directory))
    assert all(record.provenance.session == 3 for record in read_runs(retried.directory))


def test_terminal_failures_settle_a_run_at_once(tmp_path: Path) -> None:
    solvers = (SolverSpec("input-order"), SolverSpec("test-refuser"))
    campaign = tiny_campaign(solvers=solvers, retries=3)
    summary = run(campaign, tmp_path, registry=REGISTRY)
    assert (summary.completed, summary.failed, summary.errors) == (2, 2, 2)
    failures = read_failures(summary.directory)
    assert {(f.solver.name, f.provenance.attempt, f.category) for f in failures} == {
        ("test-refuser", 1, Category.TERMINAL)
    }
    assert failures[0].error.type == "dispatchatlas.core.errors.SolverError"
    again = run(campaign, tmp_path, registry=REGISTRY, retry_failed=True)
    assert (again.attempted, again.failed) == (2, 2)
    attempts = sorted(f.provenance.attempt for f in read_failures(summary.directory))
    assert attempts == [1, 1, 2, 2]


def test_another_configuration_is_refused_unless_fresh(tmp_path: Path) -> None:
    campaign = tiny_campaign()
    first = run(campaign, tmp_path, registry=REGISTRY)
    changed = replace(campaign, seed=campaign.seed + 1)
    with pytest.raises(ConfigError, match="another configuration"):
        run(changed, tmp_path, registry=REGISTRY)
    assert _hashes(first.directory)
    fresh = run(changed, tmp_path, registry=REGISTRY, fresh=True)
    assert fresh.session == 1
    assert fresh.done
    assert {record.config_hash for record in read_runs(fresh.directory)} == {changed.config_hash}


def test_a_campaign_directory_has_one_runner_at_a_time(tmp_path: Path) -> None:
    campaign = tiny_campaign()
    with (
        Store(campaign_directory(tmp_path, campaign.id)).locked(),
        pytest.raises(ConfigError, match="in use by another campaign runner"),
    ):
        run(campaign, tmp_path, registry=REGISTRY)
    assert run(campaign, tmp_path, registry=REGISTRY).done


def test_invalid_or_foreign_records_are_run_again(tmp_path: Path) -> None:
    campaign = tiny_campaign()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    records = read_runs(summary.directory)
    planned = read_plan(summary.directory).runs

    def path_of(index: int) -> Path:
        item = planned[index]
        return run_path(summary.directory, item.solver, item.problem, item.replicate, item.protocol)

    edited = json.loads(path_of(0).read_text())
    edited["outcome"]["value"] += 1.0
    path_of(0).write_text(json.dumps(edited))
    foreign = json.loads(path_of(1).read_text())
    foreign["run_id"] = "run-ffffffffffffffff"
    path_of(1).write_text(json.dumps(foreign))
    path_of(2).write_text("{ not json")

    with pytest.raises(SerializationError):
        read_runs(summary.directory)
    again = run(campaign, tmp_path, registry=REGISTRY)
    assert again.attempted == 3
    assert [r.result_hash for r in read_runs(summary.directory)] == [r.result_hash for r in records]


def test_moving_instance_files_keeps_the_results(tmp_path: Path) -> None:
    from dispatchatlas.campaigns import FileSource  # noqa: PLC0415
    from dispatchatlas.core import save_problem  # noqa: PLC0415

    save_problem(chain_problem(), tmp_path / "old" / "chain.json")
    campaign = replace(
        tiny_campaign(instances=(FileSource(Path("old/chain.json")),)), base=tmp_path
    )
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    (tmp_path / "old").rename(tmp_path / "new")
    moved = replace(campaign, instances=(FileSource(Path("new/chain.json")),))
    again = run(moved, tmp_path / "out", registry=REGISTRY)
    assert again.attempted == 0
    assert again.completed == first.completed == first.planned


def test_changed_problems_are_rerun(tmp_path: Path) -> None:
    from dispatchatlas.campaigns import FileSource  # noqa: PLC0415
    from dispatchatlas.core import save_problem  # noqa: PLC0415

    save_problem(chain_problem(), tmp_path / "chain.json")
    campaign = replace(tiny_campaign(instances=(FileSource(Path("chain.json")),)), base=tmp_path)
    first = run(campaign, tmp_path / "out", registry=REGISTRY)
    save_problem(
        replace(chain_problem(), objectives=("total_completion_time",)), tmp_path / "chain.json"
    )
    second = run(campaign, tmp_path / "out", registry=REGISTRY)
    assert second.attempted == first.planned
    assert {r.outcome.objective for r in read_runs(second.directory)} == {"total_completion_time"}
