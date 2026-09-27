"""Replay: completed runs reproduce, and edited records are caught."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.campaigns import (
    Protocol,
    SolverSpec,
    Verdict,
    read_plan,
    read_runs,
    replay,
    run,
)
from dispatchatlas.campaigns.store import run_path
from dispatchatlas.core import ConfigError, StopCriteria
from tests.campaigns.solvers import registry, tiny_campaign

REGISTRY = registry()


def _record_path(directory: Path, index: int) -> Path:
    item = read_plan(directory).runs[index]
    return run_path(directory, item.solver, item.problem, item.replicate, item.protocol)


def test_every_completed_run_reproduces(tmp_path: Path) -> None:
    campaign = tiny_campaign()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    report = replay(campaign, tmp_path, registry=REGISTRY)
    assert report.ok
    assert report.count(Verdict.MATCH) == summary.completed == len(report.entries)
    assert [entry.run_id for entry in report.entries] == [
        item.run_id for item in read_plan(summary.directory).runs
    ]


def test_replay_on_worker_processes_matches_too(tmp_path: Path) -> None:
    campaign = tiny_campaign(replicates=1)
    run(campaign, tmp_path, registry=REGISTRY)
    report = replay(campaign, tmp_path, registry=REGISTRY, backend="process", workers=2)
    assert report.ok
    assert report.count(Verdict.MATCH) == len(report.entries) == 6


def test_an_edited_record_is_a_mismatch_with_its_differences(tmp_path: Path) -> None:
    campaign = tiny_campaign()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    path = _record_path(summary.directory, 1)
    document = json.loads(path.read_text())
    recorded = document["outcome"]["value"]
    document["outcome"]["value"] = recorded - 1.0
    document["outcome"]["objectives"]["makespan"] = recorded - 1.0
    path.write_text(json.dumps(document))

    report = replay(campaign, tmp_path, registry=REGISTRY)
    assert not report.ok
    (entry,) = [entry for entry in report.entries if entry.verdict is Verdict.MISMATCH]
    assert entry.run_id == document["run_id"]
    assert "does not match the result hash" in entry.detail
    fields = {difference.field: difference for difference in entry.differences}
    assert fields["value"].recorded == recorded - 1.0
    assert fields["value"].replayed == recorded
    assert "objectives" in fields
    assert report.count(Verdict.MATCH) == len(report.entries) - 1


def test_an_edited_result_hash_is_a_mismatch(tmp_path: Path) -> None:
    campaign = tiny_campaign()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    path = _record_path(summary.directory, 0)
    document = json.loads(path.read_text())
    document["result_hash"] = "0" * 64
    path.write_text(json.dumps(document))
    report = replay(campaign, tmp_path, registry=REGISTRY)
    mismatched = [entry for entry in report.entries if entry.verdict is Verdict.MISMATCH]
    assert [entry.run_id for entry in mismatched] == [document["run_id"]]
    assert mismatched[0].differences == ()


def test_runs_stopped_by_the_clock_are_skipped(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("input-order"), SolverSpec("random-search")),
        protocols=(Protocol("clock", StopCriteria(time_limit=0.05)),),
        replicates=1,
    )
    run(campaign, tmp_path, registry=REGISTRY)
    report = replay(campaign, tmp_path, registry=REGISTRY)
    verdicts = {(entry.solver, entry.verdict) for entry in report.entries}
    assert verdicts == {("input-order", Verdict.MATCH), ("random-search", Verdict.SKIPPED)}
    assert all(
        "wall-clock limit" in entry.detail
        for entry in report.entries
        if entry.verdict is Verdict.SKIPPED
    )
    assert report.ok


def test_selected_runs_and_errors(tmp_path: Path) -> None:
    campaign = tiny_campaign()
    summary = run(campaign, tmp_path, registry=REGISTRY, max_runs=3)
    runs = read_plan(summary.directory).runs
    report = replay(campaign, tmp_path, registry=REGISTRY, runs=[runs[0].run_id, runs[5].run_id])
    assert [(entry.run_id, entry.verdict) for entry in report.entries] == [
        (runs[0].run_id, Verdict.MATCH),
        (runs[5].run_id, Verdict.SKIPPED),
    ]
    assert report.entries[1].detail == "the run has no completed record"
    with pytest.raises(ConfigError, match="the plan has no runs run-nope"):
        replay(campaign, tmp_path, registry=REGISTRY, runs=["run-nope"])
    with pytest.raises(ConfigError, match="another configuration"):
        replay(replace(campaign, seed=99), tmp_path, registry=REGISTRY)
    with pytest.raises(ConfigError, match="holds no campaign results"):
        replay(campaign, tmp_path / "elsewhere", registry=REGISTRY)
    assert len(read_runs(summary.directory)) == 3
