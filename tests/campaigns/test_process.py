"""The process backend: equal results, hard time limits, and crashing workers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from dispatchatlas.campaigns import (
    Category,
    FamilySource,
    Protocol,
    SolverSpec,
    read_failures,
    read_runs,
    run,
)
from dispatchatlas.campaigns.records import ErrorKind
from dispatchatlas.core import StopCriteria
from tests.campaigns.solvers import registry, tiny_campaign

REGISTRY = registry()
ONE_PROBLEM = (FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),)


def test_process_workers_reproduce_the_sequential_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    solvers = (
        SolverSpec("input-order"),
        SolverSpec("random-search"),
        SolverSpec("hill-climbing", "hc", {"restart_after": 5}),
        SolverSpec("test-environment"),
        SolverSpec("test-tabled", parameters={"weights": {"a": 1.0, "b": 2.0}}),
    )
    sequential = run(tiny_campaign(solvers=solvers), tmp_path / "s", registry=REGISTRY)
    parallel = run(
        tiny_campaign(solvers=solvers),
        tmp_path / "p",
        registry=REGISTRY,
        backend="process",
        workers=2,
    )
    assert sequential.done
    assert parallel.done
    first, second = read_runs(sequential.directory), read_runs(parallel.directory)
    assert [(r.run_id, r.result_hash) for r in first] == [(r.run_id, r.result_hash) for r in second]
    assert [r.schedule for r in first] == [r.schedule for r in second]
    assert {r.provenance.worker for r in second} <= {"worker-1", "worker-2"}
    assert all(r.provenance.cpu_seconds is not None for r in second)
    if sys.platform.startswith("linux"):
        assert all(r.provenance.peak_rss_bytes for r in second)
    probes = [r for r in second if r.solver.id == "test-environment"]
    assert {r.outcome.extras["omp_threads"] for r in probes} == {"1"}
    probes = [r for r in first if r.solver.id == "test-environment"]
    assert {r.outcome.extras["omp_threads"] for r in probes} == {""}
    tabled = [r for r in second if r.solver.id == "test-tabled"]
    assert {r.outcome.extras["weights"] for r in tabled} == {"a=1.0,b=2.0"}


def test_a_hard_limit_stops_a_solver_that_ignores_its_time_limit(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(
            SolverSpec("input-order"),
            SolverSpec("test-sleeper", parameters={"seconds": 30.0}),
        ),
        protocols=(Protocol("clock", StopCriteria(time_limit=0.2, max_evaluations=10)),),
        instances=ONE_PROBLEM,
        backend="process",
        timeout_factor=1.0,
        grace=0.3,
        retries=2,
    )
    started = time.perf_counter()
    summary = run(campaign, tmp_path, registry=REGISTRY)
    assert time.perf_counter() - started < 20.0
    assert (summary.completed, summary.failed, summary.pending) == (1, 1, 0)
    (failure,) = read_failures(summary.directory)
    assert failure.solver.name == "test-sleeper"
    assert failure.category is Category.TERMINAL
    assert failure.error.kind is ErrorKind.TIMEOUT
    assert "hard limit of 0.5 s" in failure.error.message
    assert failure.provenance.hard_limit == pytest.approx(0.5)
    assert 0.4 <= failure.provenance.wall_seconds < 20.0
    (record,) = read_runs(summary.directory)
    assert record.provenance.hard_limit == pytest.approx(0.5)


def test_a_crashing_worker_costs_only_its_attempt(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-crasher"), SolverSpec("input-order")),
        instances=ONE_PROBLEM,
        backend="process",
        retries=1,
    )
    summary = run(campaign, tmp_path, registry=REGISTRY)
    assert (summary.completed, summary.failed, summary.errors) == (1, 1, 2)
    failures = read_failures(summary.directory)
    assert [(f.provenance.attempt, f.category, f.error.kind) for f in failures] == [
        (1, Category.RECOVERABLE, ErrorKind.WORKER_EXIT),
        (2, Category.RECOVERABLE, ErrorKind.WORKER_EXIT),
    ]
    assert "exit code 7" in failures[0].error.message

    again = run(campaign.with_execution(retries=2), tmp_path, registry=REGISTRY)
    assert (again.attempted, again.failed) == (1, 1)
    failures = read_failures(summary.directory)
    assert [(f.provenance.attempt, f.provenance.session) for f in failures] == [
        (1, 1),
        (2, 1),
        (3, 2),
    ]


def test_a_solver_that_cannot_reach_a_worker_fails_terminally(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-unpicklable"), SolverSpec("input-order")),
        instances=ONE_PROBLEM,
        backend="process",
        retries=2,
    )
    summary = run(campaign, tmp_path, registry=REGISTRY)
    assert (summary.completed, summary.failed, summary.errors) == (1, 1, 1)
    (failure,) = read_failures(summary.directory)
    assert failure.error.kind is ErrorKind.TRANSFER
    assert failure.category is Category.TERMINAL
    assert "pickle" in failure.error.message.lower()
