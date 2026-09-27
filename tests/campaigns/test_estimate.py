"""Estimates: run counts and time bounds from the protocols."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.campaigns import Protocol, SolverSpec, estimate, run
from dispatchatlas.core import StopCriteria
from tests.campaigns.solvers import registry, tiny_campaign

REGISTRY = registry()
PROTOCOLS = (
    Protocol("clock", StopCriteria(time_limit=2.0, max_evaluations=10)),
    Protocol("evals", StopCriteria(max_evaluations=10)),
)


def test_counts_runs_and_bounds_their_duration() -> None:
    campaign = tiny_campaign(
        protocols=PROTOCOLS, backend="process", workers=2, timeout_factor=1.5, grace=0.5, retries=1
    )
    result = estimate(campaign, registry=REGISTRY)
    assert (result.planned, result.completed, result.failed, result.remaining) == (20, 0, 0, 20)
    assert (result.problems, result.solvers, result.protocols, result.exclusions) == (2, 3, 2, 0)
    assert result.by_protocol == {"clock": 10, "evals": 10}
    assert result.by_solver == {"input-order": 4, "random-search": 8, "hc": 8}
    assert (result.timed, result.untimed, result.workers) == (10, 10, 2)
    assert result.budget_seconds == pytest.approx(20.0)
    # Each timed attempt may take 1.5 * 2 + 0.5 seconds, and a run gets two attempts.
    assert result.worst_case_seconds == pytest.approx(10 * 2 * 3.5)
    assert result.worst_case_wall_seconds == pytest.approx(35.0)
    assert result.budget_wall_seconds == pytest.approx(10.0)


def test_without_hard_limits_the_time_limit_bounds_each_attempt() -> None:
    campaign = tiny_campaign(protocols=PROTOCOLS, backend="sequential")
    result = estimate(campaign, registry=REGISTRY)
    assert result.worst_case_seconds == pytest.approx(result.budget_seconds) == pytest.approx(20.0)
    overridden = estimate(campaign, registry=REGISTRY, backend="process", workers=4)
    assert overridden.workers == 4
    assert overridden.worst_case_seconds == pytest.approx(10 * (2.0 + 5.0))


def test_an_existing_store_reduces_the_remaining_runs(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("input-order"), SolverSpec("random-search"), SolverSpec("test-refuser"))
    )
    run(campaign, tmp_path, registry=REGISTRY, max_runs=5)
    result = estimate(campaign, root=tmp_path, registry=REGISTRY)
    # Four runs per problem; the first five are the four of the first problem,
    # among them the refuser's, which its failure settles, and one of the second.
    assert (result.planned, result.completed, result.failed, result.remaining) == (8, 4, 1, 3)
    assert sum(result.by_solver.values()) == 3
    elsewhere = estimate(campaign, root=tmp_path / "empty", registry=REGISTRY)
    assert (elsewhere.completed, elsewhere.remaining) == (0, 8)
    changed = estimate(replace(campaign, seed=5), root=tmp_path, registry=REGISTRY)
    assert changed.completed == 0
