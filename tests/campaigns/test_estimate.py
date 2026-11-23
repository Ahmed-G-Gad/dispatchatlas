"""Estimates: run counts and time bounds from the protocols."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dispatchatlas.campaigns import Cost, CostModel, Protocol, SolverSpec, estimate, run
from dispatchatlas.core import ConfigError, StopCriteria
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


# ------------------------------------------------------------ projected seconds

FIRST, SECOND = "flow-shop-3x2-1", "flow-shop-3x2-2"
"""The ids of the two problems of the tiny campaigns."""


def _costs(per_evaluation: dict[str, float]) -> CostModel:
    """A model with one group for each solver, at the given seconds per evaluation."""
    return CostModel(
        {name: Cost(seconds, 1) for name, seconds in per_evaluation.items()},
        lambda solver, problem: solver,
    )


COSTS = _costs({"input-order": 0.5, "random-search": 0.125, "hc": 0.25})
"""The made-up costs of the three solvers of the tiny campaign, binary fractions."""


def test_a_cost_model_projects_the_seconds_of_the_remaining_runs() -> None:
    campaign = tiny_campaign(protocols=(Protocol("evals", StopCriteria(max_evaluations=10)),))
    result = estimate(campaign, registry=REGISTRY, costs=COSTS)
    # Two runs of input order at 5 seconds, four of random search at 1.25, four of hc at 2.5.
    assert result.projected_seconds == pytest.approx(2 * 5.0 + 4 * 1.25 + 4 * 2.5)
    # The model changes nothing else, and no model leaves the projection unset.
    plain = estimate(campaign, registry=REGISTRY)
    assert plain.projected_seconds is None
    assert replace(result, projected_seconds=None) == plain


def test_a_projection_is_at_most_the_time_limit_of_its_run() -> None:
    clock = Protocol("clock", StopCriteria(time_limit=2.0, max_evaluations=10))
    campaign = tiny_campaign(protocols=(clock, Protocol("evals", StopCriteria(max_evaluations=10))))
    result = estimate(campaign, registry=REGISTRY, costs=COSTS)
    # Under the clock: 2 * min(5, 2) + 4 * min(1.25, 2) + 4 * min(2.5, 2); under the evaluations,
    # 2 * 5 + 4 * 1.25 + 4 * 2.5.
    assert result.projected_seconds == pytest.approx((4.0 + 5.0 + 8.0) + (10.0 + 5.0 + 10.0))


def test_a_run_with_only_a_time_limit_is_projected_at_its_limit_without_the_model() -> None:
    campaign = tiny_campaign(protocols=(Protocol("clock", StopCriteria(time_limit=3.0)),))
    result = estimate(campaign, registry=REGISTRY, costs=_costs({}))
    assert result.projected_seconds == pytest.approx(10 * 3.0)


def test_a_run_the_model_cannot_project_is_refused_and_not_left_out() -> None:
    campaign = tiny_campaign(protocols=(Protocol("evals", StopCriteria(max_evaluations=10)),))
    without_hc = _costs({"input-order": 0.5, "random-search": 0.125})
    with pytest.raises(ConfigError, match=r"no run of group 'hc' was measured"):
        estimate(campaign, registry=REGISTRY, costs=without_hc)
    iterations = tiny_campaign(protocols=(Protocol("steps", StopCriteria(max_iterations=5)),))
    with pytest.raises(ConfigError, match="no evaluation limit and no time limit"):
        estimate(iterations, registry=REGISTRY, costs=COSTS)
    # Without a model the same campaign estimates as before.
    assert estimate(iterations, registry=REGISTRY).untimed == 10


def test_a_cap_gives_a_run_the_time_limit_the_estimate_counts() -> None:
    caps = {"evals": {FIRST: 2.0, SECOND: 4.0}}
    campaign = tiny_campaign(
        solvers=(SolverSpec("random-search", time_caps=caps), SolverSpec("input-order")),
        protocols=(Protocol("evals", StopCriteria(max_evaluations=10)),),
        backend="process",
        workers=2,
        timeout_factor=1.5,
        grace=0.5,
        retries=1,
    )
    result = estimate(campaign, registry=REGISTRY)
    # The four runs of random search are timed by their caps, the two of input order are not.
    assert (result.remaining, result.timed, result.untimed) == (6, 4, 2)
    assert result.budget_seconds == pytest.approx(2 * 2.0 + 2 * 4.0)
    # Each of two attempts may take 1.5 * cap + 0.5 seconds.
    assert result.worst_case_seconds == pytest.approx(2 * 2 * (1.5 * 2.0 + 0.5 + 1.5 * 4.0 + 0.5))
    # A run is projected at most to its cap: 10 evaluations at 0.5 s is 5 s.
    slow = _costs({"input-order": 0.5, "random-search": 0.5})
    projected = estimate(campaign, registry=REGISTRY, costs=slow).projected_seconds
    assert projected == pytest.approx(2 * 2.0 + 2 * 4.0 + 2 * 5.0)
    uncapped = tiny_campaign(
        solvers=(SolverSpec("random-search"), SolverSpec("input-order")),
        protocols=(Protocol("evals", StopCriteria(max_evaluations=10)),),
    )
    assert estimate(uncapped, registry=REGISTRY).untimed == 6
