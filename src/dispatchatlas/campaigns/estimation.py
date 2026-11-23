"""Estimates: how many runs a campaign needs, and how long they can take.

:func:`estimate` plans a campaign without running anything and bounds the
duration of its solves from the protocols. The solve of a run whose protocol
sets ``time_limit`` takes at most its hard limit when the process backend
enforces one, and otherwise relies on the solver honouring the time limit.
Every attempt may take that long, and a run gets up to ``1 + retries``
attempts. The time a worker takes to begin a solve, which the start-up limit
bounds, is not counted. A run whose protocol bounds only evaluations or
iterations has no duration bound, so those runs are counted apart, unless a
time cap of its solver (:attr:`~dispatchatlas.campaigns.config.SolverSpec.time_caps`)
gives it a time limit.

Given a cost model (:class:`~dispatchatlas.campaigns.costs.CostModel`), the
estimate also projects the seconds the remaining runs take, from their
evaluations and the measured seconds per evaluation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dispatchatlas.benchmarks.registry import FamilyRegistry
from dispatchatlas.campaigns.config import Campaign
from dispatchatlas.campaigns.costs import CostModel
from dispatchatlas.campaigns.planning import PlannedRun, prepare
from dispatchatlas.campaigns.runner import overridden
from dispatchatlas.campaigns.store import (
    Ledger,
    Store,
    campaign_directory,
    recorded_config_hash,
)
from dispatchatlas.campaigns.timeout import hard_limit
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.logs import get_logger
from dispatchatlas.core.solver import StopCriteria
from dispatchatlas.solvers.registry import SolverRegistry

__all__ = ["Estimate", "estimate"]

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class Estimate:
    """The size of a campaign and bounds on its duration.

    Attributes:
        planned: Runs in the plan.
        completed: Runs an existing store already completed.
        failed: Runs that failures in an existing store settled.
        remaining: Runs that :func:`~dispatchatlas.campaigns.runner.run` would
            attempt: neither completed nor settled.
        problems: Problems in the plan.
        solvers: Solvers in the plan.
        protocols: Protocols in the plan.
        exclusions: Problem and solver pairs that do not run.
        by_protocol: Remaining runs per protocol.
        by_solver: Remaining runs per solver.
        workers: Runs that execute at once.
        timed: Remaining runs that have a time limit, their protocol's or a
            cap's.
        untimed: Remaining runs without a time limit, whose duration has no bound.
        budget_seconds: Sum of the time limits of the timed runs.
        worst_case_seconds: Sum of the longest time the solve of every
            attempt of every timed run may take: its hard limit when
            enforced, else its time limit, times ``1 + retries``.
        projected_seconds: Seconds the remaining runs are projected to take
            in all, or ``None`` when the estimate was made without a cost
            model. A run contributes its evaluation limit times the model's
            seconds per evaluation for its solver and problem, at most its
            time limit; a run with a time limit and no evaluation limit
            contributes the time limit.
    """

    planned: int
    completed: int
    failed: int
    remaining: int
    problems: int
    solvers: int
    protocols: int
    exclusions: int
    by_protocol: Mapping[str, int]
    by_solver: Mapping[str, int]
    workers: int
    timed: int
    untimed: int
    budget_seconds: float
    worst_case_seconds: float
    projected_seconds: float | None = None

    @property
    def budget_wall_seconds(self) -> float:
        """:attr:`budget_seconds` spread evenly over the workers."""
        return self.budget_seconds / self.workers

    @property
    def worst_case_wall_seconds(self) -> float:
        """:attr:`worst_case_seconds` spread evenly over the workers."""
        return self.worst_case_seconds / self.workers


def estimate(
    campaign: Campaign,
    *,
    root: str | Path | None = None,
    registry: SolverRegistry | None = None,
    families: FamilyRegistry | None = None,
    backend: str | None = None,
    workers: int | str | None = None,
    costs: CostModel | None = None,
) -> Estimate:
    """Plan a campaign and bound the time its remaining runs can take.

    Args:
        campaign: The campaign.
        root: Output root. When given, runs that the campaign's directory
            already completed, or whose failures settled them, are left out.
        registry: Solver registry. Defaults to the built-in solvers and plugins.
        families: Benchmark-family registry. Defaults to the built-in families
            and plugins.
        backend: Override of the campaign's execution backend.
        workers: Override of the campaign's worker count, or ``"auto"``.
        costs: A cost model that projects the seconds the remaining runs
            take. ``None`` leaves :attr:`Estimate.projected_seconds` unset and
            changes nothing else.

    Returns:
        The estimate.

    Raises:
        ConfigError: If the campaign cannot be planned. Also, with ``costs``,
            if the model measured no group of a remaining run, or a remaining
            run has neither an evaluation limit nor a time limit, so that
            nothing projects its duration.
    """
    execution = overridden(campaign.execution, backend, workers)
    plan = prepare(campaign, registry=registry, families=families).plan
    ledger = Ledger()
    if root is not None:
        directory = campaign_directory(root, campaign.id)
        if _same_configuration(directory, plan.config_hash):
            # A record kept aside that a session brings back needs no run.
            ledger = Store(directory).ledger(plan).restored()
    remaining = ledger.pending(plan, execution.max_attempts)
    failed = sum(1 for item in plan.runs if ledger.closed(item.run_id, execution.max_attempts))
    protocols = {protocol.name: protocol for protocol in plan.protocols}
    specs = {spec.name: spec for spec in campaign.solvers}
    timed = untimed = 0
    budget = worst = projected = 0.0
    for item in remaining:
        stop, _ = specs[item.solver].capped(protocols[item.protocol], item.problem)
        if costs is not None:
            projected += _projected(costs, item, stop)
        if stop.time_limit is None:
            untimed += 1
            continue
        timed += 1
        budget += stop.time_limit
        limit = hard_limit(stop, execution)
        worst += execution.max_attempts * (stop.time_limit if limit is None else limit)
    return Estimate(
        planned=len(plan.runs),
        completed=len(ledger.completed),
        failed=failed,
        remaining=len(remaining),
        problems=len(plan.problems),
        solvers=len(plan.solvers),
        protocols=len(plan.protocols),
        exclusions=len(plan.exclusions),
        by_protocol=dict(Counter(item.protocol for item in remaining)),
        by_solver=dict(Counter(item.solver for item in remaining)),
        workers=execution.worker_count(),
        timed=timed,
        untimed=untimed,
        budget_seconds=budget,
        worst_case_seconds=worst,
        projected_seconds=None if costs is None else projected,
    )


def _projected(costs: CostModel, item: PlannedRun, stop: StopCriteria) -> float:
    """Return the seconds a run is projected to take under its stop criteria."""
    if stop.max_evaluations is None:
        if stop.time_limit is None:
            raise ConfigError(
                f"run {item.run_id} ({item.solver} on {item.problem}, protocol {item.protocol}) "
                "has no evaluation limit and no time limit, so nothing projects its duration"
            )
        return stop.time_limit
    spent = costs.seconds(item.solver, item.problem, stop.max_evaluations)
    return spent if stop.time_limit is None else min(spent, stop.time_limit)


def _same_configuration(directory: Path, config_hash: str) -> bool:
    """Return whether a campaign directory holds results of this configuration."""
    found = recorded_config_hash(directory)
    if found is not None and found != config_hash:
        _LOG.warning("%s holds results of another configuration; counting none", directory)
    return found == config_hash
