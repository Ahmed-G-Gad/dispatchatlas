"""Replay: run completed runs again and check that they reproduce.

:func:`replay` re-executes runs of a campaign from their recorded seeds, with
the same solver parameters and stop criteria, and compares result hashes.
Before that it checks each record against its own result hash, so a record
edited after it was written is caught even though its run reproduces. It
never writes to the store.

A run stopped by its wall-clock limit is skipped: how far it got depends on
the speed and load of the machine. Other runs reproduce exactly on any
machine where the solver's arithmetic does; transcendental functions of the
C library may differ in their last bit between platforms, which can steer a
stochastic search elsewhere. Each mismatch lists the outcome fields that
differ.
"""

from __future__ import annotations

import contextlib
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from dispatchatlas.benchmarks.registry import FamilyRegistry
from dispatchatlas.campaigns.backends import backend_for
from dispatchatlas.campaigns.config import Campaign
from dispatchatlas.campaigns.execution import Completion, Result
from dispatchatlas.campaigns.planning import PlannedRun, prepare
from dispatchatlas.campaigns.records import RunRecord
from dispatchatlas.campaigns.runner import overridden, tasks_for
from dispatchatlas.campaigns.store import campaign_directory, read_runs, recorded_config_hash
from dispatchatlas.core.errors import ConfigError
from dispatchatlas.core.logs import get_logger
from dispatchatlas.solvers.registry import SolverRegistry

__all__ = ["Difference", "ReplayEntry", "ReplayReport", "Verdict", "replay"]

_LOG = get_logger(__name__)


class Verdict(StrEnum):
    """What replaying one run showed."""

    MATCH = "match"
    """The run reproduced its recorded result, and the record is intact."""
    MISMATCH = "mismatch"
    """The result differs, or the record no longer matches its result hash."""
    SKIPPED = "skipped"
    """The run cannot be replayed, or has no completed record."""
    FAILED = "failed"
    """The replay attempt failed."""


@dataclass(frozen=True)
class Difference:
    """One outcome field whose recorded and replayed values differ.

    Attributes:
        field: Name of the field.
        recorded: The value in the record.
        replayed: The value the replay produced.
    """

    field: str
    recorded: Any
    replayed: Any


@dataclass(frozen=True)
class ReplayEntry:
    """The verdict on one run.

    Attributes:
        run_id: Run id.
        solver: Solver name.
        problem: Problem id.
        protocol: Protocol name.
        replicate: Replicate number.
        verdict: What the replay showed.
        detail: Explanation of a verdict other than a match.
        differences: Differing outcome fields, for a mismatch.
    """

    run_id: str
    solver: str
    problem: str
    protocol: str
    replicate: int
    verdict: Verdict
    detail: str = ""
    differences: tuple[Difference, ...] = ()


@dataclass(frozen=True)
class ReplayReport:
    """The verdicts of a replay, in plan order.

    Attributes:
        campaign: Campaign id.
        config_hash: Config hash of the campaign.
        entries: One entry per selected run.
    """

    campaign: str
    config_hash: str
    entries: tuple[ReplayEntry, ...]

    def count(self, verdict: Verdict) -> int:
        """Return how many runs got a verdict."""
        return sum(1 for entry in self.entries if entry.verdict is verdict)

    @property
    def ok(self) -> bool:
        """Whether no run mismatched or failed."""
        return not any(
            entry.verdict in (Verdict.MISMATCH, Verdict.FAILED) for entry in self.entries
        )


def replay(
    campaign: Campaign,
    root: str | Path,
    *,
    runs: Iterable[str] | None = None,
    registry: SolverRegistry | None = None,
    families: FamilyRegistry | None = None,
    backend: str | None = None,
    workers: int | str | None = None,
    affinity: str | Sequence[int] | None = None,
) -> ReplayReport:
    """Re-execute completed runs and compare their results with the records.

    Args:
        campaign: The campaign the results belong to.
        root: Output root the campaign ran under.
        runs: Ids of the runs to replay. Defaults to every completed run.
        registry: Solver registry. Defaults to the built-in solvers and plugins.
        families: Benchmark-family registry. Defaults to the built-in families
            and plugins.
        backend: Override of the campaign's execution backend.
        workers: Override of the campaign's worker count, or ``"auto"``.
        affinity: Override of the campaign's ``affinity``: ``"inherit"``,
            ``"pin"``, or processor indices.

    Returns:
        The verdict on every selected run.

    Raises:
        ConfigError: If the directory holds another configuration, has no
            results, a requested run id is not in the plan, or runs are to
            be replayed and the workers cannot be pinned as ``affinity``
            asks.
    """
    execution = overridden(campaign.execution, backend, workers, affinity)
    prepared = prepare(campaign, registry=registry, families=families)
    plan = prepared.plan
    directory = campaign_directory(root, campaign.id)
    _check_store(directory, plan.config_hash)
    records = {record.run_id: record for record in read_runs(directory, verify=False)}
    if runs is None:
        selected = [item for item in plan.runs if item.run_id in records]
    else:
        wanted = set(runs)
        unknown = sorted(identifier for identifier in wanted if not plan.has_run(identifier))
        if unknown:
            raise ConfigError(f"the plan has no runs {', '.join(unknown)}")
        selected = [item for item in plan.runs if item.run_id in wanted]

    verdicts: dict[str, ReplayEntry] = {}
    replayable: list[PlannedRun] = []
    for item in selected:
        record = records.get(item.run_id)
        if record is None:
            verdicts[item.run_id] = _entry(item, Verdict.SKIPPED, "the run has no completed record")
        elif not record.outcome.replayable:
            detail = "the run stopped at its wall-clock limit, which does not reproduce"
            verdicts[item.run_id] = _entry(item, Verdict.SKIPPED, detail)
        else:
            replayable.append(item)
    attempts = {item.run_id: records[item.run_id].provenance.attempt for item in replayable}
    queue = deque(tasks_for(prepared, replayable, execution, attempts.__getitem__))
    count = execution.worker_count()
    chosen = backend_for(
        execution.backend,
        count,
        cpus=execution.worker_cpus(count) if queue else None,
        threads=execution.threads_per_worker,
        startup_limit=execution.startup_limit,
    )
    with contextlib.closing(chosen.run(queue)) as results:
        for result in results:
            item = plan.run(result.run_id)
            verdicts[item.run_id] = _judge(item, records[item.run_id], result)
    report = ReplayReport(
        campaign=campaign.id,
        config_hash=plan.config_hash,
        entries=tuple(verdicts[item.run_id] for item in selected),
    )
    _LOG.info(
        "replay of campaign %s: %d matched, %d mismatched, %d failed, %d skipped",
        campaign.id,
        report.count(Verdict.MATCH),
        report.count(Verdict.MISMATCH),
        report.count(Verdict.FAILED),
        report.count(Verdict.SKIPPED),
    )
    return report


def _check_store(directory: Path, config_hash: str) -> None:
    found = recorded_config_hash(directory)
    if found is None:
        raise ConfigError(f"{directory} holds no campaign results")
    if found != config_hash:
        raise ConfigError(
            f"{directory} holds results of another configuration (config hash "
            f"{found[:12]}, now {config_hash[:12]})"
        )


def _entry(
    item: PlannedRun, verdict: Verdict, detail: str = "", differences: Sequence[Difference] = ()
) -> ReplayEntry:
    return ReplayEntry(
        run_id=item.run_id,
        solver=item.solver,
        problem=item.problem,
        protocol=item.protocol,
        replicate=item.replicate,
        verdict=verdict,
        detail=detail,
        differences=tuple(differences),
    )


def _judge(item: PlannedRun, record: RunRecord, result: Result) -> ReplayEntry:
    """Compare a replayed attempt with the record of the run."""
    if not isinstance(result, Completion):
        return _entry(item, Verdict.FAILED, f"the replay failed: {result.error.message}")
    problems = list(record.verify())
    if record.seed != item.seed:
        problems.append(f"the record's seed {record.seed} is not the planned {item.seed}")
    if result.result_hash != record.result_hash:
        problems.append("the replayed result differs from the recorded one")
    if not problems:
        return _entry(item, Verdict.MATCH)
    differences = _differences(record.outcome.hashed(), result.outcome.hashed())
    return _entry(item, Verdict.MISMATCH, "; ".join(problems), differences)


def _differences(recorded: dict[str, Any], replayed: dict[str, Any]) -> list[Difference]:
    found: list[Difference] = []
    for key in sorted(set(recorded) | set(replayed)):
        before, after = recorded.get(key), replayed.get(key)
        if before == after:
            continue
        if isinstance(before, list) and isinstance(after, list):
            index = next(
                (i for i, (a, b) in enumerate(zip(before, after, strict=False)) if a != b),
                min(len(before), len(after)),
            )
            found.append(
                Difference(
                    key,
                    f"{len(before)} items, item {index}: {_item(before, index)}",
                    f"{len(after)} items, item {index}: {_item(after, index)}",
                )
            )
        else:
            found.append(Difference(key, before, after))
    return found


def _item(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None
