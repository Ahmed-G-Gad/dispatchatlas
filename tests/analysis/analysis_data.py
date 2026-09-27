"""Datasets for the analysis tests: a real tiny campaign, and synthetic runs.

``make_dataset`` builds a dataset without running solvers, so a test can state
every run's outcome: a number is a succeeded run with that objective value, a
tuple a succeeded run with that objective vector, a list of tuples a front,
``"infeasible"`` a completed run without a feasible schedule, ``"failed"`` a
run whose attempts failed, and ``None`` a run that has no record yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from dispatchatlas.analysis.dataset import Dataset, Run, RunStatus
from dispatchatlas.campaigns import (
    Campaign,
    Execution,
    FamilySource,
    Outcome,
    Protocol,
    Provenance,
    RunRecord,
    SolverSpec,
    run,
)
from dispatchatlas.campaigns.planning import Plan, PlannedProblem, PlannedRun
from dispatchatlas.campaigns.records import ProblemRef, SolverEntry
from dispatchatlas.core import Status, StopCriteria, StopReason

type Entry = float | tuple[float, ...] | list[tuple[float, ...]] | str | None

SOLVERS = (
    SolverSpec("input-order"),
    SolverSpec("random-search"),
    SolverSpec("hill-climbing", "hc", {"restart_after": 5}),
    SolverSpec("shortest-processing-time"),
)


def run_tiny_campaign(
    root: Path,
    *,
    protocols: tuple[Protocol, ...] = (Protocol("evals", StopCriteria(max_evaluations=15)),),
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    replicates: int = 3,
) -> Path:
    """Run four cheap solvers on small flow shops and return the campaign directory."""
    campaign = Campaign(
        id="tiny",
        seed=11,
        instances=(FamilySource("flow-shop", seeds, {"jobs": 4, "machines": 3}),),
        solvers=SOLVERS,
        protocols=protocols,
        replicates=replicates,
        execution=Execution(backend="sequential"),
    )
    return run(campaign, root).directory


@pytest.fixture(scope="module")
def campaign_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A campaign directory of 56 runs: 4 solvers, 7 problems, 3 replicates."""
    return run_tiny_campaign(tmp_path_factory.mktemp("campaign"))


def make_dataset(
    cells: Mapping[str, Mapping[str, Sequence[Entry]]],
    *,
    objectives: tuple[str, ...] = ("makespan",),
    protocols: tuple[str, ...] = ("p",),
    deterministic: Sequence[str] = (),
) -> Dataset:
    """Build a dataset from ``cells[solver][problem]``, one entry per replicate."""
    problem_ids = list(dict.fromkeys(problem for row in cells.values() for problem in row))
    problems = tuple(
        PlannedProblem(problem, None, "0" * 64, 0, 3, 2, objectives, ()) for problem in problem_ids
    )
    solvers = tuple(
        SolverEntry(name, name, None, {}, "random-sampling", name not in deterministic, True, False)
        for name in cells
    )
    stops = tuple(Protocol(name, StopCriteria(max_evaluations=10)) for name in protocols)
    planned: list[PlannedRun] = []
    runs: list[Run] = []
    replicates = max(len(entries) for row in cells.values() for entries in row.values())
    for protocol in stops:
        for problem in problems:
            for replicate in range(replicates):
                for solver in solvers:
                    entries = cells[solver.name].get(problem.id, ())
                    if replicate >= len(entries):
                        continue
                    run_id = f"run-{protocol.name}-{solver.name}-{problem.id}-{replicate}"
                    planned.append(
                        PlannedRun(run_id, problem.id, solver.name, protocol.name, replicate, 7)
                    )
                    runs.append(
                        _run(run_id, solver, problem, protocol, replicate, entries[replicate])
                    )
    plan = Plan(
        campaign="synthetic",
        config_hash="0" * 64,
        seed=0,
        replicates=replicates,
        problems=problems,
        solvers=solvers,
        protocols=stops,
        runs=tuple(planned),
    )
    return Dataset(plan, tuple(runs), solvers, problems, stops)


def _run(
    run_id: str,
    solver: SolverEntry,
    problem: PlannedProblem,
    protocol: Protocol,
    replicate: int,
    entry: Entry,
) -> Run:
    def make(status: RunStatus, record: RunRecord | None = None) -> Run:
        return Run(run_id, solver.name, problem.id, protocol.name, replicate, 7, status, record)

    if entry is None:
        return make("missing")
    if entry == "failed":
        return make("failed")
    names = problem.objectives
    front: tuple[tuple[float, ...], ...] = ()
    feasible = entry != "infeasible"
    if isinstance(entry, str):
        vector: tuple[float, ...] = tuple(1e6 for _ in names)
    elif isinstance(entry, list):
        front = tuple(tuple(point) for point in entry)
        vector = front[0]
    elif isinstance(entry, tuple):
        vector = entry
    else:
        vector = tuple(float(entry) for _ in names)
    outcome = Outcome(
        status=Status.FEASIBLE if feasible else Status.VIOLATED,
        stop_reason=StopReason.EVALUATION_LIMIT,
        objective=names[0],
        value=vector[0],
        objectives=dict(zip(names, vector, strict=True)),
        bound=None,
        gap=None,
        feasible=feasible,
        violations=() if feasible else ("deadline",),
        evaluations=10,
        iterations=10,
        elapsed=0.01,
        front=front,
    )
    record = RunRecord(
        run_id=run_id,
        campaign="synthetic",
        config_hash="0" * 64,
        problem=ProblemRef(problem.id, None, "0" * 64),
        solver=solver,
        protocol=protocol,
        replicate=replicate,
        seed=7,
        outcome=outcome,
        result_hash=outcome.result_hash(),
        schedule=None,
        provenance=Provenance(1, 1, "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", 0.01),
    )
    return make("succeeded" if feasible else "infeasible", record)
