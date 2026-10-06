"""What a solver does once in a process, and the one instant an attempt counts from."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import ClassVar

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
from dispatchatlas.campaigns.execution import (
    Completion,
    Failure,
    Task,
    attempt,
    prepare,
    prepare_task,
)
from dispatchatlas.campaigns.records import ErrorKind
from dispatchatlas.core import (
    MissingDependencyError,
    Problem,
    Solver,
    SolveResult,
    SolverInfo,
    StopCriteria,
    problem_digest,
)
from dispatchatlas.core.clock import clock_start
from dispatchatlas.solvers import default_registry
from dispatchatlas.solvers.rules import RandomSearch
from tests.campaigns.solvers import ColdStart, Phased, Prepared, registry, tiny_campaign
from tests.helpers import chain_problem, random_problem

REGISTRY = registry()
ROOT = Path(__file__).resolve().parents[2]
ONE_PROBLEM = (FamilySource("flow-shop", (1,), {"jobs": 3, "machines": 2}),)

_SWEEP = """
import json
import sys

from dispatchatlas.campaigns.execution import prepare
from dispatchatlas.core import DispatchAtlasError, StopCriteria
from dispatchatlas.solvers import default_registry
from tests.helpers import chain_problem

solvers = default_registry(plugins=False)
late, solved = {}, 0
for info in solvers.infos():
    if info.requires:
        continue
    solver = solvers.create(info.id)
    prepare(solver)
    before = set(sys.modules)
    try:
        solver.solve(chain_problem(), StopCriteria(max_evaluations=30), 2)
    except DispatchAtlasError:
        continue
    solved += 1
    new = sorted(set(sys.modules) - before)
    if new:
        late[info.id] = new[:5]
print(json.dumps({"late": late, "solved": solved}))
"""


def _task(solver: Solver, stop: StopCriteria) -> Task:
    problem = random_problem(5)
    return Task("run-0", 1, problem, problem_digest(problem), solver, stop, seed=3)


def test_what_a_solver_prepares_is_not_charged_to_its_first_run() -> None:
    stop = StopCriteria(time_limit=0.2)
    solver = Prepared(seconds=0.6, key="first-run")
    began = time.perf_counter()
    first = attempt(_task(solver, stop))
    assert time.perf_counter() - began >= 0.6
    assert isinstance(first, Completion)
    # The 0.6 s were paid before the clock of the attempt started.
    assert first.measurement.wall_seconds < 0.5
    assert first.outcome.elapsed < 0.5
    assert first.outcome.trace[0].elapsed < 0.2
    assert first.outcome.evaluations > 1
    assert first.notes == ()
    # A later run of the process pays nothing and is timed alike.
    later = attempt(_task(solver, stop))
    assert isinstance(later, Completion)
    assert later.outcome.trace[0].elapsed < 0.2
    assert later.notes == ()


def test_a_first_call_cost_paid_inside_solve_is_charged_to_the_first_run() -> None:
    stop = StopCriteria(time_limit=0.2)
    solver = ColdStart(seconds=0.6, key="charged")
    first = attempt(_task(solver, stop))
    assert isinstance(first, Completion)
    # Without a prepare method the cost is inside the time limit: no search is left.
    assert first.outcome.evaluations == 1
    assert first.outcome.trace[0].elapsed >= 0.6
    assert any(note.startswith("the run took") for note in first.notes)
    later = attempt(_task(solver, stop))
    assert isinstance(later, Completion)
    assert later.outcome.evaluations > 1
    assert later.notes == ()


def test_an_attempt_with_an_origin_counts_from_it_and_skips_the_preparation() -> None:
    stop = StopCriteria(time_limit=0.6)
    solver = Prepared(seconds=0.3, key="origin")
    task = _task(solver, stop)
    assert prepare_task(task) is None
    origin = time.perf_counter()
    time.sleep(0.3)
    result = attempt(task, origin=origin)
    assert isinstance(result, Completion)
    # The 0.3 s between the origin and the call are inside the wall time and the limit.
    assert 0.6 <= result.measurement.wall_seconds < 0.85
    assert result.outcome.trace[0].elapsed >= 0.3
    assert 0.6 <= result.outcome.elapsed < 0.85
    assert result.notes == ()


def test_a_preparation_that_raises_fails_the_attempt_before_its_solve() -> None:
    task = _task(Prepared(broken=True), StopCriteria(max_evaluations=5))
    workers = []
    for failure in (attempt(task), prepare_task(task, "worker-1")):
        assert isinstance(failure, Failure)
        assert failure.error.kind is ErrorKind.EXCEPTION
        assert failure.error.message == "the backend cannot be loaded"
        assert failure.category is Category.TERMINAL
        assert failure.solve_started is False
        assert failure.measurement.cpu_seconds is None
        workers.append(failure.measurement.worker)
    assert workers == ["main", "worker-1"]


def test_preparing_calls_the_hook_of_a_solver_that_has_one() -> None:
    before = time.perf_counter()
    prepare(RandomSearch())
    prepare(Prepared(seconds=0.2, key="hook"))
    assert time.perf_counter() - before >= 0.2
    again = time.perf_counter()
    prepare(Prepared(seconds=0.2, key="hook"))
    assert time.perf_counter() - again < 0.2
    assert "numpy.random" in sys.modules


@pytest.mark.parametrize(
    ("solver_id", "package"),
    [("cp-sat", "ortools"), ("restricted-neighbourhood-search", "ortools"), ("milp", "pulp")],
)
def test_the_solvers_with_a_backend_import_it_when_prepared(solver_id: str, package: str) -> None:
    solver = default_registry(plugins=False).create(solver_id)
    if find_spec(package) is None:
        with pytest.raises(MissingDependencyError, match=package):
            prepare(solver)
    else:
        prepare(solver)
        assert package in sys.modules


def test_no_built_in_solver_imports_a_module_during_its_first_solve() -> None:
    # A fresh interpreter: in this one, earlier tests have imported everything.
    done = subprocess.run(
        [sys.executable, "-c", _SWEEP],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    report = json.loads(done.stdout.strip().splitlines()[-1])
    # A module a solver imports on its first solve is charged to the first run of every
    # process: import it in the solver's module, or in the solver's prepare method.
    assert report["late"] == {}
    assert report["solved"] >= 60


def test_process_workers_prepare_a_solver_before_the_start_of_its_solve(tmp_path: Path) -> None:
    limit = 0.3
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-prepared", parameters={"seconds": 0.5, "key": "workers"}),),
        protocols=(Protocol("clock", StopCriteria(time_limit=limit)),),
        instances=ONE_PROBLEM,
        replicates=4,
        backend="process",
        workers=2,
    )
    summary = run(campaign, tmp_path, registry=REGISTRY)
    records = read_runs(summary.directory)
    assert len(records) == 4
    # Each of the two workers paid 0.5 s for its first run, before the instant
    # its hard limit, its wall time, and its solver's clock count from.
    for record in records:
        assert record.provenance.wall_seconds < limit + 0.2
        assert record.outcome.trace[0].elapsed < limit
        assert record.outcome.evaluations > 1
        assert record.notes == ()
    # The preparation is part of what the first run of a worker waited for its solve.
    waited = [record.provenance.waited_seconds or 0.0 for record in records]
    assert max(waited) >= 0.5


def test_a_preparation_that_raises_in_a_worker_fails_the_run(tmp_path: Path) -> None:
    campaign = tiny_campaign(
        solvers=(SolverSpec("test-prepared", parameters={"broken": True}),),
        instances=ONE_PROBLEM,
        replicates=1,
        backend="process",
        workers=1,
    )
    summary = run(campaign, tmp_path, registry=REGISTRY)
    assert (summary.completed, summary.failed) == (0, 1)
    (failure,) = read_failures(summary.directory)
    assert failure.error.kind is ErrorKind.EXCEPTION
    assert failure.error.type == "dispatchatlas.core.errors.SolverError"
    assert failure.solve_started is False


@dataclass(frozen=True)
class _Probe:
    """Records where a clock created inside its solve starts."""

    seen: list[float] = field(default_factory=list)
    info: ClassVar[SolverInfo] = RandomSearch.info

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        self.seen.append(clock_start())
        return RandomSearch().solve(problem, stop, seed)


def test_an_attempt_counts_from_the_instant_the_worker_states_as_its_start() -> None:
    probe = _Probe()
    problem = chain_problem()
    task = Task("r", 1, problem, problem_digest(problem), probe, StopCriteria(max_evaluations=5), 1)
    origin = time.perf_counter() - 1.0
    result = attempt(task, origin=origin)
    assert isinstance(result, Completion)
    assert probe.seen == [origin]
    assert result.measurement.wall_seconds >= 1.0
    assert result.outcome.elapsed >= 1.0


def test_a_solver_that_budgets_phases_gives_each_its_share_inside_an_attempt() -> None:
    stop = StopCriteria(time_limit=0.6)
    own = attempt(_task(Phased(own_origin=True), stop))
    assert isinstance(own, Completion)
    assert int(own.outcome.extras["first"]) > 1
    assert int(own.outcome.extras["second"]) > 1
    assert 0.6 <= own.measurement.wall_seconds < 0.9
    assert own.notes == ()
    # Without an origin of its own, the second phase counts its half from the
    # start of the attempt: that half is over when the phase begins.
    shared = attempt(_task(Phased(own_origin=False), stop))
    assert isinstance(shared, Completion)
    assert int(shared.outcome.extras["first"]) > 1
    assert int(shared.outcome.extras["second"]) == 1
    assert shared.measurement.wall_seconds < 0.5
