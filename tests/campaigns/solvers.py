"""Solvers with scripted behaviour, and campaign factories, for the campaign tests.

The solver classes live at module level so that spawned worker processes can
import them when they unpickle a run.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

from dispatchatlas.campaigns import (
    Campaign,
    Execution,
    FamilySource,
    InstanceSource,
    Protocol,
    SolverSpec,
)
from dispatchatlas.campaigns.config import THREAD_VARIABLES
from dispatchatlas.campaigns.processors import allowed_cpus
from dispatchatlas.core import FEATURES, Problem, SolverError, SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers import SolverRegistry, default_registry
from dispatchatlas.solvers.rules import InputOrder, RandomSearch
from dispatchatlas.solvers.search import Search


def _info(solver_id: str, **options: Any) -> SolverInfo:
    return SolverInfo(
        id=solver_id,
        name=solver_id,
        family="random-sampling",
        summary="A solver whose behaviour the campaign tests script.",
        **options,
    )


@dataclass(frozen=True)
class Sleeper:
    """Ignores its stop criteria and sleeps."""

    seconds: float = 60.0
    info: ClassVar[SolverInfo] = _info("test-sleeper", stochastic=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        time.sleep(self.seconds)
        return InputOrder().solve(problem, stop, seed)


@dataclass(frozen=True)
class Crasher:
    """Ends its process at once, as a crash in native code would."""

    info: ClassVar[SolverInfo] = _info("test-crasher", stochastic=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        os._exit(7)


@dataclass(frozen=True)
class Flaky:
    """Raises a RuntimeError on its first ``failures`` calls per problem and seed.

    The calls are counted in files under ``marker``, so the count survives
    worker processes and sessions.
    """

    marker: str = ""
    failures: int = 1
    info: ClassVar[SolverInfo] = _info("test-flaky")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        path = Path(self.marker) / f"{problem.id}-{seed}"
        calls = int(path.read_text()) if path.exists() else 0
        path.write_text(str(calls + 1))
        if calls < self.failures:
            raise RuntimeError(f"transient failure {calls + 1}")
        return RandomSearch().solve(problem, stop, seed)


@dataclass(frozen=True)
class Refuser:
    """Raises a typed DispatchAtlas error, which settles a run at once."""

    info: ClassVar[SolverInfo] = _info("test-refuser", stochastic=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        raise SolverError("this solver refuses every problem")


@dataclass(frozen=True)
class Interrupter:
    """Raises KeyboardInterrupt on its first call, then behaves like input order."""

    marker: str = ""
    info: ClassVar[SolverInfo] = _info("test-interrupter", stochastic=False, iterative=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        path = Path(self.marker)
        if not path.exists():
            path.write_text("interrupted")
            raise KeyboardInterrupt
        return InputOrder().solve(problem, stop, seed)


@dataclass(frozen=True)
class NoDeadlines:
    """Input order, declaring that it does not model hard deadlines."""

    info: ClassVar[SolverInfo] = _info(
        "test-no-deadlines",
        stochastic=False,
        iterative=False,
        supports=frozenset(FEATURES) - {"deadline"},
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return InputOrder().solve(problem, stop, seed)


@dataclass(frozen=True)
class Pareto:
    """Random search that declares itself multi-objective."""

    info: ClassVar[SolverInfo] = _info("test-pareto", multi_objective=True)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return RandomSearch().solve(problem, stop, seed)


@dataclass(frozen=True)
class EnvironmentProbe:
    """Input order that reports the thread settings and processors of its process in its extras."""

    info: ClassVar[SolverInfo] = _info("test-environment", stochastic=False, iterative=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        result = InputOrder().solve(problem, stop, seed)
        extras = {
            "omp_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "thread_variables": ",".join(os.environ.get(name, "") for name in THREAD_VARIABLES),
            "cpus": ",".join(str(index) for index in allowed_cpus()),
            "thread_cpus": ",".join(str(index) for index in _thread_cpus()),
        }
        return replace(result, extras=extras)


def _thread_cpus() -> tuple[int, ...]:
    """Return the processors that any thread of this process may run on."""
    tasks = Path("/proc/self/task")
    read = getattr(os, "sched_getaffinity", None)
    if read is None or not tasks.is_dir():
        return allowed_cpus()
    found: set[int] = set()
    for task in tasks.iterdir():
        with contextlib.suppress(OSError, ValueError):
            found.update(read(int(task.name)))
    return tuple(sorted(found))


@dataclass(frozen=True)
class Threaded:
    """Input order with a ``threads`` parameter, which it only declares."""

    threads: int = 1
    info: ClassVar[SolverInfo] = _info("test-threaded", stochastic=False, iterative=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return InputOrder().solve(problem, stop, seed)


@dataclass(frozen=True)
class FourThreads:
    """Input order whose ``threads`` parameter, which it only declares, is 4 by default."""

    threads: int = 4
    info: ClassVar[SolverInfo] = _info("test-four-threads", stochastic=False, iterative=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return InputOrder().solve(problem, stop, seed)


@dataclass(frozen=True)
class Tabled:
    """Input order with a table parameter, which it reports in its extras."""

    weights: Mapping[str, float] = field(default_factory=dict)
    info: ClassVar[SolverInfo] = _info("test-tabled", stochastic=False, iterative=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        result = InputOrder().solve(problem, stop, seed)
        text = ",".join(f"{key}={value}" for key, value in sorted(self.weights.items()))
        return replace(result, extras={"weights": text})


@dataclass(frozen=True)
class Unpicklable:
    """Input order holding a lambda, which cannot be sent to a worker process."""

    hook: Callable[[], None] = field(default=lambda: None)
    info: ClassVar[SolverInfo] = _info("test-unpicklable", stochastic=False, iterative=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return InputOrder().solve(problem, stop, seed)


@dataclass(frozen=True)
class SlowArrival:
    """Input order whose instance takes ``delay`` seconds to rebuild in a worker."""

    delay: float = 1.0
    info: ClassVar[SolverInfo] = _info("test-slow-arrival", stochastic=False, iterative=False)

    def __setstate__(self, state: dict[str, Any]) -> None:
        time.sleep(state["delay"])
        self.__dict__.update(state)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        return InputOrder().solve(problem, stop, seed)


@dataclass(frozen=True)
class Stalled:
    """Evaluates the listed order inside a search, then sleeps whatever its limits.

    Its instance takes ``delay`` seconds to rebuild in a worker.
    """

    seconds: float = 60.0
    delay: float = 0.0
    info: ClassVar[SolverInfo] = _info("test-stalled", stochastic=False)

    def __setstate__(self, state: dict[str, Any]) -> None:
        time.sleep(state["delay"])
        self.__dict__.update(state)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        search = Search(problem, stop, seed, self.info)
        search.evaluate(list(range(problem.size)))
        time.sleep(self.seconds)
        return search.result()


@dataclass(frozen=True)
class Nested:
    """Starts a child process, writes its own pid and the child's to ``marker``, and sleeps."""

    marker: str = ""
    seconds: float = 60.0
    info: ClassVar[SolverInfo] = _info("test-nested", stochastic=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        # The interpreter itself, not the launcher a virtual environment may put before it.
        python = getattr(sys, "_base_executable", None) or sys.executable
        child = subprocess.Popen([python, "-c", f"import time; time.sleep({self.seconds!r})"])
        marker = Path(self.marker)
        partial = marker.with_name(marker.name + ".partial")
        partial.write_text(f"{os.getpid()} {child.pid}", encoding="ascii")
        partial.replace(marker)
        time.sleep(self.seconds)
        return InputOrder().solve(problem, stop, seed)


TEST_SOLVERS = (
    Sleeper,
    Crasher,
    Flaky,
    Refuser,
    Interrupter,
    NoDeadlines,
    Pareto,
    EnvironmentProbe,
    Threaded,
    FourThreads,
    Tabled,
    Unpicklable,
    SlowArrival,
    Stalled,
    Nested,
)


def registry() -> SolverRegistry:
    """Return the built-in solvers plus the scripted ones."""
    solvers = default_registry(plugins=False)
    for factory in TEST_SOLVERS:
        solvers.register(factory)
    return solvers


def tiny_campaign(
    campaign_id: str = "tiny",
    *,
    solvers: tuple[SolverSpec, ...] = (
        SolverSpec("input-order"),
        SolverSpec("random-search"),
        SolverSpec("hill-climbing", "hc", {"restart_after": 5}),
    ),
    protocols: tuple[Protocol, ...] = (Protocol("evals", StopCriteria(max_evaluations=15)),),
    instances: tuple[InstanceSource, ...] = (
        FamilySource("flow-shop", (1, 2), {"jobs": 3, "machines": 2}),
    ),
    replicates: int = 2,
    seed: int = 11,
    **execution: Any,
) -> Campaign:
    """Return a campaign that finishes in a fraction of a second."""
    settings = {"backend": "sequential", **execution}
    return Campaign(
        id=campaign_id,
        seed=seed,
        instances=instances,
        solvers=solvers,
        protocols=protocols,
        replicates=replicates,
        execution=Execution(**settings),
    )
