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
from dispatchatlas.core.clock import clock_origin, clock_start
from dispatchatlas.core.solver import TracePoint
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
class LockHolder:
    """Evaluates the listed order inside a search, then keeps the interpreter lock.

    One call that never returns to the interpreter runs for about ``seconds``
    seconds, so no other thread of the worker runs in that time.
    """

    seconds: float = 20.0
    info: ClassVar[SolverInfo] = _info("test-holds-lock", stochastic=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        sample = 2_000_000
        clock = time.perf_counter()
        sum(range(sample))
        count = int(self.seconds * sample / max(time.perf_counter() - clock, 1e-6))
        search = Search(problem, stop, seed, self.info)
        search.evaluate(list(range(problem.size)))
        sum(range(count))
        return search.result()


@dataclass(frozen=True)
class Overrunner:
    """Improves once at the start and once ``after`` seconds in, then sleeps whatever its limits.

    Its search has no time limit. The first incumbent is the worse of the
    listed order and its reverse, the second one the better.
    """

    after: float = 1.0
    seconds: float = 60.0
    info: ClassVar[SolverInfo] = _info("test-overrun", stochastic=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        search = Search(problem, StopCriteria(max_evaluations=10), seed, self.info)
        listed = list(range(problem.size))
        orders = sorted((listed, listed[::-1]), key=lambda order: search.probe(order)[0])
        search.evaluate(orders[1])
        time.sleep(max(0.0, self.after - search.elapsed))
        search.evaluate(orders[0])
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


@dataclass(frozen=True)
class SlowSetup:
    """Spends ``setup`` seconds before its search starts, then samples orders within its limits."""

    setup: float = 0.4
    info: ClassVar[SolverInfo] = _info("test-slow-setup")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        time.sleep(self.setup)
        return RandomSearch().solve(problem, stop, seed)


@dataclass(frozen=True)
class Tardy:
    """Schedules the listed order, then works ``seconds`` more after its clock was read."""

    seconds: float = 0.3
    info: ClassVar[SolverInfo] = _info("test-tardy", stochastic=False, iterative=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        result = InputOrder().solve(problem, stop, seed)
        time.sleep(self.seconds)
        return result


@dataclass(frozen=True)
class Scripted:
    """Returns the schedule of the listed order with a scripted trace and elapsed time.

    The trace has one point per entry of ``times`` and ``evaluations``. Every
    point holds the value of the schedule, but the last one holds ``last``
    when that is set. ``elapsed`` replaces the elapsed time when it is set.
    """

    times: tuple[float, ...] = ()
    evaluations: tuple[int, ...] = ()
    last: float | None = None
    elapsed: float | None = None
    info: ClassVar[SolverInfo] = _info("test-scripted", stochastic=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        result = InputOrder().solve(problem, stop, seed)
        value = result.value
        assert value is not None
        points = [
            TracePoint(float(at), int(count), 0, value)
            for at, count in zip(self.times, self.evaluations, strict=True)
        ]
        if points and self.last is not None:
            points[-1] = replace(points[-1], value=self.last)
        elapsed = result.elapsed if self.elapsed is None else self.elapsed
        return replace(result, trace=tuple(points), elapsed=elapsed)


@dataclass(frozen=True)
class Timeless:
    """Returns the schedule of the listed order with a time or a value that is not a number.

    ``where`` names what is not finite: the ``elapsed`` time, an ``infinite``
    elapsed time, a ``time`` of the trace, or a ``value`` of the trace.
    """

    where: str = "elapsed"
    info: ClassVar[SolverInfo] = _info("test-timeless", stochastic=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        result = InputOrder().solve(problem, stop, seed)
        value = result.value
        assert value is not None
        nan = float("nan")
        trace = (
            TracePoint(0.0, 1, 0, nan if self.where == "value" else value),
            TracePoint(nan if self.where == "time" else 0.0, 1, 0, value),
        )
        elapsed = {"elapsed": nan, "infinite": float("inf")}.get(self.where, result.elapsed)
        return replace(result, trace=trace, elapsed=elapsed)


@dataclass(frozen=True)
class Delegating:
    """Schedules the listed order after a child process has computed for ``seconds``."""

    seconds: float = 0.3
    info: ClassVar[SolverInfo] = _info("test-delegating", stochastic=False, iterative=False)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        code = f"import time\nwhile time.process_time() < {self.seconds!r}:\n    pass\n"
        subprocess.run([sys.executable, "-c", code], check=True)
        return InputOrder().solve(problem, stop, seed)


_PAID: set[str] = set()
"""The keys whose one-off cost this process has paid."""


def _pay_once(key: str, seconds: float) -> None:
    """Sleep ``seconds`` the first time ``key`` is seen in this process, as a first import does."""
    if key not in _PAID:
        _PAID.add(key)
        time.sleep(seconds)


@dataclass(frozen=True)
class ColdStart:
    """Pays ``seconds`` once in a process, inside its first solve there, then samples orders."""

    seconds: float = 0.25
    key: str = "cold"
    info: ClassVar[SolverInfo] = _info("test-cold-start")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        _pay_once(self.key, self.seconds)
        return RandomSearch().solve(problem, stop, seed)


@dataclass(frozen=True)
class Prepared:
    """Pays ``seconds`` once in a process, in ``prepare`` when that is called first.

    With ``broken`` set, ``prepare`` raises as a solver whose backend is missing does.
    """

    seconds: float = 0.25
    key: str = "prepared"
    broken: bool = False
    info: ClassVar[SolverInfo] = _info("test-prepared")

    def prepare(self) -> None:
        if self.broken:
            raise SolverError("the backend cannot be loaded")
        _pay_once(self.key, self.seconds)

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        _pay_once(self.key, self.seconds)
        return RandomSearch().solve(problem, stop, seed)


@dataclass(frozen=True)
class Phased:
    """Runs two searches in turn, each with half of the time limit.

    With ``own_origin`` set, each search runs inside a clock origin of its
    own, so its half counts from its start; otherwise both count from the
    origin the caller set.
    """

    own_origin: bool = True
    info: ClassVar[SolverInfo] = _info("test-phased")

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        assert stop.time_limit is not None
        began = clock_start()
        half = replace(stop, time_limit=stop.time_limit / 2.0)
        results = []
        for phase in range(2):
            with clock_origin() if self.own_origin else contextlib.nullcontext():
                results.append(RandomSearch().solve(problem, half, seed + phase))
        best = min(results, key=lambda result: result.value or 0.0)
        first, second = results
        return replace(
            best,
            evaluations=first.evaluations + second.evaluations,
            elapsed=time.perf_counter() - began,
            extras={"first": first.evaluations, "second": second.evaluations},
        )


TEST_SOLVERS = (
    Delegating,
    Timeless,
    ColdStart,
    Prepared,
    Phased,
    SlowSetup,
    Tardy,
    Scripted,
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
    LockHolder,
    Overrunner,
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
