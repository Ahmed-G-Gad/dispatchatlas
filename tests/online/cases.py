"""Platforms and traces shared by the tests of the online loop."""

from __future__ import annotations

from dataclasses import dataclass

from dispatchatlas.benchmarks import generate
from dispatchatlas.core import Dependency, Mode, Network, Problem, Resource, Task
from dispatchatlas.online import (
    AlternatingRenewal,
    Arrival,
    Exponential,
    Failure,
    Repair,
    Trace,
    Workflow,
    availability_events,
    poisson_arrivals,
    rate_for_load,
    workflows_from_problem,
)


@dataclass(frozen=True)
class TwoNodes:
    """Two machines, n1 and n2, with no network."""

    resources: tuple[Resource, ...] = (Resource("n1"), Resource("n2"))
    network: Network | None = None


def fork() -> Workflow:
    """Task a, then b and c, on n1 or n2.

    Durations on (n1, n2): a (2, 4), b (3, 3), c (1, 5). Mode 0 runs on n1,
    mode 1 on n2. HEFT's upward ranks are a 6, b 3, c 3, so b comes before c
    when both fall due together.
    """

    def task(name: str, on_n1: float, on_n2: float) -> Task:
        return Task(name, (Mode(on_n1, {"n1": 1.0}), Mode(on_n2, {"n2": 1.0})))

    return Workflow(
        "w",
        (task("a", 2, 4), task("b", 3, 3), task("c", 1, 5)),
        (Dependency("a", "b"), Dependency("a", "c")),
    )


def fork_trace() -> Trace:
    """The fork arrives at 0; n1 fails at 3 and is repaired at 10."""
    return Trace.of([Arrival(0.0, fork()), Failure(3.0, "n1"), Repair(10.0, "n1")])


def continuum_case(
    seed: int, *, applications: int = 6, failures: bool = True, load: float = 0.7
) -> tuple[Problem, Trace]:
    """A continuum workflow problem's platform and a trace of its workflows.

    Arrivals are Poisson at ``load``; with ``failures`` every edge, fog, and
    cloud node fails and recovers until ten arrival spans, with a mean up time
    of one span and a mean outage of a fifth of it.
    """
    problem = generate("continuum-workflow", seed=seed, applications=applications, tasks=5)
    workflows = [workflow for workflow, _ in workflows_from_problem(problem)]
    rate = rate_for_load(workflows, problem.resources, load)
    trace = poisson_arrivals(workflows, rate, seed=seed)
    if failures:
        span = trace.events[-1].time
        nodes = [r.id for r in problem.resources if r.tier != "device"]
        model = AlternatingRenewal(Exponential(span), Exponential(span / 5))
        trace = trace.merged(availability_events(nodes, model, seed=seed, horizon=10 * span))
    return problem, trace
