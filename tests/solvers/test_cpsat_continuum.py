"""CP-SAT as the exact baseline for workflow placement on the edge-fog-cloud continuum.

Tiny continuum instances, with nodes of capacity above one, communication that
depends on the pair of chosen nodes, releases, and several workflows, must reach
the optimum of a plain brute force over task orders and node choices. The
size of the model at 50 and 100 tasks, the single-worker default, determinism,
the reporting of a bound and a gap, and the refusals and flags for what the
model does not state are pinned here too.
"""

from __future__ import annotations

import random

import pytest

from dispatchatlas.benchmarks.continuum.families import ContinuumWorkflow
from dispatchatlas.core import (
    Dependency,
    Link,
    Mode,
    Network,
    Problem,
    Resource,
    Status,
    StopCriteria,
    Task,
    UnsupportedProblemError,
    check_schedule,
)
from dispatchatlas.solvers.exact import CpSat
from dispatchatlas.solvers.exact.common import decimal_scale, serial_horizon, time_values
from tests.solvers.test_exact import (
    GENEROUS,
    bound_of,
    brute_force,
    gap_of,
    requires_ortools,
    schedule_of,
    value_of,
)

pytestmark = [pytest.mark.exact, requires_ortools]


def mixed_node_problem(seed: int, objective: str) -> Problem:
    """Five tasks of two workflows on an edge node, a 2-core fog node, and a 3-core cloud node.

    Transfer times differ for every pair of tiers and co-location is free.
    Tasks demand one or two cores and some are released late.
    """
    rng = random.Random(seed)
    resources = (
        Resource("edge", capacity=1.0, tier="edge"),
        Resource("fog", capacity=2.0, tier="fog"),
        Resource("cloud", capacity=3.0, tier="cloud"),
    )
    speeds = {"edge": 1, "fog": 2, "cloud": 4}
    tasks = []
    for i in range(5):
        modes = []
        for node in rng.sample(("edge", "fog", "cloud"), rng.randint(2, 3)):
            cores = 1.0 if node == "edge" else float(rng.randint(1, 2))
            duration = float(rng.randint(2, 12)) / speeds[node]
            modes.append(Mode(duration, {node: cores}, location=node))
        tasks.append(
            Task(
                f"t{i}",
                tuple(modes),
                release=float(rng.choice((0, 0, 1, 3))),
                group="w0" if i < 3 else "w1",
            )
        )
    dependencies = tuple(
        Dependency(f"t{i}", f"t{j}", data=float(rng.randint(1, 6)))
        for j in range(1, 5)
        for i in range(j)
        if rng.random() < 0.4
    )
    links = (
        Link("edge", "fog", bandwidth=2.0, latency=0.5),
        Link("fog", "cloud", bandwidth=4.0, latency=1.0),
        Link("edge", "cloud", bandwidth=1.0, latency=2.0),
    )
    return Problem(
        f"mixed-{seed}",
        tasks=tuple(tasks),
        resources=resources,
        dependencies=dependencies,
        network=Network(links),
        objectives=(objective,),
    )


def small_workflows(applications: int, tasks: int) -> ContinuumWorkflow:
    return ContinuumWorkflow(
        applications=applications,
        tasks=tasks,
        devices=2,
        edge_nodes=1,
        fog_nodes=1,
        cloud_nodes=1,
        tiers="edge,fog,cloud",
        arrival_rate=0.5,
        objectives=("mean_response_time",),
    )


def large_workflows(applications: int) -> ContinuumWorkflow:
    return ContinuumWorkflow(
        applications=applications,
        tasks=8,
        devices=5,
        edge_nodes=3,
        fog_nodes=2,
        cloud_nodes=2,
        objectives=("mean_response_time",),
    )


@pytest.mark.parametrize("objective", ["makespan", "mean_response_time"])
@pytest.mark.parametrize("seed", range(6))
def test_cpsat_matches_brute_force_on_tiny_continuum_instances(seed, objective):
    problem = mixed_node_problem(seed, objective)
    expected = brute_force(problem)
    assert expected is not None
    result = CpSat().solve(problem, GENEROUS)
    assert result.status is Status.OPTIMAL
    assert value_of(result) == pytest.approx(expected, abs=1e-6)
    assert bound_of(result) == pytest.approx(expected, abs=1e-6)
    assert gap_of(result) == pytest.approx(0.0, abs=1e-9)
    assert check_schedule(problem, schedule_of(result)) == ()


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(("applications", "tasks"), [(1, 3), (2, 1)])
def test_cpsat_matches_brute_force_on_generated_workflows(seed, applications, tasks):
    problem = small_workflows(applications, tasks).generate(seed)
    expected = brute_force(problem)
    assert expected is not None
    result = CpSat().solve(problem, GENEROUS)
    assert result.status is Status.OPTIMAL
    assert value_of(result) == pytest.approx(expected, abs=1e-6)
    assert check_schedule(problem, schedule_of(result)) == ()


def test_cpsat_charges_the_transfer_only_between_different_nodes():
    # Both tasks on the unary edge node finish at 4; moving b to the cloud pays 5 first.
    problem = Problem(
        "colocation",
        tasks=(
            Task("a", (Mode(2, {"edge": 1}, location="edge"),)),
            Task(
                "b",
                (Mode(2, {"edge": 1}, location="edge"), Mode(1, {"cloud": 1}, location="cloud")),
            ),
        ),
        resources=(Resource("edge", tier="edge"), Resource("cloud", tier="cloud")),
        dependencies=(Dependency("a", "b", data=4.0),),
        network=Network((Link("edge", "cloud", bandwidth=1.0, latency=1.0),)),
    )
    result = CpSat().solve(problem, GENEROUS)
    assert value_of(result) == pytest.approx(4.0)
    assert [a.mode for a in schedule_of(result).assignments] == [0, 0]


def test_cpsat_runs_one_worker_by_default_and_is_deterministic():
    assert CpSat().threads == 1
    problem = small_workflows(2, 3).generate(3)
    first = CpSat().solve(problem, GENEROUS)
    second = CpSat().solve(problem, GENEROUS)
    assert first.status is second.status is Status.OPTIMAL
    assert schedule_of(first) == schedule_of(second)
    assert [point.value for point in first.trace] == [point.value for point in second.trace]
    assert first.extras["threads"] == 1
    assert CpSat(threads=3).solve(problem, GENEROUS).extras["threads"] == 3


@pytest.mark.parametrize("applications", [5, 10])
def test_cpsat_builds_the_model_of_50_and_100_task_workflows(applications):
    from dispatchatlas.solvers.exact import _cpsat  # noqa: PLC0415 - loads OR-Tools

    problem = large_workflows(applications).generate(1)
    assert problem.size == applications * 10
    scale = decimal_scale(time_values(problem, dues=False), "time values of the problem")
    model = _cpsat.IntervalModel(problem, scale, serial_horizon(problem, scale))
    assert model.model.validate() == ""
    assert model.transfer_pairs > 0
    assert model.objective("mean_response_time").divisor == scale * applications


def test_cpsat_reports_an_incumbent_bound_and_gap_on_a_50_task_workflow():
    problem = large_workflows(5).generate(1)
    result = CpSat().solve(problem, StopCriteria(time_limit=30.0, max_evaluations=1))
    assert result.status in (Status.FEASIBLE, Status.OPTIMAL)
    assert check_schedule(problem, schedule_of(result)) == ()
    assert bound_of(result) <= value_of(result) + 1e-9
    assert gap_of(result) >= 0.0
    assert result.extras["threads"] == 1
    assert int(result.extras["transfer_pairs"]) > 0
    assert int(result.extras["model_constraints"]) > 0
    assert int(result.extras["model_variables"]) > 0


def test_cpsat_flags_modes_without_a_location_as_free_transfers():
    def build(located: bool) -> Problem:
        # A mode that demands no resource has no location unless it names one.
        demands = {"cloud": 1} if located else {}
        return Problem(
            "unlocated",
            tasks=(
                Task("a", (Mode(1, {"edge": 1}, location="edge"),)),
                Task("b", (Mode(1, demands, location="cloud" if located else None),)),
            ),
            resources=(Resource("edge", tier="edge"), Resource("cloud", tier="cloud")),
            dependencies=(Dependency("a", "b", data=5.0),),
            network=Network((Link("edge", "cloud", bandwidth=1.0, latency=1.0),)),
        )

    located = CpSat().solve(build(True), GENEROUS)
    unlocated = CpSat().solve(build(False), GENEROUS)
    assert located.extras["unlocated_transfers"] == 0
    assert value_of(located) == pytest.approx(8.0)
    assert unlocated.extras["unlocated_transfers"] == 1
    assert value_of(unlocated) == pytest.approx(2.0)


def test_cpsat_refuses_a_transfer_time_it_cannot_state_exactly():
    problem = Problem(
        "thirds",
        tasks=(
            Task("a", (Mode(1, {"edge": 1}, location="edge"),)),
            Task("b", (Mode(1, {"cloud": 1}, location="cloud"),)),
        ),
        resources=(Resource("edge", tier="edge"), Resource("cloud", tier="cloud")),
        dependencies=(Dependency("a", "b", data=1.0),),
        network=Network((Link("edge", "cloud", bandwidth=3.0),)),
    )
    with pytest.raises(UnsupportedProblemError, match="decimals"):
        CpSat().solve(problem, GENEROUS)
