"""Tests of the edge-fog-cloud continuum model and its three families."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from statistics import fmean
from typing import Any

import pytest

from dispatchatlas.benchmarks import default_registry
from dispatchatlas.benchmarks.characterize import earliest_finish_times
from dispatchatlas.benchmarks.continuum import (
    DEFAULT_LINKS,
    DEFAULT_NODES,
    TIERS,
    Application,
    NodeSpec,
    Platform,
    build_problem,
    fork_join_graph,
    response_bound,
    three_tier_platform,
)
from dispatchatlas.core import (
    ConfigError,
    Dependency,
    Link,
    Mode,
    Problem,
    Status,
    StopCriteria,
    Task,
    check_schedule,
    evaluate,
    problem_digest,
    problem_from_dict,
    problem_to_dict,
)
from dispatchatlas.references import bibliography
from dispatchatlas.solvers import default_registry as solver_registry
from dispatchatlas.solvers.evaluation import Evaluator
from dispatchatlas.solvers.scheduling import Decoder

REGISTRY = default_registry(plugins=False)
FAMILIES = ("continuum-offloading", "continuum-workflow", "continuum-pipeline")
SMALL = {
    "continuum-offloading": {"applications": 8, "devices": 2},
    "continuum-workflow": {"applications": 3, "tasks": 6, "devices": 2},
    "continuum-pipeline": {"streams": 2, "frames": 3, "stages": 3, "devices": 2},
}


def _small(family: str, seed: int = 3, **overrides: Any) -> Problem:
    return REGISTRY.generate(family, seed, **{**SMALL[family], **overrides})


def _compute_tasks(problem: Problem) -> list[Task]:
    return [task for task in problem.tasks if len(task.modes) > 1]


def _location(mode: Mode) -> str:
    assert mode.location is not None
    return mode.location


def _tier(problem: Problem, mode: Mode) -> str | None:
    return problem.resource(_location(mode)).tier


def _link(platform: Platform, a: str, b: str) -> Link:
    link = platform.network.link(a, b)
    assert link is not None
    return link


def _modes_on(problem: Problem, tier: str) -> list[int]:
    """Choose for every task its first mode on ``tier``, else its first mode."""
    modes = []
    for task in problem.tasks:
        on_tier = [k for k, mode in enumerate(task.modes) if _tier(problem, mode) == tier]
        modes.append(on_tier[0] if on_tier else 0)
    return modes


# ------------------------------------------------------------------ platform


def test_platform_has_the_requested_nodes_per_tier():
    platform = three_tier_platform(devices=3, edge=2, fog=1, cloud=2)
    assert platform.tiers == TIERS
    assert [node.id for node in platform.nodes("edge")] == ["edge-0", "edge-1"]
    for tier in TIERS:
        spec = DEFAULT_NODES[tier]
        for node in platform.nodes(tier):
            assert node.tier == tier
            assert node.capacity == spec.cores
            assert (node.idle_power, node.busy_power, node.cost_rate) == (
                spec.idle_power,
                spec.busy_power,
                spec.cost_rate,
            )
            assert platform.speed(node.id) == spec.speed
    assert {node.kind for node in platform.nodes("device")} == {"device"}
    assert len(platform.resources) == 8


def test_default_links_join_every_pair_of_tiers_along_the_chain():
    pairs = {tuple(sorted((link.a, link.b))) for link in DEFAULT_LINKS}
    assert len(pairs) == len(DEFAULT_LINKS) == 10
    platform = three_tier_platform()
    access = _link(platform, "device", "edge")
    metro = _link(platform, "edge", "fog")
    wide = _link(platform, "fog", "cloud")
    through = _link(platform, "cloud", "device")
    assert through.bandwidth == min(access.bandwidth, metro.bandwidth, wide.bandwidth)
    assert through.latency == pytest.approx(access.latency + metro.latency + wide.latency)
    assert through.energy_per_unit == pytest.approx(
        access.energy_per_unit + metro.energy_per_unit + wide.energy_per_unit
    )
    assert through.cost_per_unit == pytest.approx(wide.cost_per_unit)


def test_platform_hardware_and_links_can_be_replaced():
    platform = three_tier_platform(
        cloud=2,
        nodes={"cloud": NodeSpec(cores=16, speed=6.0, idle_power=1.0, busy_power=2.0)},
        links=[Link("fog", "cloud", bandwidth=100.0, latency=0.001)],
    )
    assert platform.speed("cloud-1") == 6.0
    assert platform.resource("cloud-0").capacity == 16.0
    assert _link(platform, "cloud", "fog").bandwidth == 100.0
    assert platform.transfer_time(50.0, "fog-0", "cloud-1") == pytest.approx(0.501)
    assert platform.transfer_time(50.0, "cloud-0", "cloud-0") == 0.0


def test_platform_transfer_times_match_the_core_model():
    platform = three_tier_platform()
    problem = build_problem(
        "probe",
        platform,
        [Application("a", "device-0", 0.0, (1.0,), upload=(2.0,), download=(0.5,))],
    )
    dependency = Dependency("a-source", "a-t0", data=2.0)
    for target in ("device-0", "edge-1", "fog-0", "cloud-0"):
        assert platform.transfer_time(2.0, "device-0", target) == problem.transfer_time(
            dependency, "device-0", target
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"devices": -1},
        {"nodes": {"moon": NodeSpec()}},
        {"links": [Link("device", "moon", bandwidth=1.0)]},
    ],
)
def test_platform_rejects_invalid_arguments(arguments):
    with pytest.raises(ConfigError):
        three_tier_platform(**arguments)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"cores": 0}, "cores"),
        ({"speed": 0.0}, "speed"),
        ({"idle_power": 5.0, "busy_power": 1.0}, "busy_power"),
        ({"cost_rate": -1.0}, "cost_rate"),
    ],
)
def test_node_spec_rejects_invalid_hardware(arguments, message):
    with pytest.raises(ValueError, match=message):
        NodeSpec(**arguments)


# -------------------------------------------------------- building problems


def test_build_problem_makes_placement_a_decision_between_pinned_endpoints():
    platform = three_tier_platform(devices=2, edge=2, fog=1, cloud=1)
    request = Application("r", "device-1", 2.5, (3.0,), upload=(1.0,), download=(0.2,))
    problem = build_problem("one", platform, [request], due_factor=2.0)
    source, compute, sink = problem.tasks
    for pinned in (source, sink):
        assert len(pinned.modes) == 1
        assert pinned.modes[0].duration == 0.0
        assert pinned.modes[0].location == "device-1"
        assert not pinned.modes[0].demands
    assert [mode.location for mode in compute.modes] == [
        "device-1",
        "edge-0",
        "edge-1",
        "fog-0",
        "cloud-0",
    ]
    for mode in compute.modes:
        assert mode.duration == platform.duration(3.0, _location(mode))
        assert dict(mode.demands) == {mode.location: 1.0}
    assert {task.release for task in problem.tasks} == {2.5}
    assert {task.group for task in problem.tasks} == {"r"}
    assert [(d.before, d.after, d.data) for d in problem.dependencies] == [
        ("r-source", "r-t0", 1.0),
        ("r-t0", "r-sink", 0.2),
    ]
    assert sink.due == pytest.approx(2.5 + 2.0 * response_bound(platform, request))


def test_response_bound_of_a_request_is_its_best_placement():
    platform = three_tier_platform(devices=1)
    request = Application("r", "device-0", 0.0, (1.7,), upload=(3.0,), download=(0.4,))
    best = min(
        platform.transfer_time(3.0, "device-0", node.id)
        + 1.7 / platform.speed(node.id)
        + platform.transfer_time(0.4, node.id, "device-0")
        for node in (platform.resource("device-0"), *platform.resources[1:])
    )
    assert response_bound(platform, request) == pytest.approx(best)


def _placement_response(
    platform: Platform, application: Application, placement: Sequence[str]
) -> float:
    """Contention-free response time of one placement, by longest path."""
    finish: list[float] = []
    device = application.device
    entries, exits = set(application.entries()), set(application.exits())
    for k, work in enumerate(application.work):
        node = placement[k]
        ready = platform.transfer_time(application.upload[k], device, node) if k in entries else 0.0
        for before, after, data in application.edges:
            if after == k:
                arrival = finish[before] + platform.transfer_time(data, placement[before], node)
                ready = max(ready, arrival)
        finish.append(ready + work / platform.speed(node))
    return max(
        finish[k] + platform.transfer_time(application.download[k], placement[k], device)
        for k in exits
    )


def test_response_bound_is_exact_for_chains():
    platform = three_tier_platform(devices=1, edge=1, fog=1, cloud=1)
    chain = Application(
        "c",
        "device-0",
        0.0,
        (0.6, 1.4, 0.9),
        edges=((0, 1, 2.0), (1, 2, 0.5)),
        upload=(6.0, 0.0, 0.0),
        download=(0.0, 0.0, 0.1),
    )
    nodes = [node.id for node in platform.resources]
    best = min(
        _placement_response(platform, chain, placement)
        for placement in itertools.product(nodes, repeat=3)
    )
    assert response_bound(platform, chain) == pytest.approx(best)


def test_response_bound_is_a_lower_bound_for_workflows():
    platform = three_tier_platform(devices=1, edge=1, fog=1, cloud=1)
    diamond = Application(
        "d",
        "device-0",
        0.0,
        (1.0, 2.0, 0.5, 1.0),
        edges=((0, 1, 4.0), (0, 2, 1.0), (1, 3, 0.2), (2, 3, 3.0)),
        upload=(2.0, 0.0, 0.0, 0.0),
        download=(0.0, 0.0, 0.0, 0.3),
    )
    nodes = [node.id for node in platform.resources]
    best = min(
        _placement_response(platform, diamond, placement)
        for placement in itertools.product(nodes, repeat=4)
    )
    assert response_bound(platform, diamond) <= best + 1e-12


def test_response_bound_matches_the_generic_earliest_finish():
    problem = _small("continuum-workflow", seed=11)
    finishes = earliest_finish_times(problem)
    for i, task in enumerate(problem.tasks):
        if task.id.endswith("-sink"):
            assert task.due is not None
            assert task.due - task.release == pytest.approx(2.0 * (finishes[i] - task.release))


@pytest.mark.parametrize(
    "arguments",
    [
        {"work": ()},
        {"edges": ((1, 0, 1.0),)},
        {"edges": ((0, 1, 1.0), (0, 1, 2.0))},
        {"upload": (1.0,)},
        {"work": (1.0, -2.0)},
        {"tiers": (("edge",),)},
        {"name": " padded "},
    ],
)
def test_application_rejects_inconsistent_input(arguments):
    base = {"name": "a", "device": "device-0", "release": 0.0, "work": (1.0, 2.0)}
    with pytest.raises(ConfigError):
        Application(**{**base, **arguments})


def test_build_problem_rejects_bad_factors_and_duplicate_names():
    platform = three_tier_platform()
    request = Application("a", "device-0", 0.0, (1.0,))
    with pytest.raises(ConfigError):
        build_problem("p", platform, [request], due_factor=0.5)
    with pytest.raises(ConfigError):
        build_problem("p", platform, [request, request])
    with pytest.raises(ConfigError):
        build_problem("p", platform, [Application("b", "device-9", 0.0, (1.0,))])
    lonely = three_tier_platform(fog=0)
    with pytest.raises(ConfigError):
        build_problem("p", lonely, [Application("c", "device-0", 0.0, (1.0,), tiers=(("fog",),))])


def test_transfers_appear_in_decoded_schedules():
    platform = three_tier_platform()
    request = Application("r", "device-0", 1.0, (2.0,), upload=(1.5,), download=(0.1,))
    problem = build_problem("one", platform, [request])
    cloud = [mode.location for mode in problem.tasks[1].modes].index("cloud-0")
    decoded = Decoder(problem).decode([0, 1, 2], [0, cloud, 0])
    link = _link(platform, "device", "cloud")
    upload = link.latency + 1.5 / link.bandwidth
    download = link.latency + 0.1 / link.bandwidth
    work = 2.0 / platform.speed("cloud-0")
    assert decoded.starts[1] == pytest.approx(1.0 + upload)
    assert decoded.ends[1] == pytest.approx(1.0 + upload + work)
    assert decoded.starts[2] == pytest.approx(1.0 + upload + work + download)
    schedule = decoded.schedule(problem)
    assert check_schedule(problem, schedule) == ()

    cloud_node = DEFAULT_NODES["cloud"]
    values = evaluate(problem, schedule, ["mean_response_time", "energy", "cost"])
    makespan = decoded.makespan
    idle = sum(resource.idle_power for resource in problem.resources) * makespan
    busy = (cloud_node.busy_power - cloud_node.idle_power) / cloud_node.cores * work
    assert values["mean_response_time"] == pytest.approx(makespan - 1.0)
    assert values["energy"] == pytest.approx(idle + busy + link.energy_per_unit * 1.6)
    assert values["cost"] == pytest.approx(
        cloud_node.cost_rate / cloud_node.cores * work + link.cost_per_unit * 1.6
    )


# ----------------------------------------------------------------- families


@pytest.mark.parametrize("family", FAMILIES)
def test_family_metadata_is_complete(family):
    info = REGISTRY.info(family)
    assert info.problem_class == "continuum"
    assert info.summary.endswith(".")
    assert info.references
    assert all(reference in bibliography() for reference in info.references)
    assert tuple(info.parameters["objectives"]) == info.objectives


@pytest.mark.parametrize("family", FAMILIES)
def test_generation_is_deterministic(family):
    assert _small(family, seed=5) == _small(family, seed=5)
    assert _small(family, seed=5) != _small(family, seed=6)


GOLDEN = {
    "continuum-offloading": "6e5597be4d620a44265c9e9510ab6bd55d1859c31e50f91d3c86d65bf9c453b8",
    "continuum-workflow": "206621afd960c632e51657c969a8bd6f0256da610b79a2cd09f62a556761ce39",
    "continuum-pipeline": "132ecf53a32b9c880a5be7325c2a8a3231f9723b67b6d53803480bb6681f6a02",
}


@pytest.mark.parametrize("family", FAMILIES)
def test_default_instances_keep_their_identity(family):
    # A generator change changes its instances: update a digest only on purpose.
    assert problem_digest(REGISTRY.generate(family, 1)) == GOLDEN[family]


@pytest.mark.parametrize("family", FAMILIES)
def test_problems_round_trip_through_json(family):
    problem = _small(family, deadline_factor=3.0)
    assert problem_from_dict(problem_to_dict(problem)) == problem


@pytest.mark.parametrize("family", FAMILIES)
def test_metadata_records_parameters_and_seed(family):
    problem = _small(family, seed=9)
    assert problem.family == family
    assert problem.id.startswith(f"{family}-")
    assert problem.id.endswith("-9")
    assert problem.metadata["seed"] == 9
    for name, value in SMALL[family].items():
        assert problem.metadata[name] == value
    assert "deadline_factor" not in problem.metadata


@pytest.mark.parametrize("family", FAMILIES)
def test_placement_is_a_decision_across_tiers(family):
    problem = _small(family)
    for task in problem.tasks:
        tiers = {_tier(problem, mode) for mode in task.modes}
        if task.id.endswith(("-source", "-sink")):
            assert tiers == {"device"}
            assert task.modes[0].duration == 0.0
        else:
            assert tiers == set(TIERS)
            locations = [mode.location for mode in task.modes]
            assert len(set(locations)) == len(locations) >= 4


@pytest.mark.parametrize("family", FAMILIES)
def test_applications_are_groups_that_end_at_their_sink(family):
    problem = _small(family)
    decoded = Decoder(problem).decode(list(range(problem.size)))
    ends = {}
    for name, members in problem.groups.items():
        ids = [problem.tasks[i].id for i in members]
        assert ids[0] == f"{name}-source"
        assert ids[-1] == f"{name}-sink"
        sink = members[-1]
        assert decoded.ends[sink] == max(decoded.ends[i] for i in members)
        ends[name] = decoded.ends[sink] - problem.tasks[sink].release
    values = evaluate(problem, decoded.schedule(problem), ["mean_response_time"])
    assert values["mean_response_time"] == pytest.approx(fmean(ends.values()))


@pytest.mark.parametrize("family", FAMILIES)
def test_decoded_schedules_are_feasible(family):
    problem = _small(family)
    decoder = Decoder(problem)
    rng = random.Random(1)
    for _ in range(8):
        order = list(range(problem.size))
        rng.shuffle(order)
        modes = [rng.randrange(len(task.modes)) for task in problem.tasks]
        assert check_schedule(problem, decoder.schedule(order, modes)) == ()


@pytest.mark.parametrize("family", FAMILIES)
def test_objectives_are_computed_consistently(family):
    problem = _small(family).with_objectives(
        "makespan", "mean_response_time", "total_tardiness", "energy", "cost"
    )
    evaluator = Evaluator(problem)
    decoded = evaluator.decoder.decode(list(range(problem.size)))
    fast = evaluator.values(decoded)
    reference = evaluate(problem, decoded.schedule(problem))
    assert fast == pytest.approx(reference)
    assert all(math.isfinite(value) and value >= 0.0 for value in reference.values())
    assert reference["energy"] > 0.0
    assert reference["cost"] > 0.0


@pytest.mark.parametrize("family", FAMILIES)
def test_placement_changes_every_objective(family):
    problem = _small(family)
    evaluator = Evaluator(problem)
    order = list(range(problem.size))
    by_tier = {
        tier: evaluator.values(evaluator.decoder.decode(order, _modes_on(problem, tier)))
        for tier in ("device", "edge", "cloud")
    }
    for objective in problem.objectives:
        assert len({round(values[objective], 9) for values in by_tier.values()}) == 3
    assert by_tier["device"]["cost"] == 0.0


@pytest.mark.parametrize("family", FAMILIES)
def test_every_time_value_is_a_whole_number_of_microseconds(family):
    problem = _small(family, deadline_factor=2.5, due_factor=1.5)
    values = [task.release for task in problem.tasks]
    values += [task.due for task in problem.tasks if task.due is not None]
    values += [task.deadline for task in problem.tasks if task.deadline is not None]
    values += [mode.duration for task in problem.tasks for mode in task.modes]
    for dependency in problem.dependencies:
        before, after = problem.task(dependency.before), problem.task(dependency.after)
        values += [
            problem.transfer_time(dependency, a.location, b.location)
            for a in before.modes
            for b in after.modes
        ]
    for value in values:
        assert value * 1e6 == pytest.approx(round(value * 1e6), rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("family", FAMILIES)
def test_hard_deadlines_are_met_by_list_scheduling(family):
    for seed in range(3):
        problem = _small(family, seed=seed, deadline_factor=3.0)
        assert all(task.deadline is not None for task in problem.tasks if task.id.endswith("sink"))
        evaluator = Evaluator(problem)
        order = sorted(range(problem.size), key=lambda i: (problem.tasks[i].release, i))
        score, _ = evaluator.evaluate(order)
        assert score.feasible


TINY = {
    "continuum-offloading": {"applications": 5, "devices": 2},
    "continuum-workflow": {"applications": 2, "tasks": 4, "devices": 2},
    "continuum-pipeline": {"streams": 2, "frames": 2, "stages": 2, "devices": 2},
}


@pytest.mark.exact
@pytest.mark.parametrize("family", FAMILIES)
def test_an_exact_optimum_bounds_the_heuristics(family):
    pytest.importorskip("ortools")
    solvers = solver_registry(plugins=False)
    if "cp-sat" not in solvers:
        pytest.skip("the cp-sat solver is not registered")
    problem = REGISTRY.generate(family, 1, **TINY[family])
    exact = solvers.create("cp-sat").solve(problem, StopCriteria(time_limit=30.0), seed=1)
    assert exact.status == Status.OPTIMAL
    assert exact.schedule is not None
    assert check_schedule(problem, exact.schedule) == ()
    assert exact.value is not None
    for solver in ("heft", "random-search"):
        if solver in solvers:
            result = solvers.create(solver).solve(problem, StopCriteria(max_evaluations=200), 1)
            assert result.value is not None
            assert result.value >= exact.value - 1e-6


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "parameters",
    [
        {"devices": 0},
        {"edge_nodes": -1},
        {"tiers": "device,moon"},
        {"tiers": ""},
        {"tiers": "fog", "fog_nodes": 0},
        {"due_factor": 0.5},
        {"deadline_factor": 0.9},
        {"objectives": ("makespan", "makespan")},
        {"objectives": ("no-such-objective",)},
    ],
)
def test_invalid_parameters_are_refused(family, parameters):
    with pytest.raises(ConfigError):
        REGISTRY.create(family, **parameters)


@pytest.mark.parametrize("family", FAMILIES)
def test_negative_seeds_are_refused(family):
    with pytest.raises(ValueError, match="seed"):
        REGISTRY.create(family).generate(-1)


# ------------------------------------------------------ offloading requests


def test_offloading_releases_follow_a_poisson_process():
    problem = REGISTRY.generate("continuum-offloading", 2, applications=400, arrival_rate=5.0)
    releases = [problem.tasks[members[0]].release for members in problem.groups.values()]
    gaps = [b - a for a, b in zip([0.0, *releases], releases, strict=False)]
    assert all(gap >= 0.0 for gap in gaps)
    assert fmean(gaps) == pytest.approx(1 / 5.0, rel=0.15)


def test_offloading_devices_take_turns_and_tiers_can_be_restricted():
    problem = REGISTRY.generate(
        "continuum-offloading", 4, applications=6, devices=3, tiers="edge,cloud"
    )
    sources = [task for task in problem.tasks if task.id.endswith("-source")]
    assert [task.modes[0].location for task in sources] == [f"device-{i % 3}" for i in range(6)]
    for task in _compute_tasks(problem):
        assert {_tier(problem, mode) for mode in task.modes} == {"edge", "cloud"}
    local = REGISTRY.generate("continuum-offloading", 4, applications=3, tiers="device")
    assert all(len(task.modes) == 1 for task in local.tasks)


# --------------------------------------------------------------- workflows


@pytest.mark.parametrize("structure", ["layered", "fork-join"])
@pytest.mark.parametrize("tasks", [10, 30, 100])
def test_workflow_graphs_stay_sparse(structure, tasks):
    problem = REGISTRY.generate(
        "continuum-workflow", 7, applications=2, tasks=tasks, structure=structure
    )
    for name in problem.groups:
        internal = [
            d
            for d in problem.dependencies
            if d.before.startswith(f"{name}-t") and d.after.startswith(f"{name}-t")
        ]
        assert len(internal) <= 2 * 2 * tasks
        density = len(internal) / (tasks * (tasks - 1) / 2)
        assert density <= {10: 0.6, 30: 0.2, 100: 0.06}[tasks]


def test_fork_join_graph_forks_and_joins():
    graph = fork_join_graph(5)
    assert graph.edges == ((0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4))
    assert graph.levels == (0, 1, 1, 1, 2)
    assert fork_join_graph(2).edges == ((0, 1),)
    assert fork_join_graph(1).edges == ()


def test_workflow_uploads_to_entry_tasks_and_returns_exit_results():
    problem = REGISTRY.generate("continuum-workflow", 3, applications=1, tasks=6)
    internal = [
        d for d in problem.dependencies if not d.before.endswith("-source") and "-t" in d.after
    ]
    has_predecessor = {d.after for d in internal}
    has_successor = {d.before for d in internal}
    uploads = {d.after for d in problem.dependencies if d.before.endswith("-source")}
    results = {d.before for d in problem.dependencies if d.after.endswith("-sink")}
    compute = {task.id for task in _compute_tasks(problem)}
    assert uploads == compute - has_predecessor
    assert results == compute - has_successor
    assert all(dependency.data > 0.0 for dependency in problem.dependencies)


def test_pinned_tasks_run_on_edge_nodes_only():
    problem = REGISTRY.generate(
        "continuum-workflow", 5, applications=3, tasks=8, pinned_fraction=0.5
    )
    for name in problem.groups:
        tiers = [
            {_tier(problem, mode) for mode in task.modes}
            for task in problem.tasks
            if task.group == name and "-t" in task.id
        ]
        assert sum(1 for item in tiers if item == {"edge"}) == 4
        assert sum(1 for item in tiers if item == set(TIERS)) == 4


def test_workflow_data_scale_with_the_ratio():
    low = REGISTRY.generate("continuum-workflow", 8, ccr=0.5)
    high = REGISTRY.generate("continuum-workflow", 8, ccr=2.0)
    for a, b in zip(low.dependencies, high.dependencies, strict=True):
        assert b.data == pytest.approx(4.0 * a.data, abs=3e-3)


def test_workflows_arrive_together_or_as_a_poisson_process():
    batch = REGISTRY.generate("continuum-workflow", 1)
    assert {task.release for task in batch.tasks} == {0.0}
    stream = REGISTRY.generate("continuum-workflow", 1, arrival_rate=2.0)
    releases = sorted({task.release for task in stream.tasks})
    assert len(releases) == 4
    assert releases[0] > 0.0


# ---------------------------------------------------------------- pipelines


def test_pipeline_frames_repeat_every_period():
    problem = REGISTRY.generate("continuum-pipeline", 12, streams=3, frames=4, period=0.8)
    by_stream: dict[str, list[tuple[float, list[float], list[float]]]] = {}
    for name, members in problem.groups.items():
        tasks = [problem.tasks[i] for i in members]
        durations = [mode.duration for task in tasks for mode in task.modes]
        names = {t.id for t in tasks}
        data = [d.data for d in problem.dependencies if d.before in names]
        by_stream.setdefault(name.split("f")[0], []).append((tasks[0].release, durations, data))
    for frames in by_stream.values():
        releases = [release for release, _, _ in frames]
        assert [b - a for a, b in itertools.pairwise(sorted(releases))] == pytest.approx(
            [0.8] * 3, abs=1e-5
        )
        assert all(item[1:] == frames[0][1:] for item in frames)


def test_pipeline_data_shrink_along_the_chain():
    problem = REGISTRY.generate("continuum-pipeline", 2, stages=4, reduction=0.25)
    name = next(iter(problem.groups))
    chain = [d.data for d in problem.dependencies if d.before.startswith(f"{name}-")]
    assert len(chain) == 5
    for first, second in itertools.pairwise(chain):
        assert second == pytest.approx(first * 0.25, abs=6e-4)


def test_pipeline_frames_are_listed_by_release():
    problem = REGISTRY.generate("continuum-pipeline", 4)
    releases = [problem.tasks[members[0]].release for members in problem.groups.values()]
    assert releases == sorted(releases)
