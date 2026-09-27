from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import pytest

from dispatchatlas.benchmarks import default_registry
from dispatchatlas.benchmarks.project import MultiModeRcpsp, Rcpsp, _network, project_problem
from dispatchatlas.core import ConfigError, ModelError, check_schedule
from dispatchatlas.core.model import Problem
from dispatchatlas.core.schedule import Assignment, Schedule

REGISTRY = default_registry(plugins=False)


def _edges(problem: Problem) -> list[tuple[int, int]]:
    index = problem.task_index
    return [(index[d.before], index[d.after]) for d in problem.dependencies]


def _reaches(count: int, edges: list[tuple[int, int]], skip: tuple[int, int]) -> bool:
    """Return whether ``skip[1]`` is reachable from ``skip[0]`` without the arc ``skip``."""
    successors = defaultdict(list)
    for edge in edges:
        if edge != skip:
            successors[edge[0]].append(edge[1])
    stack, seen = [skip[0]], set()
    while stack:
        node = stack.pop()
        for following in successors[node]:
            if following == skip[1]:
                return True
            if following not in seen:
                seen.add(following)
                stack.append(following)
    return False


def _earliest_start_schedule(problem: Problem) -> Schedule:
    """Schedule every task in its first mode as early as precedence allows."""
    ends: dict[str, float] = {}
    assignments = []
    for i in problem.topological_order:
        task = problem.tasks[i]
        start = max((ends[d.before] for d in problem.incoming[i]), default=0.0)
        ends[task.id] = start + task.modes[0].duration
        assignments.append(Assignment(task.id, 0, start, ends[task.id]))
    return Schedule.of(assignments)


@pytest.mark.parametrize(("complexity", "arcs"), [(1.5, 42), (1.8, 52), (2.1, 62)])
@pytest.mark.parametrize("seed", range(4))
def test_networks_have_psplibs_arc_counts(complexity, arcs, seed):
    problem = Rcpsp(network_complexity=complexity).generate(seed)
    assert len(problem.dependencies) == arcs


@pytest.mark.parametrize("seed", range(8))
def test_networks_keep_progens_structure(seed):
    problem = Rcpsp(activities=40, network_complexity=1.8).generate(seed)
    edges = _edges(problem)
    assert all(before < after for before, after in edges)
    predecessors = Counter(after for _, after in edges)
    successors = Counter(before for before, _ in edges)
    assert [i for i in range(40) if not predecessors[i]] == [0, 1, 2]
    assert [i for i in range(40) if not successors[i]] == [37, 38, 39]
    assert max(predecessors.values()) <= 3
    assert max(successors.values()) <= 3
    for edge in edges:
        assert not _reaches(40, edges, edge), f"arc {edge} is redundant"


def test_resource_factor_quarter_requests_one_resource_per_activity():
    problem = Rcpsp(resource_factor=0.25).generate(3)
    assert all(len(task.modes[0].demands) == 1 for task in problem.tasks)


@pytest.mark.parametrize("factor", [0.5, 0.75, 1.0])
def test_requests_meet_the_resource_factor(factor):
    problem = Rcpsp(resource_factor=factor).generate(5)
    requests = [len(task.modes[0].demands) for task in problem.tasks]
    assert sum(requests) == math.ceil(factor * 30 * 4)
    assert min(requests) >= 1


def test_requests_follow_progens_two_steps():
    problem = Rcpsp(activities=6, resources=3, resource_factor=0.5).generate(2)
    rng = random.Random(2)
    _network(rng, Rcpsp(activities=6, resources=3, resource_factor=0.5))
    durations = [rng.randint(1, 10) for _ in range(6)]
    chosen = [{rng.randrange(3)} for _ in range(6)]
    free = [(a, r) for a in range(6) for r in range(3) if r not in chosen[a]]
    for a, r in rng.sample(free, 9 - 6):
        chosen[a].add(r)
    for task, duration, requested in zip(problem.tasks, durations, chosen, strict=True):
        assert task.modes[0].duration == duration
        assert set(task.modes[0].demands) == {f"r{r + 1}" for r in requested}


def test_zero_resource_factor_requests_nothing():
    problem = Rcpsp(activities=8, resource_factor=0.0).generate(0)
    assert all(not task.modes[0].demands for task in problem.tasks)
    assert all(resource.capacity == 1.0 for resource in problem.resources)


@pytest.mark.parametrize("strength", [0.0, 0.2, 0.5, 1.0])
def test_capacities_follow_the_resource_strength(strength):
    problem = Rcpsp(resource_strength=strength).generate(7)
    schedule = _earliest_start_schedule(problem)
    for resource in problem.resources:
        k_min = max(task.modes[0].demands.get(resource.id, 0.0) for task in problem.tasks)
        times = sorted({item.start for item in schedule.assignments})
        k_max = max(
            sum(
                problem.task(item.task).modes[0].demands.get(resource.id, 0.0)
                for item in schedule.assignments
                if item.start <= time < item.end
            )
            for time in times
        )
        expected = k_min + math.floor(strength * (k_max - k_min) + 0.5)
        assert resource.capacity == max(expected, 1.0)
        assert resource.kind == "renewable"


def test_full_strength_makes_the_earliest_start_schedule_feasible():
    problem = Rcpsp(resource_strength=1.0).generate(4)
    assert check_schedule(problem, _earliest_start_schedule(problem)) == ()
    scarce = Rcpsp(resource_strength=0.0).generate(4)
    codes = {
        violation.code for violation in check_schedule(scarce, _earliest_start_schedule(scarce))
    }
    assert codes == {"capacity"}


def test_ranges_and_ids():
    problem = Rcpsp(
        activities=12, min_duration=3, max_duration=5, min_demand=2, max_demand=4
    ).generate(1)
    assert [task.id for task in problem.tasks] == [f"a{i:02d}" for i in range(1, 13)]
    assert [resource.id for resource in problem.resources] == ["r1", "r2", "r3", "r4"]
    assert all(3 <= task.modes[0].duration <= 5 for task in problem.tasks)
    assert all(
        2 <= amount <= 4 for task in problem.tasks for amount in task.modes[0].demands.values()
    )
    assert problem.id == "rcpsp-12x4-nc1.5-rf0.5-rs0.5-dmin3-dmax5-qmin2-qmax4-1"


@pytest.mark.parametrize("seed", range(6))
def test_multi_mode_activities_trade_duration_against_demand(seed):
    problem = MultiModeRcpsp(activities=15, resources=3, resource_factor=0.6).generate(seed)
    capacities = {resource.id: resource.capacity for resource in problem.resources}
    for task in problem.tasks:
        modes = task.modes
        assert 2 <= len(modes) <= 3
        durations = [mode.duration for mode in modes]
        assert durations == sorted(set(durations))
        requested = set(modes[0].demands)
        assert requested
        assert all(set(mode.demands) == requested for mode in modes)
        columns = {r: [mode.demands[r] for mode in modes] for r in requested}
        assert all(column == sorted(column, reverse=True) for column in columns.values())
        assert any(len(set(column)) == len(modes) for column in columns.values())
        for mode in modes:
            assert all(amount <= capacities[r] for r, amount in mode.demands.items())
        for a in modes:
            for b in modes:
                if a is not b:
                    no_longer = a.duration <= b.duration
                    no_heavier = all(a.demands[r] <= b.demands[r] for r in requested)
                    assert not (no_longer and no_heavier), f"a mode of {task.id} is dominated"


def test_multi_mode_capacities_fit_every_mode_at_zero_strength():
    problem = MultiModeRcpsp(resource_strength=0.0).generate(3)
    for resource in problem.resources:
        largest = max(
            mode.demands.get(resource.id, 0.0) for task in problem.tasks for mode in task.modes
        )
        assert resource.capacity == max(largest, 1.0)


@pytest.mark.parametrize(
    ("family", "parameters", "message"),
    [
        ("rcpsp", {"start_activities": 20, "finish_activities": 11}, "cannot exceed"),
        ("rcpsp", {"network_complexity": 0.0}, "positive"),
        ("rcpsp", {"resource_factor": 1.5}, "resource_factor"),
        ("rcpsp", {"resource_strength": -0.1}, "resource_strength"),
        ("rcpsp", {"max_degree": 0}, "max_degree"),
        ("multi-mode-rcpsp", {"resource_factor": 0.0}, "positive"),
        ("multi-mode-rcpsp", {"max_modes": 4, "max_duration": 3}, "duration range"),
        ("multi-mode-rcpsp", {"max_demand": 2}, "demand range"),
        ("multi-mode-rcpsp", {"min_modes": 3, "max_modes": 2}, "cannot exceed"),
    ],
)
def test_invalid_parameters_are_refused(family, parameters, message):
    with pytest.raises(ConfigError, match=message):
        REGISTRY.create(family, **parameters)


def test_project_builder():
    problem = project_problem(
        [[(3.0, [2.0, 0.0])], [(1.0, [0.0, 1.0]), (2.0, [0.0, 0.0])]],
        [(0, 1)],
        [2.0, 5.0],
        activity_ids=["x", "y"],
    )
    assert [task.id for task in problem.tasks] == ["x", "y"]
    assert dict(problem.task("y").modes[1].demands) == {}
    assert [(d.before, d.after) for d in problem.dependencies] == [("x", "y")]
    assert [(r.id, r.capacity) for r in problem.resources] == [("r1", 2.0), ("r2", 5.0)]


@pytest.mark.parametrize(
    "build",
    [
        lambda: project_problem([], [], [1.0]),
        lambda: project_problem([[(1.0, [1.0])]], [], [1.0, 2.0]),
        lambda: project_problem([[(1.0, [1.0])]], [(0, 1)], [1.0]),
        lambda: project_problem([[(1.0, [1.0])]], [], [1.0], activity_ids=["a", "b"]),
        lambda: project_problem([[(1.0, [3.0])]], [], [1.0]),
    ],
)
def test_project_builder_refuses_malformed_data(build):
    with pytest.raises(ModelError):
        build()
