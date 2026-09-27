"""The data model: every validation rule, derived views, features, and immutability."""

from __future__ import annotations

import copy
import dataclasses
import math
import pickle
from collections.abc import Callable
from typing import Any

import pytest

from dispatchatlas.core import model
from dispatchatlas.core._frozen import FrozenMap
from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import (
    FEATURES,
    Dependency,
    Link,
    Mode,
    Network,
    Problem,
    Resource,
    SetupTimes,
    Task,
)
from dispatchatlas.core.serialization import problem_digest, problem_to_dict
from tests.helpers import chain_problem, random_problem


def _problem(**changes: Any) -> Problem:
    """A valid two-machine problem, with fields replaced by ``changes``."""
    fields: dict[str, Any] = {
        "id": "base",
        "tasks": (Task.on("a", "m1", 3), Task.on("b", "m2", 2)),
        "resources": (Resource("m1"), Resource("m2")),
    }
    fields.update(changes)
    return Problem(**fields)


def _issues(**changes: Any) -> tuple[str, ...]:
    """The issues a problem built from ``changes`` is rejected with."""
    with pytest.raises(ModelError) as caught:
        _problem(**changes)
    assert caught.value.issues, "a problem-level error lists its issues"
    for issue in caught.value.issues:
        assert issue in str(caught.value)
    return caught.value.issues


def _raises(build: Callable[[], object], fragment: str) -> None:
    with pytest.raises(ModelError) as caught:
        build()
    assert fragment in str(caught.value), str(caught.value)


# ----------------------------------------------------------------- resources


@pytest.mark.parametrize(
    ("build", "fragment"),
    [
        (lambda: Resource(""), "resource id must be a non-empty string"),
        (lambda: Resource(" m"), "resource id must be a non-empty string"),
        (lambda: Resource(7), "resource id must be a non-empty string"),  # type: ignore[arg-type]
        (lambda: Resource("m", capacity=0), "capacity must be positive"),
        (lambda: Resource("m", capacity=-2), "capacity must be positive"),
        (lambda: Resource("m", capacity=math.inf), "capacity must be finite"),
        (lambda: Resource("m", capacity=math.nan), "capacity must be finite"),
        (lambda: Resource("m", capacity=True), "capacity must be a number"),
        (lambda: Resource("m", capacity="1"), "capacity must be a number"),  # type: ignore[arg-type]
        (lambda: Resource("m", kind=""), "kind must be a non-empty string"),
        (lambda: Resource("m", tier="edge "), "tier must be a non-empty string"),
        (lambda: Resource("m", idle_power=-1), "idle_power must be at least 0.0"),
        (lambda: Resource("m", busy_power=-1), "busy_power must be at least 0.0"),
        (lambda: Resource("m", idle_power=5, busy_power=3), "busy_power must be at least idle"),
        (lambda: Resource("m", cost_rate=-0.5), "cost_rate must be at least 0.0"),
        (lambda: Resource("m", cost_rate=math.inf), "cost_rate must be finite"),
        (lambda: Resource("m", metadata=[("k", 1)]), "metadata must be a mapping"),  # type: ignore[arg-type]
        (lambda: Resource("m", metadata={"": 1}), "metadata keys must be non-empty strings"),
        (lambda: Resource("m", metadata={3: 1}), "metadata keys must be non-empty strings"),  # type: ignore[dict-item]
        (lambda: Resource("m", metadata={3: 1, "k": 2}), "metadata keys must be non-empty"),  # type: ignore[dict-item]
        (lambda: Resource("m", metadata={"k": [1]}), "metadata value for 'k' must be a string"),  # type: ignore[dict-item]
        (lambda: Resource("m", metadata={"k": None}), "metadata value for 'k' must be a string"),  # type: ignore[dict-item]
        (lambda: Resource("m", metadata={"k": math.nan}), "metadata value for 'k' must be finite"),
        (lambda: Resource("m", metadata={"k": -math.inf}), "metadata value for 'k' must be finite"),
    ],
)
def test_resource_rejects_invalid_fields(build: Callable[[], object], fragment: str) -> None:
    _raises(build, fragment)


def test_resource_normalises_numbers_and_derives_its_endpoint() -> None:
    machine = Resource("m", capacity=1, idle_power=2, busy_power=7, cost_rate=3)
    assert machine.capacity == 1.0
    assert isinstance(machine.capacity, float)
    assert (machine.idle_power, machine.busy_power, machine.cost_rate) == (2.0, 7.0, 3.0)
    assert machine.unary
    assert machine.endpoint == "m"
    node = Resource("n", capacity=8, kind="node", tier="edge", metadata={"z": 1, "a": "x"})
    assert not node.unary
    assert node.endpoint == "edge"
    assert list(node.metadata) == ["a", "z"]


# --------------------------------------------------------------------- modes


@pytest.mark.parametrize(
    ("build", "fragment"),
    [
        (lambda: Mode(-1), "mode duration must be at least 0.0"),
        (lambda: Mode(math.inf), "mode duration must be finite"),
        (lambda: Mode(math.nan), "mode duration must be finite"),
        (lambda: Mode("3"), "mode duration must be a number"),  # type: ignore[arg-type]
        (lambda: Mode(1, [("m", 1)]), "mode demands must be a mapping"),  # type: ignore[arg-type]
        (lambda: Mode(1, {"": 1}), "demanded resource id must be a non-empty string"),
        (lambda: Mode(1, {None: 1}), "demanded resource id must be a non-empty string"),  # type: ignore[dict-item]
        (lambda: Mode(1, {2: 1, "m": 1}), "demanded resource id must be a non-empty string"),  # type: ignore[dict-item]
        (lambda: Mode(1, {"m": 0}), "demand on 'm' must be positive"),
        (lambda: Mode(1, {"m": -1}), "demand on 'm' must be positive"),
        (lambda: Mode(1, {"m": math.inf}), "demand on 'm' must be finite"),
        (lambda: Mode(1, {"m": 1}, location=""), "mode location must be a non-empty string"),
    ],
)
def test_mode_rejects_invalid_fields(build: Callable[[], object], fragment: str) -> None:
    _raises(build, fragment)


def test_mode_sorts_demands_and_infers_a_single_location() -> None:
    mode = Mode(2, {"z": 1, "a": 2})
    assert list(mode.demands) == ["a", "z"]
    assert mode.demands["a"] == 2.0
    assert mode.location is None
    assert Mode(2, {"m": 1}).location == "m"
    assert Mode(2, {"m": 1}, location="n").location == "n"
    assert Mode(0).demands == {}
    assert Mode(0).location is None


# --------------------------------------------------------------------- tasks


@pytest.mark.parametrize(
    ("build", "fragment"),
    [
        (lambda: Task("", (Mode(1),)), "task id must be a non-empty string"),
        (lambda: Task("a", ()), "task 'a' needs at least one Mode"),
        (lambda: Task("a", ("fast",)), "task 'a' needs at least one Mode"),  # type: ignore[arg-type]
        (lambda: Task("a", (Mode(1),), release=-1), "task 'a' release must be at least 0.0"),
        (lambda: Task("a", (Mode(1),), release=math.inf), "task 'a' release must be finite"),
        (lambda: Task("a", (Mode(1),), due=-1), "task 'a' due must be at least 0.0"),
        (lambda: Task("a", (Mode(1),), due=math.nan), "task 'a' due must be finite"),
        (lambda: Task("a", (Mode(1),), deadline=-1), "task 'a' deadline must be at least 0.0"),
        (lambda: Task("a", (Mode(1),), deadline=math.inf), "task 'a' deadline must be finite"),
        (lambda: Task("a", (Mode(1),), weight=-1), "task 'a' weight must be at least 0.0"),
        (lambda: Task("a", (Mode(1),), weight=False), "task 'a' weight must be a number"),
        (lambda: Task("a", (Mode(1),), group=""), "task 'a' group must be a non-empty string"),
        (lambda: Task("a", (Mode(1),), metadata={"k": {}}), "metadata value for 'k'"),  # type: ignore[dict-item]
    ],
)
def test_task_rejects_invalid_fields(build: Callable[[], object], fragment: str) -> None:
    _raises(build, fragment)


def test_task_on_builds_a_single_mode_task() -> None:
    task = Task.on("a", "pool", 4, demand=2, release=1, due=9, weight=3, group="j")
    assert task.modes == (Mode(4.0, {"pool": 2.0}),)
    assert task.modes[0].location == "pool"
    assert (task.release, task.due, task.deadline, task.weight, task.group) == (
        1.0,
        9.0,
        None,
        3.0,
        "j",
    )
    flexible = Task("b", [Mode(5, {"m1": 1}), Mode(3, {"m2": 1})])  # type: ignore[arg-type]
    assert isinstance(flexible.modes, tuple)
    assert flexible.min_duration == 3.0


# -------------------------------------------------------------- dependencies


@pytest.mark.parametrize(
    ("build", "fragment"),
    [
        (lambda: Dependency("", "b"), "dependency 'before' id must be a non-empty string"),
        (lambda: Dependency("a", " b"), "dependency 'after' id must be a non-empty string"),
        (lambda: Dependency("a", "a"), "task 'a' cannot depend on itself"),
        (lambda: Dependency("a", "b", lag=-1), "dependency 'a' -> 'b' lag must be at least 0.0"),
        (lambda: Dependency("a", "b", lag=math.inf), "dependency 'a' -> 'b' lag must be finite"),
        (lambda: Dependency("a", "b", data=-3), "dependency 'a' -> 'b' data must be at least"),
        (lambda: Dependency("a", "b", data=True), "dependency 'a' -> 'b' data must be a number"),
    ],
)
def test_dependency_rejects_invalid_fields(build: Callable[[], object], fragment: str) -> None:
    _raises(build, fragment)


# ------------------------------------------------------------------- network


@pytest.mark.parametrize(
    ("build", "fragment"),
    [
        (lambda: Link("", "b"), "link endpoint must be a non-empty string"),
        (lambda: Link("a", None), "link endpoint must be a non-empty string"),  # type: ignore[arg-type]
        (lambda: Link("a", "b", bandwidth=0), "bandwidth must be positive"),
        (lambda: Link("a", "b", bandwidth=-5), "bandwidth must be positive"),
        (lambda: Link("a", "b", bandwidth=math.nan), "bandwidth must be finite"),
        (lambda: Link("a", "b", latency=-1), "latency must be at least 0.0"),
        (lambda: Link("a", "b", latency=math.inf), "latency must be finite"),
        (lambda: Link("a", "b", energy_per_unit=-1), "energy_per_unit must be at least 0.0"),
        (lambda: Link("a", "b", cost_per_unit=math.inf), "cost_per_unit must be finite"),
        (lambda: Network(("edge",)), "network links must be Link objects"),  # type: ignore[arg-type]
        (lambda: Network((Link("a", "b"), Link("a", "b"))), "defines the link 'a' <-> 'b' twice"),
        (lambda: Network((Link("b", "a"), Link("a", "b"))), "defines the link 'a' <-> 'b' twice"),
        (lambda: Network((Link("a", "a"), Link("a", "a"))), "defines the link 'a' <-> 'a' twice"),
    ],
)
def test_links_and_networks_reject_invalid_fields(
    build: Callable[[], object], fragment: str
) -> None:
    _raises(build, fragment)


def test_link_transfer_time_is_latency_plus_volume_over_bandwidth() -> None:
    link = Link("edge", "cloud", bandwidth=4, latency=0.5)
    assert link.transfer_time(10) == 0.5 + 10 / 4
    assert link.transfer_time(0) == 0.0
    assert Link("a", "b", latency=2).transfer_time(1e9) == 2.0
    assert Link("a", "b").transfer_time(5) == 0.0


def test_network_links_are_symmetric() -> None:
    near = Link("edge", "edge", bandwidth=10)
    far = Link("edge", "cloud", bandwidth=2)
    network = Network([near, far])  # type: ignore[arg-type]
    assert network.links == (near, far)
    assert network.link("edge", "cloud") is far
    assert network.link("cloud", "edge") is far
    assert network.link("edge", "edge") is near
    assert network.link("cloud", "cloud") is None
    assert network.endpoints == frozenset({"edge", "cloud"})


# --------------------------------------------------------------------- setups


@pytest.mark.parametrize(
    ("build", "fragment"),
    [
        (lambda: SetupTimes([("m", None, "a")]), "setup times must be a mapping"),  # type: ignore[arg-type]
        (lambda: SetupTimes({("m", "a"): 1}), "setup keys must be (resource, previous, following)"),  # type: ignore[dict-item]
        (lambda: SetupTimes({"m": 1}), "setup keys must be (resource, previous, following)"),  # type: ignore[dict-item]
        (lambda: SetupTimes({("", None, "a"): 1}), "setup resource must be a non-empty string"),
        (lambda: SetupTimes({("m", "", "a"): 1}), "setup previous task must be a non-empty"),
        (lambda: SetupTimes({("m", "a", None): 1}), "setup following task must be a non-empty"),  # type: ignore[dict-item]
        (lambda: SetupTimes({("m", None, "a"): -1}), "must be at least 0.0"),
        (lambda: SetupTimes({("m", None, "a"): math.nan}), "must be finite"),
    ],
)
def test_setup_times_reject_invalid_entries(build: Callable[[], object], fragment: str) -> None:
    _raises(build, fragment)


def test_setup_times_drop_zeros_and_default_to_zero() -> None:
    setups = SetupTimes({("m", None, "a"): 2, ("m", "a", "b"): 0, ("n", "b", "a"): 1.5})
    assert dict(setups.times) == {("m", None, "a"): 2.0, ("n", "b", "a"): 1.5}
    assert setups.resources == frozenset({"m", "n"})
    assert setups.get("m", None, "a") == 2.0
    assert setups.get("m", "a", "b") == 0.0
    assert setups.get("x", "a", "b") == 0.0
    assert SetupTimes().times == {}


# ------------------------------------------------------------------- problems


def test_problem_rejects_invalid_scalar_fields() -> None:
    _raises(lambda: _problem(id=""), "problem id must be a non-empty string")
    _raises(lambda: _problem(family=" shop"), "problem family must be a non-empty string")
    _raises(lambda: _problem(metadata={"k": object()}), "metadata value for 'k'")


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"tasks": ()}, "the problem has no tasks"),
        ({"tasks": ("a",)}, "every task must be a Task"),
        ({"resources": ("m1",)}, "every resource must be a Resource"),
        ({"dependencies": (("a", "b"),)}, "every dependency must be a Dependency"),
        (
            {"tasks": (Task.on("a", "m1", 1), Task.on("a", "m2", 1))},
            "duplicate task ids: ['a']",
        ),
        (
            {"resources": (Resource("m1"), Resource("m2"), Resource("m1"))},
            "duplicate resource ids: ['m1']",
        ),
        (
            {"tasks": (Task.on("a", "m1", 1), Task.on("b", "gpu", 1))},
            "task 'b' mode 0 demands unknown resource 'gpu'",
        ),
        (
            {"tasks": (Task.on("a", "m1", 1), Task.on("b", "m2", 1, demand=2))},
            "task 'b' mode 0 demands 2.0 of 'm2', above its capacity 1.0",
        ),
        (
            {"tasks": (Task.on("a", "m1", 1), Task("b", (Mode(1, {"m2": 1}, location="x"),)))},
            "task 'b' mode 0 is located on unknown resource 'x'",
        ),
        ({"dependencies": (Dependency("a", "z"),)}, "dependency refers to unknown task 'z'"),
        (
            {"dependencies": (Dependency("a", "b"), Dependency("a", "b", lag=1))},
            "dependency 'a' -> 'b' is declared twice",
        ),
        (
            {"dependencies": (Dependency("a", "b"), Dependency("b", "a"))},
            "the dependencies form a cycle",
        ),
        ({"objectives": ()}, "the problem declares no objective"),
        ({"objectives": ("makespan", "makespan")}, "an objective is declared twice"),
        ({"objectives": ("speed",)}, "unknown objective 'speed'"),
        ({"objectives": (3,)}, "unknown objective 3"),
        (
            {
                "tasks": (
                    Task.on("a", "m1", 1, group="j"),
                    Task.on("b", "m2", 1, group="j", weight=2),
                )
            },
            "tasks of group 'j' have different weights",
        ),
    ],
)
def test_problem_reports_each_structural_defect(changes: dict[str, Any], expected: str) -> None:
    assert expected in _issues(**changes)


def test_demand_rounding_just_above_capacity_is_accepted() -> None:
    problem = _problem(
        tasks=(Task.on("a", "m1", 1, demand=0.1 + 0.2), Task.on("b", "m2", 1)),
        resources=(Resource("m1", capacity=0.3), Resource("m2")),
    )
    assert problem.tasks[0].modes[0].demands["m1"] > 0.3


def test_problem_accepts_a_single_objective_name_and_orders_objectives() -> None:
    assert _problem(objectives="energy").objectives == ("energy",)
    problem = _problem(objectives=["total_tardiness", "makespan"])
    assert problem.objectives == ("total_tardiness", "makespan")


def test_problem_lists_every_defect_at_once() -> None:
    issues = _issues(
        tasks=(
            Task.on("a", "gpu", 1, group="j"),
            Task.on("b", "m2", 1, group="j", weight=2),
            Task.on("b", "m2", 1),
        ),
        objectives=("speed",),
        setups=SetupTimes({("m9", None, "b"): 1.0}),
    )
    expected = {
        "duplicate task ids: ['b']",
        "task 'a' mode 0 demands unknown resource 'gpu'",
        "unknown objective 'speed'",
        "tasks of group 'j' have different weights",
        "setup refers to unknown resource 'm9'",
    }
    assert expected <= set(issues)
    empty = _issues(tasks=(), objectives=("speed",))
    assert empty == ("the problem has no tasks", "unknown objective 'speed'")


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {"setups": SetupTimes({("gpu", None, "a"): 1})},
            "setup refers to unknown resource 'gpu'",
        ),
        (
            {
                "resources": (Resource("m1", capacity=2), Resource("m2")),
                "setups": SetupTimes({("m1", None, "a"): 1}),
            },
            "setups are allowed on capacity-1 resources only: 'm1'",
        ),
        ({"setups": SetupTimes({("m1", "x", "a"): 1})}, "setup refers to unknown task 'x'"),
        ({"setups": SetupTimes({("m1", "a", "y"): 1})}, "setup refers to unknown task 'y'"),
        (
            {
                "resources": (Resource("m1"), Resource("m2")),
                "tasks": (Task.on("a", "m1", 1, demand=0.5), Task.on("b", "m2", 1)),
                "setups": SetupTimes({("m1", None, "b"): 1}),
            },
            "task 'a' must hold the whole of machine 'm1', which has setup times",
        ),
        (
            {
                "tasks": (Task.on("a", "m1", 0), Task.on("b", "m2", 1)),
                "setups": SetupTimes({("m1", None, "b"): 1}),
            },
            "task 'a' has a zero-duration mode on machine 'm1', which has setup times",
        ),
    ],
)
def test_setup_rules(changes: dict[str, Any], expected: str) -> None:
    assert expected in _issues(**changes)


def test_zero_duration_modes_are_allowed_on_machines_without_setups() -> None:
    setups = SetupTimes({("m2", None, "b"): 1, ("m1", None, "a"): 0})
    problem = _problem(tasks=(Task.on("a", "m1", 0), Task.on("b", "m2", 1)), setups=setups)
    assert problem.setups is not None
    assert problem.setups.resources == frozenset({"m2"})


@pytest.mark.parametrize(
    ("tasks", "resources", "expected"),
    [
        (
            (Task.on("a", "p", 1), Task.on("b", "m2", 1)),
            (Resource("p", capacity=2), Resource("m2")),
            "task 'a' must hold the whole of capacity-1 machine 'p' in a permutation problem",
        ),
        (
            (Task.on("a", "m1", 1, demand=0.5), Task.on("b", "m2", 1)),
            (Resource("m1"), Resource("m2")),
            "task 'a' must hold the whole of capacity-1 machine 'm1' in a permutation problem",
        ),
        (
            (Task.on("a", "m1", 1, group="j"), Task.on("b", "m1", 2, group="j")),
            (Resource("m1"),),
            "tasks 'a' and 'b' of group 'j' can both use machine 'm1'; in a permutation "
            "problem a group uses a machine at most once",
        ),
        (
            (
                Task("a", (Mode(1, {"m1": 1}), Mode(1, {"m2": 1})), group="j"),
                Task.on("b", "m2", 2, group="j"),
            ),
            (Resource("m1"), Resource("m2")),
            "tasks 'a' and 'b' of group 'j' can both use machine 'm2'; in a permutation "
            "problem a group uses a machine at most once",
        ),
    ],
)
def test_permutation_rules(
    tasks: tuple[Task, ...], resources: tuple[Resource, ...], expected: str
) -> None:
    assert expected in _issues(tasks=tasks, resources=resources, permutation=True)


def test_permutation_problems_accept_whole_machines_used_once_per_group() -> None:
    tasks = (
        Task("a", (Mode(1, {"m1": 1}), Mode(2, {"m1": 1})), group="j"),
        Task.on("b", "m2", 0, group="j"),
        Task.on("c", "m1", 3),
        Task.on("d", "m1", 3),
    )
    problem = _problem(tasks=tasks, dependencies=(Dependency("a", "b"),), permutation=True)
    assert "permutation" in problem.features


def test_network_needs_a_link_for_every_possible_transfer() -> None:
    resources = (Resource("m1", tier="edge"), Resource("m2", tier="cloud"), Resource("m3"))
    tasks = (
        Task.on("a", "m1", 1),
        Task("b", (Mode(1, {"m1": 1}), Mode(1, {"m2": 1}), Mode(1, {"m3": 1}))),
    )
    issues = _issues(
        tasks=tasks,
        resources=resources,
        dependencies=(Dependency("a", "b", data=5),),
        network=Network((Link("edge", "edge"),)),
    )
    assert issues == (
        "the network has no link between endpoints 'cloud' and 'edge'",
        "the network has no link between endpoints 'edge' and 'm3'",
    )
    # Without data, or on the same resource, nothing needs a link.
    problem = _problem(
        tasks=(Task.on("a", "m1", 1), Task.on("b", "m1", 1), Task.on("c", "m3", 1)),
        resources=resources,
        dependencies=(Dependency("a", "b", data=5), Dependency("a", "c")),
        network=Network(),
    )
    assert problem.network == Network()


# -------------------------------------------------------------- derived views


def test_lookups_and_graph_views() -> None:
    problem = _problem(
        tasks=(
            Task.on("a", "m1", 3, group="j1"),
            Task.on("b", "m2", 2, group="j2"),
            Task.on("c", "m1", 1, group="j1"),
            Task.on("d", "m2", 1),
        ),
        dependencies=(Dependency("c", "b"), Dependency("a", "b", lag=2), Dependency("a", "c")),
    )
    assert dict(problem.task_index) == {"a": 0, "b": 1, "c": 2, "d": 3}
    assert dict(problem.resource_index) == {"m1": 0, "m2": 1}
    assert problem.task("c") is problem.tasks[2]
    assert problem.resource("m2") is problem.resources[1]
    assert problem.incoming[1] == (Dependency("c", "b"), Dependency("a", "b", lag=2))
    assert problem.predecessors == ((), (2, 0), (0,), ())
    assert problem.successors == ((1, 2), (), (1,), ())
    assert problem.topological_order == (0, 2, 1, 3)
    assert dict(problem.groups) == {"j1": (0, 2), "j2": (1,)}
    assert problem.size == 4
    assert problem.summary() == {
        "id": "base",
        "family": None,
        "tasks": 4,
        "resources": 2,
        "dependencies": 3,
        "modes": 4,
        "groups": 2,
        "objectives": ["makespan"],
        "features": ["groups", "lags", "precedence"],
    }


def test_topological_order_breaks_ties_by_lowest_index() -> None:
    tasks = tuple(Task.on(name, "m1", 1) for name in "edcba")
    problem = _problem(tasks=tasks, dependencies=(Dependency("e", "a"), Dependency("b", "d")))
    assert problem.topological_order == (0, 2, 3, 1, 4)
    for seed in range(20):
        random = random_problem(seed, tasks=15, density=0.3)
        position = {task: n for n, task in enumerate(random.topological_order)}
        assert sorted(position) == list(range(random.size))
        for after, befores in enumerate(random.predecessors):
            assert all(position[before] < position[after] for before in befores)


def test_long_precedence_chains_validate_without_recursion() -> None:
    count = 5000
    tasks = tuple(Task.on(f"t{i}", "m1", 1) for i in range(count))
    chain = tuple(Dependency(f"t{i}", f"t{i + 1}") for i in range(count - 1))
    problem = _problem(tasks=tasks, dependencies=chain)
    assert problem.topological_order == tuple(range(count))
    cyclic = (*chain, Dependency(f"t{count - 1}", "t0"))
    assert "the dependencies form a cycle" in _issues(tasks=tasks, dependencies=cyclic)


def test_transfer_time_between_resources() -> None:
    resources = (
        Resource("e1", tier="edge"),
        Resource("e2", tier="edge"),
        Resource("c1", tier="cloud"),
    )
    tasks = (
        Task("a", (Mode(1, {"e1": 1}), Mode(1, {"c1": 1}))),
        Task("b", (Mode(1, {"e1": 1}), Mode(1, {"e2": 1}), Mode(1, {"c1": 1}))),
    )
    network = Network(
        (Link("edge", "edge", bandwidth=10, latency=1), Link("edge", "cloud", bandwidth=2))
    )
    dependency = Dependency("a", "b", data=20)
    problem = _problem(
        tasks=tasks, resources=resources, dependencies=(dependency,), network=network
    )
    assert problem.transfer_time(dependency, "e1", "e1") == 0.0
    assert problem.transfer_time(dependency, "e1", "e2") == 1 + 20 / 10
    assert problem.transfer_time(dependency, "e1", "c1") == 20 / 2
    assert problem.transfer_time(dependency, "c1", "e2") == 20 / 2
    assert problem.transfer_time(dependency, None, "e2") == 0.0
    assert problem.transfer_time(Dependency("a", "b"), "e1", "c1") == 0.0
    no_network = _problem(tasks=tasks, resources=resources, dependencies=(dependency,))
    assert no_network.transfer_time(dependency, "e1", "c1") == 0.0


def test_exclusive_resources_are_those_every_demand_takes_whole() -> None:
    problem = _problem(
        tasks=(
            Task.on("a", "m1", 1),
            Task.on("b", "wide", 1, demand=2),
            Task("c", (Mode(1, {"wide": 2, "shared": 1}), Mode(1, {"mixed": 3}))),
            Task.on("d", "mixed", 1, demand=1),
            Task.on("e", "shared", 1, demand=1),
        ),
        resources=(
            Resource("m1"),
            Resource("wide", capacity=2),
            Resource("shared", capacity=2),
            Resource("mixed", capacity=3),
            Resource("idle"),
        ),
    )
    assert problem.exclusive_resources == frozenset({"m1", "wide"})


def _feature_cases() -> list[tuple[str, dict[str, Any]]]:
    two_modes = Task("b", (Mode(2, {"m1": 1}), Mode(1, {"m2": 1})))
    return [
        ("precedence", {"dependencies": (Dependency("a", "b"),)}),
        ("lags", {"dependencies": (Dependency("a", "b", lag=1),)}),
        ("modes", {"tasks": (Task.on("a", "m1", 3), two_modes)}),
        (
            "multi-resource",
            {"tasks": (Task.on("a", "m1", 3), Task("b", (Mode(1, {"m1": 1, "m2": 1}),)))},
        ),
        ("cumulative", {"resources": (Resource("m1"), Resource("m2"), Resource("p", capacity=2))}),
        ("release", {"tasks": (Task.on("a", "m1", 3, release=1), Task.on("b", "m2", 2))}),
        ("due", {"tasks": (Task.on("a", "m1", 3, due=0), Task.on("b", "m2", 2))}),
        ("deadline", {"tasks": (Task.on("a", "m1", 3, deadline=9), Task.on("b", "m2", 2))}),
        ("setups", {"setups": SetupTimes({("m1", None, "a"): 1})}),
        (
            "network",
            {
                "dependencies": (Dependency("a", "b", data=1),),
                "network": Network((Link("m1", "m2"),)),
            },
        ),
        ("permutation", {"permutation": True}),
        ("groups", {"tasks": (Task.on("a", "m1", 3, group="j"), Task.on("b", "m2", 2))}),
    ]


def test_feature_cases_cover_every_feature() -> None:
    assert sorted(name for name, _ in _feature_cases()) == sorted(FEATURES)


@pytest.mark.parametrize(("feature", "changes"), _feature_cases())
def test_each_feature_is_detected_alone(feature: str, changes: dict[str, Any]) -> None:
    assert _problem(**changes).features == frozenset({feature}) | (
        {"precedence"} if feature in ("lags", "network") else set()
    )


def test_features_ignore_neutral_values() -> None:
    assert _problem().features == frozenset()
    neutral = _problem(
        tasks=(Task.on("a", "m1", 3, release=0), Task.on("b", "m2", 2)),
        dependencies=(Dependency("a", "b", lag=0, data=5),),
        setups=SetupTimes({("m1", None, "a"): 0}),
    )
    assert neutral.features == frozenset({"precedence"})
    no_data = _problem(dependencies=(Dependency("a", "b"),), network=Network())
    assert no_data.features == frozenset({"precedence"})


# ------------------------------------------------------------ derived copies


def test_renamed_keeps_everything_but_the_id() -> None:
    problem = random_problem(3, network=True, setups=True, deadlines=True)
    features = problem.features
    renamed = problem.renamed("copy")
    assert renamed.id == "copy"
    assert problem.id == "random-3"
    assert renamed == dataclasses.replace(problem, id="copy")
    assert renamed.features == features
    assert renamed.task_index == problem.task_index
    assert problem_to_dict(renamed) == {**problem_to_dict(problem), "id": "copy"}
    assert problem_digest(renamed) != problem_digest(problem)
    assert problem_digest(renamed.renamed(problem.id)) == problem_digest(problem)
    with pytest.raises(ModelError, match="problem id"):
        problem.renamed(" copy")


def test_renamed_skips_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = chain_problem()

    def refuse(_: Problem) -> list[str]:
        raise AssertionError("validation ran")

    monkeypatch.setattr(model, "_structural_issues", refuse)
    assert problem.renamed("other").id == "other"
    with pytest.raises(AssertionError, match="validation ran"):
        problem.with_objectives("makespan")


def test_with_objectives_returns_a_validated_copy() -> None:
    problem = chain_problem()
    changed = problem.with_objectives("total_tardiness", "makespan")
    assert changed.objectives == ("total_tardiness", "makespan")
    assert problem.objectives == ("makespan",)
    assert changed.tasks == problem.tasks
    with pytest.raises(ModelError, match="unknown objective 'speed'"):
        problem.with_objectives("speed")
    with pytest.raises(ModelError, match="declares no objective"):
        problem.with_objectives()


# --------------------------------------------------------------- immutability


def test_model_objects_are_immutable() -> None:
    problem = random_problem(1, network=True, setups=True)
    task = problem.tasks[0]
    mode = task.modes[0]
    resource = problem.resources[0]
    assert problem.network is not None
    assert problem.setups is not None
    link = problem.network.links[0]
    for target, name, value in (
        (problem, "id", "x"),
        (problem, "tasks", ()),
        (task, "release", 5.0),
        (mode, "duration", 1.0),
        (resource, "capacity", 3.0),
        (Dependency("a", "b"), "lag", 1.0),
        (link, "bandwidth", 1.0),
        (problem.network, "links", ()),
        (problem.setups, "times", {}),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(target, name, value)
    for mapping in (
        mode.demands,
        task.metadata,
        resource.metadata,
        problem.metadata,
        problem.setups.times,
        problem.task_index,
        problem.resource_index,
        problem.groups,
        FEATURES,
    ):
        assert isinstance(mapping, FrozenMap)
        with pytest.raises(TypeError):
            mapping["new"] = 1  # type: ignore[index]


def test_inputs_are_copied_not_shared() -> None:
    demands = {"m1": 1.0}
    metadata: dict[str, Any] = {"source": "test"}
    times: dict[tuple[str, str | None, str], float] = {("m1", None, "a"): 2.0}
    tasks = [Task("a", (Mode(3, demands),)), Task.on("b", "m2", 2)]
    problem = _problem(tasks=tasks, metadata=metadata, setups=SetupTimes(times))
    demands["m1"] = 9.0
    metadata["source"] = "changed"
    times[("m1", None, "a")] = 7.0
    tasks.pop()
    assert problem.tasks[0].modes[0].demands["m1"] == 1.0
    assert problem.metadata["source"] == "test"
    assert problem.setups is not None
    assert problem.setups.get("m1", None, "a") == 2.0
    assert isinstance(problem.tasks, tuple)
    assert len(problem.tasks) == 2


def test_equal_problems_hash_alike() -> None:
    first, second = random_problem(4, network=True), random_problem(4, network=True)
    assert first == second
    assert hash(first) == hash(second)
    assert first != random_problem(5, network=True)


def _flow_shop() -> Problem:
    """A permutation flow shop with a zero processing time."""
    return Problem(
        "flow",
        tasks=(
            Task.on("j1m1", "m1", 3, group="j1"),
            Task.on("j1m2", "m2", 0, group="j1"),
            Task.on("j2m1", "m1", 2, group="j2"),
            Task.on("j2m2", "m2", 4, group="j2"),
        ),
        resources=(Resource("m1"), Resource("m2")),
        dependencies=(Dependency("j1m1", "j1m2"), Dependency("j2m1", "j2m2")),
        permutation=True,
        metadata={"source": "hand", "jobs": 2, "ratio": 0.5, "flag": True},
    )


def _warmed(problem: Problem) -> Problem:
    """Read every cached view, so that copies carry them too."""
    _ = (problem.task_index, problem.resource_index, problem.groups, problem.features)
    _ = (problem.exclusive_resources, problem.incoming, problem.topological_order)
    if problem.network is not None and problem.network.links:
        link = problem.network.links[0]
        problem.network.link(link.a, link.b)
    if problem.setups is not None:
        _ = problem.setups.resources
    return problem


@pytest.mark.parametrize("protocol", [pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL])
def test_problems_survive_pickle_and_deep_copies(protocol: int) -> None:
    problems = [_flow_shop(), chain_problem()] + [
        random_problem(seed, network=seed % 2 == 0, setups=seed % 3 == 0, deadlines=True)
        for seed in range(6)
    ]
    for problem in map(_warmed, problems):
        for again in (pickle.loads(pickle.dumps(problem, protocol)), copy.deepcopy(problem)):
            assert again == problem
            assert hash(again) == hash(problem)
            assert problem_digest(again) == problem_digest(problem)
            assert again.task_index == problem.task_index
            assert again.features == problem.features
            assert again.renamed("copy") == problem.renamed("copy")
        assert copy.copy(problem) == problem


def test_every_model_object_pickles() -> None:
    problem = _warmed(random_problem(2, network=True, setups=True))
    assert problem.network is not None
    assert problem.setups is not None
    parts: list[object] = [
        problem.tasks[0],
        problem.tasks[0].modes[0],
        problem.resources[0],
        problem.dependencies[0],
        problem.network,
        problem.network.links[0],
        problem.setups,
        FEATURES,
    ]
    for part in parts:
        assert pickle.loads(pickle.dumps(part)) == part
        assert copy.deepcopy(part) == part
