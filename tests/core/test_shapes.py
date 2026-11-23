"""Problem shapes: which problems are job shops, and how a solver requires a shape."""

from __future__ import annotations

import copy
import dataclasses
import pickle
import re
from typing import Any

import pytest

from dispatchatlas import core
from dispatchatlas.core._frozen import FrozenMap
from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import (
    FEATURES,
    SHAPES,
    Dependency,
    Link,
    Mode,
    Network,
    Problem,
    Resource,
    SetupTimes,
    Task,
)
from dispatchatlas.core.solver import SolverInfo
from tests.helpers import chain_problem, random_problem, small_job_shop

JOB_SHOP = frozenset({"job-shop"})


def _problem(**changes: Any) -> Problem:
    """Two independent tasks on two machines, with fields replaced by ``changes``."""
    fields: dict[str, Any] = {
        "id": "base",
        "tasks": (Task.on("a", "m1", 3), Task.on("b", "m2", 2)),
        "resources": (Resource("m1"), Resource("m2")),
    }
    fields.update(changes)
    return Problem(**fields)


def _three(*dependencies: Dependency, name: str = "three") -> Problem:
    """Three tasks on three machines, joined by ``dependencies``."""
    return Problem(
        name,
        tasks=(Task.on("a", "m1", 1), Task.on("b", "m2", 1), Task.on("c", "m3", 1)),
        resources=(Resource("m1"), Resource("m2"), Resource("m3")),
        dependencies=dependencies,
    )


def _info(**changes: Any) -> SolverInfo:
    fields: dict[str, Any] = {
        "id": "toy-solver",
        "name": "Toy solver",
        "family": "local-search",
        "summary": "Does nothing useful.",
    }
    fields.update(changes)
    return SolverInfo(**fields)


# ------------------------------------------------------------------- the names


def test_the_shapes_are_named_and_described() -> None:
    assert set(SHAPES) == {"job-shop"}
    assert core.SHAPES is SHAPES
    assert isinstance(SHAPES, FrozenMap)
    for description in SHAPES.values():
        assert description.endswith(".")
    with pytest.raises(TypeError):
        SHAPES["flow-shop"] = "Not allowed."  # type: ignore[index]


def test_a_shape_never_shares_its_name_with_a_feature() -> None:
    # Solver.unsupported() returns features and a missing shape in one set.
    assert set(SHAPES).isdisjoint(FEATURES)


# --------------------------------------------------------------- job shops


def test_chains_of_tasks_on_machines_are_a_job_shop() -> None:
    assert small_job_shop().shapes == JOB_SHOP
    # a -> b on one machine, and c on another.
    assert chain_problem().shapes == JOB_SHOP
    # Without dependencies every task is a job of its own.
    assert _problem().shapes == JOB_SHOP
    # A job may come back to a machine it visited.
    assert _three(Dependency("a", "b"), Dependency("b", "c")).shapes == JOB_SHOP
    assert isinstance(small_job_shop().shapes, frozenset)


def test_a_task_with_two_successors_or_two_predecessors_is_not_in_a_chain() -> None:
    assert _three(Dependency("a", "b"), Dependency("a", "c")).shapes == frozenset()
    assert _three(Dependency("a", "c"), Dependency("b", "c")).shapes == frozenset()
    diamond = Problem(
        "diamond",
        tasks=tuple(Task.on(name, f"m{i}", 1) for i, name in enumerate("abcd")),
        resources=tuple(Resource(f"m{i}") for i in range(4)),
        dependencies=(
            Dependency("a", "b"),
            Dependency("a", "c"),
            Dependency("b", "d"),
            Dependency("c", "d"),
        ),
    )
    assert diamond.shapes == frozenset()


def test_one_branch_in_one_job_ends_the_shape() -> None:
    job_shop = small_job_shop()
    assert job_shop.shapes == JOB_SHOP
    joined = dataclasses.replace(
        job_shop, dependencies=(*job_shop.dependencies, Dependency("j0-o0", "j1-o1"))
    )
    assert joined.shapes == frozenset()


def test_a_task_needs_one_mode_that_holds_one_machine() -> None:
    two_machines = Task("a", (Mode(3, {"m1": 1, "m2": 1}),))
    two_modes = Task("a", (Mode(3, {"m1": 1}), Mode(2, {"m2": 1})))
    same_machine = Task("a", (Mode(3, {"m1": 1}), Mode(2, {"m1": 1})))
    holds_nothing = Task("a", (Mode(0, {}),))
    other = Task.on("b", "m2", 2)
    for first in (two_machines, two_modes, same_machine, holds_nothing):
        assert _problem(tasks=(first, other)).shapes == frozenset()
    assert _problem(tasks=(Task("a", (Mode(3, {"m1": 1}),)), other)).shapes == JOB_SHOP


def test_the_machine_must_run_one_task_at_a_time() -> None:
    pool = (Resource("p", capacity=2),)
    # Every task takes the whole pool, so it is a machine of capacity 2.
    whole = (Task.on("a", "p", 3, demand=2), Task.on("b", "p", 2, demand=2))
    assert _problem(tasks=whole, resources=pool).shapes == JOB_SHOP
    # One task takes a share of it, so tasks can run side by side.
    share = (Task.on("a", "p", 3, demand=2), Task.on("b", "p", 2, demand=1))
    assert _problem(tasks=share, resources=pool).shapes == frozenset()
    units = (Task.on("a", "p", 3), Task.on("b", "p", 2))
    assert _problem(tasks=units, resources=pool).shapes == frozenset()
    # A capacity-1 resource that tasks share in halves is not flagged as cumulative.
    halves = (Task.on("a", "m1", 3, demand=0.5), Task.on("b", "m1", 2, demand=0.5))
    problem = _problem(tasks=halves, resources=(Resource("m1"),))
    assert "cumulative" not in problem.features
    assert problem.shapes == frozenset()


@pytest.mark.parametrize(
    "changes",
    [
        {"tasks": (Task.on("a", "m1", 3, release=2), Task.on("b", "m2", 2, due=1, deadline=9))},
        {
            "tasks": (Task.on("a", "m1", 3, group="j"), Task.on("b", "m2", 2, group="j")),
            "dependencies": (Dependency("a", "b", lag=2),),
        },
        {"setups": SetupTimes({("m1", None, "a"): 1})},
        {"permutation": True},
        {"dependencies": (Dependency("a", "b", data=4),), "network": Network((Link("m1", "m2"),))},
    ],
)
def test_other_features_do_not_decide_the_shape(changes: dict[str, Any]) -> None:
    problem = _problem(**changes)
    assert problem.features
    assert problem.shapes == JOB_SHOP


def _written_out(problem: Problem) -> bool:
    """The definition of the job shop, from the problem's fields alone."""
    before = {task.id: 0 for task in problem.tasks}
    after = dict(before)
    for dependency in problem.dependencies:
        after[dependency.before] += 1
        before[dependency.after] += 1
    capacity = {resource.id: resource.capacity for resource in problem.resources}
    for task in problem.tasks:
        if before[task.id] > 1 or after[task.id] > 1 or len(task.modes) != 1:
            return False
        demands = task.modes[0].demands
        if len(demands) != 1:
            return False
        (machine,) = demands
        takes_it_whole = all(
            mode.demands[machine] == capacity[machine]
            for other in problem.tasks
            for mode in other.modes
            if machine in mode.demands
        )
        if not takes_it_whole:
            return False
    return True


def test_the_shape_agrees_with_its_definition_on_random_problems() -> None:
    outcomes = []
    for seed in range(80):
        problem = random_problem(
            seed,
            tasks=6,
            machines=3,
            pools=1 if seed % 8 == 0 else 0,
            max_modes=2 if seed % 5 == 0 else 1,
            density=0.12,
        )
        outcomes.append("job-shop" in problem.shapes)
        assert ("job-shop" in problem.shapes) == _written_out(problem), problem.id
    assert 20 <= sum(outcomes) <= 60


def test_shapes_are_kept_by_copies_and_by_pickling() -> None:
    problem = small_job_shop()
    assert problem.shapes is problem.shapes
    for again in (
        problem.renamed("copy"),
        copy.deepcopy(problem),
        pickle.loads(pickle.dumps(problem)),
    ):
        assert again.shapes == JOB_SHOP


# ----------------------------------------------------------- the solver gate


def test_a_solver_requires_no_shape_unless_it_declares_one() -> None:
    assert _info().shape is None
    assert _info(shape="job-shop").shape == "job-shop"


def test_a_shape_the_library_does_not_define_is_refused() -> None:
    with pytest.raises(ModelError, match=re.escape("declares unknown shape 'flow-shop'")):
        _info(shape="flow-shop")
    with pytest.raises(ModelError, match="declares unknown shape ''"):
        _info(shape="")


def test_a_missing_shape_is_unsupported() -> None:
    needs = _info(shape="job-shop")
    assert needs.unsupported(small_job_shop()) == frozenset()
    assert needs.accepts(small_job_shop())
    fork = _three(Dependency("a", "b"), Dependency("a", "c"), name="fork")
    assert needs.unsupported(fork) == JOB_SHOP
    assert not needs.accepts(fork)
    # A solver without a requirement reports features only, as it always did.
    assert _info().unsupported(fork) == fork.features - _info().supports
    assert _info().accepts(fork)


def test_a_missing_shape_is_reported_with_the_features_not_modelled() -> None:
    problem = random_problem(3, tasks=6, max_modes=2, pools=0, network=True)
    assert "job-shop" not in problem.shapes
    narrow = _info(shape="job-shop", supports={"precedence", "groups"})
    assert narrow.unsupported(problem) == (problem.features - {"precedence", "groups"}) | JOB_SHOP
    # Where the problem has the shape, only its features count.
    job_shop = dataclasses.replace(small_job_shop(), setups=SetupTimes({("m0", None, "j0-o0"): 1}))
    assert job_shop.shapes == JOB_SHOP
    assert narrow.unsupported(job_shop) == frozenset({"setups"})
