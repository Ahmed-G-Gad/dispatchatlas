"""The evaluator: fast objectives equal the registry's, partial decodes, scores, and plugins."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any

import pytest

from dispatchatlas.core import objectives as registry
from dispatchatlas.core.model import (
    Dependency,
    Link,
    Mode,
    Network,
    Problem,
    Resource,
    Task,
)
from dispatchatlas.core.objectives import (
    Objective,
    evaluate,
    is_built_in,
    register_objective,
    registered_objectives,
)
from dispatchatlas.core.validation import deadline_excess
from dispatchatlas.solvers.evaluation import Evaluator, Score
from dispatchatlas.solvers.operators import random_modes, random_order, random_topological_order
from tests.helpers import chain_problem, random_problem

EVERY = tuple(item.name for item in registered_objectives() if is_built_in(item.name))

FEATURE_SETS: list[dict[str, Any]] = [
    {},
    {"pools": 2, "max_modes": 3},
    {"network": True},
    {"network": True, "pools": 0, "max_modes": 2},
    {"setups": True, "deadlines": True},
    {"setups": True, "network": True, "deadlines": True, "pools": 2},
    {"releases": False, "lags": False, "machines": 1, "pools": 0},
]


def _enriched(seed: int, **options: Any) -> Problem:
    """A random problem with powers, costs, group weights, loose tasks, and zero durations."""
    rng = random.Random(seed)
    base = random_problem(seed, **options)
    resources = []
    for resource in base.resources:
        idle = rng.uniform(0, 5)
        resources.append(
            replace(
                resource,
                idle_power=idle,
                busy_power=idle + rng.uniform(0, 20),
                cost_rate=rng.uniform(0, 2),
            )
        )
    weights = {f"g{k}": rng.choice([0.5, 1.0, 2.0, 3.0]) for k in range(4)}
    tasks = []
    for task in base.tasks:
        modes = task.modes
        if not options.get("setups") and rng.random() < 0.15:
            modes = tuple(replace(mode, duration=0.0) for mode in modes)
        group = task.group if rng.random() < 0.75 else None
        weight = weights[group] if group is not None else rng.choice([0.0, 1.0, 4.0])
        due = task.due if task.due is not None or rng.random() < 0.5 else float(rng.randint(1, 30))
        tasks.append(replace(task, modes=modes, group=group, weight=weight, due=due))
    return replace(base, resources=tuple(resources), tasks=tuple(tasks), objectives=EVERY)


def _flow_shop(seed: int) -> Problem:
    """A permutation flow shop with zero processing times, due dates, and powers."""
    rng = random.Random(seed)
    jobs, machines = 5, 3
    tasks, dependencies = [], []
    for j in range(jobs):
        for m in range(machines):
            tasks.append(
                Task.on(
                    f"j{j}m{m}",
                    f"m{m}",
                    float(rng.choice([0, 0, 1, 3, 7])),
                    group=f"j{j}",
                    weight=float(j + 1),
                    due=float(rng.randint(5, 25)) if m == machines - 1 else None,
                )
            )
            if m:
                dependencies.append(Dependency(f"j{j}m{m - 1}", f"j{j}m{m}"))
    return Problem(
        f"flow-{seed}",
        tasks=tuple(tasks),
        resources=tuple(
            Resource(f"m{m}", idle_power=1.0 + m, busy_power=4.0 + m, cost_rate=0.5)
            for m in range(machines)
        ),
        dependencies=tuple(dependencies),
        permutation=True,
        objectives=EVERY,
    )


def _modes(problem: Problem, rng: random.Random) -> list[int] | None:
    """Explicit modes, builder-chosen modes, or a mix of both."""
    draw = rng.random()
    if draw < 0.3:
        return None
    modes = random_modes(problem, rng)
    if draw < 0.6:
        modes = [mode if rng.random() < 0.5 else -1 for mode in modes]
    return modes


def _assert_close(fast: dict[str, float], reference: dict[str, float]) -> None:
    assert list(fast) == list(reference)
    for name, value in reference.items():
        assert math.isclose(fast[name], value, rel_tol=1e-9, abs_tol=1e-9), (name, fast, value)


# ------------------------------------------------------- complete schedules


@pytest.mark.parametrize("options", FEATURE_SETS)
def test_fast_objectives_equal_the_registry_on_complete_schedules(
    options: dict[str, Any],
) -> None:
    for seed in range(10):
        problem = _enriched(seed, tasks=10, **options)
        evaluator = Evaluator(problem)
        rng = random.Random(seed)
        for _ in range(4):
            decoded = evaluator.decoder.decode(random_order(problem, rng), _modes(problem, rng))
            schedule = decoded.schedule(problem)
            reference = evaluate(problem, schedule)
            _assert_close(evaluator.values(decoded), reference)
            vector = evaluator.vector(decoded)
            assert vector == pytest.approx(tuple(reference[name] for name in EVERY))
            score = evaluator.score(decoded)
            assert score.violation == pytest.approx(deadline_excess(problem, schedule))
            assert score.value == pytest.approx(reference[problem.objectives[0]])


def test_fast_objectives_equal_the_registry_on_permutation_flow_shops() -> None:
    for seed in range(8):
        problem = _flow_shop(seed)
        evaluator = Evaluator(problem, "total_tardiness")
        rng = random.Random(seed)
        for _ in range(4):
            decoded = evaluator.decoder.decode(random_order(problem, rng))
            reference = evaluate(problem, decoded.schedule(problem))
            _assert_close(evaluator.values(decoded), reference)
            assert evaluator.score(decoded).value == pytest.approx(reference["total_tardiness"])


# -------------------------------------------------------- partial schedules


def _restricted(problem: Problem, placed: set[int]) -> Problem:
    """The problem made of the placed tasks and the dependencies among them."""
    tasks = tuple(task for i, task in enumerate(problem.tasks) if i in placed)
    names = {task.id for task in tasks}
    return replace(
        problem,
        tasks=tasks,
        dependencies=tuple(
            d for d in problem.dependencies if d.before in names and d.after in names
        ),
        setups=None,
    )


@pytest.mark.parametrize("options", FEATURE_SETS)
def test_partial_decodes_score_only_the_placed_tasks(options: dict[str, Any]) -> None:
    for seed in range(8):
        problem = _enriched(seed, tasks=10, **{**options, "releases": False})
        evaluator = Evaluator(problem)
        rng = random.Random(seed)
        for _ in range(4):
            prefix = random_topological_order(problem, rng)[: rng.randint(1, problem.size - 1)]
            decoded = evaluator.decoder.decode(prefix, _modes(problem, rng))
            assert sorted(decoded.sequence) == sorted(prefix)
            restricted = _restricted(problem, set(prefix))
            reference = evaluate(restricted, decoded.schedule(problem))
            _assert_close(evaluator.values(decoded), reference)


def test_partial_decodes_add_no_phantom_transfers() -> None:
    problem = Problem(
        "phantom",
        tasks=(Task.on("a", "edge", 1), Task.on("b", "cloud", 1), Task.on("c", "cloud", 1)),
        resources=(Resource("edge", tier="e"), Resource("cloud", tier="c", cost_rate=2.0)),
        dependencies=(Dependency("a", "b", data=10), Dependency("a", "c", data=5)),
        network=Network((Link("e", "c", bandwidth=5, energy_per_unit=0.5, cost_per_unit=1.0),)),
        objectives=("cost", "energy", "makespan"),
    )
    evaluator = Evaluator(problem)
    only_a = evaluator.decoder.decode([0])
    assert evaluator.values(only_a) == {"cost": 0.0, "energy": 0.0, "makespan": 1.0}
    a_and_b = evaluator.decoder.decode([0, 1])
    # a -> b crosses the network (10 units); a -> c has no placed successor.
    assert evaluator.values(a_and_b) == {"cost": 2.0 + 10.0, "energy": 5.0, "makespan": 4.0}
    complete = evaluator.decoder.decode([0, 1, 2])
    assert evaluator.values(complete) == pytest.approx(
        evaluate(problem, complete.schedule(problem))
    )
    assert evaluator.values(complete)["cost"] == 2.0 + 2.0 + 15.0


def test_an_empty_decode_scores_zero() -> None:
    problem = _enriched(3, tasks=8, deadlines=True)
    evaluator = Evaluator(problem)
    empty = evaluator.decoder.decode([])
    assert evaluator.values(empty) == dict.fromkeys(EVERY, 0.0)
    assert evaluator.score(empty) == Score(0.0, 0.0)


# ------------------------------------------------------------------ scoring


def test_scores_compare_violation_first() -> None:
    assert Score(0.0, 10.0) < Score(0.0, 11.0) < Score(0.5, 0.0) < Score(1.0, -5.0)
    assert Score(0.0, 3.0).feasible
    assert not Score(1e-9, 3.0).feasible


def test_the_score_prices_deadline_overruns() -> None:
    problem = Problem(
        "late",
        tasks=(Task.on("a", "m", 4, deadline=3), Task.on("b", "m", 2, deadline=5)),
        resources=(Resource("m"),),
    )
    evaluator = Evaluator(problem)
    score, decoded = evaluator.evaluate([0, 1])
    assert decoded.ends == (4.0, 6.0)
    assert score == Score(1.0 + 1.0, 6.0)
    assert not score.feasible
    score, _ = evaluator.evaluate([1, 0])
    assert score == Score(3.0, 6.0)
    assert evaluator.evaluations == 2
    evaluator.score(decoded)
    assert evaluator.evaluations == 2


def test_the_primary_objective_defaults_to_the_first_one() -> None:
    problem = chain_problem().with_objectives("total_tardiness", "makespan")
    assert Evaluator(problem).objective.name == "total_tardiness"
    assert Evaluator(problem, "makespan").objective.name == "makespan"
    assert Evaluator(problem, "energy").objective.name == "energy"
    assert Evaluator(problem).decoder.insertion
    assert not Evaluator(problem, insertion=False).decoder.insertion


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))


@pytest.mark.usefixtures("scratch_registry")
def test_plugin_objectives_use_their_registered_function() -> None:
    def placed(problem: Problem, schedule: Any) -> float:
        return float(len(schedule))

    register_objective(Objective("placed", "max", "count", "Tasks placed.", placed))
    problem = chain_problem().with_objectives("placed", "makespan")
    evaluator = Evaluator(problem)
    partial = evaluator.decoder.decode([0, 2])
    assert evaluator.values(partial) == {"placed": 2.0, "makespan": 4.0}
    assert evaluator.vector(partial) == (-2.0, 4.0)
    assert evaluator.score(partial) == Score(0.0, -2.0)


@pytest.mark.usefixtures("scratch_registry")
def test_a_plugin_that_replaces_a_built_in_objective_is_used() -> None:
    doubled = Objective(
        "makespan", "min", "time", "Twice the makespan.", lambda p, s: 2 * s.makespan
    )
    register_objective(doubled, replace=True)
    evaluator = Evaluator(chain_problem())
    score, decoded = evaluator.evaluate([0, 1, 2])
    assert decoded.makespan == 5.0
    assert score == Score(0.0, 10.0)
    assert evaluator.values(decoded) == {"makespan": 10.0}


def test_zero_duration_modes_count_no_busy_energy_or_cost() -> None:
    problem = Problem(
        "instant",
        tasks=(Task("a", (Mode(0, {"m": 1}),)), Task.on("b", "m", 2)),
        resources=(Resource("m", idle_power=1, busy_power=3, cost_rate=5),),
        objectives=("energy", "cost"),
    )
    evaluator = Evaluator(problem)
    decoded = evaluator.decoder.decode([0, 1])
    assert evaluator.values(decoded) == {"energy": 1 * 2 + 2 * 2, "cost": 5 * 2}
    assert evaluator.values(decoded) == evaluate(problem, decoded.schedule(problem))
