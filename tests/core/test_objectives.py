"""Built-in objectives against hand-computed values, and the objective registry."""

from __future__ import annotations

import dataclasses

import pytest

from dispatchatlas.core import objectives as registry
from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Dependency, Link, Mode, Network, Problem, Resource, Task
from dispatchatlas.core.objectives import (
    TARDY_TOLERANCE,
    Objective,
    evaluate,
    get_objective,
    is_built_in,
    is_registered,
    register_objective,
    registered_objectives,
    unregister_objective,
)
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.validation import check_schedule
from tests.helpers import chain_problem

BUILT_IN = (
    "cost",
    "energy",
    "makespan",
    "max_lateness",
    "mean_response_time",
    "tardy_tasks",
    "total_completion_time",
    "total_flow_time",
    "total_tardiness",
)


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a test register objectives without leaking them into other tests."""
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))


def _problem() -> Problem:
    """Four tasks, two jobs with weights and releases, powers, costs, and a network.

    Job ``j1`` is ``a`` then ``b`` (weight 2, releases 1 and 2); ``c`` and
    ``d`` are single-task jobs of weights 3 and 1. ``a`` holds machine ``m1``
    and half of pool ``p``. ``a`` sends 10 data units to ``c``.
    """
    return Problem(
        "hand",
        tasks=(
            Task(
                "a", (Mode(3, {"m1": 1, "p": 2}, location="m1"),), release=1, group="j1", weight=2
            ),
            Task.on("b", "m1", 2, release=2, due=6, group="j1", weight=2),
            Task("c", (Mode(4, {"m2": 1}), Mode(6, {"m1": 1})), due=3, weight=3),
            Task("d", (Mode(5, {"m1": 1}), Mode(1, {"m2": 1})), due=10, deadline=12),
        ),
        resources=(
            Resource("m1", tier="edge", idle_power=2, busy_power=10, cost_rate=3),
            Resource("m2", tier="cloud", idle_power=1, busy_power=5, cost_rate=1),
            Resource("p", capacity=4, idle_power=4, busy_power=12, cost_rate=2),
            Resource("spare", idle_power=1, busy_power=1),
        ),
        dependencies=(Dependency("a", "b"), Dependency("a", "c", data=10)),
        network=Network(
            (Link("edge", "cloud", bandwidth=5, latency=1, energy_per_unit=0.5, cost_per_unit=0.1),)
        ),
        objectives=BUILT_IN,
    )


def _remote() -> Schedule:
    """``c`` runs on ``m2``, so ``a``'s data crosses the network (3 time units)."""
    return Schedule.of(
        [
            Assignment("a", 0, 1, 4),
            Assignment("b", 0, 4, 6),
            Assignment("c", 0, 7, 11),
            Assignment("d", 1, 0, 1),
        ]
    )


def _local() -> Schedule:
    """``c`` runs on ``m1`` next to ``a``, so no data crosses the network."""
    return Schedule.of(
        [
            Assignment("a", 0, 1, 4),
            Assignment("b", 0, 4, 6),
            Assignment("c", 1, 6, 12),
            Assignment("d", 1, 0, 1),
        ]
    )


def _chain_schedule() -> Schedule:
    """The optimal schedule of :func:`tests.helpers.chain_problem`."""
    return Schedule.of(
        [Assignment("a", 0, 0, 3), Assignment("b", 0, 3, 5), Assignment("c", 0, 0, 4)]
    )


def test_hand_schedules_are_feasible() -> None:
    problem = _problem()
    assert check_schedule(problem, _remote()) == ()
    assert check_schedule(problem, _local()) == ()


def test_objectives_of_the_remote_schedule() -> None:
    values = evaluate(_problem(), _remote())
    assert values == pytest.approx(
        {
            "makespan": 11.0,
            # Jobs: j1 ends at 6 (release 1, weight 2), c at 11 (weight 3), d at 1.
            "total_completion_time": 2 * 6 + 3 * 11 + 1 * 1,
            "total_flow_time": 2 * (6 - 1) + 3 * 11 + 1 * 1,
            "mean_response_time": ((6 - 1) + 11 + 1) / 3,
            # Only c is late: 11 - 3, weighted by 3. b ends exactly at its due date.
            "total_tardiness": 3 * (11 - 3),
            "max_lateness": 11 - 3,
            "tardy_tasks": 1.0,
            # Idle power of all four resources over the makespan, plus busy power
            # above idle for each held fraction, plus 0.5 J per transferred unit.
            "energy": (2 + 1 + 4 + 1) * 11
            + (10 - 2) * 3
            + (12 - 4) * (2 / 4) * 3
            + (10 - 2) * 2
            + (5 - 1) * 4
            + (5 - 1) * 1
            + 0.5 * 10,
            "cost": 3 * 3 + 2 * (2 / 4) * 3 + 3 * 2 + 1 * 4 + 1 * 1 + 0.1 * 10,
        }
    )
    assert values["energy"] == pytest.approx(165.0)
    assert values["cost"] == pytest.approx(24.0)


def test_objectives_of_the_local_schedule() -> None:
    values = evaluate(_problem(), _local())
    assert values == pytest.approx(
        {
            "makespan": 12.0,
            "total_completion_time": 2 * 6 + 3 * 12 + 1,
            "total_flow_time": 2 * 5 + 3 * 12 + 1,
            "mean_response_time": 6.0,
            "total_tardiness": 3 * 9,
            "max_lateness": 9.0,
            "tardy_tasks": 1.0,
            "energy": 8 * 12 + (8 * 3 + 8 * 0.5 * 3) + 8 * 2 + 8 * 6 + 4 * 1,
            "cost": (3 * 3 + 2 * 0.5 * 3) + 3 * 2 + 3 * 6 + 1 * 1,
        }
    )


def test_evaluate_returns_the_requested_objectives_in_order() -> None:
    problem, schedule = _problem(), _remote()
    assert list(evaluate(problem, schedule)) == list(BUILT_IN)
    assert list(evaluate(problem, schedule, ["cost", "makespan"])) == ["cost", "makespan"]
    assert evaluate(chain_problem(), _chain_schedule()) == {"makespan": 5.0}


def test_due_date_objectives_without_due_dates_are_zero() -> None:
    problem = dataclasses.replace(chain_problem(), tasks=(Task.on("a", "m1", 3),), dependencies=())
    schedule = Schedule.of([Assignment("a", 0, 0, 3)])
    names = ("total_tardiness", "max_lateness", "tardy_tasks")
    assert evaluate(problem, schedule, names) == dict.fromkeys(names, 0.0)


def test_max_lateness_can_be_negative_and_tardiness_cannot() -> None:
    problem = Problem(
        "early",
        tasks=(Task.on("a", "m", 2, due=10), Task.on("b", "m", 2, due=7)),
        resources=(Resource("m"),),
    )
    schedule = Schedule.of([Assignment("a", 0, 0, 2), Assignment("b", 0, 2, 4)])
    values = evaluate(problem, schedule, ["max_lateness", "total_tardiness", "tardy_tasks"])
    assert values == {"max_lateness": -3.0, "total_tardiness": 0.0, "tardy_tasks": 0.0}


def test_a_task_is_tardy_only_beyond_the_tolerance() -> None:
    problem = Problem("due", tasks=(Task.on("a", "m", 2, due=2),), resources=(Resource("m"),))
    for end, tardy in ((2.0, 0.0), (2.0 + TARDY_TOLERANCE / 2, 0.0), (2.0 + 1e-6, 1.0)):
        schedule = Schedule.of([Assignment("a", 0, end - 2, end)])
        assert evaluate(problem, schedule, ["tardy_tasks"]) == {"tardy_tasks": tardy}


def test_zero_weight_jobs_count_nothing_in_weighted_sums() -> None:
    problem = Problem(
        "weights",
        tasks=(Task.on("a", "m", 2, weight=0, due=0), Task.on("b", "m", 3, weight=4, due=0)),
        resources=(Resource("m"),),
    )
    schedule = Schedule.of([Assignment("a", 0, 0, 2), Assignment("b", 0, 2, 5)])
    values = evaluate(problem, schedule, ["total_completion_time", "total_tardiness"])
    assert values == {"total_completion_time": 20.0, "total_tardiness": 20.0}
    assert evaluate(problem, schedule, ["mean_response_time"]) == {"mean_response_time": 3.5}


def test_evaluate_requires_every_task_exactly_once() -> None:
    problem = chain_problem()
    partial = Schedule.of([Assignment("a", 0, 0, 3), Assignment("c", 0, 0, 4)])
    with pytest.raises(ModelError, match="schedule does not assign task 'b'"):
        evaluate(problem, partial)
    doubled = Schedule.of([*_chain_schedule().assignments, Assignment("a", 0, 5, 8)])
    with pytest.raises(ModelError, match="task 'a' is assigned more than once"):
        evaluate(problem, doubled)


# ------------------------------------------------------------------ registry


def test_the_built_in_objectives_are_registered_and_minimised() -> None:
    names = [objective.name for objective in registered_objectives()]
    assert names == sorted(names)
    assert set(BUILT_IN) <= set(names)
    for name in BUILT_IN:
        objective = get_objective(name)
        assert objective.name == name
        assert objective.sense == "min"
        assert objective.unit
        assert objective.description.endswith(".")
        assert is_registered(name)
        assert is_built_in(name)
    assert not is_registered("speed")
    assert not is_built_in("speed")


def test_unknown_objectives_are_reported_with_the_known_names() -> None:
    with pytest.raises(ModelError, match="unknown objective 'speed'; known: cost, energy"):
        get_objective("speed")
    with pytest.raises(ModelError, match="unknown objective 'speed'"):
        evaluate(chain_problem(), _chain_schedule(), ["speed"])


def test_minimized_orients_values_by_sense() -> None:
    lowest = Objective("low", "min", "unit", "Lower is better.", lambda p, s: 0.0)
    highest = Objective("high", "max", "unit", "Higher is better.", lambda p, s: 0.0)
    assert lowest.minimized(4.5) == 4.5
    assert highest.minimized(4.5) == -4.5
    assert highest.minimized(-2.0) == 2.0


@pytest.mark.usefixtures("scratch_registry")
def test_plugins_register_new_objectives() -> None:
    def busy_tasks(problem: Problem, schedule: Schedule) -> float:
        return float(sum(1 for item in schedule.assignments if item.end > item.start))

    with pytest.raises(ModelError, match="unknown objective 'busy_tasks'"):
        chain_problem().with_objectives("busy_tasks")
    register_objective(Objective("busy_tasks", "max", "count", "Tasks that take time.", busy_tasks))
    assert is_registered("busy_tasks")
    assert not is_built_in("busy_tasks")
    assert "busy_tasks" in [objective.name for objective in registered_objectives()]
    problem = chain_problem().with_objectives("makespan", "busy_tasks")
    assert evaluate(problem, _chain_schedule()) == {"makespan": 5.0, "busy_tasks": 3.0}
    assert get_objective("busy_tasks").minimized(3.0) == -3.0


@pytest.mark.usefixtures("scratch_registry")
def test_registration_refuses_duplicates_and_invalid_senses() -> None:
    constant = Objective("constant", "min", "unit", "Always one.", lambda p, s: 1.0)
    register_objective(constant)
    with pytest.raises(ModelError, match="objective 'constant' is already registered"):
        register_objective(constant)
    upward = Objective(
        "constant",
        "sideways",  # type: ignore[arg-type]
        "unit",
        "Invalid.",
        lambda p, s: 1.0,
    )
    with pytest.raises(ModelError, match="objective 'constant' has invalid sense 'sideways'"):
        register_objective(upward, replace=True)
    assert get_objective("constant") is constant


@pytest.mark.usefixtures("scratch_registry")
def test_plugins_can_override_a_built_in_objective() -> None:
    original = get_objective("makespan")
    shifted = Objective(
        "makespan", "min", "time", "Makespan plus one.", lambda p, s: s.makespan + 1.0
    )
    with pytest.raises(ModelError, match="already registered"):
        register_objective(shifted)
    register_objective(shifted, replace=True)
    assert get_objective("makespan") is shifted
    assert not is_built_in("makespan")
    assert evaluate(chain_problem(), _chain_schedule()) == {"makespan": 6.0}
    register_objective(original, replace=True)
    assert is_built_in("makespan")
    assert evaluate(chain_problem(), _chain_schedule()) == {"makespan": 5.0}


@pytest.mark.usefixtures("scratch_registry")
def test_unregistering_removes_plugins_and_restores_built_ins() -> None:
    register_objective(Objective("span", "max", "time", "The makespan.", lambda p, s: s.makespan))
    unregister_objective("span")
    assert not is_registered("span")
    with pytest.raises(ModelError, match="unknown objective 'span'"):
        unregister_objective("span")
    original = get_objective("makespan")
    with pytest.raises(ModelError, match="objective 'makespan' is built in and cannot be removed"):
        unregister_objective("makespan")
    shifted = Objective("makespan", "min", "time", "Plus one.", lambda p, s: s.makespan + 1.0)
    register_objective(shifted, replace=True)
    unregister_objective("makespan")
    assert is_built_in("makespan")
    assert get_objective("makespan") == original
    assert evaluate(chain_problem(), _chain_schedule()) == {"makespan": 5.0}


def test_the_registry_module_is_not_shadowed() -> None:
    import dispatchatlas.core.objectives as module  # noqa: PLC0415

    assert module is registry
    assert module.registered_objectives is registered_objectives
