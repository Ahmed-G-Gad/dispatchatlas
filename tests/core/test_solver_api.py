"""The solver contract: stop criteria, solver metadata, results, and the protocol."""

from __future__ import annotations

import copy
import dataclasses
import math
import pickle
import re
from typing import Any

import pytest

from dispatchatlas.benchmarks import FamilyInfo
from dispatchatlas.benchmarks import default_registry as family_registry
from dispatchatlas.core import BoundPoint as ExportedBoundPoint
from dispatchatlas.core._frozen import FrozenMap
from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import FEATURES, Problem
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.serialization import digest, schedule_to_dict
from dispatchatlas.core.solver import (
    SCHEDULES,
    SOLVER_FAMILIES,
    BoundPoint,
    FrontPoint,
    Solver,
    SolveResult,
    SolverInfo,
    Status,
    StopCriteria,
    StopReason,
    TracePoint,
)
from dispatchatlas.solvers import default_registry as solver_registry
from dispatchatlas.solvers import solve
from dispatchatlas.solvers.rules import InputOrder
from tests.helpers import chain_problem, random_problem

# ------------------------------------------------------------ stop criteria


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"time_limit": 0}, "time_limit must be a positive number of seconds"),
        ({"time_limit": -1.5}, "time_limit must be a positive number of seconds"),
        ({"time_limit": math.nan}, "time_limit must be a positive number of seconds"),
        ({"time_limit": True}, "time_limit must be a positive number of seconds"),
        ({"time_limit": "5"}, "time_limit must be a positive number of seconds"),
        ({"time_limit": math.inf}, "time_limit must be finite"),
        ({"max_evaluations": 0}, "max_evaluations must be a positive integer"),
        ({"max_evaluations": 2.0}, "max_evaluations must be a positive integer"),
        ({"max_evaluations": True}, "max_evaluations must be a positive integer"),
        ({"max_iterations": -3}, "max_iterations must be a positive integer"),
        ({"max_iterations": "3"}, "max_iterations must be a positive integer"),
        ({"patience": 0}, "patience must be a positive integer"),
        ({"patience": False}, "patience must be a positive integer"),
        ({"target": True}, "target must be a number"),
        ({"target": "7"}, "target must be a number"),
        ({"target": math.nan}, "target must be a number"),
    ],
)
def test_stop_criteria_reject_invalid_values(values: dict[str, Any], message: str) -> None:
    with pytest.raises(ModelError, match=re.escape(message)):
        StopCriteria(**values)


def test_stop_criteria_normalise_numbers() -> None:
    stop = StopCriteria(time_limit=2, max_evaluations=10, max_iterations=3, target=-4, patience=1)
    assert stop.time_limit == 2.0
    assert isinstance(stop.time_limit, float)
    assert stop.target == -4.0
    assert isinstance(stop.target, float)
    assert StopCriteria(target=math.inf).target == math.inf
    with pytest.raises(dataclasses.FrozenInstanceError):
        stop.time_limit = 5.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "bounded"),
    [
        ({}, False),
        ({"target": 3.0}, False),
        ({"patience": 5}, False),
        ({"target": 3.0, "patience": 5}, False),
        ({"time_limit": 1.0}, True),
        ({"max_evaluations": 1}, True),
        ({"max_iterations": 1}, True),
        ({"max_iterations": 1, "patience": 2}, True),
    ],
)
def test_bounded_means_a_time_evaluation_or_iteration_limit(
    values: dict[str, Any], bounded: bool
) -> None:
    assert StopCriteria(**values).bounded is bounded


def test_stop_criteria_round_trip_through_dicts() -> None:
    assert StopCriteria().to_dict() == {}
    stop = StopCriteria(time_limit=1.5, max_evaluations=7, max_iterations=3, target=2, patience=4)
    assert stop.to_dict() == {
        "time_limit": 1.5,
        "max_evaluations": 7,
        "max_iterations": 3,
        "target": 2.0,
        "patience": 4,
    }
    assert StopCriteria.from_dict(stop.to_dict()) == stop
    assert StopCriteria.from_dict({"max_evaluations": 5}) == StopCriteria(max_evaluations=5)
    with pytest.raises(ModelError, match=re.escape("unknown stop criteria: ['budget', 'seed']")):
        StopCriteria.from_dict({"budget": 3, "seed": 1, "time_limit": 1.0})
    with pytest.raises(ModelError, match="max_iterations must be a positive integer"):
        StopCriteria.from_dict({"max_iterations": 0})


def test_progress_is_the_largest_used_fraction_of_a_set_limit() -> None:
    stop = StopCriteria(time_limit=10.0, max_evaluations=100, max_iterations=4)
    assert stop.progress(50, 1, 1.0) == 0.5
    assert stop.progress(10, 3, 1.0) == 0.75
    # By default the clock counts only when no evaluation or iteration limit is set.
    assert stop.progress(10, 1, 9.0) == 0.25
    assert stop.progress(10, 1, 9.0, deterministic=False) == 0.9
    assert stop.progress(500, 9, 99.0) == 1.0
    assert StopCriteria(time_limit=10.0).progress(0, 0, 2.5) == 0.25
    assert StopCriteria(target=1.0, patience=3).progress(10**6, 10**6, 1e6) == 0.0
    assert StopCriteria().progress(0, 0, 0.0) == 0.0


def test_progress_follows_the_schedule_setting() -> None:
    assert SCHEDULES == ("deterministic", "binding")
    both = StopCriteria(time_limit=10.0, max_evaluations=1000)
    assert both.schedule == "deterministic"
    assert both.progress(100, 0, 9.0) == 0.1
    assert both.schedule_clock(100, 0, 9.0) == "evaluations"
    binding = StopCriteria(time_limit=10.0, max_evaluations=1000, schedule="binding")
    assert binding.progress(100, 0, 9.0) == 0.9
    assert binding.schedule_clock(100, 0, 9.0) == "time"
    assert binding.schedule_clock(950, 0, 9.0) == "evaluations"
    # The keyword overrides the setting for one reading.
    assert binding.progress(100, 0, 9.0, deterministic=True) == 0.1
    assert both.progress(100, 0, 9.0, deterministic=False) == 0.9
    assert both.schedule_clock(100, 0, 9.0, deterministic=False) == "time"
    # With the time limit as the only limit both settings read the clock.
    for schedule in SCHEDULES:
        timed = StopCriteria(time_limit=10.0, schedule=schedule)  # type: ignore[arg-type]
        assert timed.progress(100, 0, 9.0) == 0.9
        assert timed.schedule_clock(100, 0, 9.0) == "time"
    counted = StopCriteria(max_evaluations=10, max_iterations=4)
    assert counted.schedule_clock(5, 3, 0.0) == "iterations"
    assert counted.schedule_clock(5, 2, 0.0) == "evaluations"
    assert StopCriteria(target=1.0).schedule_clock(5, 2, 1.0) is None


def test_the_schedule_setting_says_whether_a_schedule_reads_the_clock() -> None:
    assert StopCriteria(time_limit=1.0).schedule_timed
    assert StopCriteria(time_limit=1.0, max_evaluations=5, schedule="binding").schedule_timed
    assert not StopCriteria(time_limit=1.0, max_evaluations=5).schedule_timed
    assert not StopCriteria(time_limit=1.0, max_iterations=5).schedule_timed
    assert not StopCriteria(max_evaluations=5, schedule="binding").schedule_timed


def test_the_schedule_is_written_only_when_it_is_not_the_default() -> None:
    assert StopCriteria(time_limit=1.0, max_evaluations=5).to_dict() == {
        "time_limit": 1.0,
        "max_evaluations": 5,
    }
    explicit = StopCriteria(max_evaluations=5, schedule="deterministic")
    assert explicit.to_dict() == {"max_evaluations": 5}
    binding = StopCriteria(time_limit=1.0, max_evaluations=5, schedule="binding")
    assert binding.to_dict() == {"time_limit": 1.0, "max_evaluations": 5, "schedule": "binding"}
    assert StopCriteria.from_dict(binding.to_dict()) == binding
    assert StopCriteria.from_dict({"max_evaluations": 5}).schedule == "deterministic"
    assert binding != StopCriteria(time_limit=1.0, max_evaluations=5)
    # The setting bounds nothing.
    assert not StopCriteria(schedule="binding").bounded
    for value in ("clock", "", None, 1):
        with pytest.raises(ModelError, match="schedule must be one of deterministic, binding"):
            StopCriteria(max_evaluations=5, schedule=value)  # type: ignore[arg-type]


# --------------------------------------------------------------- solver info


def _info(**changes: Any) -> SolverInfo:
    fields: dict[str, Any] = {
        "id": "toy-solver",
        "name": "Toy solver",
        "family": "local-search",
        "summary": "Does nothing useful.",
    }
    fields.update(changes)
    return SolverInfo(**fields)


def test_solver_info_defaults_and_normalisation() -> None:
    info = _info()
    assert (info.stochastic, info.iterative, info.multi_objective) == (True, True, False)
    assert info.supports == frozenset(FEATURES)
    assert (info.references, info.requires) == ((), ())
    assert info.parameters == {}
    parameters = {"rate": 0.5}
    shaped = _info(
        id="cp-sat.v9",
        references=["kolisch1996serial"],
        supports={"precedence"},
        requires=["exact"],
        parameters=parameters,
    )
    parameters["rate"] = 0.9
    assert shaped.references == ("kolisch1996serial",)
    assert shaped.supports == frozenset({"precedence"})
    assert shaped.requires == ("exact",)
    assert shaped.parameters == {"rate": 0.5}
    assert isinstance(shaped.parameters, FrozenMap)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": "Toy"}, "solver id 'Toy' must be lowercase kebab case"),
        ({"id": "toy solver"}, "solver id 'toy solver' must be lowercase kebab case"),
        ({"id": "toy--solver"}, "must be lowercase kebab case"),
        ({"id": "-toy"}, "must be lowercase kebab case"),
        ({"id": "toy_solver"}, "must be lowercase kebab case"),
        ({"id": ""}, "must be lowercase kebab case"),
        ({"family": "Local Search"}, "solver family 'Local Search' must be lowercase kebab case"),
        ({"name": ""}, "solver 'toy-solver' needs a name and a summary"),
        ({"summary": ""}, "solver 'toy-solver' needs a name and a summary"),
        ({"supports": {"precedence", "teleport"}}, "declares unknown features ['teleport']"),
    ],
)
def test_solver_info_rejects_invalid_metadata(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ModelError, match=re.escape(message)):
        _info(**changes)


def test_accepts_and_unsupported_compare_features() -> None:
    problem = random_problem(2, network=True, setups=True, deadlines=True)
    assert {"setups", "network", "precedence"} <= problem.features
    everything = _info()
    assert everything.accepts(problem)
    assert everything.unsupported(problem) == frozenset()
    narrow = _info(supports={"precedence", "groups"})
    assert not narrow.accepts(problem)
    assert narrow.unsupported(problem) == problem.features - {"precedence", "groups"}
    single = Problem("one", tasks=chain_problem().tasks[:1], resources=chain_problem().resources)
    assert single.features == frozenset()
    assert _info(supports=frozenset()).accepts(single)


def test_solver_families_are_documented_kebab_case_names() -> None:
    assert "matheuristic" in SOLVER_FAMILIES
    for family, description in SOLVER_FAMILIES.items():
        _info(family=family)
        assert description.endswith(".")
    with pytest.raises(TypeError):
        SOLVER_FAMILIES["new"] = "Not allowed."  # type: ignore[index]


# ------------------------------------------------------------------ results


def _result(**changes: Any) -> SolveResult:
    schedule = Schedule.of([Assignment("a", 0, 0.0, 3.0)])
    fields: dict[str, Any] = {
        "solver": "toy-solver",
        "problem": "p",
        "objective": "makespan",
        "status": Status.FEASIBLE,
        "schedule": schedule,
        "objectives": {"makespan": 10.0, "energy": 4.0},
        "stop_reason": StopReason.COMPLETED,
        "elapsed": 0.25,
    }
    fields.update(changes)
    return SolveResult(**fields)


def test_result_defaults() -> None:
    result = _result()
    assert (result.evaluations, result.iterations) == (0, 0)
    assert (result.bound, result.trace, result.front, result.extras) == (None, (), (), {})
    assert result.value == 10.0
    assert _result(objective="energy").value == 4.0


@pytest.mark.parametrize(
    ("status", "feasible"),
    [
        (Status.OPTIMAL, True),
        (Status.FEASIBLE, True),
        (Status.VIOLATED, False),
        (Status.INFEASIBLE, False),
        (Status.UNKNOWN, False),
    ],
)
def test_result_feasibility_follows_the_status(status: Status, feasible: bool) -> None:
    assert _result(status=status).feasible is feasible


def test_result_without_a_schedule_has_no_value_or_gap() -> None:
    result = _result(status=Status.UNKNOWN, schedule=None, objectives={}, bound=3.0)
    assert result.value is None
    assert result.gap is None


@pytest.mark.parametrize(
    ("value", "bound", "gap"),
    [
        (10.0, None, None),
        (10.0, 8.0, 0.2),
        (10.0, 10.0, 0.0),
        (-10.0, -12.0, 0.2),
        (0.0, 0.0, 0.0),
        (0.0, 1e-12, 1e-3),
    ],
)
def test_gap_is_relative_to_the_value(value: float, bound: float | None, gap: float | None) -> None:
    result = _result(objectives={"makespan": value}, bound=bound)
    if gap is None:
        assert result.gap is None
    else:
        assert result.gap == pytest.approx(gap)


def test_trace_and_front_points() -> None:
    point = TracePoint(0.5, 12, 3, 42.0)
    assert point.feasible
    assert FrontPoint((1.0, 2.0)).schedule is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        point.value = 1.0  # type: ignore[misc]


def test_bound_points_and_the_bounds_of_a_result() -> None:
    point = BoundPoint(0.5, 42.0)
    assert (point.elapsed, point.value) == (0.5, 42.0)
    assert ExportedBoundPoint is BoundPoint
    with pytest.raises(dataclasses.FrozenInstanceError):
        point.value = 1.0  # type: ignore[misc]
    assert _result().bounds == ()
    points = (point, BoundPoint(1.5, 44.0))
    assert _result(bounds=points).bounds == points
    assert _result(bounds=points) != _result()


def test_status_and_stop_reason_values_are_stable() -> None:
    assert [reason.value for reason in StopReason] == [
        "completed",
        "time-limit",
        "evaluation-limit",
        "iteration-limit",
        "target-reached",
        "stagnation",
    ]
    assert [status.value for status in Status] == [
        "optimal",
        "feasible",
        "violated",
        "infeasible",
        "unknown",
    ]
    assert StopReason("time-limit") is StopReason.TIME_LIMIT
    assert Status("violated") == "violated"


# ----------------------------------------------------------------- protocol


def test_the_solver_protocol_is_structural() -> None:
    assert isinstance(InputOrder(), Solver)
    assert not isinstance(object(), Solver)

    class Plugin:
        info = _info(id="plugin-solver")

        def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
            return _result(solver=self.info.id, problem=problem.id)

    plugin = Plugin()
    assert isinstance(plugin, Solver)
    assert plugin.solve(chain_problem(), StopCriteria()).problem == "chain"


# ------------------------------------------------------------------ pickling


def test_results_and_their_schedules_survive_pickle_and_deep_copies() -> None:
    problem = random_problem(4, tasks=8, deadlines=True)
    result = solve(problem, "nsga-ii", max_evaluations=60, seed=2)
    assert result.schedule is not None
    assert result.front
    for schedule in (result.schedule, *(point.schedule for point in result.front)):
        assert schedule is not None
        assert schedule.by_task
    for again in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
        assert again == result
        assert again.schedule is not None
        assert again.schedule.by_task == result.schedule.by_task
        assert digest(schedule_to_dict(again.schedule)) == digest(schedule_to_dict(result.schedule))
        assert again.front == result.front
        assert (again.status, again.stop_reason) == (result.status, result.stop_reason)


def test_results_with_bounds_survive_pickle_and_deep_copies() -> None:
    result = _result(bound=48.5, bounds=(BoundPoint(0.1, 40.0), BoundPoint(0.4, 48.5)))
    for again in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
        assert again == result
        assert again.bounds == result.bounds


def test_solver_and_family_metadata_pickle() -> None:
    infos: list[SolverInfo | FamilyInfo] = [
        *solver_registry(plugins=False).infos(),
        *family_registry(plugins=False).infos(),
    ]
    for info in infos:
        again = pickle.loads(pickle.dumps(info))
        assert again == info
        assert again.parameters == info.parameters
        assert copy.deepcopy(info) == info
