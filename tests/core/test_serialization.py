"""JSON formats and canonical hashing: round trips, wire rules, and stable digests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas.core.errors import SerializationError
from dispatchatlas.core.model import (
    Dependency,
    Link,
    Mode,
    Network,
    Problem,
    Resource,
    SetupTimes,
    Task,
)
from dispatchatlas.core.objectives import evaluate
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.serialization import (
    PROBLEM_FORMAT,
    RESULT_FORMAT,
    SCHEDULE_FORMAT,
    canonical_json,
    digest,
    dumps_problem,
    load_problem,
    load_result,
    load_schedule,
    loads_problem,
    problem_digest,
    problem_from_dict,
    problem_to_dict,
    result_from_dict,
    result_to_dict,
    save_problem,
    save_result,
    save_schedule,
    schedule_from_dict,
    schedule_to_dict,
)
from dispatchatlas.core.solver import (
    FrontPoint,
    SolveResult,
    Status,
    StopCriteria,
    StopReason,
    TracePoint,
)
from dispatchatlas.solvers.rules import InputOrder
from tests.helpers import chain_problem, random_problem

ROOT = Path(__file__).resolve().parents[2]

RICH_DIGEST = "7e8a193e73eb0bd60841bc1cb3feacdb4c835921add4d6f2ba80c2dbaf432d5b"
"""SHA-256 of the canonical form of :func:`_rich_problem`. Changing it breaks stored digests."""

TINY_DIGEST = "43c716265f267f7debaf72c233f6198c77ea5af135b8f35a6233ebdb8de3191e"
"""SHA-256 of the canonical form of :func:`_tiny_problem`."""


def _tiny_problem() -> Problem:
    return Problem("tiny", tasks=(Task.on("t", "cpu", 1.0),), resources=(Resource("cpu"),))


def _rich_problem() -> Problem:
    """A problem that sets every field of the format to a non-default value."""
    return Problem(
        "rich",
        tasks=(
            Task(
                "a",
                (
                    Mode(3, {"m1": 1, "pool": 2}, location="m1"),
                    Mode(4.5, {"m2": 1}),
                    Mode(0, location="m2"),
                ),
                release=1.5,
                due=10,
                deadline=20,
                weight=2,
                group="job",
                metadata={"label": "first", "rank": 1, "ratio": 0.5, "flag": True},
            ),
            Task.on("b", "m1", 2, group="job", weight=2),
            Task.on("c", "pool", 1, demand=0.5, due=4),
        ),
        resources=(
            Resource("m1", tier="edge", idle_power=1, busy_power=3, cost_rate=0.25),
            Resource("m2", kind="node", tier="cloud", metadata={"cores": 8}),
            Resource("pool", capacity=4),
        ),
        dependencies=(Dependency("a", "b", lag=1), Dependency("a", "c", data=6)),
        objectives=("makespan", "energy", "cost"),
        network=Network(
            (
                Link("edge", "cloud", bandwidth=2, latency=0.5, energy_per_unit=0.1),
                Link("edge", "pool", cost_per_unit=0.2),
                Link("cloud", "pool", bandwidth=8),
            )
        ),
        setups=SetupTimes({("m1", None, "b"): 1, ("m1", "a", "b"): 0.5, ("m1", "b", "a"): 2}),
        family="hand-made",
        metadata={"seed": 7, "note": "every field"},
    )


def _permutation_problem() -> Problem:
    tasks = (
        Task.on("j1m1", "m1", 3, group="j1"),
        Task.on("j1m2", "m2", 2, group="j1"),
        Task.on("j2m1", "m1", 1, group="j2"),
        Task.on("j2m2", "m2", 0, group="j2"),
    )
    return Problem(
        "flow",
        tasks=tasks,
        resources=(Resource("m1"), Resource("m2")),
        dependencies=(Dependency("j1m1", "j1m2"), Dependency("j2m1", "j2m2")),
        permutation=True,
    )


# ---------------------------------------------------------------- round trips


@pytest.mark.parametrize(
    "build", [_tiny_problem, _rich_problem, _permutation_problem, chain_problem]
)
def test_problems_round_trip_through_dicts_text_and_files(build: Any, tmp_path: Path) -> None:
    problem = build()
    assert problem_from_dict(problem_to_dict(problem)) == problem
    assert loads_problem(dumps_problem(problem)) == problem
    assert loads_problem(dumps_problem(problem, indent=None)) == problem
    path = save_problem(problem, tmp_path / "nested" / "problem.json")
    assert path == tmp_path / "nested" / "problem.json"
    assert load_problem(path) == problem
    assert load_problem(str(path)) == problem
    assert [item.name for item in path.parent.iterdir()] == ["problem.json"]
    assert problem_digest(load_problem(path)) == problem_digest(problem)


def test_generated_problems_with_every_feature_round_trip() -> None:
    for seed in range(20):
        problem = random_problem(
            seed, network=seed % 2 == 0, setups=seed % 3 == 0, deadlines=True, pools=seed % 3
        )
        again = loads_problem(dumps_problem(problem))
        assert again == problem
        assert problem_to_dict(again) == problem_to_dict(problem)


def test_the_wire_format_omits_defaults() -> None:
    assert problem_to_dict(chain_problem()) == {
        "format": "dispatchatlas.problem",
        "version": 1,
        "id": "chain",
        "objectives": ["makespan"],
        "resources": [{"id": "m1"}, {"id": "m2"}],
        "tasks": [
            {"id": "a", "modes": [{"duration": 3.0, "demands": {"m1": 1.0}}]},
            {"id": "b", "modes": [{"duration": 2.0, "demands": {"m1": 1.0}}], "due": 4.0},
            {"id": "c", "modes": [{"duration": 4.0, "demands": {"m2": 1.0}}]},
        ],
        "dependencies": [{"before": "a", "after": "b"}],
    }


def test_the_wire_format_writes_every_non_default_field() -> None:
    document = problem_to_dict(_rich_problem())
    assert document["resources"][0] == {
        "id": "m1",
        "tier": "edge",
        "idle_power": 1.0,
        "busy_power": 3.0,
        "cost_rate": 0.25,
    }
    assert document["resources"][2] == {"id": "pool", "capacity": 4.0}
    # A mode's location is written only where it is not its single demand.
    assert document["tasks"][0]["modes"] == [
        {"duration": 3.0, "demands": {"m1": 1.0, "pool": 2.0}, "location": "m1"},
        {"duration": 4.5, "demands": {"m2": 1.0}},
        {"duration": 0.0, "location": "m2"},
    ]
    assert document["network"]["links"] == [
        {"a": "edge", "b": "cloud", "bandwidth": 2.0, "latency": 0.5, "energy_per_unit": 0.1},
        {"a": "edge", "b": "pool", "cost_per_unit": 0.2},
        {"a": "cloud", "b": "pool", "bandwidth": 8.0},
    ]
    assert document["setups"] == [
        ["m1", None, "b", 1.0],
        ["m1", "a", "b", 0.5],
        ["m1", "b", "a", 2.0],
    ]
    assert document["family"] == "hand-made"
    assert document["metadata"] == {"note": "every field", "seed": 7}
    assert "permutation" not in document
    assert problem_to_dict(_permutation_problem())["permutation"] is True


def test_schedules_round_trip(tmp_path: Path) -> None:
    schedule = Schedule.of(
        [Assignment("b", 0, 3.0, 5.0), Assignment("a", 0, 0.0, 3.0), Assignment("c", 0, 0, 4)]
    )
    document = schedule_to_dict(schedule, "chain")
    assert document == {
        "format": "dispatchatlas.schedule",
        "version": 1,
        "problem": "chain",
        "assignments": [["a", 0, 0.0, 3.0], ["c", 0, 0, 4], ["b", 0, 3.0, 5.0]],
    }
    assert "problem" not in schedule_to_dict(schedule)
    assert schedule_from_dict(json.loads(json.dumps(document))) == schedule
    path = save_schedule(schedule, tmp_path / "schedule.json", "chain")
    assert load_schedule(path) == schedule
    assert json.loads(path.read_text(encoding="utf-8"))["problem"] == "chain"


def test_a_solver_result_survives_a_json_round_trip() -> None:
    problem = chain_problem().with_objectives("makespan", "total_tardiness")
    stop = StopCriteria(time_limit=5, max_evaluations=10, target=1.5, patience=2)
    result = InputOrder().solve(problem, stop, seed=1)
    assert result.schedule is not None
    text = json.dumps(
        {
            "schedule": schedule_to_dict(result.schedule, result.problem),
            "stop": stop.to_dict(),
            "status": result.status.value,
            "stop_reason": result.stop_reason.value,
            "objectives": dict(result.objectives),
            "trace": [dataclasses.astuple(point) for point in result.trace],
        },
        allow_nan=False,
    )
    document = json.loads(text)
    schedule = schedule_from_dict(document["schedule"])
    assert schedule == result.schedule
    assert evaluate(problem, schedule) == document["objectives"] == result.objectives
    assert StopCriteria.from_dict(document["stop"]) == stop
    assert Status(document["status"]) is result.status
    assert StopReason(document["stop_reason"]) is result.stop_reason
    assert tuple(TracePoint(*row) for row in document["trace"]) == result.trace


# ------------------------------------------------------------- canonical form


def test_canonical_json_sorts_keys_and_drops_spaces() -> None:
    value = {"b": 1, "a": {"d": [1, 2.5, 0.1, 1e-7, 3.0], "c": None, "e": True}}
    assert canonical_json(value) == '{"a":{"c":null,"d":[1,2.5,0.1,1e-07,3.0],"e":true},"b":1}'
    assert canonical_json({"name": "é", "x": "☃"}) == '{"name":"\\u00e9","x":"\\u2603"}'
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


@pytest.mark.parametrize(
    "value", [math.nan, [math.inf], {"x": -math.inf}, {1, 2}, object(), {1: "a", "b": 2}]
)
def test_canonical_json_refuses_values_json_cannot_hold(value: Any) -> None:
    with pytest.raises(SerializationError, match="cannot be written as canonical JSON"):
        canonical_json(value)
    with pytest.raises(SerializationError):
        digest(value)


def test_digest_is_the_sha256_of_the_canonical_bytes() -> None:
    value = {"b": 1, "a": [1, 2.5, "x", None, True]}
    expected = hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()
    assert digest(value) == expected
    assert digest(value) == "750c50ecaa8edefe1a0aaddd25757e16de2043660da2f8782f76f4e709a7bc82"
    assert digest([]) == hashlib.sha256(b"[]").hexdigest()


def test_problem_digests_are_pinned() -> None:
    assert problem_digest(_rich_problem()) == RICH_DIGEST
    assert problem_digest(_tiny_problem()) == TINY_DIGEST
    assert problem_digest(_tiny_problem()) == digest(
        {
            "format": PROBLEM_FORMAT,
            "version": 1,
            "id": "tiny",
            "objectives": ["makespan"],
            "resources": [{"id": "cpu"}],
            "tasks": [{"id": "t", "modes": [{"duration": 1.0, "demands": {"cpu": 1.0}}]}],
        }
    )


def test_digests_ignore_construction_order_and_number_spelling() -> None:
    forward = Problem(
        "order",
        tasks=(Task("a", (Mode(1, {"x": 1, "y": 2}),), metadata={"p": 1, "q": 2}),),
        resources=(Resource("x"), Resource("y", capacity=2)),
        setups=SetupTimes({("x", None, "a"): 1, ("x", "a", "a"): 2}),
    )
    backward = Problem(
        "order",
        tasks=(Task("a", (Mode(1.0, {"y": 2.0, "x": 1}),), metadata={"q": 2, "p": 1}),),
        resources=(Resource("x", capacity=1), Resource("y", capacity=2.0)),
        setups=SetupTimes({("x", "a", "a"): 2.0, ("x", None, "a"): 1}),
    )
    assert problem_digest(forward) == problem_digest(backward)
    assert problem_digest(forward) != problem_digest(forward.renamed("other"))


def test_digests_are_stable_across_processes() -> None:
    script = (
        "from tests.core.test_serialization import _rich_problem, _tiny_problem\n"
        "from dispatchatlas.core.serialization import problem_digest\n"
        "print(problem_digest(_rich_problem()), problem_digest(_tiny_problem()))\n"
    )
    for hash_seed in ("1", "4242"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        output = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert output == [RICH_DIGEST, TINY_DIGEST]


# ------------------------------------------------------------------- errors


def _document(**changes: Any) -> dict[str, Any]:
    document = json.loads(json.dumps(problem_to_dict(_rich_problem())))
    document.update(changes)
    return {key: value for key, value in document.items() if value is not None}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"format": SCHEDULE_FORMAT}, "expected format 'dispatchatlas.problem'"),
        ({"format": None}, "expected format 'dispatchatlas.problem', found None"),
        ({"version": 2}, "unsupported dispatchatlas.problem version 2"),
        ({"version": None}, "unsupported dispatchatlas.problem version None"),
        ({"version": "1"}, "unsupported dispatchatlas.problem version '1'"),
        ({"tasks": None}, "malformed problem document: KeyError('tasks')"),
        ({"tasks": "abc"}, "malformed problem document"),
        ({"setups": [["m1", None]]}, "malformed problem document"),
        ({"setups": [["m1", None, "b", "slow"]]}, "setup time ('m1', None, 'b') must be a number"),
        ({"network": {"links": [{"a": "edge"}]}}, "malformed problem document: KeyError('b')"),
        ({"objectives": ["makespan", "speed"]}, "unknown objective 'speed'"),
        ({"id": ""}, "problem id must be a non-empty string"),
        ({"dependencies": [{"before": "a", "after": "a"}]}, "task 'a' cannot depend on itself"),
    ],
)
def test_invalid_problem_documents_raise_serialization_errors(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(SerializationError) as caught:
        problem_from_dict(_document(**changes))
    assert message in str(caught.value)


def test_problem_text_and_files_must_hold_a_json_object(tmp_path: Path) -> None:
    with pytest.raises(SerializationError, match="invalid JSON"):
        loads_problem("{")
    with pytest.raises(SerializationError, match="must be a JSON object"):
        loads_problem("[]")
    with pytest.raises(SerializationError, match="must be a JSON object"):
        problem_from_dict([])  # type: ignore[arg-type]
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(SerializationError, match="invalid JSON"):
        load_problem(path)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"format": PROBLEM_FORMAT, "version": 1}, "expected format 'dispatchatlas.schedule'"),
        ({"format": SCHEDULE_FORMAT, "version": 0}, "unsupported dispatchatlas.schedule version"),
        ({"format": SCHEDULE_FORMAT, "version": 1}, "malformed schedule document: KeyError"),
        ({"format": SCHEDULE_FORMAT, "version": 1, "assignments": [["a", 0]]}, "IndexError"),
        (
            {"format": SCHEDULE_FORMAT, "version": 1, "assignments": [["a", "x", 0, 1]]},
            "ValueError",
        ),
        ({"format": SCHEDULE_FORMAT, "version": 1, "assignments": 5}, "TypeError"),
    ],
)
def test_invalid_schedule_documents_raise_serialization_errors(
    document: dict[str, Any], message: str
) -> None:
    with pytest.raises(SerializationError) as caught:
        schedule_from_dict(document)
    assert message in str(caught.value)


def test_schedule_files_are_json_without_nan(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[", encoding="utf-8")
    with pytest.raises(SerializationError, match="invalid JSON"):
        load_schedule(path)
    broken = Schedule.of([Assignment("a", 0, math.nan, 3.0)])
    with pytest.raises(SerializationError, match="cannot write"):
        save_schedule(broken, tmp_path / "nan.json")
    assert not (tmp_path / "nan.json").exists()
    opaque = Schedule.of([Assignment("a", object(), 0.0, 3.0)])  # type: ignore[arg-type]
    with pytest.raises(SerializationError, match="cannot write"):
        save_schedule(opaque, tmp_path / "opaque.json")
    assert sorted(item.name for item in tmp_path.iterdir()) == ["bad.json"]


def test_infinite_bandwidth_is_omitted_and_restored() -> None:
    problem = _rich_problem()
    assert problem.network is not None
    assert math.isinf(problem.network.links[1].bandwidth)
    text = dumps_problem(problem)
    assert "Infinity" not in text
    assert "NaN" not in text
    assert loads_problem(text).network == problem.network


# ------------------------------------------------------------------- results


def _full_result(**changes: Any) -> SolveResult:
    """A result of :func:`chain_problem` that sets every field to a non-default value."""
    schedule = Schedule.of(
        [Assignment("a", 0, 0.0, 3.0), Assignment("b", 0, 3.0, 5.0), Assignment("c", 0, 0, 4)]
    )
    result = SolveResult(
        solver="hand-solver",
        problem="chain",
        objective="makespan",
        status=Status.OPTIMAL,
        schedule=schedule,
        objectives={"makespan": 5.0, "total_tardiness": 1.0},
        stop_reason=StopReason.TARGET_REACHED,
        elapsed=0.25,
        evaluations=12,
        iterations=4,
        bound=5.0,
        trace=(TracePoint(0.01, 1, 0, 7.0, False), TracePoint(0.2, 9, 3, 5.0)),
        front=(FrontPoint((5.0, 1.0), schedule), FrontPoint((6.0, 0.0))),
        extras={"backend": "hand", "nodes": 3, "ratio": 0.5, "proven": True},
    )
    return dataclasses.replace(result, **changes)


def _result_document(**changes: Any) -> dict[str, Any]:
    document = json.loads(json.dumps(result_to_dict(_full_result())))
    document.update(changes)
    return {key: value for key, value in document.items() if value != "drop"}


def test_results_round_trip_through_dicts_and_files(tmp_path: Path) -> None:
    result = _full_result()
    assert result.schedule is not None
    document = result_to_dict(result)
    assert result_from_dict(json.loads(json.dumps(document, allow_nan=False))) == result
    assert document["format"] == RESULT_FORMAT
    assert document["version"] == 1
    assert document["trace"] == [[0.01, 1, 0, 7.0, False], [0.2, 9, 3, 5.0, True]]
    assert document["schedule"] == schedule_to_dict(result.schedule, "chain")
    assert document["front"][1] == {"values": [6.0, 0.0], "schedule": None}
    assert not {"problem_digest", "stop", "seed", "parameters"} & set(document)
    path = save_result(result, tmp_path / "nested" / "result.json")
    assert load_result(path) == result
    assert load_result(str(path)) == result
    assert path.read_text(encoding="utf-8").endswith("}\n")
    empty = _full_result(schedule=None, objectives={}, status=Status.UNKNOWN, trace=(), front=())
    assert result_from_dict(result_to_dict(empty)) == empty


def test_a_result_records_how_its_run_was_made() -> None:
    problem = chain_problem()
    stop = StopCriteria(max_evaluations=10, target=4.0)
    result = InputOrder().solve(problem, stop, seed=3)
    document = result_to_dict(
        result, problem=problem, stop=stop, seed=3, parameters={"weights": (1, 2), "rule": "x"}
    )
    assert document["problem_digest"] == problem_digest(problem)
    assert StopCriteria.from_dict(document["stop"]) == stop
    assert document["seed"] == 3
    assert document["parameters"] == {"weights": [1, 2], "rule": "x"}
    assert result_from_dict(json.loads(json.dumps(document))) == result
    with pytest.raises(SerializationError, match="the result is of problem 'chain', not 'other'"):
        result_to_dict(result, problem=problem.renamed("other"))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"format": SCHEDULE_FORMAT}, "expected format 'dispatchatlas.result'"),
        ({"version": 2}, "unsupported dispatchatlas.result version 2"),
        ({"solver": "drop"}, "a result document lacks the field 'solver'"),
        ({"problem": 7}, "the problem of a result must be a string, got 7"),
        ({"status": "great"}, "status of a result must be one of optimal, feasible, violated"),
        ({"stop_reason": 3}, "the stop reason of a result must be one of completed"),
        ({"objectives": [5.0]}, "the objectives of a result must be a JSON object"),
        ({"objectives": {"makespan": "5"}}, "value of 'makespan' of a result must be a finite"),
        ({"elapsed": math.nan}, "the elapsed time of a result must be a finite number"),
        ({"evaluations": -1}, "evaluation count of a result must be a non-negative integer"),
        ({"iterations": True}, "iteration count of a result must be a non-negative integer"),
        ({"bound": "low"}, "the bound of a result must be a finite number"),
        ({"trace": [[0.1, 1, 0, 5.0]]}, "a trace row of a result lists elapsed"),
        ({"trace": [[0.1, 1, 0, 5.0, "yes"]]}, "a trace row of a result lists elapsed"),
        ({"front": [{"schedule": None}]}, "a front point lacks the field 'values'"),
        ({"front": [{"values": [1.0, None]}]}, "front point value of a result must be a finite"),
        ({"extras": {"nodes": [1]}}, "extra 'nodes' of a result must be a string, number"),
        ({"schedule": {"format": PROBLEM_FORMAT, "version": 1}}, "format 'dispatchatlas.schedule"),
    ],
)
def test_invalid_result_documents_raise_serialization_errors(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(SerializationError) as caught:
        result_from_dict(_result_document(**changes))
    assert message in str(caught.value)


def test_result_files_are_json_objects_without_nan(tmp_path: Path) -> None:
    broken = _full_result(elapsed=math.inf)
    with pytest.raises(SerializationError, match="cannot write"):
        save_result(broken, tmp_path / "inf.json")
    with pytest.raises(SerializationError, match="cannot write"):
        save_result(_full_result(), tmp_path / "opaque.json", parameters={"rule": object()})
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(SerializationError, match="invalid JSON"):
        load_result(path)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SerializationError, match="must be a JSON object"):
        load_result(path)
