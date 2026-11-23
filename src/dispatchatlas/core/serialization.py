"""JSON formats for problems, schedules, and solver results, and canonical hashing.

Problem documents are the interchange format of DispatchAtlas: generators
write them, campaigns store them, and every solver reads them. Fields equal to
their default are omitted, so small problems stay readable.

Example document::

    {
      "format": "dispatchatlas.problem",
      "version": 1,
      "id": "two-jobs",
      "objectives": ["makespan"],
      "resources": [{"id": "m1"}],
      "tasks": [
        {"id": "a", "modes": [{"duration": 3.0, "demands": {"m1": 1.0}}]},
        {"id": "b", "modes": [{"duration": 2.0, "demands": {"m1": 1.0}}]}
      ],
      "dependencies": [{"before": "a", "after": "b"}]
    }

A schedule document lists ``[task, mode, start, end]`` rows. A result
document holds every field of a :class:`~dispatchatlas.core.solver.SolveResult`,
its schedules as schedule documents and its trace as ``[elapsed, evaluations,
iteration, value, feasible]`` rows, and, when the writer knows them, how the
run was made: the digest of the problem, the stop criteria, the seed, and the
solver's parameters. A result that has bounds over time also holds them under
``bounds`` as ``[elapsed, value]`` rows; a result without them has no such key.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from dispatchatlas.core.errors import ModelError, SerializationError
from dispatchatlas.core.model import (
    Dependency,
    Link,
    Mode,
    Network,
    Problem,
    Resource,
    Scalar,
    SetupTimes,
    Task,
)
from dispatchatlas.core.schedule import Assignment, Schedule
from dispatchatlas.core.solver import (
    BoundPoint,
    FrontPoint,
    SolveResult,
    Status,
    StopCriteria,
    StopReason,
    TracePoint,
)

__all__ = [
    "PROBLEM_FORMAT",
    "RESULT_FORMAT",
    "SCHEDULE_FORMAT",
    "canonical_json",
    "digest",
    "dumps_problem",
    "load_problem",
    "load_result",
    "load_schedule",
    "loads_problem",
    "problem_digest",
    "problem_from_dict",
    "problem_to_dict",
    "result_from_dict",
    "result_to_dict",
    "save_problem",
    "save_result",
    "save_schedule",
    "schedule_from_dict",
    "schedule_to_dict",
]

PROBLEM_FORMAT = "dispatchatlas.problem"
SCHEDULE_FORMAT = "dispatchatlas.schedule"
RESULT_FORMAT = "dispatchatlas.result"
_VERSION = 1


def canonical_json(value: Any) -> str:
    """Serialise ``value`` to canonical JSON: sorted keys, no spaces, ASCII only.

    Raises:
        SerializationError: If ``value`` holds NaN, infinity, or a type JSON
            cannot represent.
    """
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise SerializationError(f"value cannot be written as canonical JSON: {error}") from error


def digest(value: Any) -> str:
    """Return the SHA-256 hex digest of ``value``'s canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


# ------------------------------------------------------------------ problems


def problem_to_dict(problem: Problem) -> dict[str, Any]:
    """Convert a problem to a JSON-ready dictionary."""
    document: dict[str, Any] = {
        "format": PROBLEM_FORMAT,
        "version": _VERSION,
        "id": problem.id,
        "objectives": list(problem.objectives),
        "resources": [_resource_to_dict(resource) for resource in problem.resources],
        "tasks": [_task_to_dict(task) for task in problem.tasks],
    }
    if problem.dependencies:
        document["dependencies"] = [_dependency_to_dict(d) for d in problem.dependencies]
    if problem.network is not None:
        document["network"] = {"links": [_link_to_dict(link) for link in problem.network.links]}
    if problem.setups is not None and problem.setups.times:
        document["setups"] = [
            [resource, previous, following, time]
            for (resource, previous, following), time in sorted(
                problem.setups.times.items(),
                key=lambda item: (item[0][0], item[0][1] or "", item[0][2]),
            )
        ]
    if problem.permutation:
        document["permutation"] = True
    if problem.family is not None:
        document["family"] = problem.family
    if problem.metadata:
        document["metadata"] = dict(problem.metadata)
    return document


def problem_from_dict(document: Mapping[str, Any]) -> Problem:
    """Build a problem from a dictionary produced by :func:`problem_to_dict`.

    Raises:
        SerializationError: If the document is not a problem document of a
            supported version, or its content is invalid.
    """
    _require_format(document, PROBLEM_FORMAT)
    try:
        resources = tuple(_resource_from_dict(item) for item in document["resources"])
        tasks = tuple(_task_from_dict(item) for item in document["tasks"])
        dependencies = tuple(
            _dependency_from_dict(item) for item in document.get("dependencies", ())
        )
        network = None
        if document.get("network") is not None:
            network = Network(tuple(_link_from_dict(item) for item in document["network"]["links"]))
        setups = None
        if document.get("setups"):
            setups = SetupTimes({(row[0], row[1], row[2]): row[3] for row in document["setups"]})
        return Problem(
            id=document["id"],
            tasks=tasks,
            resources=resources,
            dependencies=dependencies,
            objectives=tuple(document.get("objectives", ("makespan",))),
            network=network,
            setups=setups,
            permutation=bool(document.get("permutation", False)),
            family=document.get("family"),
            metadata=document.get("metadata", {}),
        )
    except (KeyError, TypeError, IndexError) as error:
        raise SerializationError(f"malformed problem document: {error!r}") from error
    except ModelError as error:
        raise SerializationError(str(error)) from error


def dumps_problem(problem: Problem, *, indent: int | None = 2) -> str:
    """Serialise a problem to a JSON string."""
    return json.dumps(problem_to_dict(problem), indent=indent, allow_nan=False) + "\n"


def loads_problem(text: str) -> Problem:
    """Parse a problem from a JSON string."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise SerializationError(f"invalid JSON: {error}") from error
    if not isinstance(document, Mapping):
        raise SerializationError("a problem document must be a JSON object")
    return problem_from_dict(document)


def save_problem(problem: Problem, path: str | Path, *, indent: int | None = 2) -> Path:
    """Write a problem to a JSON file and return its path."""
    from dispatchatlas.core.io import atomic_write_text  # noqa: PLC0415

    return atomic_write_text(Path(path), dumps_problem(problem, indent=indent))


def load_problem(path: str | Path) -> Problem:
    """Read a problem from a JSON file."""
    return loads_problem(Path(path).read_text(encoding="utf-8"))


def problem_digest(problem: Problem) -> str:
    """Return the SHA-256 digest of a problem's canonical JSON form."""
    return digest(problem_to_dict(problem))


# ----------------------------------------------------------------- schedules


def schedule_to_dict(schedule: Schedule, problem_id: str | None = None) -> dict[str, Any]:
    """Convert a schedule to a JSON-ready dictionary of compact rows."""
    document: dict[str, Any] = {
        "format": SCHEDULE_FORMAT,
        "version": _VERSION,
        "assignments": [
            [item.task, item.mode, item.start, item.end] for item in schedule.assignments
        ],
    }
    if problem_id is not None:
        document["problem"] = problem_id
    return document


def schedule_from_dict(document: Mapping[str, Any]) -> Schedule:
    """Build a schedule from a dictionary produced by :func:`schedule_to_dict`."""
    _require_format(document, SCHEDULE_FORMAT)
    try:
        return Schedule.of(
            Assignment(str(row[0]), int(row[1]), float(row[2]), float(row[3]))
            for row in document["assignments"]
        )
    except (KeyError, TypeError, IndexError, ValueError) as error:
        raise SerializationError(f"malformed schedule document: {error!r}") from error


def save_schedule(schedule: Schedule, path: str | Path, problem_id: str | None = None) -> Path:
    """Write a schedule to a JSON file and return its path.

    Raises:
        SerializationError: If the schedule holds a value JSON cannot represent,
            such as a non-finite time.
    """
    from dispatchatlas.core.io import atomic_write_text  # noqa: PLC0415

    try:
        text = json.dumps(schedule_to_dict(schedule, problem_id), indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SerializationError(f"cannot write {path} as JSON: {error}") from error
    return atomic_write_text(Path(path), text + "\n")


def load_schedule(path: str | Path) -> Schedule:
    """Read a schedule from a JSON file."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SerializationError(f"invalid JSON in {path}: {error}") from error
    return schedule_from_dict(document)


# ------------------------------------------------------------------- results


def result_to_dict(
    result: SolveResult,
    *,
    problem: Problem | None = None,
    stop: StopCriteria | None = None,
    seed: int | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a solver result to a JSON-ready dictionary.

    Every field of the result is written, ``bounds`` only when the result has
    bounds over time. The keyword arguments record how the run was made, which
    the result itself does not say; each is written only when given, as
    ``problem_digest``, ``stop``, ``seed``, and ``parameters``.

    Args:
        result: The result.
        problem: The problem solved, whose digest identifies it beyond its id.
        stop: The stop criteria of the run.
        seed: The seed of the run.
        parameters: The solver's parameters.

    Raises:
        SerializationError: If ``problem`` is not the problem the result names.
    """
    if problem is not None and problem.id != result.problem:
        raise SerializationError(f"the result is of problem {result.problem!r}, not {problem.id!r}")
    document: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "version": _VERSION,
        "solver": result.solver,
        "problem": result.problem,
        "objective": result.objective,
        "status": result.status.value,
        "stop_reason": result.stop_reason.value,
        "objectives": dict(result.objectives),
        "bound": result.bound,
        "elapsed": result.elapsed,
        "evaluations": result.evaluations,
        "iterations": result.iterations,
        "trace": [
            [point.elapsed, point.evaluations, point.iteration, point.value, point.feasible]
            for point in result.trace
        ],
        "front": [
            {
                "values": list(point.values),
                "schedule": None if point.schedule is None else schedule_to_dict(point.schedule),
            }
            for point in result.front
        ],
        "extras": dict(result.extras),
        "schedule": None
        if result.schedule is None
        else schedule_to_dict(result.schedule, result.problem),
    }
    if result.bounds:
        document["bounds"] = [[point.elapsed, point.value] for point in result.bounds]
    if problem is not None:
        document["problem_digest"] = problem_digest(problem)
    if stop is not None:
        document["stop"] = stop.to_dict()
    if seed is not None:
        document["seed"] = seed
    if parameters is not None:
        document["parameters"] = _plain(parameters)
    return document


def result_from_dict(document: Mapping[str, Any]) -> SolveResult:
    """Build a solver result from a dictionary produced by :func:`result_to_dict`.

    How the run was made (``problem_digest``, ``stop``, ``seed``, and
    ``parameters``) is not part of a result; read it from the document.

    Raises:
        SerializationError: If the document is not a result document of a
            supported version, or a field is missing or holds a value of the
            wrong type.
    """
    _require_format(document, RESULT_FORMAT)
    schedule = document.get("schedule")
    bound = document.get("bound")
    return SolveResult(
        solver=_text(_field(document, "solver"), "solver"),
        problem=_text(_field(document, "problem"), "problem"),
        objective=_text(_field(document, "objective"), "objective"),
        status=_member(Status, _field(document, "status"), "status"),
        schedule=None if schedule is None else schedule_from_dict(schedule),
        objectives={
            _text(name, "objective name"): _number(value, f"value of {name!r}")
            for name, value in _table(_field(document, "objectives"), "objectives").items()
        },
        stop_reason=_member(StopReason, _field(document, "stop_reason"), "stop reason"),
        elapsed=_number(_field(document, "elapsed"), "elapsed time"),
        evaluations=_count(document.get("evaluations", 0), "evaluation count"),
        iterations=_count(document.get("iterations", 0), "iteration count"),
        bound=None if bound is None else _number(bound, "bound"),
        trace=tuple(_trace_point(row) for row in _items(document.get("trace", []), "trace")),
        front=tuple(_front_point(item) for item in _items(document.get("front", []), "front")),
        extras={
            _text(name, "extra name"): _scalar(value, f"extra {name!r}")
            for name, value in _table(document.get("extras", {}), "extras").items()
        },
        bounds=tuple(_bound_point(row) for row in _items(document.get("bounds", []), "bounds")),
    )


def save_result(
    result: SolveResult,
    path: str | Path,
    *,
    problem: Problem | None = None,
    stop: StopCriteria | None = None,
    seed: int | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> Path:
    """Write a solver result to a JSON file and return its path.

    The keyword arguments are those of :func:`result_to_dict`.

    Raises:
        SerializationError: If the result holds a value JSON cannot represent,
            such as a number that is not finite, or ``problem`` does not match.
    """
    from dispatchatlas.core.io import atomic_write_text  # noqa: PLC0415

    document = result_to_dict(result, problem=problem, stop=stop, seed=seed, parameters=parameters)
    try:
        text = json.dumps(document, indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise SerializationError(f"cannot write {path} as JSON: {error}") from error
    return atomic_write_text(Path(path), text + "\n")


def load_result(path: str | Path) -> SolveResult:
    """Read a solver result from a JSON file."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SerializationError(f"invalid JSON in {path}: {error}") from error
    return result_from_dict(document)


# ------------------------------------------------------------------- helpers


def _require_format(document: Mapping[str, Any], expected: str) -> None:
    if not isinstance(document, Mapping):
        raise SerializationError(f"a {expected} document must be a JSON object")
    if document.get("format") != expected:
        raise SerializationError(f"expected format {expected!r}, found {document.get('format')!r}")
    if document.get("version") != _VERSION:
        raise SerializationError(
            f"unsupported {expected} version {document.get('version')!r}; "
            f"this release reads version {_VERSION}"
        )


def _plain(value: Any) -> Any:
    """Return ``value`` with mappings as dictionaries and sequences as lists, recursively."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _field(document: Mapping[str, Any], key: str, what: str = "a result document") -> Any:
    if key not in document:
        raise SerializationError(f"{what} lacks the field {key!r}")
    return document[key]


def _text(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise SerializationError(f"the {what} of a result must be a string, got {value!r}")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SerializationError(f"the {what} of a result must be a finite number, got {value!r}")
    return float(value)


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SerializationError(
            f"the {what} of a result must be a non-negative integer, got {value!r}"
        )
    return value


def _table(value: object, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SerializationError(f"the {what} of a result must be a JSON object, got {value!r}")
    return value


def _items(value: object, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise SerializationError(f"the {what} of a result must be a list, got {value!r}")
    return value


def _scalar(value: object, what: str) -> Scalar:
    if not isinstance(value, (str, int, float, bool)):
        raise SerializationError(
            f"the {what} of a result must be a string, number, or boolean, got {value!r}"
        )
    return value


def _member[E: StrEnum](kind: type[E], value: object, what: str) -> E:
    if isinstance(value, str):
        try:
            return kind(value)
        except ValueError:
            pass
    known = ", ".join(item.value for item in kind)
    raise SerializationError(f"the {what} of a result must be one of {known}, got {value!r}")


def _trace_point(row: object) -> TracePoint:
    """Read an ``[elapsed, evaluations, iteration, value, feasible]`` trace row."""
    items = _items(row, "trace row")
    if len(items) != 5 or not isinstance(items[4], bool):
        raise SerializationError(
            "a trace row of a result lists elapsed, evaluations, iteration, value, and "
            f"feasible, got {row!r}"
        )
    elapsed, evaluations, iteration, value, feasible = items
    return TracePoint(
        _number(elapsed, "trace time"),
        _count(evaluations, "trace evaluation count"),
        _count(iteration, "trace iteration"),
        _number(value, "trace value"),
        feasible,
    )


def _bound_point(row: object) -> BoundPoint:
    """Read an ``[elapsed, value]`` bound row."""
    items = _items(row, "bound row")
    if len(items) != 2:
        raise SerializationError(f"a bound row of a result lists elapsed and value, got {row!r}")
    elapsed, value = items
    return BoundPoint(_number(elapsed, "bound time"), _number(value, "bound value"))


def _front_point(item: object) -> FrontPoint:
    fields = _table(item, "front point")
    values = _items(_field(fields, "values", "a front point"), "front point values")
    schedule = fields.get("schedule")
    return FrontPoint(
        tuple(_number(value, "front point value") for value in values),
        None if schedule is None else schedule_from_dict(schedule),
    )


def _resource_to_dict(resource: Resource) -> dict[str, Any]:
    item: dict[str, Any] = {"id": resource.id}
    if resource.capacity != 1.0:
        item["capacity"] = resource.capacity
    if resource.kind != "machine":
        item["kind"] = resource.kind
    if resource.tier is not None:
        item["tier"] = resource.tier
    for name in ("idle_power", "busy_power", "cost_rate"):
        if getattr(resource, name) != 0.0:
            item[name] = getattr(resource, name)
    if resource.metadata:
        item["metadata"] = dict(resource.metadata)
    return item


def _resource_from_dict(item: Mapping[str, Any]) -> Resource:
    return Resource(
        id=item["id"],
        capacity=item.get("capacity", 1.0),
        kind=item.get("kind", "machine"),
        tier=item.get("tier"),
        idle_power=item.get("idle_power", 0.0),
        busy_power=item.get("busy_power", 0.0),
        cost_rate=item.get("cost_rate", 0.0),
        metadata=item.get("metadata", {}),
    )


def _mode_to_dict(mode: Mode) -> dict[str, Any]:
    item: dict[str, Any] = {"duration": mode.duration}
    if mode.demands:
        item["demands"] = dict(mode.demands)
    implied = next(iter(mode.demands)) if len(mode.demands) == 1 else None
    if mode.location is not None and mode.location != implied:
        item["location"] = mode.location
    return item


def _mode_from_dict(item: Mapping[str, Any]) -> Mode:
    return Mode(
        duration=item["duration"],
        demands=item.get("demands", {}),
        location=item.get("location"),
    )


def _task_to_dict(task: Task) -> dict[str, Any]:
    item: dict[str, Any] = {"id": task.id, "modes": [_mode_to_dict(m) for m in task.modes]}
    if task.release != 0.0:
        item["release"] = task.release
    for name in ("due", "deadline", "group"):
        if getattr(task, name) is not None:
            item[name] = getattr(task, name)
    if task.weight != 1.0:
        item["weight"] = task.weight
    if task.metadata:
        item["metadata"] = dict(task.metadata)
    return item


def _task_from_dict(item: Mapping[str, Any]) -> Task:
    return Task(
        id=item["id"],
        modes=tuple(_mode_from_dict(mode) for mode in item["modes"]),
        release=item.get("release", 0.0),
        due=item.get("due"),
        deadline=item.get("deadline"),
        weight=item.get("weight", 1.0),
        group=item.get("group"),
        metadata=item.get("metadata", {}),
    )


def _dependency_to_dict(dependency: Dependency) -> dict[str, Any]:
    item: dict[str, Any] = {"before": dependency.before, "after": dependency.after}
    if dependency.lag != 0.0:
        item["lag"] = dependency.lag
    if dependency.data != 0.0:
        item["data"] = dependency.data
    return item


def _dependency_from_dict(item: Mapping[str, Any]) -> Dependency:
    return Dependency(
        before=item["before"],
        after=item["after"],
        lag=item.get("lag", 0.0),
        data=item.get("data", 0.0),
    )


def _link_to_dict(link: Link) -> dict[str, Any]:
    item: dict[str, Any] = {"a": link.a, "b": link.b}
    if not math.isinf(link.bandwidth):
        item["bandwidth"] = link.bandwidth
    for name in ("latency", "energy_per_unit", "cost_per_unit"):
        if getattr(link, name) != 0.0:
            item[name] = getattr(link, name)
    return item


def _link_from_dict(item: Mapping[str, Any]) -> Link:
    return Link(
        a=item["a"],
        b=item["b"],
        bandwidth=item.get("bandwidth", math.inf),
        latency=item.get("latency", 0.0),
        energy_per_unit=item.get("energy_per_unit", 0.0),
        cost_per_unit=item.get("cost_per_unit", 0.0),
    )
