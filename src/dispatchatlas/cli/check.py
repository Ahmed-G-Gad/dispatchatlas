"""``dispatchatlas check``: validate a problem, and a schedule against it."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dispatchatlas.campaigns.records import RUN_SCHEMA, RunRecord
from dispatchatlas.cli._common import (
    PROG,
    Commands,
    add_command,
    add_problem_arguments,
    emit_fields,
    emit_json,
    existing_file,
    read_problem,
    text,
    violation_to_dict,
)
from dispatchatlas.core.errors import ModelError, SerializationError
from dispatchatlas.core.io import read_json
from dispatchatlas.core.objectives import evaluate
from dispatchatlas.core.schedule import Schedule
from dispatchatlas.core.serialization import (
    RESULT_FORMAT,
    SCHEDULE_FORMAT,
    result_from_dict,
    schedule_from_dict,
)
from dispatchatlas.core.validation import check_schedule

__all__ = ["register"]

_INCOMPLETE = frozenset({"unknown-task", "duplicate-task", "missing-task", "invalid-mode"})
"""Violation codes after which a schedule has no objective values."""


def register(commands: Commands) -> None:
    """Add the ``check`` command."""
    parser = add_command(
        commands,
        "check",
        _check,
        "Validate a problem, and a schedule against it.",
        description="Validate a problem: read it and check its structure. With SCHEDULE, "
        "also check the schedule against every constraint of the problem and compute its "
        "objective values. SCHEDULE is a schedule document, a result document written by "
        f"'{PROG} solve --output', or a run record of a campaign. The exit code is 1 when "
        "the problem is invalid or the schedule breaks a constraint.",
        epilog=f"examples:\n  {PROG} check js.json\n  {PROG} check js.json result.json --json",
    )
    add_problem_arguments(parser)
    parser.add_argument(
        "schedule",
        metavar="SCHEDULE",
        nargs="?",
        help="schedule document, result document, or campaign run record to check",
    )


def _check(arguments: argparse.Namespace) -> int:
    try:
        problem = read_problem(arguments.problem, arguments.format_name, arguments.instance)
    except (ModelError, SerializationError) as error:
        report: dict[str, Any] = {"problem": arguments.problem, "valid": False, "error": str(error)}
        if arguments.json:
            emit_json(report)
        else:
            emit_fields([("problem", f"{arguments.problem}: invalid"), ("defects", str(error))])
        return 1
    report = {"problem": problem.summary(), "valid": True}
    if arguments.schedule is not None:
        schedule, kind, named = _read_schedule(existing_file(arguments.schedule))
        violations = check_schedule(problem, schedule)
        complete = not any(item.code in _INCOMPLETE for item in violations)
        report["schedule"] = {
            "path": arguments.schedule,
            "kind": kind,
            "problem": named,
            "assignments": len(schedule),
            "feasible": not violations,
            "violations": [violation_to_dict(item) for item in violations],
            "objectives": evaluate(problem, schedule) if complete else None,
        }
    if arguments.json:
        emit_json(report)
    else:
        _print(report)
    checked = report.get("schedule")
    return 0 if checked is None or checked["feasible"] else 1


def _read_schedule(path: Path) -> tuple[Schedule, str, str | None]:
    """Read the schedule of a schedule document, a result document, or a campaign run record.

    Returns:
        The schedule, the kind of document, and the id of the problem the
        document names, if it names one.

    Raises:
        SerializationError: If the file holds none of these documents, or
            one without a schedule.
    """
    document = read_json(path)
    if not isinstance(document, Mapping):
        raise SerializationError(f"{path} does not hold a JSON object")
    if document.get("format") == SCHEDULE_FORMAT:
        named = document.get("problem")
        return schedule_from_dict(document), "schedule", named if isinstance(named, str) else None
    if document.get("format") == RESULT_FORMAT:
        result = result_from_dict(document)
        if result.schedule is None:
            raise SerializationError(
                f"{path}: the result holds no schedule (status {result.status.value})"
            )
        return result.schedule, "result", result.problem
    if document.get("schema") == RUN_SCHEMA:
        record = RunRecord.from_dict(document)
        if record.schedule is None:
            raise SerializationError(
                f"{path}: the run record holds no schedule; its campaign did not record them"
            )
        return record.schedule, "run record", record.problem.id
    raise SerializationError(
        f"{path} is not a schedule document, a result document, or a campaign run record"
    )


def _print(report: dict[str, Any]) -> None:
    problem = report["problem"]
    rows: list[tuple[str, object]] = [
        (
            "problem",
            f"{problem['id']}: valid, {problem['tasks']} tasks, {problem['resources']} "
            f"resources, {problem['dependencies']} dependencies",
        ),
        ("features", problem["features"] or "none"),
        ("objectives", problem["objectives"]),
    ]
    schedule = report.get("schedule")
    if schedule is not None:
        named = schedule["problem"]
        source = f"{schedule['path']} ({schedule['kind']}, {schedule['assignments']} assignments)"
        if named is not None and named != problem["id"]:
            source += f", written for problem {named}"
        violations = schedule["violations"]
        verdict = "feasible" if not violations else f"{len(violations)} violations"
        rows += [("schedule", source), ("verdict", verdict)]
        rows += [(item["code"], item["message"]) for item in violations]
        if schedule["objectives"] is not None:
            values = ", ".join(
                f"{name} {text(value)}" for name, value in schedule["objectives"].items()
            )
            rows.append(("values", values))
    emit_fields(rows)
