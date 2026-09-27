"""The PSPLIB format of single-mode (``.sm``) and multi-mode (``.mm``) projects.

PSPLIB (Kolisch and Sprecher 1997) files list the jobs with their modes,
successors, durations, and requests, then the resource availabilities. Job 1
and the last job are a dummy source and sink with zero duration and no
requests; the reader drops them, so ``J30`` instances have 30 tasks. Tasks keep
their PSPLIB job numbers (``a02`` to ``a31`` in ``J30``) and renewable
resources their numbers (``r1``, ``r2``, ...).

A mode that requests more of a renewable resource than is available can never
run, and is dropped. The model has renewable resources only. A nonrenewable
resource whose availability covers the largest request of every job's
remaining modes cannot constrain a schedule and is dropped. One that can is
refused, unless ``ignore_nonrenewable=True`` drops it too, which turns the
instance into a relaxation of the published one. Doubly constrained resources
are refused.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from dispatchatlas.benchmarks.formats._text import parse_error, to_integer
from dispatchatlas.benchmarks.project import project_problem
from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Problem, Scalar

__all__ = ["read_psplib"]

_LABEL = re.compile(r"([RND])\s*(\d+)")


@dataclass(frozen=True)
class _Job:
    number: int
    successors: tuple[int, ...]
    modes: list[tuple[int, list[int]]]


def read_psplib(
    text: str, *, problem_id: str = "psplib", ignore_nonrenewable: bool = False
) -> Problem:
    """Read a PSPLIB project in the ``.sm`` or ``.mm`` format.

    Args:
        text: The instance.
        problem_id: Id of the problem.
        ignore_nonrenewable: Drop nonrenewable resources even when they
            constrain the choice of modes.

    Returns:
        The project, built by :func:`~dispatchatlas.benchmarks.project.project_problem`.
        Its metadata records the ``horizon`` of the file, and ``dropped_modes``
        or ``ignored_nonrenewable`` when the reader dropped anything.

    Raises:
        SerializationError: If the text is malformed.
        ModelError: If the instance needs a resource the model lacks, or a job
            has no mode that fits the renewable resources.
    """
    lines = text.splitlines()
    count = _field(lines, "jobs (incl. supersource/sink", problem_id)
    if _field(lines, "projects", problem_id, default=1) != 1:
        raise parse_error(problem_id, "only files with one project are supported")
    horizon = _field(lines, "horizon", problem_id, default=0)
    kinds = _kinds(lines, problem_id)
    jobs = _jobs(lines, count, len(kinds), problem_id)
    availability = _availability(lines, len(kinds), problem_id)
    if "D" in kinds:
        raise ModelError(f"{problem_id}: doubly constrained resources are not supported")

    real = [job for job in jobs if not (job.number in (1, count) and _is_dummy(job))]
    if not real:
        raise parse_error(problem_id, "the project has no job besides the dummy source and sink")
    renewable = [k for k, kind in enumerate(kinds) if kind == "R"]
    usable: list[list[tuple[int, list[int]]]] = []
    for job in real:
        fitting = [
            (duration, demands)
            for duration, demands in job.modes
            if all(demands[k] <= availability[k] for k in renewable)
        ]
        if not fitting:
            raise ModelError(
                f"{problem_id}: job {job.number} has no mode that fits the renewable resources"
            )
        usable.append(fitting)
    binding = [
        k
        for k, kind in enumerate(kinds)
        if kind == "N"
        and sum(max(demands[k] for _, demands in options) for options in usable) > availability[k]
    ]
    if binding and not ignore_nonrenewable:
        raise ModelError(
            f"{problem_id}: {len(binding)} nonrenewable resources can constrain the modes, "
            "and the model has none; pass ignore_nonrenewable=True to drop them"
        )
    modes: list[list[tuple[float, Sequence[float]]]] = [
        [(float(duration), [float(demands[k]) for k in renewable]) for duration, demands in options]
        for options in usable
    ]
    dropped = sum(len(job.modes) for job in real) - sum(len(options) for options in usable)
    index = {job.number: position for position, job in enumerate(real)}
    arcs = [
        (index[job.number], index[successor])
        for job in real
        for successor in job.successors
        if successor in index
    ]
    width = len(str(count))
    metadata: dict[str, Scalar] = {
        "format": "psplib",
        "problem_class": "multi-mode-rcpsp" if any(len(job.modes) > 1 for job in real) else "rcpsp",
        "horizon": horizon,
    }
    if dropped:
        metadata["dropped_modes"] = dropped
    if binding:
        metadata["ignored_nonrenewable"] = len(binding)
    return project_problem(
        modes,
        arcs,
        [float(availability[k]) for k in renewable],
        activity_ids=[f"a{job.number:0{width}d}" for job in real],
        problem_id=problem_id,
        metadata=metadata,
    )


def _field(lines: Sequence[str], label: str, source: str, default: int | None = None) -> int:
    """Return the integer after the colon of the first line that starts with ``label``."""
    for line in lines:
        if line.strip().startswith(label) and ":" in line:
            tokens = line.split(":", 1)[1].split()
            if tokens:
                return to_integer(tokens[0], f"the value of {label!r}", source)
    if default is None:
        raise parse_error(source, f"no line {label!r} found")
    return default


def _section(lines: Sequence[str], title: str, source: str) -> int:
    """Return the index of the line that follows the section title ``title``."""
    for number, line in enumerate(lines):
        if line.strip().upper().startswith(title):
            return number + 1
    raise parse_error(source, f"no section {title!r} found")


def _kinds(lines: Sequence[str], source: str) -> list[str]:
    """Return the kind (R, N, or D) of every resource column, in column order."""
    start = _section(lines, "REQUESTS/DURATIONS:", source)
    header = lines[start] if start < len(lines) else ""
    if "duration" not in header:
        raise parse_error(source, "the requests section lacks its column header")
    return [kind for kind, _ in _LABEL.findall(header.split("duration", 1)[1])]


def _jobs(lines: Sequence[str], count: int, resources: int, source: str) -> list[_Job]:
    """Read the precedence and request sections into jobs 1 to ``count``."""
    successors: dict[int, tuple[int, ...]] = {}
    declared: dict[int, int] = {}
    for line in _rows(lines, _section(lines, "PRECEDENCE RELATIONS:", source) + 1):
        values = [to_integer(token, "a precedence value", source) for token in line]
        if len(values) < 3 or len(values) != 3 + values[2]:
            raise parse_error(source, f"malformed precedence line {' '.join(line)!r}")
        successors[values[0]] = tuple(values[3:])
        declared[values[0]] = values[1]
    requests: dict[int, list[tuple[int, list[int]]]] = {}
    current: int | None = None
    for line in _rows(lines, _section(lines, "REQUESTS/DURATIONS:", source) + 1):
        values = [to_integer(token, "a request value", source) for token in line]
        if len(values) == 3 + resources:
            current = values[0]
            requests.setdefault(current, []).append((values[2], values[3:]))
        elif len(values) == 2 + resources and current is not None:
            requests[current].append((values[1], values[2:]))
        else:
            raise parse_error(source, f"malformed request line {' '.join(line)!r}")
    numbers = list(range(1, count + 1))
    if sorted(successors) != numbers or sorted(requests) != numbers:
        raise parse_error(
            source, f"the precedence and request sections must list jobs 1 to {count}"
        )
    jobs = []
    for number in numbers:
        if len(requests[number]) != declared[number]:
            raise parse_error(
                source,
                f"job {number} declares {declared[number]} modes but lists {len(requests[number])}",
            )
        unknown = [successor for successor in successors[number] if successor not in successors]
        if unknown:
            raise parse_error(source, f"job {number} has unknown successors {unknown}")
        jobs.append(_Job(number, successors[number], requests[number]))
    return jobs


def _availability(lines: Sequence[str], resources: int, source: str) -> list[int]:
    """Read the resource availabilities."""
    rows = _rows(lines, _section(lines, "RESOURCEAVAILABILITIES:", source) + 1)
    if not rows or len(rows[0]) != resources:
        raise parse_error(source, f"the availabilities must list {resources} values")
    return [to_integer(token, "an availability", source, minimum=0) for token in rows[0]]


def _rows(lines: Sequence[str], start: int) -> list[list[str]]:
    """Return the token rows from ``start`` up to the next line of asterisks."""
    rows = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("*"):
            break
        if stripped and not stripped.startswith("-"):
            rows.append(stripped.split())
    return rows


def _is_dummy(job: _Job) -> bool:
    """Return whether a job takes no time and requests nothing in any mode."""
    return all(duration == 0 and not any(demands) for duration, demands in job.modes)
