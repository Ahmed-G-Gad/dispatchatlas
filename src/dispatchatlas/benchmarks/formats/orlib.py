"""The OR-Library text format of job-shop and flow-shop instances.

An instance is a size line ``n m`` followed by one line per job that lists the
``machine time`` pairs of its operations in processing order. OR-Library
(Beasley 1990) numbers machines from 0; Taillard's generator program can also
number them from 1, which ``first_machine=1`` reads. Lines before the size
line, such as a description or the ``#`` comments of the JSPLIB collection, are
skipped. Processing times may be zero, as in ``orb07`` and in Heller's flow
shops.

The files ``jobshop1.txt`` (82 job shops) and ``flowshop1.txt`` (31 flow
shops) hold many instances: each starts with a line ``instance <name>`` and a
description line, and lines of ``+`` separate them. :func:`split_orlib` cuts
such a file into single instances, and the readers do it themselves when given
``instance=``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dispatchatlas.benchmarks.formats._text import (
    check_first_machine,
    is_integer,
    parse_error,
    to_integer,
)
from dispatchatlas.benchmarks.shop import flow_shop_problem, job_shop_problem
from dispatchatlas.core.model import Problem, Scalar

__all__ = ["read_orlib_flow_shop", "read_orlib_job_shop", "split_orlib"]

_HEADER = re.compile(r"instance\s+(\S+)")


def split_orlib(text: str) -> dict[str, str]:
    """Split a multi-instance OR-Library file into one text per instance.

    Args:
        text: Contents of a file such as ``jobshop1.txt`` or ``flowshop1.txt``.

    Returns:
        The instance texts keyed by instance name, in file order. Each text
        holds the description line, the size line, and the job lines. Empty
        when the text has no ``instance <name>`` lines.

    Raises:
        SerializationError: If an instance name appears twice.
    """
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        match = _HEADER.fullmatch(stripped)
        if match is not None:
            name = match.group(1)
            if name in blocks:
                raise parse_error("OR-Library file", f"instance {name!r} appears twice")
            current = blocks[name] = []
        elif current is not None and stripped and not stripped.startswith("+"):
            current.append(stripped)
    return {name: "\n".join(lines) + "\n" for name, lines in blocks.items()}


def read_orlib_job_shop(
    text: str,
    *,
    instance: str | None = None,
    problem_id: str | None = None,
    first_machine: int = 0,
) -> Problem:
    """Read a job shop in the OR-Library format.

    Args:
        text: One instance, or a multi-instance file when ``instance`` is given.
        instance: Name of the instance to extract from a multi-instance file.
        problem_id: Id of the problem. Defaults to ``instance``, else
            ``"orlib-job-shop"``.
        first_machine: Number of the first machine in the text, 0 or 1.

    Returns:
        The job shop, built by :func:`~dispatchatlas.benchmarks.shop.job_shop_problem`.

    Raises:
        SerializationError: If the text is malformed or lacks the instance.
        ConfigError: If ``first_machine`` is neither 0 nor 1.
    """
    shop = _read(text, instance, problem_id, "orlib-job-shop", first_machine)
    return job_shop_problem(
        shop.routes,
        machines=shop.machines,
        problem_id=shop.problem_id,
        metadata=shop.metadata("orlib-job-shop", "job-shop"),
    )


def read_orlib_flow_shop(
    text: str,
    *,
    instance: str | None = None,
    problem_id: str | None = None,
    first_machine: int = 0,
    permutation: bool = True,
) -> Problem:
    """Read a flow shop in the OR-Library format.

    Every job must visit the machines in increasing order, as in OR-Library's
    ``flowshop1.txt`` and the files of Taillard's generator.

    Args:
        text: One instance, or a multi-instance file when ``instance`` is given.
        instance: Name of the instance to extract from a multi-instance file.
        problem_id: Id of the problem. Defaults to ``instance``, else
            ``"orlib-flow-shop"``.
        first_machine: Number of the first machine in the text, 0 or 1.
        permutation: Require one common job order on every machine, as in the
            permutation flow-shop benchmarks.

    Returns:
        The flow shop, built by :func:`~dispatchatlas.benchmarks.shop.flow_shop_problem`.

    Raises:
        SerializationError: If the text is malformed, lacks the instance, or
            describes jobs that do not visit the machines in order.
        ConfigError: If ``first_machine`` is neither 0 nor 1.
    """
    shop = _read(text, instance, problem_id, "orlib-flow-shop", first_machine)
    in_order = list(range(shop.machines))
    for j, route in enumerate(shop.routes):
        visited = [machine for machine, _ in route]
        if visited != in_order:
            numbers = [machine + shop.first_machine for machine in visited]
            raise parse_error(
                shop.problem_id,
                f"job {j} visits machines {numbers}; a flow shop visits them in increasing order",
            )
    return flow_shop_problem(
        [[time for _, time in route] for route in shop.routes],
        problem_id=shop.problem_id,
        permutation=permutation,
        metadata=shop.metadata("orlib-flow-shop", "flow-shop"),
    )


@dataclass(frozen=True)
class _Shop:
    """One parsed OR-Library instance with 0-based machine indices."""

    problem_id: str
    instance: str | None
    description: str
    machines: int
    first_machine: int
    routes: list[list[tuple[int, float]]]

    def metadata(self, format_name: str, problem_class: str) -> dict[str, Scalar]:
        values: dict[str, Scalar] = {"format": format_name, "problem_class": problem_class}
        if self.instance is not None:
            values["instance"] = self.instance
        if self.description:
            values["description"] = self.description
        return values


def _read(
    text: str,
    instance: str | None,
    problem_id: str | None,
    default_id: str,
    first_machine: int,
) -> _Shop:
    base = check_first_machine(first_machine)
    if instance is not None:
        blocks = split_orlib(text)
        if instance not in blocks:
            known = ", ".join(blocks) if blocks else "none"
            raise parse_error(default_id, f"no instance {instance!r}; the text holds: {known}")
        text = blocks[instance]
    source = problem_id or instance or default_id
    description: list[str] = []
    size: tuple[int, int] | None = None
    body: list[str] = []
    for line in text.splitlines():
        tokens = line.split()
        if size is not None:
            body.extend(tokens)
        elif len(tokens) == 2 and all(is_integer(token) for token in tokens):
            size = (int(tokens[0]), int(tokens[1]))
        elif tokens:
            description.append(" ".join(tokens))
    if size is None:
        raise parse_error(source, "no size line 'jobs machines' found")
    jobs, machines = size
    if jobs < 1 or machines < 1:
        raise parse_error(source, f"the size line needs positive counts, got {jobs} x {machines}")
    expected = 2 * jobs * machines
    if len(body) != expected:
        hint = "; read one instance at a time" if len(body) > expected else ""
        raise parse_error(
            source,
            f"expected {expected} numbers for {jobs} jobs on {machines} machines, "
            f"found {len(body)}{hint}",
        )
    values = [to_integer(token, "every value", source) for token in body]
    routes = []
    for j in range(jobs):
        route = []
        for p in range(machines):
            number, time = values[2 * (j * machines + p)], values[2 * (j * machines + p) + 1]
            if not base <= number < base + machines:
                raise parse_error(
                    source,
                    f"job {j} names machine {number}, outside {base}..{base + machines - 1}",
                )
            if time < 0:
                raise parse_error(source, f"job {j} has a negative processing time {time}")
            route.append((number - base, float(time)))
        routes.append(route)
    return _Shop(
        problem_id=source,
        instance=instance,
        description=" ".join(description),
        machines=machines,
        first_machine=base,
        routes=routes,
    )
