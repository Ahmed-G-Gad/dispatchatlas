"""The FJSPLib ``.fjs`` format of flexible job-shop instances.

The format of the collection compiled by Mastrolilli and Gambardella (2000),
which holds the instances of Brandimarte (1993) among others: a header line
``n m`` with an optional third number, the average number of machines per
operation, which is ignored; then per job the number of operations and, per
operation, the number ``k`` of machines that can run it followed by ``k``
``machine time`` pairs. Machines are numbered from 1. Some mirrors number them
from 0, which ``first_machine=0`` reads. Blank lines are ignored.
"""

from __future__ import annotations

from dispatchatlas.benchmarks.formats._text import (
    Tokens,
    check_first_machine,
    parse_error,
    to_integer,
)
from dispatchatlas.benchmarks.shop import flexible_job_shop_problem
from dispatchatlas.core.model import Problem

__all__ = ["read_fjs"]


def read_fjs(text: str, *, problem_id: str = "fjs", first_machine: int = 1) -> Problem:
    """Read a flexible job shop in the ``.fjs`` format.

    Every machine option of an operation becomes one mode.

    Args:
        text: The instance.
        problem_id: Id of the problem.
        first_machine: Number of the first machine in the text, 0 or 1.

    Returns:
        The flexible job shop, built by
        :func:`~dispatchatlas.benchmarks.shop.flexible_job_shop_problem`.

    Raises:
        SerializationError: If the text is empty, truncated, or has trailing
            values, a count below 1, a machine out of range, or a negative
            processing time.
        ConfigError: If ``first_machine`` is neither 0 nor 1.
    """
    base = check_first_machine(first_machine)
    lines = [line.split() for line in text.splitlines() if line.strip()]
    if not lines:
        raise parse_error(problem_id, "the text is empty")
    header = lines[0]
    if not 2 <= len(header) <= 3:
        raise parse_error(problem_id, f"the header must give jobs and machines, got {header}")
    jobs = to_integer(header[0], "the number of jobs", problem_id, minimum=1)
    machines = to_integer(header[1], "the number of machines", problem_id, minimum=1)
    tokens = Tokens([token for line in lines[1:] for token in line], problem_id)
    data: list[list[list[tuple[int, float]]]] = []
    for j in range(jobs):
        operations = []
        for p in range(tokens.integer(f"operation count of job {j}", minimum=1)):
            options = []
            for _ in range(tokens.integer(f"machine count of job {j} operation {p}", minimum=1)):
                number = tokens.integer("a machine number")
                time = tokens.integer("a processing time", minimum=0)
                if not base <= number < base + machines:
                    raise parse_error(
                        problem_id,
                        f"job {j} operation {p} names machine {number}, "
                        f"outside {base}..{base + machines - 1}",
                    )
                options.append((number - base, float(time)))
            operations.append(options)
        data.append(operations)
    if tokens.remaining:
        raise parse_error(problem_id, f"{tokens.remaining} values follow the last job")
    return flexible_job_shop_problem(
        data,
        machines=machines,
        problem_id=problem_id,
        metadata={"format": "fjs", "problem_class": "flexible-job-shop"},
    )
