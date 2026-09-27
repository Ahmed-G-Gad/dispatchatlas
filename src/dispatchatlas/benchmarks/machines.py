"""Unrelated parallel machines, with optional setups, release dates, and due dates.

Each job is a single operation that runs on exactly one of ``m`` machines, and
its processing time depends on the machine. With sequence-dependent setups, a
machine that finishes job ``a`` must wait ``S[k][a][b]`` before it can start
job ``b``; the first job on a machine needs no setup. This is the
``R|S_ijk|Cmax`` benchmark of Vallada and Ruiz (2011): processing times
uniform in ``[1, 99]`` and setup times uniform in ``[1, 9]``, ``[1, 49]``,
``[1, 99]``, or ``[1, 124]``.

Due dates follow the tardiness-factor scheme of the OR-Library
weighted-tardiness instances (Crauwels, Potts and Van Wassenhove 1998): an
integer drawn uniformly from ``[P (1 - TF - RDD / 2), P (1 - TF + RDD / 2)]``,
where ``TF`` is the tardiness factor and ``RDD`` the relative range of due
dates. On one machine ``P`` is the total processing time; here it is the mean
machine load, the sum over jobs of their mean processing time divided by
``m``. Weights, when requested, are integers uniform in ``[1, 10]`` as in the
same instances.

The problems use the conventions of :mod:`dispatchatlas.benchmarks.shop`:
machines ``m0``, ``m1``, ... and jobs ``j0``, ``j1``, ... zero-padded to one
width. Each job is a task with one mode per machine.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import ClassVar

from dispatchatlas.benchmarks._common import (
    check_bool,
    check_int,
    check_int_range,
    check_matrix,
    check_number,
    check_objectives,
    check_per_item,
    floor_tolerant,
    instance_id,
    instance_metadata,
    numbered_ids,
    round_half_up,
)
from dispatchatlas.benchmarks.registry import FamilyInfo, FamilyRegistry
from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Mode, Problem, Resource, Scalar, SetupTimes, Task

__all__ = ["VALLADA_RUIZ_SETUPS", "ParallelMachines", "parallel_machine_problem", "register"]

VALLADA_RUIZ_SETUPS: tuple[int, ...] = (9, 49, 99, 124)
"""The largest setup time of each setup class of Vallada and Ruiz (2011)."""


def parallel_machine_problem(
    times: Sequence[Sequence[float]],
    *,
    setups: Sequence[Sequence[Sequence[float]]] | None = None,
    releases: Sequence[float] | None = None,
    due_dates: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    problem_id: str = "parallel-machines",
    objectives: Sequence[str] = ("makespan",),
    family: str | None = None,
    metadata: Mapping[str, Scalar] | None = None,
) -> Problem:
    """Build an unrelated parallel-machine problem.

    Args:
        times: ``times[j][k]`` is the processing time of job ``j`` on machine
            ``k``.
        setups: ``setups[k][a][b]`` is the setup time machine ``k`` needs
            between job ``a`` and job ``b``. Diagonal entries are ignored. No
            setup precedes the first job on a machine. The model forbids zero
            processing times on a machine with a non-zero setup time.
        releases: Release date of each job.
        due_dates: Due date of each job.
        weights: Weight of each job.
        problem_id: Id of the problem.
        objectives: Objectives to optimise, primary first.
        family: Id of the family that generated the data, if any.
        metadata: Annotations stored on the problem.

    Returns:
        The problem.

    Raises:
        ModelError: If the matrices are empty or ragged, a per-job list has the
            wrong length, or a job has a zero processing time on a machine with
            setup times.
    """
    rows = check_matrix(times, "parallel-machine times")
    count = len(rows)
    jobs = numbered_ids("j", count)
    machines = numbered_ids("m", len(rows[0]))
    starts = check_per_item(releases, count, "releases")
    dues = check_per_item(due_dates, count, "due_dates")
    job_weights = check_per_item(weights, count, "weights")
    tasks = tuple(
        Task(
            job,
            tuple(Mode(time, {machine: 1.0}) for machine, time in zip(machines, row, strict=True)),
            release=starts[j] if starts is not None else 0.0,
            due=dues[j] if dues is not None else None,
            weight=job_weights[j] if job_weights is not None else 1.0,
        )
        for j, (job, row) in enumerate(zip(jobs, rows, strict=True))
    )
    setup_times = None
    if setups is not None:
        tables = [check_matrix(table, "setup times") for table in setups]
        if len(tables) != len(machines) or any(
            len(table) != count or len(table[0]) != count for table in tables
        ):
            raise ModelError(f"setups must hold one {count} x {count} table per machine")
        setup_times = SetupTimes(
            {
                (machine, jobs[a], jobs[b]): table[a][b]
                for machine, table in zip(machines, tables, strict=True)
                for a in range(count)
                for b in range(count)
                if a != b
            }
        )
    return Problem(
        problem_id,
        tasks=tasks,
        resources=tuple(Resource(machine) for machine in machines),
        objectives=tuple(objectives),
        setups=setup_times,
        family=family,
        metadata=metadata or {},
    )


@dataclass(frozen=True)
class ParallelMachines:
    """Unrelated parallel machines in the style of Vallada and Ruiz (2011).

    Draws, in order: the processing times job by job and machine by machine;
    the setup times machine by machine, then by preceding job and following
    job; the release dates; the due dates; the weights. Draws of an option that
    is switched off are skipped.

    Attributes:
        jobs: Number of jobs.
        machines: Number of machines.
        min_time: Smallest processing time.
        max_time: Largest processing time.
        max_setup: Largest setup time. Setups are uniform in ``[1, max_setup]``
            for every machine and ordered pair of jobs. 0 means no setups.
            Vallada and Ruiz use 9, 49, 99, and 124.
        release_span: Release dates are uniform integers in
            ``[0, release_span * P]``, ``P`` being the mean machine load. 0
            releases every job at time 0.
        tardiness_factor: Tardiness factor ``TF`` of the due dates, in
            ``[0, 1]``, or ``None`` for no due dates. A job's due date is its
            release date plus the drawn value, and never earlier than its
            release date.
        due_date_range: Relative range ``RDD`` of the due dates.
        weighted: Draw weights uniform in ``[1, 10]`` instead of weight 1.
        objectives: Objectives the problems declare, primary first.
    """

    jobs: int = 20
    machines: int = 5
    min_time: int = 1
    max_time: int = 99
    max_setup: int = 0
    release_span: float = 0.0
    tardiness_factor: float | None = None
    due_date_range: float = 0.6
    weighted: bool = False
    objectives: tuple[str, ...] = ("makespan",)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="parallel-machines",
        name="Unrelated parallel machines",
        problem_class="parallel-machines",
        summary="Independent jobs each run on one of several unrelated machines, "
        "optionally with sequence-dependent setup times as in Vallada and Ruiz "
        "(2011), release dates, and due dates.",
        references=("vallada2011genetic", "crauwels1998local"),
    )
    _tags: ClassVar[Mapping[str, str]] = {
        "min_time": "pmin",
        "max_time": "pmax",
        "max_setup": "s",
        "release_span": "r",
        "tardiness_factor": "tf",
        "due_date_range": "rdd",
        "weighted": "w",
        "objectives": "",
    }

    def __post_init__(self) -> None:
        check_int("jobs", self.jobs)
        check_int("machines", self.machines)
        check_int_range("min_time", self.min_time, "max_time", self.max_time)
        check_int("max_setup", self.max_setup, minimum=0)
        object.__setattr__(self, "release_span", check_number("release_span", self.release_span))
        tardiness = self.tardiness_factor
        if tardiness is not None:
            tardiness = check_number("tardiness_factor", tardiness, maximum=1.0)
        object.__setattr__(self, "tardiness_factor", tardiness)
        spread = check_number("due_date_range", self.due_date_range)
        if tardiness is None and spread != type(self).due_date_range:
            raise ValueError("due_date_range needs tardiness_factor")
        object.__setattr__(self, "due_date_range", spread)
        check_bool("weighted", self.weighted)
        object.__setattr__(
            self,
            "objectives",
            check_objectives(
                self.objectives,
                due_dates=tardiness is not None,
                due_parameter="tardiness_factor",
            ),
        )

    def generate(self, seed: int = 0) -> Problem:
        """Return the parallel-machine problem for ``seed``."""
        problem_id = instance_id(self, f"{self.jobs}x{self.machines}", seed, self._tags)
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        n, m = self.jobs, self.machines
        times = [
            [float(rng.randint(self.min_time, self.max_time)) for _ in range(m)] for _ in range(n)
        ]
        setups = None
        if self.max_setup:
            setups = [
                [
                    [float(rng.randint(1, self.max_setup)) if a != b else 0.0 for b in range(n)]
                    for a in range(n)
                ]
                for _ in range(m)
            ]
        load = sum(fmean(row) for row in times) / m
        releases = None
        if self.release_span > 0.0:
            latest = floor_tolerant(self.release_span * load)
            releases = [float(rng.randint(0, latest)) for _ in range(n)]
        due_dates = None
        if self.tardiness_factor is not None:
            low = round_half_up(load * (1.0 - self.tardiness_factor - self.due_date_range / 2.0))
            high = round_half_up(load * (1.0 - self.tardiness_factor + self.due_date_range / 2.0))
            offsets = [max(0, rng.randint(low, high)) for _ in range(n)]
            base = releases if releases is not None else [0.0] * n
            due_dates = [start + offset for start, offset in zip(base, offsets, strict=True)]
        weights = [float(rng.randint(1, 10)) for _ in range(n)] if self.weighted else None
        return parallel_machine_problem(
            times,
            setups=setups,
            releases=releases,
            due_dates=due_dates,
            weights=weights,
            problem_id=problem_id,
            objectives=self.objectives,
            family=self.info.id,
            metadata=metadata,
        )


def register(registry: FamilyRegistry) -> None:
    """Add the parallel-machine family to ``registry``."""
    registry.register(ParallelMachines)
