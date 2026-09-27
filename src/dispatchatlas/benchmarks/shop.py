"""Classic shop-scheduling families and the builders that map shop data onto the model.

Five families generate the classic machine environments:

- ``flow-shop``: every job visits machines 0, 1, ..., m-1 in that order.
- ``job-shop``: every job visits every machine once, in its own order.
- ``open-shop``: every job needs every machine once, in any order.
- ``flexible-job-shop``: a job is a chain of operations, and each operation runs
  on one machine chosen from its own eligible set.
- ``hybrid-flow-shop``: jobs visit a series of stages in order, each on one of
  the stage's parallel machines.

Processing times follow Taillard (1993): independent integers drawn uniformly
from ``[min_time, max_time]``, ``[1, 99]`` by default.

The builders (:func:`flow_shop_problem` and its siblings) turn plain matrices
into problems. The families, the exact Taillard instances in
:mod:`dispatchatlas.benchmarks.taillard`, and the file readers in
:mod:`dispatchatlas.benchmarks.formats` all use them, so a shop instance has the
same shape whatever its origin:

- Machines are unary resources ``m0``, ``m1``, ... and jobs are task groups
  ``j0``, ``j1``, .... Indices are zero-padded to a common width, so ids sort in
  index order.
- A flow-shop or open-shop operation is task ``<job>-<machine>``, for example
  ``j03-m2``. Operation ``p`` of a job in the other shops is task
  ``<job>-o<p>``, or ``<job>-s<stage>`` in a hybrid flow shop.
- Consecutive operations of a job are joined by finish-to-start dependencies.
- An operation that may run on several machines has one mode per machine.
- A job's due date sits on its final operation and its weight on every one of
  its operations, as :mod:`dispatchatlas.core.objectives` expects.

Due dates and weights, when requested, follow the job-shop weighted-tardiness
benchmark of Singer and Pinedo (1998): job ``j`` is due at
``floor(f * P_j)``, where ``P_j`` is its total processing time and ``f`` the
due-date tightness (1.3, 1.5, and 1.6 in that benchmark), and the first 20 % of
the jobs weigh 4, the next 60 % weigh 2, and the rest weigh 1. An operation with
several modes contributes its mean duration to ``P_j``.
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
    check_objectives,
    check_optional_positive,
    check_per_item,
    floor_tolerant,
    instance_id,
    instance_metadata,
    numbered_ids,
)
from dispatchatlas.benchmarks.registry import FamilyInfo, FamilyRegistry
from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import Dependency, Mode, Problem, Resource, Scalar, Task

__all__ = [
    "FlexibleJobShop",
    "FlowShop",
    "HybridFlowShop",
    "JobShop",
    "OpenShop",
    "flexible_job_shop_problem",
    "flow_shop_problem",
    "hybrid_flow_shop_problem",
    "job_shop_problem",
    "open_shop_problem",
    "register",
]

type Alternatives = Sequence[tuple[int, float]]
"""The ``(machine index, processing time)`` options of one operation."""


# ------------------------------------------------------------------ builders


def flow_shop_problem(
    times: Sequence[Sequence[float]],
    *,
    problem_id: str = "flow-shop",
    permutation: bool = True,
    due_dates: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    objectives: Sequence[str] = ("makespan",),
    family: str | None = None,
    metadata: Mapping[str, Scalar] | None = None,
) -> Problem:
    """Build a flow shop from a job-by-machine processing-time matrix.

    Args:
        times: ``times[j][k]`` is the processing time of job ``j`` on machine
            ``k``. Every job visits the machines in increasing index order.
        problem_id: Id of the problem.
        permutation: Require every machine to process the jobs in one common
            order, as in the permutation flow shop.
        due_dates: Due date of each job, placed on its last operation.
        weights: Weight of each job, placed on all its operations.
        objectives: Objectives to optimise, primary first.
        family: Id of the family that generated the data, if any.
        metadata: Annotations stored on the problem.

    Returns:
        The problem.

    Raises:
        ModelError: If the matrix is empty or ragged, or a per-job list has the
            wrong length.
    """
    rows = check_matrix(times, "flow-shop times")
    jobs = numbered_ids("j", len(rows))
    machines = numbered_ids("m", len(rows[0]))
    operations = [[[(machines[k], time)] for k, time in enumerate(row)] for row in rows]
    task_ids = [[f"{job}-{machine}" for machine in machines] for job in jobs]
    return _chain_problem(
        problem_id,
        jobs,
        task_ids,
        operations,
        [Resource(machine) for machine in machines],
        permutation=permutation,
        due_dates=due_dates,
        weights=weights,
        objectives=objectives,
        family=family,
        metadata=metadata,
    )


def job_shop_problem(
    routes: Sequence[Alternatives],
    *,
    machines: int | None = None,
    problem_id: str = "job-shop",
    due_dates: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    objectives: Sequence[str] = ("makespan",),
    family: str | None = None,
    metadata: Mapping[str, Scalar] | None = None,
) -> Problem:
    """Build a job shop from each job's route of machines and processing times.

    Args:
        routes: ``routes[j]`` lists the ``(machine index, processing time)``
            of job ``j``'s operations in processing order. Machine indices
            start at 0.
        machines: Number of machines. Defaults to one more than the largest
            index in the routes.
        problem_id: Id of the problem.
        due_dates: Due date of each job, placed on its last operation.
        weights: Weight of each job, placed on all its operations.
        objectives: Objectives to optimise, primary first.
        family: Id of the family that generated the data, if any.
        metadata: Annotations stored on the problem.

    Returns:
        The problem.

    Raises:
        ModelError: If a job has no operation, a machine index is out of range,
            or a per-job list has the wrong length.
    """
    options = [[[step] for step in route] for route in routes]
    return flexible_job_shop_problem(
        options,
        machines=machines,
        problem_id=problem_id,
        due_dates=due_dates,
        weights=weights,
        objectives=objectives,
        family=family,
        metadata=metadata,
    )


def flexible_job_shop_problem(
    jobs: Sequence[Sequence[Alternatives]],
    *,
    machines: int | None = None,
    problem_id: str = "flexible-job-shop",
    due_dates: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    objectives: Sequence[str] = ("makespan",),
    family: str | None = None,
    metadata: Mapping[str, Scalar] | None = None,
) -> Problem:
    """Build a flexible job shop from each operation's machine alternatives.

    Args:
        jobs: ``jobs[j][p]`` lists the ``(machine index, processing time)``
            options of operation ``p`` of job ``j``. Each option becomes one
            mode. Machine indices start at 0.
        machines: Number of machines. Defaults to one more than the largest
            index used.
        problem_id: Id of the problem.
        due_dates: Due date of each job, placed on its last operation.
        weights: Weight of each job, placed on all its operations.
        objectives: Objectives to optimise, primary first.
        family: Id of the family that generated the data, if any.
        metadata: Annotations stored on the problem.

    Returns:
        The problem.

    Raises:
        ModelError: If a job or an operation is empty, a machine index is out
            of range, or a per-job list has the wrong length.
    """
    data = [[list(options) for options in job] for job in jobs]
    if not data or any(not job for job in data):
        raise ModelError("every job needs at least one operation")
    if any(not options for job in data for options in job):
        raise ModelError("every operation needs at least one machine")
    used = [machine for job in data for options in job for machine, _ in options]
    for machine in used:
        if isinstance(machine, bool) or not isinstance(machine, int):
            raise ModelError(f"machine index {machine!r} is not an integer")
    count = max(used) + 1 if machines is None else machines
    for machine in used:
        if not 0 <= machine < count:
            raise ModelError(f"machine index {machine} is outside 0..{count - 1}")
    machine_ids = numbered_ids("m", count)
    job_ids = numbered_ids("j", len(data))
    positions = numbered_ids("o", max(len(job) for job in data))
    operations = [
        [[(machine_ids[machine], time) for machine, time in options] for options in job]
        for job in data
    ]
    task_ids = [
        [f"{job_id}-{positions[p]}" for p in range(len(job))]
        for job_id, job in zip(job_ids, data, strict=True)
    ]
    return _chain_problem(
        problem_id,
        job_ids,
        task_ids,
        operations,
        [Resource(machine) for machine in machine_ids],
        permutation=False,
        due_dates=due_dates,
        weights=weights,
        objectives=objectives,
        family=family,
        metadata=metadata,
    )


def hybrid_flow_shop_problem(
    times: Sequence[Sequence[Sequence[float]]],
    *,
    problem_id: str = "hybrid-flow-shop",
    due_dates: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    objectives: Sequence[str] = ("makespan",),
    family: str | None = None,
    metadata: Mapping[str, Scalar] | None = None,
) -> Problem:
    """Build a hybrid flow shop from per-stage, per-machine processing times.

    Machines are named ``s<stage>-m<k>``. Each job has one operation per stage,
    with one mode per machine of the stage.

    Args:
        times: ``times[j][s][k]`` is the processing time of job ``j`` at stage
            ``s`` on the stage's machine ``k``. Every job must list the same
            number of stages and, per stage, the same number of machines.
        problem_id: Id of the problem.
        due_dates: Due date of each job, placed on its last operation.
        weights: Weight of each job, placed on all its operations.
        objectives: Objectives to optimise, primary first.
        family: Id of the family that generated the data, if any.
        metadata: Annotations stored on the problem.

    Returns:
        The problem.

    Raises:
        ModelError: If the stage layout is empty or differs between jobs, or a
            per-job list has the wrong length.
    """
    data = [[list(stage) for stage in job] for job in times]
    if not data or not data[0] or any(not stage for stage in data[0]):
        raise ModelError("hybrid flow-shop times need at least one job, stage, and machine")
    widths = [len(stage) for stage in data[0]]
    if any([len(stage) for stage in job] != widths for job in data):
        raise ModelError("every job must list the same stages with the same machine counts")
    stages = numbered_ids("s", len(widths))
    slots = numbered_ids("m", max(widths))
    stage_machines = [
        [f"{stage}-{slot}" for slot in slots[:width]]
        for stage, width in zip(stages, widths, strict=True)
    ]
    job_ids = numbered_ids("j", len(data))
    operations = [
        [
            list(zip(stage_machines[s], stage_times, strict=True))
            for s, stage_times in enumerate(job)
        ]
        for job in data
    ]
    task_ids = [[f"{job_id}-{stage}" for stage in stages] for job_id in job_ids]
    resources = [Resource(machine) for machines in stage_machines for machine in machines]
    return _chain_problem(
        problem_id,
        job_ids,
        task_ids,
        operations,
        resources,
        permutation=False,
        due_dates=due_dates,
        weights=weights,
        objectives=objectives,
        family=family,
        metadata=metadata,
    )


def open_shop_problem(
    times: Sequence[Sequence[float]],
    *,
    problem_id: str = "open-shop",
    objectives: Sequence[str] = ("makespan",),
    family: str | None = None,
    metadata: Mapping[str, Scalar] | None = None,
) -> Problem:
    """Build an open shop from a job-by-machine processing-time matrix.

    Each operation holds its machine and a unary resource ``job-<j>`` that
    stands for its job, so a job runs on at most one machine at a time while
    its operations may run in any order.

    Args:
        times: ``times[j][k]`` is the processing time of job ``j`` on machine
            ``k``.
        problem_id: Id of the problem.
        objectives: Objectives to optimise, primary first.
        family: Id of the family that generated the data, if any.
        metadata: Annotations stored on the problem.

    Returns:
        The problem.

    Raises:
        ModelError: If the matrix is empty or ragged.
    """
    rows = check_matrix(times, "open-shop times")
    jobs = numbered_ids("j", len(rows))
    locks = numbered_ids("job-", len(rows))
    machines = numbered_ids("m", len(rows[0]))
    tasks = [
        Task(
            f"{job}-{machine}",
            (Mode(time, {machine: 1.0, lock: 1.0}, location=machine),),
            group=job,
        )
        for job, lock, row in zip(jobs, locks, rows, strict=True)
        for machine, time in zip(machines, row, strict=True)
    ]
    resources = [Resource(machine) for machine in machines]
    resources += [Resource(lock, kind="job") for lock in locks]
    return Problem(
        problem_id,
        tasks=tuple(tasks),
        resources=tuple(resources),
        objectives=tuple(objectives),
        family=family,
        metadata=metadata or {},
    )


# ------------------------------------------------------------------ families


@dataclass(frozen=True)
class FlowShop:
    """Flow shops with Taillard's processing-time scheme.

    Draws, in order: the processing times machine by machine and, within a
    machine, job by job, the order of Taillard's generator. Due dates and
    weights, when requested, are computed from the times without further draws.

    Attributes:
        jobs: Number of jobs.
        machines: Number of machines.
        min_time: Smallest processing time.
        max_time: Largest processing time.
        permutation: Require one common job order on every machine.
        due_date_tightness: Factor ``f`` of the due dates ``floor(f * P_j)``,
            or ``None`` for no due dates.
        weighted: Give jobs the weights 4, 2, and 1 of Singer and Pinedo (1998)
            instead of weight 1.
        objectives: Objectives the problems declare, primary first.
    """

    jobs: int = 20
    machines: int = 5
    min_time: int = 1
    max_time: int = 99
    permutation: bool = True
    due_date_tightness: float | None = None
    weighted: bool = False
    objectives: tuple[str, ...] = ("makespan",)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="flow-shop",
        name="Flow shop",
        problem_class="flow-shop",
        summary="Jobs visit every machine in the same order, with integer processing "
        "times drawn uniformly as in Taillard (1993), optionally in one common job "
        "order on every machine.",
        references=("taillard1993benchmarks", "singer1998computational"),
    )
    _tags: ClassVar[Mapping[str, str]] = {
        "min_time": "pmin",
        "max_time": "pmax",
        "permutation": "perm",
        "due_date_tightness": "due",
        "weighted": "w",
        "objectives": "",
    }

    def __post_init__(self) -> None:
        check_int("jobs", self.jobs)
        check_int("machines", self.machines)
        check_int_range("min_time", self.min_time, "max_time", self.max_time)
        check_bool("permutation", self.permutation)
        check_bool("weighted", self.weighted)
        tightness = check_optional_positive("due_date_tightness", self.due_date_tightness)
        object.__setattr__(self, "due_date_tightness", tightness)
        object.__setattr__(
            self,
            "objectives",
            check_objectives(
                self.objectives,
                due_dates=tightness is not None,
                due_parameter="due_date_tightness",
            ),
        )

    def generate(self, seed: int = 0) -> Problem:
        """Return the flow shop for ``seed``."""
        problem_id = instance_id(self, f"{self.jobs}x{self.machines}", seed, self._tags)
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        by_machine = [
            [rng.randint(self.min_time, self.max_time) for _ in range(self.jobs)]
            for _ in range(self.machines)
        ]
        times = [[float(by_machine[k][j]) for k in range(self.machines)] for j in range(self.jobs)]
        return flow_shop_problem(
            times,
            problem_id=problem_id,
            permutation=self.permutation,
            due_dates=_due_dates([sum(row) for row in times], self.due_date_tightness),
            weights=_weights(self.jobs) if self.weighted else None,
            objectives=self.objectives,
            family=self.info.id,
            metadata=metadata,
        )


@dataclass(frozen=True)
class JobShop:
    """Job shops with Taillard's processing-time and routing scheme.

    Draws, in order, as in Taillard's generator: every processing time, job by
    job and operation by operation; then each job's machine order, built by
    swapping position ``i`` with a uniformly drawn position ``i..m-1`` for
    ``i = 0, ..., m-1``.

    Attributes:
        jobs: Number of jobs.
        machines: Number of machines. Every job visits each machine once.
        min_time: Smallest processing time.
        max_time: Largest processing time.
        due_date_tightness: Factor ``f`` of the due dates ``floor(f * P_j)``,
            or ``None`` for no due dates.
        weighted: Give jobs the weights 4, 2, and 1 of Singer and Pinedo (1998)
            instead of weight 1.
        objectives: Objectives the problems declare, primary first.
    """

    jobs: int = 10
    machines: int = 10
    min_time: int = 1
    max_time: int = 99
    due_date_tightness: float | None = None
    weighted: bool = False
    objectives: tuple[str, ...] = ("makespan",)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="job-shop",
        name="Job shop",
        problem_class="job-shop",
        summary="Each job visits every machine once in its own random order, with "
        "processing times and routes drawn as in Taillard (1993).",
        references=("taillard1993benchmarks", "singer1998computational"),
    )
    _tags: ClassVar[Mapping[str, str]] = {
        "min_time": "pmin",
        "max_time": "pmax",
        "due_date_tightness": "due",
        "weighted": "w",
        "objectives": "",
    }

    def __post_init__(self) -> None:
        check_int("jobs", self.jobs)
        check_int("machines", self.machines)
        check_int_range("min_time", self.min_time, "max_time", self.max_time)
        check_bool("weighted", self.weighted)
        tightness = check_optional_positive("due_date_tightness", self.due_date_tightness)
        object.__setattr__(self, "due_date_tightness", tightness)
        object.__setattr__(
            self,
            "objectives",
            check_objectives(
                self.objectives,
                due_dates=tightness is not None,
                due_parameter="due_date_tightness",
            ),
        )

    def generate(self, seed: int = 0) -> Problem:
        """Return the job shop for ``seed``."""
        problem_id = instance_id(self, f"{self.jobs}x{self.machines}", seed, self._tags)
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        times, orders = _taillard_draws(rng, self.jobs, self.machines, self.min_time, self.max_time)
        routes = [
            [(order[i], float(job_times[i])) for i in range(self.machines)]
            for job_times, order in zip(times, orders, strict=True)
        ]
        return job_shop_problem(
            routes,
            machines=self.machines,
            problem_id=problem_id,
            due_dates=_due_dates([float(sum(row)) for row in times], self.due_date_tightness),
            weights=_weights(self.jobs) if self.weighted else None,
            objectives=self.objectives,
            family=self.info.id,
            metadata=metadata,
        )


@dataclass(frozen=True)
class OpenShop:
    """Open shops with Taillard's data scheme.

    Draws, in order, as in Taillard's open-shop generator (the same procedure
    as his job shops): every processing time, job by job; then a random machine
    permutation per job that assigns the ``i``-th time of a job to a machine.

    Attributes:
        jobs: Number of jobs.
        machines: Number of machines. Every job needs each machine once.
        min_time: Smallest processing time.
        max_time: Largest processing time.
        objectives: Objectives the problems declare, primary first.
    """

    jobs: int = 5
    machines: int = 5
    min_time: int = 1
    max_time: int = 99
    objectives: tuple[str, ...] = ("makespan",)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="open-shop",
        name="Open shop",
        problem_class="open-shop",
        summary="Each job needs every machine once in any order and one machine at "
        "a time, with processing times drawn as in Taillard (1993).",
        references=("taillard1993benchmarks", "gonzalez1976open"),
    )
    _tags: ClassVar[Mapping[str, str]] = {"min_time": "pmin", "max_time": "pmax", "objectives": ""}

    def __post_init__(self) -> None:
        check_int("jobs", self.jobs)
        check_int("machines", self.machines)
        check_int_range("min_time", self.min_time, "max_time", self.max_time)
        object.__setattr__(self, "objectives", check_objectives(self.objectives, due_dates=False))

    def generate(self, seed: int = 0) -> Problem:
        """Return the open shop for ``seed``."""
        problem_id = instance_id(self, f"{self.jobs}x{self.machines}", seed, self._tags)
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        times, orders = _taillard_draws(rng, self.jobs, self.machines, self.min_time, self.max_time)
        matrix = [[0.0] * self.machines for _ in range(self.jobs)]
        for j, (job_times, order) in enumerate(zip(times, orders, strict=True)):
            for i, machine in enumerate(order):
                matrix[j][machine] = float(job_times[i])
        return open_shop_problem(
            matrix,
            problem_id=problem_id,
            objectives=self.objectives,
            family=self.info.id,
            metadata=metadata,
        )


@dataclass(frozen=True)
class FlexibleJobShop:
    """Flexible job shops in the style of Brandimarte (1993).

    Draws, job by job: the number of operations, uniform in
    ``[min_operations, max_operations]``; then per operation the number of
    eligible machines, uniform in ``[1, flexibility]``, the eligible machines
    as a uniform sample of that size, and one processing time per eligible
    machine, uniform in ``[min_time, max_time]``.

    Attributes:
        jobs: Number of jobs.
        machines: Number of machines.
        min_operations: Fewest operations of a job.
        max_operations: Most operations of a job.
        flexibility: Most machines eligible for one operation.
        min_time: Smallest processing time.
        max_time: Largest processing time.
        due_date_tightness: Factor ``f`` of the due dates ``floor(f * P_j)``,
            where an operation counts with its mean time over its machines, or
            ``None`` for no due dates.
        weighted: Give jobs the weights 4, 2, and 1 of Singer and Pinedo (1998)
            instead of weight 1.
        objectives: Objectives the problems declare, primary first.
    """

    jobs: int = 10
    machines: int = 6
    min_operations: int = 5
    max_operations: int = 7
    flexibility: int = 3
    min_time: int = 1
    max_time: int = 20
    due_date_tightness: float | None = None
    weighted: bool = False
    objectives: tuple[str, ...] = ("makespan",)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="flexible-job-shop",
        name="Flexible job shop",
        problem_class="flexible-job-shop",
        summary="Jobs are chains of operations, each of which runs on one machine "
        "from its own random eligible set with machine-dependent times, as in "
        "Brandimarte (1993).",
        references=("brandimarte1993routing", "singer1998computational"),
    )
    _tags: ClassVar[Mapping[str, str]] = {
        "min_operations": "omin",
        "max_operations": "omax",
        "flexibility": "flex",
        "min_time": "pmin",
        "max_time": "pmax",
        "due_date_tightness": "due",
        "weighted": "w",
        "objectives": "",
    }

    def __post_init__(self) -> None:
        check_int("jobs", self.jobs)
        check_int("machines", self.machines)
        check_int_range(
            "min_operations", self.min_operations, "max_operations", self.max_operations
        )
        check_int("flexibility", self.flexibility)
        if self.flexibility > self.machines:
            raise ValueError(
                f"flexibility ({self.flexibility}) cannot exceed machines ({self.machines})"
            )
        check_int_range("min_time", self.min_time, "max_time", self.max_time)
        check_bool("weighted", self.weighted)
        tightness = check_optional_positive("due_date_tightness", self.due_date_tightness)
        object.__setattr__(self, "due_date_tightness", tightness)
        object.__setattr__(
            self,
            "objectives",
            check_objectives(
                self.objectives,
                due_dates=tightness is not None,
                due_parameter="due_date_tightness",
            ),
        )

    def generate(self, seed: int = 0) -> Problem:
        """Return the flexible job shop for ``seed``."""
        problem_id = instance_id(self, f"{self.jobs}x{self.machines}", seed, self._tags)
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        jobs: list[list[list[tuple[int, float]]]] = []
        for _ in range(self.jobs):
            operations = []
            for _ in range(rng.randint(self.min_operations, self.max_operations)):
                eligible = sorted(
                    rng.sample(range(self.machines), rng.randint(1, self.flexibility))
                )
                operations.append(
                    [
                        (machine, float(rng.randint(self.min_time, self.max_time)))
                        for machine in eligible
                    ]
                )
            jobs.append(operations)
        work = [sum(fmean(time for _, time in options) for options in job) for job in jobs]
        return flexible_job_shop_problem(
            jobs,
            machines=self.machines,
            problem_id=problem_id,
            due_dates=_due_dates(work, self.due_date_tightness),
            weights=_weights(self.jobs) if self.weighted else None,
            objectives=self.objectives,
            family=self.info.id,
            metadata=metadata,
        )


@dataclass(frozen=True)
class HybridFlowShop:
    """Hybrid flow shops with identical or unrelated parallel machines per stage.

    Draws, in order: the number of machines of every stage, uniform in
    ``[min_machines, max_machines]``; then job by job and stage by stage the
    processing time, one draw shared by the stage's machines when they are
    identical, or one draw per machine when they are unrelated.

    Attributes:
        jobs: Number of jobs.
        stages: Number of stages every job visits in order.
        min_machines: Fewest parallel machines of a stage.
        max_machines: Most parallel machines of a stage.
        unrelated: Draw an independent processing time per machine instead of
            one time shared by the stage's identical machines.
        min_time: Smallest processing time.
        max_time: Largest processing time.
        due_date_tightness: Factor ``f`` of the due dates ``floor(f * P_j)``,
            where a stage counts with its mean time over its machines, or
            ``None`` for no due dates.
        weighted: Give jobs the weights 4, 2, and 1 of Singer and Pinedo (1998)
            instead of weight 1.
        objectives: Objectives the problems declare, primary first.
    """

    jobs: int = 20
    stages: int = 5
    min_machines: int = 1
    max_machines: int = 3
    unrelated: bool = False
    min_time: int = 1
    max_time: int = 99
    due_date_tightness: float | None = None
    weighted: bool = False
    objectives: tuple[str, ...] = ("makespan",)

    info: ClassVar[FamilyInfo] = FamilyInfo(
        id="hybrid-flow-shop",
        name="Hybrid flow shop",
        problem_class="hybrid-flow-shop",
        summary="Jobs visit a series of stages in order, each on one of the stage's "
        "identical or unrelated parallel machines.",
        references=("ruiz2010hybrid", "taillard1993benchmarks", "singer1998computational"),
    )
    _tags: ClassVar[Mapping[str, str]] = {
        "min_machines": "kmin",
        "max_machines": "kmax",
        "unrelated": "unrel",
        "min_time": "pmin",
        "max_time": "pmax",
        "due_date_tightness": "due",
        "weighted": "w",
        "objectives": "",
    }

    def __post_init__(self) -> None:
        check_int("jobs", self.jobs)
        check_int("stages", self.stages)
        check_int_range("min_machines", self.min_machines, "max_machines", self.max_machines)
        check_bool("unrelated", self.unrelated)
        check_int_range("min_time", self.min_time, "max_time", self.max_time)
        check_bool("weighted", self.weighted)
        tightness = check_optional_positive("due_date_tightness", self.due_date_tightness)
        object.__setattr__(self, "due_date_tightness", tightness)
        object.__setattr__(
            self,
            "objectives",
            check_objectives(
                self.objectives,
                due_dates=tightness is not None,
                due_parameter="due_date_tightness",
            ),
        )

    def generate(self, seed: int = 0) -> Problem:
        """Return the hybrid flow shop for ``seed``."""
        problem_id = instance_id(self, f"{self.jobs}x{self.stages}", seed, self._tags)
        metadata = instance_metadata(self, seed)
        rng = random.Random(seed)
        widths = [rng.randint(self.min_machines, self.max_machines) for _ in range(self.stages)]
        times: list[list[list[float]]] = []
        for _ in range(self.jobs):
            job = []
            for width in widths:
                if self.unrelated:
                    job.append(
                        [float(rng.randint(self.min_time, self.max_time)) for _ in range(width)]
                    )
                else:
                    job.append([float(rng.randint(self.min_time, self.max_time))] * width)
            times.append(job)
        work = [sum(fmean(stage) for stage in job) for job in times]
        return hybrid_flow_shop_problem(
            times,
            problem_id=problem_id,
            due_dates=_due_dates(work, self.due_date_tightness),
            weights=_weights(self.jobs) if self.weighted else None,
            objectives=self.objectives,
            family=self.info.id,
            metadata=metadata,
        )


def register(registry: FamilyRegistry) -> None:
    """Add the shop families to ``registry``."""
    for factory in (FlowShop, JobShop, OpenShop, FlexibleJobShop, HybridFlowShop):
        registry.register(factory)


# ------------------------------------------------------------------ helpers


def _taillard_draws(
    rng: random.Random, jobs: int, machines: int, low: int, high: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Draw job-shop times and machine orders in the order of Taillard's generator."""
    times = [[rng.randint(low, high) for _ in range(machines)] for _ in range(jobs)]
    orders = []
    for _ in range(jobs):
        order = list(range(machines))
        for i in range(machines):
            k = rng.randint(i, machines - 1)
            order[i], order[k] = order[k], order[i]
        orders.append(order)
    return times, orders


def _chain_problem(
    problem_id: str,
    job_ids: Sequence[str],
    task_ids: Sequence[Sequence[str]],
    operations: Sequence[Sequence[Sequence[tuple[str, float]]]],
    resources: Sequence[Resource],
    *,
    permutation: bool,
    due_dates: Sequence[float] | None,
    weights: Sequence[float] | None,
    objectives: Sequence[str],
    family: str | None,
    metadata: Mapping[str, Scalar] | None,
) -> Problem:
    """Build a problem whose jobs are chains of operations with machine options."""
    dues = check_per_item(due_dates, len(job_ids), "due_dates")
    job_weights = check_per_item(weights, len(job_ids), "weights")
    tasks: list[Task] = []
    dependencies: list[Dependency] = []
    for j, job in enumerate(job_ids):
        previous: str | None = None
        last = len(operations[j]) - 1
        for p, options in enumerate(operations[j]):
            task_id = task_ids[j][p]
            modes = tuple(Mode(time, {machine: 1.0}) for machine, time in options)
            tasks.append(
                Task(
                    task_id,
                    modes,
                    due=dues[j] if dues is not None and p == last else None,
                    weight=job_weights[j] if job_weights is not None else 1.0,
                    group=job,
                )
            )
            if previous is not None:
                dependencies.append(Dependency(previous, task_id))
            previous = task_id
    return Problem(
        problem_id,
        tasks=tuple(tasks),
        resources=tuple(resources),
        dependencies=tuple(dependencies),
        objectives=tuple(objectives),
        permutation=permutation,
        family=family,
        metadata=metadata or {},
    )


def _due_dates(work: Sequence[float], tightness: float | None) -> list[float] | None:
    """Return ``floor(tightness * work)`` per job, or ``None`` without a tightness."""
    if tightness is None:
        return None
    return [float(floor_tolerant(tightness * total)) for total in work]


def _weights(count: int) -> list[float]:
    """Return the Singer and Pinedo (1998) weights: 4 for the first 20 % of jobs, 2, then 1.

    The shares are rounded half up: jobs ``0 .. round(0.2 n) - 1`` weigh 4 and
    jobs up to ``round(0.8 n) - 1`` weigh 2.
    """
    heavy = (2 * count + 5) // 10
    medium = (8 * count + 5) // 10
    return [4.0 if j < heavy else 2.0 if j < medium else 1.0 for j in range(count)]
