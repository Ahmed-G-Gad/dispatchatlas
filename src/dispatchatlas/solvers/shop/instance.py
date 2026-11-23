"""The job-shop view of a problem, which the schedule builders of this package read.

A problem has the ``"job-shop"`` shape of :data:`~dispatchatlas.core.model.SHAPES`
when its dependencies join the tasks into chains and every task runs in one mode
on one machine. :class:`JobShop` reads such a problem as a job shop: each chain is
a job, and its tasks, in chain order, are the job's operations. The view holds
what the builders of :mod:`~dispatchatlas.solvers.shop.decoding` and the graph
functions of :mod:`~dispatchatlas.solvers.shop.graph` read: tuples of Python
numbers indexed by task index, for their loops, and the slot layout that job
sequences permute and random keys sort over, :attr:`JobShop.slot_job`, as a
NumPy array.

The shape leaves open the features that change when an operation may start or
which schedules are feasible, such as lags, setups, and deadlines. The builders
model none of them, so :meth:`JobShop.from_problem` refuses each, and
:data:`JOB_SHOP_FEATURES` lists the features the view accepts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.core.errors import ModelError
from dispatchatlas.core.model import FEATURES, SHAPES, Problem

__all__ = ["JOB_SHOP_FEATURES", "JobShop"]

JOB_SHOP_FEATURES: frozenset[str] = frozenset({"precedence", "release", "due", "groups"})
"""The features of :data:`~dispatchatlas.core.model.FEATURES` a job shop may have.

The chains of operations are the precedence, and the builders start no operation
before its release time. Due dates and groups are read only by objectives, which
the evaluator computes from the schedule, so a solver built on the view can
declare all four in :attr:`~dispatchatlas.core.solver.SolverInfo.supports`.
"""


@dataclass(frozen=True, eq=False)
class JobShop:
    """A problem with the job-shop shape, read as jobs of operations on machines.

    Build one with :meth:`from_problem`, which checks the problem. Every per-task
    tuple is indexed by task index, the position of the task in ``problem.tasks``,
    which is the index a :class:`~dispatchatlas.solvers.scheduling.Decoded`
    schedule uses too. Two views compare equal only when they are the same object.

    Attributes:
        problem: The problem the view reads.
        operations: The task indices of each job's operations, in processing
            order. Job ``j`` is the chain whose first task comes ``j``-th among
            the first tasks of the chains, in task order.
        job: The job of each task.
        machine: The machine of each task: the index of the resource it runs on
            in ``problem.resources``.
        duration: The processing time of each task.
        release: The release time of each task, before which it cannot start.
        predecessor: The task before each task in its job, ``-1`` for the first
            operation of a job.
        successor: The task after each task in its job, ``-1`` for the last
            operation of a job.
        slot_job: The job of each of the :attr:`size` slots of a job sequence:
            job 0 once per operation, then job 1, and so on. Every permutation of
            it is a job sequence of the shop. The array is read-only.
    """

    problem: Problem
    operations: tuple[tuple[int, ...], ...]
    job: tuple[int, ...]
    machine: tuple[int, ...]
    duration: tuple[float, ...]
    release: tuple[float, ...]
    predecessor: tuple[int, ...]
    successor: tuple[int, ...]
    slot_job: NDArray[np.intp]

    @classmethod
    def from_problem(cls, problem: Problem) -> JobShop:
        """Read ``problem`` as a job shop.

        The problem must have the ``"job-shop"`` shape, and every feature it has
        must be in :data:`JOB_SHOP_FEATURES`. The view refuses these features,
        which its builders do not model:

        - ``"lags"``: the builders start an operation as soon as its job
          predecessor completes, so a lag would be ignored;
        - ``"deadline"``: the builders ignore deadlines, so a schedule could
          miss one;
        - ``"setups"`` and ``"network"``: the builders add no setup or transfer
          time between operations;
        - ``"permutation"``: the builders let each machine order the jobs in its
          own way;
        - ``"cumulative"``: the view's machines have capacity 1, as in a job
          shop;
        - ``"modes"`` and ``"multi-resource"``: a problem with either lacks the
          shape, which gives every task one mode on one machine.

        Args:
            problem: The problem.

        Returns:
            The view.

        Raises:
            ModelError: If the problem lacks the shape or has a refused feature,
                with one issue for the shape and one for each refused feature.
        """
        issues: list[str] = []
        if "job-shop" not in problem.shapes:
            issues.append(f"it lacks the 'job-shop' shape: {SHAPES['job-shop']}")
        for name in sorted(problem.features - JOB_SHOP_FEATURES):
            issues.append(
                f"it has the feature {name!r}, which the view does not model: " + FEATURES[name]
            )
        if issues:
            raise ModelError(f"problem {problem.id!r} is not a job shop", tuple(issues))
        successors = problem.successors
        operations: list[tuple[int, ...]] = []
        for first, before in enumerate(problem.predecessors):
            if before:
                continue
            chain = [first]
            while successors[chain[-1]]:
                chain.append(successors[chain[-1]][0])
            operations.append(tuple(chain))
        size = problem.size
        job = [0] * size
        predecessor = [-1] * size
        successor = [-1] * size
        for number, route in enumerate(operations):
            for step, task in enumerate(route):
                job[task] = number
                if step > 0:
                    predecessor[task] = route[step - 1]
                if step + 1 < len(route):
                    successor[task] = route[step + 1]
        resource_index = problem.resource_index
        modes = [task.modes[0] for task in problem.tasks]
        slot_job = np.repeat(
            np.arange(len(operations), dtype=np.intp), [len(route) for route in operations]
        )
        slot_job.flags.writeable = False
        return cls(
            problem,
            tuple(operations),
            tuple(job),
            tuple(resource_index[next(iter(mode.demands))] for mode in modes),
            tuple(mode.duration for mode in modes),
            tuple(task.release for task in problem.tasks),
            tuple(predecessor),
            tuple(successor),
            slot_job,
        )

    @property
    def size(self) -> int:
        """Number of operations, which is the number of tasks of the problem."""
        return len(self.job)

    @property
    def jobs(self) -> int:
        """Number of jobs."""
        return len(self.operations)

    @property
    def machines(self) -> int:
        """Number of machines: the resources of the problem, in their order."""
        return len(self.problem.resources)

    def __repr__(self) -> str:
        return (
            f"JobShop(problem={self.problem.id!r}, jobs={self.jobs}, "
            f"machines={self.machines}, operations={self.size})"
        )
