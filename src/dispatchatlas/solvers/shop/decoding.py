"""Schedule builders that decode job sequences into job-shop schedules.

A **job sequence** (Bierwirth 1995, doi:10.1007/BF01719250) lists every job once
per operation, and its ``k``-th occurrence of job ``j`` stands for the ``k``-th
operation of ``j``. Every permutation of
:attr:`~dispatchatlas.solvers.shop.instance.JobShop.slot_job` is therefore a job
sequence, and every job sequence is feasible: no builder here repairs a sequence
or calls :func:`~dispatchatlas.solvers.scheduling.precedence_feasible`. The
position of an operation's occurrence is its **priority**, and an operation
comes before another in the priority when its occurrence comes first. Random keys
(Bean 1994, doi:10.1287/ijoc.6.2.154) reach the same builders through
:func:`sequence_from_keys`, which sorts the slots of a job sequence by their keys.

Three builders, named by :data:`BUILDERS`, turn a sequence into a schedule:

- **semi-active** (:func:`semi_active`): the operations, in sequence order, each
  start at the latest of the completion of their job predecessor, the
  availability of their machine, and their release time. No operation fills an
  idle interval left earlier on its machine.
- **active** (:func:`giffler_thompson`): the procedure of Giffler and Thompson
  (1960, doi:10.1287/opre.8.4.487). At each step it takes the earliest
  completion time C* of the operations that can be scheduled next and the
  machine M* of an operation that attains it, and schedules, among the
  operations on M* that can start before C*, the one that comes first in the
  priority.
- **non-delay** (:func:`giffler_thompson` with ``non_delay=True``): at each step
  the operation that can start earliest is scheduled, so no machine stays idle
  while an operation that needs it is ready.

**Ties.** When several operations attain C*, or the earliest start time, the one
that comes first in the priority fixes the machine. Positions in a sequence are
unique, so no other tie arises in any builder. Random keys sort stably, so equal
keys keep the order of their slots.

**Ranks.** A job sequence lists the operations of a job in their route order, so
it cannot express a priority that ranks an operation ahead of its job
predecessor, such as shortest processing time. :func:`giffler_thompson_ranked`
builds the same active and non-delay schedules from one rank per operation, a
lower rank coming first in the priority, under the same tie rule. The ranks must
be pairwise different, so that no tie arises there either.

Every builder returns a :class:`~dispatchatlas.solvers.scheduling.Decoded`
schedule of plain Python numbers, which
:class:`~dispatchatlas.solvers.evaluation.Evaluator` scores and
:meth:`~dispatchatlas.solvers.search.Search.assess_built` counts as one
evaluation. :class:`JobSequenceDecoder` fixes the builder of a run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.core.errors import ModelError
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.shop.instance import JobShop

__all__ = [
    "BUILDERS",
    "BuilderName",
    "JobSequence",
    "JobSequenceDecoder",
    "OperationRanks",
    "check_sequence",
    "giffler_thompson",
    "giffler_thompson_ranked",
    "semi_active",
    "sequence_from_keys",
]

type BuilderName = Literal["semi-active", "active", "non-delay"]
"""The name of a schedule builder, one of :data:`BUILDERS`."""

type JobSequence = Sequence[int] | NDArray[np.integer[Any]]
"""A job sequence: one job index per operation, as a sequence or an integer array."""

type OperationRanks = Sequence[int] | NDArray[np.integer[Any]]
"""Operation ranks: one integer per task, indexed by task index, as a sequence or an array."""

BUILDERS: tuple[BuilderName, ...] = ("semi-active", "active", "non-delay")
"""The names of the schedule builders, as :class:`JobSequenceDecoder` takes them."""


def sequence_from_keys(
    keys: Sequence[float] | NDArray[np.floating[Any]], slot_job: JobSequence
) -> NDArray[np.intp]:
    """Return the job sequence that sorting the slots of ``slot_job`` by ``keys`` gives.

    Key ``i`` belongs to slot ``i``, which holds job ``slot_job[i]``. The slots
    are sorted by ascending key with a stable sort, which is a sort by
    ``(key, slot)``: slots with equal keys keep their order. The jobs of the
    sorted slots form the sequence.

    Args:
        keys: One finite key per slot.
        slot_job: The job of each slot, such as
            :attr:`~dispatchatlas.solvers.shop.instance.JobShop.slot_job`.

    Returns:
        A new array that holds the job sequence.

    Raises:
        ValueError: If the keys and the slots differ in number or shape, or a key
            is not finite.
    """
    values = np.asarray(keys, dtype=np.float64)
    slots = np.asarray(slot_job, dtype=np.intp)
    if values.ndim != 1 or values.shape != slots.shape:
        raise ValueError(f"expected {slots.size} keys in one row, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("every key must be finite")
    return slots[np.argsort(values, kind="stable")]


def check_sequence(shop: JobShop, sequence: JobSequence) -> tuple[str, ...]:
    """Return every way in which ``sequence`` fails to be a job sequence of ``shop``.

    A job sequence of ``shop`` is a row of :attr:`JobShop.size` integers in which
    every job occurs exactly as often as it has operations. The builders make a
    fast form of this check on every decode and raise
    :class:`~dispatchatlas.core.errors.ModelError` with these lines when it fails.
    Calling it checks a sequence without decoding it, as tests do.

    Args:
        shop: The job shop.
        sequence: The candidate sequence.

    Returns:
        One readable line per defect, empty for a job sequence.
    """
    array = np.asarray(sequence)
    if array.ndim != 1:
        return (f"a job sequence is one row of job indices, got shape {array.shape}",)
    if array.size and array.dtype.kind not in "iu":
        return (f"a job sequence holds integers, got {array.dtype}",)
    issues: list[str] = []
    if array.size != shop.size:
        issues.append(f"it has {array.size} entries for the {shop.size} operations")
    values = array.astype(np.int64, copy=False)
    unknown = sorted({int(value) for value in values if not 0 <= value < shop.jobs})
    if unknown:
        issues.append(f"it names jobs outside 0..{shop.jobs - 1}: {unknown[:5]}")
    counts = np.bincount(values[(values >= 0) & (values < shop.jobs)], minlength=shop.jobs)
    for job, (count, operations) in enumerate(zip(counts, shop.operations, strict=True)):
        if count != len(operations):
            issues.append(f"job {job} occurs {count} times for its {len(operations)} operations")
    return tuple(issues)


def _tasks(shop: JobShop, sequence: JobSequence) -> list[int]:
    """Return the task of each position of a job sequence, after checking the sequence.

    A row of integers is a job sequence exactly when sorting it gives the sorted
    slot layout :attr:`JobShop.slot_job`.
    """
    array = np.asarray(sequence)
    if (
        array.ndim != 1
        or array.dtype.kind not in "iu"
        or not np.array_equal(np.sort(array), shop.slot_job)
    ):
        raise ModelError(
            f"the sequence is not a job sequence of problem {shop.problem.id!r}",
            check_sequence(shop, sequence),
        )
    operations = shop.operations
    taken = [0] * shop.jobs
    tasks: list[int] = []
    for job in array.tolist():
        tasks.append(operations[job][taken[job]])
        taken[job] += 1
    return tasks


def _ranks(shop: JobShop, rank: OperationRanks) -> list[int]:
    """Return the rank of each task as a list, after checking that every task has its own."""
    array = np.asarray(rank)
    # Distinct ranks are checked on the sorted ranks: np.unique would import numpy.ma during the
    # first decode of a process, which the warm-up rule of the campaigns forbids.
    if (
        array.ndim != 1
        or array.size != shop.size
        or array.dtype.kind not in "iu"
        or bool(np.any(np.diff(np.sort(array)) == 0))
    ):
        raise ModelError(
            f"the ranks are not one distinct integer per operation of problem {shop.problem.id!r}",
            _rank_issues(shop, array),
        )
    return [int(value) for value in array.tolist()]


def _rank_issues(shop: JobShop, array: NDArray[Any]) -> tuple[str, ...]:
    """Return every way in which ``array`` fails to hold one distinct integer rank per task."""
    if array.ndim != 1:
        return (f"the ranks are one row of integers, one per operation, got shape {array.shape}",)
    issues: list[str] = []
    if array.size != shop.size:
        issues.append(f"there are {array.size} ranks for the {shop.size} operations")
    if array.size and array.dtype.kind not in "iu":
        issues.append(f"the ranks are integers, got {array.dtype}")
    else:
        values, counts = np.unique(array, return_counts=True)
        repeated = values[counts > 1]
        if repeated.size:
            issues.append(f"ranks occur more than once: {repeated[:5].tolist()}")
    return tuple(issues)


def semi_active(shop: JobShop, sequence: JobSequence) -> Decoded:
    """Decode a job sequence into a semi-active schedule.

    The operations are placed in sequence order. Each starts at the latest of the
    completion of its job predecessor, the availability of its machine, which is
    the completion of the operation placed on the machine before it, and its
    release time. No operation fills an idle interval left earlier on its machine,
    so every machine processes its operations in sequence order.

    Args:
        shop: The job shop.
        sequence: A job sequence of ``shop``.

    Returns:
        The schedule, whose sequence lists the operations in the order they were
        placed.

    Raises:
        ModelError: If ``sequence`` is not a job sequence of ``shop``.
    """
    tasks = _tasks(shop, sequence)
    job, machine, duration, release = shop.job, shop.machine, shop.duration, shop.release
    job_free = [0.0] * shop.jobs
    machine_free = [0.0] * shop.machines
    starts = [0.0] * shop.size
    ends = [0.0] * shop.size
    for task in tasks:
        owner = job[task]
        used = machine[task]
        start = max(job_free[owner], machine_free[used], release[task])
        end = start + duration[task]
        job_free[owner] = machine_free[used] = end
        starts[task] = start
        ends[task] = end
    return Decoded(tuple(tasks), (0,) * shop.size, tuple(starts), tuple(ends))


def giffler_thompson(shop: JobShop, priority: JobSequence, *, non_delay: bool = False) -> Decoded:
    """Decode a job sequence into an active schedule, or into a non-delay one.

    The operations that can be scheduled next are the first unscheduled
    operation of each job. Each can start at the latest of the completion of its
    job predecessor, the availability of its machine, which is the completion of
    the last operation scheduled on it, and its release time, and would complete
    its processing time later. An operation comes before another in the priority
    when its occurrence in ``priority`` comes first.

    Active, the procedure of Giffler and Thompson (1960): let C* be the earliest
    completion time of an operation that can be scheduled next. Of the
    operations that attain C*, the one that comes first in the priority fixes
    the machine M*. The conflict set holds that operation and the operations on
    M* that can start before C*, which include it unless it takes no time. The
    operation of the conflict set that comes first in the priority is scheduled
    at its earliest start.

    Non-delay, with ``non_delay=True``: let S* be the earliest start time of an
    operation that can be scheduled next. Of the operations that attain S*, the
    one that comes first in the priority is scheduled at S*. It fixes the
    machine, and it comes first in the priority among the operations on that
    machine that can start at S*, which form the conflict set of the non-delay
    rule.

    Positions in a sequence are unique, so no other tie arises. Each operation is
    appended to its machine, so no idle interval is filled later. Each step looks
    at every job, so a decode takes time proportional to the number of
    operations times the number of jobs. :func:`giffler_thompson_ranked` runs the
    same procedure on a priority that is one rank per operation.

    Args:
        shop: The job shop.
        priority: A job sequence of ``shop``, which ranks the operations.
        non_delay: Build a non-delay schedule instead of an active one.

    Returns:
        The schedule, whose sequence lists the operations in the order they were
        scheduled.

    Raises:
        ModelError: If ``priority`` is not a job sequence of ``shop``.
    """
    rank = [0] * shop.size
    for position, task in enumerate(_tasks(shop, priority)):
        rank[task] = position
    return _giffler_thompson(shop, rank, non_delay)


def giffler_thompson_ranked(
    shop: JobShop, rank: OperationRanks, *, non_delay: bool = False
) -> Decoded:
    """Decode per-operation ranks into an active schedule, or into a non-delay one.

    The procedure is that of :func:`giffler_thompson`, with the priority given as
    one rank per operation instead of a job sequence: ``rank[task]`` is the rank
    of task ``task``, and an operation comes before another in the priority when
    its rank is lower. Of the operations that attain C*, the one of lowest rank
    fixes the machine M*, and the operation of lowest rank in the conflict set is
    scheduled. With non-delay, the operation of lowest rank among those that
    attain S* is scheduled.

    The ranks must be pairwise different, which leaves no other tie. A
    priority rule makes them so by sorting the operations on a key such as the
    rule's value, then the job, then the operation, and ranking them by their
    place in the sort. Only the first unscheduled operation of each job can be
    scheduled next, so the ranks of one job's operations are never compared with
    each other: a rank may put an operation ahead of its job predecessor, which a
    job sequence cannot do. When the ranks are the positions of a job
    sequence's occurrences, the result is that of :func:`giffler_thompson`.

    Args:
        shop: The job shop.
        rank: One integer per task, indexed by task index, all different.
        non_delay: Build a non-delay schedule instead of an active one.

    Returns:
        The schedule, whose sequence lists the operations in the order they were
        scheduled.

    Raises:
        ModelError: If ``rank`` does not hold one integer per task of ``shop``, or
            two tasks share a rank.
    """
    return _giffler_thompson(shop, _ranks(shop, rank), non_delay)


def _giffler_thompson(shop: JobShop, rank: list[int], non_delay: bool) -> Decoded:
    """Run the procedure of :func:`giffler_thompson` on one distinct rank per task."""
    operations, machine, duration = shop.operations, shop.machine, shop.duration
    release = shop.release
    step = [0] * shop.jobs
    job_free = [0.0] * shop.jobs
    machine_free = [0.0] * shop.machines
    # The earliest start of each job's next operation, and the key that picks the
    # operation fixing the machine: its earliest completion, or start, then its rank.
    earliest = [0.0] * shop.jobs
    key: list[tuple[float, int]] = [(0.0, 0)] * shop.jobs
    # The jobs whose next operation runs on each machine, and the jobs not done.
    waiting: list[list[int]] = [[] for _ in range(shop.machines)]
    unfinished = list(range(shop.jobs))

    def update(job: int) -> None:
        task = operations[job][step[job]]
        start = max(job_free[job], machine_free[machine[task]], release[task])
        earliest[job] = start
        key[job] = (start if non_delay else start + duration[task], rank[task])

    for job in unfinished:
        update(job)
        waiting[machine[operations[job][0]]].append(job)
    starts = [0.0] * shop.size
    ends = [0.0] * shop.size
    placed: list[int] = []
    while unfinished:
        # The job of the operation that fixes the machine M*, and the jobs waiting for M*.
        chosen = min(unfinished, key=key.__getitem__)
        used = machine[operations[chosen][step[chosen]]]
        queue = waiting[used]
        if not non_delay:
            # The conflict set: that operation and those on M* that can start before C*.
            completion, best = key[chosen]
            for job in queue:
                if earliest[job] < completion and key[job][1] < best:
                    chosen, best = job, key[job][1]
        task = operations[chosen][step[chosen]]
        start = earliest[chosen]
        end = start + duration[task]
        starts[task] = start
        ends[task] = end
        placed.append(task)
        job_free[chosen] = machine_free[used] = end
        queue.remove(chosen)
        # The machine is busy until ``end`` now, which can delay every job waiting for it.
        for job in queue:
            update(job)
        step[chosen] += 1
        if step[chosen] < len(operations[chosen]):
            update(chosen)
            waiting[machine[operations[chosen][step[chosen]]]].append(chosen)
        else:
            unfinished.remove(chosen)
    return Decoded(tuple(placed), (0,) * shop.size, tuple(starts), tuple(ends))


class JobSequenceDecoder:
    """Decodes the job sequences of one job shop with one schedule builder.

    Args:
        shop: The job shop.
        builder: ``"semi-active"`` decodes with :func:`semi_active`, ``"active"``
            with :func:`giffler_thompson`, and ``"non-delay"`` with
            :func:`giffler_thompson` and ``non_delay=True``.

    Attributes:
        shop: The job shop.
        builder: The name of the builder.

    Raises:
        ValueError: If ``builder`` is not one of :data:`BUILDERS`.
    """

    def __init__(self, shop: JobShop, builder: BuilderName) -> None:
        if builder not in BUILDERS:
            raise ValueError(f"unknown builder {builder!r}; expected one of {list(BUILDERS)}")
        self.shop = shop
        self.builder = builder

    def decode(self, sequence: JobSequence) -> Decoded:
        """Decode a job sequence with the builder.

        Raises:
            ModelError: If ``sequence`` is not a job sequence of the shop.
        """
        if self.builder == "semi-active":
            return semi_active(self.shop, sequence)
        return giffler_thompson(self.shop, sequence, non_delay=self.builder == "non-delay")
