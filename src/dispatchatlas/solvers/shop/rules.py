"""Dispatching rules for the job shop under the Giffler-Thompson procedure.

A dispatching rule ranks the operations by one static attribute, and the
procedure of Giffler and Thompson (1960, doi:10.1287/opre.8.4.487) builds one
active schedule from the ranks, in which a lower rank comes first in the
priority. The solvers here run
:func:`~dispatchatlas.solvers.shop.decoding.giffler_thompson_ranked`, because a
job sequence cannot carry these rules: it lists the operations of a job in
their route order, and shortest processing time can rank a job's later
operation ahead of its earlier one.

- :class:`GtShortestProcessingTime`: in each conflict set the operation with the
  shortest processing time is scheduled (Smith 1956,
  doi:10.1002/nav.3800030106).
- :class:`GtMostWorkRemaining`: the operation whose job has the most processing
  time left is scheduled. The work left counts the operation's own processing
  time and that of the job's later operations (Demirkol et al. 1998, p. 139,
  doi:10.1016/S0377-2217(97)00019-2; Nguyen et al. 2013, p. 627,
  doi:10.1109/TEVC.2012.2227326).

**Ties.** The ranks must be pairwise different, so the operations are sorted by
the key ``(rule value, job, operation)`` and ranked by their place in the sort.
The rule value is ranked in ascending order: the processing time for the first
rule, and minus the work remaining for the second. Operations of equal value
therefore go in order of their job index, and then of their position in the job,
which is the operation index. The procedure compares the first unscheduled
operation of each job only, so the operation index only makes the ranks
different. The same key fixes the machine when several operations attain the
earliest completion time.

A rule is deterministic: a run builds one schedule and evaluates it once, and
the seed has no effect. Both rules accept every feature in
:data:`~dispatchatlas.solvers.shop.instance.JOB_SHOP_FEATURES`; the build starts
no operation before its release time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import ClassVar

from dispatchatlas.core.model import Problem
from dispatchatlas.core.solver import SolveResult, SolverInfo, StopCriteria
from dispatchatlas.solvers.search import Search
from dispatchatlas.solvers.shop.decoding import giffler_thompson_ranked
from dispatchatlas.solvers.shop.instance import JOB_SHOP_FEATURES, JobShop

__all__ = ["GtMostWorkRemaining", "GtShortestProcessingTime"]


def _rank_by(shop: JobShop, values: Sequence[float]) -> list[int]:
    """Return each task's rank when the tasks are sorted by ``(value, job, operation)``."""
    place = [0] * shop.size
    for route in shop.operations:
        for index, task in enumerate(route):
            place[task] = index
    ordered = sorted(range(shop.size), key=lambda task: (values[task], shop.job[task], place[task]))
    rank = [0] * shop.size
    for position, task in enumerate(ordered):
        rank[task] = position
    return rank


def _work_remaining(shop: JobShop) -> list[float]:
    """Return the processing time of each task and of every later task of its job."""
    remaining = [0.0] * shop.size
    for route in shop.operations:
        total = 0.0
        for task in reversed(route):
            total += shop.duration[task]
            remaining[task] = total
    return remaining


def _solve(
    info: SolverInfo,
    problem: Problem,
    stop: StopCriteria,
    seed: int,
    values: Callable[[JobShop], Sequence[float]],
) -> SolveResult:
    """Build the one schedule that ranking the tasks by ``values`` gives, and evaluate it once."""
    search = Search(problem, stop, seed, info)
    shop = JobShop.from_problem(problem)
    rank = _rank_by(shop, values(shop))
    with search:
        search.assess_built(partial(giffler_thompson_ranked, shop, rank))
    return search.result()


@dataclass(frozen=True)
class GtShortestProcessingTime:
    """Giffler-Thompson with shortest processing time: the shortest operation goes first.

    The run builds the active schedule of the Giffler-Thompson procedure in which
    the operation of each conflict set with the shortest processing time is
    scheduled, and evaluates it once. Operations of equal processing time go in
    order of their job index, then of their operation index, and the same order
    picks the operation that fixes the machine among those that attain the
    earliest completion time.
    """

    info: ClassVar[SolverInfo] = SolverInfo(
        id="gt-shortest-processing-time",
        name="Giffler-Thompson shortest processing time",
        family="dispatching-rule",
        summary="Builds one active job-shop schedule with the Giffler-Thompson procedure, "
        "scheduling the operation with the shortest processing time in each conflict set.",
        references=("giffler1960algorithms", "smith1956various"),
        stochastic=False,
        iterative=False,
        supports=JOB_SHOP_FEATURES,
        shape="job-shop",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Build the schedule of the shortest-processing-time rule and evaluate it once."""
        return _solve(self.info, problem, stop, seed, lambda shop: shop.duration)


@dataclass(frozen=True)
class GtMostWorkRemaining:
    """Giffler-Thompson with most work remaining: the job with the most work left goes first.

    The run builds the active schedule of the Giffler-Thompson procedure in which
    the operation whose job has the most processing time left is scheduled in
    each conflict set, and evaluates it once. The work left counts the
    operation's own processing time and that of the job's later operations
    (Demirkol et al. 1998, p. 139; Nguyen et al. 2013, p. 627), summed from the
    end of the job, so it is exact for integer processing times. Operations with
    equal work left go in order of their job index, then of their operation
    index, and the same order picks the operation that fixes the machine among
    those that attain the earliest completion time.
    """

    info: ClassVar[SolverInfo] = SolverInfo(
        id="gt-most-work-remaining",
        name="Giffler-Thompson most work remaining",
        family="dispatching-rule",
        summary="Builds one active job-shop schedule with the Giffler-Thompson procedure, "
        "scheduling the operation whose job has the most processing time left in each "
        "conflict set.",
        references=("giffler1960algorithms", "demirkol1998benchmarks", "nguyen2013computational"),
        stochastic=False,
        iterative=False,
        supports=JOB_SHOP_FEATURES,
        shape="job-shop",
    )

    def solve(self, problem: Problem, stop: StopCriteria, seed: int = 0) -> SolveResult:
        """Build the schedule of the most-work-remaining rule and evaluate it once."""
        return _solve(
            self.info,
            problem,
            stop,
            seed,
            lambda shop: [-work for work in _work_remaining(shop)],
        )
