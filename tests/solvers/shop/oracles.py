"""Naive reference implementations that the tests of the job-shop rules compare the library with.

They follow the published procedures line by line and ignore speed: the
Giffler-Thompson oracle scans every candidate at every step, and shares no code
with the library's builders beyond the job-shop view.
"""

from __future__ import annotations

from collections.abc import Sequence

from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.shop.instance import JobShop


def naive_giffler_thompson(
    shop: JobShop, rank: Sequence[float], *, non_delay: bool = False
) -> Decoded:
    """Build the active, or non-delay, schedule of the Giffler-Thompson procedure for ranks.

    A lower rank comes first. Of the operations that can be scheduled next and
    attain the earliest completion time C*, the one of lowest rank fixes the
    machine M*; the conflict set holds it and the operations on M* that can
    start before C*; the one of lowest rank is scheduled at its earliest start.
    With ``non_delay``, the operation of lowest rank among those that attain the
    earliest start S* is scheduled at S*.
    """
    next_step = [0] * shop.jobs
    job_free = [0.0] * shop.jobs
    machine_free = [0.0] * shop.machines
    starts = [0.0] * shop.size
    ends = [0.0] * shop.size
    placed: list[int] = []
    for _ in range(shop.size):
        candidates: list[tuple[int, float]] = []
        for job, route in enumerate(shop.operations):
            if next_step[job] < len(route):
                task = route[next_step[job]]
                start = max(job_free[job], machine_free[shop.machine[task]], shop.release[task])
                candidates.append((task, start))
        if non_delay:
            earliest = min(start for _, start in candidates)
            chosen = min(
                (task for task, start in candidates if start == earliest), key=rank.__getitem__
            )
        else:
            completion = min(start + shop.duration[task] for task, start in candidates)
            fixing = min(
                (task for task, start in candidates if start + shop.duration[task] == completion),
                key=rank.__getitem__,
            )
            conflict = [
                task
                for task, start in candidates
                if shop.machine[task] == shop.machine[fixing]
                and (start < completion or task == fixing)
            ]
            chosen = min(conflict, key=rank.__getitem__)
        start = dict(candidates)[chosen]
        end = start + shop.duration[chosen]
        starts[chosen], ends[chosen] = start, end
        job_free[shop.job[chosen]] = machine_free[shop.machine[chosen]] = end
        next_step[shop.job[chosen]] += 1
        placed.append(chosen)
    return Decoded(tuple(placed), (0,) * shop.size, tuple(starts), tuple(ends))
