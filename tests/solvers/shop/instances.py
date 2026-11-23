"""Job shops, job sequences, and schedule checks for the tests of the job-shop package."""

from __future__ import annotations

import random
from functools import cache

import numpy as np
from numpy.typing import NDArray

from dispatchatlas.benchmarks import taillard_job_shop
from dispatchatlas.core import Dependency, Problem, Resource, Task, derive_seed
from dispatchatlas.solvers.scheduling import Decoded
from dispatchatlas.solvers.shop.instance import JobShop


@cache
def ta01() -> JobShop:
    """Taillard's first job shop, 15 jobs on 15 machines, as the library rebuilds it."""
    return JobShop.from_problem(taillard_job_shop("ta01"))


def random_job_shop(
    index: int, *, jobs: int = 4, machines: int = 3, releases: bool = True, revisits: bool = False
) -> Problem:
    """Return a small random job shop.

    Each job visits every machine once in a random order, or, with ``revisits``,
    draws the machine of each of its ``machines`` operations at random, so that
    it may return to a machine, even straight away. Processing times are 1 to
    9. With ``releases``, about a third of the operations get a release time of
    1 to 12, which may fall after their job predecessor completes.
    """
    rng = random.Random(derive_seed(7, "tests/shop/random-job-shop", index))
    tasks: list[Task] = []
    dependencies: list[Dependency] = []
    for job in range(jobs):
        route = (
            [rng.randrange(machines) for _ in range(machines)]
            if revisits
            else rng.sample(range(machines), machines)
        )
        for step, machine in enumerate(route):
            release = float(rng.randint(1, 12)) if releases and rng.random() < 0.3 else 0.0
            task_id = f"j{job}-o{step}"
            tasks.append(
                Task.on(
                    task_id,
                    f"m{machine}",
                    float(rng.randint(1, 9)),
                    release=release,
                    group=f"j{job}",
                )
            )
            if step:
                dependencies.append(Dependency(f"j{job}-o{step - 1}", task_id))
    return Problem(
        f"random-job-shop-{index}",
        tasks=tuple(tasks),
        resources=tuple(Resource(f"m{machine}") for machine in range(machines)),
        dependencies=tuple(dependencies),
    )


def random_sequences(shop: JobShop, count: int, namespace: str) -> list[NDArray[np.intp]]:
    """Return ``count`` uniformly random job sequences of ``shop``, seeded by ``namespace``."""
    rng = np.random.default_rng(derive_seed(7, namespace))
    return [rng.permutation(shop.slot_job) for _ in range(count)]


def _ready(shop: JobShop, decoded: Decoded, task: int) -> float:
    """Return when ``task`` may start: after its release and its job predecessor."""
    before = shop.predecessor[task]
    return max(shop.release[task], decoded.ends[before] if before >= 0 else 0.0)


def _by_machine(shop: JobShop, decoded: Decoded) -> list[list[int]]:
    """Return the operations of each machine by start time, then completion time."""
    orders: list[list[int]] = [[] for _ in range(shop.machines)]
    for task in sorted(range(shop.size), key=lambda t: (decoded.starts[t], decoded.ends[t])):
        orders[shop.machine[task]].append(task)
    return orders


def left_shift(shop: JobShop, decoded: Decoded) -> str | None:
    """Describe an operation that could start earlier without delaying another, if any.

    An operation can when it starts later than its job, its machine, and its
    release time allow, or when it fits into an idle interval of its machine
    before its start, after its release and its job predecessor. A schedule
    with no such operation is active.
    """
    for order in _by_machine(shop, decoded):
        free = 0.0
        idle: list[tuple[float, float]] = []
        for task in order:
            ready = _ready(shop, decoded, task)
            start = decoded.starts[task]
            if start > max(ready, free):
                return f"task {task} starts at {start}, not at {max(ready, free)}"
            for low, high in idle:
                if max(low, ready) + shop.duration[task] <= high:
                    return f"task {task} fits into the idle interval [{low}, {high})"
            if start > free:
                idle.append((free, start))
            free = decoded.ends[task]
    return None


def idle_while_ready(shop: JobShop, decoded: Decoded) -> str | None:
    """Describe an operation whose machine idles while the operation is ready, if any.

    A schedule with no such operation is non-delay.
    """
    for order in _by_machine(shop, decoded):
        idle: list[tuple[float, float]] = []
        free = 0.0
        for task in order:
            start = decoded.starts[task]
            if start > free:
                idle.append((free, start))
            free = decoded.ends[task]
        for task in order:
            ready = _ready(shop, decoded, task)
            start = decoded.starts[task]
            for low, high in idle:
                if max(low, ready) < min(high, start):
                    return (
                        f"task {task} is ready at {ready} while its machine idles in "
                        f"[{low}, {high})"
                    )
    return None
