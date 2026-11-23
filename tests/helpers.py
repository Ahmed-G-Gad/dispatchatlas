"""Problem factories shared by the test suite."""

from __future__ import annotations

import random

from dispatchatlas.core import (
    Dependency,
    Link,
    Mode,
    Network,
    Problem,
    Resource,
    SetupTimes,
    Task,
)


def chain_problem() -> Problem:
    """Two machines, three tasks, one dependency. Optimal makespan is 5."""
    return Problem(
        "chain",
        tasks=(Task.on("a", "m1", 3), Task.on("b", "m1", 2, due=4), Task.on("c", "m2", 4)),
        resources=(Resource("m1"), Resource("m2")),
        dependencies=(Dependency("a", "b"),),
    )


def small_job_shop() -> Problem:
    """Three jobs on three machines, each job a chain through every machine once.

    Job ``j0`` visits machines ``m0``, ``m1``, ``m2`` for 3, 2, and 2; ``j1``
    visits ``m0``, ``m2``, ``m1`` for 2, 1, and 4; ``j2`` visits ``m1``,
    ``m0``, ``m2`` for 3, 2, and 3. The machines have 7, 9, and 6 units of
    work and the longest job takes 8, yet the optimal makespan is 11.
    """
    routes = (
        ((0, 3), (1, 2), (2, 2)),
        ((0, 2), (2, 1), (1, 4)),
        ((1, 3), (0, 2), (2, 3)),
    )
    tasks: list[Task] = []
    dependencies: list[Dependency] = []
    for job, route in enumerate(routes):
        for step, (machine, duration) in enumerate(route):
            tasks.append(Task.on(f"j{job}-o{step}", f"m{machine}", duration, group=f"j{job}"))
            if step:
                dependencies.append(Dependency(f"j{job}-o{step - 1}", f"j{job}-o{step}"))
    return Problem(
        "small-job-shop",
        tasks=tuple(tasks),
        resources=tuple(Resource(f"m{machine}") for machine in range(3)),
        dependencies=tuple(dependencies),
    )


def random_problem(
    seed: int,
    *,
    tasks: int = 12,
    machines: int = 3,
    pools: int = 1,
    max_modes: int = 3,
    density: float = 0.2,
    setups: bool = False,
    network: bool = False,
    releases: bool = True,
    deadlines: bool = False,
    lags: bool = True,
) -> Problem:
    """Return a random problem that exercises every decoder feature."""
    rng = random.Random(seed)
    resources = [
        Resource(f"m{r}", tier=("edge" if r % 2 == 0 else "cloud") if network else None)
        for r in range(machines)
    ]
    resources += [Resource(f"pool{r}", capacity=float(rng.randint(2, 4))) for r in range(pools)]
    items = []
    for i in range(tasks):
        modes = []
        for _ in range(rng.randint(1, max_modes)):
            machine = f"m{rng.randrange(machines)}"
            demands = {machine: 1.0}
            if pools and rng.random() < 0.4:
                pool = f"pool{rng.randrange(pools)}"
                demands[pool] = float(rng.randint(1, 2))
            modes.append(Mode(float(rng.randint(1, 9)), demands, location=machine))
        items.append(
            Task(
                f"t{i}",
                tuple(modes),
                release=float(rng.randint(0, 5)) if releases and rng.random() < 0.3 else 0.0,
                due=float(rng.randint(5, 40)) if rng.random() < 0.3 else None,
                deadline=float(rng.randint(10, 60)) if deadlines and rng.random() < 0.3 else None,
                group=f"g{i % 4}",
            )
        )
    dependencies = []
    for j in range(tasks):
        for i in range(j):
            if rng.random() < density:
                dependencies.append(
                    Dependency(
                        f"t{i}",
                        f"t{j}",
                        lag=float(rng.randint(0, 2)) if lags and rng.random() < 0.3 else 0.0,
                        data=float(rng.randint(0, 50)) if network else 0.0,
                    )
                )
    setup_times = None
    if setups:
        times = {}
        for r in range(machines):
            for a in [None, *range(tasks)]:
                for b in range(tasks):
                    if a != b and rng.random() < 0.3:
                        times[(f"m{r}", None if a is None else f"t{a}", f"t{b}")] = float(
                            rng.randint(1, 3)
                        )
        setup_times = SetupTimes(times)
    links = None
    if network:
        links = Network(
            (
                Link(
                    "edge",
                    "edge",
                    bandwidth=10.0,
                    latency=0.5,
                    energy_per_unit=0.1,
                    cost_per_unit=0.01,
                ),
                Link(
                    "edge",
                    "cloud",
                    bandwidth=5.0,
                    latency=2.0,
                    energy_per_unit=0.2,
                    cost_per_unit=0.02,
                ),
                Link("cloud", "cloud", bandwidth=50.0, latency=0.1),
            )
        )
    return Problem(
        f"random-{seed}",
        tasks=tuple(items),
        resources=tuple(resources),
        dependencies=tuple(dependencies),
        setups=setup_times,
        network=links,
        objectives=("makespan", "total_tardiness", "total_completion_time"),
    )
