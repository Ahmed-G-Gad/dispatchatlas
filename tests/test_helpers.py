"""The shared problem factories state what they promise."""

from __future__ import annotations

import itertools

from tests.helpers import small_job_shop


def test_the_small_job_shop_has_three_jobs_on_three_machines() -> None:
    problem = small_job_shop()
    assert problem.size == 9
    assert sorted(problem.groups) == ["j0", "j1", "j2"]
    assert {len(members) for members in problem.groups.values()} == {3}
    assert [resource.id for resource in problem.resources] == ["m0", "m1", "m2"]
    for members in problem.groups.values():
        machines = [next(iter(problem.tasks[i].modes[0].demands)) for i in members]
        assert sorted(machines) == ["m0", "m1", "m2"]


def test_the_small_job_shop_has_the_optimal_makespan_its_docstring_states() -> None:
    problem = small_job_shop()
    routes = {
        job: [
            (next(iter(problem.tasks[i].modes[0].demands)), problem.tasks[i].modes[0].duration)
            for i in members
        ]
        for job, members in problem.groups.items()
    }
    loads: dict[str, float] = {}
    for route in routes.values():
        for machine, duration in route:
            loads[machine] = loads.get(machine, 0.0) + duration
    assert sorted(loads.values()) == [6, 7, 9]
    assert max(sum(duration for _, duration in route) for route in routes.values()) == 8

    # Every semi-active schedule follows from one sequence of jobs, a job's k-th
    # appearance standing for its k-th task, and an optimal schedule is among them.
    best = float("inf")
    for sequence in set(itertools.permutations([job for job in routes for _ in routes[job]])):
        step = dict.fromkeys(routes, 0)
        job_free = dict.fromkeys(routes, 0.0)
        machine_free = dict.fromkeys(loads, 0.0)
        for job in sequence:
            machine, duration = routes[job][step[job]]
            start = max(job_free[job], machine_free[machine])
            job_free[job] = machine_free[machine] = start + duration
            step[job] += 1
        best = min(best, max(job_free.values()))
    assert best == 11
