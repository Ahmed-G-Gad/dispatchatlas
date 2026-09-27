"""Classic instances with zero processing times, and the model rules they meet.

hel1 and hel2 (Heller 1960) are permutation flow shops with zero processing
times, and orb07 (Applegate and Cook 1991) is a job shop with one. A zero-time
operation occupies an instant on its machine and takes its place in the
machine's job order, so decoding a job permutation must reproduce the classic
completion-time recurrence of the permutation flow shop.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from dispatchatlas.benchmarks import default_registry
from dispatchatlas.benchmarks.formats import read_orlib_flow_shop, read_orlib_job_shop
from dispatchatlas.benchmarks.machines import ParallelMachines, parallel_machine_problem
from dispatchatlas.benchmarks.shop import flow_shop_problem, job_shop_problem, open_shop_problem
from dispatchatlas.core import ConfigError, ModelError, check_schedule
from dispatchatlas.core.model import Mode, Problem, Resource, Task
from dispatchatlas.solvers.operators import random_order
from dispatchatlas.solvers.scheduling import Decoder
from tests.benchmarks.fixtures import HEL1, HEL2, ORB07

REGISTRY = default_registry(plugins=False)


def processing_times(text: str) -> list[list[int]]:
    """Return ``p[j][k]`` of an OR-Library flow-shop text, parsed without the reader."""
    rows = [line.split() for line in text.splitlines()[1:] if line.strip()]
    jobs, machines = int(rows[0][0]), int(rows[0][1])
    times = []
    for row in rows[1 : 1 + jobs]:
        values = [int(token) for token in row]
        assert values[0::2] == list(range(machines))
        times.append(values[1::2])
    return times


def completion_times(times: list[list[int]], sequence: list[int]) -> dict[int, list[int]]:
    """Return ``C[j][k]`` of the permutation flow-shop recurrence for a job sequence.

    ``C[j][k] = max(C[previous job][k], C[j][k - 1]) + p[j][k]``, with zero for
    a missing previous job or machine.
    """
    machines = len(times[0])
    result: dict[int, list[int]] = {}
    previous = [0] * machines
    for job in sequence:
        row = []
        ready = 0
        for k in range(machines):
            ready = max(ready, previous[k]) + times[job][k]
            row.append(ready)
        result[job] = row
        previous = row
    return result


@pytest.mark.parametrize(("name", "text", "zeros"), [("hel1", HEL1, 32), ("hel2", HEL2, 4)])
def test_heller_permutations_follow_the_flow_shop_recurrence(name, text, zeros):
    times = processing_times(text)
    assert sum(time == 0 for row in times for time in row) == zeros
    problem = read_orlib_flow_shop(text, problem_id=name)
    assert problem.permutation
    groups = sorted(problem.groups)
    decoder = Decoder(problem)
    rng = random.Random(name)
    sequence = list(range(len(times)))
    for _ in range(5):
        rng.shuffle(sequence)
        order = [index for job in sequence for index in problem.groups[groups[job]]]
        schedule = decoder.schedule(order)
        expected = completion_times(times, sequence)
        assert schedule.makespan == expected[sequence[-1]][-1]
        for job, row in expected.items():
            ends = [schedule.by_task[problem.tasks[i].id].end for i in problem.groups[groups[job]]]
            assert ends == row
        assert check_schedule(problem, schedule) == ()


def test_orb07_decodes_to_feasible_schedules():
    problem = read_orlib_job_shop(ORB07, problem_id="orb07")
    assert [task.id for task in problem.tasks if task.modes[0].duration == 0.0] == ["j9-o9"]
    rng = random.Random(7)
    for insertion in (True, False):
        decoder = Decoder(problem, insertion=insertion)
        for _ in range(10):
            schedule = decoder.schedule(random_order(problem, rng))
            assert check_schedule(problem, schedule) == ()
            zero = schedule.by_task["j9-o9"]
            assert zero.start == zero.end >= schedule.by_task["j9-o8"].end


def test_permutation_flow_shops_accept_zero_times():
    problem = flow_shop_problem([[0.0, 2.0], [3.0, 0.0]])
    schedule = Decoder(problem).schedule([2, 3, 0, 1])
    assert schedule.makespan == 5.0
    assert check_schedule(problem, schedule) == ()


def _single_task(demand: float, capacity: float) -> Problem:
    return Problem(
        "single",
        tasks=(Task("a", (Mode(1.0, {"m": demand}),)),),
        resources=(Resource("m", capacity=capacity),),
        permutation=True,
    )


@pytest.mark.parametrize(
    "build",
    [
        # A job that visits a machine twice.
        lambda: dataclasses.replace(
            job_shop_problem([[(0, 1.0), (1, 2.0), (0, 3.0)]]), permutation=True
        ),
        # An open shop, whose job resource serves every operation of the job.
        lambda: dataclasses.replace(open_shop_problem([[1.0, 2.0]]), permutation=True),
        # A machine of capacity 2, and a task holding half a machine.
        lambda: _single_task(1.0, 2.0),
        lambda: _single_task(0.5, 1.0),
    ],
)
def test_permutation_problems_that_break_the_model_rules_are_refused(build):
    with pytest.raises(ModelError, match="permutation problem"):
        build()


def test_zero_times_are_refused_on_machines_with_setup_times():
    times = [[0.0, 2.0], [1.0, 3.0]]
    with pytest.raises(ModelError, match="zero-duration mode on machine 'm0'"):
        parallel_machine_problem(times, setups=[[[0.0, 1.0], [1.0, 0.0]], [[0.0] * 2] * 2])
    problem = parallel_machine_problem(times, setups=[[[0.0] * 2] * 2, [[0.0, 1.0], [1.0, 0.0]]])
    assert problem.setups is not None
    assert problem.setups.resources == {"m1"}


def test_the_parallel_machine_family_never_draws_zero_times():
    with pytest.raises(ValueError, match="min_time"):
        ParallelMachines(min_time=0, max_setup=9)
    with pytest.raises(ConfigError, match="min_time"):
        REGISTRY.create("parallel-machines", min_time=0, max_setup=9)
