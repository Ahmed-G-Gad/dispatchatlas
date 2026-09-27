"""Taillard's instances against the verification data published with them.

The matrices below are copied from the OR-Library files that distribute
Taillard's (1993) generator: ``flowshop2.txt`` prints the processing times of
ta001, and ``jobshop2.txt`` and ``openshop.txt`` print a 4 x 4 instance whose
seeds are those of the first open shop. The first job of ta01 is copied from
the published instance file in the JSPLIB collection.
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from dispatchatlas.benchmarks import taillard
from dispatchatlas.benchmarks.taillard import (
    MODULUS,
    MULTIPLIER,
    flow_shop_instances,
    flow_shop_matrix,
    job_shop_instances,
    job_shop_matrices,
    open_shop_instances,
    open_shop_matrices,
    taillard_flow_shop,
    taillard_instance,
    taillard_job_shop,
    taillard_open_shop,
    unif,
)
from dispatchatlas.core import ConfigError, check_schedule
from dispatchatlas.solvers.operators import random_order
from dispatchatlas.solvers.scheduling import Decoder

TA001 = [
    [54, 83, 15, 71, 77, 36, 53, 38, 27, 87, 76, 91, 14, 29, 12, 77, 32, 87, 68, 94],
    [79, 3, 11, 99, 56, 70, 99, 60, 5, 56, 3, 61, 73, 75, 47, 14, 21, 86, 5, 77],
    [16, 89, 49, 15, 89, 45, 60, 23, 57, 64, 7, 1, 63, 41, 63, 47, 26, 75, 77, 40],
    [66, 58, 31, 68, 78, 91, 13, 59, 49, 85, 85, 9, 39, 41, 56, 40, 54, 77, 51, 31],
    [58, 56, 20, 85, 53, 35, 53, 41, 69, 13, 86, 72, 8, 49, 47, 87, 58, 18, 68, 28],
]
"""``d[i][j]`` of ta001, machine ``i`` and job ``j``, from flowshop2.txt."""

VERIFICATION_TIMES = [[54, 9, 38, 95], [34, 15, 19, 34], [61, 89, 28, 7], [2, 70, 87, 29]]
VERIFICATION_MACHINES = [[3, 4, 1, 1], [1, 1, 2, 3], [4, 2, 3, 2], [2, 3, 4, 4]]
"""``d[i][j]`` and ``M[i][j]`` for seeds 1166510396 and 164000672, from jobshop2.txt."""

TA01_JOB_0 = [
    6,
    94,
    12,
    66,
    4,
    10,
    7,
    53,
    3,
    26,
    2,
    15,
    10,
    65,
    11,
    82,
    8,
    10,
    14,
    27,
    9,
    93,
    13,
    92,
    5,
    96,
    0,
    70,
    1,
    83,
]
"""The first job of ta01 as ``machine time`` pairs, machines numbered from 0."""

TABLES_DIGEST = "d74b12d4ba25930070b47edb168689b834246bb5e13a1b2fb41d805cc7b8e882"
"""SHA-256 of the seed tables as parsed from flowshop2.txt, jobshop2.txt, and openshop.txt."""


def test_unif_is_the_lehmer_generator():
    rng = random.Random(0)
    for _ in range(1000):
        seed = rng.randint(1, MODULUS - 1)
        following, value = unif(seed, 1, 99)
        assert following == MULTIPLIER * seed % MODULUS
        assert value == 1 + int(following / MODULUS * 99)


def test_flow_shop_generator_reproduces_ta001():
    assert flow_shop_matrix(873654221, 20, 5) == TA001


def test_job_shop_generator_reproduces_the_verification_matrices():
    assert job_shop_matrices(1166510396, 164000672, 4, 4) == (
        VERIFICATION_TIMES,
        VERIFICATION_MACHINES,
    )


def test_open_shop_generator_reproduces_the_verification_matrices():
    first = open_shop_instances()[0]
    assert (first.time_seed, first.machine_seed) == (1166510396, 164000672)
    assert open_shop_matrices(1166510396, 164000672, 4, 4) == (
        VERIFICATION_TIMES,
        VERIFICATION_MACHINES,
    )


def test_seed_tables_match_the_published_files():
    tables = {
        "flow": [[i.name, i.jobs, i.machines, i.time_seed] for i in flow_shop_instances()],
        "job": [
            [i.name, i.jobs, i.machines, i.time_seed, i.machine_seed] for i in job_shop_instances()
        ],
        "open": [
            [
                i.name,
                i.jobs,
                i.machines,
                i.time_seed,
                i.machine_seed,
                i.upper_bound,
                i.optimal,
                i.lower_bound,
            ]
            for i in open_shop_instances()
        ],
    }
    text = json.dumps(tables, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(text.encode()).hexdigest() == TABLES_DIGEST


def test_table_sizes_and_names():
    flows, jobs, opens = flow_shop_instances(), job_shop_instances(), open_shop_instances()
    assert [len(flows), len(jobs), len(opens)] == [120, 80, 60]
    assert (flows[0].name, flows[-1].name) == ("ta001", "ta120")
    assert (jobs[0].name, jobs[41].name, jobs[-1].name) == ("ta01", "ta42", "ta80")
    assert (opens[0].name, opens[-1].name) == ("tai4x4-1", "tai20x20-10")
    assert (flows[-1].jobs, flows[-1].machines) == (500, 20)
    assert (jobs[41].time_seed, jobs[41].machine_seed) == (2116959593, 1546338557)
    assert all(i.machine_seed is None for i in flows)
    assert sum(i.optimal for i in opens) == 39


def test_ta001_problem():
    problem = taillard_flow_shop("ta001")
    assert problem.id == "ta001"
    assert problem.size == 100
    assert problem.permutation
    for j in range(20):
        durations = [problem.task(f"j{j:02d}-m{k}").modes[0].duration for k in range(5)]
        assert durations == [TA001[k][j] for k in range(5)]
    assert problem.metadata["time_seed"] == 873654221
    assert problem.metadata["suite"] == "taillard-flow-shop"
    assert problem.family is None


def test_ta01_matches_the_published_instance():
    problem = taillard_job_shop("ta01")
    assert problem.size == 225
    route = []
    for p in range(15):
        mode = problem.task(f"j00-o{p:02d}").modes[0]
        route += [int(next(iter(mode.demands))[1:]), int(mode.duration)]
    assert route == TA01_JOB_0


def test_open_shop_problem_uses_the_verification_data_and_bounds():
    problem = taillard_open_shop("tai4x4-1")
    for j in range(4):
        for i in range(4):
            machine = VERIFICATION_MACHINES[i][j] - 1
            mode = problem.task(f"j{j}-m{machine}").modes[0]
            assert mode.duration == VERIFICATION_TIMES[i][j]
            assert set(mode.demands) == {f"m{machine}", f"job-{j}"}
    assert (problem.metadata["upper_bound"], problem.metadata["lower_bound"]) == (193, 186)
    assert problem.metadata["optimal"] is True


@pytest.mark.parametrize(
    ("problem_class", "instance", "name"),
    [
        ("flow-shop", "ta001", "ta001"),
        ("flow-shop", "TA1", "ta001"),
        ("flow-shop", 120, "ta120"),
        ("flow-shop", " 57 ", "ta057"),
        ("job-shop", "ta42", "ta42"),
        ("job-shop", 1, "ta01"),
        ("open-shop", "tai4x4-1", "tai4x4-1"),
        ("open-shop", "TAI_20x20_10", "tai20x20-10"),
        ("open-shop", "tai7_7_3", "tai7x7-3"),
        ("open-shop", 60, "tai20x20-10"),
    ],
)
def test_instances_are_found_by_name_or_number(problem_class, instance, name):
    assert taillard_instance(problem_class, instance).name == name


@pytest.mark.parametrize(
    ("problem_class", "instance"),
    [
        ("flow-shop", "ta000"),
        ("flow-shop", 121),
        ("flow-shop", "ta01x"),
        ("flow-shop", True),
        ("job-shop", 81),
        ("open-shop", "tai4x4-11"),
        ("open-shop", "tai6x6-1"),
        ("open-shop", "tai4x5-1"),
        ("tardy-shop", 1),
    ],
)
def test_unknown_instances_are_refused(problem_class, instance):
    with pytest.raises(ConfigError):
        taillard_instance(problem_class, instance)


@pytest.mark.parametrize(
    "build",
    [
        lambda: taillard_flow_shop("ta011"),
        lambda: taillard_job_shop("ta11"),
        lambda: taillard_open_shop("tai5x5-3"),
    ],
)
def test_taillard_problems_decode_to_feasible_schedules(build):
    problem = build()
    decoder = Decoder(problem)
    schedule = decoder.schedule(random_order(problem, random.Random(0)))
    assert check_schedule(problem, schedule) == ()


@pytest.mark.slow
def test_every_instance_builds():
    for item in flow_shop_instances():
        assert taillard_flow_shop(item.name).size == item.jobs * item.machines
    for item in job_shop_instances():
        assert taillard_job_shop(item.name).size == item.jobs * item.machines
    for item in open_shop_instances():
        assert taillard.taillard_open_shop(item.name).size == item.jobs * item.machines
