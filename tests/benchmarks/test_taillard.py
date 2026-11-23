"""Taillard's instances against the verification data published with them.

The matrices below are copied from the OR-Library files that distribute
Taillard's (1993) generator: ``flowshop2.txt`` prints the processing times of
ta001, and ``jobshop2.txt`` and ``openshop.txt`` print a 4 x 4 instance whose
seeds are those of the first open shop. The first job of ta01 is copied from
the published instance file in the JSPLIB collection. The last tests check the route digests
that the library stores for ta01 to ta80 and the function that compares a rebuilt job shop
with them.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import replace
from typing import Any

import pytest

from dispatchatlas.benchmarks import taillard
from dispatchatlas.benchmarks.shop import job_shop_problem
from dispatchatlas.benchmarks.taillard import (
    JOB_SHOP_ROUTE_DIGESTS,
    MODULUS,
    MULTIPLIER,
    check_taillard_job_shop,
    flow_shop_instances,
    flow_shop_matrix,
    job_shop_instances,
    job_shop_matrices,
    job_shop_route_digest,
    open_shop_instances,
    open_shop_matrices,
    taillard_flow_shop,
    taillard_instance,
    taillard_job_shop,
    taillard_job_shop_seeded,
    taillard_open_shop,
    unif,
)
from dispatchatlas.core import (
    ConfigError,
    Dependency,
    ModelError,
    Problem,
    Resource,
    Task,
    check_schedule,
)
from dispatchatlas.solvers.operators import random_order
from dispatchatlas.solvers.scheduling import Decoder
from tests.helpers import random_problem

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


# ------------------------------------------------------------ the stored route digests


def _routes(name: str) -> list[list[tuple[int, float]]]:
    """Return the routes of a published job shop straight from the generator's matrices.

    Job ``j`` visits, in order, machine ``M[i][j] - 1`` for ``d[i][j]``.
    """
    item = taillard_instance("job-shop", name)
    assert item.machine_seed is not None
    durations, order = job_shop_matrices(
        item.time_seed, item.machine_seed, item.jobs, item.machines
    )
    return [
        [(order[i][j] - 1, float(durations[i][j])) for i in range(item.machines)]
        for j in range(item.jobs)
    ]


def _oracle(name: str) -> str:
    """Return the route digest of a published job shop written out from its routes alone.

    One line for each job, the name and then ``m<machine>:<time>`` for each operation, the
    machine index zero-padded to the width the shop builders give resource ids.
    """
    width = len(str(taillard_instance("job-shop", name).machines - 1))
    lines = [
        f"{name} {' '.join(f'm{machine:0{width}d}:{time:g}' for machine, time in route)}\n"
        for route in _routes(name)
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def _rebuilt(name: str, routes: list[list[tuple[int, float]]]) -> Problem:
    """Return a job shop named ``name`` with the given routes, built by the shop builder."""
    item = taillard_instance("job-shop", name)
    return job_shop_problem(routes, machines=item.machines, problem_id=name)


def test_every_published_job_shop_has_a_stored_route_digest():
    names = [item.name for item in job_shop_instances()]
    assert list(JOB_SHOP_ROUTE_DIGESTS) == names
    assert (names[0], names[-1]) == ("ta01", "ta80")
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in JOB_SHOP_ROUTE_DIGESTS.values())
    assert len(set(JOB_SHOP_ROUTE_DIGESTS.values())) == 80
    with pytest.raises(TypeError):
        JOB_SHOP_ROUTE_DIGESTS["ta01"] = "0" * 64  # type: ignore[index]


def test_the_stored_route_digests_are_those_of_the_routes_the_generator_draws():
    # The oracle reads the generator's matrices and no problem, so it is independent of the
    # builder and of the digest function; the check of every rebuilt problem against these
    # entries is made by test_the_published_job_shops_keep_their_routes.
    for name, stored in JOB_SHOP_ROUTE_DIGESTS.items():
        assert _oracle(name) == stored, name
    problem = taillard_job_shop("ta01")
    assert job_shop_route_digest(problem) == _oracle("ta01")
    check_taillard_job_shop(problem)


@pytest.mark.parametrize("name", ["ta01", "ta42", "ta80"])
def test_the_check_accepts_a_job_shop_built_from_the_generators_routes_by_other_code(name):
    # Only the routes are digested: the metadata of the library's builder are not.
    problem = _rebuilt(name, _routes(name))
    assert problem != taillard_job_shop(name)
    check_taillard_job_shop(problem)
    assert job_shop_route_digest(problem) == JOB_SHOP_ROUTE_DIGESTS[name]


def _changed_time(routes: list[list[tuple[int, float]]]) -> None:
    machine, time = routes[3][7]
    routes[3][7] = (machine, time + 1.0)


def _swapped_machines(routes: list[list[tuple[int, float]]]) -> None:
    (first, one), (second, other) = routes[0][0], routes[0][1]
    routes[0][0], routes[0][1] = (second, one), (first, other)


def _swapped_operations(routes: list[list[tuple[int, float]]]) -> None:
    routes[5][2], routes[5][3] = routes[5][3], routes[5][2]


def _swapped_jobs(routes: list[list[tuple[int, float]]]) -> None:
    routes[0], routes[1] = routes[1], routes[0]


def _dropped_job(routes: list[list[tuple[int, float]]]) -> None:
    routes.pop()


@pytest.mark.parametrize(
    "change",
    [_changed_time, _swapped_machines, _swapped_operations, _swapped_jobs, _dropped_job],
    ids=["time", "machines", "operations", "jobs", "missing-job"],
)
def test_the_check_refuses_a_job_shop_whose_routes_differ(change: Any):
    routes = _routes("ta01")
    change(routes)
    problem = _rebuilt("ta01", routes)
    found = job_shop_route_digest(problem)
    assert found != JOB_SHOP_ROUTE_DIGESTS["ta01"]
    with pytest.raises(ModelError, match="'ta01' does not have the routes") as refused:
        check_taillard_job_shop(problem)
    assert refused.value.issues == (
        f"stored route digest: {JOB_SHOP_ROUTE_DIGESTS['ta01']}",
        f"digest of the routes found: {found}",
    )


def test_the_check_refuses_the_routes_of_one_job_shop_under_the_name_of_another():
    problem = _rebuilt("ta02", _routes("ta01"))
    with pytest.raises(ModelError, match="'ta02' does not have the routes"):
        check_taillard_job_shop(problem)


@pytest.mark.parametrize(
    "make",
    [
        lambda: replace(taillard_job_shop("ta01"), id="ta81"),
        lambda: replace(taillard_job_shop("ta01"), id="my-job-shop"),
        lambda: taillard_job_shop_seeded(15, 15, 1),
        lambda: taillard_flow_shop("ta001"),
    ],
    ids=["ta81", "other-name", "seeded", "flow-shop"],
)
def test_the_check_refuses_a_name_that_has_no_stored_digest(make: Any):
    with pytest.raises(ModelError, match="is not one of the job shops ta01 to ta80"):
        check_taillard_job_shop(make())


def test_a_problem_that_is_not_a_job_shop_has_no_route_digest():
    several_modes = replace(random_problem(2, tasks=8, max_modes=2, pools=1), id="ta01")
    ungrouped = Problem(
        "ta01",
        tasks=(Task.on("a0", "m0", 2.0), Task.on("a1", "m1", 3.0)),
        resources=(Resource("m0"), Resource("m1")),
        dependencies=(Dependency("a0", "a1"),),
    )
    assert "job-shop" not in several_modes.shapes
    assert "job-shop" in ungrouped.shapes
    with pytest.raises(ModelError, match="is not a job shop"):
        job_shop_route_digest(several_modes)
    with pytest.raises(ModelError, match="is not a job shop"):
        check_taillard_job_shop(several_modes)
    with pytest.raises(ModelError, match="belong to no job group"):
        job_shop_route_digest(ungrouped)
    with pytest.raises(ModelError, match="belong to no job group"):
        check_taillard_job_shop(ungrouped)
