from __future__ import annotations

import random

import pytest

from dispatchatlas.benchmarks.formats import (
    load_instance,
    read_fjs,
    read_orlib_flow_shop,
    read_orlib_job_shop,
    read_psplib,
    split_orlib,
)
from dispatchatlas.core import ConfigError, ModelError, SerializationError, check_schedule
from dispatchatlas.core.model import Problem
from dispatchatlas.solvers.operators import random_order
from dispatchatlas.solvers.scheduling import Decoder
from tests.benchmarks.fixtures import (
    FJS,
    FLOW_SHOP,
    FT06,
    FT06_OPTIMUM,
    FT06_ROUTE_0,
    ORLIB_JOB_SHOPS,
    PSPLIB_MM,
    PSPLIB_SM,
)

# ------------------------------------------------------------------ OR-Library job shops


def _route(problem: Problem, job: str) -> list[tuple[int, float]]:
    tasks = [problem.tasks[i] for i in problem.groups[job]]
    return [(int(next(iter(t.modes[0].demands))[1:]), t.modes[0].duration) for t in tasks]


def test_ft06_reads_as_a_job_shop():
    problem = read_orlib_job_shop(FT06, problem_id="ft06")
    assert problem.id == "ft06"
    assert problem.size == 36
    assert len(problem.dependencies) == 30
    assert [r.id for r in problem.resources] == [f"m{k}" for k in range(6)]
    assert _route(problem, "j0") == FT06_ROUTE_0
    assert (
        problem.metadata["description"] == "Fisher and Thompson 6x6 instance, alternate name (mt06)"
    )
    assert problem.metadata["problem_class"] == "job-shop"


def test_ft06_schedules_respect_the_known_optimum():
    problem = read_orlib_job_shop(FT06)
    machine_load = {r.id: 0.0 for r in problem.resources}
    for task in problem.tasks:
        machine_load[next(iter(task.modes[0].demands))] += task.modes[0].duration
    job_length = [
        sum(problem.tasks[i].modes[0].duration for i in members)
        for members in problem.groups.values()
    ]
    assert max(*machine_load.values(), *job_length) <= FT06_OPTIMUM
    decoder = Decoder(problem)
    rng = random.Random(0)
    for _ in range(200):
        schedule = decoder.schedule(random_order(problem, rng))
        assert check_schedule(problem, schedule) == ()
        assert schedule.makespan >= FT06_OPTIMUM


def test_machines_numbered_from_one():
    shifted = FT06.splitlines()
    body = [
        " ".join(str(int(v) + 1) if k % 2 == 0 else v for k, v in enumerate(line.split()))
        for line in shifted[2:]
    ]
    text = "\n".join([*shifted[:2], *body])
    assert _route(read_orlib_job_shop(text, first_machine=1), "j0") == FT06_ROUTE_0


def test_split_orlib_cuts_a_multi_instance_file():
    blocks = split_orlib(ORLIB_JOB_SHOPS)
    assert list(blocks) == ["ft06", "tiny"]
    assert blocks["tiny"] == "Two jobs on two machines\n2 2\n0 3 1 2\n1 4 0 0\n"
    assert split_orlib(FT06) == {}


def test_readers_extract_named_instances_and_accept_zero_times():
    problem = read_orlib_job_shop(ORLIB_JOB_SHOPS, instance="tiny")
    assert problem.id == "tiny"
    assert problem.metadata["instance"] == "tiny"
    assert _route(problem, "j1") == [(1, 4.0), (0, 0.0)]
    ft06 = read_orlib_job_shop(ORLIB_JOB_SHOPS, instance="ft06")
    assert _route(ft06, "j0") == FT06_ROUTE_0


def test_duplicate_instance_names_are_refused():
    with pytest.raises(SerializationError, match="twice"):
        split_orlib(ORLIB_JOB_SHOPS + " instance tiny\n 1 1\n 0 1\n")


@pytest.mark.parametrize(
    ("text", "options", "message"),
    [
        ("no numbers here\n", {}, "no size line"),
        ("2 2\n0 1 1 1\n0 1\n", {}, "expected 8 numbers"),
        ("1 2\n0 1 1 1\n0 1 1 1\n", {}, "read one instance at a time"),
        ("1 2\n0 1 2 1\n", {}, "outside 0..1"),
        ("1 2\n0 1 1 -3\n", {}, "negative"),
        ("1 2\n0 1 1 x\n", {}, "integer"),
        ("0 2\n", {}, "positive"),
        (ORLIB_JOB_SHOPS, {"instance": "ft10"}, "no instance 'ft10'"),
    ],
)
def test_malformed_job_shops_are_refused(text, options, message):
    with pytest.raises(SerializationError, match=message):
        read_orlib_job_shop(text, **options)


def test_first_machine_must_be_zero_or_one():
    with pytest.raises(ConfigError, match="first_machine"):
        read_orlib_job_shop(FT06, first_machine=2)


# ------------------------------------------------------------------ OR-Library flow shops


def test_flow_shop_reads_times_by_job_and_machine():
    problem = read_orlib_flow_shop(FLOW_SHOP, problem_id="toy")
    assert problem.permutation
    assert [t.id for t in problem.tasks] == ["j0-m0", "j0-m1", "j1-m0", "j1-m1", "j2-m0", "j2-m1"]
    assert [t.modes[0].duration for t in problem.tasks] == [3.0, 2.0, 1.0, 4.0, 2.0, 0.0]
    assert problem.metadata["format"] == "orlib-flow-shop"
    assert not read_orlib_flow_shop(FLOW_SHOP, permutation=False).permutation


def test_flow_shops_must_visit_machines_in_order():
    with pytest.raises(SerializationError, match="increasing order"):
        read_orlib_flow_shop("2 2\n0 1 1 1\n1 1 0 1\n")


# ------------------------------------------------------------------ FJS


def test_fjs_gives_one_mode_per_machine():
    problem = read_fjs(FJS, problem_id="toy")
    assert problem.id == "toy"
    assert [r.id for r in problem.resources] == ["m0", "m1", "m2"]
    assert [t.id for t in problem.tasks] == ["j0-o0", "j0-o1", "j1-o0", "j1-o1", "j1-o2"]
    options = {t.id: [(next(iter(m.demands)), m.duration) for m in t.modes] for t in problem.tasks}
    assert options == {
        "j0-o0": [("m0", 5.0), ("m2", 4.0)],
        "j0-o1": [("m1", 3.0)],
        "j1-o0": [("m0", 2.0)],
        "j1-o1": [("m1", 4.0), ("m2", 6.0)],
        "j1-o2": [("m2", 1.0)],
    }
    assert len(problem.dependencies) == 3
    assert problem.metadata["problem_class"] == "flexible-job-shop"


def test_fjs_and_orlib_read_ft06_alike():
    rows = [line.split() for line in FT06.splitlines()[2:]]
    fjs = "6 6 1\n" + "\n".join(
        "6 " + " ".join(f"1 {int(row[k]) + 1} {row[k + 1]}" for k in range(0, 12, 2))
        for row in rows
    )
    flexible = read_fjs(fjs, problem_id="ft06")
    classic = read_orlib_job_shop(FT06, problem_id="ft06")
    assert flexible.tasks == classic.tasks
    assert flexible.dependencies == classic.dependencies
    assert flexible.resources == classic.resources


def test_fjs_machines_numbered_from_zero_and_blank_lines():
    text = "\n" + FJS.replace(" 1 5 3 4 1 2 3", " 0 5 2 4 1 1 3").replace(
        "1 1 2 2 2 4 3 6 1 3 1", "1 0 2 2 1 4 2 6 1 2 1"
    )
    shifted = read_fjs(text, first_machine=0)
    assert shifted.tasks == read_fjs(FJS).tasks


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("2\n", "header"),
        ("1 2 3 4\n1 1 1 1\n", "header"),
        ("1 2\n2 1 1 3\n", "ends before"),
        ("1 2\n1 1 1 3 7\n", "follow the last job"),
        ("1 2\n1 1 3 3\n", "outside 1..2"),
        ("1 2\n0\n", "at least 1"),
        ("1 2\n1 1 1 -2\n", "at least 0"),
        ("x 2\n1 1 1 1\n", "integer"),
    ],
)
def test_malformed_fjs_is_refused(text, message):
    with pytest.raises(SerializationError, match=message):
        read_fjs(text)


# ------------------------------------------------------------------ PSPLIB


def test_psplib_single_mode():
    problem = read_psplib(PSPLIB_SM, problem_id="toy")
    assert [t.id for t in problem.tasks] == ["a2", "a3", "a4", "a5"]
    assert [(d.before, d.after) for d in problem.dependencies] == [("a2", "a4"), ("a3", "a5")]
    assert [(r.id, r.capacity) for r in problem.resources] == [("r1", 3.0), ("r2", 3.0)]
    assert [(t.modes[0].duration, dict(t.modes[0].demands)) for t in problem.tasks] == [
        (3.0, {"r1": 2.0}),
        (4.0, {"r1": 1.0, "r2": 1.0}),
        (2.0, {"r2": 3.0}),
        (5.0, {"r1": 2.0, "r2": 2.0}),
    ]
    assert dict(problem.metadata) == {"format": "psplib", "problem_class": "rcpsp", "horizon": 20}


def test_psplib_multi_mode_with_a_slack_budget():
    problem = read_psplib(PSPLIB_MM.format(budget=9), problem_id="mm")
    assert [len(t.modes) for t in problem.tasks] == [2, 2, 2]
    assert [(m.duration, dict(m.demands)) for m in problem.task("a3").modes] == [
        (3.0, {"r1": 3.0}),
        (6.0, {"r1": 1.0}),
    ]
    assert problem.metadata["dropped_modes"] == 1
    assert problem.metadata["problem_class"] == "multi-mode-rcpsp"
    assert "ignored_nonrenewable" not in problem.metadata


def test_psplib_refuses_a_binding_nonrenewable_budget():
    text = PSPLIB_MM.format(budget=8)
    with pytest.raises(ModelError, match="ignore_nonrenewable"):
        read_psplib(text)
    relaxed = read_psplib(text, ignore_nonrenewable=True)
    assert relaxed.metadata["ignored_nonrenewable"] == 1
    assert [r.id for r in relaxed.resources] == ["r1"]


def test_psplib_refuses_jobs_without_a_fitting_mode():
    text = PSPLIB_SM.replace("  3      1     4       1    1", "  3      1     4       5    1")
    with pytest.raises(ModelError, match="job 3 has no mode"):
        read_psplib(text)


def test_psplib_refuses_doubly_constrained_resources():
    text = PSPLIB_SM.replace("jobnr. mode duration  R 1  R 2", "jobnr. mode duration  R 1  D 1")
    with pytest.raises(ModelError, match="doubly constrained"):
        read_psplib(text)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("jobs (incl. supersource/sink ):  6", "", "no line"),
        (
            "   2        1          1           4",
            "   2        1          2           4",
            "precedence line",
        ),
        ("  5      1     5       2    2", "  5      1     5       2    2    7", "request line"),
        ("  6      1     0       0    0\n", "", "jobs 1 to 6"),
        ("    3    3\n", "    3\n", "availabilities"),
        ("RESOURCEAVAILABILITIES:", "AVAILABILITIES:", "no section"),
        ("projects                      :  1", "projects                      :  2", "one project"),
        (
            "   5        1          1           6",
            "   5        1          1           9",
            "unknown successors",
        ),
    ],
)
def test_malformed_psplib_is_refused(old, new, message):
    assert old in PSPLIB_SM
    with pytest.raises(SerializationError, match=message):
        read_psplib(PSPLIB_SM.replace(old, new))


# ------------------------------------------------------------------ load_instance


def test_load_instance_infers_the_format(tmp_path):
    (tmp_path / "Mk99.fjs").write_text(FJS)
    (tmp_path / "j301_1.sm").write_text(PSPLIB_SM)
    (tmp_path / "j102_3.mm").write_text(PSPLIB_MM.format(budget=9))
    assert load_instance(tmp_path / "Mk99.fjs").id == "Mk99"
    assert load_instance(str(tmp_path / "j301_1.sm")).id == "j301_1"
    assert load_instance(tmp_path / "j102_3.mm").metadata["problem_class"] == "multi-mode-rcpsp"


def test_load_instance_with_an_explicit_format(tmp_path):
    path = tmp_path / "jobshop1.txt"
    path.write_text(ORLIB_JOB_SHOPS)
    problem = load_instance(path, format="orlib-job-shop", instance="ft06")
    assert problem.id == "ft06"
    renamed = load_instance(path, "orlib-job-shop", instance="tiny", problem_id="small")
    assert renamed.id == "small"
    flow = tmp_path / "flow.txt"
    flow.write_text(FLOW_SHOP)
    assert load_instance(flow, format="orlib-flow-shop").id == "flow"


@pytest.mark.parametrize(
    ("name", "options", "message"),
    [
        ("data.txt", {}, "cannot infer"),
        ("data.txt", {"format": "excel"}, "unknown instance format"),
        ("data.fjs", {"instance": "x"}, "does not accept"),
    ],
)
def test_load_instance_refuses_bad_requests(tmp_path, name, options, message):
    path = tmp_path / name
    path.write_text(FJS)
    with pytest.raises(ConfigError, match=message):
        load_instance(path, **options)
