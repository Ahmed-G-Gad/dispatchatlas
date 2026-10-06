"""Tests of the streaming reader of the Alibaba 2018 batch-task table."""

from __future__ import annotations

import itertools
import re
import time
from collections import deque
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import pytest

from dispatchatlas import benchmarks
from dispatchatlas.benchmarks.alibaba import (
    COLUMNS,
    SKIP_REASONS,
    STATUSES,
    WORK_RULES,
    iter_alibaba_batch,
    parse_alibaba_task_name,
)
from dispatchatlas.benchmarks.continuum import three_tier_platform
from dispatchatlas.benchmarks.continuum.platform import Platform
from dispatchatlas.benchmarks.wfcommons import read_wfcommons
from dispatchatlas.core import (
    ConfigError,
    Link,
    Network,
    Problem,
    Resource,
    SerializationError,
    check_schedule,
)
from dispatchatlas.solvers import solve
from dispatchatlas.solvers.scheduling import Decoder

PLATFORM = three_tier_platform(devices=1, edge=1, fog=1, cloud=1)

# Hand-written rows. Job j_1 has the graph of the example in the documentation
# of the trace, with its rows out of task order; j_2 has independent tasks, one
# of them with a span of zero; j_3 has a single task.
LINES = [
    "M5_3_4,2,j_1,1,Terminated,160,200,100,0.5",
    "M1,4,j_1,1,Terminated,100,120,50,0.25",
    "R4_2,1,j_1,1,Terminated,140,150,100,0.5",
    "M2_1,3,j_1,1,Terminated,120,140,200,1.25",
    "M3_1,1,j_1,1,Terminated,125,160,100,0.75",
    "task_QUJD,1,j_2,10,Terminated,300,360,75,0.1",
    "MergeTask,5,j_2,10,Terminated,290,290,50,0.2",
    "M1,1,j_3,1,Terminated,50,57,100,0.3",
]
SPANS = {"M5_3_4": 40.0, "M1": 20.0, "R4_2": 10.0, "M2_1": 20.0, "M3_1": 35.0}
CORE_SECONDS = {"M5_3_4": 80.0, "M1": 40.0, "R4_2": 10.0, "M2_1": 120.0, "M3_1": 35.0}
EDGES = [
    ("M1", "M2_1"),
    ("M1", "M3_1"),
    ("R4_2", "M5_3_4"),
    ("M2_1", "R4_2"),
    ("M3_1", "M5_3_4"),
]

# Task 30 of a job, depending on tasks 1 to 29, as the 50 characters of the
# trace leave it: cut inside its last number, and cut inside a suffix.
LONG_CODE = "J30_" + "_".join(str(number) for number in range(1, 30))
CUT_INSIDE_A_NUMBER = LONG_CODE[:50]
CUT_INSIDE_A_SUFFIX = LONG_CODE[:44] + "_Stg12"

# A job whose tasks carry the start time of the job, as most jobs of the trace
# do, so the span of a task includes the time it waited for the tasks it
# depends on. The last task starts after the task it depends on has ended.
WAITING = [
    "M1,1,j_4,1,Terminated,100,120,100,0.5",
    "M2_1,1,j_4,1,Terminated,100,150,100,0.5",
    "M3_1,1,j_4,1,Terminated,100,130,100,0.5",
    "R4_2_3,1,j_4,1,Terminated,100,190,100,0.5",
    "J5_4,1,j_4,1,Terminated,200,230,100,0.5",
]
WAITING_SPANS = {"M1": 20.0, "M2_1": 50.0, "M3_1": 30.0, "R4_2_3": 90.0, "J5_4": 30.0}
WAITING_AFTER = {"M1": 20.0, "M2_1": 30.0, "M3_1": 10.0, "R4_2_3": 40.0, "J5_4": 30.0}

# Characters that must not reach a name: a byte-order mark and a zero-width space.
BYTE_ORDER_MARK = chr(0xFEFF)
ZERO_WIDTH_SPACE = chr(0x200B)

# Fields of unusual length, to show that refusing them takes linear time.
LONG = 300_000


def row(name: str, job: str = "j_9", **fields: object) -> str:
    """A row of a finished task from second 10 to second 20, with ``fields`` replaced."""
    values: dict[str, object] = {
        "task_name": name,
        "instance_num": 1,
        "job_name": job,
        "task_type": 1,
        "status": "Terminated",
        "start_time": 10,
        "end_time": 20,
        "plan_cpu": 100,
        "plan_mem": 0.5,
    }
    assert set(fields) <= set(values)
    values.update(fields)
    return ",".join(str(values[column]) for column in COLUMNS)


def read(lines: Any = LINES, **options: Any) -> list[Problem]:
    return list(iter_alibaba_batch(lines, PLATFORM, **options))


def outcome(lines: Any, **options: Any) -> tuple[list[str], list[tuple[str, str]]]:
    """The ids of the problems read from ``lines`` and the reported skips."""
    skipped: list[tuple[str, str]] = []
    problems = read(lines, on_skip=lambda job, reason: skipped.append((job, reason)), **options)
    return [problem.id for problem in problems], skipped


# ------------------------------------------------------------------ task names


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("M1", (1, ())),
        ("R2_1", (2, (1,))),
        ("J4_2_3", (4, (2, 3))),
        ("m12_7", (12, (7,))),
        ("AB3_2_2_1", (3, (2, 1))),
        ("M01_002", (1, (2,))),
        ("M" * 48 + "1", (1, ())),
        ("task_QUJDREVGRw==", None),
        ("task_1", None),
        ("task_" + "QUJD" * 20, None),
        ("MergeTask", None),
        ("M", None),
        ("7", None),
        ("_M1", None),
        ("", None),
    ],
)
def test_task_names_decode_to_a_number_and_its_dependencies(name, expected):
    assert parse_alibaba_task_name(name) == expected


@pytest.mark.parametrize(
    "name",
    ["M3_1_2_", "M1_", "M" * 49 + "1", CUT_INSIDE_A_NUMBER, CUT_INSIDE_A_SUFFIX, LONG_CODE],
)
def test_cut_off_task_names_are_refused(name):
    with pytest.raises(SerializationError, match="cut off"):
        parse_alibaba_task_name(name)


def test_a_cut_can_leave_a_name_that_looks_complete():
    assert len(CUT_INSIDE_A_NUMBER) == len(CUT_INSIDE_A_SUFFIX) == 50
    assert CUT_INSIDE_A_NUMBER.endswith("_18_1")
    assert CUT_INSIDE_A_SUFFIX.endswith("_16_1_Stg12")


@pytest.mark.parametrize("name", ["M2_1_Stg3", "M7_Stg3", "M1__2", "M1_x", "M1x", "M3x_1", "R2-1"])
def test_names_that_start_like_a_code_and_are_not_one_are_refused(name):
    with pytest.raises(SerializationError, match="starts like a dependency code and is not one"):
        parse_alibaba_task_name(name)


@pytest.mark.parametrize("name", [None, 5, b"M1", ["M1"]])
def test_a_task_name_must_be_a_string(name):
    with pytest.raises(ConfigError, match="a task name must be a string"):
        parse_alibaba_task_name(name)


# ------------------------------------------------------------------------ jobs


def test_reads_one_problem_per_job():
    first, second, third = read()
    assert [problem.id for problem in (first, second, third)] == ["j_1", "j_2", "j_3"]
    assert [task.id for task in first.tasks] == list(SPANS)
    for task in first.tasks:
        assert [mode.location for mode in task.modes] == ["edge-0", "fog-0", "cloud-0"]
        for mode in task.modes:
            assert mode.location is not None
            assert mode.duration == PLATFORM.duration(SPANS[task.id], mode.location)
            assert dict(mode.demands) == {mode.location: 1.0}
        assert task.group == "j_1"
        assert task.release == 0.0
    assert [(d.before, d.after, d.data, d.lag) for d in first.dependencies] == [
        (before, after, 0.0, 0.0) for before, after in EDGES
    ]
    assert first.resources == PLATFORM.resources
    assert first.network == PLATFORM.network
    assert first.objectives == ("makespan",)
    assert "network" not in first.features
    assert [task.id for task in second.tasks] == ["task_QUJD", "MergeTask"]
    assert second.dependencies == ()
    assert second.task("MergeTask").min_duration == 0.0
    assert [task.id for task in third.tasks] == ["M1"]


def test_problem_ids_are_the_job_names_and_distinct():
    lines = [row("M1", job=f"j_{number}") for number in range(40)]
    problems = read(lines)
    ids = [problem.id for problem in problems]
    assert ids == [f"j_{number}" for number in range(40)]
    assert len(set(ids)) == len(ids)
    assert all(problem.metadata["job"] == problem.id for problem in problems)


def test_problems_and_tasks_record_the_fields_of_the_trace():
    first, second, _ = read()
    assert dict(first.metadata) == {
        "source": "alibaba-cluster-trace-v2018",
        "job": "j_1",
        "arrival": 100,
        "work": "duration",
    }
    assert dict(first.task("M5_3_4").metadata) == {
        "instance_num": 2,
        "plan_cpu": 100.0,
        "plan_mem": 0.5,
        "task_type": "1",
        "status": "Terminated",
        "start_time": 160,
        "end_time": 200,
    }
    assert second.metadata["arrival"] == 290
    assert second.task("task_QUJD").metadata["task_type"] == "10"


def test_empty_numbers_are_left_out_of_the_metadata():
    line = row("task_QQ==", instance_num="", plan_cpu="", plan_mem="")
    (problem,) = read([line])
    assert dict(problem.tasks[0].metadata) == {
        "task_type": "1",
        "status": "Terminated",
        "start_time": 10,
        "end_time": 20,
    }
    assert problem.tasks[0].modes[0].duration == PLATFORM.duration(10.0, "edge-0")


def test_core_seconds_multiply_the_span_by_the_requested_cores():
    first, second, third = read(work="core_seconds")
    for task in first.tasks:
        assert task.modes[2].duration == PLATFORM.duration(CORE_SECONDS[task.id], "cloud-0")
    assert first.metadata["work"] == "core_seconds"
    assert second.task("task_QUJD").modes[0].duration == PLATFORM.duration(45.0, "edge-0")
    assert second.task("MergeTask").min_duration == 0.0
    assert third.tasks[0].modes[0].duration == PLATFORM.duration(7.0, "edge-0")
    assert first.dependencies == read()[0].dependencies


def test_work_after_dependencies_leaves_out_the_wait_that_the_times_show():
    (spans,) = read(WAITING)
    (after,) = read(WAITING, work="after_dependencies")
    for task in spans.tasks:
        assert task.modes[2].duration == PLATFORM.duration(WAITING_SPANS[task.id], "cloud-0")
    for task in after.tasks:
        assert task.modes[2].duration == PLATFORM.duration(WAITING_AFTER[task.id], "cloud-0")
    assert after.metadata["work"] == "after_dependencies"
    assert after.metadata["arrival"] == 100
    assert after.dependencies == spans.dependencies
    assert [task.metadata for task in after.tasks] == [task.metadata for task in spans.tasks]
    # The job took 130 seconds. On one node of speed 1 with a core for every
    # task, the makespan is the longest path, M1 -> M2_1 -> R4_2_3 -> J5_4:
    # 190 seconds of spans, and 120 seconds of work after dependencies.
    alone = Platform((Resource("n-0", capacity=8.0, tier="n"),), Network())
    (spans,) = iter_alibaba_batch(WAITING, alone)
    (after,) = iter_alibaba_batch(WAITING, alone, work="after_dependencies")
    assert solve(spans, "heft").value == 190.0
    result = solve(after, "heft")
    assert result.value == 120.0
    assert result.schedule is not None
    assert check_schedule(after, result.schedule) == ()


def test_work_after_dependencies_is_the_span_where_no_task_waited():
    for plain, after in zip(read(), read(work="after_dependencies"), strict=True):
        assert after.tasks == plain.tasks
        assert after.dependencies == plain.dependencies


def test_work_after_dependencies_skips_a_job_whose_times_contradict_a_dependency():
    lines = [row("M1", end_time=30), row("R2_1", end_time=25)]
    assert outcome(lines, work="after_dependencies") == ([], [("j_9", "dependency-time")])
    assert outcome(lines) == (["j_9"], [])
    assert outcome(lines, work="core_seconds") == (["j_9"], [])
    # A task that ends with the task it depends on has a work of zero, which is read.
    ending_together = [row("M1", end_time=30), row("R2_1", end_time=30)]
    (problem,) = read(ending_together, work="after_dependencies")
    assert problem.task("R2_1").min_duration == 0.0
    assert problem.task("M1").min_duration == PLATFORM.duration(20.0, "cloud-0")


def test_tiers_objectives_and_statuses_are_configurable():
    (problem, *_) = read(tiers=["device", "cloud"], objectives=("makespan", "energy"))
    assert [mode.location for mode in problem.tasks[0].modes] == ["device-0", "cloud-0"]
    assert problem.objectives == ("makespan", "energy")
    (problem, *_) = read(tiers=deque(["fog", "edge"]), statuses=deque(["Terminated"]))
    assert [mode.location for mode in problem.tasks[0].modes] == ["fog-0", "edge-0"]
    (problem, *_) = read(tiers=("cloud",), objectives="cost")
    assert [mode.location for mode in problem.tasks[0].modes] == ["cloud-0"]
    assert problem.objectives == ("cost",)
    lines = [row("M1"), row("R2_1", status="Failed")]
    assert outcome(lines) == ([], [("j_9", "status")])
    (problem,) = read(lines, statuses=("Terminated", "Failed"))
    assert problem.task("R2_1").metadata["status"] == "Failed"
    assert outcome(lines, statuses=["Failed"]) == ([], [("j_9", "status")])
    assert outcome(lines[1:], statuses=["Failed"]) == ([], [("j_9", "missing-dependency")])


def test_dependencies_carry_edge_data_across_the_network():
    problem = read(edge_data=2.0)[0]
    assert {dependency.data for dependency in problem.dependencies} == {2.0}
    assert "network" in problem.features
    decoder = Decoder(problem)
    order = [1, 3, 4, 2, 0]
    # M1 on the edge and M2_1 in the cloud: 2 MB cross the metro and wide-area links.
    decoded = decoder.decode(order, [0, 0, 0, 2, 0])
    link = PLATFORM.network.link("edge", "cloud")
    assert isinstance(link, Link)
    assert decoded.starts[3] == pytest.approx(decoded.ends[1] + link.latency + 2.0 / link.bandwidth)
    local = decoder.decode(order, [0, 0, 0, 0, 0])
    assert local.starts[3] == pytest.approx(local.ends[1])
    assert check_schedule(problem, decoded.schedule(problem)) == ()


def test_solvers_schedule_the_jobs():
    first, second, third = read()
    for problem in (first, second, third):
        result = solve(problem, "heft")
        assert result.schedule is not None
        assert check_schedule(problem, result.schedule) == ()
    # The cloud is the fastest tier and has a core for every task, so the
    # makespan is the longest path, M1 -> M3_1 -> M5_3_4, at the cloud's speed.
    longest = sum(PLATFORM.duration(SPANS[name], "cloud-0") for name in ("M1", "M3_1", "M5_3_4"))
    assert solve(first, "heft").value == pytest.approx(longest)
    assert solve(third, "heft").value == PLATFORM.duration(7.0, "cloud-0")


# --------------------------------------------------------------- skipped jobs


@pytest.mark.parametrize(
    ("lines", "options", "reason"),
    [
        ([row("M1"), row("R2_1", status="Failed")], {}, "status"),
        ([row("M1", status="Running", end_time=0)], {}, "status"),
        ([row("M1", status="Waiting", start_time=0, end_time=0, instance_num="")], {}, "status"),
        ([row("M1", status="Done")], {}, "status"),
        ([row("M1", status="")], {}, "status"),
        ([row("M1", status=" Terminated")], {}, "status"),
        ([row("M1", end_time=0)], {}, "time"),
        ([row("M1", start_time=0)], {}, "time"),
        ([row("M1", start_time=0, end_time=0)], {}, "time"),
        ([row("M1", start_time=30)], {}, "time"),
        ([row("M1", start_time="")], {}, "time"),
        ([row("M1", end_time="")], {}, "time"),
        ([row("M1", start_time=-5, end_time=-1)], {}, "time"),
        ([row("M1", instance_num="")], {"work": "core_seconds"}, "demand"),
        ([row("M1", plan_cpu="")], {"work": "core_seconds"}, "demand"),
        ([row("M1", instance_num=0)], {"work": "core_seconds"}, "demand"),
        ([row("M1", instance_num=-2)], {"work": "core_seconds"}, "demand"),
        ([row("M1", plan_cpu=0)], {"work": "core_seconds"}, "demand"),
        ([row("M1", plan_cpu=-100)], {"work": "core_seconds"}, "demand"),
        ([row("M1"), row("M2"), row("M3_1_2_")], {}, "truncated-name"),
        ([row("M1"), row(CUT_INSIDE_A_NUMBER)], {}, "truncated-name"),
        ([row("M1"), row(CUT_INSIDE_A_SUFFIX)], {}, "truncated-name"),
        ([row("M1"), row("M2_1_Stg3")], {}, "unknown-name"),
        ([row("M1"), row("M2x")], {}, "unknown-name"),
        ([row("M1"), row("R1")], {}, "duplicate-task"),
        ([row("M1"), row("M01")], {}, "duplicate-task"),
        ([row("M1"), row("M2_1"), row("M1")], {}, "duplicate-task"),
        ([row("MergeTask"), row("MergeTask")], {}, "duplicate-task"),
        ([row("R2_1")], {}, "missing-dependency"),
        ([row("M1"), row("J3_1_2")], {}, "missing-dependency"),
        ([row("M1_1")], {}, "cycle"),
        ([row("M1_2"), row("M2_1")], {}, "cycle"),
        ([row("M1"), row("M2_1_4"), row("M3_2"), row("M4_3")], {}, "cycle"),
        (
            [row("M1", end_time=30), row("R2_1")],
            {"work": "after_dependencies"},
            "dependency-time",
        ),
        (
            [row("M1"), row("M2", end_time=40), row("J3_1_2", start_time=5, end_time=39)],
            {"work": "after_dependencies"},
            "dependency-time",
        ),
        ([row("M1")], {"min_tasks": 2}, "too-few-tasks"),
        ([row("M1"), row("M2")], {"max_tasks": 1}, "too-many-tasks"),
    ],
)
def test_a_job_that_cannot_be_read_completely_is_skipped_whole(lines, options, reason):
    assert reason in SKIP_REASONS
    assert outcome(lines, **options) == ([], [("j_9", reason)])


def test_a_skipped_job_leaves_the_jobs_around_it_alone():
    lines = [row("M1", job="j_8"), row("M1", end_time=0), row("R2_1"), row("M7", job="j_10")]
    assert outcome(lines) == (["j_8", "j_10"], [("j_9", "time")])
    assert outcome(lines, max_tasks=1) == (["j_8", "j_10"], [("j_9", "time")])
    assert outcome(lines, min_tasks=2) == (
        [],
        [("j_8", "too-few-tasks"), ("j_9", "time"), ("j_10", "too-few-tasks")],
    )


def test_the_reported_reason_is_the_first_that_applies():
    defects = [
        ("status", [row("M1", status="Failed")]),
        ("time", [row("M2", end_time=0)]),
        ("demand", [row("M3", instance_num="")]),
        ("truncated-name", [row("M4_1_")]),
        ("unknown-name", [row("M5_1_Stg2")]),
        ("duplicate-task", [row("M6"), row("R6")]),
        ("missing-dependency", [row("M7_99")]),
        ("cycle", [row("M8_8")]),
    ]
    assert [reason for reason, _ in defects] == list(SKIP_REASONS[:8])
    for first in range(len(defects)):
        lines = [line for _, lines in reversed(defects[first:]) for line in lines]
        expected = [("j_9", defects[first][0])]
        assert outcome(lines, work="core_seconds", min_tasks=50) == ([], expected)
    assert outcome([row("M1")], min_tasks=2, max_tasks=5) == ([], [("j_9", "too-few-tasks")])
    six = [row(f"M{number}") for number in range(1, 7)]
    assert outcome(six, min_tasks=2, max_tasks=5) == ([], [("j_9", "too-many-tasks")])
    assert outcome(six[:5], min_tasks=2, max_tasks=5) == (["j_9"], [])
    assert outcome(six[:2], min_tasks=2, max_tasks=2) == (["j_9"], [])
    # Times that contradict a dependency come after the graph and before the size.
    late = [row("M1", end_time=30), row("R2_1")]
    options = {"work": "after_dependencies"}
    assert outcome([*late, row("M3_3")], **options) == ([], [("j_9", "cycle")])
    assert outcome([*late, row("M3_7")], **options) == ([], [("j_9", "missing-dependency")])
    assert outcome(late, min_tasks=3, **options) == ([], [("j_9", "dependency-time")])
    assert outcome(late, max_tasks=1, **options) == ([], [("j_9", "dependency-time")])


def test_work_too_large_to_represent_stops_the_reader_only_in_a_job_that_is_read():
    big = row("M1", end_time="9" * 18, instance_num="9" * 18, plan_cpu="1e300")
    others = [
        (row("M2", status="Failed"), "status"),
        (row("M2", end_time=0), "time"),
        (row("M2", plan_cpu=""), "demand"),
        (row("M2_1_"), "truncated-name"),
        (row("M2_1_Stg3"), "unknown-name"),
        (row("R1"), "duplicate-task"),
        (row("M2_7"), "missing-dependency"),
        (row("M2_2"), "cycle"),
    ]
    for other, reason in others:
        for lines in ([big, other], [other, big]):
            assert outcome(lines, work="core_seconds") == ([], [("j_9", reason)])
    assert outcome([big], work="core_seconds", min_tasks=2) == ([], [("j_9", "too-few-tasks")])
    two = [big, row("M2")]
    assert outcome(two, work="core_seconds", max_tasks=1) == ([], [("j_9", "too-many-tasks")])
    for lines in ([big], two, [row("M2"), big]):
        with pytest.raises(SerializationError, match="job 'j_9' has a task whose work is too"):
            read(lines, work="core_seconds")
    # The jobs before the one that is refused are read first.
    stream = iter_alibaba_batch([row("M1", job="j_8"), big], PLATFORM, work="core_seconds")
    assert next(stream).id == "j_8"
    with pytest.raises(SerializationError, match="job 'j_9' has a task whose work is too large"):
        next(stream)
    # Core-seconds that a floating-point number holds are read, although their
    # product with 100 overflows: one second of 100 instances at 1.7e306 cores.
    alone = Platform((Resource("a-0", tier="a"),), Network())
    line = row("M1", start_time=1, end_time=2, instance_num=100, plan_cpu="1.7e308")
    (problem,) = iter_alibaba_batch([line], alone, work="core_seconds")
    assert problem.tasks[0].modes[0].duration == 1.7e308


def test_skips_need_no_callback_and_equal_times_are_read():
    lines = [row("M1", job="j_8", end_time=0), row("M1", start_time=20), row("R2_1")]
    (problem,) = read(lines)
    assert problem.id == "j_9"
    assert problem.task("M1").min_duration == 0.0
    assert problem.metadata["arrival"] == 10


def test_a_repeated_dependency_counts_once():
    (problem,) = read([row("M1"), row("R2_1_1")])
    assert [(d.before, d.after) for d in problem.dependencies] == [("M1", "R2_1_1")]


# ---------------------------------------------------------------------- source


def test_reads_paths_files_and_iterables_alike(tmp_path):
    expected = read()
    path = tmp_path / "batch_task.csv"
    path.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    assert read(path) == expected
    assert read(str(path)) == expected
    with path.open(encoding="utf-8") as handle:
        assert read(handle) == expected
    assert read(iter(LINES)) == expected
    assert read(tuple(LINES)) == expected
    assert read([]) == []
    assert read(["", "\n", "  \r\n"]) == []


def test_a_byte_order_mark_and_windows_line_endings_are_accepted(tmp_path):
    expected = read()
    path = tmp_path / "batch_task.csv"
    path.write_bytes(("\r\n\r\n".join(LINES) + "\r\n \r\n").encode("utf-8-sig"))
    assert read(path) == expected
    with path.open(encoding="utf-8", newline="") as handle:
        assert read(handle) == expected
    assert read(["\ufeff" + LINES[0] + "\r\n", *LINES[1:]]) == expected
    assert read(["\ufeff\n", *LINES]) == expected


def test_a_string_source_is_a_path_and_never_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for text in (LINES[0], "\n".join(LINES)):
        jobs = iter_alibaba_batch(text, PLATFORM)
        with pytest.raises(OSError, match="j_1"):
            next(jobs)
    with pytest.raises(ConfigError, match="a source path must be text without a null character"):
        iter_alibaba_batch("batch\0task.csv", PLATFORM)


def test_the_source_is_read_one_job_at_a_time():
    consumed = 0

    def counted() -> Iterator[str]:
        nonlocal consumed
        for line in LINES:
            consumed += 1
            yield line

    jobs = iter_alibaba_batch(counted(), PLATFORM)
    assert consumed == 0
    assert next(jobs).id == "j_1"
    assert consumed == 6
    assert next(jobs).id == "j_2"
    assert consumed == 8

    def endless() -> Iterator[str]:
        for number in itertools.count(1):
            skipped = number % 2 == 0
            yield row("M1", job=f"j_{number}", end_time=0 if skipped else 20)
            yield row("R2_1", job=f"j_{number}")

    skips: list[str] = []
    stream = iter_alibaba_batch(endless(), PLATFORM, on_skip=lambda job, _: skips.append(job))
    assert [problem.id for problem in itertools.islice(stream, 3)] == ["j_1", "j_3", "j_5"]
    assert skips == ["j_2", "j_4"]


def test_a_job_that_returns_among_the_last_jobs_is_refused():
    lines = [row("M1", job="j_7"), row("M1", job="j_3"), row("M2", job="j_7")]
    jobs = iter_alibaba_batch(lines, PLATFORM)
    assert [next(jobs).id, next(jobs).id] == ["j_7", "j_3"]
    with pytest.raises(SerializationError, match="row 3: job 'j_7' returns after other jobs"):
        next(jobs)
    # A skipped job is remembered like one that is read.
    skipped = [row("M1", job="j_7", end_time=0), row("M1", job="j_3"), row("M2", job="j_7")]
    with pytest.raises(SerializationError, match="row 3: job 'j_7' returns after other jobs"):
        read(skipped)


def test_a_job_that_returns_after_1024_other_jobs_is_read_as_a_job_of_its_own():
    between = [row("M1", job=f"j_{number}") for number in range(100, 100 + 1024)]
    lines = [row("M1", job="j_7"), *between, row("M2", job="j_7")]
    problems = read(lines)
    assert len(problems) == 1026
    assert [problem.id for problem in (problems[0], problems[-1])] == ["j_7", "j_7"]
    assert [task.id for task in problems[0].tasks] == ["M1"]
    assert [task.id for task in problems[-1].tasks] == ["M2"]
    with pytest.raises(SerializationError, match="row 1025: job 'j_7' returns after other jobs"):
        read(lines[:1] + lines[2:])


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("M1,1,j_9,1,Terminated,10,20,100", "row 2 has 8 fields, expected the 9"),
        (row("M1") + ",extra", "row 2 has 10 fields"),
        ("task_name,instance_num", "row 2 has 2 fields"),
        (",".join(COLUMNS), "row 2: instance_num must be an integer of at most 18 digits"),
        (row("M1", plan_mem="0,5"), "row 2 has 10 fields"),
        (row(""), "row 2 must name its task and its job"),
        (row("M1", job=""), "row 2 must name its task and its job"),
        (row(" M1"), "row 2 must name its task and its job"),
        (row("M1", job="j_9 "), "row 2 must name its task and its job"),
        (5, "row 2 is not a line of text"),
        (None, "row 2 is not a line of text"),
        (b"M1,1,j_9,1,Terminated,10,20,100,0.5", "row 2 is not a line of text"),
    ],
)
def test_malformed_rows_are_refused_with_their_number(line, message):
    with pytest.raises(SerializationError, match=message):
        read([row("M1", job="j_8"), line])
    # A malformed row is refused even inside a job that is skipped anyway.
    with pytest.raises(SerializationError, match=message):
        read([row("M1", status="Failed"), line])


def test_a_first_row_without_a_job_name_is_refused():
    with pytest.raises(SerializationError, match="row 1 must name its task and its job"):
        read([row("M1", job="")])


@pytest.mark.parametrize(
    ("line", "column"),
    [
        (row('"R2_1"'), "task_name"),
        (row("'R2_1'"), "task_name"),
        (row('R2"_1'), "task_name"),
        (row("M1", job='"j_9"'), "job_name"),
        (row("M1", job="j'9"), "job_name"),
        (row(BYTE_ORDER_MARK + "R2_1"), "task_name"),
        (row(ZERO_WIDTH_SPACE + "R2_1"), "task_name"),
        (row("R2_1" + ZERO_WIDTH_SPACE), "task_name"),
        (row("M\t1"), "task_name"),
        (row("M1\0"), "task_name"),
        (row("M1" + chr(0x7F)), "task_name"),
        (row("M" + chr(0xE9) + "1"), "task_name"),
        (row("M1", job="j_" + chr(0x4E5D)), "job_name"),
        (row("M1", job="j\t9"), "job_name"),
    ],
)
def test_names_with_quotes_or_characters_outside_printable_ascii_are_refused(line, column):
    message = f"row 2: {column} .* has a quote or a character outside printable ASCII"
    with pytest.raises(SerializationError, match=message):
        read([row("M1", job="j_8"), line])
    # Read as it stands, such a name would be the name of an independent task.
    with pytest.raises(SerializationError, match=message):
        read([row("M1"), line])
    # A malformed name is refused even inside a job that is skipped anyway.
    with pytest.raises(SerializationError, match=message):
        read([row("M1", status="Failed"), line])


def test_quoted_tables_are_refused_and_not_read_as_independent_tasks():
    quoted = ['"R2_1",1,"j_1","1","Terminated",20,30,100,0.5', row('"M1"', job='"j_1"')]
    with pytest.raises(SerializationError, match="row 1: task_name '\"R2_1\"' has a quote"):
        outcome(quoted)
    with pytest.raises(SerializationError, match="the table is UTF-8 text without quoted fields"):
        read([row("M1"), row('"R2_1"')])


def test_names_of_printable_ascii_are_read_as_they_stand():
    names = ["Merge Task", "task_QUJD+/==", "x#y;z", "M-1", "(x)", "~", "!"]
    (problem,) = read([row(name, job="job 7 (a/b)") for name in names])
    assert problem.id == "job 7 (a/b)"
    assert [task.id for task in problem.tasks] == names
    assert problem.dependencies == ()


def test_a_byte_order_mark_is_read_only_at_the_start_and_only_as_utf_8(tmp_path):
    expected = read()
    text = "\n".join(LINES) + "\n"
    path = tmp_path / "batch_task.csv"
    path.write_bytes(text.encode("utf-8-sig"))
    for encoding in ("utf-8", "utf-8-sig"):
        with path.open(encoding=encoding) as handle:
            assert read(handle) == expected
    # Decoded with another encoding, the mark becomes part of the first task name.
    for encoding in ("cp1252", "latin-1"):
        with (
            path.open(encoding=encoding) as handle,
            pytest.raises(SerializationError, match=r"row 1: task_name .* printable ASCII"),
        ):
            read(handle)
    # Two marked tables joined into one file: the second mark is inside the table.
    path.write_bytes(text.encode("utf-8-sig") + text.replace("j_", "k_").encode("utf-8-sig"))
    with pytest.raises(SerializationError, match=r"row 9: task_name .*M5_3_4' has a quote or"):
        read(path)
    with pytest.raises(SerializationError, match="row 1: task_name"):
        read([BYTE_ORDER_MARK * 2 + LINES[0]])


def test_an_exception_from_on_skip_ends_the_iteration():
    class RefusedError(Exception):
        pass

    def refuse(job: str, reason: str) -> None:
        raise RefusedError(f"{job}: {reason}")

    lines = [row("M1", job="j_7"), row("M1", job="j_8", status="Failed"), row("M1", job="j_9")]
    assert outcome(lines) == (["j_7", "j_9"], [("j_8", "status")])
    stream = iter_alibaba_batch(lines, PLATFORM, on_skip=refuse)
    assert next(stream).id == "j_7"
    with pytest.raises(RefusedError, match="j_8: status"):
        next(stream)
    # The generator is finished: the job after the skipped one is never read.
    assert list(stream) == []


def test_the_file_of_a_path_stays_open_until_the_generator_ends(tmp_path, monkeypatch):
    path = tmp_path / "batch_task.csv"
    path.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    broken = tmp_path / "broken.csv"
    broken.write_text(LINES[0] + "\nbroken\n", encoding="utf-8")
    opened: list[Any] = []
    original = Path.open

    def recording(self: Path, *args: Any, **options: Any) -> Any:
        handle = original(self, *args, **options)
        opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", recording)
    stream = iter_alibaba_batch(path, PLATFORM)
    assert isinstance(stream, Generator)
    assert opened == []
    assert next(stream).id == "j_1"
    assert len(opened) == 1
    assert not opened[0].closed
    stream.close()
    assert opened[0].closed
    assert list(stream) == []
    assert len(read(path)) == 3
    assert opened[1].closed
    with pytest.raises(SerializationError, match="row 2 has 1 fields"):
        read(broken)
    assert opened[2].closed
    assert len(opened) == 3


def test_a_source_that_is_not_a_path_is_left_to_the_caller(tmp_path):
    path = tmp_path / "batch_task.csv"
    path.write_text("\n".join(LINES) + "\nbroken\n", encoding="utf-8")
    with path.open(encoding="utf-8") as handle:
        stream = iter_alibaba_batch(handle, PLATFORM)
        assert next(stream).id == "j_1"
        stream.close()
        assert not handle.closed
        with pytest.raises(SerializationError, match="has 1 fields"):
            list(iter_alibaba_batch(handle, PLATFORM))
        assert not handle.closed

    def lines() -> Iterator[str]:
        yield from LINES

    source = lines()
    stream = iter_alibaba_batch(source, PLATFORM)
    assert next(stream).id == "j_1"
    stream.close()
    # The reader took the five rows of j_1 and the row that ended the job.
    assert next(source) == LINES[6]


@pytest.mark.parametrize("column", ["instance_num", "start_time", "end_time"])
@pytest.mark.parametrize(
    "text",
    [
        "one",
        "1.5",
        "10.0",
        "1e3",
        "1_000",
        "+5",
        " 5",
        "5 ",
        "\t5",
        "-",
        "--5",
        "5-",
        "0x10",
        "nan",
        "inf",
        "\u0665",  # an Arabic-Indic digit
        "\uff15",  # a full-width digit
        "\u00b2",  # a superscript digit
        "9" * 19,
        "-" + "9" * 19,
    ],
)
def test_integer_fields_take_ascii_digits_and_a_minus_sign_only(column, text):
    message = f"row 1: {column} must be an integer of at most 18 digits, got {text!r}"
    with pytest.raises(SerializationError, match=re.escape(message)):
        read([row("M1", **{column: text})])


def test_integer_fields_accept_leading_zeros_and_eighteen_digits():
    line = row("M1", instance_num="007", start_time="010", end_time="9" * 18)
    (problem,) = read([line])
    metadata = problem.tasks[0].metadata
    assert (metadata["instance_num"], metadata["start_time"]) == (7, 10)
    assert metadata["end_time"] == 10**18 - 1
    assert problem.tasks[0].modes[0].duration == PLATFORM.duration(float(10**18 - 11), "edge-0")
    assert outcome([row("M1", start_time="-" + "9" * 18)]) == ([], [("j_9", "time")])
    assert outcome([row("M1", end_time="-0")]) == ([], [("j_9", "time")])


@pytest.mark.parametrize("column", ["plan_cpu", "plan_mem"])
@pytest.mark.parametrize(
    "text",
    [
        "full",
        "nan",
        "NaN",
        "inf",
        "-inf",
        "Infinity",
        "1_000.5",
        "+1.5",
        " 1.5",
        "1.5 ",
        "1e999",
        "-1e999",
        "1e",
        "e5",
        ".",
        "-",
        "--1",
        "1.2.3",
        "1e5.0",
        "0x1p3",
        "\u0661.\u0665",  # Arabic-Indic digits
        "\uff11.5",  # a full-width digit
    ],
)
def test_decimal_fields_take_finite_ascii_decimals_only(column, text):
    message = f"row 1: {column} must be a finite decimal number, got {text!r}"
    with pytest.raises(SerializationError, match=re.escape(message)):
        read([row("M1", **{column: text})])


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("100", 100.0),
        ("100.0", 100.0),
        ("1e2", 100.0),
        ("1E+2", 100.0),
        (".5", 0.5),
        ("5.", 5.0),
        ("2.5e-1", 0.25),
        ("-1", -1.0),
        ("-0.0", 0.0),
    ],
)
def test_decimal_fields_take_fractions_and_exponents(text, value):
    (problem,) = read([row("M1", plan_cpu=text, plan_mem=text)])
    metadata = problem.tasks[0].metadata
    assert (metadata["plan_cpu"], metadata["plan_mem"]) == (value, value)
    assert str(metadata["plan_cpu"]) == str(value)


def test_long_fields_are_handled_in_linear_time():
    started = time.perf_counter()
    with pytest.raises(SerializationError, match=r"row 1: start_time must be an integer") as caught:
        read([row("M1", start_time="9" * LONG)])
    assert f"({LONG} characters)" in str(caught.value)
    assert len(str(caught.value)) < 200
    with pytest.raises(SerializationError, match="row 1: plan_cpu must be a finite decimal"):
        read([row("M1", plan_cpu="1" * LONG + "x")])
    with pytest.raises(SerializationError, match="row 1: plan_cpu must be a finite decimal"):
        read([row("M1", plan_cpu="1" * LONG)])
    with pytest.raises(SerializationError, match="row 1: plan_mem must be a finite decimal"):
        read([row("M1", plan_mem="1e" + "9" * LONG)])
    (problem,) = read([row("M1", plan_mem="0." + "3" * LONG)])
    assert problem.tasks[0].metadata["plan_mem"] == pytest.approx(1 / 3)
    assert outcome([row("M1"), row("M2" + "_1" * LONG + "x")]) == ([], [("j_9", "truncated-name")])
    assert outcome([row("M1"), row("task_" + "A" * LONG)]) == (["j_9"], [])
    assert outcome([row("M1"), row("M" * LONG)]) == (["j_9"], [])
    with pytest.raises(SerializationError, match="cut off") as caught:
        parse_alibaba_task_name("M2" + "_1" * LONG)
    assert len(str(caught.value)) < 300
    with pytest.raises(SerializationError, match=r"job 'jjjj.*characters\) returns after"):
        read([row("M1", job="j" * LONG), row("M1", job="j_1"), row("M2", job="j" * LONG)])
    assert time.perf_counter() - started < 20.0


def test_files_that_are_missing_or_not_text_are_refused_when_read(tmp_path):
    jobs = iter_alibaba_batch(tmp_path / "absent.csv", PLATFORM)
    with pytest.raises(OSError, match="absent"):
        next(jobs)
    path = tmp_path / "batch_task.csv"
    path.write_bytes(row("M1").encode() + b"\n\xff\xfe\n")
    with pytest.raises(SerializationError, match="not UTF-8 text"):
        read(path)


def test_work_that_cannot_be_represented_is_refused():
    lines = [row("M1", end_time="9" * 18, instance_num="9" * 18, plan_cpu="1e300")]
    with pytest.raises(SerializationError, match="job 'j_9' has a task whose work is too large"):
        read(lines, work="core_seconds")
    assert [problem.id for problem in read(lines)] == ["j_9"]
    slow = Platform((Resource("a-0", tier="a", metadata={"speed": 1e-300}),), Network())
    with pytest.raises(SerializationError, match="job 'j_9' cannot be built on the platform"):
        list(iter_alibaba_batch(lines, slow))


# ------------------------------------------------------------------- arguments


@pytest.mark.parametrize(
    "options",
    [
        {"objectives": ("makespan", "speed")},
        {"objectives": ("makespan", "makespan")},
        {"objectives": ()},
        {"objectives": None},
        {"statuses": ()},
        {"statuses": None},
        {"statuses": ("Terminated", "Done")},
        {"statuses": ("Failed", "Failed")},
        {"statuses": ("terminated",)},
        {"statuses": "Terminated"},
        {"statuses": {"Terminated"}},
        {"min_tasks": 0},
        {"min_tasks": 1.5},
        {"min_tasks": True},
        {"max_tasks": 0},
        {"min_tasks": 3, "max_tasks": 2},
        {"work": "cores"},
        {"work": None},
        {"edge_data": -1.0},
        {"edge_data": float("nan")},
        {"edge_data": "2"},
        {"on_skip": "print"},
        {"tiers": ("orbit",)},
        {"tiers": ("cloud", "orbit")},
        {"tiers": ("cloud", "cloud")},
        {"tiers": ()},
        {"tiers": 3},
        {"tiers": ("cloud", 3)},
        {"tiers": "cloud"},
        {"tiers": {"cloud"}},
        {"tiers": b"cloud"},
        {"tiers": iter(["cloud"])},
        {"tiers": {"cloud": 1}.keys()},
        {"tiers": ("fog", "orbit")},
        {"statuses": b"Terminated"},
        {"statuses": iter(["Terminated"])},
        {"objectives": "total_tardiness"},
        {"objectives": ("makespan", "max_lateness")},
        {"objectives": ["tardy_tasks"]},
    ],
)
def test_invalid_arguments_are_refused_before_the_source_is_read(options):
    class Unread:
        def __iter__(self) -> Iterator[str]:
            raise AssertionError("the source was read")

    with pytest.raises(ConfigError, match="invalid arguments for iter_alibaba_batch"):
        iter_alibaba_batch(Unread(), PLATFORM, **options)
    with pytest.raises(AssertionError, match="the source was read"):
        next(iter_alibaba_batch(Unread(), PLATFORM))


def test_a_string_is_refused_for_tiers_and_statuses():
    message = "tiers must be a sequence of tier names, got 'cloud'; write one tier as a list"
    with pytest.raises(ConfigError, match=message):
        iter_alibaba_batch(LINES, PLATFORM, tiers="cloud")
    message = "statuses must be a non-empty sequence of task statuses, got 'Terminated'; write"
    with pytest.raises(ConfigError, match=message):
        iter_alibaba_batch(LINES, PLATFORM, statuses="Terminated")


def test_objectives_that_need_due_dates_are_refused():
    assert all(task.due is None for problem in read() for task in problem.tasks)
    for name in ("total_tardiness", "max_lateness", "tardy_tasks"):
        with pytest.raises(ConfigError, match=rf"objectives \['{name}'\] need due dates"):
            iter_alibaba_batch(LINES, PLATFORM, objectives=("makespan", name))


def test_the_readers_onto_a_platform_refuse_the_same_tiers_and_objectives():
    document = {
        "name": "one-task",
        "schemaVersion": "1.5",
        "workflow": {
            "specification": {"tasks": [{"id": "a", "parents": [], "children": []}], "files": []},
            "execution": {"tasks": [{"id": "a", "runtimeInSeconds": 1.0}]},
        },
    }
    half = Platform((Resource("a-0", capacity=0.5, tier="a"),), Network())
    cases: list[tuple[Platform, dict[str, Any], str]] = [
        (PLATFORM, {"tiers": "cloud"}, "tiers must be a sequence of tier names, got 'cloud'"),
        (PLATFORM, {"tiers": {"cloud"}}, "tiers must be a sequence of tier names"),
        (PLATFORM, {"tiers": ("cloud", 3)}, "tiers must be a sequence of tier names"),
        (PLATFORM, {"tiers": 3}, "tiers must be a sequence of tier names"),
        (PLATFORM, {"tiers": ("cloud", "cloud")}, "tiers repeat a name"),
        (PLATFORM, {"tiers": ()}, r"no node in tiers \[\]"),
        (PLATFORM, {"tiers": ("orbit", "cloud")}, r"no node in tiers \['orbit'\]"),
        (three_tier_platform(fog=0), {"tiers": ("fog", "cloud")}, r"no node in tiers \['fog'\]"),
        (three_tier_platform(edge=0, fog=0, cloud=0), {}, "no node in tiers other than 'device'"),
        (half, {}, r"node 'a-0' has capacity 0\.5, less than the one core a task holds"),
        (PLATFORM, {"objectives": "total_tardiness"}, "need due dates"),
        (PLATFORM, {"objectives": ("makespan", "tardy_tasks")}, "need due dates"),
        (PLATFORM, {"objectives": ["max_lateness"]}, "need due dates"),
    ]
    for platform, options, message in cases:
        with pytest.raises(ConfigError, match=message):
            iter_alibaba_batch(LINES, platform, **options)
        with pytest.raises(ConfigError, match=message):
            read_wfcommons(document, platform, **options)
    for tiers in (["device", "cloud"], ("cloud", "edge"), deque(["fog"])):
        (job, *_) = read(tiers=tiers)
        workflow = read_wfcommons(document, PLATFORM, tiers=tiers)
        expected = [f"{tier}-0" for tier in tiers]
        assert [mode.location for mode in job.tasks[0].modes] == expected
        assert [mode.location for mode in workflow.tasks[0].modes] == expected


@pytest.mark.parametrize("source", [None, 5, b"batch_task.csv", bytearray(b"M1")])
def test_sources_that_are_neither_paths_nor_iterables_are_refused(source):
    with pytest.raises(ConfigError, match="source must be a path or an iterable"):
        iter_alibaba_batch(source, PLATFORM)


def test_platforms_that_cannot_run_the_jobs_are_refused():
    with pytest.raises(ConfigError, match="platform must be a Platform"):
        iter_alibaba_batch(LINES, PLATFORM.resources)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="no node in tiers"):
        iter_alibaba_batch(LINES, three_tier_platform(edge=0, fog=0, cloud=0))
    with pytest.raises(ConfigError, match=r"no node in tiers \['fog'\]"):
        iter_alibaba_batch(LINES, three_tier_platform(fog=0), tiers=("fog", "cloud"))
    apart = Platform((Resource("a-0", tier="a"), Resource("b-0", tier="b")), Network())
    assert [problem.id for problem in iter_alibaba_batch(LINES, apart)] == ["j_1", "j_2", "j_3"]
    # A missing link is refused in the form of every other argument, before any reading.
    message = (
        "^invalid arguments for iter_alibaba_batch: edge_data needs a link between "
        "'a' and 'b', which the platform network lacks$"
    )
    with pytest.raises(ConfigError, match=message):
        iter_alibaba_batch(LINES, apart, edge_data=1.0)
    twins = Platform((Resource("a-0", tier="a"), Resource("a-1", tier="a")), Network())
    message = (
        "^invalid arguments for iter_alibaba_batch: edge_data needs a link between 'a' and 'a'"
    )
    with pytest.raises(ConfigError, match=message):
        iter_alibaba_batch(LINES, twins, edge_data=1.0)
    # The tiers are checked first, so a tier without a node is reported and not its links.
    with pytest.raises(ConfigError, match=r"^invalid arguments .*: the platform has no node"):
        iter_alibaba_batch(LINES, apart, tiers=("a", "c"), edge_data=1.0)
    alone = Platform((Resource("a-0", tier="a"),), Network())
    assert len(list(iter_alibaba_batch(LINES, alone, edge_data=1.0))) == 3
    half = Platform((Resource("a-0", capacity=0.5, tier="a"),), Network())
    with pytest.raises(ConfigError, match=r"node 'a-0' has capacity 0\.5, less than the one core"):
        iter_alibaba_batch(LINES, half)


# ------------------------------------------------------------------ the module


def test_the_reader_is_exported_and_its_constants_are_consistent():
    assert benchmarks.iter_alibaba_batch is iter_alibaba_batch
    assert benchmarks.parse_alibaba_task_name is parse_alibaba_task_name
    assert len(COLUMNS) == 9
    assert len(set(SKIP_REASONS)) == len(SKIP_REASONS) == 11
    assert STATUSES[0] == "Terminated"
    assert WORK_RULES == ("duration", "core_seconds", "after_dependencies")
    assert read() == read()
